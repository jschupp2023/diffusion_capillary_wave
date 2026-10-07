"""Numerical checks for shared-basis temporal comparisons."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import pdist

from data_analysis.correlations.compare_pod_lagged_covariance import (
    analyze_recording, checked_sampling_frequency, feature_vectors, held_out_centroid, lag_weights,
    physical_lag_covariance, read_signature,
    add_whitening_summary, whiten_lagged_covariance, whitened_identity_baseline,
    metric_features, entry_mask, WHITENED, DIAG_ONLY, OFFDIAG_ONLY,
    single_matrix_features,
)
from data_analysis.correlations.pod_lagged_covariance import lagged_covariances, normalize


class TemporalComparisonTests(unittest.TestCase):
    def test_single_matrix_distance_is_plain_rms_without_lag_weights(self):
        matrices = np.array([[[.8, -.2], [.1, .4]], [[.6, .1], [-.4, .5]]])
        distance = pdist(single_matrix_features(matrices))[0]
        self.assertAlmostEqual(distance, np.sqrt(np.mean((matrices[0]-matrices[1])**2)))
        np.testing.assert_array_equal(single_matrix_features(matrices), matrices.reshape(2, -1)/2)

    def test_component_features_select_entries_without_symmetrizing_or_mutating(self):
        values = np.arange(2*3*3*3, dtype=float).reshape(2, 3, 3, 3)
        original = values.copy()
        weights = np.array([.2, .3, .5])
        diagonal = np.diagonal(values, axis1=-2, axis2=-1)
        expected_diag = (diagonal*np.sqrt(weights)[None, :, None]/np.sqrt(3)).reshape(2, -1)
        np.testing.assert_allclose(metric_features(values, weights, DIAG_ONLY), expected_diag)
        off = np.stack([values[..., i, j] for i in range(3) for j in range(3) if i!=j], axis=-1)
        expected_off = (off*np.sqrt(weights)[None, :, None]/np.sqrt(6)).reshape(2, -1)
        np.testing.assert_allclose(metric_features(values, weights, OFFDIAG_ONLY), expected_off)
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(entry_mask(3, DIAG_ONLY) | entry_mask(3, OFFDIAG_ONLY), True)
        np.testing.assert_allclose(metric_features(values, weights, WHITENED), feature_vectors(values, weights))
        with self.assertRaisesRegex(ValueError, 'rank >= 2'):
            metric_features(values[..., :1, :1], weights, OFFDIAG_ONLY)

    def test_component_distances_reconstruct_full_squared_distance(self):
        k = 4
        values = np.random.default_rng(88).normal(size=(7, 5, k, k))
        weights = lag_weights(np.array([.05, .1, .5, 1., 2.]))
        full = pdist(metric_features(values, weights, WHITENED))
        diag = pdist(metric_features(values, weights, DIAG_ONLY))
        off = pdist(metric_features(values, weights, OFFDIAG_ONLY))
        np.testing.assert_allclose(full**2, diag**2/k + (k-1)*off**2/k, atol=1e-14)

    def test_components_isolate_autocorrelation_and_cross_coordinate_class_information(self):
        groups = np.repeat([0, 1], 3)
        eligible = np.ones(6, bool)
        for signal_metric, location, other_metric in ((DIAG_ONLY, (0, 0), OFFDIAG_ONLY),
                                                       (OFFDIAG_ONLY, (0, 1), DIAG_ONLY)):
            values = np.zeros((6, 1, 2, 2))
            values[:, 0, location[0], location[1]] = groups+.01*np.tile([-1, 0, 1], 2)
            good = metric_features(values, np.ones(1), signal_metric)
            empty = metric_features(values, np.ones(1), other_metric)
            self.assertEqual(held_out_centroid(good, groups, eligible, 2)[1], 1)
            self.assertEqual(held_out_centroid(empty, groups, eligible, 2)[1], .5)

    def test_full_whitening_removes_static_covariance_but_preserves_lag_dependence(self):
        # Identical scalar lag dependence with different SPD static covariances.
        q = np.array([[.8, -.6], [.6, .8]])
        c0 = q @ np.diag([1., 9.]) @ q.T
        lag_factors = np.array([1., .7, -.2])
        c = lag_factors[:, None, None] * c0
        kernel, w, _, diagnostics = whiten_lagged_covariance(c)
        np.testing.assert_allclose(kernel, lag_factors[:, None, None]*np.eye(2), atol=1e-12)
        np.testing.assert_allclose(w, w.T, atol=1e-14)
        self.assertTrue(diagnostics['identity_verified'])
        self.assertEqual(diagnostics['clipped_eigenvalues'], 0)
        np.testing.assert_allclose(diagnostics['condition_number'], 9)
        # Diagonal normalization retains static off-diagonal structure.
        original = normalize(c, np.sqrt(np.diag(c0)))
        self.assertGreater(abs(original[0, 0, 1]), .5)
        changed = c.copy()
        changed[1] = .3*c0
        k_changed = whiten_lagged_covariance(changed)[0]
        self.assertGreater(np.linalg.norm(kernel[1]-k_changed[1]), .5)

    def test_whitening_matches_covariance_of_transformed_coordinates(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(1001, 3))
        x[:, 1] = np.roll(x[:, 0], 2)+.1*x[:, 1]
        x = x @ np.array([[2., .5, 0], [.1, 1., .8], [0, .2, 3.]])
        x -= x.mean(axis=0)
        c = lagged_covariances(x, np.array([0, 1, 2, 5]))
        kernel, w, _, _ = whiten_lagged_covariance(c)
        expected = lagged_covariances(x @ w, np.array([0, 1, 2, 5]))
        np.testing.assert_allclose(kernel, expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(kernel[2]-kernel[2].T), .5)

    def test_eigenvalue_floor_reports_identity_failure_without_hiding_it(self):
        c0 = np.diag([1e-14, 1.])
        kernel, w, _, d = whiten_lagged_covariance(np.stack([c0, .5*c0]), 1e-8)
        self.assertTrue(np.isfinite(w).all())
        self.assertEqual(d['clipped_eigenvalues'], 1)
        self.assertFalse(d['identity_verified'])
        self.assertAlmostEqual(d['identity_frobenius_error'], 1-1e-6)
        np.testing.assert_allclose(kernel[0], np.diag([1e-6, 1.]))
        self.assertAlmostEqual(d['condition_number']/1e14, 1)
        self.assertAlmostEqual(d['regularized_condition_number']/1e8, 1)
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'):
            whiten_lagged_covariance(np.array([np.diag([-1., 1.])]))
        for eps in (0, -1, 1, float('nan')):
            with self.assertRaises(ValueError):
                whiten_lagged_covariance(np.array([np.eye(2)]), eps)

    def test_verified_identity_static_control_has_no_class_information(self):
        values = np.broadcast_to(np.eye(2), (6, 1, 2, 2)).copy()
        values[:3, :, 0, 0] += 1e-14  # Deliberately class-dependent numerical residue.
        original = values.copy()
        control, verified = whitened_identity_baseline(values, np.zeros(6, int))
        self.assertTrue(verified)
        np.testing.assert_array_equal(values, original)  # Never mutate the measured K(0).
        np.testing.assert_allclose(pdist(feature_vectors(control, np.ones(1))), 0, atol=0)
        confusion, accuracy = held_out_centroid(feature_vectors(control, np.ones(1)),
                                                np.repeat([0, 1], 3), np.ones(6, bool), 2)
        self.assertEqual(accuracy, .5)
        control, verified = whitened_identity_baseline(values, np.ones(6, int))
        self.assertFalse(verified)
        np.testing.assert_array_equal(control, original)

    def test_identity_centroid_ties_are_exact_for_unequal_class_sizes(self):
        groups = np.repeat(np.arange(8), [16, 15, 16, 16, 16, 15, 15, 16])
        # 0.1 cannot be represented exactly; naive means differ with sample count.
        features = np.tile((np.eye(10)/10).reshape(1, -1), (len(groups), 1))
        confusion, accuracy = held_out_centroid(features, groups, np.ones(len(groups), bool), 8)
        self.assertEqual(accuracy, .125)
        self.assertEqual(confusion[:, 1:].sum(), 0)

    def test_derived_whitening_reuses_cached_covariance(self):
        cov = np.tile(np.array([np.diag([1e-9, 1.]), np.diag([5e-10, .5])]), (2, 1, 1, 1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with h5py.File(output/'cache.h5', 'w') as cache:
                cache.create_dataset('covariance', data=cov)
                cache.attrs['configuration'] = 'original projection config'
                data, diagnostic = add_whitening_summary(cache.create_group('summary'),
                    cache['covariance'][:], 1e-10, ['a', 'b'], np.array([0]), output)
                self.assertFalse(np.any(diagnostic['clipped_eigenvalues']))
                del cache['summary']
                _, diagnostic = add_whitening_summary(cache.create_group('summary'),
                    cache['covariance'][:], 1e-8, ['a', 'b'], np.array([0]), output)
                self.assertTrue(np.all(diagnostic['clipped_eigenvalues']==1))
                np.testing.assert_array_equal(cache['covariance'][:], cov)
                self.assertEqual(cache.attrs['configuration'], 'original projection config')
                self.assertEqual(cache['summary/whitening'].attrs['eps_rel'], 1e-8)
            self.assertTrue((output/'whitening_diagnostics.csv').exists())

    def test_time_grid_accepts_float32_rounding_but_rejects_gaps(self):
        t = (np.arange(100001)/115200).astype('float32').astype(float)
        fs, error = checked_sampling_frequency(t)
        self.assertAlmostEqual(fs/115200, 1, places=6)
        self.assertLess(error, .05)
        t[50000:] += 1/115200
        with self.assertRaisesRegex(ValueError, 'uniform sampling'):
            checked_sampling_frequency(t)

    def test_fft_matches_direct_including_direction_and_fractional_lags(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(127, 3))
        x[:, 1] = np.roll(x[:, 0], 3) + .1*x[:, 1]
        x -= x.mean(axis=0)
        direct = lagged_covariances(x, np.arange(10))
        result = physical_lag_covariance(x, 1000, np.array([0, 1, 3, 4.5, 9])/1000)
        expected = np.stack([direct[0], direct[1], direct[3],
                             .5*(direct[4]+direct[5]), direct[9]])
        np.testing.assert_allclose(result, expected, atol=1e-12)
        self.assertGreater(result[2, 0, 1], result[2, 1, 0] + .5)
        with self.assertRaises(ValueError):
            physical_lag_covariance(x, 1000, np.array([.126]))

    def test_correlation_removes_coordinate_amplitude_changes(self):
        x = np.random.default_rng(5).normal(size=(401, 3))
        x -= x.mean(axis=0)
        lag = np.array([0, .001, .005])
        def corr(values):
            return normalize(physical_lag_covariance(values, 1000, lag),
                             np.sqrt(np.mean(values**2, axis=0)))
        np.testing.assert_allclose(corr(x), corr(x*np.array([2, 4, .3])), atol=1e-12)

    def test_distance_has_rms_units_and_time_weights(self):
        t = np.array([0, 1, 2, 5.])
        w = lag_weights(t)
        np.testing.assert_allclose(w, [.125, .5, .375])
        values = np.zeros((2, 3, 2, 2))
        values[1] = 3
        np.testing.assert_allclose(pdist(feature_vectors(values, w)), [3])

    def test_whole_recording_holdout_excludes_references(self):
        features = np.array([[1e6], [0], [.1], [-1e6], [10], [10.1]])
        groups = np.repeat([0, 1], 3)
        eligible = np.array([False, True, True, False, True, True])
        confusion, accuracy = held_out_centroid(features, groups, eligible, 2)
        np.testing.assert_array_equal(confusion, [[2, 0], [0, 2]])
        self.assertEqual(accuracy, 1)

    def test_projection_includes_higher_source_modes_and_checks_grid(self):
        # Common mode includes a component outside the leading source subspace.
        # Omitting source mode 3 therefore changes the measured trajectory.
        rng = np.random.default_rng(7)
        coefficients = rng.normal(size=(501, 4))
        modes = np.eye(4).reshape(4, 2, 2)
        reference = np.array([[[.8, 0, .6, 0], [0, 1, 0, 0]]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pod.h5'
            with h5py.File(path, 'w') as f:
                f.attrs['instantaneous_spatial_mean_removed'] = True
                f.attrs['temporal_mean_field_removed'] = True
                f.create_dataset('pod/modes', data=modes)
                for name, data, units in (
                    ('grid/x', [0., 1.], 'microns'), ('grid/y', [0., 1.], 'microns'),
                    ('grid/time', np.arange(501)/1000, 'seconds'),
                    ('reduced/coefficients', coefficients, 'microns'),
                    ('preprocessing/frame_spatial_mean', rng.normal(size=501), 'microns')):
                    f.create_dataset(name, data=data).attrs['units'] = units
            args = Namespace(source_ranks=[2, 4], rank=2, batch_size=71,
                             lags_ms=np.array([0., 1., 5.]))
            grid = (np.array([0., 1.]), np.array([0., 1.]))
            result = analyze_recording(path, reference, grid, read_signature(path), args)
            aligned = (coefficients-coefficients.mean(axis=0)) @ reference[0].T
            expected = lagged_covariances(aligned, np.array([0, 1, 5]))
            np.testing.assert_allclose(result['covariance'][0], expected, atol=1e-12)
            self.assertGreater(result['relative_projection_error'][0, 0], .3)
            self.assertEqual(result['relative_projection_error'][0, 1], 0)
            np.testing.assert_allclose(result['reference_subspace_capture'][0], [.82, 1])
            with self.assertRaisesRegex(ValueError, 'grid mismatch'):
                analyze_recording(path, reference, (grid[0]+1, grid[1]), read_signature(path), args)
            # The simple diagnostic must use exact sample delays, not the
            # physical-lag list retained in the older workflow's arguments.
            args.single_lag_samples = [0, 2, 10]
            args.single_lag_reference_fs = 1000.
            result = analyze_recording(path, reference, grid, read_signature(path), args)
            expected = lagged_covariances(aligned, np.array([0, 2, 10]))
            np.testing.assert_allclose(result['covariance'][0], expected, atol=1e-12)
            expected_correlation = normalize(expected, np.sqrt(np.mean(aligned**2, axis=0)))
            np.testing.assert_allclose(result['correlation'][0], expected_correlation, atol=1e-12)
            args.single_lag_reference_fs = 900.
            with self.assertRaisesRegex(ValueError, 'matching sampling frequencies'):
                analyze_recording(path, reference, grid, read_signature(path), args)


if __name__ == '__main__':
    unittest.main()
