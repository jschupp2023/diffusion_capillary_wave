import unittest
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from itertools import combinations
from unittest.mock import patch
import h5py
import numpy as np
from data_analysis.energy.pod_capillary_energy import (capillary_stiffness, energy_series, energy_statistics,
                                  length_scale, matrix_diagnostics, quadrature_weights,
                                  coefficient_second_moment, mean_energies, parse_args, run,
                                  select_references, temporal_weights, field_energies,
                                  reconstructed_energy_series, select_energy_frames)


class CapillaryEnergyTests(unittest.TestCase):
    def test_frame_sampling_is_unique_reproducible_and_unbiased_for_time_mean(self):
        t = np.array([0., .1, .4, 1., 2.])
        energy = np.array([2., 8., 1., 3., 7.])
        indices, weights = select_energy_frames(t, 2, 12345)
        other, _ = select_energy_frames(t, 2, 12345)
        np.testing.assert_array_equal(indices, other)
        self.assertEqual(len(set(indices)), 2)
        np.testing.assert_allclose(weights, temporal_weights(t)[indices]*len(t)/2)
        estimates = [np.dot(energy[list(p)], temporal_weights(t)[list(p)]*len(t)/2) for p in combinations(range(len(t)), 2)]
        self.assertAlmostEqual(np.mean(estimates), np.trapezoid(energy, t)/(t[-1]-t[0]))
        all_indices, all_weights = select_energy_frames(t, 2000, 42)
        np.testing.assert_array_equal(all_indices, np.arange(len(t)))
        np.testing.assert_allclose(all_weights, temporal_weights(t))
    def test_constant_mode_zero_affine_modes_have_analytic_energy(self):
        x, y = np.linspace(0, 3e-4, 11), np.linspace(0, 2e-4, 9)
        xx, yy = np.meshgrid(x, y)
        modes = np.array([np.ones_like(xx), xx/x[-1], yy/y[-1]])
        gamma = .0728
        k, info = capillary_stiffness(modes, x, y, 'surface', gamma)
        area = x[-1]*y[-1]
        expected = np.diag([0., gamma*area/x[-1]**2, gamma*area/y[-1]**2])
        np.testing.assert_allclose(k, expected, atol=1e-14)
        self.assertAlmostEqual(info['measure']/area, 1)
        a = np.array([[4e-6, 2e-6, 3e-6]])
        energy = energy_series(a, k)[0]
        self.assertAlmostEqual(energy/(.5*gamma*area*((2e-6/x[-1])**2+(3e-6/y[-1])**2)), 1)
        self.assertTrue(matrix_diagnostics(k)['positive_semidefinite'])

    def test_quadratic_form_matches_direct_surface_integration(self):
        rng = np.random.default_rng(41)
        x, y = np.linspace(0, 4e-4, 8), np.linspace(0, 3e-4, 7)
        modes = rng.normal(size=(4, len(y), len(x)))
        a = rng.normal(size=(5, 4))*1e-6
        k, _ = capillary_stiffness(modes, x, y, gamma=.0728)
        field = np.einsum('tr,ryx->tyx', a, modes)
        gx = np.gradient(field, x, axis=-1, edge_order=2)
        gy = np.gradient(field, y, axis=-2, edge_order=2)
        weights = np.outer(quadrature_weights(y), quadrature_weights(x))
        direct = .5*.0728*np.sum((gx**2+gy**2)*weights, axis=(1, 2))
        np.testing.assert_allclose(energy_series(a, k), direct, rtol=1e-13)

    def test_centerline_uses_only_x_slope_and_energy_scales_with_gamma(self):
        x, y = np.linspace(0, 2e-4, 9), np.linspace(0, 3e-4, 7)
        xx, yy = np.meshgrid(x, y)
        modes = np.array([xx/x[-1], yy/y[-1], np.ones_like(xx)])
        k, _ = capillary_stiffness(modes, x, y, 'centerline')
        np.testing.assert_allclose(k, np.diag([1/x[-1], 0., 0.]), atol=1e-10)
        physical, _ = capillary_stiffness(modes, x, y, 'centerline', .0728)
        np.testing.assert_allclose(physical, .0728*k)

    def test_time_average_uses_elapsed_time_and_units_are_explicit(self):
        result = energy_statistics(np.array([0., 1., 3.]), np.array([0., 2., 6.]))
        self.assertAlmostEqual(result['time_averaged_energy'], 3.)
        self.assertNotEqual(result['time_averaged_energy'], result['sample_mean'])
        self.assertEqual(length_scale('microns'), 1e-6)
        self.assertEqual(length_scale('µm'), 1e-6)
        with self.assertRaises(ValueError):
            length_scale('unknown')
        with self.assertRaises(ValueError):
            matrix_diagnostics(np.diag([-1., 2.]))

    def test_exact_plane_area_flat_surface_and_small_slope_limit(self):
        x, y = np.linspace(0, 3e-4, 11), np.linspace(0, 2e-4, 9)
        xx, yy = np.meshgrid(x, y)
        fields = np.array([np.zeros_like(xx), 3*xx+4*yy, 1e-10*xx])
        quad, exact = field_energies(fields, x, y)
        area = x[-1]*y[-1]
        self.assertEqual(exact[0], 0.)
        np.testing.assert_allclose(quad[1], .5*25*area, rtol=1e-14)
        np.testing.assert_allclose(exact[1], (np.sqrt(26)-1)*area, rtol=1e-14)
        self.assertTrue(np.all(exact <= quad*(1+1e-14)))
        self.assertGreater(exact[2], 0.)  # sqrt(1+s)-1 would round to zero here.
        np.testing.assert_allclose(exact[2], quad[2], rtol=1e-14)
        cq, ce = field_energies(fields, x, y, 'centerline')
        np.testing.assert_allclose(cq[1], .5*9*x[-1], rtol=1e-14)
        np.testing.assert_allclose(ce[1], (np.sqrt(10)-1)*x[-1], rtol=1e-14)

    def test_batched_nested_reconstruction_matches_direct_fields(self):
        rng = np.random.default_rng(841)
        x, y = np.linspace(0, 3e-4, 7), np.linspace(0, 2e-4, 6)
        basis = rng.normal(size=(5, 42))
        a = rng.normal(size=(13, 5))*1e-5
        ranks = [1, 3, 5]
        for batch_size in [1, 4, 100]:
            quad, exact = reconstructed_energy_series(a, basis, x, y, ranks, 'surface', batch_size)
            for ri, rank in enumerate(ranks):
                field = (a[:, :rank] @ basis[:rank]).reshape(13, 6, 7)
                slopes = np.gradient(field, x, axis=-1, edge_order=2)**2 + np.gradient(field, y, axis=-2, edge_order=2)**2
                weights = np.outer(quadrature_weights(y), quadrature_weights(x))
                direct = np.sum((np.sqrt(1+slopes)-1)*weights, axis=(1, 2))
                np.testing.assert_allclose(exact[ri], direct, rtol=1e-13)
                k, _ = capillary_stiffness(basis[:rank].reshape(rank, 6, 7), x, y)
                np.testing.assert_allclose(quad[ri], energy_series(a[:, :rank], k), rtol=1e-13)

    def test_mean_restoration_includes_slope_interactions_before_exact_energy(self):
        x, y = np.linspace(0, 3., 7), np.linspace(0, 2., 6)
        xx, yy = np.meshgrid(x, y)
        basis = np.stack([xx.ravel(), yy.ravel()])
        a = np.array([[.2, .3], [-.4, .1]])
        mean = .5*xx - .2*yy + 7.
        quad, exact = reconstructed_energy_series(a, basis, x, y, [1, 2], 'surface', 1, mean)
        for i, rank in enumerate([1, 2]):
            slope2 = (a[:, 0]+.5)**2 + ((a[:, 1] if rank == 2 else 0)-.2)**2
            np.testing.assert_allclose(quad[i], 3*slope2, rtol=1e-13)
            np.testing.assert_allclose(exact[i], 6*(np.sqrt(1+slope2)-1), rtol=1e-13)
        with self.assertRaises(ValueError):
            reconstructed_energy_series(a, basis, x, y, [2], 'surface', 1, np.zeros((1, 1)))

    def test_mean_contraction_matches_integrated_trajectories_for_every_rank(self):
        rng = np.random.default_rng(22)
        t = np.cumsum(rng.uniform(.01, .1, 23))
        b = rng.normal(size=(23, 5)) + 2.  # Nonzero mean must not be removed.
        overlap = rng.normal(size=(4, 5))
        gradient = rng.normal(size=(7, 4))
        matrix = gradient.T @ gradient
        ranks = [1, 2, 4]
        expected = [energy_statistics(t, energy_series((b @ overlap.T)[:, :r]*1e-6, matrix[:r, :r]))['time_averaged_energy']
                    for r in ranks]
        for batch_size in (1, 7, 100):
            moment = coefficient_second_moment(b, t, 5, 1e-6, batch_size)
            np.testing.assert_allclose(mean_energies(moment, overlap, matrix, ranks), expected, rtol=1e-13)
        np.testing.assert_allclose(temporal_weights(np.array([0., 2.])), [.5, .5])
        with self.assertRaises(ValueError):
            temporal_weights(np.array([0., 0.]))

    def test_default_study_and_distinct_reproducible_selection(self):
        args = parse_args([])
        self.assertEqual(args.rank, [10, 100, 1000])
        self.assertEqual(args.source_rank, 1000)
        self.assertEqual(args.references, 3)
        self.assertFalse(args.exact)
        self.assertTrue(parse_args(['--exact']).exact)
        local_args = parse_args(['--basis-scope', 'per-power'])
        self.assertEqual(local_args.references, 2)
        self.assertNotEqual(local_args.output_dir, args.output_dir)
        refs = select_references(list(range(128)), 3, 12345)
        self.assertEqual(refs, select_references(list(range(128)), 3, 12345))
        self.assertEqual(len(set(refs)), 3)
        self.assertNotEqual(refs, select_references(list(range(128)), 3, 42))

    def test_small_study_outputs_match_direct_energy_and_resume_without_projection(self):
        self._check_small_study('global')

    def test_per_power_study_uses_only_local_bases_and_reuses_moments(self):
        self._check_small_study('per-power')

    def test_per_recording_study_matches_own_coefficients_and_aggregates_repetitions(self):
        self._check_small_study('per-recording')

    def _check_small_study(self, scope):
        rng = np.random.default_rng(71)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, y = np.linspace(0, 300, 6), np.linspace(0, 200, 5)
            t = .01*np.linspace(0, 1, 17)**1.3  # Check actual time weights, not a sample mean.
            for power in ['0p000', '0p04', '0p08', '0p20']:
                for rep in [1, 2, 3]:
                    folder = root/power/f'Ca_ac_{power}_rep{rep}'
                    folder.mkdir(parents=True)
                    modes = np.linalg.qr(rng.normal(size=(30, 4)))[0].T.reshape(4, 5, 6)
                    with h5py.File(folder/'pod_2d_r4.h5', 'w') as f:
                        f.attrs['instantaneous_spatial_mean_removed'] = True
                        f.attrs['temporal_mean_field_removed'] = True
                        f.create_dataset('pod/modes', data=modes)
                        mean = f.create_dataset('preprocessing/temporal_mean_field', data=x[None, :]/30+y[:, None]/40)
                        mean.attrs['units'] = 'microns'
                        mean.attrs['subtracted_before_pod'] = True
                        for name, values, units in [('grid/x', x, 'microns'), ('grid/y', y, 'microns'),
                                                    ('grid/time', t, 'seconds'),
                                                    ('reduced/coefficients', rng.normal(size=(17, 4))+1, 'microns'),
                                                    ('preprocessing/frame_spatial_mean', np.zeros(17), 'microns')]:
                            f.create_dataset(name, data=values).attrs['units'] = units
            args = parse_args(['--data-root', str(root), '--rank', '1', '2', '4', '--source-rank', '4',
                               '--pod-rank', '4', '--gamma', '.0728', '--batch-size', '5', '--field-batch-size', '2', '--basis-scope', scope])
            with patch('pod_capillary_energy.cached_energy_trajectories', side_effect=AssertionError('exact should be opt-in')), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)
            with h5py.File(args.output_dir/'capillary_energy.h5') as cache:
                self.assertNotIn('trajectories', cache['recordings/0p08_rep1'])
            self.assertNotIn('Ebar_exact', (args.output_dir/'repetition_energy.csv').read_text().splitlines()[0])
            args.exact = True
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            out = args.output_dir
            rows = list(csv.DictReader((out/'repetition_energy.csv').read_text().splitlines()))
            self.assertEqual(len(rows), {'global': 54, 'per-power': 36, 'per-recording': 18}[scope])
            self.assertEqual({r['power'] for r in rows}, {'0p08', '0p20'})
            config = json.loads((out/'configuration.json').read_text())
            self.assertEqual(len({r['source']['path'] for r in config['references']}), {'global': 3, 'per-power': 4, 'per-recording': 6}[scope])
            summaries = list(csv.DictReader((out/'power_energy_summary.csv').read_text().splitlines()))
            self.assertEqual(len(summaries), {'global': 18, 'per-power': 12, 'per-recording': 6}[scope])
            self.assertTrue(all(int(s['repetitions']) == 3 for s in summaries))
            if scope == 'per-recording':
                self.assertTrue(all(r['power'] == r['reference_power'] and r['repetition'] == r['reference_repetition'] for r in rows))
                self.assertTrue(all(float(r['reference_subspace_capture']) == 1. for r in rows))
            if scope == 'per-power':
                self.assertTrue(all(r['power'] == r['reference_power'] for r in rows))
                for power in ['0p08', '0p20']:
                    local_refs = [r for r in config['references'] if r['power'] == power]
                    self.assertEqual({r['reference_slot'] for r in local_refs}, {1, 2})
                    self.assertEqual(len({r['repetition'] for r in local_refs}), 2)
                for rank in ['1', '2', '4']:
                    for slot in ['1', '2']:
                        curve = [s for s in summaries if s['rank'] == rank and s['reference_slot'] == slot]
                        self.assertEqual({s['power'] for s in curve}, {'0p08', '0p20'})
            with h5py.File(out/'capillary_energy.h5') as cache:
                for row in rows:
                    rank = int(row['rank'])
                    reference = cache[f"references/basis_{row['reference_id']}"]
                    k = reference['K_gamma'][:rank, :rank]
                    with h5py.File(row['source_file']) as source:
                        if scope == 'per-recording':
                            self.assertNotIn('modes', reference)
                            basis = source['pod/modes'][:rank].reshape(rank, -1)
                            aligned = source['reduced/coefficients'][:, :rank]*1e-6
                        else:
                            basis = reference['modes'][:rank]
                            p = basis @ source['pod/modes'][:].reshape(4, -1).T
                            aligned = source['reduced/coefficients'][:] @ p.T * 1e-6
                    expected = energy_statistics(t, energy_series(aligned, k))['time_averaged_energy']
                    np.testing.assert_allclose(float(row['time_averaged_energy']), expected, rtol=1e-12)
                    np.testing.assert_allclose(float(row['Ebar_quad']), expected, rtol=1e-12)
                    field = (aligned @ basis).reshape(len(t), len(y), len(x))
                    slopes = np.gradient(field, x*1e-6, axis=-1, edge_order=2)**2 + np.gradient(field, y*1e-6, axis=-2, edge_order=2)**2
                    weight = np.outer(quadrature_weights(y*1e-6), quadrature_weights(x*1e-6))
                    # Independent stable area formula; plain sqrt(1+s)-1 loses
                    # accuracy for the near-flat frames in this fixture.
                    direct_exact = .0728*np.sum(np.expm1(.5*np.log1p(slopes))*weight, axis=(1, 2))
                    exact_mean = energy_statistics(t, direct_exact)['time_averaged_energy']
                    np.testing.assert_allclose(float(row['Ebar_exact']), exact_mean, rtol=1e-9)
                    recording = cache[f"recordings/{row['power']}_rep{row['repetition']}"]
                    bi = list(recording['reference_ids'][:]).index(int(row['reference_id']))
                    ri = args.rank.index(rank)
                    np.testing.assert_allclose(recording['trajectories/E_exact'][bi, ri], direct_exact, rtol=1e-9)
                    np.testing.assert_allclose(recording['trajectories/E_quad'][bi, ri], energy_series(aligned, k), rtol=1e-11)
                self.assertNotIn('time_seconds', cache['recordings/0p08_rep1'])
                for power in ['0p08', '0p20']:
                    ids = cache[f'recordings/{power}_rep1/reference_ids'][:]
                    expected = [r['reference_id'] for r in config['references'] if scope == 'global' or
                                (r['power'] == power and (scope != 'per-recording' or r['repetition'] == 1))]
                    self.assertEqual(list(ids), expected)
            self.assertTrue((out/'energy_over_power.pdf').is_file())
            self.assertFalse((out/'example_energy_trajectories.png').exists())
            args.gamma = 1.
            with patch('pod_capillary_energy.coefficient_second_moment', side_effect=AssertionError('recomputed moment')), \
                 (contextlib.nullcontext() if scope == 'per-recording' else patch('pod_capillary_energy.load_basis', side_effect=AssertionError('reloaded modes'))), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)
            proxy_rows = list(csv.DictReader((out/'repetition_energy.csv').read_text().splitlines()))
            for physical, proxy in zip(rows, proxy_rows):
                np.testing.assert_allclose(float(physical['time_averaged_energy'])/.0728,
                                           float(proxy['time_averaged_energy']), rtol=1e-13)
                np.testing.assert_allclose(float(physical['Ebar_exact'])/.0728,
                                           float(proxy['Ebar_exact']), rtol=1e-13)
                self.assertEqual(proxy['units'], 'm²')
            # Backfill an old quadratic-only recording and resume a partial trajectory.
            with h5py.File(out/'capillary_energy.h5', 'a') as cache:
                del cache['recordings/0p08_rep1/trajectories']
                partial = cache['recordings/0p08_rep2/trajectories']
                partial.attrs['complete'] = False
                partial.attrs['completed_samples'] = 5
                prefix = partial['geometric_exact'][:, :, :5]
                partial['geometric_exact'][:, :, 5:] = np.nan
                partial['geometric_quad'][:, :, 5:] = np.nan
            with patch('pod_capillary_energy.coefficient_second_moment', side_effect=AssertionError('recomputed moment')), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)

            with h5py.File(out/'capillary_energy.h5') as cache:
                restored = cache['recordings/0p08_rep2/trajectories']
                np.testing.assert_array_equal(restored['geometric_exact'][:, :, :5], prefix)
                self.assertTrue(np.isfinite(restored['geometric_exact'][:]).all())
                self.assertTrue(restored.attrs['complete'])
            resumed_rows = list(csv.DictReader((out/'repetition_energy.csv').read_text().splitlines()))
            for previous, resumed in zip(proxy_rows, resumed_rows):
                self.assertEqual(previous['Ebar_exact'], resumed['Ebar_exact'])
            # Sampled reconstruction must match the corresponding full cached frames.
            args.exact_frames = 7
            with patch('pod_capillary_energy.coefficient_second_moment', side_effect=AssertionError('recomputed moment')), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)
            with h5py.File(args.output_dir/'capillary_energy.h5') as cache:
                for record in cache['recordings'].values():
                    full = record['trajectories']
                    sampled = record['sampled_trajectories_n7_seed12345']
                    indices = sampled['frame_indices'][:]
                    self.assertEqual(len(set(indices)), 7)
                    for observable in ['quad', 'exact']:
                        np.testing.assert_allclose(sampled[f'E_{observable}'][:], full[f'E_{observable}'][:, :, indices], rtol=1e-11)
                        np.testing.assert_allclose(record[f'Ebar_{observable}'][:], sampled[f'E_{observable}'][:] @ sampled['mean_weights'][:], rtol=1e-13)
            # A different seed needs new overlaps, but reuses the costly source moments.
            args.seed = 42
            args.output_dir = root/'different_seed'
            with patch('pod_capillary_energy.coefficient_second_moment', side_effect=AssertionError('recomputed moment')), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)

            if scope == 'per-recording':
                args.include_low_powers = True
                args.restore_temporal_mean = True
                args.output_dir = root/'restored_including_low_powers'
                with patch('pod_capillary_energy.cached_second_moment', side_effect=AssertionError('restoration does not need full moments')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                restored_rows = list(csv.DictReader((args.output_dir/'repetition_energy.csv').read_text().splitlines()))
                self.assertEqual(len(restored_rows), 36)
                self.assertEqual({r['power'] for r in restored_rows}, {'0p000', '0p04', '0p08', '0p20'})
                restored_config = json.loads((args.output_dir/'configuration.json').read_text())
                self.assertFalse(restored_config['energy_temporal_mean_removed'])
                self.assertTrue(restored_config['restore_temporal_mean'])
                indices, weights = select_energy_frames(t, args.exact_frames, args.frame_seed)
                for row in restored_rows:
                    rank = int(row['rank'])
                    with h5py.File(row['source_file']) as source:
                        modes = source['pod/modes'][:rank].reshape(rank, -1)
                        coefficients = source['reduced/coefficients'][:][indices, :rank]*1e-6
                        mean = source['preprocessing/temporal_mean_field'][:]*1e-6
                    field = (coefficients @ modes).reshape(len(indices), len(y), len(x))+mean
                    quad, exact = field_energies(field, x*1e-6, y*1e-6)
                    np.testing.assert_allclose(float(row['Ebar_exact']), args.gamma*exact @ weights, rtol=1e-12)
                    np.testing.assert_allclose(float(row['Ebar_quad']), args.gamma*quad @ weights, rtol=1e-12)
                    self.assertEqual(row['Ebar_quad'], row['time_averaged_energy'])
                    self.assertNotIn('quadratic_sampling_relative_error', row)
                with patch('pod_capillary_energy.reconstructed_energy_series', side_effect=AssertionError('recomputed restored samples')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                args.restore_temporal_mean = False
                with self.assertRaisesRegex(ValueError, 'different projection setup'), contextlib.redirect_stdout(io.StringIO()):
                    run(args)


if __name__ == '__main__':
    unittest.main()
