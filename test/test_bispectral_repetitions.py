"""Scientific checks for repetition distances and sparse spectral evaluation."""

import unittest

import numpy as np

from bis_bic import wavelet_bispectral_metrics
from data_analysis.bispectrum.compare_bispectral_repetitions import (
    distance_matrices, group_summary, log_area_weights, permutation_separation,
)
from data_analysis.bispectrum.sparse_bispectrum import sparse_grid, sparse_wavelet_metrics
from data_analysis.bispectrum.compare_center_bispectrum import weighted_scores


class SparseTests(unittest.TestCase):
    def test_matches_dense_metrics_at_selected_frequencies(self):
        fs = 4096
        t = np.arange(8192) / fs
        z = np.random.default_rng(1).normal(size=len(t))
        dense = wavelet_bispectral_metrics(z, t, fs, 50, 1600, Ns=64)
        sparse, _ = sparse_wavelet_metrics(z, t, fs, 50, 1600, Ns=64, axis_samples=12)
        grid = sparse_grid(fs, 50, 1600, Ns=64, axis_samples=12)
        idx = np.ix_(grid.output_positions, grid.output_positions)
        self.assertEqual((dense.sample_start, dense.sample_stop), (sparse.sample_start, sparse.sample_stop))
        np.testing.assert_array_equal(sparse.frequency_hz[[0, -1]], dense.frequency_hz[[0, -1]])
        np.testing.assert_allclose(sparse.complex_bispectrum, dense.complex_bispectrum[idx], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(sparse.bicoherence, dense.bicoherence[idx], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(grid.transform_frequency_hz[grid.sum_indices],
                                   grid.frequency_hz[:, None] + grid.frequency_hz[None, :])

    def test_zero_signal_remains_undefined(self):
        t = np.arange(1024) / 4096
        result, _ = sparse_wavelet_metrics(np.zeros(len(t)), t, 4096, 50, 1600, Ns=32, axis_samples=8)
        self.assertTrue(np.isnan(result.bicoherence).all())
        self.assertTrue(np.all(result.magnitude == 0))


class DistanceTests(unittest.TestCase):
    def test_raw_pod_scores_match_repetition_grid_and_amplitude_invariance(self):
        # Multiplying a signal by two multiplies B by eight, leaves b unchanged,
        # and must disappear from the centered shape distance on either grid.
        fs = 4096
        t = np.arange(4096) / fs
        z = np.random.default_rng(7).normal(size=len(t))
        raw = wavelet_bispectral_metrics(z, t, fs, 50, 1600, Ns=64)
        pod = wavelet_bispectral_metrics(2*z, t, fs, 50, 1600, Ns=64)
        data = {"frequency_hz": raw.frequency_hz}
        for treatment in ("with_mean", "without_mean"):
            for source, result in (("experiment", raw), ("pod", pod)):
                data[f"{treatment}_{source}_complex_bispectrum"] = result.complex_bispectrum
                data[f"{treatment}_{source}_bicoherence"] = result.bicoherence
        metadata = {"sampling_frequency_hz": fs, "fb_low_hz": 50,
                    "fb_high_hz": 1600, "num_scales_requested": 64, "wavelet": "cgau1"}
        scores = weighted_scores(data, metadata, axis_samples=12, maximum_sum=600)
        self.assertEqual(len(scores), 8)
        grid = sparse_grid(fs, 50, 1600, Ns=64, axis_samples=12)
        for key, item in scores.items():
            self.assertAlmostEqual(item["bispectrum_log"], np.log10(8))
            self.assertAlmostEqual(item["bispectrum_shape"], 0)
            self.assertAlmostEqual(item["bicoherence"], 0)
            self.assertAlmostEqual(item["retained_log_area_fraction"], 1)
            if key.startswith("sparse12"):
                self.assertEqual(item["axis_samples"], len(grid.frequency_hz))

    def test_shape_removes_uniform_magnitude_scaling(self):
        logs = np.array([[1., 2., 3.], [2., 3., 4.]])
        b = np.array([[.1, .2, .3], [.3, .4, .5]])
        distances, _ = distance_matrices(logs, b, np.array([.2, .3, .5]))
        self.assertAlmostEqual(distances["bispectrum_log"][0, 1], 1)
        self.assertAlmostEqual(distances["bispectrum_shape"][0, 1], 0)
        self.assertAlmostEqual(distances["bicoherence"][0, 1], .2)

    def test_common_valid_domain_and_coverage(self):
        logs = np.array([[1., 2., np.nan], [2., np.nan, 3.]])
        b = np.zeros_like(logs)
        distances, info = distance_matrices(logs, b, np.array([.2, .3, .5]))
        self.assertAlmostEqual(info["retained_log_area_fraction"], .2)
        self.assertAlmostEqual(distances["bispectrum_log"][0, 1], 1)

    def test_folded_weights_and_sum_domain(self):
        f = np.array([100., 200., 400.])
        up, weights = log_area_weights(f, 500)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertTrue(np.all(weights[f[up[0]] + f[up[1]] > 500] == 0))
        _, full = log_area_weights(f)
        # Endpoint widths are half the interior width on this logarithmic grid.
        self.assertAlmostEqual(full[0], 1/16)
        self.assertAlmostEqual(full[1], 1/4)

    def test_group_means_exclude_self_pairs(self):
        x = np.array([0., 1., 10., 11.])
        distance = abs(x[:, None] - x[None, :])
        result, blocks = group_summary(distance, ["A", "A", "B", "B"])
        np.testing.assert_allclose(blocks, [[1, 10], [10, 1]])
        self.assertEqual(result["between"][0]["between_to_within_ratio"], 10)
        self.assertEqual(result["within"]["A"]["pairs"], 1)

    def test_permutation_moves_repetitions_and_is_reproducible(self):
        x = np.r_[np.arange(4), 20 + np.arange(4)].astype(float)
        distance = abs(x[:, None] - x[None, :])
        labels = ["A"]*4 + ["B"]*4
        a = permutation_separation(distance, labels, count=499, seed=3)
        b = permutation_separation(distance, labels, count=499, seed=3)
        self.assertEqual(a, b)
        self.assertGreater(a["one_sided_pvalue_unadjusted"], 0)
        self.assertLess(a["one_sided_pvalue_unadjusted"], .1)


if __name__ == "__main__":
    unittest.main()
