"""Scientific regression checks: python -m unittest test_bis_bic -v."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pywt

from bis_bic import frequency_grid, wavelet_bispectral_metrics, wav_bicoherence
from data_analysis.bispectrum.compare_center_bispectrum import load_signals


class BispectralTests(unittest.TestCase):
    def test_frequency_sums_at_nonzero_lower_bound(self):
        for wavelet in ("cgau1", "cmor1.5-1.0"):
            scales, frequencies, indices, sums = frequency_grid(
                115200, 150, 57000, 1024, wavelet
            )
            np.testing.assert_allclose(
                frequencies[sums], frequencies[indices, None] + frequencies[None, indices],
                rtol=1e-12, atol=1e-9,
            )
            np.testing.assert_allclose(
                pywt.scale2frequency(wavelet, scales) * 115200, frequencies
            )
            self.assertGreaterEqual(frequencies[indices[0]], 150)
            self.assertLessEqual(frequencies[indices[-1]], 28500 + 1e-9)

    def test_rejects_legacy_custom_scale_grid(self):
        scales = 34500 / np.linspace(57000, 150, 1024)
        with self.assertRaisesRegex(ValueError, "exact f1"):
            frequency_grid(115200, 150, 57000, scales=scales)

    def test_direct_time_average_and_original_normalization(self):
        fs = 2048
        t = np.arange(2048) / fs
        z = np.random.default_rng(91).normal(size=len(t))
        result = wavelet_bispectral_metrics(z, t, fs, 50, 800, Ns=16)
        # Independently calculate every triple at its physical frequencies.
        for i, f1 in enumerate(result.frequency_hz):
            for j, f2 in enumerate(result.frequency_hz):
                scales = pywt.frequency2scale("cgau1", np.array([f1, f2, f1 + f2]) / fs)
                w, _ = pywt.cwt(z - z.mean(), scales, "cgau1", 1 / fs, method="conv")
                w = w[:, result.sample_start : result.sample_stop]
                q = w[0] * w[1] * w[2].conj()
                np.testing.assert_allclose(result.complex_bispectrum[i, j], q.mean(), atol=1e-12)
                np.testing.assert_allclose(result.normalization_magnitude[i, j], np.abs(q).mean(), atol=1e-12)
                np.testing.assert_allclose(result.bicoherence[i, j], abs(q.mean()) / np.abs(q).mean(), atol=1e-12)

    def test_amplitude_and_constant_offset(self):
        fs = 2048
        t = np.arange(4096) / fs
        z = np.random.default_rng(8).normal(size=len(t))
        a = wavelet_bispectral_metrics(z, t, fs, 50, 800, Ns=16)
        b = wavelet_bispectral_metrics(3 * z + 200, t, fs, 50, 800, Ns=16)
        np.testing.assert_allclose(b.complex_bispectrum, 27 * a.complex_bispectrum, atol=1e-11)
        np.testing.assert_allclose(b.bicoherence, a.bicoherence, atol=1e-12)
        np.testing.assert_array_equal(a.bicoherence, a.bicoherence.T)

    def test_phase_locked_triad_vs_changing_biphase(self):
        fs = 4096
        t = np.arange(fs * 8) / fs
        # A narrow wavelet separates the three tones for this analytic check.
        f1, f2 = 120, 200
        common = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
        locked = common + np.cos(2 * np.pi * (f1 + f2) * t + 0.4)
        phase = np.random.default_rng(20).uniform(-np.pi, np.pi, 32)
        varying = common + np.cos(2 * np.pi * (f1 + f2) * t + phase[np.arange(len(t)) // 1024])
        kwargs = dict(fb_low=40, fb_high=800, Ns=20, wavelet="cmor3.0-1.0")
        a = wavelet_bispectral_metrics(locked, t, fs, **kwargs)
        b = wavelet_bispectral_metrics(varying, t, fs, **kwargs)
        i = np.argmin(abs(a.frequency_hz - f1))
        j = np.argmin(abs(a.frequency_hz - f2))
        self.assertGreater(a.bicoherence[i, j], 0.95)
        self.assertLess(b.bicoherence[i, j], 0.4)

    def test_zero_signal_is_undefined_and_wrappers_work(self):
        t = np.arange(2048) / 2048
        f, magnitude, b = wav_bicoherence(np.zeros(len(t)), t, 2048, 50, 800, Ns=16)
        self.assertEqual(magnitude.shape, (len(f), len(f)))
        self.assertTrue(np.all(magnitude == 0))
        self.assertTrue(np.isnan(b).all())

    def test_invalid_timing_and_short_record(self):
        with self.assertRaisesRegex(ValueError, "uniformly sampled"):
            wavelet_bispectral_metrics(np.ones(1024), np.arange(1024) / 1000, 2048, 50, 800, Ns=16)
        with self.assertRaisesRegex(ValueError, "too short"):
            wavelet_bispectral_metrics(np.ones(10), np.arange(10) / 2048, 2048, 50, 800, Ns=16)


class MeanTreatmentTests(unittest.TestCase):
    def test_rank_truncation_and_both_mean_treatments(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pod_path, cache_path = folder / "pod.h5", folder / "raw.npz"
            t = np.arange(32) / 2048
            coefficients = np.column_stack((np.sin(np.arange(32)), np.cos(np.arange(32))))
            modes = np.zeros((2, 2, 2))
            modes[:, 1, 1] = [2, 3]
            mean = 20 + np.arange(32) ** 2
            offset = 7.0
            raw = coefficients @ modes[:, 1, 1] + mean + offset
            with h5py.File(pod_path, "w") as h:
                h.attrs["source_file"] = "experiment.h5"
                h.create_dataset("pod/modes", data=modes)
                h.create_dataset("reduced/coefficients", data=coefficients).attrs["units"] = "microns"
                h.create_dataset("grid/time", data=t)
                for axis in ("x", "y"):
                    h.create_dataset(f"grid/{axis}", data=[0, 1]).attrs["units"] = "microns"
                h.create_dataset("preprocessing/frame_spatial_mean", data=mean).attrs["subtracted_before_pod"] = True
                h.create_dataset("preprocessing/temporal_mean_field", data=np.full((2, 2), offset)).attrs["subtracted_before_pod"] = True
            cache = dict(time=t, center_signal=raw, center_y_index=1, center_x_index=1,
                         center_y_coordinate=1, center_x_coordinate=1, signal_units="microns",
                         source_file="experiment.h5")
            np.savez(cache_path, **cache)
            _, _, signals, metadata = load_signals(cache_path, pod_path, 1)
            np.testing.assert_allclose(signals["with_mean_pod"], 2 * coefficients[:, 0] + mean + offset)
            np.testing.assert_allclose(signals["without_mean_pod"], 2 * coefficients[:, 0] + offset)
            np.testing.assert_allclose(signals["without_mean_experiment"], coefficients @ modes[:, 1, 1] + offset)
            for treatment in ("with_mean", "without_mean"):
                np.testing.assert_allclose(
                    signals[f"{treatment}_experiment"] - signals[f"{treatment}_pod"],
                    3 * coefficients[:, 1], atol=1e-12,
                )
            self.assertEqual(metadata["restored_static_center_offset"], offset)
            cache["time"] = t + 1 / 2048
            np.savez(cache_path, **cache)
            with self.assertRaisesRegex(ValueError, "timestamps"):
                load_signals(cache_path, pod_path, 1)


if __name__ == "__main__":
    unittest.main()
