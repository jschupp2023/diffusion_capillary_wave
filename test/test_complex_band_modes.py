"""Tests for the shared complex frequency-band decomposition."""

import argparse
import unittest

import numpy as np

from data_analysis.pod.complex_band_modes import (
    BandDefinition,
    _decoder_diagnostics,
    fit_complex_band_modes,
    make_bands,
    parse_band,
    transform_trajectory,
)


class ComplexBandModeTests(unittest.TestCase):
    def test_band_parsing_rank_broadcast_and_validation(self):
        self.assertEqual(parse_band("80:115"), (80., 115.))
        bands = make_bands(((150, 190), (60, 100)), (2,), nyquist_hz=500)
        self.assertEqual([(b.low_hz, b.high_hz, b.rank) for b in bands],
                         [(60., 100., 2), (150., 190., 2)])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_band("80-115")
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_bands(((60, 100), (90, 130)), (1,), nyquist_hz=500)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            make_bands(((450, 510),), (1,), nyquist_hz=500)

    def test_recovers_complex_directions_and_reconstructs_selected_bands(self):
        fs = 1000.
        time = np.arange(12000) / fs
        first = np.array([1., np.exp(.35j), .35 * np.exp(-.7j), 0.], complex)
        second = np.array([.15j, .2, np.exp(-.45j), .8 * np.exp(.6j)], complex)
        first /= np.linalg.norm(first)
        second /= np.linalg.norm(second)

        def trajectory(phase):
            z1 = 2.2 * np.exp(1j * (2 * np.pi * 80 * time + phase))
            z2 = 1.3 * np.exp(1j * (2 * np.pi * 180 * time - .4 * phase))
            return np.real(z1[:, None] * first.conj()[None, :]
                           + z2[:, None] * second.conj()[None, :])

        signals = {"rep1": trajectory(.2), "rep2": trajectory(1.1)}
        bands = (BandDefinition(65, 95, 1), BandDefinition(165, 195, 1))
        vectors, eigenvalues, captures = fit_complex_band_modes(
            signals, signals.__getitem__, fs, bands, filter_order=4, trim_samples=250)

        self.assertGreater(abs(np.vdot(first, vectors[:, 0])), .995)
        self.assertGreater(abs(np.vdot(second, vectors[:, 1])), .995)
        self.assertTrue(np.all(captures > .999))
        self.assertEqual(len(eigenvalues), 2)

        result = transform_trajectory(
            trajectory(.7), fs, bands, vectors, filter_order=4, trim_samples=250)
        self.assertLess(result["selected_band_nrmse"], .01)
        self.assertEqual(result["y_osc"].shape, (11500, 4))
        np.testing.assert_allclose(result["y_osc"][:, 0::2], result["z_osc"].real)
        np.testing.assert_allclose(result["y_osc"][:, 1::2], result["z_osc"].imag)

    def test_real_decoder_matches_complex_reconstruction(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
        vectors, _ = np.linalg.qr(vectors)
        z = rng.normal(size=(20, 3)) + 1j * rng.normal(size=(20, 3))
        diagnostics = _decoder_diagnostics(vectors)
        state = np.empty((len(z), 2 * z.shape[1]))
        state[:, 0::2] = z.real
        state[:, 1::2] = z.imag
        expected = np.real(z @ vectors.conj().T)
        actual = state @ diagnostics["real_decoder"].T
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertEqual(diagnostics["complex_effective_rank"], 3)
        wide = _decoder_diagnostics(np.ones((2, 3), dtype=complex))
        self.assertIsNone(wide["complex_condition_number"])


if __name__ == "__main__":
    unittest.main()
