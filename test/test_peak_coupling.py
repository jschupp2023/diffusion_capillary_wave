"""Scientific controls for peak coupling, not just implementation snapshots."""
import unittest
import numpy as np

from data_analysis.correlations.peak_coupling import (
    holm, lag_correlations, pair_correlations, phase_surrogate, shifted_null,
    targeted_bicoherence,
)


class PeakCouplingTests(unittest.TestCase):
    def test_phase_locked_triad_vs_random_biphase(self):
        rng = np.random.default_rng(14)
        fs, nseg, windows = 1024, 256, 60
        time = np.arange(nseg*windows)/fs
        phase = np.repeat(rng.uniform(-np.pi, np.pi, (windows, 3)), nseg, axis=0)
        frequencies = np.array([64., 96., 160.])
        independent = np.exp(1j*(2*np.pi*time[:, None]*frequencies+phase))
        locked = independent.copy()
        locked[:, 2] = np.exp(1j*(2*np.pi*time*160+phase[:, 0]+phase[:, 1]+.4))
        cells = np.array([[0, 1, 2, 16, 24, 40]])
        owners = np.array([0])
        a = targeted_bicoherence(locked, cells, owners, nseg)[0][0]
        b = targeted_bicoherence(independent, cells, owners, nseg)[0][0]
        self.assertGreater(a, .7)
        self.assertLess(b, .15)

    def test_surrogate_preserves_full_cross_spectrum(self):
        rng = np.random.default_rng(93)
        for n in [511, 512]:
            c = rng.normal(size=(n, 3))+1j*rng.normal(size=(n, 3))
            c[:, 1] += .8*c[:, 0]
            surrogate = phase_surrogate(c, rng)
            a, b = np.fft.fft(c, axis=0), np.fft.fft(surrogate, axis=0)
            np.testing.assert_allclose(a[:, :, None]*a[:, None, :].conj(),
                                       b[:, :, None]*b[:, None, :].conj(), rtol=1e-11, atol=1e-10)

    def test_coherent_lines_are_not_automatically_evidence_against_phase_null(self):
        # A fixed triad of pure tones has b²~1 even without a dynamical coupling.
        # Random Fourier phases must retain that null difficulty, not manufacture
        # significance by independently scrambling overlapping time windows.
        time = np.arange(4096)/1024
        c = np.exp(2j*np.pi*time[:, None]*np.array([64., 96., 160.]))
        cells = np.array([[0, 1, 2, 16, 24, 40]])
        owners = np.array([0])
        surrogate = phase_surrogate(c, np.random.default_rng(8))
        a = targeted_bicoherence(c, cells, owners, 256)[0][0]
        b = targeted_bicoherence(surrogate, cells, owners, 256)[0][0]
        self.assertGreater(a, .999)
        self.assertGreater(b, .999)

    def test_envelope_dependence_and_positive_lag_convention(self):
        rng = np.random.default_rng(42)
        common = np.convolve(rng.normal(size=2000), np.ones(15)/15, mode='same')
        delayed = np.r_[rng.normal(size=8), common[:-8]]
        x = np.column_stack((common, delayed))
        pairs = np.array([[0, 1]])
        lags = np.arange(-12, 13)
        curve = lag_correlations(x, pairs, lags)[0]
        self.assertEqual(lags[np.argmax(curve)], 8)
        aligned = np.column_stack((common, common+.1*rng.normal(size=len(common))))
        actual = pair_correlations(aligned, pairs)[0]
        null = shifted_null(aligned, pairs, 999, rng, 1000.)[:, 0]
        self.assertGreater(actual, np.quantile(null, .999))

    def test_holm_controls_family_in_original_order(self):
        np.testing.assert_allclose(holm([.04, .001, .03]), [.06, .003, .06])
        self.assertEqual(len(holm([])), 0)


if __name__ == '__main__':
    unittest.main()
