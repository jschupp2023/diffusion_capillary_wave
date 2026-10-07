"""Compare ensemble-moment algebra with explicit spatial reconstructions."""
import unittest
import numpy as np

from data_analysis.ensemble.compare_ensemble_curve_repetitions import (
    curve_sizes, distances, moment_kernels, snapshot_gram, weights,
)


class EnsembleCurveTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.u = rng.normal(size=(8, 3, 7))
        self.a = rng.normal(size=(17, 8, 3))
        self.z = np.einsum('trk,rkp->trp', self.a, self.u)
        self.overlap = (self.u.reshape(24, 7) @ self.u.reshape(24, 7).T).reshape(8, 3, 8, 3)

    def explicit(self, indices):
        z = self.z[:, indices]
        mean = z.mean(axis=1)
        centered = z-mean[:, None]
        covariance = np.einsum('trp,trq->tpq', centered, centered)/(len(indices)-1)
        return mean, covariance

    def test_snapshot_gram_uses_physical_coordinates(self):
        gram = snapshot_gram(self.a, self.a, self.overlap, 7)
        explicit = np.einsum('tip,tjp->tij', self.z, self.z)/7
        np.testing.assert_allclose(gram, explicit, atol=1e-12)

    def test_different_sizes_and_overlapping_subsets(self):
        gram = snapshot_gram(self.a, self.a, self.overlap, 7)
        kernels = moment_kernels(gram)
        left_indices, right_indices = [0, 1, 3], list(range(8))
        result = distances(weights([left_indices], 8), weights([right_indices], 8), kernels, kernels, kernels)
        left, right = self.explicit(left_indices), self.explicit(right_indices)
        for m in range(2):
            self.assertAlmostEqual(result[m][0], np.sqrt(np.mean((left[m]-right[m])**2)))

    def test_cross_power_kernels(self):
        aa = snapshot_gram(self.a[:, :4], self.a[:, :4], self.overlap[:4, :, :4], 7)
        bb = snapshot_gram(self.a[:, 4:], self.a[:, 4:], self.overlap[4:, :, 4:], 7)
        ab = snapshot_gram(self.a[:, :4], self.a[:, 4:], self.overlap[:4, :, 4:], 7)
        result = distances(weights([[0, 2, 3]], 4), weights([[0, 1]], 4),
                           moment_kernels(aa), moment_kernels(bb), moment_kernels(ab))
        left, right = self.explicit([0, 2, 3]), self.explicit([4, 5])
        for m in range(2):
            self.assertAlmostEqual(result[m][0], np.sqrt(np.mean((left[m]-right[m])**2)))

    def test_norm_curves_match_explicit_statistics(self):
        gram = snapshot_gram(self.a, self.a, self.overlap, 7)
        subset = [1, 3, 5, 6]
        result = curve_sizes(gram, subset)
        explicit = self.explicit(subset)
        np.testing.assert_allclose(result[0], np.sqrt(np.mean(explicit[0]**2, axis=1)))
        np.testing.assert_allclose(result[1], np.sqrt(np.mean(explicit[1]**2, axis=(1, 2))))


if __name__ == '__main__':
    unittest.main()
