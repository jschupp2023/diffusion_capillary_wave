"""Validate low-rank spatial comparisons against explicit spatial matrices."""
import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform

from data_analysis.ensemble.compare_mean_covariance_repetitions import (
    held_out_from_inner, spatial_covariance_distances,
)


class SpatialCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(14)
        self.factors = self.rng.normal(size=(6, 3, 11))
        self.covariances = np.array([f.T @ f for f in self.factors])

    def test_distances_match_explicit_reconstruction(self):
        distance, _ = spatial_covariance_distances(self.factors)
        explicit = squareform(pdist(self.covariances.reshape(6, -1))) / 11
        np.testing.assert_allclose(distance, explicit, atol=1e-12)

    def test_independent_basis_rotations_preserve_distances(self):
        rotated = []
        for factor in self.factors:
            rotation, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
            rotated.append(rotation @ factor)
        original, _ = spatial_covariance_distances(self.factors)
        changed, _ = spatial_covariance_distances(np.array(rotated))
        np.testing.assert_allclose(changed, original, atol=1e-12)

    def test_held_out_reference_is_mean_covariance(self):
        _, inner = spatial_covariance_distances(self.factors)
        labels = np.repeat(['A', 'B'], 3)
        errors, relative = held_out_from_inner(inner, labels)
        for i in range(6):
            others = np.flatnonzero((labels == labels[i]) & (np.arange(6) != i))
            reference = self.covariances[others].mean(axis=0)
            difference = np.linalg.norm(self.covariances[i]-reference)
            self.assertAlmostEqual(errors[i], difference / 11)
            self.assertAlmostEqual(relative[i], difference / np.linalg.norm(reference))

    def test_zero_covariance_has_zero_absolute_error(self):
        distance, inner = spatial_covariance_distances(np.zeros((4, 2, 5)))
        errors, relative = held_out_from_inner(inner, ['A', 'A', 'B', 'B'])
        np.testing.assert_array_equal(distance, 0)
        np.testing.assert_array_equal(errors, 0)
        self.assertTrue(np.isnan(relative).all())


if __name__ == '__main__':
    unittest.main()
