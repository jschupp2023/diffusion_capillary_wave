import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.energy.pod_gravitational_energy import compute, highpass_spatial_mean


class PodGravitationalEnergyTests(unittest.TestCase):
    @staticmethod
    def _write_file(path, time, spatial_mean):
        time, spatial_mean = np.asarray(time), np.asarray(spatial_mean)
        with h5py.File(path, "w") as f:
            f.attrs["instantaneous_spatial_mean_removed"] = True
            f.attrs["temporal_mean_field_removed"] = True
            f.create_dataset("pod/modes", data=np.ones((1, 3, 3)))
            f.create_dataset("grid/time", data=time)
            for axis in ("x", "y"):
                dataset = f.create_dataset(f"grid/{axis}", data=[0.0, 1.0, 2.0])
                dataset.attrs["units"] = "m"
            coefficients = f.create_dataset(
                "reduced/coefficients", data=np.full((len(time), 1), 2.0)
            )
            coefficients.attrs["units"] = "m"
            temporal_mean = f.create_dataset(
                "preprocessing/temporal_mean_field", data=np.full((3, 3), 3.0)
            )
            saved_spatial_mean = f.create_dataset(
                "preprocessing/frame_spatial_mean", data=spatial_mean
            )
            for dataset in (temporal_mean, saved_spatial_mean):
                dataset.attrs["units"] = "m"
                dataset.attrs["subtracted_before_pod"] = True

    def test_half_rho_g_prefactor_with_and_without_spatial_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pod.h5"
            self._write_file(path, [0.0], [7.0])

            _, _, without_mean, with_mean = compute(
                path, rank=1, frames=1, seed=0, rho=2.0, gravity=3.0
            )

            area = 4.0
            np.testing.assert_allclose(without_mean, [0.5 * 2.0 * 3.0 * 5.0**2 * area])
            np.testing.assert_allclose(with_mean, [0.5 * 2.0 * 3.0 * 12.0**2 * area])

    def test_highpass_is_applied_to_full_spatial_mean_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pod.h5"
            time = np.linspace(0.0, 2.0, 201)
            self._write_file(path, time, np.full(len(time), 7.0))

            _, _, without_mean, with_mean = compute(
                path, rank=1, frames=len(time), seed=0, rho=2.0, gravity=3.0,
                spatial_mean_highpass_hz=5.0,
            )

            np.testing.assert_allclose(with_mean, without_mean, atol=1e-10)

    def test_highpass_suppresses_low_frequency_and_preserves_high_frequency(self):
        time = np.arange(4000) / 1000.0
        low = np.sin(2 * np.pi * 2 * time)
        high = np.sin(2 * np.pi * 100 * time)
        filtered = highpass_spatial_mean(low + high, time, 20.0)
        interior = slice(500, -500)
        low_projection = 2 * np.mean(filtered[interior] * low[interior])
        high_projection = 2 * np.mean(filtered[interior] * high[interior])
        self.assertLess(abs(low_projection), .01)
        self.assertAlmostEqual(high_projection, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
