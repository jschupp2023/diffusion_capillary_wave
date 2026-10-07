import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import h5py
import numpy as np

from data_analysis.energy.pod_gravitational_energy import highpass_spatial_mean
from data_analysis.pod.store_highpass_spatial_mean import (
    OUTPUT_GROUP,
    run,
    spatial_mean_energy_retained,
)


class StoreHighpassSpatialMeanTests(unittest.TestCase):
    def make_condition(self, root: Path) -> tuple[Path, dict[str, np.ndarray]]:
        condition = root / "0p20"
        repetition = condition / "Ca_ac_test_rep1"
        repetition.mkdir(parents=True)
        path = repetition / "pod_2d_r3.h5"
        time = np.arange(2_000, dtype=np.float64) / 1_000.0
        spatial_mean = 3.0 + np.sin(2 * np.pi * 5 * time) + 0.2 * np.sin(
            2 * np.pi * 80 * time
        )
        modes = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
        coefficients = np.arange(6_000, dtype=np.float32).reshape(2_000, 3)
        with h5py.File(path, "w") as handle:
            handle.attrs["rank"] = 3
            handle.create_dataset("pod/modes", data=modes)
            time_dataset = handle.create_dataset("grid/time", data=time)
            time_dataset.attrs["units"] = "seconds"
            mean_dataset = handle.create_dataset(
                "preprocessing/frame_spatial_mean", data=spatial_mean
            )
            mean_dataset.attrs["units"] = "microns"
            handle.create_dataset("reduced/coefficients", data=coefficients)
            handle.create_dataset("sentinel", data=np.array([11, 22, 33]))
        original = {
            "time": time,
            "spatial_mean": spatial_mean,
            "modes": modes,
            "coefficients": coefficients,
            "sentinel": np.array([11, 22, 33]),
        }
        return path, original

    def arguments(self, root: Path, *, dry_run: bool = False) -> Namespace:
        return Namespace(
            condition=Path("0p20"),
            root=root,
            rank=3,
            cutoffs_hz=[10.0, 20.0, 100.0],
            dry_run=dry_run,
        )

    def test_appends_filtered_trajectories_without_changing_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = self.make_condition(root)

            run(self.arguments(root))

            with h5py.File(path, "r") as handle:
                np.testing.assert_array_equal(handle["grid/time"], original["time"])
                np.testing.assert_array_equal(
                    handle["preprocessing/frame_spatial_mean"],
                    original["spatial_mean"],
                )
                np.testing.assert_array_equal(handle["pod/modes"], original["modes"])
                np.testing.assert_array_equal(
                    handle["reduced/coefficients"], original["coefficients"]
                )
                np.testing.assert_array_equal(handle["sentinel"], original["sentinel"])
                group = handle[OUTPUT_GROUP]
                self.assertEqual(
                    set(group),
                    {"cutoff_10_hz", "cutoff_20_hz", "cutoff_100_hz"},
                )
                for name, cutoff in (
                    ("cutoff_10_hz", 10.0),
                    ("cutoff_20_hz", 20.0),
                    ("cutoff_100_hz", 100.0),
                ):
                    dataset = group[name]
                    self.assertEqual(dataset.shape, original["spatial_mean"].shape)
                    self.assertEqual(dataset.dtype, np.dtype("float64"))
                    self.assertTrue(np.isfinite(dataset[:]).all())
                    self.assertEqual(float(dataset.attrs["cutoff_hz"]), cutoff)
                    self.assertEqual(dataset.attrs["units"], "microns")
                    self.assertEqual(
                        dataset.attrs["source_dataset"],
                        "/preprocessing/frame_spatial_mean",
                    )

            size_after_first_run = path.stat().st_size
            run(self.arguments(root))
            self.assertEqual(path.stat().st_size, size_after_first_run)

    def test_dry_run_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, original = self.make_condition(root)
            size_before = path.stat().st_size

            stdout = StringIO()
            with redirect_stdout(stdout):
                plans = run(self.arguments(root, dry_run=True))

            self.assertEqual(path.stat().st_size, size_before)
            with h5py.File(path, "r") as handle:
                self.assertNotIn(OUTPUT_GROUP, handle)
            expected = []
            for cutoff_hz in self.arguments(root).cutoffs_hz:
                filtered = highpass_spatial_mean(
                    original["spatial_mean"], original["time"], cutoff_hz
                )
                expected.append(
                    float(
                        np.mean(filtered**2)
                        / np.mean(original["spatial_mean"] ** 2)
                    )
                )
            np.testing.assert_allclose(
                [fraction for _, fraction in plans[0].energy_retained],
                expected,
                rtol=1e-14,
                atol=0.0,
            )
            self.assertIn("spatial-mean squared-norm energy retained", stdout.getvalue())
            self.assertIn("10 Hz: rho=", stdout.getvalue())

    def test_energy_retention_rejects_zero_energy_original(self):
        with self.assertRaisesRegex(ValueError, "zero mean-square energy"):
            spatial_mean_energy_retained(np.zeros(10), np.zeros(10))


if __name__ == "__main__":
    unittest.main()
