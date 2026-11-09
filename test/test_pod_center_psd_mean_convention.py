import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.psd.pod_center_psd import reconstruct_point_signals


class PodCenterPsdMeanConventionTests(unittest.TestCase):
    def make_pod(self, path: Path, mean_removed: bool) -> None:
        with h5py.File(path, "w") as output:
            output.attrs["instantaneous_spatial_mean_removed"] = mean_removed
            pod = output.create_group("pod")
            modes = np.zeros((2, 2, 2), dtype=np.float32)
            modes[:, 1, 1] = [2.0, -1.0]
            pod.create_dataset("modes", data=modes)
            reduced = output.create_group("reduced")
            coefficients = reduced.create_dataset(
                "coefficients",
                data=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            )
            coefficients.attrs["units"] = "microns"
            preprocessing = output.create_group("preprocessing")
            mean = preprocessing.create_dataset(
                "frame_spatial_mean", data=np.asarray([10.0, 20.0, 30.0])
            )
            mean.attrs["subtracted_before_pod"] = mean_removed
            grid = output.create_group("grid")
            grid.create_dataset("time", data=np.arange(3, dtype=float))
            grid.create_dataset("x", data=np.arange(2, dtype=float))
            grid.create_dataset("y", data=np.arange(2, dtype=float))

    def test_restores_mean_only_when_it_was_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expected_modal = np.asarray([0.0, 2.0, 4.0])
            for removed in (False, True):
                path = base / f"pod_{removed}.h5"
                self.make_pod(path, removed)
                _, modal, mean, physical, *_ = reconstruct_point_signals(
                    path, 2, None, None, 2, show_progress=False
                )
                np.testing.assert_allclose(modal, expected_modal)
                np.testing.assert_allclose(mean, [10.0, 20.0, 30.0])
                np.testing.assert_allclose(
                    physical,
                    expected_modal + mean if removed else expected_modal,
                )


if __name__ == "__main__":
    unittest.main()
