import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np

from data_analysis.pod.plot_highpass_spatial_mean import run


class PlotHighpassSpatialMeanTests(unittest.TestCase):
    def test_writes_combined_plot_to_condition_pod_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = root / "0p35"
            time = np.arange(400) / 1000.0
            for repetition in (1, 2):
                repetition_directory = condition / f"Ca_ac_test_rep{repetition}"
                repetition_directory.mkdir(parents=True)
                with h5py.File(repetition_directory / "pod_2d_r3.h5", "w") as handle:
                    handle.attrs["rank"] = 3
                    handle.create_dataset("pod/modes", data=np.zeros((3, 2, 2)))
                    time_dataset = handle.create_dataset("grid/time", data=time)
                    time_dataset.attrs["units"] = "seconds"
                    mean_dataset = handle.create_dataset(
                        "preprocessing/frame_spatial_mean",
                        data=np.sin(2 * np.pi * 5 * time) + repetition,
                    )
                    mean_dataset.attrs["units"] = "microns"

            output = run(
                Namespace(
                    condition=Path("0p35"), cutoff_hz=20.0, root=root,
                    output_dir=None, rank=3, dpi=40, max_repetitions=None,
                )
            )

            self.assertEqual(
                output,
                condition / "pod_analysis"
                / "all_repetitions_spatial_mean_highpass_20Hz_0p35.png",
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
