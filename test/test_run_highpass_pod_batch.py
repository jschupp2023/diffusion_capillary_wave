import io
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.pod.run_highpass_pod_batch import (
    parse_args,
    run_batch,
    validate_args,
)


class RunHighpassPodBatchTests(unittest.TestCase):
    def make_recording(self, root: Path) -> Path:
        directory = root / "0p00" / "rep1"
        directory.mkdir(parents=True)
        path = directory / "test_data_roi-none_cal-true_100k.hdf5"
        n_frames = 400
        sampling_hz = 1_000.0
        time = np.arange(n_frames, dtype=np.float64) / sampling_hz
        y, x = np.mgrid[:4, :5]
        pattern = (x - x.mean()) + 0.4 * (y - y.mean())
        frame_numbers = np.arange(10, 10 + n_frames, dtype=np.int32)
        with h5py.File(path, "w") as output:
            output.attrs["test_attribute"] = "preserved"
            meta = output.create_group("meta")
            meta.attrs.update(
                t_units="seconds",
                x_units="microns",
                y_units="microns",
                z_units="microns",
            )
            meta.create_dataset("t", data=time)
            meta.create_dataset("x", data=np.arange(5, dtype=np.float32))
            meta.create_dataset("y", data=np.arange(4, dtype=np.float32))
            meta.create_dataset("frames", data=frame_numbers)
            main = output.create_group("main")
            for frame_number, instant in zip(frame_numbers, time):
                field = (
                    3.0 * np.sin(2 * np.pi * 5 * instant)
                    + 0.3 * np.sin(2 * np.pi * 100 * instant) * pattern
                )
                main.create_dataset(
                    f"{frame_number:05d}", data=field.astype(np.float32)
                )
        return path

    def test_end_to_end_writes_production_pod_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "raw"
            output_root = base / "reduced"
            scratch_root = base / "scratch"
            source = self.make_recording(raw_root)
            args = parse_args(
                [
                    "--input-root",
                    str(raw_root),
                    "--output-root",
                    str(output_root),
                    "--scratch-root",
                    str(scratch_root),
                    "--powers",
                    "0p00",
                    "--rank",
                    "2",
                    "--training-frames",
                    "100",
                    "--oversampling",
                    "1",
                    "--power-iterations",
                    "0",
                    "--batch-size",
                    "64",
                    "--filter-row-chunk",
                    "2",
                    "--blas-threads",
                    "1",
                    "--compression",
                    "none",
                ]
            )
            validate_args(args)
            first_log = io.StringIO()

            self.assertEqual(run_batch(args, first_log), 0)

            result = output_root / "0p00" / "rep1" / "pod_2d_r2.h5"
            self.assertTrue(result.is_file())
            with h5py.File(result, "r") as pod:
                self.assertEqual(pod.attrs["source_file"], str(source.resolve()))
                self.assertTrue(pod.attrs["input_full_field_highpass_applied"])
                self.assertEqual(pod.attrs["input_highpass_cutoff_hz"], 50.0)
                self.assertFalse(pod.attrs["instantaneous_spatial_mean_removed"])
                self.assertTrue(pod.attrs["temporal_mean_field_removed"])
                self.assertEqual(pod["reduced/coefficients"].shape, (400, 2))
                np.testing.assert_array_equal(
                    pod["source/frame_number"][:], np.arange(10, 410)
                )
            self.assertEqual(list(scratch_root.iterdir()), [])

            second_log = io.StringIO()
            self.assertEqual(run_batch(args, second_log), 0)
            self.assertIn("SKIP existing", second_log.getvalue())


if __name__ == "__main__":
    unittest.main()
