import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.pod.run_raw_highpass_pod_batch import (
    default_label,
    parse_args,
    run_batch,
)


class RunRawHighpassPodBatchTests(unittest.TestCase):
    def make_tree(self, root: Path) -> None:
        for condition, reps in (("0p04", (1, 2)), ("0p20", (1,))):
            for rep in reps:
                directory = root / condition / f"Ca_ac_test_rep{rep}"
                directory.mkdir(parents=True)
                (directory / f"raw_rep{rep}.hdf5").touch()

    def make_valid_tree(self, root: Path) -> None:
        directory = root / "0p04" / "Ca_ac_test_rep1"
        directory.mkdir(parents=True)
        path = directory / "raw_rep1.hdf5"
        time = np.arange(400, dtype=np.float32) / 1_000.0
        y, x = np.mgrid[:4, :4]
        pattern = (x - x.mean()) + 0.5 * (y - y.mean())
        with h5py.File(path, "w") as handle:
            meta = handle.create_group("meta")
            meta.attrs.update(
                t_units="seconds", x_units="microns", y_units="microns", z_units="microns"
            )
            meta.create_dataset("t", data=time)
            meta.create_dataset("x", data=np.arange(4, dtype=np.float32))
            meta.create_dataset("y", data=np.arange(4, dtype=np.float32))
            meta.create_dataset("frames", data=np.arange(400, dtype=np.int32))
            main = handle.create_group("main")
            for index, value in enumerate(time):
                field = (
                    3.0 * np.sin(2 * np.pi * 5 * value)
                    + 0.2 * np.sin(2 * np.pi * 100 * value) * pattern
                )
                main.create_dataset(f"{index:05d}", data=field.astype(np.float32))

    def test_default_pilot_configuration(self):
        args = parse_args(["--dry-run"])
        self.assertEqual(args.cutoff_hz, 50.0)
        self.assertEqual(args.spatial_stride, 4)
        self.assertEqual(args.pod_frames, 5_000)
        self.assertEqual(args.rank, 100)
        self.assertEqual(args.treatments, ["raw", "full_highpass"])
        self.assertEqual(
            default_label(args),
            "hp50_stride4_n5000_r100_raw-full_highpass",
        )

    def test_dry_run_filters_jobs_and_writes_manifest_without_computing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "raw"
            output_root = base / "output"
            scratch_root = base / "scratch"
            scratch_root.mkdir()
            self.make_tree(raw_root)
            args = parse_args(
                [
                    "--input-root",
                    str(raw_root),
                    "--output-root",
                    str(output_root),
                    "--scratch-root",
                    str(scratch_root),
                    "--powers",
                    "0p04",
                    "--reps",
                    "2",
                    "--dry-run",
                ]
            )
            log = io.StringIO()

            self.assertEqual(run_batch(args, log), 0)

            batch = output_root / default_label(args)
            manifest = (batch / "manifest.csv").read_text()
            self.assertIn("0p04", manifest)
            self.assertIn("Ca_ac_test_rep2", manifest)
            self.assertNotIn("Ca_ac_test_rep1", manifest)
            config = json.loads((batch / "configuration.json").read_text())
            self.assertEqual(config["spatial_stride"], 4)
            self.assertEqual(config["rank"], 100)
            output = log.getvalue()
            self.assertIn("--treatments raw full_highpass", output)
            self.assertIn("DRY RUN [1/1]", output)
            self.assertFalse((batch / "all_recordings_summary.csv").exists())

    def test_end_to_end_batch_is_resumable_and_aggregates_results(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw_root = base / "raw"
            output_root = base / "output"
            scratch_root = base / "scratch"
            scratch_root.mkdir()
            self.make_valid_tree(raw_root)
            argv = [
                "--input-root",
                str(raw_root),
                "--output-root",
                str(output_root),
                "--scratch-root",
                str(scratch_root),
                "--spatial-stride",
                "1",
                "--pod-frames",
                "100",
                "--rank",
                "2",
                "--oversampling",
                "1",
                "--power-iterations",
                "0",
                "--filter-columns",
                "4",
                "--blas-threads",
                "1",
            ]
            args = parse_args(argv)
            first_log = io.StringIO()

            self.assertEqual(run_batch(args, first_log), 0)

            batch = output_root / default_label(args)
            result = batch / "0p04" / "Ca_ac_test_rep1"
            for name in (
                "summary.csv",
                "mode_diagnostics.csv",
                "comparison.png",
                "completion.json",
            ):
                self.assertTrue((result / name).is_file(), name)
            self.assertTrue((batch / "all_recordings_summary.csv").is_file())
            self.assertTrue((batch / "condition_summary.csv").is_file())

            second_log = io.StringIO()
            self.assertEqual(run_batch(parse_args(argv), second_log), 0)
            self.assertIn("SKIP complete [1/1]", second_log.getvalue())


if __name__ == "__main__":
    unittest.main()
