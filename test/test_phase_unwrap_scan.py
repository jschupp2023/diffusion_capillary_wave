import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.pod.phase_unwrap_scan import (
    frame_ambiguity_metrics,
    scan_recording,
)


def _write_recording(path: Path, fields: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        main = handle.create_group("main")
        meta = handle.create_group("meta")
        meta.create_dataset("t", data=np.arange(len(fields), dtype=float))
        meta.create_dataset("x", data=np.arange(fields.shape[2], dtype=float))
        meta.create_dataset("y", data=np.arange(fields.shape[1], dtype=float))
        meta.create_dataset("frames", data=np.arange(len(fields)))
        meta.attrs["z_units"] = "microns"
        for index, field in enumerate(fields):
            main.create_dataset(f"{index:05d}", data=field)


class PhaseUnwrapScanTests(unittest.TestCase):
    def test_frame_metrics_detect_ambiguous_wrapped_cells(self):
        clean = np.zeros((8, 9))
        ambiguous = clean.copy()
        vortex_phase = np.array(
            [[0.86055566, -1.44647274], [-2.88414841, -3.03774646]]
        )
        ambiguous[3:5, 4:6] = vortex_phase / (2 * np.pi)
        clean_metrics = frame_ambiguity_metrics(clean, q=1.0)
        ambiguous_metrics = frame_ambiguity_metrics(ambiguous, q=1.0)
        self.assertEqual(clean_metrics["wrapped_residue_fraction"], 0.0)
        self.assertGreater(ambiguous_metrics["wrapped_residue_fraction"], 0.0)
        self.assertGreater(ambiguous_metrics["edge_abs_p99"], clean_metrics["edge_abs_p99"])

    def test_scan_writes_separate_read_only_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.h5"
            output = root / "scan"
            y, x = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 13), indexing="ij")
            fields = np.stack([0.05 * index + 0.1 * x + 0.1 * y for index in range(9)])
            fields[5, 5:, 6:] += 4.0
            _write_recording(source, fields)
            before = source.read_bytes()

            returned = scan_recording(
                source, 0, None, 2, 1.0, output_dir=output, robust_z=2.0, dpi=30
            )

            self.assertEqual(returned, output.resolve())
            self.assertEqual(source.read_bytes(), before)
            for name in ("metrics.csv", "summary.json", "timeline.png"):
                self.assertTrue((output / name).is_file())
            with (output / "summary.json").open() as handle:
                summary = json.load(handle)
            self.assertTrue(summary["source_opened_read_only"])
            self.assertEqual(summary["sample_count"], 5)
            with (output / "metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["source_position"]) for row in rows], [0, 2, 4, 6, 8])


if __name__ == "__main__":
    unittest.main()
