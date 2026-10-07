import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.pod.phase_unwrap_diagnostic import (
    correct_window,
    integer_consistency,
    load_raw_window,
    run_diagnostic,
)


def _write_raw(path: Path, fields: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        main = handle.create_group("main")
        meta = handle.create_group("meta")
        meta.create_dataset("t", data=np.arange(len(fields), dtype=float) * 0.1)
        meta.create_dataset("x", data=np.linspace(-2.0, 2.0, fields.shape[2]))
        meta.create_dataset("y", data=np.linspace(-1.0, 1.0, fields.shape[1]))
        meta.create_dataset("frames", data=np.arange(10, 10 + len(fields)))
        meta.attrs["t_units"] = "seconds"
        meta.attrs["x_units"] = "microns"
        meta.attrs["y_units"] = "microns"
        meta.attrs["z_units"] = "microns"
        for number, field in zip(range(10, 10 + len(fields)), fields):
            main.create_dataset(f"{number:05d}", data=field)


class PhaseUnwrapDiagnosticTests(unittest.TestCase):
    def setUp(self):
        y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, 9), np.linspace(-2.0, 2.0, 11), indexing="ij"
        )
        base = 0.8 * x + 0.55 * y + 0.12 * x * y
        self.q = 1.25
        self.clean = np.stack([base + 0.08 * index for index in range(5)])
        self.recorded = self.clean.copy()
        # Local integer-cycle artifacts disappear on rewrapping and should be repaired.
        self.recorded[:, 2:6, 6:9] += 2 * self.q
        self.recorded[3:, 6:8, 1:4] -= self.q
        self.recorded[2, 4, 5] = np.nan

    def test_window_uses_source_positions_and_clips_neighbors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.h5"
            _write_raw(source, self.recorded)
            window = load_raw_window(source, timestep=0, neighbors=3)
            np.testing.assert_array_equal(window.info.source_positions, [0, 1, 2, 3])
            np.testing.assert_array_equal(window.info.frame_numbers, [10, 11, 12, 13])
            self.assertEqual(window.target_local_index, 0)
            self.assertTrue(window.invalid[2, 4, 5])

    def test_both_methods_recover_synthetic_branches_and_integer_corrections(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.h5"
            _write_raw(source, self.recorded)
            window = load_raw_window(source, timestep=2, neighbors=2)
            result = correct_window(window, self.q)
            valid = ~window.invalid
            np.testing.assert_allclose(result.independent_2d[valid], self.clean[valid], atol=1e-10)
            np.testing.assert_allclose(result.joint_3d[valid], self.clean[valid], atol=1e-10)
            for corrected in (result.independent_2d, result.joint_3d):
                check = integer_consistency(corrected, window.fields, window.invalid, self.q)
                self.assertLess(check["max_distance_to_integer_cycles"], 1e-10)
            # The physical +0.08 per-frame mean drift was retained, not demeaned.
            means = np.nanmean(result.independent_2d, axis=(1, 2))
            np.testing.assert_allclose(np.diff(means), 0.08, atol=1e-10)

    def test_end_to_end_outputs_are_separate_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.h5"
            output = root / "diagnostic"
            _write_raw(source, self.recorded)
            before = source.read_bytes()

            returned = run_diagnostic(
                source, timestep=2, neighbors=1, q=self.q,
                output_dir=output, dpi=35,
            )

            self.assertEqual(returned, output.resolve())
            self.assertEqual(source.read_bytes(), before)
            for name in (
                "target_surfaces.png", "target_line_profiles.png",
                "target_correction_cycles.png", "window_diagnostics.png",
                "corrections.npz", "summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "summary.json").open() as handle:
                summary = json.load(handle)
            self.assertTrue(summary["source_opened_read_only"])
            self.assertFalse(summary["periodic_boundary_connections"])
            self.assertEqual(summary["q"], self.q)
            self.assertFalse(
                summary["global_cycle_alignment"]["spatial_mean_removed_per_frame"]
            )
            self.assertIn("target_candidate_assessment", summary)
            self.assertTrue(summary["automatic_action"].startswith("preserve_original"))
            with np.load(output / "corrections.npz") as saved:
                self.assertEqual(saved["corrected_joint_3d"].shape, (3, 9, 11))

            with self.assertRaises(FileExistsError):
                run_diagnostic(
                    source, timestep=2, neighbors=1, q=self.q,
                    output_dir=output, dpi=35,
                )


if __name__ == "__main__":
    unittest.main()
