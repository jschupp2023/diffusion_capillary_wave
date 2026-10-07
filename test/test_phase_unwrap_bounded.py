import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data_analysis.pod.phase_unwrap_bounded import (
    bounded_correction,
    load_raw_interval,
    run_bounded_diagnostic,
    spatial_metrics,
)


def _write_raw(path: Path, fields: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        main = handle.create_group("main")
        meta = handle.create_group("meta")
        meta.create_dataset("t", data=np.arange(len(fields), dtype=float) * 0.1)
        meta.create_dataset("x", data=np.arange(fields.shape[2], dtype=float))
        meta.create_dataset("y", data=np.arange(fields.shape[1], dtype=float))
        meta.create_dataset("frames", data=np.arange(len(fields)))
        meta.attrs["z_units"] = "microns"
        for index, field in enumerate(fields):
            main.create_dataset(f"{index:05d}", data=field)


class BoundedPhaseUnwrapTests(unittest.TestCase):
    def setUp(self):
        y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, 12), np.linspace(-1.0, 1.0, 13), indexing="ij"
        )
        self.q = 2.0
        self.clean = np.stack([0.15 * x + 0.12 * y + 0.05 * t for t in range(5)])
        self.recorded = self.clean.copy()
        self.recorded[1:4, 3:9, 5:11] += 5 * self.q

    def test_joint_solver_reduces_large_edges_with_fixed_clean_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.h5"
            _write_raw(source, self.recorded)
            window = load_raw_interval(source, 0, 4, 2)
            result = bounded_correction(
                window,
                self.q,
                carrier_degree=2,
                carrier_iterations=3,
                force=True,
            )
            before = spatial_metrics(
                window.fields, window.invalid, self.q / 2, self.q
            )["p999"][2]
            after = spatial_metrics(
                result.corrected, window.invalid, self.q / 2, self.q
            )["p999"][2]
            self.assertLess(after, before)
            self.assertEqual(np.count_nonzero(result.correction_cycles[0]), 0)
            self.assertEqual(np.count_nonzero(result.correction_cycles[-1]), 0)
            np.testing.assert_allclose(
                result.corrected - window.fields,
                self.q * result.correction_cycles,
                atol=1e-12,
            )

    def test_smooth_target_gate_is_exact_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.h5"
            _write_raw(source, self.clean)
            window = load_raw_interval(source, 0, 4, 2)
            result = bounded_correction(window, self.q, carrier_degree=2)
            self.assertTrue(result.gate_preserved_original)
            np.testing.assert_array_equal(result.corrected, window.fields)
            self.assertEqual(np.count_nonzero(result.correction_cycles), 0)

    def test_end_to_end_outputs_are_separate_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.h5"
            output = root / "bounded"
            _write_raw(source, self.recorded)
            before = source.read_bytes()
            returned = run_bounded_diagnostic(
                source,
                0,
                4,
                2,
                self.q,
                output_dir=output,
                carrier_degree=2,
                carrier_iterations=3,
                force=True,
                dpi=30,
            )
            self.assertEqual(returned, output.resolve())
            self.assertEqual(source.read_bytes(), before)
            for name in (
                "target_comparison.png",
                "window_diagnostics.png",
                "bounded_candidate.npz",
                "summary.json",
            ):
                self.assertTrue((output / name).is_file())
            with (output / "summary.json").open() as handle:
                summary = json.load(handle)
            self.assertTrue(summary["source_opened_read_only"])
            self.assertEqual(summary["target"]["source_position"], 2)


if __name__ == "__main__":
    unittest.main()
