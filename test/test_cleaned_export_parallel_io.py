from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from data_analysis.export_cleaned_dataset import (
    SPATIAL_QC_FIELDS,
    _restore_frame_hybrid,
    export_cleaned_dataset,
)
from data_analysis.pod.phase_unwrap_gradient_poisson import restore_frame_poisson
from data_analysis.temporal_alignment import DEFAULT_Q, DEFAULT_TAU


def _write_recording(path: Path, fields: np.ndarray) -> None:
    n_frames, ny, nx = fields.shape
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("t", data=np.arange(n_frames) / 115200.0)
        meta.create_dataset("x", data=np.arange(nx, dtype=np.float64))
        meta.create_dataset("y", data=np.arange(ny, dtype=np.float64))
        meta.create_dataset("frames", data=np.arange(n_frames, dtype=np.int64))
        main = handle.create_group("main")
        for index, field in enumerate(fields):
            main.create_dataset(f"{index:05d}", data=field.astype(np.float32))


def test_parallel_workers_write_exact_fields_directly_to_scratch(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    fields = rng.normal(scale=0.05, size=(7, 16, 18)).astype(np.float32)
    # Exercise both unchanged frames and integer-cycle restoration.
    fields[2, :, 9:] += np.float32(3 * DEFAULT_Q)
    fields[5, 8:, :] -= np.float32(4 * DEFAULT_Q)

    source = tmp_path / "source.hdf5"
    output = tmp_path / "cleaned.hdf5"
    scratch = tmp_path / "scratch"
    _write_recording(source, fields)
    source_stat = (source.stat().st_size, source.stat().st_mtime_ns)

    export_cleaned_dataset(
        raw_input=source,
        output_file=output,
        q=DEFAULT_Q,
        tau=DEFAULT_TAU,
        method="integer",
        temporal_align=False,
        apply_highpass_50hz=False,
        workers=2,
        chunk_size=2,
        scratch_dir=scratch,
        stage_input_to_scratch=True,
    )

    expected = np.stack(
        [
            restore_frame_poisson(
                frame,
                q=DEFAULT_Q,
                tau=DEFAULT_TAU,
                method="integer",
            ).restored
            for frame in fields
        ]
    )
    with h5py.File(output, "r") as handle:
        actual = np.stack([handle[f"main/{index:05d}"][:] for index in range(7)])
    np.testing.assert_array_equal(actual, expected)

    assert source_stat == (source.stat().st_size, source.stat().st_mtime_ns)
    assert list(scratch.glob("mmap_spatial_*.dat")) == []
    assert list(scratch.glob("staged_input_*.hdf5")) == []


def test_export_rejects_invalid_parallel_settings(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    _write_recording(source, np.zeros((2, 4, 4), dtype=np.float32))

    for workers, chunk_size in ((0, 1), (1, 0)):
        try:
            export_cleaned_dataset(
                raw_input=source,
                output_file=tmp_path / f"bad_{workers}_{chunk_size}.hdf5",
                workers=workers,
                chunk_size=chunk_size,
            )
        except ValueError as exc:
            assert "workers and chunk_size must be positive" in str(exc)
        else:
            raise AssertionError("Invalid parallel settings were accepted")


def test_hybrid_cliff_gate_skips_expensive_phase_diagnostics() -> None:
    clean = np.zeros((16, 16), dtype=np.float32)
    with patch(
        "data_analysis.export_cleaned_dataset.wrapped_phase_quality",
        side_effect=AssertionError("phase diagnostics should be bypassed"),
    ):
        restored, qc = _restore_frame_hybrid(
            clean,
            q=DEFAULT_Q,
            tau=DEFAULT_TAU,
        )

    np.testing.assert_array_equal(restored, clean)
    qc_by_name = dict(zip(SPATIAL_QC_FIELDS, qc, strict=True))
    assert qc_by_name["modified_fraction"] == 0.0
    assert qc_by_name["cliff_gate_passed"] == 0.0
    assert np.isnan(qc_by_name["mask_fraction"])
