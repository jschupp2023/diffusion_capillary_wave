from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from data_analysis.audit_cleaning_changes import audit_recording
from data_analysis.run_clean_dataset_batch import CleanJob, select_jobs_from_audit
from data_analysis.temporal_alignment import DEFAULT_Q, DEFAULT_TAU


def _write_recording(path: Path, fields: np.ndarray) -> None:
    n_frames, ny, nx = fields.shape
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("t", data=np.arange(n_frames, dtype=np.float64) / 115200.0)
        meta.create_dataset("x", data=np.arange(nx, dtype=np.float64))
        meta.create_dataset("y", data=np.arange(ny, dtype=np.float64))
        meta.create_dataset("frames", data=np.arange(n_frames, dtype=np.int64))
        main = handle.create_group("main")
        for index, field in enumerate(fields):
            main.create_dataset(f"{index:05d}", data=np.asarray(field, dtype=np.float32))


def test_audit_bypasses_clean_low_forcing_recording(tmp_path: Path) -> None:
    raw = tmp_path / "clean.hdf5"
    fields = np.zeros((8, 16, 16), dtype=np.float32)
    _write_recording(raw, fields)

    result = audit_recording(raw, workers=1, chunk_size=3)

    assert result["highpass_filter_included"] is False
    assert result["spatial"]["gate_passed_frame_count"] == 0
    assert result["spatial"]["modified_frame_count"] == 0
    assert result["temporal"]["correction_event_count"] == 0
    assert result["accepted_discrete_correction_detected"] is False
    assert result["requires_materialized_cleaned_dataset"] is False
    assert result["recommendation"] == "reuse_raw_apply_50hz_during_pod"


def test_audit_detects_temporal_integer_cycle_event(tmp_path: Path) -> None:
    raw = tmp_path / "temporal_slip.hdf5"
    fields = np.zeros((10, 16, 16), dtype=np.float32)
    fields[4:] += 4.0 * DEFAULT_Q
    _write_recording(raw, fields)

    result = audit_recording(raw, workers=1, chunk_size=4)

    assert result["spatial"]["modified_frame_count"] == 0
    assert result["temporal"]["correction_event_count"] == 1
    assert result["temporal"]["correction_frame_positions"] == [4]
    assert result["temporal"]["max_abs_accumulated_offset_cycles"] == 4
    assert result["requires_materialized_cleaned_dataset"] is True
    assert result["recommendation"] == "materialize_cleaned_dataset"


def test_partial_audit_cannot_authorize_deployment(tmp_path: Path) -> None:
    raw = tmp_path / "partial.hdf5"
    _write_recording(raw, np.zeros((10, 16, 16), dtype=np.float32))

    result = audit_recording(raw, workers=1, chunk_size=3, max_frames=5)

    assert result["complete_recording"] is False
    assert result["deployment_decision_valid"] is False
    assert result["requires_materialized_cleaned_dataset"] is False
    assert result["recommendation"].startswith("partial_audit_only")


def test_batch_manifest_selects_only_changed_recordings(tmp_path: Path) -> None:
    unchanged = tmp_path / "unchanged.hdf5"
    changed = tmp_path / "changed.hdf5"
    _write_recording(unchanged, np.zeros((8, 16, 16), dtype=np.float32))
    changed_fields = np.zeros((8, 16, 16), dtype=np.float32)
    changed_fields[4:] += 4.0 * DEFAULT_Q
    _write_recording(changed, changed_fields)

    unchanged_result = audit_recording(unchanged, workers=1, chunk_size=4)
    changed_result = audit_recording(changed, workers=1, chunk_size=4)
    manifest = tmp_path / "audit.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "highpass_filter_included": False,
                "settings": {
                    "q_um": DEFAULT_Q,
                    "tau_um": DEFAULT_TAU,
                    "spatial_method": "hybrid",
                    "temporal_alignment_audited": True,
                },
                "recordings": [unchanged_result, changed_result],
            }
        ),
        encoding="utf-8",
    )
    jobs = [
        CleanJob("0p04", "rep1", unchanged, tmp_path / "out1.hdf5"),
        CleanJob("0p20", "rep1", changed, tmp_path / "out2.hdf5"),
    ]

    selected, reuse_raw = select_jobs_from_audit(
        jobs,
        manifest,
        q=DEFAULT_Q,
        tau=DEFAULT_TAU,
        method="hybrid",
        temporal_align=True,
    )

    assert [job.input_path for job in selected] == [changed]
    assert [job.input_path for job in reuse_raw] == [unchanged]
