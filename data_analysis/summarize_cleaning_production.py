"""Collect compact production-cleaning statistics without copying field data.

The script reads only metadata and the small per-frame diagnostic arrays stored
in cleaned HDF5 files.  It writes JSON, CSV, and Markdown summaries that can be
copied off the lab computer instead of the large cleaned recordings.

An audit JSON produced by :mod:`data_analysis.audit_cleaning_changes` may also
be supplied.  Complete audit recordings that were not materialized are then
included, which is useful for documenting low-forcing conditions for which the
production cleaner accepted no correction.

Example
-------
python -m data_analysis.summarize_cleaning_production \
    --cleaned-root /disk/hyk049/jschupp/cleaned_experiment \
    --audit-json /path/to/cleaning_audit.json \
    --powers 0p0 0p07 0p08 0p10 0p15 0p18 0p20 0p25 \
    --output-dir /disk/hyk049/jschupp/cleaning_statistics
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import h5py
import numpy as np


HDF5_SUFFIXES = {".h5", ".hdf5"}
CONDITION_PATTERN = re.compile(r"0p\d{1,2}")
VPP_PATTERN = re.compile(r"0[._](\d+)vpp", re.IGNORECASE)
REVIEW_FLAGS = {
    "boundary_growth": 1,
    "unresolved_boundary": 2,
    "gate_skipped_severe_gradient": 4,
    "mask_above_70pct": 8,
    "gate_passed_empty_mask": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        required=True,
        help="Root containing materialized cleaned HDF5 recordings.",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional cleaning-audit JSON. May be supplied multiple times. "
            "Only complete recordings absent from the cleaned tree are added."
        ),
    )
    parser.add_argument(
        "--powers",
        nargs="+",
        help="Restrict output to conditions such as 0p08 0p20.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the compact JSON, CSV, and Markdown summaries.",
    )
    parser.add_argument(
        "--top-frames",
        type=int,
        default=10,
        help="Number of most strongly modified frames retained per recording.",
    )
    return parser.parse_args()


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _condition_from_paths(*paths: str | Path) -> str:
    for raw in paths:
        if not raw:
            continue
        path = Path(str(raw))
        for part in path.parts:
            if CONDITION_PATTERN.fullmatch(part):
                return part
        match = VPP_PATTERN.search(path.name)
        if match:
            hundredths = int(round(float(f"0.{match.group(1)}") * 100.0))
            return "0p0" if hundredths == 0 else f"0p{hundredths:02d}"
    return "unknown"


def _condition_sort_key(condition: str) -> tuple[int, float | str]:
    if CONDITION_PATTERN.fullmatch(condition):
        return (0, float(condition.replace("p", ".")))
    return (1, condition)


def _first_frame_shape(h5: h5py.File) -> tuple[int, ...] | None:
    if "main" not in h5:
        return None
    main = h5["main"]
    for value in main.values():
        if isinstance(value, h5py.Dataset):
            return tuple(int(v) for v in value.shape)
    return None


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in ("p0", "p50", "p90", "p99", "p100")}
    return {
        key: float(np.quantile(finite, probability))
        for key, probability in (
            ("p0", 0.0),
            ("p50", 0.5),
            ("p90", 0.9),
            ("p99", 0.99),
            ("p100", 1.0),
        )
    }


def _read_array(group: h5py.Group, name: str, *, default: Any) -> np.ndarray:
    if name not in group:
        return np.asarray(default)
    return np.asarray(group[name][...])


def summarize_cleaned_file(path: Path, top_frames: int) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        if "cleaning_diagnostics/spatial" not in h5:
            raise ValueError("missing /cleaning_diagnostics/spatial")
        spatial = h5["cleaning_diagnostics/spatial"]
        modified = np.asarray(spatial["modified_fraction"][...], dtype=np.float64)
        n_frames = int(modified.size)
        frame_shape = _first_frame_shape(h5)
        if frame_shape is None or len(frame_shape) != 2:
            raise ValueError(f"could not determine 2-D frame shape, found {frame_shape}")
        n_pixels = int(np.prod(frame_shape))

        frame_ids = (
            np.asarray(h5["meta/frames"][...], dtype=np.int64)
            if "meta/frames" in h5
            else np.arange(n_frames, dtype=np.int64)
        )
        if frame_ids.size != n_frames:
            raise ValueError("frame-ID and diagnostic lengths differ")

        source_file = str(_scalar(h5.attrs.get("source_file", "")))
        condition = _condition_from_paths(path, source_file)
        gate = _read_array(spatial, "cliff_gate_passed", default=np.zeros(n_frames, bool)).astype(bool)
        mask = _read_array(spatial, "mask_fraction", default=np.full(n_frames, np.nan)).astype(float)
        initial_mask = _read_array(
            spatial, "initial_mask_fraction", default=np.full(n_frames, np.nan)
        ).astype(float)
        raw_gradient = _read_array(
            spatial, "raw_max_gradient_um", default=np.full(n_frames, np.nan)
        ).astype(float)
        restored_gradient = _read_array(
            spatial, "restored_max_gradient_um", default=np.full(n_frames, np.nan)
        ).astype(float)
        review_bits = _read_array(
            spatial, "review_flag_bits", default=np.zeros(n_frames, np.int32)
        ).astype(np.int64)
        growth_added = _read_array(
            spatial, "boundary_growth_added_pixels", default=np.zeros(n_frames, np.int32)
        ).astype(np.int64)
        unresolved = _read_array(
            spatial,
            "unresolved_boundary_violation_pixels",
            default=np.zeros(n_frames, np.int32),
        ).astype(np.int64)

        temporal_offsets = (
            np.asarray(h5["meta/temporal_offset_cycles"][...], dtype=np.int64)
            if "meta/temporal_offset_cycles" in h5
            else np.zeros(n_frames, dtype=np.int64)
        )
        uncertainty = (
            np.asarray(h5["meta/temporal_uncertainty_flags"][...], dtype=bool)
            if "meta/temporal_uncertainty_flags" in h5
            else np.zeros(n_frames, dtype=bool)
        )

        modified_positions = np.flatnonzero(modified > 0)
        order = modified_positions[np.argsort(modified[modified_positions])[::-1]]
        order = order[: max(0, top_frames)]
        top = [
            {
                "frame_position": int(position),
                "frame_id": int(frame_ids[position]),
                "modified_fraction": float(modified[position]),
                "mask_fraction": _finite_or_none(mask[position]),
                "raw_max_gradient_um": _finite_or_none(raw_gradient[position]),
                "restored_max_gradient_um": _finite_or_none(restored_gradient[position]),
                "review_flag_bits": int(review_bits[position]),
            }
            for position in order
        ]

        review_counts = {
            "any": int(np.count_nonzero(review_bits)),
            **{
                name: int(np.count_nonzero(review_bits & bit))
                for name, bit in REVIEW_FLAGS.items()
            },
        }
        modified_pixel_count = int(np.rint(np.sum(modified) * n_pixels))
        total_pixel_observations = int(n_frames * n_pixels)
        finite_mask = np.where(np.isfinite(mask), mask, 0.0)

        return {
            "source_kind": "cleaned_hdf5",
            "condition": condition,
            "realization": path.parent.name,
            "cleaned_file": str(path),
            "source_file": source_file,
            "cleaning_version": str(_scalar(h5.attrs.get("cleaning_version", "unknown"))),
            "spatial_method": str(_scalar(h5.attrs.get("spatial_method", "unknown"))),
            "q_um": _finite_or_none(h5.attrs.get("q_microns")),
            "tau_um": _finite_or_none(h5.attrs.get("tau_microns")),
            "n_frames": n_frames,
            "frame_shape": list(frame_shape),
            "n_pixels_per_frame": n_pixels,
            "total_pixel_observations": total_pixel_observations,
            "spatial": {
                "gate_passed_frame_count": int(np.count_nonzero(gate)),
                "modified_frame_count": int(modified_positions.size),
                "modified_pixel_count": modified_pixel_count,
                "modified_frame_fraction": float(modified_positions.size / n_frames),
                "modified_pixel_fraction": float(modified_pixel_count / total_pixel_observations),
                "masked_pixel_count": int(np.rint(np.sum(finite_mask) * n_pixels)),
                "modified_fraction_quantiles_all_frames": _quantiles(modified),
                "modified_fraction_quantiles_modified_frames": _quantiles(
                    modified[modified_positions]
                ),
                "final_mask_fraction_quantiles_evaluated_frames": _quantiles(mask),
                "initial_mask_fraction_quantiles_evaluated_frames": _quantiles(initial_mask),
                "max_raw_gradient_um": _finite_or_none(np.nanmax(raw_gradient, initial=np.nan)),
                "max_restored_gradient_um": _finite_or_none(
                    np.nanmax(restored_gradient, initial=np.nan)
                ),
                "boundary_growth_frame_count": int(np.count_nonzero(growth_added > 0)),
                "boundary_growth_added_pixel_count": int(np.sum(growth_added)),
                "unresolved_boundary_frame_count": int(np.count_nonzero(unresolved > 0)),
                "unresolved_boundary_pixel_count": int(np.sum(unresolved)),
                "review_counts": review_counts,
                "top_modified_frames": top,
            },
            "temporal": {
                "alignment_enabled": bool(_scalar(h5.attrs.get("temporal_dc_aligned", False))),
                "correction_event_count": int(
                    _scalar(h5.attrs.get("temporal_corrections_count", 0))
                ),
                "affected_frame_count": int(np.count_nonzero(temporal_offsets)),
                "large_correction_count": int(
                    _scalar(h5.attrs.get("temporal_large_corrections_count", 0))
                ),
                "small_correction_count": int(
                    _scalar(h5.attrs.get("temporal_small_corrections_count", 0))
                ),
                "uncertain_correction_count": int(np.count_nonzero(uncertainty)),
                "max_abs_accumulated_offset_cycles": int(
                    np.max(np.abs(temporal_offsets), initial=0)
                ),
                "max_residual_increment_um": _finite_or_none(
                    h5.attrs.get("max_residual_increment_um")
                ),
            },
            "highpass": {
                "applied": bool(_scalar(h5.attrs.get("highpass_filter_applied", False))),
                "cutoff_hz": _finite_or_none(h5.attrs.get("highpass_cutoff_hz")),
            },
        }


def _raw_frame_pixels(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as h5:
            shape = _first_frame_shape(h5)
        return int(np.prod(shape)) if shape is not None and len(shape) == 2 else None
    except OSError:
        return None


def normalize_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    input_file = str(record.get("input_file", ""))
    n_frames = int(record.get("n_scanned_frames", record.get("n_source_frames", 0)))
    spatial = dict(record.get("spatial", {}))
    temporal = dict(record.get("temporal", {}))
    n_pixels = _raw_frame_pixels(Path(input_file))
    modified_pixels = int(spatial.get("modified_pixel_count", 0))
    total_pixels = n_frames * n_pixels if n_pixels is not None else None
    modified_frames = int(spatial.get("modified_frame_count", 0))
    return {
        "source_kind": "audit_only",
        "condition": _condition_from_paths(input_file),
        "realization": Path(input_file).parent.name,
        "source_file": input_file,
        "complete_recording": bool(record.get("complete_recording", False)),
        "q_um": _finite_or_none(record.get("q_um")),
        "tau_um": _finite_or_none(record.get("tau_um")),
        "n_frames": n_frames,
        "n_pixels_per_frame": n_pixels,
        "total_pixel_observations": total_pixels,
        "spatial": {
            "gate_passed_frame_count": int(spatial.get("gate_passed_frame_count", 0)),
            "modified_frame_count": modified_frames,
            "modified_pixel_count": modified_pixels,
            "modified_frame_fraction": modified_frames / n_frames if n_frames else None,
            "modified_pixel_fraction": (
                modified_pixels / total_pixels if total_pixels else (0.0 if modified_pixels == 0 else None)
            ),
            "max_raw_gradient_um": _finite_or_none(spatial.get("max_raw_gradient_um")),
            "max_restored_gradient_um": _finite_or_none(
                spatial.get("max_restored_gradient_um")
            ),
            "review_counts": dict(spatial.get("review_counts", {})),
        },
        "temporal": {
            "alignment_enabled": bool(record.get("temporal_alignment_audited", False)),
            "correction_event_count": int(temporal.get("correction_event_count", 0)),
            "affected_frame_count": int(temporal.get("affected_frame_count", 0)),
            "large_correction_count": int(temporal.get("large_correction_count", 0)),
            "small_correction_count": int(temporal.get("small_correction_count", 0)),
            "uncertain_correction_count": int(temporal.get("uncertain_correction_count", 0)),
            "max_abs_accumulated_offset_cycles": int(
                temporal.get("max_abs_accumulated_offset_cycles", 0)
            ),
            "max_residual_increment_um": _finite_or_none(
                temporal.get("max_residual_increment_um")
            ),
        },
        "highpass": {"applied": False, "cutoff_hz": None},
    }


def discover_cleaned_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in HDF5_SUFFIXES
        and not path.name.endswith(".tmp")
    )


def _sum_nested(records: Iterable[dict[str, Any]], section: str, field: str) -> int:
    return int(sum(int(record.get(section, {}).get(field, 0) or 0) for record in records))


def aggregate_condition(condition: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    n_frames = sum(int(record["n_frames"]) for record in records)
    known_total_pixels = [record.get("total_pixel_observations") for record in records]
    total_pixels = (
        int(sum(int(value) for value in known_total_pixels))
        if all(value is not None for value in known_total_pixels)
        else None
    )
    modified_frames = _sum_nested(records, "spatial", "modified_frame_count")
    modified_pixels = _sum_nested(records, "spatial", "modified_pixel_count")
    raw_maxima = [
        record["spatial"].get("max_raw_gradient_um")
        for record in records
        if record["spatial"].get("max_raw_gradient_um") is not None
    ]
    restored_maxima = [
        record["spatial"].get("max_restored_gradient_um")
        for record in records
        if record["spatial"].get("max_restored_gradient_um") is not None
    ]
    review_counts = {
        name: int(
            sum(int(record["spatial"].get("review_counts", {}).get(name, 0)) for record in records)
        )
        for name in ("any", *REVIEW_FLAGS.keys())
    }
    return {
        "condition": condition,
        "recording_count": len(records),
        "cleaned_recording_count": sum(record["source_kind"] == "cleaned_hdf5" for record in records),
        "audit_only_recording_count": sum(record["source_kind"] == "audit_only" for record in records),
        "n_frames": n_frames,
        "total_pixel_observations": total_pixels,
        "spatial_gate_passed_frame_count": _sum_nested(records, "spatial", "gate_passed_frame_count"),
        "spatial_modified_frame_count": modified_frames,
        "spatial_modified_frame_fraction": modified_frames / n_frames if n_frames else None,
        "spatial_modified_pixel_count": modified_pixels,
        "spatial_modified_pixel_fraction": (
            modified_pixels / total_pixels if total_pixels else (0.0 if modified_pixels == 0 else None)
        ),
        "max_raw_gradient_um": max(raw_maxima) if raw_maxima else None,
        "max_restored_gradient_um": max(restored_maxima) if restored_maxima else None,
        "boundary_growth_frame_count": _sum_nested(records, "spatial", "boundary_growth_frame_count"),
        "unresolved_boundary_frame_count": _sum_nested(
            records, "spatial", "unresolved_boundary_frame_count"
        ),
        "review_counts": review_counts,
        "temporal_correction_event_count": _sum_nested(records, "temporal", "correction_event_count"),
        "temporal_affected_frame_count": _sum_nested(records, "temporal", "affected_frame_count"),
        "temporal_uncertain_correction_count": _sum_nested(
            records, "temporal", "uncertain_correction_count"
        ),
    }


def _percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.6g}%"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _recording_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        spatial = record["spatial"]
        temporal = record["temporal"]
        rows.append(
            {
                "condition": record["condition"],
                "realization": record["realization"],
                "source_kind": record["source_kind"],
                "n_frames": record["n_frames"],
                "n_pixels_per_frame": record.get("n_pixels_per_frame"),
                "gate_passed_frames": spatial.get("gate_passed_frame_count"),
                "modified_frames": spatial.get("modified_frame_count"),
                "modified_frame_fraction": spatial.get("modified_frame_fraction"),
                "modified_pixels": spatial.get("modified_pixel_count"),
                "modified_pixel_fraction": spatial.get("modified_pixel_fraction"),
                "max_raw_gradient_um": spatial.get("max_raw_gradient_um"),
                "max_restored_gradient_um": spatial.get("max_restored_gradient_um"),
                "review_frames": spatial.get("review_counts", {}).get("any", 0),
                "boundary_growth_frames": spatial.get("boundary_growth_frame_count", 0),
                "unresolved_boundary_frames": spatial.get("unresolved_boundary_frame_count", 0),
                "temporal_correction_events": temporal.get("correction_event_count", 0),
                "temporal_affected_frames": temporal.get("affected_frame_count", 0),
                "temporal_uncertain_corrections": temporal.get("uncertain_correction_count", 0),
                "source_file": record.get("source_file"),
                "cleaned_file": record.get("cleaned_file"),
            }
        )
    return rows


def _write_markdown(path: Path, conditions: list[dict[str, Any]], issues: list[dict[str, str]]) -> None:
    lines = [
        "# Production cleaning statistics",
        "",
        "Spatial counts describe the masked-harmonic stage before temporal alignment and the 50 Hz filter.",
        "",
        "| Forcing | Recs (clean/audit) | Frames | Spatially modified frames | Modified pixels | Review frames | Temporal events |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        lines.append(
            "| {condition} | {recording_count} ({cleaned_recording_count}/{audit_only_recording_count}) "
            "| {n_frames:,} | {spatial_modified_frame_count:,} ({frame_pct}) "
            "| {spatial_modified_pixel_count:,} ({pixel_pct}) | {review:,} | {temporal:,} |".format(
                **row,
                frame_pct=_percent(row["spatial_modified_frame_fraction"]),
                pixel_pct=_percent(row["spatial_modified_pixel_fraction"]),
                review=row["review_counts"]["any"],
                temporal=row["temporal_correction_event_count"],
            )
        )
    if issues:
        lines.extend(["", "## Files not summarized", ""])
        lines.extend(f"- `{item['path']}`: {item['reason']}" for item in issues)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cleaned_root = args.cleaned_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected = set(args.powers or [])
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_records: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    files = discover_cleaned_files(cleaned_root)
    print(f"Discovered {len(files)} candidate HDF5 files below {cleaned_root}")
    for index, path in enumerate(files, start=1):
        try:
            record = summarize_cleaned_file(path, args.top_frames)
            if selected and record["condition"] not in selected:
                continue
            cleaned_records.append(record)
            print(
                f"[{index}/{len(files)}] {record['condition']}/{record['realization']}: "
                f"{record['spatial']['modified_frame_count']:,} modified frames"
            )
        except (OSError, KeyError, ValueError) as exc:
            issues.append({"path": str(path), "reason": str(exc)})

    cleaned_sources = {record["source_file"] for record in cleaned_records if record["source_file"]}
    audit_records: list[dict[str, Any]] = []
    for audit_path in args.audit_json:
        audit_path = audit_path.expanduser().resolve()
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        for source in payload.get("recordings", []):
            if not source.get("complete_recording", False):
                continue
            record = normalize_audit_record(source)
            if record["source_file"] in cleaned_sources:
                continue
            if selected and record["condition"] not in selected:
                continue
            audit_records.append(record)

    records = sorted(
        [*cleaned_records, *audit_records],
        key=lambda row: (_condition_sort_key(row["condition"]), row["realization"], row["source_file"]),
    )
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)
    conditions = [
        aggregate_condition(condition, by_condition[condition])
        for condition in sorted(by_condition, key=_condition_sort_key)
    ]

    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cleaned_root": str(cleaned_root),
        "audit_jsons": [str(path.expanduser().resolve()) for path in args.audit_json],
        "selected_powers": sorted(selected, key=_condition_sort_key) if selected else None,
        "notes": [
            "Spatial diagnostics precede temporal alignment and high-pass filtering.",
            "Modified pixels are output pixels shifted by a nonzero integer multiple of q.",
            "Audit-only records are complete recordings absent from the cleaned HDF5 tree.",
        ],
        "condition_summary": conditions,
        "recordings": records,
        "issues": issues,
    }
    json_path = output_dir / "cleaning_statistics.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    condition_fields = [
        "condition",
        "recording_count",
        "cleaned_recording_count",
        "audit_only_recording_count",
        "n_frames",
        "total_pixel_observations",
        "spatial_gate_passed_frame_count",
        "spatial_modified_frame_count",
        "spatial_modified_frame_fraction",
        "spatial_modified_pixel_count",
        "spatial_modified_pixel_fraction",
        "max_raw_gradient_um",
        "max_restored_gradient_um",
        "boundary_growth_frame_count",
        "unresolved_boundary_frame_count",
        "temporal_correction_event_count",
        "temporal_affected_frame_count",
        "temporal_uncertain_correction_count",
    ]
    _write_csv(output_dir / "cleaning_statistics_by_condition.csv", conditions, condition_fields)
    recording_rows = _recording_csv_rows(records)
    _write_csv(
        output_dir / "cleaning_statistics_by_recording.csv",
        recording_rows,
        list(recording_rows[0].keys()) if recording_rows else ["condition"],
    )
    _write_markdown(output_dir / "cleaning_statistics_summary.md", conditions, issues)

    print(
        f"Wrote {len(records)} recordings across {len(conditions)} conditions to {output_dir}"
    )
    for name in (
        "cleaning_statistics.json",
        "cleaning_statistics_by_condition.csv",
        "cleaning_statistics_by_recording.csv",
        "cleaning_statistics_summary.md",
    ):
        print(f"  {output_dir / name}")
    if issues:
        print(f"Warning: {len(issues)} candidate HDF5 files were not summarized; see JSON/Markdown.")


if __name__ == "__main__":
    main()
