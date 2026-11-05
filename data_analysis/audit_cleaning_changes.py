"""Read-only audit of whether production cleaning would alter recordings.

The audit runs the same accepted spatial integer-cycle corrections and temporal
kinematic selector as :mod:`data_analysis.export_cleaned_dataset`, but never
writes reconstructed fields and never applies the 50 Hz high-pass filter.

For low-forcing recordings, frames that fail the production hybrid cliff gate
take a cheap bypass: they cannot receive a spatial correction, so phase-quality
and harmonic-continuation calculations are unnecessary. The temporally relevant
spatial mean and quadrant channels are still collected for every frame.

Examples
--------
Audit low-forcing conditions in the standard experiment tree::

    python data_analysis/audit_cleaning_changes.py \
        --input-root /disk/hyk049/DHM_new_experiment \
        --powers 0p04 0p08 --workers 8 \
        --output runs/cleaning_audits/low_forcing.json

Audit one recording::

    python data_analysis/audit_cleaning_changes.py \
        --input-file /path/to/recording.hdf5 --workers 8

The resulting JSON can be supplied to ``run_clean_dataset_batch.py`` with
``--audit-summary`` to materialize only recordings with accepted discrete
corrections.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/capillarywave-matplotlib-{os.getuid()}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np

from data_analysis.export_cleaned_dataset import (
    HYBRID_PARAMETERS,
    SPATIAL_QC_FIELDS,
    _restore_frame_hybrid,
    hybrid_cliff_gate_diagnostics,
)
from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.pod.phase_unwrap_gradient_poisson import restore_frame_poisson
from data_analysis.run_clean_dataset_batch import (
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    discover_jobs,
)
from data_analysis.temporal_alignment import (
    DEFAULT_Q,
    DEFAULT_TAU,
    extract_spatial_channels,
    select_temporal_kinematic,
)

AUDIT_SCHEMA_VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        help=f"Experiment tree root (default when no --input-file: {DEFAULT_INPUT_ROOT}).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        action="append",
        help="Audit one HDF5 recording; repeat for multiple recordings.",
    )
    parser.add_argument("--powers", nargs="+", help="Condition folders to scan, e.g. 0p04 0p08.")
    parser.add_argument("--reps", type=int, nargs="+", help="Optional repetition numbers.")
    parser.add_argument("--q", type=float, default=DEFAULT_Q, help="Optical cycle height [um].")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Spatial cliff threshold [um].")
    parser.add_argument(
        "--method",
        choices=("hybrid", "integer", "poisson"),
        default="hybrid",
        help="Production spatial method to audit (default: hybrid).",
    )
    parser.add_argument(
        "--no-temporal-align",
        dest="temporal_align",
        action="store_false",
        help="Audit only spatial corrections.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-files", type=int, help="Audit only the first N discovered recordings.")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Pilot scan of the first N frames. Partial scans cannot authorize deployment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Audit JSON path (default: timestamped path under runs/cleaning_audits).",
    )
    args = parser.parse_args()
    if args.input_file and args.input_root:
        parser.error("Use either --input-file or --input-root, not both.")
    if args.input_file and (args.powers or args.reps):
        parser.error("--powers/--reps apply only to --input-root discovery.")
    if args.workers < 1 or args.chunk_size < 1:
        parser.error("--workers and --chunk-size must be positive.")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2.")
    return args


def _audit_chunk_worker(
    input_path_str: str,
    start: int,
    end: int,
    q: float,
    tau: float,
    method: str,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact correction/QC channels without returning cleaned frames."""
    count = end - start
    channels = np.empty((count, 6), dtype=np.float64)
    modified_fraction = np.zeros(count, dtype=np.float64)
    gate_passed = np.zeros(count, dtype=bool)
    review_bits = np.zeros(count, dtype=np.uint8)
    raw_max_gradient = np.zeros(count, dtype=np.float64)
    restored_max_gradient = np.zeros(count, dtype=np.float64)
    qc_index = {name: index for index, name in enumerate(SPATIAL_QC_FIELDS)}

    with h5py.File(input_path_str, "r") as handle:
        main = handle["main"]
        for local_index, position in enumerate(range(start, end)):
            raw = np.asarray(main[_frame_key(position)], dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError(f"Frame /main/{_frame_key(position)} contains NaN or infinity.")

            if method == "hybrid":
                gate = hybrid_cliff_gate_diagnostics(raw, tau)
                gate_passed[local_index] = bool(gate["gate_passed"])
                raw_max_gradient[local_index] = float(gate["raw_max_gradient_um"])
                if gate_passed[local_index]:
                    restored, frame_qc = _restore_frame_hybrid(raw, q=q, tau=tau)
                    modified_fraction[local_index] = frame_qc[qc_index["modified_fraction"]]
                    review_bits[local_index] = np.uint8(frame_qc[qc_index["review_flag_bits"]])
                    restored_max_gradient[local_index] = frame_qc[
                        qc_index["restored_max_gradient_um"]
                    ]
                else:
                    # The production cleaner sets cycles=0 whenever this gate
                    # fails, irrespective of the phase-quality mask.
                    restored = raw
                    restored_max_gradient[local_index] = raw_max_gradient[local_index]
                    if raw_max_gradient[local_index] > 2.0 * tau:
                        review_bits[local_index] |= np.uint8(4)
            else:
                result = restore_frame_poisson(raw, q=q, tau=tau, method=method)
                restored = result.restored
                modified_fraction[local_index] = result.pct_pixels_modified / 100.0
                raw_max_gradient[local_index] = result.orig_max_dx
                restored_max_gradient[local_index] = result.rest_max_dx

            channels[local_index] = extract_spatial_channels(restored)

    return (
        start,
        end,
        channels,
        modified_fraction,
        gate_passed,
        review_bits,
        np.column_stack((raw_max_gradient, restored_max_gradient)),
    )


def audit_recording(
    input_path: Path,
    *,
    q: float = DEFAULT_Q,
    tau: float = DEFAULT_TAU,
    method: str = "hybrid",
    temporal_align: bool = True,
    workers: int = 8,
    chunk_size: int = 500,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Audit one recording and return a JSON-serializable summary."""
    input_path = input_path.expanduser().resolve()
    full_info = inspect_input(input_path, 0, None)
    n_source_frames = full_info.n_frames
    n_frames = min(n_source_frames, max_frames) if max_frames is not None else n_source_frames
    info = full_info if n_frames == n_source_frames else inspect_input(input_path, 0, n_frames)
    n_pixels = int(np.prod(info.frame_shape))

    channels = np.empty((n_frames, 6), dtype=np.float64)
    modified_fraction = np.zeros(n_frames, dtype=np.float64)
    gate_passed = np.zeros(n_frames, dtype=bool)
    review_bits = np.zeros(n_frames, dtype=np.uint8)
    gradients = np.zeros((n_frames, 2), dtype=np.float64)
    chunks = [(start, min(start + chunk_size, n_frames)) for start in range(0, n_frames, chunk_size)]

    started = time.perf_counter()
    completed = 0
    report_every = max(1, len(chunks) // 10)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _audit_chunk_worker,
                str(input_path),
                start,
                end,
                q,
                tau,
                method,
            )
            for start, end in chunks
        ]
        for future in as_completed(futures):
            start, end, chunk_channels, chunk_modified, chunk_gate, chunk_review, chunk_grad = (
                future.result()
            )
            channels[start:end] = chunk_channels
            modified_fraction[start:end] = chunk_modified
            gate_passed[start:end] = chunk_gate
            review_bits[start:end] = chunk_review
            gradients[start:end] = chunk_grad
            completed += 1
            if completed % report_every == 0 or completed == len(chunks):
                print(
                    f"  {input_path.name}: {completed}/{len(chunks)} chunks "
                    f"({100.0 * completed / len(chunks):.0f}%)",
                    flush=True,
                )

    temporal_result = None
    if temporal_align:
        mean_increment = np.r_[0.0, np.diff(channels[:, 0])]
        quadrant_increment = np.vstack(
            (np.zeros((1, 4)), np.diff(channels[:, 2:], axis=0))
        )
        temporal_result = select_temporal_kinematic(
            mean_increment,
            quadrant_increment,
            q=q,
        )

    spatial_positions = np.flatnonzero(modified_fraction > 0)
    temporal_positions = (
        np.flatnonzero(temporal_result.shifts != 0)
        if temporal_result is not None
        else np.empty(0, dtype=np.int64)
    )
    temporal_affected = (
        np.flatnonzero(temporal_result.offsets != 0)
        if temporal_result is not None
        else np.empty(0, dtype=np.int64)
    )
    uncertainty_positions = (
        np.flatnonzero(temporal_result.uncertainty_flags)
        if temporal_result is not None
        else np.empty(0, dtype=np.int64)
    )
    complete = n_frames == n_source_frames
    accepted_change = bool(spatial_positions.size or temporal_positions.size)
    decision_valid = bool(complete)

    if decision_valid and accepted_change:
        recommendation = "materialize_cleaned_dataset"
    elif decision_valid:
        recommendation = "reuse_raw_apply_50hz_during_pod"
    else:
        recommendation = "partial_audit_only_scan_full_recording_before_deployment"

    result: dict[str, Any] = {
        "input_file": str(input_path),
        "source_size_bytes": int(input_path.stat().st_size),
        "source_mtime_ns": int(input_path.stat().st_mtime_ns),
        "n_source_frames": int(n_source_frames),
        "n_scanned_frames": int(n_frames),
        "complete_recording": complete,
        "deployment_decision_valid": decision_valid,
        "q_um": float(q),
        "tau_um": float(tau),
        "spatial_method": method,
        "temporal_alignment_audited": bool(temporal_align),
        "highpass_filter_included": False,
        "accepted_discrete_correction_detected": accepted_change,
        "requires_materialized_cleaned_dataset": bool(decision_valid and accepted_change),
        "recommendation": recommendation,
        "spatial": {
            "gate_passed_frame_count": int(np.count_nonzero(gate_passed)),
            "modified_frame_count": int(spatial_positions.size),
            "modified_pixel_count": int(np.rint(np.sum(modified_fraction) * n_pixels)),
            "max_modified_frame_fraction": float(modified_fraction.max(initial=0.0)),
            "modified_frame_positions": spatial_positions.astype(int).tolist(),
            "modified_frame_ids": info.frame_numbers[spatial_positions].astype(int).tolist(),
            "max_raw_gradient_um": float(gradients[:, 0].max(initial=0.0)),
            "max_restored_gradient_um": float(gradients[:, 1].max(initial=0.0)),
            "review_counts": {
                "any": int(np.count_nonzero(review_bits)),
                "boundary_growth": int(np.count_nonzero(review_bits & 1)),
                "unresolved_boundary": int(np.count_nonzero(review_bits & 2)),
                "gate_skipped_severe_gradient": int(np.count_nonzero(review_bits & 4)),
                "mask_above_70pct": int(np.count_nonzero(review_bits & 8)),
                "gate_passed_empty_mask": int(np.count_nonzero(review_bits & 16)),
            },
        },
        "temporal": {
            "correction_event_count": int(temporal_positions.size),
            "affected_frame_count": int(temporal_affected.size),
            "large_correction_count": int(
                temporal_result.n_large_corrections if temporal_result is not None else 0
            ),
            "small_correction_count": int(
                temporal_result.n_small_corrections if temporal_result is not None else 0
            ),
            "uncertain_correction_count": int(uncertainty_positions.size),
            "max_abs_accumulated_offset_cycles": int(
                np.max(np.abs(temporal_result.offsets), initial=0)
                if temporal_result is not None
                else 0
            ),
            "max_residual_increment_um": float(
                temporal_result.max_residual_increment_um if temporal_result is not None else 0.0
            ),
            "correction_frame_positions": temporal_positions.astype(int).tolist(),
            "correction_frame_ids": info.frame_numbers[temporal_positions].astype(int).tolist(),
            "uncertain_frame_positions": uncertainty_positions.astype(int).tolist(),
            "uncertain_frame_ids": info.frame_numbers[uncertainty_positions].astype(int).tolist(),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    if method == "hybrid":
        result["hybrid_parameters"] = dict(HYBRID_PARAMETERS)
        result["audit_fast_path"] = (
            "Frames failing the exact production cliff gate bypass phase/mask analysis; "
            "their production spatial correction is identically zero."
        )
    return result


def _discover_inputs(args: argparse.Namespace) -> tuple[list[Path], list[dict[str, str]]]:
    if args.input_file:
        return [path.expanduser().resolve() for path in args.input_file], []

    input_root = (args.input_root or DEFAULT_INPUT_ROOT).expanduser().resolve()
    jobs, issues = discover_jobs(
        input_root=input_root,
        output_root=DEFAULT_OUTPUT_ROOT,
        powers=args.powers,
        reps=args.reps,
        method=args.method,
    )
    return [job.input_path.resolve() for job in jobs], [
        {"directory": str(issue.directory), "reason": issue.reason} for issue in issues
    ]


def main() -> None:
    args = parse_args()
    inputs, discovery_issues = _discover_inputs(args)
    if args.max_files is not None:
        inputs = inputs[: args.max_files]
    if not inputs:
        raise RuntimeError("No input recordings were discovered.")

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path("runs/cleaning_audits") / f"cleaning_audit_{timestamp}.json"
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=== Read-only cleaning-change audit ===")
    print(f"Recordings: {len(inputs)} | method={args.method} | q={args.q:.7g} um | tau={args.tau:g} um")
    print("50 Hz high-pass: EXCLUDED from audit and deployment decision")

    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for index, input_path in enumerate(inputs, start=1):
        print(f"\n[{index}/{len(inputs)}] {input_path}", flush=True)
        try:
            result = audit_recording(
                input_path,
                q=args.q,
                tau=args.tau,
                method=args.method,
                temporal_align=args.temporal_align,
                workers=args.workers,
                chunk_size=args.chunk_size,
                max_frames=args.max_frames,
            )
            results.append(result)
            print(
                "  accepted corrections: "
                f"spatial_frames={result['spatial']['modified_frame_count']}, "
                f"temporal_events={result['temporal']['correction_event_count']} -> "
                f"{result['recommendation']}",
                flush=True,
            )
        except Exception as exc:
            failed.append(
                {
                    "input_file": str(input_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  FAILED: {failed[-1]['error']}", flush=True)

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "read-only accepted discrete cleaning correction audit",
        "highpass_filter_included": False,
        "settings": {
            "q_um": float(args.q),
            "tau_um": float(args.tau),
            "spatial_method": args.method,
            "temporal_alignment_audited": bool(args.temporal_align),
            "max_frames": args.max_frames,
        },
        "recording_count": len(results),
        "materialize_count": int(
            sum(bool(result["requires_materialized_cleaned_dataset"]) for result in results)
        ),
        "reuse_raw_count": int(
            sum(result["recommendation"] == "reuse_raw_apply_50hz_during_pod" for result in results)
        ),
        "discovery_issues": discovery_issues,
        "failed": failed,
        "recordings": results,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"\nAudit written to {args.output}")
    print(
        f"Materialize: {payload['materialize_count']} | reuse raw: {payload['reuse_raw_count']} "
        f"| failed: {len(failed)}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
