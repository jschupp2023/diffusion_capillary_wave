"""Run data cleaning and high-pass filtering sequentially over the experiment tree.

Expected input layout::

    INPUT_ROOT/
      0p20/
        Ca_ac_0p001660_rep1/
          11272025_c_0.2vpp_data_roi-none_cal-true.hdf5

The same relative directory structure is created below ``--output-root``::

    OUTPUT_ROOT/
      0p20/
        Ca_ac_0p001660_rep1/
          11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5

Example
-------
    # 1. Dry run to inspect discovered files:
    python data_analysis/run_clean_dataset_batch.py --powers 0p20 --dry-run

    # 2. Pilot test on just the first file:
    python data_analysis/run_clean_dataset_batch.py --powers 0p20 --max-files 1

    # 3. Full unattended batch run on all 0p20 files:
    python data_analysis/run_clean_dataset_batch.py --powers 0p20 --workers 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import TextIO

# Some cleaning dependencies import Matplotlib for optional diagnostics. Keep
# its cache off read-only home directories on compute nodes.
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/capillarywave-matplotlib-{os.getuid()}")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.export_cleaned_dataset import export_cleaned_dataset
from data_analysis.temporal_alignment import DEFAULT_Q, DEFAULT_TAU

DEFAULT_INPUT_ROOT = Path("/disk/hyk049/DHM_new_experiment")
DEFAULT_OUTPUT_ROOT = Path("/disk/hyk049/jschupp/cleaned_experiment")
CONDITION_PATTERN = re.compile(r"0p\d{1,2}")
# Most conditions use ``Ca_ac_..._repN`` directories, while 0p07 uses the
# shorter ``repN`` convention. Both contain one direct raw HDF5 recording.
REALIZATION_PATTERN = re.compile(r"(?:Ca_ac.*|rep\d+)")
REPETITION_NUMBER_PATTERN = re.compile(r"(?:^|_)rep(\d+)(?:$|_)")
HDF5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class CleanJob:
    condition: str
    realization: str
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class DiscoveryIssue:
    directory: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Input tree root (default: {DEFAULT_INPUT_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root of the mirrored output tree (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--powers",
        nargs="+",
        help="Only these condition directories (e.g. --powers 0p20 or --powers 0p08 0p20).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        nargs="+",
        help="Only these repetition numbers within every selected power.",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=DEFAULT_Q,
        help=f"Optical cycle height in microns (default: {DEFAULT_Q}).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help=f"Spatial unwrapping threshold in microns (default: {DEFAULT_TAU}).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="integer",
        choices=["integer", "poisson", "hybrid"],
        help="Spatial restoration method (default: integer).",
    )
    parser.add_argument(
        "--no-temporal-align",
        dest="temporal_align",
        action="store_false",
        help="Disable temporal alignment.",
    )
    parser.add_argument(
        "--no-highpass-50hz",
        dest="apply_highpass_50hz",
        action="store_false",
        help="Disable 50 Hz Butterworth high-pass filter.",
    )
    parser.add_argument(
        "--highpass-cutoff-hz",
        type=float,
        default=50.0,
        help="High-pass filter cutoff frequency in Hz (default: 50.0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker processes per file for spatial restoration (default: 8).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Frame chunk size for spatial worker pool (default: 500).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Run only the first N valid inputs; useful for pilot testing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Subsample only the first M frames of each recording (for quick testing).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        help="Directory for temporary memory maps.",
    )
    parser.add_argument(
        "--stage-input-to-scratch",
        action="store_true",
        help=(
            "Copy each input recording into --scratch-dir before cleaning. "
            "Use this with node-local SSD or /dev/shm, not the source filesystem."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cleaned outputs; otherwise they are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and print jobs without running the cleaner.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Combined log path (default: timestamped file in output root).",
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        help=(
            "JSON from audit_cleaning_changes.py. Only recordings for which the "
            "complete audit accepted a discrete correction are materialized."
        ),
    )
    return parser.parse_args()


def discover_jobs(
    input_root: Path,
    output_root: Path,
    powers: list[str] | None = None,
    reps: list[int] | None = None,
    method: str = "integer",
) -> tuple[list[CleanJob], list[DiscoveryIssue]]:
    """Discover valid inputs using the expected two-level directory layout."""
    jobs: list[CleanJob] = []
    issues: list[DiscoveryIssue] = []

    if not input_root.exists():
        issues.append(DiscoveryIssue(directory=input_root, reason="input root does not exist"))
        return jobs, issues

    condition_directories = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and CONDITION_PATTERN.fullmatch(path.name)
    )

    if powers:
        condition_directories = [p for p in condition_directories if p.name in powers]

    for condition_directory in condition_directories:
        realization_directories = sorted(
            path
            for path in condition_directory.iterdir()
            if path.is_dir() and REALIZATION_PATTERN.fullmatch(path.name)
        )
        if reps:
            selected_reps = set(reps)
            realization_directories = [
                path
                for path in realization_directories
                if (
                    (match := REPETITION_NUMBER_PATTERN.search(path.name))
                    and int(match.group(1)) in selected_reps
                )
            ]

        for realization_directory in realization_directories:
            input_files = sorted(
                path
                for path in realization_directory.iterdir()
                if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
            )
            if len(input_files) != 1:
                issues.append(
                    DiscoveryIssue(
                        directory=realization_directory,
                        reason=(
                            "expected exactly one direct .h5/.hdf5 file; "
                            f"found {len(input_files)}"
                        ),
                    )
                )
                continue

            input_path = input_files[0]
            relative_directory = realization_directory.relative_to(input_root)
            cleaner_tag = "_hybrid" if method == "hybrid" else ""
            output_filename = (
                f"{input_path.stem}_cleaned{cleaner_tag}_hp50hz.hdf5"
            )
            output_path = output_root / relative_directory / output_filename

            jobs.append(
                CleanJob(
                    condition=condition_directory.name,
                    realization=realization_directory.name,
                    input_path=input_path,
                    output_path=output_path,
                )
            )

    return jobs, issues


def announce(log: TextIO, message: str) -> None:
    """Write a timestamped message to both stdout and the log file."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def select_jobs_from_audit(
    jobs: list[CleanJob],
    audit_path: Path,
    *,
    q: float,
    tau: float,
    method: str,
    temporal_align: bool,
) -> tuple[list[CleanJob], list[CleanJob]]:
    """Validate an audit manifest and partition jobs into materialize/reuse-raw."""
    audit_path = audit_path.expanduser().resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError(
            f"Unsupported audit schema in {audit_path}: {payload.get('schema_version')!r}"
        )
    if payload.get("highpass_filter_included") is not False:
        raise ValueError("Audit manifest must explicitly exclude the high-pass filter.")

    settings = payload.get("settings", {})
    expected = {
        "spatial_method": method,
        "temporal_alignment_audited": bool(temporal_align),
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(
                f"Audit setting {key}={settings.get(key)!r} does not match batch setting {value!r}."
            )
    for key, value in (("q_um", q), ("tau_um", tau)):
        if not math.isclose(float(settings.get(key, math.nan)), float(value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Audit setting {key}={settings.get(key)!r} does not match batch setting {value!r}."
            )

    records: dict[Path, dict] = {}
    for record in payload.get("recordings", []):
        path = Path(record["input_file"]).expanduser().resolve()
        if path in records:
            raise ValueError(f"Duplicate input in audit manifest: {path}")
        records[path] = record

    selected: list[CleanJob] = []
    reuse_raw: list[CleanJob] = []
    for job in jobs:
        input_path = job.input_path.expanduser().resolve()
        if input_path not in records:
            raise ValueError(f"Selected batch input is missing from audit manifest: {input_path}")
        record = records[input_path]
        if not record.get("complete_recording") or not record.get("deployment_decision_valid"):
            raise ValueError(f"Audit is partial or not deployment-valid for {input_path}")
        stat = input_path.stat()
        if int(record.get("source_size_bytes", -1)) != stat.st_size:
            raise ValueError(f"Input size changed since audit: {input_path}")
        if int(record.get("source_mtime_ns", -1)) != stat.st_mtime_ns:
            raise ValueError(f"Input modification time changed since audit: {input_path}")
        if record.get("requires_materialized_cleaned_dataset"):
            selected.append(job)
        else:
            reuse_raw.append(job)
    return selected, reuse_raw


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.log_file is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = args.output_root / f"batch_clean_{timestamp_str}.log"

    jobs, issues = discover_jobs(
        input_root=args.input_root,
        output_root=args.output_root,
        powers=args.powers,
        reps=args.reps,
        method=args.method,
    )

    if args.max_files is not None and args.max_files > 0:
        jobs = jobs[:args.max_files]

    audit_reuse_raw: list[CleanJob] = []
    if args.audit_summary is not None:
        jobs, audit_reuse_raw = select_jobs_from_audit(
            jobs,
            args.audit_summary,
            q=args.q,
            tau=args.tau,
            method=args.method,
            temporal_align=args.temporal_align,
        )

    with open(args.log_file, "a", encoding="utf-8") as log:
        announce(log, "=== Data Cleaning Batch Runner Initialized ===")
        announce(log, f"Input Root:  {args.input_root}")
        announce(log, f"Output Root: {args.output_root}")
        announce(log, f"Selected Powers: {args.powers or 'ALL'}")
        announce(log, f"Discovered Jobs: {len(jobs)}")
        announce(log, f"Discovery Issues: {len(issues)}")
        if args.audit_summary is not None:
            announce(log, f"Audit Summary: {args.audit_summary}")
            announce(
                log,
                f"Audit selection: materialize {len(jobs)} | reuse raw {len(audit_reuse_raw)}",
            )

        for issue in issues:
            announce(log, f"  Issue: {issue.directory} -> {issue.reason}")

        if args.dry_run:
            announce(log, "\n--- DRY RUN: Planned Jobs ---")
            for i, job in enumerate(jobs, start=1):
                exists_tag = " [EXISTS - SKIPPED]" if job.output_path.exists() and not args.overwrite else ""
                size_mb = job.input_path.stat().st_size / (1024.0**2) if job.input_path.exists() else 0
                announce(log, f"[{i:3d}/{len(jobs):3d}] {job.condition}/{job.realization} ({size_mb:.1f} MB){exists_tag}")
                announce(log, f"        In:  {job.input_path}")
                announce(log, f"        Out: {job.output_path}")
            if audit_reuse_raw:
                announce(
                    log,
                    f"\n--- Audit-bypassed recordings ({len(audit_reuse_raw)}; keep raw) ---",
                )
                for job in audit_reuse_raw:
                    announce(log, f"        {job.condition}/{job.realization}: {job.input_path}")
            announce(log, "\nDry run complete. No files were modified.")
            return

        # Execute jobs
        succeeded = 0
        skipped = 0
        failed = 0
        job_results = []

        total_jobs = len(jobs)
        batch_start = time.perf_counter()

        for idx, job in enumerate(jobs, start=1):
            announce(log, f"\n[{idx}/{total_jobs}] Processing {job.condition}/{job.realization}...")
            announce(log, f"  Input:  {job.input_path}")
            announce(log, f"  Output: {job.output_path}")

            if job.output_path.exists() and not args.overwrite:
                announce(log, "  Output already exists and --overwrite not set. Skipping.")
                skipped += 1
                job_results.append({
                    "condition": job.condition,
                    "realization": job.realization,
                    "status": "SKIPPED",
                    "output_path": str(job.output_path),
                })
                continue

            job_start = time.perf_counter()
            try:
                export_cleaned_dataset(
                    raw_input=job.input_path,
                    output_file=job.output_path,
                    q=args.q,
                    tau=args.tau,
                    method=args.method,
                    temporal_align=args.temporal_align,
                    apply_highpass_50hz=args.apply_highpass_50hz,
                    highpass_cutoff_hz=args.highpass_cutoff_hz,
                    workers=args.workers,
                    chunk_size=args.chunk_size,
                    scratch_dir=args.scratch_dir,
                    stage_input_to_scratch=args.stage_input_to_scratch,
                    max_frames=args.max_frames,
                    overwrite=args.overwrite,
                )
                job_elapsed = time.perf_counter() - job_start
                succeeded += 1
                announce(log, f"  Job Succeeded in {job_elapsed:.1f} s.")
                job_results.append({
                    "condition": job.condition,
                    "realization": job.realization,
                    "status": "SUCCESS",
                    "elapsed_sec": job_elapsed,
                    "output_path": str(job.output_path),
                })
            except Exception as exc:
                job_elapsed = time.perf_counter() - job_start
                failed += 1
                err_msg = f"{type(exc).__name__}: {exc}"
                announce(log, f"  JOB FAILED after {job_elapsed:.1f} s: {err_msg}")
                log.write(traceback.format_exc() + "\n")
                log.flush()
                job_results.append({
                    "condition": job.condition,
                    "realization": job.realization,
                    "status": "FAILED",
                    "error": err_msg,
                    "elapsed_sec": job_elapsed,
                    "input_path": str(job.input_path),
                })

        batch_elapsed = time.perf_counter() - batch_start
        announce(log, "\n=== Batch Run Finished ===")
        announce(log, f"Total Jobs: {total_jobs} | Succeeded: {succeeded} | Skipped: {skipped} | Failed: {failed}")
        announce(log, f"Total Time: {batch_elapsed:.1f} s ({batch_elapsed/60.0:.1f} min)")

        summary_file = args.output_root / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_file.write_text(json.dumps({
            "total_jobs": total_jobs,
            "succeeded": succeeded,
            "skipped": skipped,
            "failed": failed,
            "audit_summary": str(args.audit_summary) if args.audit_summary else None,
            "audit_reuse_raw_count": len(audit_reuse_raw),
            "audit_reuse_raw_inputs": [str(job.input_path) for job in audit_reuse_raw],
            "total_elapsed_sec": batch_elapsed,
            "jobs": job_results,
        }, indent=2) + "\n")
        announce(log, f"Batch summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
