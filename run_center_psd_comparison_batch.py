"""Create raw-versus-POD center PSD comparisons for every experiment.

Expected reduced-data layout::

    REDUCED_ROOT/0p20/Ca_ac_0p001660_rep13/pod_2d_r1000.h5

The matching raw file is taken from the POD file's ``source_file`` attribute.
If that path is unavailable, the script falls back to the same relative folder
under ``--raw-root`` and requires exactly one direct HDF5 child there. All PNGs
are written into one output directory with unique power/realization filenames.
Failures are logged and skipped without stopping the batch.

Example
-------
    python run_center_psd_comparison_batch.py --dry-run
    python run_center_psd_comparison_batch.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
from typing import TextIO

import h5py


DEFAULT_REDUCED_ROOT = Path("/disk/hyk049/jschupp/diffusion_capillary_wave")
DEFAULT_RAW_ROOT = Path("/disk/hyk049/DHM_new_experiment")
CONDITION_PATTERN = re.compile(r"0p\d{1,2}")
REALIZATION_PATTERN = re.compile(r"Ca_ac.*")
HDF5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class ComparisonJob:
    condition: str
    realization: str
    raw_path: Path
    pod_path: Path
    output_path: Path


@dataclass(frozen=True)
class DiscoveryIssue:
    directory: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced-root",
        type=Path,
        default=DEFAULT_REDUCED_ROOT,
        help=f"Reduced-data tree root (default: {DEFAULT_REDUCED_ROOT}).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Fallback raw-data tree root (default: {DEFAULT_RAW_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "center_psd_comparisons",
        help="Flat PNG output folder (default: ./center_psd_comparisons).",
    )
    parser.add_argument(
        "--pod-rank",
        type=int,
        default=1_000,
        help="Expected POD result rank/filename (default: 1000).",
    )
    parser.add_argument(
        "--reconstruction-rank",
        type=int,
        help="Modes used in each reconstruction (default: all stored modes).",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=8_192,
        help="Samples per Welch segment (default: 8192).",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=0.5,
        help="Fractional Welch overlap (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8_192,
        help="POD coefficient rows reconstructed per batch (default: 8192).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Plot resolution (default: 200 dpi).",
    )
    parser.add_argument(
        "--comparison-script",
        type=Path,
        default=Path(__file__).with_name("compare_raw_pod_center_psd.py"),
        help="Single-case comparison script (default: next to this script).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log path (default: timestamped file in the output folder).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Process only the first N jobs; useful for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing plots; otherwise completed jobs are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and print matched jobs without computing PSDs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pod_rank < 1:
        raise ValueError("--pod-rank must be positive.")
    if args.reconstruction_rank is not None and args.reconstruction_rank < 1:
        raise ValueError("--reconstruction-rank must be positive.")
    if args.nperseg < 2:
        raise ValueError("--nperseg must be at least 2.")
    if not 0 <= args.overlap_fraction < 1:
        raise ValueError("--overlap-fraction must lie in [0, 1).")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive.")


def _attribute_path(value: object) -> Path | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser()


def _raw_file_from_reduced(
    pod_path: Path,
    raw_root: Path,
    condition: str,
    realization: str,
) -> tuple[Path | None, str | None]:
    try:
        with h5py.File(pod_path, "r") as handle:
            source_path = _attribute_path(handle.attrs.get("source_file"))
    except OSError as error:
        return None, f"could not open POD file: {error}"

    if source_path is not None:
        if source_path.is_file():
            return source_path.resolve(), None
        try:
            source_relative = source_path.relative_to(DEFAULT_RAW_ROOT)
        except ValueError:
            source_relative = None
        if source_relative is not None:
            remapped = raw_root / source_relative
            if remapped.is_file():
                return remapped.resolve(), None

    raw_directory = raw_root / condition / realization
    if not raw_directory.is_dir():
        source_note = f"; source_file={source_path}" if source_path else ""
        return None, f"matching raw folder does not exist: {raw_directory}{source_note}"
    candidates = sorted(
        path
        for path in raw_directory.iterdir()
        if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
    )
    if len(candidates) != 1:
        return None, (
            "expected exactly one raw .h5/.hdf5 file in "
            f"{raw_directory}; found {len(candidates)}"
        )
    return candidates[0].resolve(), None


def discover_jobs(
    reduced_root: Path,
    raw_root: Path,
    output_directory: Path,
    pod_rank: int,
) -> tuple[list[ComparisonJob], list[DiscoveryIssue]]:
    jobs: list[ComparisonJob] = []
    issues: list[DiscoveryIssue] = []
    condition_directories = sorted(
        path
        for path in reduced_root.iterdir()
        if path.is_dir() and CONDITION_PATTERN.fullmatch(path.name)
    )
    for condition_directory in condition_directories:
        realization_directories = sorted(
            path
            for path in condition_directory.iterdir()
            if path.is_dir() and REALIZATION_PATTERN.fullmatch(path.name)
        )
        for realization_directory in realization_directories:
            pod_path = realization_directory / f"pod_2d_r{pod_rank}.h5"
            if not pod_path.is_file():
                issues.append(
                    DiscoveryIssue(
                        realization_directory,
                        f"missing rank-{pod_rank} POD file: {pod_path.name}",
                    )
                )
                continue
            raw_path, raw_error = _raw_file_from_reduced(
                pod_path,
                raw_root,
                condition_directory.name,
                realization_directory.name,
            )
            if raw_path is None:
                issues.append(
                    DiscoveryIssue(
                        realization_directory,
                        raw_error or "could not identify raw source file",
                    )
                )
                continue
            filename = (
                f"{condition_directory.name}__{realization_directory.name}"
                "__raw_vs_pod_center_psd.png"
            )
            jobs.append(
                ComparisonJob(
                    condition=condition_directory.name,
                    realization=realization_directory.name,
                    raw_path=raw_path,
                    pod_path=pod_path.resolve(),
                    output_path=(output_directory / filename).resolve(),
                )
            )
    return jobs, issues


def announce(log: TextIO, message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def build_command(args: argparse.Namespace, job: ComparisonJob) -> list[str]:
    command = [
        sys.executable,
        str(args.comparison_script),
        str(job.raw_path),
        str(job.pod_path),
        "--output",
        str(job.output_path),
        "--label",
        f"{job.condition} — {job.realization}",
        "--nperseg",
        str(args.nperseg),
        "--overlap-fraction",
        str(args.overlap_fraction),
        "--batch-size",
        str(args.batch_size),
        "--dpi",
        str(args.dpi),
    ]
    if args.reconstruction_rank is not None:
        command.extend(("--reconstruction-rank", str(args.reconstruction_rank)))
    if args.overwrite:
        command.append("--overwrite")
    return command


def run_and_tee(command: list[str], log: TextIO) -> int:
    announce(log, f"Command: {shlex.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise


def run_batch(args: argparse.Namespace, log: TextIO) -> int:
    reduced_root = args.reduced_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    comparison_script = args.comparison_script.expanduser().resolve()
    args.comparison_script = comparison_script
    if not reduced_root.is_dir():
        raise FileNotFoundError(f"Reduced-data root does not exist: {reduced_root}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw-data root does not exist: {raw_root}")
    if not comparison_script.is_file():
        raise FileNotFoundError(
            f"Comparison script does not exist: {comparison_script}"
        )

    announce(log, f"Reduced-data root: {reduced_root}")
    announce(log, f"Raw-data root: {raw_root}")
    announce(log, f"Output directory: {output_directory}")
    announce(
        log,
        f"Settings: pod_rank={args.pod_rank}, "
        f"reconstruction_rank={args.reconstruction_rank or 'all'}, "
        f"nperseg={args.nperseg}, overlap={args.overlap_fraction:g}",
    )
    jobs, issues = discover_jobs(
        reduced_root, raw_root, output_directory, args.pod_rank
    )
    for issue in issues:
        announce(log, f"SKIP invalid {issue.directory}: {issue.reason}")
    if args.max_files is not None:
        jobs = jobs[: args.max_files]
    announce(
        log,
        f"Discovered {len(jobs)} valid matched job(s); "
        f"{len(issues)} invalid folder(s).",
    )
    if not jobs:
        announce(log, "No valid matched jobs found.")
        return 1

    succeeded = 0
    skipped_existing = 0
    failed = 0
    batch_started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        label = f"[{index}/{len(jobs)}] {job.condition}/{job.realization}"
        if job.output_path.exists() and not args.overwrite:
            announce(log, f"SKIP existing {label}: {job.output_path}")
            skipped_existing += 1
            continue
        announce(log, f"START {label}")
        announce(log, f"Raw input: {job.raw_path}")
        announce(log, f"POD input: {job.pod_path}")
        announce(log, f"Output: {job.output_path}")
        if args.dry_run:
            announce(log, f"DRY RUN complete for {label}")
            continue

        job_started = time.perf_counter()
        try:
            return_code = run_and_tee(build_command(args, job), log)
        except KeyboardInterrupt:
            announce(log, f"INTERRUPTED during {label}")
            return 130
        except BaseException:
            failed += 1
            announce(log, f"FAILED to launch {label}; continuing.")
            traceback.print_exc(file=log)
            log.flush()
            continue
        elapsed = time.perf_counter() - job_started
        if return_code == 0 and job.output_path.is_file():
            succeeded += 1
            announce(log, f"DONE {label} in {elapsed:.1f} s")
        else:
            failed += 1
            announce(
                log,
                f"FAILED {label} after {elapsed:.1f} s "
                f"(exit code {return_code}); continuing.",
            )

    elapsed = time.perf_counter() - batch_started
    if args.dry_run:
        announce(log, f"Dry run listed {len(jobs)} job(s) in {elapsed:.1f} s.")
        return 0
    announce(
        log,
        "Batch summary: "
        f"{succeeded} succeeded, {skipped_existing} existing skipped, "
        f"{len(issues)} invalid skipped, {failed} failed; "
        f"elapsed {elapsed:.1f} s.",
    )
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else output_directory / f"center_psd_batch_{timestamp}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        announce(log, f"Batch log: {log_path}")
        try:
            return run_batch(args, log)
        except BaseException:
            announce(log, "Batch setup failed.")
            traceback.print_exc(file=log)
            log.flush()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
