"""Run ``pod_2d.py`` sequentially over the experiment directory tree.

Expected input layout::

    INPUT_ROOT/
      0p04/
        Ca_ac_0p001250_rep9/
          11222025_i_0.15vpp_data_roi-none_cal-true.hdf5

The same relative directory structure is created below ``--output-root``.
Each valid ``Ca_ac*`` directory must contain exactly one direct child with an
``.h5`` or ``.hdf5`` extension. Invalid directories and failed POD runs are
logged and skipped without stopping the remaining jobs.

Example
-------
    python run_pod_batch.py --rank 100 --training-frames 5000 --dry-run

    OPENBLAS_NUM_THREADS=24 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
        python run_pod_batch.py --rank 100 --training-frames 5000
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


DEFAULT_INPUT_ROOT = Path("/disk/hyk049/DHM_new_experiment")
CONDITION_PATTERN = re.compile(r"0p\d{1,2}")
REALIZATION_PATTERN = re.compile(r"Ca_ac.*")
HDF5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class PodJob:
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
        default=Path.cwd(),
        help="Root of the mirrored output tree (default: current directory).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=100,
        help="Final POD rank passed to pod_2d.py (default: 100).",
    )
    parser.add_argument(
        "--training-frames",
        type=int,
        default=5_000,
        help="Training frames passed to pod_2d.py (default: 5000).",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=20,
        help="Candidate modes beyond the final rank (default: 20).",
    )
    parser.add_argument(
        "--power-iterations",
        type=int,
        default=1,
        help="Randomized-SVD power iterations (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Frame batch size passed to pod_2d.py (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12_345,
        help="Random seed passed to every POD run (default: 12345).",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Stored mode/coefficient type (default: float32).",
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="Output HDF5 compression (default: lzf).",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        choices=range(1, 10),
        default=4,
        metavar="1-9",
        help="gzip level when --compression=gzip (default: 4).",
    )
    parser.add_argument(
        "--keep-spatial-mean",
        action="store_true",
        help="Pass --keep-spatial-mean to pod_2d.py.",
    )
    parser.add_argument(
        "--no-temporal-centering",
        action="store_true",
        help="Pass --no-temporal-centering to pod_2d.py.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        help=(
            "Set OpenBLAS/MKL/OMP thread counts for every POD subprocess. "
            "By default, inherit the current environment."
        ),
    )
    parser.add_argument(
        "--pod-script",
        type=Path,
        default=Path(__file__).with_name("pod_2d.py"),
        help="Path to pod_2d.py (default: next to this script).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Combined log path (default: timestamped file in output root).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Run only the first N valid inputs; useful for a test batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing POD outputs; otherwise they are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and print jobs without running pod_2d.py.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("rank", "training_frames", "batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.oversampling < 0:
        raise ValueError("--oversampling must be non-negative.")
    if args.power_iterations < 0:
        raise ValueError("--power-iterations must be non-negative.")
    if args.blas_threads is not None and args.blas_threads < 1:
        raise ValueError("--blas-threads must be positive.")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive.")


def discover_jobs(
    input_root: Path,
    output_root: Path,
    rank: int,
) -> tuple[list[PodJob], list[DiscoveryIssue]]:
    """Discover valid inputs using the expected two-level directory layout."""
    jobs: list[PodJob] = []
    issues: list[DiscoveryIssue] = []

    condition_directories = sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and CONDITION_PATTERN.fullmatch(path.name)
    )
    for condition_directory in condition_directories:
        realization_directories = sorted(
            path
            for path in condition_directory.iterdir()
            if path.is_dir() and REALIZATION_PATTERN.fullmatch(path.name)
        )
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

            relative_directory = realization_directory.relative_to(input_root)
            output_path = (
                output_root
                / relative_directory
                / f"pod_2d_r{rank}.h5"
            )
            jobs.append(
                PodJob(
                    condition=condition_directory.name,
                    realization=realization_directory.name,
                    input_path=input_files[0],
                    output_path=output_path,
                )
            )

    return jobs, issues


def announce(log: TextIO, message: str) -> None:
    """Write a timestamped batch message to both the terminal and log."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def build_command(args: argparse.Namespace, job: PodJob) -> list[str]:
    command = [
        sys.executable,
        str(args.pod_script),
        "--input",
        str(job.input_path),
        "--output",
        str(job.output_path),
        "--rank",
        str(args.rank),
        "--training-frames",
        str(args.training_frames),
        "--oversampling",
        str(args.oversampling),
        "--power-iterations",
        str(args.power_iterations),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--storage-dtype",
        args.storage_dtype,
        "--compression",
        args.compression,
        "--gzip-level",
        str(args.gzip_level),
    ]
    if args.keep_spatial_mean:
        command.append("--keep-spatial-mean")
    if args.no_temporal_centering:
        command.append("--no-temporal-centering")
    if args.overwrite:
        command.append("--overwrite")
    return command


def subprocess_environment(blas_threads: int | None) -> dict[str, str]:
    environment = os.environ.copy()
    if blas_threads is not None:
        thread_count = str(blas_threads)
        for variable in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            environment[variable] = thread_count
    return environment


def run_and_tee(
    command: list[str],
    environment: dict[str, str],
    log: TextIO,
) -> int:
    """Run one command and copy its combined output to terminal and log."""
    command_line = shlex.join(command)
    announce(log, f"Command: {command_line}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
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
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    pod_script = args.pod_script.expanduser().resolve()
    args.pod_script = pod_script

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not pod_script.is_file():
        raise FileNotFoundError(f"POD script does not exist: {pod_script}")

    announce(log, f"Input root: {input_root}")
    announce(log, f"Output root: {output_root}")
    announce(
        log,
        f"POD settings: rank={args.rank}, "
        f"training_frames={args.training_frames}, "
        f"oversampling={args.oversampling}, "
        f"power_iterations={args.power_iterations}, "
        f"batch_size={args.batch_size}",
    )
    if args.blas_threads is not None:
        announce(log, f"BLAS threads per POD process: {args.blas_threads}")

    jobs, issues = discover_jobs(input_root, output_root, args.rank)
    for issue in issues:
        announce(log, f"SKIP invalid folder {issue.directory}: {issue.reason}")

    if args.max_files is not None:
        jobs = jobs[: args.max_files]
    announce(
        log,
        f"Discovered {len(jobs)} valid job(s); "
        f"{len(issues)} invalid folder(s).",
    )

    if not jobs:
        announce(log, "No valid inputs found.")
        return 1

    environment = subprocess_environment(args.blas_threads)
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
        announce(log, f"Input: {job.input_path}")
        announce(log, f"Output: {job.output_path}")
        if args.dry_run:
            announce(log, f"DRY RUN complete for {label}")
            continue

        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        job_started = time.perf_counter()
        try:
            return_code = run_and_tee(
                build_command(args, job), environment, log
            )
        except KeyboardInterrupt:
            announce(log, f"INTERRUPTED during {label}")
            return 130
        except BaseException:
            failed += 1
            announce(log, f"FAILED to launch {label}")
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
    _validate_args(args)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else output_root / f"pod_batch_{timestamp}.log"
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
