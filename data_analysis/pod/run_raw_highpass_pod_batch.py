"""Run raw-versus-high-pass POD comparisons over the lab experiment tree.

The default pilot is intentionally bounded: a 50 Hz cutoff, every fourth
spatial point (50 x 50 for a 200 x 200 frame), 5,000 stratified snapshots,
and rank 100. Jobs run sequentially because the raw inputs contain 100,001
individually keyed frames and concurrent reads are usually counterproductive.

Expected input layout matches :mod:`data_analysis.pod.run_pod_batch`::

    /disk/hyk049/DHM_new_experiment/
      0p04/
        Ca_ac_..._rep1/
          recording.hdf5

Use ``--dry-run`` before starting the batch. Completed jobs are validated and
skipped on restart. Temporary raw and filtered memmaps should be placed on fast
local scratch with ``--scratch-root``; they are removed after each job.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
from typing import TextIO

from data_analysis.pod.compare_raw_highpass_pod import TREATMENT_LABELS
from data_analysis.pod.run_pod_batch import DEFAULT_INPUT_ROOT, discover_jobs


EXPECTED_OUTPUTS = ("summary.csv", "mode_diagnostics.csv", "comparison.png")
AGGREGATE_METRICS = (
    "total_energy_relative_to_raw",
    "constant_subspace_energy_fraction",
    "constant_energy_relative_to_raw_constant",
    "rank_energy_fraction",
    "spatial_mean_energy_retained_fraction",
    "centered_spatial_mean_energy_retained_fraction",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Lab raw-data tree (default: {DEFAULT_INPUT_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/pod_highpass_comparison_batch"),
        help="Parent directory for the configuration-labelled batch directory.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/tmp"),
        help="Fast local directory for temporary memmaps (default: /tmp).",
    )
    parser.add_argument("--cutoff-hz", type=float, default=50.0)
    parser.add_argument("--spatial-stride", type=int, default=4)
    parser.add_argument("--pod-frames", type=int, default=5_000)
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--oversampling", type=int, default=20)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--filter-columns", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--treatments",
        nargs="+",
        choices=tuple(TREATMENT_LABELS),
        default=["raw", "full_highpass"],
        help="Treatments to compare (default: raw full_highpass).",
    )
    parser.add_argument(
        "--powers",
        nargs="+",
        help="Only these condition directories, for example 0p04 0p20 0p35.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        nargs="+",
        help="Only these repetition numbers within every selected power.",
    )
    parser.add_argument("--max-files", type=int, help="Run only the first N selected files.")
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=4,
        help="OpenBLAS/OMP/MKL threads used by each sequential job (default: 4).",
    )
    parser.add_argument(
        "--comparison-script",
        type=Path,
        default=Path(__file__).with_name("compare_raw_highpass_pod.py"),
    )
    parser.add_argument(
        "--label",
        help="Batch-directory name; default encodes cutoff, stride, frames, rank, and treatments.",
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute completed jobs. A changed configuration still requires a new label.",
    )
    return parser.parse_args(argv)


def _cutoff_tag(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("+", "")


def default_label(args: argparse.Namespace) -> str:
    treatment_tag = "-".join(args.treatments)
    return (
        f"hp{_cutoff_tag(args.cutoff_hz)}_stride{args.spatial_stride}_"
        f"n{args.pod_frames}_r{args.rank}_{treatment_tag}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "spatial_stride",
        "pod_frames",
        "rank",
        "filter_columns",
        "dpi",
        "blas_threads",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.oversampling < 0 or args.power_iterations < 0 or args.seed < 0:
        raise ValueError("Oversampling, power iterations, and seed must be nonnegative.")
    if not args.cutoff_hz > 0:
        raise ValueError("--cutoff-hz must be positive.")
    if "raw" not in args.treatments:
        raise ValueError("--treatments must include raw as the comparison reference.")
    if len(set(args.treatments)) != len(args.treatments):
        raise ValueError("--treatments must not contain duplicates.")
    if args.reps is not None and any(rep < 1 for rep in args.reps):
        raise ValueError("--reps values must be positive.")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive.")


def announce(log: TextIO, message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def _rep_number(realization: str) -> int | None:
    match = re.search(r"_rep(\d+)$", realization)
    return int(match.group(1)) if match else None


def select_jobs(jobs, powers: list[str] | None, reps: list[int] | None):
    selected = jobs
    if powers is not None:
        requested = set(powers)
        available = {job.condition for job in jobs}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested power directories were not found: {missing}")
        selected = [job for job in selected if job.condition in requested]
    if reps is not None:
        requested_reps = set(reps)
        selected = [
            job for job in selected if _rep_number(job.realization) in requested_reps
        ]
    return selected


def configuration(args: argparse.Namespace, input_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "input_root": str(input_root),
        "cutoff_hz": float(args.cutoff_hz),
        "spatial_stride": int(args.spatial_stride),
        "pod_frames": int(args.pod_frames),
        "rank": int(args.rank),
        "oversampling": int(args.oversampling),
        "power_iterations": int(args.power_iterations),
        "filter_columns": int(args.filter_columns),
        "seed": int(args.seed),
        "dpi": int(args.dpi),
        "treatments": list(args.treatments),
        "powers": list(args.powers) if args.powers is not None else None,
        "reps": list(args.reps) if args.reps is not None else None,
        "comparison_script": str(args.comparison_script.expanduser().resolve()),
    }


def _encoded_configuration(config: dict[str, object]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _configuration_digest(config: dict[str, object]) -> str:
    return hashlib.sha256(_encoded_configuration(config).encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def job_output_directory(batch_directory: Path, job) -> Path:
    return batch_directory / job.condition / job.realization


def completion_path(output_directory: Path) -> Path:
    return output_directory / "completion.json"


def is_complete(output_directory: Path, job, config_digest: str) -> bool:
    marker = completion_path(output_directory)
    if not marker.is_file() or any(
        not (output_directory / name).is_file() for name in EXPECTED_OUTPUTS
    ):
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("configuration_sha256") == config_digest
        and payload.get("source_fingerprint") == source_fingerprint(job.input_path)
    )


def build_command(
    args: argparse.Namespace,
    job,
    output_directory: Path,
    scratch_root: Path,
    *,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(args.comparison_script.expanduser().resolve()),
        "--input",
        str(job.input_path),
        "--cutoff-hz",
        str(args.cutoff_hz),
        "--spatial-stride",
        str(args.spatial_stride),
        "--pod-frames",
        str(args.pod_frames),
        "--rank",
        str(args.rank),
        "--oversampling",
        str(args.oversampling),
        "--power-iterations",
        str(args.power_iterations),
        "--filter-columns",
        str(args.filter_columns),
        "--seed",
        str(args.seed),
        "--dpi",
        str(args.dpi),
        "--treatments",
        *args.treatments,
        "--output-dir",
        str(output_directory),
        "--scratch-dir",
        str(scratch_root),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def subprocess_environment(blas_threads: int, repository_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    count = str(blas_threads)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = count
    environment.setdefault("MPLCONFIGDIR", "/tmp/capillarywave-matplotlib")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repository_root)
        if not existing_pythonpath
        else str(repository_root) + os.pathsep + existing_pythonpath
    )
    return environment


def run_and_tee(command: list[str], environment: dict[str, str], log: TextIO) -> int:
    announce(log, f"Command: {shlex.join(command)}")
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
        with process.stdout:
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


def write_manifest(batch_directory: Path, jobs, input_root: Path) -> None:
    rows = [
        {
            "condition": job.condition,
            "realization": job.realization,
            "repetition": _rep_number(job.realization),
            "source_relative": str(job.input_path.relative_to(input_root)),
            "source_file": str(job.input_path),
            "output_directory": str(job_output_directory(batch_directory, job)),
        }
        for job in jobs
    ]
    if rows:
        _atomic_csv(batch_directory / "manifest.csv", list(rows[0]), rows)


def summarize(batch_directory: Path, jobs, input_root: Path, config_digest: str) -> int:
    rows: list[dict[str, object]] = []
    for job in jobs:
        output_directory = job_output_directory(batch_directory, job)
        if not is_complete(output_directory, job, config_digest):
            continue
        with (output_directory / "summary.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "condition": job.condition,
                        "realization": job.realization,
                        "repetition": _rep_number(job.realization),
                        "source_relative": str(job.input_path.relative_to(input_root)),
                        **row,
                    }
                )
    if not rows:
        return 0
    _atomic_csv(batch_directory / "all_recordings_summary.csv", list(rows[0]), rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), str(row["treatment"])), []).append(row)
    aggregate_rows = []
    for (condition, treatment), selected in sorted(grouped.items()):
        aggregate: dict[str, object] = {
            "condition": condition,
            "treatment": treatment,
            "n_recordings": len(selected),
        }
        for metric in AGGREGATE_METRICS:
            values = [float(row[metric]) for row in selected]
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else ""
            )
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_sd"] = variance**0.5 if variance != "" else ""
        aggregate_rows.append(aggregate)
    _atomic_csv(
        batch_directory / "condition_summary.csv",
        list(aggregate_rows[0]),
        aggregate_rows,
    )
    return len({(row["condition"], row["realization"]) for row in rows})


def run_batch(args: argparse.Namespace, log: TextIO) -> int:
    _validate_args(args)
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scratch_root = args.scratch_root.expanduser().resolve()
    comparison_script = args.comparison_script.expanduser().resolve()
    args.comparison_script = comparison_script
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not scratch_root.is_dir():
        raise FileNotFoundError(f"Scratch root does not exist: {scratch_root}")
    if not comparison_script.is_file():
        raise FileNotFoundError(f"Comparison script does not exist: {comparison_script}")

    label = args.label or default_label(args)
    batch_directory = output_root / label
    batch_directory.mkdir(parents=True, exist_ok=True)
    config = configuration(args, input_root)
    config_digest = _configuration_digest(config)
    config_path = batch_directory / "configuration.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if _encoded_configuration(existing) != _encoded_configuration(config):
            raise ValueError(
                f"Existing batch configuration differs: {config_path}. "
                "Choose a new --label or matching settings."
            )
    else:
        _atomic_json(config_path, config)

    discovered, issues = discover_jobs(input_root, batch_directory, args.rank)
    jobs = select_jobs(discovered, args.powers, args.reps)
    if args.max_files is not None:
        jobs = jobs[: args.max_files]
    if not jobs:
        raise ValueError("No jobs matched the requested powers/repetitions.")

    write_manifest(batch_directory, jobs, input_root)
    issue_rows = [
        {"directory": str(issue.directory), "reason": issue.reason} for issue in issues
    ]
    if issue_rows:
        _atomic_csv(batch_directory / "discovery_issues.csv", list(issue_rows[0]), issue_rows)

    announce(log, f"Input root: {input_root}")
    announce(log, f"Batch output: {batch_directory}")
    announce(log, f"Scratch root: {scratch_root}")
    announce(
        log,
        f"Settings: cutoff={args.cutoff_hz:g} Hz, stride={args.spatial_stride}, "
        f"POD frames={args.pod_frames}, rank={args.rank}, "
        f"treatments={','.join(args.treatments)}",
    )
    for issue in issues:
        announce(log, f"DISCOVERY ISSUE: {issue.directory}: {issue.reason}")
    announce(log, f"Selected {len(jobs)} job(s); {len(issues)} discovery issue(s).")

    environment = subprocess_environment(
        args.blas_threads, comparison_script.parents[2]
    )
    succeeded = skipped = failed = 0
    started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        output_directory = job_output_directory(batch_directory, job)
        label_text = f"[{index}/{len(jobs)}] {job.condition}/{job.realization}"
        complete = is_complete(output_directory, job, config_digest)
        if complete and not args.overwrite:
            skipped += 1
            announce(log, f"SKIP complete {label_text}")
            continue
        output_directory.mkdir(parents=True, exist_ok=True)
        generated_exists = any(
            (output_directory / name).exists() for name in EXPECTED_OUTPUTS
        )
        command = build_command(
            args,
            job,
            output_directory,
            scratch_root,
            overwrite=args.overwrite or generated_exists,
        )
        if args.dry_run:
            announce(log, f"DRY RUN {label_text}: {shlex.join(command)}")
            continue

        announce(log, f"START {label_text}")
        job_started = time.perf_counter()
        try:
            return_code = run_and_tee(command, environment, log)
            if return_code != 0 or any(
                not (output_directory / name).is_file() for name in EXPECTED_OUTPUTS
            ):
                raise RuntimeError(f"comparison exited with code {return_code}")
            marker = {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "configuration_sha256": config_digest,
                "source_fingerprint": source_fingerprint(job.input_path),
                "elapsed_seconds": time.perf_counter() - job_started,
            }
            _atomic_json(completion_path(output_directory), marker)
            succeeded += 1
            announce(log, f"DONE {label_text} in {marker['elapsed_seconds']:.1f} s")
        except KeyboardInterrupt:
            announce(log, f"INTERRUPTED {label_text}")
            return 130
        except BaseException as exc:
            failed += 1
            announce(log, f"FAILED {label_text}: {exc}")
            traceback.print_exc(file=log)
            log.flush()
        finally:
            summarize(batch_directory, jobs, input_root, config_digest)

    completed = summarize(batch_directory, jobs, input_root, config_digest)
    elapsed = time.perf_counter() - started
    if args.dry_run:
        announce(log, f"Dry run listed {len(jobs)} job(s); no comparisons executed.")
        return 0
    announce(
        log,
        f"Batch summary: {succeeded} succeeded, {skipped} complete skipped, "
        f"{failed} failed, {completed}/{len(jobs)} complete; elapsed {elapsed:.1f} s.",
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else output_root / f"raw_highpass_pod_batch_{timestamp}.log"
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
