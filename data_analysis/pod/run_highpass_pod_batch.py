"""Run production POD on raw recordings after full-field temporal high-pass.

This is the production counterpart to the diagnostic raw/high-pass comparison.
For every recording it:

1. materializes the complete trajectory in local scratch;
2. applies a zero-phase Butterworth high-pass to every spatial pixel over the
   complete recording;
3. writes a temporary filtered HDF5 input in scratch;
4. invokes the standard :mod:`data_analysis.pod.pod_2d` implementation with
   the instantaneous spatial mean retained and temporal mean-field centering;
5. records the raw source and filter provenance in the POD output; and
6. removes all temporary full-field data.

Only the compact POD file is retained permanently.  Jobs are sequential and
existing outputs are skipped unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import TextIO

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.pod.run_pod_batch import (
    DEFAULT_INPUT_ROOT,
    PodJob,
    announce,
    build_command,
    discover_jobs,
    run_and_tee,
    subprocess_environment,
)


DEFAULT_SCRATCH_ROOT = Path("/dev/shm") / (
    f"{os.environ.get('USER', 'user')}_highpass_pod"
)
FILTER_TYPE = "butterworth_zero_phase_sosfiltfilt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--powers", nargs="+", required=True)
    parser.add_argument("--reps", type=int, nargs="+")
    parser.add_argument(
        "--file-pattern",
        default="*_data_roi-none_cal-true.hdf5",
        help="Raw input filename glob/substring.",
    )
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--cutoff-hz", type=float, default=50.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument(
        "--filter-row-chunk",
        type=int,
        default=4,
        help="Rows filtered together in float64 (default: 4).",
    )
    parser.add_argument("--rank", type=int, default=200)
    parser.add_argument("--training-frames", type=int, default=5_000)
    parser.add_argument("--oversampling", type=int, default=20)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument(
        "--storage-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--compression", choices=("lzf", "gzip", "none"), default="lzf"
    )
    parser.add_argument("--gzip-level", type=int, choices=range(1, 10), default=4)
    parser.add_argument("--blas-threads", type=int, default=24)
    parser.add_argument(
        "--pod-script", type=Path, default=Path(__file__).with_name("pod_2d.py")
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "filter_order",
        "filter_row_chunk",
        "rank",
        "training_frames",
        "batch_size",
        "blas_threads",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not np.isfinite(args.cutoff_hz) or args.cutoff_hz <= 0:
        raise ValueError("--cutoff-hz must be finite and positive.")
    if args.oversampling < 0 or args.power_iterations < 0:
        raise ValueError("--oversampling and --power-iterations must be non-negative.")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive.")

    # These attributes are consumed by run_pod_batch.build_command.  They are
    # fixed deliberately so raw and cleaned conditions use the same convention.
    args.keep_spatial_mean = True
    args.no_temporal_centering = False


def _copy_root_attributes(source: h5py.File, destination: h5py.File) -> None:
    for name, value in source.attrs.items():
        destination.attrs[name] = value


def materialize_highpass_input(
    input_path: Path,
    filtered_path: Path,
    work_directory: Path,
    cutoff_hz: float,
    filter_order: int,
    row_chunk: int,
) -> dict[str, float | int | str]:
    """Create one temporary, full-field high-pass HDF5 recording."""
    info = inspect_input(input_path, 0, None)
    if info.n_frames < 3:
        raise ValueError("At least three frames are required for temporal filtering.")

    time_steps = np.diff(info.time)
    if not np.isfinite(time_steps).all() or np.any(time_steps <= 0):
        raise ValueError("Time coordinates must be finite and strictly increasing.")
    mean_dt = float(np.mean(time_steps))
    relative_dt_spread = float(np.max(np.abs(time_steps - mean_dt)) / mean_dt)
    if relative_dt_spread > 5e-2:
        raise ValueError(
            "The high-pass filter assumes uniform sampling, but the maximum "
            f"relative timestep deviation is {relative_dt_spread:.3g}."
        )
    if relative_dt_spread > 1e-3:
        print(
            "  Note: stored timestep deviations reach "
            f"{100 * relative_dt_spread:.3f}%; using the mean sampling rate "
            "to match the established cleaning pipeline.",
            flush=True,
        )
    sampling_hz = 1.0 / mean_dt
    if cutoff_hz >= 0.5 * sampling_hz:
        raise ValueError(
            f"Cutoff {cutoff_hz:g} Hz must be below Nyquist "
            f"({0.5 * sampling_hz:g} Hz)."
        )

    n_frames = info.n_frames
    ny, nx = info.frame_shape
    trajectory_path = work_directory / "full_field.float32.dat"
    trajectory = np.memmap(
        trajectory_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_frames, ny, nx),
    )
    report_every = max(1, n_frames // 5)

    print(
        f"Materializing {n_frames:,} frames ({ny}x{nx}) from {input_path}",
        flush=True,
    )
    started = time.perf_counter()
    with h5py.File(input_path, "r") as source:
        frames = source["main"]
        for position, frame_number in enumerate(info.frame_numbers):
            trajectory[position] = np.asarray(
                frames[_frame_key(frame_number)], dtype=np.float32
            )
            if (position + 1) % report_every == 0 or position + 1 == n_frames:
                print(
                    f"  Read {position + 1:,}/{n_frames:,} frames "
                    f"in {time.perf_counter() - started:.1f} s",
                    flush=True,
                )
    trajectory.flush()

    sos = butter(
        filter_order,
        cutoff_hz,
        btype="highpass",
        fs=sampling_hz,
        output="sos",
    )
    n_row_chunks = (ny + row_chunk - 1) // row_chunk
    filter_started = time.perf_counter()
    print(
        f"Filtering every pixel at {cutoff_hz:g} Hz "
        f"(order {filter_order}, zero phase)...",
        flush=True,
    )
    for chunk_index, y_start in enumerate(range(0, ny, row_chunk), start=1):
        y_stop = min(ny, y_start + row_chunk)
        row_data = np.asarray(trajectory[:, y_start:y_stop, :], dtype=np.float64)
        row_filtered = sosfiltfilt(sos, row_data, axis=0).astype(np.float32)
        trajectory[:, y_start:y_stop, :] = row_filtered
        del row_data, row_filtered
        if chunk_index % 5 == 0:
            gc.collect()
        if (
            chunk_index % max(1, n_row_chunks // 5) == 0
            or chunk_index == n_row_chunks
        ):
            print(
                f"  Filtered rows {y_start}:{y_stop} "
                f"({chunk_index}/{n_row_chunks}) in "
                f"{time.perf_counter() - filter_started:.1f} s",
                flush=True,
            )
    trajectory.flush()

    temporary_filtered_path = filtered_path.with_suffix(filtered_path.suffix + ".tmp")
    temporary_filtered_path.unlink(missing_ok=True)
    write_started = time.perf_counter()
    print(f"Writing temporary filtered input {filtered_path}", flush=True)
    try:
        with h5py.File(input_path, "r") as source, h5py.File(
            temporary_filtered_path, "w"
        ) as destination:
            source.copy("meta", destination)
            _copy_root_attributes(source, destination)
            destination.attrs["source_file"] = str(input_path.resolve())
            destination.attrs["highpass_filter_applied"] = True
            destination.attrs["highpass_cutoff_hz"] = float(cutoff_hz)
            destination.attrs["highpass_filter_order"] = int(filter_order)
            destination.attrs["highpass_filter_type"] = FILTER_TYPE
            destination.attrs["highpass_sampling_hz"] = float(sampling_hz)
            destination.attrs["highpass_applied_to"] = "every spatial pixel"

            output_frames = destination.create_group("main", track_order=True)
            for position, frame_number in enumerate(info.frame_numbers):
                output_frames.create_dataset(
                    _frame_key(frame_number),
                    data=trajectory[position],
                    dtype=np.float32,
                    track_times=False,
                )
                if (position + 1) % report_every == 0 or position + 1 == n_frames:
                    print(
                        f"  Wrote {position + 1:,}/{n_frames:,} frames "
                        f"in {time.perf_counter() - write_started:.1f} s",
                        flush=True,
                    )
        os.replace(temporary_filtered_path, filtered_path)
    except BaseException:
        temporary_filtered_path.unlink(missing_ok=True)
        raise
    finally:
        del trajectory
        trajectory_path.unlink(missing_ok=True)

    return {
        "cutoff_hz": float(cutoff_hz),
        "filter_order": int(filter_order),
        "filter_type": FILTER_TYPE,
        "sampling_hz": float(sampling_hz),
        "relative_dt_spread": relative_dt_spread,
    }


def _write_provenance(
    pod_path: Path,
    raw_source: Path,
    filter_metadata: dict[str, float | int | str],
) -> None:
    """Attach production preprocessing provenance before output promotion."""
    with h5py.File(pod_path, "r+") as output:
        output.attrs["source_file"] = str(raw_source.resolve())
        output.attrs["input_full_field_highpass_applied"] = True
        output.attrs["input_highpass_cutoff_hz"] = filter_metadata["cutoff_hz"]
        output.attrs["input_highpass_filter_order"] = filter_metadata["filter_order"]
        output.attrs["input_highpass_filter_type"] = filter_metadata["filter_type"]
        output.attrs["input_highpass_sampling_hz"] = filter_metadata["sampling_hz"]
        output.attrs["input_highpass_applied_to"] = "every spatial pixel"
        output.attrs["input_highpass_zero_phase"] = True
        output.attrs["production_preprocessing"] = (
            "full-field temporal high-pass; temporal mean-field centering; "
            "instantaneous spatial mean retained"
        )
        preprocessing = output["preprocessing"]
        preprocessing.attrs["full_field_highpass_applied"] = True
        preprocessing.attrs["highpass_cutoff_hz"] = filter_metadata["cutoff_hz"]
        preprocessing.attrs["highpass_filter_order"] = filter_metadata["filter_order"]
        preprocessing.attrs["highpass_filter_type"] = filter_metadata["filter_type"]
        preprocessing.attrs["highpass_sampling_hz"] = filter_metadata["sampling_hz"]


def _existing_output_matches(
    output_path: Path,
    raw_source: Path,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    """Return whether an existing output has this production configuration."""
    try:
        with h5py.File(output_path, "r") as output:
            checks = {
                "rank": int(output.attrs.get("rank", -1)) == args.rank,
                "source_file": str(output.attrs.get("source_file", ""))
                == str(raw_source.resolve()),
                "full_field_highpass": bool(
                    output.attrs.get("input_full_field_highpass_applied", False)
                ),
                "cutoff_hz": np.isclose(
                    float(output.attrs.get("input_highpass_cutoff_hz", np.nan)),
                    args.cutoff_hz,
                    rtol=0.0,
                    atol=1e-12,
                ),
                "filter_order": int(
                    output.attrs.get("input_highpass_filter_order", -1)
                )
                == args.filter_order,
                "spatial_mean_retained": not bool(
                    output.attrs.get("instantaneous_spatial_mean_removed", True)
                ),
                "temporal_centering": bool(
                    output.attrs.get("temporal_mean_field_removed", False)
                ),
            }
    except (OSError, ValueError, TypeError) as error:
        return False, f"cannot validate HDF5 output: {error}"
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return False, "mismatched or missing: " + ", ".join(failed)
    return True, "matching production high-pass POD"


def run_batch(args: argparse.Namespace, log: TextIO) -> int:
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scratch_root = args.scratch_root.expanduser().resolve()
    args.pod_script = args.pod_script.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not args.pod_script.is_file():
        raise FileNotFoundError(f"POD script does not exist: {args.pod_script}")
    scratch_root.mkdir(parents=True, exist_ok=True)

    jobs, issues = discover_jobs(
        input_root,
        output_root,
        args.rank,
        powers=args.powers,
        reps=args.reps,
        file_pattern=args.file_pattern,
    )
    if args.max_files is not None:
        jobs = jobs[: args.max_files]

    announce(log, f"Input root: {input_root}")
    announce(log, f"Output root: {output_root}")
    announce(log, f"Scratch root: {scratch_root}")
    announce(
        log,
        f"Preprocessing: full-field {args.cutoff_hz:g} Hz zero-phase "
        f"Butterworth order {args.filter_order}; spatial mean retained; "
        "temporal mean field centered",
    )
    announce(
        log,
        f"POD: rank={args.rank}, training_frames={args.training_frames}, "
        f"oversampling={args.oversampling}, power_iterations={args.power_iterations}, "
        f"batch_size={args.batch_size}, BLAS threads={args.blas_threads}",
    )
    for issue in issues:
        announce(log, f"SKIP invalid folder {issue.directory}: {issue.reason}")
    announce(log, f"Discovered {len(jobs)} valid job(s).")
    if not jobs:
        return 1

    environment = subprocess_environment(args.blas_threads)
    succeeded = skipped = failed = 0
    batch_started = time.perf_counter()

    for index, job in enumerate(jobs, start=1):
        label = f"[{index}/{len(jobs)}] {job.condition}/{job.realization}"
        if job.output_path.exists() and not args.overwrite:
            matches, reason = _existing_output_matches(
                job.output_path, job.input_path, args
            )
            if matches:
                announce(log, f"SKIP existing {label}: {job.output_path}")
                skipped += 1
            else:
                announce(
                    log,
                    f"REFUSE incompatible existing output for {label}: {reason}. "
                    "Inspect it, then rerun with --overwrite if replacement is intended.",
                )
                failed += 1
            continue
        announce(log, f"START {label}: {job.input_path}")
        if args.dry_run:
            continue

        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        provisional_output = job.output_path.with_name(
            f".{job.output_path.name}.highpass_inprogress"
        )
        provisional_output.unlink(missing_ok=True)
        job_started = time.perf_counter()
        try:
            prefix = f"{job.condition}_{job.realization}_"
            with tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root) as temporary:
                work_directory = Path(temporary)
                filtered_path = work_directory / "full_field_highpass.hdf5"
                filter_metadata = materialize_highpass_input(
                    job.input_path,
                    filtered_path,
                    work_directory,
                    args.cutoff_hz,
                    args.filter_order,
                    args.filter_row_chunk,
                )
                staged_job = PodJob(
                    condition=job.condition,
                    realization=job.realization,
                    input_path=filtered_path,
                    output_path=provisional_output,
                )
                return_code = run_and_tee(
                    build_command(args, staged_job), environment, log
                )
                if return_code != 0 or not provisional_output.is_file():
                    raise RuntimeError(f"pod_2d.py failed with exit code {return_code}")
                _write_provenance(
                    provisional_output, job.input_path, filter_metadata
                )
                os.replace(provisional_output, job.output_path)
        except KeyboardInterrupt:
            provisional_output.unlink(missing_ok=True)
            announce(log, f"INTERRUPTED during {label}")
            return 130
        except BaseException:
            provisional_output.unlink(missing_ok=True)
            failed += 1
            announce(log, f"FAILED {label}; continuing")
            traceback.print_exc(file=log)
            log.flush()
            continue

        succeeded += 1
        announce(log, f"DONE {label} in {time.perf_counter() - job_started:.1f} s")

    elapsed = time.perf_counter() - batch_started
    if args.dry_run:
        announce(log, f"Dry run listed {len(jobs)} job(s) in {elapsed:.1f} s.")
        return 0
    announce(
        log,
        f"Batch summary: {succeeded} succeeded, {skipped} existing skipped, "
        f"{len(issues)} invalid skipped, {failed} failed; elapsed {elapsed:.1f} s.",
    )
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else output_root / f"highpass_pod_batch_{timestamp}.log"
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
