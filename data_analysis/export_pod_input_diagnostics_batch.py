"""Export compact full-field diagnostics matched to production POD inputs.

The batch is driven by the final ``pod_2d_r*.h5`` tree rather than by guesses
about experimental filenames.  Each POD file identifies its source recording
and whether production applied a temporary full-field 50 Hz high-pass filter.

For sources that are already stored as cleaned/high-passed HDF5 files, the
source is analyzed directly.  For low-forcing PODs whose high-pass input was
temporary, the exact production filter is recreated one recording at a time in
local scratch and deleted afterward.  The stored POD temporal-mean field is
then subtracted, while the instantaneous spatial mean remains in the field.

Only compact numerical products are retained:

* center-point Welch frequency PSD (no figures or center trajectory), and
* sampled exact and quadratic capillary surface-energy trajectories.

The output HDF5 is restartable and accompanied by CSV/JSON manifests suitable
for transfer to another machine.  Run only one process per output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import h5py
import numpy as np

from data_analysis.energy.pod_capillary_energy import (
    field_energies,
    fingerprint,
    length_scale,
    quadrature_weights,
    select_energy_frames,
)
from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.pod.run_highpass_pod_batch import materialize_highpass_input
from data_analysis.psd.pod_center_psd import compute_welch, infer_sampling_frequency


DEFAULT_REDUCED_ROOT = Path("/disk/hyk049/jschupp/reduced_data/clean_red_data")
DEFAULT_RAW_ROOT = Path("/disk/hyk049/DHM_new_experiment")
DEFAULT_SCRATCH_ROOT = Path("/dev/shm") / (
    f"{os.environ.get('USER', 'user')}_pod_input_diagnostics"
)
CONDITION_PATTERN = re.compile(r"0p\d+")
REALIZATION_PATTERN = re.compile(r"(?:Ca_ac.*|rep\d+)")
REPETITION_PATTERN = re.compile(r"(?:.*_rep|rep)(\d+)$")
HDF5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class Job:
    power: str
    realization: str
    repetition: int
    pod_path: Path
    source_path: Path
    recreate_highpass: bool


@dataclass(frozen=True)
class Issue:
    path: Path
    reason: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced-root", type=Path, default=DEFAULT_REDUCED_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--powers", nargs="+")
    parser.add_argument("--reps", nargs="+", type=int)
    parser.add_argument("--pod-rank", type=int, default=200)
    parser.add_argument("--nperseg", type=int, default=8192)
    parser.add_argument("--overlap-fraction", type=float, default=0.5)
    parser.add_argument("--energy-frames", type=int, default=2000)
    parser.add_argument("--frame-seed", type=int, default=12345)
    parser.add_argument("--gamma", type=float, default=0.0728)
    parser.add_argument("--energy-batch-size", type=int, default=64)
    parser.add_argument("--cutoff-hz", type=float, default=50.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--filter-row-chunk", type=int, default=4)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--skip-psd", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive = (
        args.pod_rank,
        args.nperseg,
        args.energy_frames,
        args.energy_batch_size,
        args.filter_order,
        args.filter_row_chunk,
    )
    if min(positive) < 1 or args.gamma <= 0 or args.cutoff_hz <= 0:
        parser.error("Ranks, counts, gamma, and cutoff must be positive.")
    if not 0 <= args.overlap_fraction < 1 or args.frame_seed < 0:
        parser.error("Overlap must lie in [0,1) and the seed must be nonnegative.")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive.")
    if args.reps is not None and any(value < 1 for value in args.reps):
        parser.error("--reps values must be positive.")
    if args.skip_psd and args.skip_energy:
        parser.error("At least one of PSD and energy must be enabled.")
    return args


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _rep_number(name: str) -> int | None:
    match = REPETITION_PATTERN.fullmatch(name)
    return int(match.group(1)) if match else None


def _source_from_pod(
    pod_path: Path,
    raw_root: Path,
    power: str,
    realization: str,
) -> tuple[Path | None, str | None]:
    with h5py.File(pod_path, "r") as pod:
        source_text = _attribute_text(pod.attrs.get("source_file"))
    source = Path(source_text).expanduser() if source_text else None
    if source is not None and source.is_file():
        return source.resolve(), None

    # Production POD paths normally remain valid on the lab computer.  This
    # fallback handles a moved original experiment tree, including 0p07/repN.
    fallback = raw_root / power / realization
    if fallback.is_dir():
        candidates = sorted(
            path
            for path in fallback.iterdir()
            if path.is_file()
            and path.suffix.lower() in HDF5_SUFFIXES
            and not path.name.startswith("pod_2d_")
        )
        if len(candidates) == 1:
            return candidates[0].resolve(), None
        fallback_note = f"; fallback contained {len(candidates)} HDF5 files"
    else:
        fallback_note = "; fallback directory does not exist"
    return None, f"POD source_file is unavailable: {source_text!r}{fallback_note}"


def _validate_pod_configuration(pod_path: Path, rank: int) -> tuple[bool, str | None]:
    try:
        with h5py.File(pod_path, "r") as pod:
            stored_rank = (
                int(pod.attrs["rank"])
                if "rank" in pod.attrs
                else len(pod["pod/modes"])
            )
            if stored_rank != rank:
                return False, f"stored rank is {stored_rank}, expected {rank}"
            if not bool(pod.attrs.get("temporal_mean_field_removed", False)):
                return False, "POD temporal mean field was not removed"
            if bool(pod.attrs.get("instantaneous_spatial_mean_removed", True)):
                return False, "POD removed the instantaneous spatial mean"
            mean = pod["preprocessing/temporal_mean_field"]
            if not bool(mean.attrs.get("subtracted_before_pod", False)):
                return False, "saved temporal mean was not subtracted before POD"
            recreate = bool(pod.attrs.get("input_full_field_highpass_applied", False))
    except (KeyError, OSError, ValueError) as error:
        return False, f"invalid POD file: {error}"
    return recreate, None


def discover(args: argparse.Namespace) -> tuple[list[Job], list[Issue]]:
    root = args.reduced_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    if not root.is_dir():
        return [], [Issue(root, "reduced-data root does not exist")]
    jobs: list[Job] = []
    issues: list[Issue] = []
    conditions = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and CONDITION_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: float(path.name.replace("p", ".")),
    )
    if args.powers:
        available = {path.name for path in conditions}
        for missing in sorted(set(args.powers) - available):
            issues.append(Issue(root / missing, "requested power directory missing"))
        conditions = [path for path in conditions if path.name in args.powers]
    selected_reps = set(args.reps) if args.reps else None
    for condition in conditions:
        realizations = sorted(
            (
                path
                for path in condition.iterdir()
                if path.is_dir() and REALIZATION_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: (_rep_number(path.name) or -1, path.name),
        )
        if not realizations:
            issues.append(Issue(condition, "no Ca_ac_*_repN or repN folders"))
        for realization in realizations:
            repetition = _rep_number(realization.name)
            if repetition is None:
                issues.append(Issue(realization, "could not parse repetition number"))
                continue
            if selected_reps is not None and repetition not in selected_reps:
                continue
            pod_path = realization / f"pod_2d_r{args.pod_rank}.h5"
            if not pod_path.is_file():
                issues.append(Issue(realization, f"missing {pod_path.name}"))
                continue
            recreate, error = _validate_pod_configuration(pod_path, args.pod_rank)
            if error:
                issues.append(Issue(pod_path, error))
                continue
            source, error = _source_from_pod(
                pod_path, raw_root, condition.name, realization.name
            )
            if error or source is None:
                issues.append(Issue(pod_path, error or "source resolution failed"))
                continue
            jobs.append(
                Job(
                    power=condition.name,
                    realization=realization.name,
                    repetition=repetition,
                    pod_path=pod_path.resolve(),
                    source_path=source,
                    recreate_highpass=recreate,
                )
            )
    if args.max_files is not None:
        jobs = jobs[: args.max_files]
    return jobs, issues


def _check_stored_highpass(source_path: Path, args: argparse.Namespace) -> dict[str, object]:
    with h5py.File(source_path, "r") as source:
        applied = bool(source.attrs.get("highpass_filter_applied", False))
        cutoff = float(source.attrs.get("highpass_cutoff_hz", np.nan))
        order = int(source.attrs.get("highpass_filter_order", -1))
        filter_type = _attribute_text(source.attrs.get("highpass_filter_type"))
    if not applied:
        raise ValueError(
            "POD says no temporary full-field filter was applied, but its source "
            "is not marked high-pass filtered. Refusing an unmatched comparison."
        )
    if not np.isclose(cutoff, args.cutoff_hz, rtol=0.0, atol=1e-12):
        raise ValueError(f"Stored source cutoff is {cutoff:g} Hz, expected {args.cutoff_hz:g} Hz.")
    if order != args.filter_order:
        raise ValueError(f"Stored source filter order is {order}, expected {args.filter_order}.")
    return {
        "mode": "stored_preprocessed_source",
        "cutoff_hz": cutoff,
        "filter_order": order,
        "filter_type": filter_type,
    }


def _pod_mean_field(
    pod_path: Path, info, effective_path: Path
) -> tuple[np.ndarray, float]:
    with h5py.File(pod_path, "r") as pod:
        x = np.asarray(pod["grid/x"], dtype=float)
        y = np.asarray(pod["grid/y"], dtype=float)
        time_values = np.asarray(pod["grid/time"], dtype=float)
        if x.shape != info.x.shape or not np.allclose(x, info.x, rtol=0, atol=1e-7):
            raise ValueError(f"POD/effective-input x grids differ: {effective_path}")
        if y.shape != info.y.shape or not np.allclose(y, info.y, rtol=0, atol=1e-7):
            raise ValueError(f"POD/effective-input y grids differ: {effective_path}")
        if time_values.shape != info.time.shape or not np.allclose(
            time_values, info.time, rtol=0, atol=1e-12
        ):
            raise ValueError(f"POD/effective-input time grids differ: {effective_path}")
        dataset = pod["preprocessing/temporal_mean_field"]
        mean = np.asarray(dataset, dtype=float)
        mean_scale = length_scale(_attribute_text(dataset.attrs.get("units")))
    if mean.shape != info.frame_shape or not np.isfinite(mean).all():
        raise ValueError(f"Invalid POD temporal mean field: {pod_path}")
    z_scale = length_scale(info.units["z"])
    if not np.isclose(mean_scale, z_scale, rtol=0, atol=0):
        raise ValueError("POD mean-field and effective-input height units differ.")
    return mean, z_scale


def compute_center_psd(
    effective_path: Path,
    mean_field_native: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    info = inspect_input(effective_path, 0, None)
    center_y, center_x = info.frame_shape[0] // 2, info.frame_shape[1] // 2
    signal = np.empty(info.n_frames, dtype=np.float64)
    with h5py.File(info.path, "r") as source:
        frames = source["main"]
        for position, frame_number in enumerate(info.frame_numbers):
            signal[position] = float(frames[_frame_key(frame_number)][center_y, center_x])
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite raw center signal.")
    signal -= float(mean_field_native[center_y, center_x])
    sampling_hz, relative_jitter = infer_sampling_frequency(info.time)
    result, nperseg, noverlap = compute_welch(
        signal, sampling_hz, args.nperseg, args.overlap_fraction
    )
    arrays = {
        "frequency_hz": result.frequency,
        "power_spectral_density": result.density,
    }
    metadata = {
        "sampling_hz": sampling_hz,
        "maximum_relative_timestep_variation": relative_jitter,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "window": "hann",
        "detrend": "constant",
        "scaling": "density",
        "center_y_index": center_y,
        "center_x_index": center_x,
        "center_y_coordinate": float(info.y[center_y]),
        "center_x_coordinate": float(info.x[center_x]),
        "x_units": info.units["x"],
        "y_units": info.units["y"],
        "signal_units": info.units["z"],
        "samples": info.n_frames,
    }
    return arrays, metadata


def compute_energy(
    effective_path: Path,
    mean_field_native: np.ndarray,
    z_scale: float,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    info = inspect_input(effective_path, 0, None)
    indices, weights = select_energy_frames(info.time, args.energy_frames, args.frame_seed)
    x_m = info.x * length_scale(info.units["x"])
    y_m = info.y * length_scale(info.units["y"])
    mean_field_m = mean_field_native * z_scale
    energies = np.empty((2, len(indices)), dtype=np.float64)
    with h5py.File(info.path, "r") as source:
        frames = source["main"]
        for start in range(0, len(indices), args.energy_batch_size):
            stop = min(start + args.energy_batch_size, len(indices))
            block = np.stack(
                [
                    np.asarray(
                        frames[_frame_key(info.frame_numbers[position])],
                        dtype=np.float32,
                    )
                    for position in indices[start:stop]
                ]
            )
            field_m = block.astype(np.float64) * z_scale - mean_field_m[None]
            quadratic, exact = field_energies(field_m, x_m, y_m)
            energies[:, start:stop] = args.gamma * np.stack([quadratic, exact])
    means = energies @ weights
    area = float(quadrature_weights(x_m).sum() * quadrature_weights(y_m).sum())
    arrays = {
        "frame_indices": indices,
        "frame_numbers": info.frame_numbers[indices],
        "time_seconds": info.time[indices],
        "mean_weights": weights,
        "E_quadratic_J": energies[0],
        "E_exact_J": energies[1],
    }
    metadata = {
        "Ebar_quadratic_J": float(means[0]),
        "Ebar_exact_J": float(means[1]),
        "quadratic_excess_over_exact_percent": float(100.0 * (means[0] / means[1] - 1.0)),
        "rms_spatial_slope": float(np.sqrt(2.0 * means[0] / (args.gamma * area))),
        "area_m2": area,
        "frames_evaluated": len(indices),
        "full_frame_count": info.n_frames,
        "frame_seed": args.frame_seed,
        "gamma_N_per_m": args.gamma,
        "temporal_mean_removed": True,
        "instantaneous_spatial_mean_removed": False,
    }
    return arrays, metadata


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _completed_rows(cache: h5py.File) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if "recordings" in cache:
        for power in cache["recordings"].values():
            for recording in power.values():
                if bool(recording.attrs.get("complete", False)):
                    rows.append(json.loads(recording.attrs["summary_json"]))
    rows.sort(key=lambda row: (float(str(row["power"]).replace("p", ".")), int(row["repetition"])))
    return rows


def _refresh_summary(cache: h5py.File, output_dir: Path) -> int:
    rows = _completed_rows(cache)
    if rows:
        fields = list(rows[0])
        _atomic_csv(output_dir / "recording_summary.csv", fields, rows)
    return len(rows)


def _write_arrays(group: h5py.Group, arrays: dict[str, np.ndarray]) -> None:
    for name, values in arrays.items():
        dataset = group.create_dataset(name, data=values, compression="gzip", shuffle=True)
        if name == "frequency_hz":
            dataset.attrs["units"] = "Hz"
        elif name == "power_spectral_density":
            dataset.attrs["units"] = "height_units^2/Hz"
        elif name.startswith("E_"):
            dataset.attrs["units"] = "J"
        elif name == "time_seconds":
            dataset.attrs["units"] = "s"


def _configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pod_rank": args.pod_rank,
        "compute_psd": not args.skip_psd,
        "psd": {
            "window": "hann",
            "nperseg": args.nperseg,
            "overlap_fraction": args.overlap_fraction,
            "detrend": "constant",
            "scaling": "density",
        },
        "compute_energy": not args.skip_energy,
        "energy": {
            "frames": args.energy_frames,
            "frame_seed": args.frame_seed,
            "gamma_N_per_m": args.gamma,
            "spatial_derivative": "numpy.gradient edge_order=2",
            "spatial_quadrature": "trapezoidal",
        },
        "matched_preprocessing": {
            "cutoff_hz": args.cutoff_hz,
            "filter_order": args.filter_order,
            "filter_type": "butterworth_zero_phase_sosfiltfilt",
            "temporal_mean_field_removed": True,
            "instantaneous_spatial_mean_retained": True,
        },
    }


def _summary_for(
    job: Job,
    preprocessing: dict[str, object],
    psd_metadata: dict[str, object] | None,
    energy_metadata: dict[str, object] | None,
    elapsed: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "power": job.power,
        "realization": job.realization,
        "repetition": job.repetition,
        "pod_file": str(job.pod_path),
        "source_file": str(job.source_path),
        "preprocessing_mode": preprocessing["mode"],
        "highpass_cutoff_hz": preprocessing["cutoff_hz"],
        "highpass_filter_order": preprocessing["filter_order"],
    }
    if psd_metadata is not None:
        row.update(
            sampling_hz=psd_metadata["sampling_hz"],
            psd_nperseg=psd_metadata["nperseg"],
            psd_noverlap=psd_metadata["noverlap"],
            center_y_index=psd_metadata["center_y_index"],
            center_x_index=psd_metadata["center_x_index"],
            signal_units=psd_metadata["signal_units"],
        )
    if energy_metadata is not None:
        row.update(energy_metadata)
    row["elapsed_seconds"] = elapsed
    return row


def run(args: argparse.Namespace) -> int:
    args.reduced_root = args.reduced_root.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.scratch_root = args.scratch_root.expanduser().resolve()
    jobs, issues = discover(args)
    for job in jobs:
        action = "recreate 50 Hz full-field input" if job.recreate_highpass else "use stored preprocessed input"
        print(f"{job.power}/{job.realization}: {action}; {job.source_path}")
    for issue in issues:
        print(f"DISCOVERY ISSUE {issue.path}: {issue.reason}")
    print(f"Discovered {len(jobs)} jobs and {len(issues)} issues.")
    if args.dry_run:
        return int(not jobs or bool(issues))
    if not jobs:
        raise ValueError("No valid jobs. Run with --dry-run to inspect discovery.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(args)
    encoded = json.dumps(configuration, sort_keys=True)
    (args.output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    _atomic_csv(
        args.output_dir / "discovery_issues.csv",
        ["path", "reason"],
        [{"path": str(issue.path), "reason": issue.reason} for issue in issues],
    )
    manifest = [
        {
            "power": job.power,
            "realization": job.realization,
            "repetition": job.repetition,
            "pod_file": str(job.pod_path),
            "source_file": str(job.source_path),
            "recreate_highpass": job.recreate_highpass,
        }
        for job in jobs
    ]
    _atomic_csv(args.output_dir / "manifest.csv", list(manifest[0]), manifest)
    log_path = args.output_dir / (
        "batch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log"
    )
    failures = 0
    archive_path = args.output_dir / "pod_input_diagnostics.h5"
    with h5py.File(archive_path, "a") as cache, log_path.open("w", encoding="utf-8") as log:
        if cache.attrs.get("analysis_configuration", encoded) != encoded:
            raise ValueError("Output archive uses different settings; choose another --output-dir.")
        cache.attrs["analysis_configuration"] = encoded
        cache.attrs["description"] = "Full-field diagnostics matched to production POD preprocessing"

        def status(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        _refresh_summary(cache, args.output_dir)
        for index, job in enumerate(jobs, start=1):
            key = f"recordings/{job.power}/{job.realization}"
            pod_mark = json.dumps(fingerprint(job.pod_path), sort_keys=True)
            source_mark = json.dumps(fingerprint(job.source_path), sort_keys=True)
            if (
                key in cache
                and bool(cache[key].attrs.get("complete", False))
                and cache[key].attrs.get("pod_fingerprint") == pod_mark
                and cache[key].attrs.get("source_fingerprint") == source_mark
                and not args.overwrite
            ):
                status(f"[{index}/{len(jobs)}] cached {job.power}/{job.realization}")
                continue
            started = time.perf_counter()
            pending = f"_pending/{job.power}/{job.realization}"
            try:
                if pending in cache:
                    del cache[pending]
                status(f"[{index}/{len(jobs)}] starting {job.power}/{job.realization}")
                scratch_context = (
                    tempfile.TemporaryDirectory(
                        prefix=f"{job.power}_{job.repetition}_", dir=args.scratch_root
                    )
                    if job.recreate_highpass
                    else contextlib.nullcontext(None)
                )
                with scratch_context as scratch_name:
                    if job.recreate_highpass:
                        scratch = Path(str(scratch_name))
                        effective_path = scratch / "matched_highpass_input.hdf5"
                        preprocessing = materialize_highpass_input(
                            job.source_path,
                            effective_path,
                            scratch,
                            args.cutoff_hz,
                            args.filter_order,
                            args.filter_row_chunk,
                        )
                        preprocessing = {"mode": "recreated_temporary_highpass", **preprocessing}
                    else:
                        effective_path = job.source_path
                        preprocessing = _check_stored_highpass(job.source_path, args)

                    info = inspect_input(effective_path, 0, None)
                    mean_field_native, z_scale = _pod_mean_field(
                        job.pod_path, info, effective_path
                    )
                    psd_arrays = psd_metadata = None
                    energy_arrays = energy_metadata = None
                    if not args.skip_psd:
                        psd_arrays, psd_metadata = compute_center_psd(
                            effective_path, mean_field_native, args
                        )
                    if not args.skip_energy:
                        energy_arrays, energy_metadata = compute_energy(
                            effective_path, mean_field_native, z_scale, args
                        )

                elapsed = time.perf_counter() - started
                summary = _summary_for(
                    job, preprocessing, psd_metadata, energy_metadata, elapsed
                )
                group = cache.create_group(pending)
                group.attrs["complete"] = False
                group.attrs["pod_fingerprint"] = pod_mark
                group.attrs["source_fingerprint"] = source_mark
                group.attrs["preprocessing_json"] = json.dumps(preprocessing)
                group.attrs["summary_json"] = json.dumps(summary)
                if psd_arrays is not None and psd_metadata is not None:
                    psd_group = group.create_group("center_psd")
                    _write_arrays(psd_group, psd_arrays)
                    for name, value in psd_metadata.items():
                        psd_group.attrs[name] = value
                if energy_arrays is not None and energy_metadata is not None:
                    energy_group = group.create_group("capillary_energy")
                    _write_arrays(energy_group, energy_arrays)
                    for name, value in energy_metadata.items():
                        energy_group.attrs[name] = value
                group.attrs["complete"] = True
                cache.flush()
                if key in cache:
                    del cache[key]
                cache.require_group(str(Path(key).parent))
                cache.move(pending, key)
                cache.flush()
                _refresh_summary(cache, args.output_dir)
                energy_note = (
                    f"; exact={float(energy_metadata['Ebar_exact_J'])*1e12:.4g} pJ"
                    if energy_metadata is not None
                    else ""
                )
                status(f"  saved in {elapsed:.1f} s{energy_note}")
            except Exception as error:
                failures += 1
                status(f"  FAILED {job.power}/{job.realization}: {error}")
                log.write(traceback.format_exc() + "\n")
                log.flush()
                if pending in cache:
                    del cache[pending]
                    cache.flush()
        completed = _refresh_summary(cache, args.output_dir)
        cache.attrs["completed_recordings"] = completed
        cache.attrs["last_run_failures"] = failures
        status(
            f"Finished: {completed} completed, {failures} failed, "
            f"{len(issues)} discovery issues."
        )

    (args.output_dir / "README.md").write_text(
        """# POD-input diagnostics export

Transfer this complete folder to the analysis computer. The original full-field
HDF5 and POD files are not required for plotting the exported diagnostics.

`pod_input_diagnostics.h5` contains, per recording, the center frequency PSD
and sampled exact/quadratic capillary-energy trajectories. `recording_summary.csv`
contains scalar metadata and time-averaged energy estimates. `manifest.csv`,
`configuration.json`, `discovery_issues.csv`, and the logs preserve provenance.

The evaluated field matches the production POD input: 50 Hz zero-phase
full-field high-pass, stored POD temporal-mean removal, and retained
instantaneous spatial mean. Sources already stored in that form are read
directly. Otherwise the filtered full field is recreated sequentially in local
scratch and deleted after the compact diagnostics have been committed.

PSD uses a Hann Welch frequency density with constant detrending. Capillary
energy uses the full spatial grid, second-order finite differences, trapezoidal
quadrature, and uniform random frame sampling without replacement. No POD
reconstruction, spatial smoothing, or interpolation enters these exports.
""",
        encoding="utf-8",
    )
    return int(bool(failures or issues))


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
