"""Smoke-compare POD energy under two temporal high-pass treatments.

Three snapshot ensembles are compared using the same sampled times and pixels:

1. the raw field;
2. every sampled pixel high-pass filtered over the complete recording; and
3. the raw spatial fluctuations with only the instantaneous spatial mean
   replaced by its high-pass-filtered trajectory.

Each ensemble is temporally centered before POD, matching the default POD
convention in :mod:`data_analysis.pod.pod_2d`. The full recording is retained
for filtering, but ``--spatial-stride`` and ``--pod-frames`` deliberately make
this a bounded diagnostic rather than a production full-resolution POD.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from data_analysis.energy.pod_gravitational_energy import highpass_spatial_mean
from data_analysis.pod.pod_2d import (
    _frame_key,
    inspect_input,
    randomized_spatial_subspace,
    refine_pod,
    stratified_training_positions,
)


TREATMENT_LABELS = {
    "raw": "Raw field",
    "full_highpass": "High-pass at every pixel",
    "filtered_mean_only": "Only spatial mean high-pass filtered",
}


@dataclass(frozen=True)
class TreatmentDiagnostics:
    name: str
    total_energy: float
    constant_energy_fraction: float
    mode_energy_fraction: np.ndarray
    cumulative_energy_fraction: np.ndarray
    constant_alignment_squared: np.ndarray
    constant_energy_attribution_fraction: np.ndarray
    most_constant_mode: int
    rank_captured_constant_fraction: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw keyed-frame HDF5 file.")
    parser.add_argument("--cutoff-hz", type=float, default=50.0)
    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=8,
        help="Keep every Nth x/y point (default: 8, giving 25 x 25 for a 200 x 200 grid).",
    )
    parser.add_argument(
        "--pod-frames",
        type=int,
        default=5_000,
        help="Stratified full-span snapshots used by each smoke POD (default: 5000).",
    )
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--oversampling", type=int, default=10)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--filter-columns", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument(
        "--treatments",
        nargs="+",
        choices=tuple(TREATMENT_LABELS),
        default=list(TREATMENT_LABELS),
        help=(
            "Treatments to compute (default: raw full_highpass filtered_mean_only). "
            "The raw treatment is required as the comparison reference."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/pod_highpass_comparison"),
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        help=(
            "Directory for temporary raw/filtered float32 memmaps. The default "
            "is --output-dir; use fast local scratch for batch runs."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace, n_frames: int, n_space: int) -> None:
    positive = ("spatial_stride", "pod_frames", "rank", "filter_columns", "dpi")
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.oversampling < 0 or args.power_iterations < 0 or args.seed < 0:
        raise ValueError("--oversampling, --power-iterations, and --seed must be nonnegative.")
    if args.pod_frames > n_frames:
        raise ValueError(f"--pod-frames exceeds the available {n_frames:,} frames.")
    if args.rank + args.oversampling > min(args.pod_frames, n_space):
        raise ValueError("rank + oversampling exceeds the sampled matrix dimensions.")
    if not np.isfinite(args.cutoff_hz) or args.cutoff_hz <= 0:
        raise ValueError("--cutoff-hz must be finite and positive.")
    if "raw" not in args.treatments:
        raise ValueError("--treatments must include raw as the comparison reference.")
    if len(set(args.treatments)) != len(args.treatments):
        raise ValueError("--treatments must not contain duplicates.")


def _load_spatially_sampled_trajectory(
    input_path: Path,
    frame_numbers: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    output: np.memmap,
) -> None:
    report_every = max(1, len(frame_numbers) // 10)
    started = time.perf_counter()
    with h5py.File(input_path, "r") as handle:
        frame_group = handle["main"]
        for index, frame_number in enumerate(frame_numbers):
            key = _frame_key(int(frame_number))
            if key not in frame_group:
                raise KeyError(f"Missing HDF5 frame /main/{key}.")
            frame = np.asarray(frame_group[key], dtype=np.float32)
            sampled = frame[np.ix_(y_indices, x_indices)]
            if not np.isfinite(sampled).all():
                raise ValueError(f"Frame /main/{key} contains nonfinite sampled values.")
            output[index] = sampled.reshape(-1)
            completed = index + 1
            if completed % report_every == 0 or completed == len(frame_numbers):
                print(
                    f"Raw extraction: {completed:,}/{len(frame_numbers):,} frames "
                    f"({100.0 * completed / len(frame_numbers):.0f}%, "
                    f"{time.perf_counter() - started:.1f} s)",
                    flush=True,
                )
    output.flush()


def _highpass_columns(
    values: np.ndarray,
    time_values: np.ndarray,
    cutoff_hz: float,
    output: np.memmap,
    column_batch: int,
) -> None:
    steps = np.diff(np.asarray(time_values, dtype=np.float64))
    mean_step = float(np.mean(steps))
    if mean_step <= 0 or not np.allclose(steps, mean_step, rtol=0.02, atol=0.0):
        raise ValueError("High-pass filtering requires a uniformly sampled increasing time grid.")
    sampling_hz = 1.0 / mean_step
    if cutoff_hz >= 0.5 * sampling_hz:
        raise ValueError(f"High-pass cutoff must be below Nyquist ({0.5 * sampling_hz:.6g} Hz).")
    sos = butter(4, cutoff_hz, btype="highpass", fs=sampling_hz, output="sos")
    started = time.perf_counter()
    for start in range(0, values.shape[1], column_batch):
        stop = min(start + column_batch, values.shape[1])
        filtered = sosfiltfilt(
            sos,
            np.asarray(values[:, start:stop], dtype=np.float64),
            axis=0,
        )
        output[:, start:stop] = filtered.astype(np.float32)
        print(
            f"Full-field filter: {stop:,}/{values.shape[1]:,} sampled pixels "
            f"({100.0 * stop / values.shape[1]:.0f}%, "
            f"{time.perf_counter() - started:.1f} s)",
            flush=True,
        )
    output.flush()


def _sample_treatment(
    name: str,
    raw: np.ndarray,
    fully_filtered: np.ndarray,
    positions: np.ndarray,
    raw_mean: np.ndarray,
    filtered_mean: np.ndarray,
) -> np.ndarray:
    if name == "raw":
        sampled = np.asarray(raw[positions], dtype=np.float32).copy()
    elif name == "full_highpass":
        sampled = np.asarray(fully_filtered[positions], dtype=np.float32).copy()
    elif name == "filtered_mean_only":
        sampled = np.asarray(raw[positions], dtype=np.float32).copy()
        sampled -= raw_mean[positions, None].astype(np.float32)
        sampled += filtered_mean[positions, None].astype(np.float32)
    else:
        raise ValueError(f"Unknown treatment {name!r}.")

    temporal_mean = sampled.mean(axis=0, dtype=np.float64)
    sampled -= temporal_mean.astype(np.float32)[None, :]
    if not np.isfinite(sampled).all():
        raise ValueError(f"Treatment {name!r} contains nonfinite values.")
    return sampled


def _analyze_treatment(
    name: str,
    snapshots: np.ndarray,
    rank: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
) -> TreatmentDiagnostics:
    total_energy = float(np.einsum("ij,ij->", snapshots, snapshots, dtype=np.float64))
    if not np.isfinite(total_energy) or total_energy <= 0:
        raise ValueError(f"Treatment {name!r} has no finite positive snapshot energy.")

    constant = np.full(snapshots.shape[1], 1.0 / np.sqrt(snapshots.shape[1]))
    constant_coefficients = snapshots.astype(np.float64) @ constant
    constant_energy = float(constant_coefficients @ constant_coefficients)
    constant_energy_fraction = constant_energy / total_energy

    candidate_basis, _ = randomized_spatial_subspace(
        snapshots,
        rank + oversampling,
        power_iterations,
        np.random.default_rng(seed),
    )
    candidate_coefficients = snapshots @ candidate_basis
    reduced_covariance = (
        candidate_coefficients.T.astype(np.float64)
        @ candidate_coefficients.astype(np.float64)
    )
    pod = refine_pod(candidate_basis, reduced_covariance, total_energy, rank)
    mode_energy_fraction = np.square(pod.singular_values) / total_energy
    cumulative_energy_fraction = np.minimum(
        np.cumsum(mode_energy_fraction), 1.0
    )
    constant_alignment_squared = np.clip(
        np.square(pod.modes.T.astype(np.float64) @ constant), 0.0, 1.0
    )
    constant_attribution = mode_energy_fraction * constant_alignment_squared
    most_constant_mode = int(np.argmax(constant_alignment_squared))
    captured = (
        float(np.sum(constant_attribution)) / constant_energy_fraction
        if constant_energy_fraction > 0
        else np.nan
    )
    if np.isfinite(captured):
        captured = float(np.clip(captured, 0.0, 1.0))
    return TreatmentDiagnostics(
        name=name,
        total_energy=total_energy,
        constant_energy_fraction=constant_energy_fraction,
        mode_energy_fraction=mode_energy_fraction,
        cumulative_energy_fraction=cumulative_energy_fraction,
        constant_alignment_squared=constant_alignment_squared,
        constant_energy_attribution_fraction=constant_attribution,
        most_constant_mode=most_constant_mode,
        rank_captured_constant_fraction=captured,
    )


def _write_csv(
    output_directory: Path,
    diagnostics: list[TreatmentDiagnostics],
    *,
    cutoff_hz: float,
    n_frames: int,
    pod_frames: int,
    sampled_shape: tuple[int, int],
    spatial_stride: int,
    mean_energy_retained: float,
    centered_mean_energy_retained: float,
    raw_mean_rms: float,
    filtered_mean_rms: float,
    raw_centered_mean_rms: float,
    filtered_centered_mean_rms: float,
    full_filter_mean_consistency_rms: float,
) -> None:
    raw_diagnostics = next(item for item in diagnostics if item.name == "raw")
    raw_total_energy = raw_diagnostics.total_energy
    raw_constant_energy = (
        raw_diagnostics.constant_energy_fraction * raw_total_energy
    )
    with (output_directory / "summary.csv").open("w", newline="") as handle:
        fields = (
            "treatment",
            "cutoff_hz",
            "n_recording_frames",
            "n_pod_frames",
            "sampled_ny",
            "sampled_nx",
            "spatial_stride",
            "total_energy_relative_to_raw",
            "constant_subspace_energy_fraction",
            "constant_energy_relative_to_raw_constant",
            "most_constant_mode_1based",
            "most_constant_mode_alignment_squared",
            "most_constant_mode_energy_fraction",
            "rank_energy_fraction",
            "rank_captured_constant_energy_fraction",
            "spatial_mean_energy_retained_fraction",
            "centered_spatial_mean_energy_retained_fraction",
            "raw_spatial_mean_rms",
            "filtered_spatial_mean_rms",
            "raw_centered_spatial_mean_rms",
            "filtered_centered_spatial_mean_rms",
            "full_filter_mean_consistency_rms",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in diagnostics:
            mode = item.most_constant_mode
            writer.writerow(
                {
                    "treatment": item.name,
                    "cutoff_hz": cutoff_hz,
                    "n_recording_frames": n_frames,
                    "n_pod_frames": pod_frames,
                    "sampled_ny": sampled_shape[0],
                    "sampled_nx": sampled_shape[1],
                    "spatial_stride": spatial_stride,
                    "total_energy_relative_to_raw": item.total_energy / raw_total_energy,
                    "constant_subspace_energy_fraction": item.constant_energy_fraction,
                    "constant_energy_relative_to_raw_constant": (
                        item.constant_energy_fraction * item.total_energy
                        / raw_constant_energy
                    ),
                    "most_constant_mode_1based": mode + 1,
                    "most_constant_mode_alignment_squared": item.constant_alignment_squared[mode],
                    "most_constant_mode_energy_fraction": item.mode_energy_fraction[mode],
                    "rank_energy_fraction": item.cumulative_energy_fraction[-1],
                    "rank_captured_constant_energy_fraction": item.rank_captured_constant_fraction,
                    "spatial_mean_energy_retained_fraction": mean_energy_retained,
                    "centered_spatial_mean_energy_retained_fraction": centered_mean_energy_retained,
                    "raw_spatial_mean_rms": raw_mean_rms,
                    "filtered_spatial_mean_rms": filtered_mean_rms,
                    "raw_centered_spatial_mean_rms": raw_centered_mean_rms,
                    "filtered_centered_spatial_mean_rms": filtered_centered_mean_rms,
                    "full_filter_mean_consistency_rms": full_filter_mean_consistency_rms,
                }
            )

    with (output_directory / "mode_diagnostics.csv").open("w", newline="") as handle:
        fields = (
            "treatment",
            "mode_1based",
            "mode_energy_fraction",
            "cumulative_energy_fraction",
            "constant_alignment_squared",
            "constant_energy_attribution_fraction_of_total",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in diagnostics:
            for index in range(len(item.mode_energy_fraction)):
                writer.writerow(
                    {
                        "treatment": item.name,
                        "mode_1based": index + 1,
                        "mode_energy_fraction": item.mode_energy_fraction[index],
                        "cumulative_energy_fraction": item.cumulative_energy_fraction[index],
                        "constant_alignment_squared": item.constant_alignment_squared[index],
                        "constant_energy_attribution_fraction_of_total": item.constant_energy_attribution_fraction[index],
                    }
                )


def _plot(output: Path, diagnostics: list[TreatmentDiagnostics], dpi: int) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for item in diagnostics:
        modes = np.arange(1, len(item.mode_energy_fraction) + 1)
        label = TREATMENT_LABELS[item.name]
        axes[0].plot(modes, 100.0 * item.mode_energy_fraction, marker="o", label=label)
        axes[1].plot(modes, 100.0 * item.cumulative_energy_fraction, marker="o")
        axes[2].plot(modes, item.constant_alignment_squared, marker="o")
    axes[0].set_ylabel("Energy per POD mode [%]")
    axes[1].set_ylabel("Cumulative POD energy [%]")
    axes[2].set_ylabel(r"Squared overlap with constant direction $|u_i^T c|^2$")
    for axis in axes:
        axis.set_xlabel("POD mode")
        axis.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser().resolve()
    info = inspect_input(input_path, 0, None)
    y_indices = np.arange(0, info.frame_shape[0], args.spatial_stride, dtype=np.int64)
    x_indices = np.arange(0, info.frame_shape[1], args.spatial_stride, dtype=np.int64)
    sampled_shape = (len(y_indices), len(x_indices))
    n_space = int(np.prod(sampled_shape))
    _validate_args(args, info.n_frames, n_space)

    output_directory = args.output_dir.expanduser().resolve()
    outputs = (
        output_directory / "summary.csv",
        output_directory / "mode_diagnostics.csv",
        output_directory / "comparison.png",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {existing[0]}. Pass --overwrite or choose another --output-dir."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    scratch_directory = (
        args.scratch_dir.expanduser().resolve()
        if args.scratch_dir is not None
        else output_directory
    )
    if not scratch_directory.is_dir():
        raise FileNotFoundError(f"Scratch directory does not exist: {scratch_directory}")
    selected_treatments = [
        name for name in TREATMENT_LABELS if name in args.treatments
    ]

    mean_step = float(np.mean(np.diff(info.time)))
    print(
        f"Input: {input_path}\n"
        f"Recording: {info.n_frames:,} frames, {info.time[-1] - info.time[0]:.6g} s, "
        f"sampling {1.0 / mean_step:.6g} Hz\n"
        f"Smoke grid: {sampled_shape[1]} x {sampled_shape[0]} "
        f"({n_space:,} pixels, stride {args.spatial_stride}); "
        f"POD snapshots: {args.pod_frames:,}; rank: {args.rank}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="pod_highpass_", dir=scratch_directory) as temporary:
        temporary_path = Path(temporary)
        raw = np.memmap(
            temporary_path / "raw.dat",
            mode="w+",
            dtype=np.float32,
            shape=(info.n_frames, n_space),
        )
        fully_filtered = np.memmap(
            temporary_path / "full_highpass.dat",
            mode="w+",
            dtype=np.float32,
            shape=(info.n_frames, n_space),
        )
        _load_spatially_sampled_trajectory(
            input_path, info.frame_numbers, y_indices, x_indices, raw
        )
        raw_mean = np.asarray(raw.mean(axis=1, dtype=np.float64))
        filtered_mean = highpass_spatial_mean(raw_mean, info.time, args.cutoff_hz)
        mean_energy_retained = float(
            np.mean(np.square(filtered_mean)) / np.mean(np.square(raw_mean))
        )
        raw_centered_mean = raw_mean - np.mean(raw_mean)
        filtered_centered_mean = filtered_mean - np.mean(filtered_mean)
        raw_centered_mean_square = float(np.mean(np.square(raw_centered_mean)))
        if raw_centered_mean_square == 0.0:
            raise ValueError(
                "Centered spatial-mean energy ratio is undefined because the raw "
                "spatial mean is temporally constant."
            )
        centered_mean_energy_retained = float(
            np.mean(np.square(filtered_centered_mean)) / raw_centered_mean_square
        )
        raw_mean_rms = float(np.sqrt(np.mean(np.square(raw_mean))))
        filtered_mean_rms = float(np.sqrt(np.mean(np.square(filtered_mean))))
        raw_centered_mean_rms = float(np.sqrt(raw_centered_mean_square))
        filtered_centered_mean_rms = float(
            np.sqrt(np.mean(np.square(filtered_centered_mean)))
        )
        _highpass_columns(
            raw,
            info.time,
            args.cutoff_hz,
            fully_filtered,
            args.filter_columns,
        )
        full_filtered_mean = fully_filtered.mean(axis=1, dtype=np.float64)
        mean_consistency_rms = float(
            np.sqrt(np.mean(np.square(full_filtered_mean - filtered_mean)))
        )

        positions = stratified_training_positions(
            info.n_frames, args.pod_frames, np.random.default_rng(args.seed)
        )
        diagnostics = []
        for treatment_index, name in enumerate(selected_treatments):
            print(f"Analyzing {TREATMENT_LABELS[name]}...", flush=True)
            snapshots = _sample_treatment(
                name,
                raw,
                fully_filtered,
                positions,
                raw_mean,
                filtered_mean,
            )
            diagnostics.append(
                _analyze_treatment(
                    name,
                    snapshots,
                    args.rank,
                    args.oversampling,
                    args.power_iterations,
                    args.seed + treatment_index + 1,
                )
            )

    _write_csv(
        output_directory,
        diagnostics,
        cutoff_hz=args.cutoff_hz,
        n_frames=info.n_frames,
        pod_frames=args.pod_frames,
        sampled_shape=sampled_shape,
        spatial_stride=args.spatial_stride,
        mean_energy_retained=mean_energy_retained,
        centered_mean_energy_retained=centered_mean_energy_retained,
        raw_mean_rms=raw_mean_rms,
        filtered_mean_rms=filtered_mean_rms,
        raw_centered_mean_rms=raw_centered_mean_rms,
        filtered_centered_mean_rms=filtered_centered_mean_rms,
        full_filter_mean_consistency_rms=mean_consistency_rms,
    )
    _plot(outputs[2], diagnostics, args.dpi)

    print(f"Spatial-mean energy retained at {args.cutoff_hz:g} Hz: {mean_energy_retained:.6g}")
    print(
        "Temporally centered spatial-mean energy retained: "
        f"{centered_mean_energy_retained:.6g}"
    )
    raw_diagnostics = next(item for item in diagnostics if item.name == "raw")
    raw_total_energy = raw_diagnostics.total_energy
    raw_constant_energy = raw_diagnostics.constant_energy_fraction * raw_total_energy
    for item in diagnostics:
        mode = item.most_constant_mode
        print(
            f"{TREATMENT_LABELS[item.name]}: constant-subspace energy "
            f"{item.constant_energy_fraction:.6%}; most constant mode {mode + 1} "
            f"(alignment^2={item.constant_alignment_squared[mode]:.6g}, "
            f"mode energy={item.mode_energy_fraction[mode]:.6%}); "
            f"rank-{args.rank} energy={item.cumulative_energy_fraction[-1]:.6%}; "
            f"absolute constant energy vs raw="
            f"{item.constant_energy_fraction * item.total_energy / raw_constant_energy:.6%}"
        )
    print(f"Saved smoke comparison: {output_directory}")
    return output_directory


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, FileExistsError, KeyError, TypeError, ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
