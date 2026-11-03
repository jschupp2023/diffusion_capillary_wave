"""Compare generated, rank-limited reference, and raw experimental PSDs.

Only rollout conditions belonging to the repetition represented by ``pod_input``
are used.  The raw experimental center signal is given the same spatial-mean
preprocessing as the rollout: its original instantaneous spatial mean is
replaced by the spatial mean stored for the rollout's configured high-pass
cutoff.  The time-independent temporal mean field is irrelevant because Welch
detrends every segment.  An optional fourth curve estimates the rank-limited
reference PSD under the checkpoint's exact training-trim rule.  Retained frames
are never concatenated: one centered contiguous clean window is used per
eligible run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory
from data_analysis.rollout_metrics import reference_training_trim_population
from modelling.data_preparation.prepare_shared_pod_training import (
    spatial_mean_dataset_path,
)
from .pod_center_psd import select_point
from .rollout_psd import ensemble_psd, point_signals


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _centered_clean_windows(
    signals: np.ndarray,
    keep_mask: np.ndarray,
    nperseg: int,
) -> np.ndarray:
    """Extract one centered window from every sufficiently long clean run."""
    values = np.asarray(signals, dtype=np.float64)
    keep = np.asarray(keep_mask, dtype=bool)
    if values.ndim != 2 or keep.shape != values.shape:
        raise ValueError("Trimmed PSD signals and mask must have matching [trajectory, time] shapes.")
    if nperseg < 1:
        raise ValueError("Trimmed PSD window length must be positive.")

    windows = []
    for trajectory, trajectory_keep in zip(values, keep, strict=True):
        padded = np.pad(trajectory_keep.astype(np.int8), (1, 1))
        edges = np.flatnonzero(np.diff(padded))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            if stop - start < nperseg:
                continue
            window_start = start + (stop - start - nperseg) // 2
            windows.append(trajectory[window_start:window_start + nperseg])
    if not windows:
        raise ValueError(
            f"Training trim leaves no contiguous clean run of at least {nperseg} samples."
        )
    return np.stack(windows)


def _matching_experimental_signals(
    rollout: Rollout,
    raw_path: Path,
    pod_path: Path,
    point: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    if rollout.metadata.get("temporal_preprocessing", {}).get("downsample_factor", 1) != 1:
        raise ValueError("Raw experimental PSD comparison does not yet apply the saved anti-alias filter. "
                         "Use plot_rollout/psd_peaks with the saved filtered reference for downsampled runs.")
    if rollout.start_index is None:
        raise ValueError("Rollout does not contain source start indices.")
    if rollout.spatial_mean_highpass_hz is None:
        raise ValueError("This diagnostic expects a saved spatial-mean high-pass cutoff.")

    repetition = pod_path.parent.name
    selected_conditions = np.flatnonzero(rollout.repetitions == repetition)
    if not len(selected_conditions):
        raise ValueError(f"Rollout has no conditions from {repetition}.")

    raw_info = inspect_input(raw_path, 0, None)
    y_index, x_index = point
    if not (0 <= y_index < raw_info.frame_shape[0] and 0 <= x_index < raw_info.frame_shape[1]):
        raise IndexError(f"Point {point} is outside raw frame shape {raw_info.frame_shape}.")

    with h5py.File(pod_path, "r") as pod:
        recorded_raw = Path(_text(pod.attrs.get("source_file", ""))).name
        if recorded_raw and recorded_raw != raw_path.name:
            raise ValueError(
                f"POD source is {recorded_raw}, but raw input is {raw_path.name}."
            )
        required = (
            "grid/time",
            "grid/x",
            "grid/y",
            "source/frame_number",
            "preprocessing/frame_spatial_mean",
            spatial_mean_dataset_path(rollout.spatial_mean_highpass_hz),
        )
        missing = [name for name in required if name not in pod]
        if missing:
            raise KeyError(f"POD file is missing: {', '.join(missing)}")
        pod_time = np.asarray(pod["grid/time"], dtype=np.float64)
        frame_number = np.asarray(pod["source/frame_number"], dtype=np.int64)
        raw_mean = np.asarray(pod["preprocessing/frame_spatial_mean"], dtype=np.float64)
        filtered_mean = np.asarray(
            pod[spatial_mean_dataset_path(rollout.spatial_mean_highpass_hz)],
            dtype=np.float64,
        )
        pod_x = np.asarray(pod["grid/x"], dtype=np.float64)
        pod_y = np.asarray(pod["grid/y"], dtype=np.float64)
        signal_units = _text(pod["preprocessing/frame_spatial_mean"].attrs.get("units", ""))

    if raw_info.frame_shape != rollout.modes.shape[1:]:
        raise ValueError("Raw, POD, and shared-basis spatial shapes do not match.")
    if not (np.allclose(raw_info.x, pod_x) and np.allclose(raw_info.y, pod_y)):
        raise ValueError("Raw and POD spatial grids do not match.")
    if not np.allclose(raw_info.time, pod_time, rtol=0, atol=2e-7):
        raise ValueError("Raw and POD time grids do not match.")

    n_samples = rollout.reference.shape[1]
    positions = rollout.start_index[selected_conditions, None] + (
        np.arange(n_samples, dtype=np.int64)[None, :] * rollout.lag_steps
    )
    if positions.min() < 0 or positions.max() >= len(pod_time):
        raise IndexError("A selected rollout window lies outside the POD source.")
    expected_time = pod_time[positions]
    if not np.allclose(rollout.time[selected_conditions], expected_time, rtol=0, atol=2e-7):
        raise ValueError("Rollout and source time windows do not align.")

    unique_positions, inverse = np.unique(positions, return_inverse=True)
    center_values = np.empty(len(unique_positions), dtype=np.float64)
    with h5py.File(raw_path, "r") as raw:
        frames = raw["main"]
        for row, source_position in enumerate(unique_positions):
            key = _frame_key(frame_number[source_position])
            if key not in frames:
                raise KeyError(f"Missing raw frame /main/{key}.")
            center_values[row] = float(frames[key][y_index, x_index])
    raw_center = center_values[inverse].reshape(positions.shape)
    original_mean = raw_mean[positions]
    highpass_mean = filtered_mean[positions]

    # Coordinate zero must reproduce exactly the stored filtered spatial mean.
    reference_mean = rollout.reference[selected_conditions, :, 0] / np.sqrt(rollout.n_space)
    if not np.allclose(reference_mean, highpass_mean, rtol=2e-5, atol=2e-4):
        raise ValueError("Rollout reference spatial mean does not match the POD source.")

    experimental = {
        "center": raw_center - original_mean + highpass_mean,
        "center_without_spatial_mean": raw_center - original_mean,
        "spatial_mean": highpass_mean,
    }
    provenance = {
        "raw_input": str(raw_path),
        "pod_input": str(pod_path),
        "repetition": repetition,
        "selected_condition_indices": selected_conditions.tolist(),
        "selected_start_indices": rollout.start_index[selected_conditions].tolist(),
        "experimental_preprocessing": (
            "raw center - original 2-D frame spatial mean + saved filtered 2-D "
            f"spatial mean at {rollout.spatial_mean_highpass_hz:g} Hz"
        ),
        "raw_center_index": [int(y_index), int(x_index)],
        "raw_center_coordinate": [float(raw_info.y[y_index]), float(raw_info.x[x_index])],
        "signal_units": signal_units,
    }
    return selected_conditions, experimental, provenance


def analyze(
    rollout: Rollout,
    raw_input: Path,
    pod_input: Path,
    *,
    point: tuple[int, int] | None = None,
    nperseg: int = 1024,
    overlap: float = 0.5,
    include_trimmed_reference: bool = False,
    output: Path | None = None,
    overwrite: bool = False,
) -> Path:
    if nperseg < 4 or not 0 <= overlap < 1:
        raise ValueError("nperseg must be >= 4 and overlap must lie in [0, 1).")
    raw_path = raw_input.expanduser().resolve()
    pod_path = pod_input.expanduser().resolve()
    selected = select_point(
        rollout.modes.shape[1:], rollout.x, rollout.y,
        rollout.x_units, rollout.y_units, point, None,
    )
    selected_point = (selected.y_index, selected.x_index)
    selected_conditions, experimental, raw_provenance = _matching_experimental_signals(
        rollout, raw_path, pod_path, selected_point
    )

    generated_coordinates = rollout.generated[selected_conditions]
    reference_coordinates = rollout.reference[selected_conditions]
    generated = point_signals(
        generated_coordinates, rollout.modes, rollout.n_space, selected,
    )
    reference = point_signals(
        reference_coordinates, rollout.modes, rollout.n_space, selected,
    )
    trimmed_reference = None
    if include_trimmed_reference:
        reference_mask, trim_report = reference_training_trim_population(rollout)
        if reference_mask is None:
            raise ValueError(
                "The rollout does not contain an available checkpoint training-trim rule."
            )
        selected_mask = reference_mask[selected_conditions]
        trimmed_reference = {
            name: _centered_clean_windows(values, selected_mask, nperseg)
            for name, values in reference.items()
        }

    report = rollout.provenance()
    report.update(raw_provenance)
    report.update(
        estimator="Welch: Hann window, constant detrend, density scaling",
        comparison=(
            "PSD is computed separately for each trajectory, then averaged; "
            "bands show the 10--90% trajectory range; the training-trimmed "
            "reference is shown as a mean line without a window-range band"
        ),
        signals={},
    )
    if trimmed_reference is not None:
        report["trimmed_reference"] = {
            "population": "rank-20 POD reference only; generated and experimental signals are not trimmed",
            "window_selection": (
                f"one centered {nperseg}-sample window per eligible contiguous clean run"
            ),
            "window_count": int(len(trimmed_reference["center"])),
            "welch_overlap_within_window": 0,
            "selected_reference_frames": int(selected_mask.size),
            "selected_retained_reference_frames": int(selected_mask.sum()),
            "selected_removed_reference_frame_percent": float(
                100 * (~selected_mask).mean()
            ),
            "exact_training_trim": trim_report,
        }
    arrays: dict[str, np.ndarray] = {}
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 10.5), constrained_layout=True)
    names = ("center", "center_without_spatial_mean", "spatial_mean")
    colors = {
        "generated": "C0",
        "rank20_reference": "C1",
        "rank20_reference_training_trimmed": "C3",
        "experimental": "black",
    }
    labels = {
        "generated": "Neural SDE",
        "rank20_reference": "rank-20 POD reference",
        "rank20_reference_training_trimmed": "rank-20 POD reference (training-trimmed)",
        "experimental": "experimental",
    }

    for axis, name in zip(axes, names):
        frequency, generated_psd, segment, n_overlap = ensemble_psd(
            generated[name], rollout.fs, nperseg, overlap
        )
        _, reference_psd, _, _ = ensemble_psd(reference[name], rollout.fs, nperseg, overlap)
        _, experimental_psd, _, _ = ensemble_psd(experimental[name], rollout.fs, nperseg, overlap)
        values_by_population = {
            "generated": generated_psd.reshape(-1, len(frequency)),
            "rank20_reference": reference_psd.reshape(-1, len(frequency)),
        }
        if trimmed_reference is not None:
            trimmed_frequency, trimmed_psd, _, _ = ensemble_psd(
                trimmed_reference[name], rollout.fs, nperseg, 0.0
            )
            if not np.array_equal(trimmed_frequency, frequency):
                raise ValueError("Trimmed and untrimmed PSD frequency grids disagree.")
            values_by_population["rank20_reference_training_trimmed"] = (
                trimmed_psd.reshape(-1, len(frequency))
            )
        values_by_population["experimental"] = experimental_psd.reshape(
            -1, len(frequency)
        )
        positive = frequency > 0
        signal_report = {}
        for population, values in values_by_population.items():
            average = values.mean(axis=0)
            axis.loglog(
                frequency[positive],
                np.maximum(average[positive], np.finfo(float).tiny),
                color=colors[population],
                linewidth=1.35,
                label=labels[population],
            )
            if (
                len(values) > 1
                and population != "rank20_reference_training_trimmed"
            ):
                lo, hi = np.quantile(values, (0.1, 0.9), axis=0)
                axis.fill_between(
                    frequency[positive],
                    np.maximum(lo[positive], np.finfo(float).tiny),
                    np.maximum(hi[positive], np.finfo(float).tiny),
                    color=colors[population],
                    alpha=0.12,
                )
            power = values.sum(axis=-1) * (frequency[1] - frequency[0])
            signal_report[population] = {
                "mean_integrated_power": float(power.mean()),
            }
            count_key = (
                "clean_window_count"
                if population == "rank20_reference_training_trimmed"
                else "trajectory_count"
            )
            signal_report[population][count_key] = int(len(values))
            arrays[f"{population}_{name}"] = values
        ratio_populations = ["generated", "rank20_reference"]
        if trimmed_reference is not None:
            ratio_populations.append("rank20_reference_training_trimmed")
        for population in ratio_populations:
            signal_report[population]["power_over_experimental"] = float(
                signal_report[population]["mean_integrated_power"]
                / signal_report["experimental"]["mean_integrated_power"]
            )
        report["signals"][name] = signal_report
        axis.set(
            title=name.replace("_", " "),
            xlabel="Frequency [Hz]",
            ylabel=f"PSD [{rollout.units}$^2$/Hz]",
        )
        axis.grid(True, which="both", linestyle="--", alpha=0.25)
        axis.legend(frameon=False)

    report.update(
        nperseg=int(segment),
        noverlap=int(n_overlap),
        frequency_spacing_hz=float(frequency[1] - frequency[0]),
    )
    out = output_directory(rollout, "psd_with_experimental", output, overwrite)
    np.savez_compressed(out / "psd_comparison.npz", frequency_hz=frequency, **arrays)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    figure.savefig(out / "psd_comparison.png", dpi=180)
    plt.close(figure)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--pod-input", type=Path, required=True)
    parser.add_argument("--point", nargs=2, type=int, metavar=("Y", "X"))
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--include-trimmed-reference",
        action="store_true",
        help=(
            "Add a rank-20 reference PSD estimated from one centered contiguous "
            "clean window per run under the checkpoint's exact training-trim rule."
        ),
    )
    args = parser.parse_args(argv)
    rollout = Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard)
    print(
        analyze(
            rollout,
            args.raw_input,
            args.pod_input,
            point=None if args.point is None else tuple(args.point),
            nperseg=args.nperseg,
            overlap=args.overlap,
            include_trimmed_reference=args.include_trimmed_reference,
            output=args.output,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
