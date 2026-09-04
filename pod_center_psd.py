"""Compute Welch PSDs at one point from a saved 2-D POD result.

Only the selected spatial point is reconstructed,

    z_preprocessed(t) = coefficients(t, :) @ modes(:, y, x),

so no full 200 x 200 frames are formed. Four plots are saved: the
directly reprojected preprocessed signal, that signal after restoring the saved
instantaneous spatial mean, the spatial mean by itself, and an overlay comparing
the latter two PSDs. The temporal mean
field is a time-independent offset and is not restored; SciPy Welch detrends
each segment by its mean, so that constant would not affect either point PSD.

Examples
--------
Use the geometric center of a repetition folder::

    python pod_center_psd.py /path/to/Ca_ac_..._rep1

Use flattened C-order index 20000 exactly::

    python pod_center_psd.py /path/to/Ca_ac_..._rep1 --flat-index 20000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class PointSelection:
    y_index: int
    x_index: int
    flat_index: int
    x_coordinate: float
    y_coordinate: float
    x_units: str
    y_units: str


@dataclass(frozen=True)
class WelchResult:
    frequency: np.ndarray
    density: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Reduced-data repetition folder or a POD HDF5 file.",
    )
    parser.add_argument(
        "--pod-rank",
        type=int,
        default=1_000,
        help="Rank in the POD filename when input is a folder (default: 1000).",
    )
    parser.add_argument(
        "--reconstruction-rank",
        type=int,
        help="Modes used for reconstruction (default: all stored modes).",
    )
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--point",
        type=int,
        nargs=2,
        metavar=("Y_INDEX", "X_INDEX"),
        help="2-D array indices (default: geometric center).",
    )
    location.add_argument(
        "--flat-index",
        type=int,
        help="C-order flattened spatial index, for example 20000.",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        help="Sampling frequency in Hz (default: inferred from grid/time).",
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
        help="Fractional overlap between Welch segments (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8_192,
        help="Coefficient rows projected per batch (default: 8192).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: <repetition-folder>/pod_psd).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Plot resolution (default: 200 dpi).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PSD plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pod_rank < 1:
        raise ValueError("--pod-rank must be positive.")
    if args.reconstruction_rank is not None and args.reconstruction_rank < 1:
        raise ValueError("--reconstruction-rank must be positive.")
    if args.flat_index is not None and args.flat_index < 0:
        raise ValueError("--flat-index must be non-negative.")
    if args.sampling_frequency is not None and args.sampling_frequency <= 0:
        raise ValueError("--sampling-frequency must be positive.")
    if args.nperseg < 2:
        raise ValueError("--nperseg must be at least 2.")
    if not 0 <= args.overlap_fraction < 1:
        raise ValueError("--overlap-fraction must lie in [0, 1).")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")


def resolve_pod_file(input_path: Path, pod_rank: int) -> Path:
    input_path = input_path.expanduser()
    if input_path.is_file():
        return input_path.resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    expected = input_path / f"pod_2d_r{pod_rank}.h5"
    if not expected.is_file():
        raise FileNotFoundError(
            f"Expected rank-{pod_rank} POD file does not exist: {expected}"
        )
    return expected.resolve()


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def select_point(
    mode_shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    x_units: str,
    y_units: str,
    point: tuple[int, int] | list[int] | None,
    flat_index: int | None,
) -> PointSelection:
    height, width = mode_shape
    if flat_index is not None:
        if flat_index >= height * width:
            raise IndexError(
                f"Flat index {flat_index:,} is outside a {height} x {width} "
                f"field containing {height * width:,} points."
            )
        y_index, x_index = np.unravel_index(flat_index, mode_shape, order="C")
    elif point is not None:
        y_index, x_index = int(point[0]), int(point[1])
    else:
        y_index, x_index = height // 2, width // 2

    if not 0 <= y_index < height or not 0 <= x_index < width:
        raise IndexError(
            f"Point ({y_index}, {x_index}) is outside field shape {mode_shape}."
        )
    resolved_flat_index = int(np.ravel_multi_index((y_index, x_index), mode_shape))
    return PointSelection(
        y_index=int(y_index),
        x_index=int(x_index),
        flat_index=resolved_flat_index,
        x_coordinate=float(x[x_index]),
        y_coordinate=float(y[y_index]),
        x_units=x_units,
        y_units=y_units,
    )


def infer_sampling_frequency(time_values: np.ndarray) -> tuple[float, float]:
    if time_values.ndim != 1 or len(time_values) < 2:
        raise ValueError("grid/time must contain at least two values.")
    differences = np.diff(time_values)
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        raise ValueError("grid/time must be finite and strictly increasing.")
    mean_step = float((time_values[-1] - time_values[0]) / (len(time_values) - 1))
    sampling_frequency = 1.0 / mean_step
    maximum_relative_jitter = float(
        np.max(np.abs(differences - mean_step)) / mean_step
    )
    return sampling_frequency, maximum_relative_jitter


def reconstruct_point_signals(
    pod_path: Path,
    reconstruction_rank: int | None,
    point: tuple[int, int] | list[int] | None,
    flat_index: int | None,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PointSelection,
    int,
    str,
]:
    """Reconstruct one point and return signals with/without spatial mean."""
    started = time.perf_counter()
    with h5py.File(pod_path, "r") as handle:
        required = (
            "pod/modes",
            "reduced/coefficients",
            "preprocessing/frame_spatial_mean",
            "grid/time",
            "grid/x",
            "grid/y",
        )
        for dataset_name in required:
            if dataset_name not in handle:
                raise KeyError(f"Missing dataset {dataset_name!r} in {pod_path}.")

        modes = handle["pod/modes"]
        coefficients = handle["reduced/coefficients"]
        spatial_mean_dataset = handle["preprocessing/frame_spatial_mean"]
        spatial_mean_was_removed = bool(
            spatial_mean_dataset.attrs.get(
                "subtracted_before_pod",
                handle.attrs.get("instantaneous_spatial_mean_removed", False),
            )
        )
        if not spatial_mean_was_removed:
            raise ValueError(
                "The instantaneous spatial mean was not removed before this POD; "
                "adding frame_spatial_mean would double-count it."
            )
        time_values = np.asarray(handle["grid/time"], dtype=np.float64)
        x_dataset = handle["grid/x"]
        y_dataset = handle["grid/y"]
        x = np.asarray(x_dataset, dtype=np.float64)
        y = np.asarray(y_dataset, dtype=np.float64)

        if modes.ndim != 3:
            raise ValueError(f"pod/modes must be three-dimensional; got {modes.shape}.")
        if coefficients.ndim != 2:
            raise ValueError(
                f"reduced/coefficients must be two-dimensional; got "
                f"{coefficients.shape}."
            )
        if coefficients.shape[1] != len(modes):
            raise ValueError(
                f"Stored mode and coefficient counts differ: {len(modes)} and "
                f"{coefficients.shape[1]}."
            )
        if not (
            len(coefficients) == len(time_values) == len(spatial_mean_dataset)
        ):
            raise ValueError(
                "Coefficient, time, and spatial-mean lengths must be identical."
            )
        if modes.shape[1:] != (len(y), len(x)):
            raise ValueError(
                f"Mode shape {modes.shape[1:]} does not match y/x grid "
                f"{(len(y), len(x))}."
            )

        rank = len(modes) if reconstruction_rank is None else reconstruction_rank
        if rank > len(modes):
            raise ValueError(
                f"Requested reconstruction rank {rank}, but only {len(modes)} "
                "modes are stored."
            )
        selected = select_point(
            modes.shape[1:],
            x,
            y,
            _attribute_text(x_dataset.attrs.get("units")),
            _attribute_text(y_dataset.attrs.get("units")),
            point,
            flat_index,
        )
        point_modes = np.asarray(
            modes[:rank, selected.y_index, selected.x_index], dtype=np.float64
        )
        if not np.isfinite(point_modes).all():
            raise ValueError("Selected POD mode values contain nonfinite values.")

        preprocessed_signal = np.empty(len(coefficients), dtype=np.float64)
        report_every = max(1, (len(coefficients) + batch_size - 1) // batch_size // 10)
        for batch_number, start in enumerate(
            range(0, len(coefficients), batch_size), start=1
        ):
            stop = min(start + batch_size, len(coefficients))
            coefficient_block = np.asarray(
                coefficients[start:stop, :rank], dtype=np.float64
            )
            preprocessed_signal[start:stop] = coefficient_block @ point_modes
            if batch_number % report_every == 0 or stop == len(coefficients):
                print(
                    f"Reconstructed {stop:,}/{len(coefficients):,} samples",
                    flush=True,
                )

        spatial_mean = np.asarray(spatial_mean_dataset, dtype=np.float64)
        signal_units = _attribute_text(coefficients.attrs.get("units"))

    if not np.isfinite(preprocessed_signal).all() or not np.isfinite(spatial_mean).all():
        raise ValueError("Reconstructed signal or spatial mean contains nonfinite values.")
    with_spatial_mean = preprocessed_signal + spatial_mean
    print(
        f"Point reconstruction completed in {time.perf_counter() - started:.2f} s.",
        flush=True,
    )
    return (
        time_values,
        preprocessed_signal,
        spatial_mean,
        with_spatial_mean,
        selected,
        rank,
        signal_units,
    )


def compute_welch(
    signal: np.ndarray,
    sampling_frequency: float,
    nperseg: int,
    overlap_fraction: float,
) -> tuple[WelchResult, int, int]:
    actual_nperseg = min(nperseg, len(signal))
    noverlap = int(round(overlap_fraction * actual_nperseg))
    noverlap = min(noverlap, actual_nperseg - 1)
    frequency, density = welch(
        signal,
        fs=sampling_frequency,
        window="hann",
        nperseg=actual_nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
    )
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("Welch PSD contains invalid values.")
    return WelchResult(frequency, density), actual_nperseg, noverlap


def _coordinate_label(value: float, units: str) -> str:
    return f"{value:.3f} {units}" if units else f"{value:.3f}"


def plot_psd(
    result: WelchResult,
    output_path: Path,
    title: str,
    point: PointSelection,
    rank: int,
    signal_units: str,
    sampling_frequency: float,
    nperseg: int,
    noverlap: int,
    dpi: int,
) -> None:
    positive = (result.frequency > 0) & (result.density > 0)
    if not np.any(positive):
        raise ValueError("PSD contains no positive-frequency positive-density values.")

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.loglog(
        result.frequency[positive],
        result.density[positive],
        color="C0",
        linewidth=1.25,
    )
    density_units = f"{signal_units}$^2$/Hz" if signal_units else "value$^2$/Hz"
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title(title)
    axis.grid(True, which="both", linestyle="--", alpha=0.32)
    point_description = (
        f"index (y, x) = ({point.y_index}, {point.x_index}), "
        f"flat = {point.flat_index:,}\n"
        f"coordinate = ({_coordinate_label(point.x_coordinate, point.x_units)}, "
        f"{_coordinate_label(point.y_coordinate, point.y_units)})\n"
        f"rank = {rank:,}, $f_s$ = {sampling_frequency:,.3f} Hz\n"
        f"Welch: Hann, nperseg = {nperseg:,}, overlap = {noverlap:,}"
    )
    axis.text(
        0.02,
        0.03,
        point_description,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_spatial_mean_psd(
    result: WelchResult,
    output_path: Path,
    experiment_name: str,
    signal_units: str,
    sampling_frequency: float,
    nperseg: int,
    noverlap: int,
    dpi: int,
) -> None:
    positive = (result.frequency > 0) & (result.density > 0)
    if not np.any(positive):
        raise ValueError("PSD contains no positive-frequency positive-density values.")

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.loglog(
        result.frequency[positive],
        result.density[positive],
        color="C1",
        linewidth=1.25,
    )
    density_units = f"{signal_units}$^2$/Hz" if signal_units else "value$^2$/Hz"
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title(f"PSD of instantaneous spatial mean only\n{experiment_name}")
    axis.grid(True, which="both", linestyle="--", alpha=0.32)
    axis.text(
        0.02,
        0.03,
        "No reduced POD dynamics included\n"
        f"$f_s$ = {sampling_frequency:,.3f} Hz\n"
        f"Welch: Hann, nperseg = {nperseg:,}, overlap = {noverlap:,}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_restored_and_spatial_mean_psds(
    restored: WelchResult,
    spatial_mean: WelchResult,
    output_path: Path,
    experiment_name: str,
    point: PointSelection,
    rank: int,
    signal_units: str,
    sampling_frequency: float,
    nperseg: int,
    noverlap: int,
    dpi: int,
) -> None:
    """Overlay the restored center-point and spatial-mean PSDs."""
    if not np.array_equal(restored.frequency, spatial_mean.frequency):
        raise ValueError("Compared PSDs do not have identical frequency bins.")
    restored_positive = (restored.frequency > 0) & (restored.density > 0)
    mean_positive = (spatial_mean.frequency > 0) & (spatial_mean.density > 0)
    if not np.any(restored_positive) or not np.any(mean_positive):
        raise ValueError(
            "Both compared PSDs must contain positive-frequency, positive-density "
            "values."
        )

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.loglog(
        restored.frequency[restored_positive],
        restored.density[restored_positive],
        color="C0",
        linewidth=1.3,
        label="center reconstruction + spatial mean",
    )
    axis.loglog(
        spatial_mean.frequency[mean_positive],
        spatial_mean.density[mean_positive],
        color="C1",
        linewidth=1.3,
        label="spatial mean only",
    )
    density_units = f"{signal_units}$^2$/Hz" if signal_units else "value$^2$/Hz"
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title(
        "Center-point PSD with spatial mean versus spatial-mean PSD\n"
        f"{experiment_name}"
    )
    axis.grid(True, which="both", linestyle="--", alpha=0.32)
    axis.legend(loc="best", frameon=False)
    details = (
        f"index (y, x) = ({point.y_index}, {point.x_index}), "
        f"flat = {point.flat_index:,}\n"
        f"rank = {rank:,}, $f_s$ = {sampling_frequency:,.3f} Hz\n"
        f"Welch: Hann, nperseg = {nperseg:,}, overlap = {noverlap:,}"
    )
    axis.text(
        0.02,
        0.03,
        details,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    validate_args(args)
    pod_path = resolve_pod_file(args.input, args.pod_rank)
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pod_path.parent / "pod_psd"
    )
    preprocessed_path = output_directory / "center_psd_preprocessed.png"
    restored_path = output_directory / "center_psd_with_spatial_mean.png"
    spatial_mean_path = output_directory / "spatial_mean_psd.png"
    comparison_path = output_directory / "center_psd_vs_spatial_mean_psd.png"
    existing = [
        path
        for path in (
            preprocessed_path,
            restored_path,
            spatial_mean_path,
            comparison_path,
        )
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    (
        time_values,
        preprocessed_signal,
        spatial_mean,
        restored_signal,
        selected,
        reconstruction_rank,
        signal_units,
    ) = reconstruct_point_signals(
        pod_path,
        args.reconstruction_rank,
        args.point,
        args.flat_index,
        args.batch_size,
    )
    inferred_frequency, relative_jitter = infer_sampling_frequency(time_values)
    sampling_frequency = (
        inferred_frequency
        if args.sampling_frequency is None
        else args.sampling_frequency
    )
    print(
        f"Selected point: (y, x) = ({selected.y_index}, {selected.x_index}), "
        f"flat index {selected.flat_index:,}.\n"
        f"Sampling frequency: {sampling_frequency:,.6f} Hz "
        f"(time-grid maximum relative step variation {relative_jitter:.3%}).",
        flush=True,
    )

    preprocessed_psd, nperseg, noverlap = compute_welch(
        preprocessed_signal,
        sampling_frequency,
        args.nperseg,
        args.overlap_fraction,
    )
    restored_psd, restored_nperseg, restored_noverlap = compute_welch(
        restored_signal,
        sampling_frequency,
        args.nperseg,
        args.overlap_fraction,
    )
    assert (restored_nperseg, restored_noverlap) == (nperseg, noverlap)
    spatial_mean_psd, mean_nperseg, mean_noverlap = compute_welch(
        spatial_mean,
        sampling_frequency,
        args.nperseg,
        args.overlap_fraction,
    )
    assert (mean_nperseg, mean_noverlap) == (nperseg, noverlap)

    experiment_name = pod_path.parent.name
    plot_psd(
        preprocessed_psd,
        preprocessed_path,
        f"Center-point PSD — preprocessed POD reconstruction\n{experiment_name}",
        selected,
        reconstruction_rank,
        signal_units,
        sampling_frequency,
        nperseg,
        noverlap,
        args.dpi,
    )
    plot_psd(
        restored_psd,
        restored_path,
        f"Center-point PSD — spatial mean restored\n{experiment_name}",
        selected,
        reconstruction_rank,
        signal_units,
        sampling_frequency,
        nperseg,
        noverlap,
        args.dpi,
    )
    plot_spatial_mean_psd(
        spatial_mean_psd,
        spatial_mean_path,
        experiment_name,
        signal_units,
        sampling_frequency,
        nperseg,
        noverlap,
        args.dpi,
    )
    plot_restored_and_spatial_mean_psds(
        restored_psd,
        spatial_mean_psd,
        comparison_path,
        experiment_name,
        selected,
        reconstruction_rank,
        signal_units,
        sampling_frequency,
        nperseg,
        noverlap,
        args.dpi,
    )
    print(f"Saved {preprocessed_path.resolve()}")
    print(f"Saved {restored_path.resolve()}")
    print(f"Saved {spatial_mean_path.resolve()}")
    print(f"Saved {comparison_path.resolve()}")
    return (
        preprocessed_path.resolve(),
        restored_path.resolve(),
        spatial_mean_path.resolve(),
        comparison_path.resolve(),
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
