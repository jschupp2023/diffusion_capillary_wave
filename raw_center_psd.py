"""Compute a Welch PSD at the geometric center of a raw experiment HDF5 file.

The expected input layout is a ``/main`` group containing one 2-D dataset per
frame and ``/meta/t``, ``/meta/x``, and ``/meta/y`` arrays. Only one scalar is
read from each frame; complete frames are never assembled in memory.

Example
-------
    python raw_center_psd.py experiment_data_roi-none_cal-true.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pod_2d import _frame_key, inspect_input
from pod_center_psd import compute_welch, infer_sampling_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw experimental HDF5 file.")
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        help="Sampling frequency in Hz (default: inferred from /meta/t).",
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
        "--output-dir",
        type=Path,
        help="Output folder (default: <input-directory>/raw_center_psd).",
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
        help="Replace existing outputs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sampling_frequency is not None and args.sampling_frequency <= 0:
        raise ValueError("--sampling-frequency must be positive.")
    if args.nperseg < 2:
        raise ValueError("--nperseg must be at least 2.")
    if not 0 <= args.overlap_fraction < 1:
        raise ValueError("--overlap-fraction must lie in [0, 1).")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")


def read_center_signal(input_path: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    float,
    float,
    dict[str, str],
]:
    """Read the geometric-center value from every keyed frame."""
    info = inspect_input(input_path, start=0, stop=None)
    center_y = info.frame_shape[0] // 2
    center_x = info.frame_shape[1] // 2
    center_signal = np.empty(info.n_frames, dtype=np.float64)
    report_every = max(1, info.n_frames // 10)
    started = time.perf_counter()

    with h5py.File(info.path, "r") as handle:
        frame_group = handle["main"]
        for position, frame_number in enumerate(info.frame_numbers):
            key = _frame_key(frame_number)
            try:
                frame = frame_group[key]
            except KeyError as error:
                raise KeyError(f"Missing HDF5 frame /main/{key}.") from error
            if frame.shape != info.frame_shape:
                raise ValueError(
                    f"Frame /main/{key} has shape {frame.shape}; expected "
                    f"{info.frame_shape}."
                )
            value = float(frame[center_y, center_x])
            if not np.isfinite(value):
                raise ValueError(
                    f"Frame /main/{key} has a nonfinite value at the center."
                )
            center_signal[position] = value
            completed = position + 1
            if completed % report_every == 0 or completed == info.n_frames:
                print(
                    f"Read center point: {completed:,}/{info.n_frames:,} frames "
                    f"({100 * completed / info.n_frames:.0f}%, "
                    f"{time.perf_counter() - started:.1f} s)",
                    flush=True,
                )

    return (
        info.time,
        center_signal,
        center_y,
        center_x,
        float(info.x[center_x]),
        float(info.y[center_y]),
        info.units,
    )


def plot_psd(
    frequency: np.ndarray,
    density: np.ndarray,
    output_path: Path,
    experiment_name: str,
    center_y: int,
    center_x: int,
    x_coordinate: float,
    y_coordinate: float,
    units: dict[str, str],
    sampling_frequency: float,
    nperseg: int,
    noverlap: int,
    dpi: int,
) -> None:
    positive = (frequency > 0) & (density > 0)
    if not np.any(positive):
        raise ValueError("PSD contains no positive-frequency positive-density values.")

    signal_units = units.get("z", "")
    density_units = f"{signal_units}$^2$/Hz" if signal_units else "value$^2$/Hz"
    x_units = units.get("x", "")
    y_units = units.get("y", "")
    x_label = f"{x_coordinate:.3f} {x_units}".strip()
    y_label = f"{y_coordinate:.3f} {y_units}".strip()

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.loglog(
        frequency[positive],
        density[positive],
        color="C0",
        linewidth=1.25,
    )
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title(f"Raw experimental center-point PSD\n{experiment_name}")
    axis.grid(True, which="both", linestyle="--", alpha=0.32)
    details = (
        f"index (y, x) = ({center_y}, {center_x})\n"
        f"coordinate = ({x_label}, {y_label})\n"
        f"$f_s$ = {sampling_frequency:,.3f} Hz\n"
        f"Welch: Hann, nperseg = {nperseg:,}, overlap = {noverlap:,}\n"
        "raw calibrated values; no spatial-mean subtraction"
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


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    validate_args(args)
    started = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / "raw_center_psd"
    )
    image_path = output_directory / f"{input_path.stem}_center_psd.png"
    data_path = output_directory / f"{input_path.stem}_center_psd.npz"
    existing = [path for path in (image_path, data_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    (
        time_values,
        center_signal,
        center_y,
        center_x,
        x_coordinate,
        y_coordinate,
        units,
    ) = read_center_signal(input_path)
    inferred_frequency, relative_jitter = infer_sampling_frequency(time_values)
    sampling_frequency = (
        inferred_frequency
        if args.sampling_frequency is None
        else args.sampling_frequency
    )
    print(
        f"Center: (y, x) = ({center_y}, {center_x}); "
        f"sampling frequency: {sampling_frequency:,.6f} Hz; "
        f"time-grid maximum relative step variation: {relative_jitter:.3%}.",
        flush=True,
    )
    result, nperseg, noverlap = compute_welch(
        center_signal,
        sampling_frequency,
        args.nperseg,
        args.overlap_fraction,
    )
    np.savez_compressed(
        data_path,
        time=time_values,
        center_signal=center_signal,
        frequency_hz=result.frequency,
        power_spectral_density=result.density,
        sampling_frequency_hz=np.float64(sampling_frequency),
        nperseg=np.int64(nperseg),
        noverlap=np.int64(noverlap),
        center_y_index=np.int64(center_y),
        center_x_index=np.int64(center_x),
        center_x_coordinate=np.float64(x_coordinate),
        center_y_coordinate=np.float64(y_coordinate),
        signal_units=np.asarray(units.get("z", "")),
        source_file=np.asarray(str(input_path)),
    )
    plot_psd(
        result.frequency,
        result.density,
        image_path,
        input_path.stem,
        center_y,
        center_x,
        x_coordinate,
        y_coordinate,
        units,
        sampling_frequency,
        nperseg,
        noverlap,
        args.dpi,
    )
    print(f"Saved {image_path.resolve()}")
    print(f"Saved {data_path.resolve()}")
    print(f"Finished in {time.perf_counter() - started:.2f} s.")
    return image_path.resolve(), data_path.resolve()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
