"""Compute an efficient wavenumber-frequency spectrum from a 2-D POD result.

The script reconstructs only the horizontal and/or vertical line through the
geometric center of the field. It then computes windowed space-time Fourier
transforms in overlapping time segments and averages their power, analogous to
Welch's method. This avoids reconstructing the full 100001 x 200 x 200 data
cube.

By default, spectra from the horizontal and vertical center lines are averaged,
all stored POD modes are used, and the removed instantaneous spatial mean is
not restored.

Example
-------
    python pod_space_time_spectrum.py /path/to/Ca_ac_..._rep1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.signal.windows import get_window

from data_analysis.psd.pod_center_psd import infer_sampling_frequency, resolve_pod_file


@dataclass(frozen=True)
class ReconstructedLines:
    values: np.ndarray
    directions: tuple[str, ...]
    spatial_grid: np.ndarray
    spatial_units: str
    signal_units: str
    time: np.ndarray
    sampling_frequency: float
    time_step_jitter: float
    reconstruction_rank: int
    center_y: int
    center_x: int


@dataclass(frozen=True)
class Spectrum:
    frequency: np.ndarray
    wavenumber: np.ndarray
    power_density: np.ndarray
    segment_length: int
    overlap: int
    segment_count: int


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
        help="Number of POD modes to reconstruct (default: all stored modes).",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "both"),
        default="both",
        help="Center-line direction(s) to analyze (default: both).",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=8_192,
        help="Samples per temporal Fourier segment (default: 8192).",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=0.5,
        help="Fractional overlap of time segments (default: 0.5).",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        help="Sampling frequency in Hz (default: inferred from grid/time).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8_192,
        help="Coefficient rows reconstructed per matrix multiply (default: 8192).",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=1,
        help="Worker threads used by SciPy FFT operations (default: 1).",
    )
    parser.add_argument(
        "--add-spatial-mean",
        action="store_true",
        help="Restore frame_spatial_mean before computing the spectrum.",
    )
    parser.add_argument(
        "--db-range",
        type=float,
        default=60.0,
        help="Displayed dynamic range below the maximum (default: 60 dB).",
    )
    parser.add_argument(
        "--max-frequency-khz",
        type=float,
        help="Maximum displayed frequency in kHz (default: Nyquist).",
    )
    parser.add_argument(
        "--max-wavenumber",
        type=float,
        help="Maximum displayed absolute wavenumber in rad/mm (default: Nyquist).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: <repetition-folder>/pod_space_time).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Plot resolution (default: 220 dpi).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing spectrum outputs.",
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
    if args.sampling_frequency is not None and args.sampling_frequency <= 0:
        raise ValueError("--sampling-frequency must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")
    if args.db_range <= 0:
        raise ValueError("--db-range must be positive.")
    if args.max_frequency_khz is not None and args.max_frequency_khz <= 0:
        raise ValueError("--max-frequency-khz must be positive.")
    if args.max_wavenumber is not None and args.max_wavenumber <= 0:
        raise ValueError("--max-wavenumber must be positive.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _uniform_spacing(grid: np.ndarray, name: str) -> float:
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError(f"{name} grid must be one-dimensional with at least 2 points.")
    differences = np.diff(grid)
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        raise ValueError(f"{name} grid must be finite and strictly increasing.")
    spacing = float(np.mean(differences))
    relative_variation = float(np.max(np.abs(differences - spacing)) / spacing)
    # Coordinate arrays are stored as float32 in some POD files, so an
    # analytically uniform linspace can acquire small step-to-step roundoff.
    # Variations below 1% are negligible at the FFT-bin resolution here.
    if relative_variation > 1e-2:
        raise ValueError(
            f"{name} grid is not sufficiently uniform; relative spacing variation "
            f"is {relative_variation:.3e}."
        )
    if relative_variation > 1e-3:
        warnings.warn(
            f"{name} grid spacing varies by {relative_variation:.3%}; using its "
            "mean spacing for the FFT.",
            stacklevel=2,
        )
    return spacing


def reconstruct_center_lines(
    pod_path: Path,
    axis: str,
    reconstruction_rank: int | None,
    sampling_frequency_override: float | None,
    batch_size: int,
    add_spatial_mean: bool,
) -> ReconstructedLines:
    """Reconstruct center lines directly from coefficients and mode slices."""
    started = time.perf_counter()
    with h5py.File(pod_path, "r") as handle:
        required = (
            "pod/modes",
            "reduced/coefficients",
            "grid/time",
            "grid/x",
            "grid/y",
            "preprocessing/frame_spatial_mean",
        )
        for dataset_name in required:
            if dataset_name not in handle:
                raise KeyError(f"Missing dataset {dataset_name!r} in {pod_path}.")

        mode_dataset = handle["pod/modes"]
        coefficient_dataset = handle["reduced/coefficients"]
        time_values = np.asarray(handle["grid/time"], dtype=np.float64)
        x_dataset = handle["grid/x"]
        y_dataset = handle["grid/y"]
        x = np.asarray(x_dataset, dtype=np.float64)
        y = np.asarray(y_dataset, dtype=np.float64)
        spatial_mean_dataset = handle["preprocessing/frame_spatial_mean"]

        if mode_dataset.ndim != 3:
            raise ValueError(
                f"pod/modes must have shape (mode, y, x); got {mode_dataset.shape}."
            )
        if coefficient_dataset.ndim != 2:
            raise ValueError(
                "reduced/coefficients must have shape (time, mode); got "
                f"{coefficient_dataset.shape}."
            )
        if coefficient_dataset.shape[1] != len(mode_dataset):
            raise ValueError("POD mode and coefficient counts do not match.")
        if not (
            len(coefficient_dataset) == len(time_values) == len(spatial_mean_dataset)
        ):
            raise ValueError(
                "Coefficient, time, and spatial-mean lengths must be identical."
            )
        if mode_dataset.shape[1:] != (len(y), len(x)):
            raise ValueError("Mode shape does not match the saved spatial grids.")

        rank = len(mode_dataset) if reconstruction_rank is None else reconstruction_rank
        if rank > len(mode_dataset):
            raise ValueError(
                f"Requested reconstruction rank {rank}, but only {len(mode_dataset)} "
                "modes are stored."
            )

        center_y = len(y) // 2
        center_x = len(x) // 2
        directions = ("x", "y") if axis == "both" else (axis,)
        if axis == "both":
            x_spacing = _uniform_spacing(x, "x")
            y_spacing = _uniform_spacing(y, "y")
            if len(x) != len(y) or not np.isclose(
                x_spacing, y_spacing, rtol=1e-5, atol=1e-12
            ):
                raise ValueError(
                    "Averaging x and y spectra requires equal grid lengths and spacing."
                )
            spatial_grid = x
            spatial_units = _attribute_text(x_dataset.attrs.get("units"))
        elif axis == "x":
            _uniform_spacing(x, "x")
            spatial_grid = x
            spatial_units = _attribute_text(x_dataset.attrs.get("units"))
        else:
            _uniform_spacing(y, "y")
            spatial_grid = y
            spatial_units = _attribute_text(y_dataset.attrs.get("units"))

        print(
            f"Loading {rank:,} modes and extracting center "
            f"line(s): {', '.join(directions)}...",
            flush=True,
        )
        mode_array = np.asarray(mode_dataset[:rank], dtype=np.float32)
        line_mode_blocks: list[np.ndarray] = []
        for direction in directions:
            if direction == "x":
                line_mode_blocks.append(mode_array[:, center_y, :])
            else:
                line_mode_blocks.append(mode_array[:, :, center_x])
        packed_line_modes = np.ascontiguousarray(
            np.concatenate(line_mode_blocks, axis=1), dtype=np.float32
        )
        del mode_array, line_mode_blocks

        n_directions = len(directions)
        n_space = len(spatial_grid)
        reconstructed = np.empty(
            (n_directions, len(coefficient_dataset), n_space), dtype=np.float32
        )
        n_batches = (len(coefficient_dataset) + batch_size - 1) // batch_size
        report_every = max(1, n_batches // 10)
        for batch_number, start in enumerate(
            range(0, len(coefficient_dataset), batch_size), start=1
        ):
            stop = min(start + batch_size, len(coefficient_dataset))
            coefficient_block = np.asarray(
                coefficient_dataset[start:stop, :rank], dtype=np.float32
            )
            projected = coefficient_block @ packed_line_modes
            reconstructed[:, start:stop, :] = projected.reshape(
                stop - start, n_directions, n_space
            ).transpose(1, 0, 2)
            if (
                batch_number == 1
                or batch_number % report_every == 0
                or batch_number == n_batches
            ):
                print(
                    f"Reconstructed {stop:,}/{len(coefficient_dataset):,} samples",
                    flush=True,
                )

        if add_spatial_mean:
            spatial_mean_was_removed = bool(
                spatial_mean_dataset.attrs.get(
                    "subtracted_before_pod",
                    handle.attrs.get("instantaneous_spatial_mean_removed", False),
                )
            )
            if not spatial_mean_was_removed:
                raise ValueError(
                    "Cannot add frame_spatial_mean: it was not removed before POD."
                )
            spatial_mean = np.asarray(spatial_mean_dataset, dtype=np.float32)
            reconstructed += spatial_mean[None, :, None]

        signal_units = _attribute_text(coefficient_dataset.attrs.get("units"))

    inferred_frequency, jitter = infer_sampling_frequency(time_values)
    sampling_frequency = (
        inferred_frequency
        if sampling_frequency_override is None
        else sampling_frequency_override
    )
    if not np.isfinite(reconstructed).all():
        raise ValueError("Reconstructed center lines contain nonfinite values.")
    print(
        f"Center-line reconstruction completed in "
        f"{time.perf_counter() - started:.2f} s.",
        flush=True,
    )
    return ReconstructedLines(
        values=reconstructed,
        directions=directions,
        spatial_grid=spatial_grid,
        spatial_units=spatial_units,
        signal_units=signal_units,
        time=time_values,
        sampling_frequency=sampling_frequency,
        time_step_jitter=jitter,
        reconstruction_rank=rank,
        center_y=center_y,
        center_x=center_x,
    )


def compute_spectrum(
    lines: ReconstructedLines,
    nperseg: int,
    overlap_fraction: float,
    fft_workers: int,
) -> Spectrum:
    """Compute an averaged, two-sided-k and one-sided-frequency spectrum."""
    n_time = lines.values.shape[1]
    segment_length = min(nperseg, n_time)
    overlap = min(
        int(round(overlap_fraction * segment_length)), segment_length - 1
    )
    step = segment_length - overlap
    starts = list(range(0, n_time - segment_length + 1, step))
    if not starts:
        raise ValueError("No complete time segment is available for the FFT.")

    spatial_spacing = _uniform_spacing(lines.spatial_grid, "spatial")
    spatial_sampling_frequency = 1.0 / spatial_spacing
    time_window = get_window("hann", segment_length, fftbins=True).astype(np.float32)
    space_window = get_window(
        "hann", lines.values.shape[2], fftbins=True
    ).astype(np.float32)
    frequencies = fft.rfftfreq(segment_length, d=1.0 / lines.sampling_frequency)
    spatial_cycles = fft.fftshift(
        fft.fftfreq(lines.values.shape[2], d=spatial_spacing)
    )

    # The saved spatial coordinates are in microns for the current data. Convert
    # cycles/micron to angular wavenumber in rad/mm. Refuse to silently apply
    # that conversion to an unknown unit.
    normalized_units = lines.spatial_units.strip().lower().replace("µ", "u")
    if normalized_units not in ("micron", "microns", "um"):
        raise ValueError(
            "Expected the spatial grid unit to be microns for rad/mm conversion; "
            f"found {lines.spatial_units!r}."
        )
    wavenumbers = 2.0 * np.pi * 1_000.0 * spatial_cycles

    power = np.zeros(
        (len(frequencies), lines.values.shape[2]), dtype=np.float64
    )
    temporal_window_energy = float(np.sum(time_window.astype(np.float64) ** 2))
    spatial_window_energy = float(np.sum(space_window.astype(np.float64) ** 2))
    density_scale = 1.0 / (
        lines.sampling_frequency
        * spatial_sampling_frequency
        * temporal_window_energy
        * spatial_window_energy
    )
    # Convert density per (cycles/micron) to density per (rad/mm).
    density_scale /= 2.0 * np.pi * 1_000.0

    total_transforms = len(lines.directions) * len(starts)
    completed = 0
    report_every = max(1, total_transforms // 10)
    started = time.perf_counter()
    for direction_index in range(len(lines.directions)):
        for start in starts:
            segment = lines.values[
                direction_index, start : start + segment_length
            ].copy()
            segment -= segment.mean(axis=0, keepdims=True)
            segment *= time_window[:, None]
            segment *= space_window[None, :]
            transformed = fft.rfft(
                segment,
                axis=0,
                workers=fft_workers,
                overwrite_x=True,
            )
            transformed = fft.fft(
                transformed,
                axis=1,
                workers=fft_workers,
                overwrite_x=True,
            )
            transformed = fft.fftshift(transformed, axes=1)
            power += transformed.real**2 + transformed.imag**2
            completed += 1
            if (
                completed == 1
                or completed % report_every == 0
                or completed == total_transforms
            ):
                print(
                    f"Space-time FFT {completed}/{total_transforms}",
                    flush=True,
                )

    power *= density_scale / total_transforms
    if segment_length % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    print(
        f"Spectrum computed from {len(starts)} time segments and "
        f"{len(lines.directions)} direction(s) in "
        f"{time.perf_counter() - started:.2f} s.",
        flush=True,
    )
    return Spectrum(
        frequency=frequencies,
        wavenumber=wavenumbers,
        power_density=power,
        segment_length=segment_length,
        overlap=overlap,
        segment_count=len(starts),
    )


def save_numerical_spectrum(
    spectrum: Spectrum,
    lines: ReconstructedLines,
    output_path: Path,
    pod_path: Path,
    spatial_mean_added: bool,
) -> None:
    np.savez_compressed(
        output_path,
        frequency_hz=spectrum.frequency,
        wavenumber_rad_per_mm=spectrum.wavenumber,
        power_density=spectrum.power_density.astype(np.float32),
        directions=np.asarray(lines.directions),
        reconstruction_rank=np.int64(lines.reconstruction_rank),
        sampling_frequency_hz=np.float64(lines.sampling_frequency),
        segment_length=np.int64(spectrum.segment_length),
        overlap=np.int64(spectrum.overlap),
        segment_count=np.int64(spectrum.segment_count),
        center_y_index=np.int64(lines.center_y),
        center_x_index=np.int64(lines.center_x),
        spatial_mean_added=np.bool_(spatial_mean_added),
        source_pod_file=np.asarray(str(pod_path.resolve())),
    )


def plot_spectrum(
    spectrum: Spectrum,
    lines: ReconstructedLines,
    output_path: Path,
    experiment_name: str,
    db_range: float,
    max_frequency_khz: float | None,
    max_wavenumber: float | None,
    spatial_mean_added: bool,
    dpi: int,
) -> None:
    positive_power = spectrum.power_density[spectrum.power_density > 0]
    if not len(positive_power):
        raise ValueError("The spectrum has no positive power values.")
    maximum_power = float(np.max(positive_power))
    floor = maximum_power * 10.0 ** (-db_range / 10.0)
    power_db = 10.0 * np.log10(
        np.maximum(spectrum.power_density, floor) / maximum_power
    )

    frequency_khz = spectrum.frequency / 1_000.0
    frequency_mask = np.ones(len(frequency_khz), dtype=bool)
    if max_frequency_khz is not None:
        frequency_mask &= frequency_khz <= max_frequency_khz
    wavenumber_mask = np.ones(len(spectrum.wavenumber), dtype=bool)
    if max_wavenumber is not None:
        wavenumber_mask &= np.abs(spectrum.wavenumber) <= max_wavenumber
    if not np.any(frequency_mask) or not np.any(wavenumber_mask):
        raise ValueError("Requested plot limits exclude every spectral bin.")

    displayed = power_db[np.ix_(frequency_mask, wavenumber_mask)]
    displayed_frequency = frequency_khz[frequency_mask]
    displayed_wavenumber = spectrum.wavenumber[wavenumber_mask]
    figure, axis = plt.subplots(figsize=(8.4, 5.8))
    image = axis.imshow(
        displayed,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            float(displayed_wavenumber[0]),
            float(displayed_wavenumber[-1]),
            float(displayed_frequency[0]),
            float(displayed_frequency[-1]),
        ),
        cmap="magma",
        vmin=-db_range,
        vmax=0,
    )
    axis.axvline(0, color="white", linewidth=0.6, alpha=0.45)
    axis.set_xlabel(r"Angular wavenumber $k$ [rad/mm]")
    axis.set_ylabel("Frequency [kHz]")
    direction_label = "+".join(lines.directions)
    mean_label = "spatial mean restored" if spatial_mean_added else "spatial mean removed"
    axis.set_title(
        f"Center-line wavenumber–frequency spectrum ({direction_label} average)\n"
        f"{experiment_name} — {mean_label}"
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Power relative to maximum [dB]")
    details = (
        f"rank {lines.reconstruction_rank:,}; "
        f"$f_s$={lines.sampling_frequency:,.1f}$ Hz\n"
        f"{spectrum.segment_count} segments; "
        f"nperseg={spectrum.segment_length:,}; "
        f"overlap={spectrum.overlap:,}"
    )
    axis.text(
        0.015,
        0.02,
        details,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "edgecolor": "white", "alpha": 0.55},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    validate_args(args)
    started = time.perf_counter()
    pod_path = resolve_pod_file(args.input, args.pod_rank)
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pod_path.parent / "pod_space_time"
    )
    stem = f"center_{args.axis}_k_frequency_spectrum"
    if args.add_spatial_mean:
        stem += "_with_spatial_mean"
    image_path = output_directory / f"{stem}.png"
    data_path = output_directory / f"{stem}.npz"
    existing = [path for path in (image_path, data_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    lines = reconstruct_center_lines(
        pod_path,
        args.axis,
        args.reconstruction_rank,
        args.sampling_frequency,
        args.batch_size,
        args.add_spatial_mean,
    )
    print(
        f"Sampling frequency: {lines.sampling_frequency:,.6f} Hz; "
        f"time-grid maximum relative step variation: "
        f"{lines.time_step_jitter:.3%}.",
        flush=True,
    )
    spectrum = compute_spectrum(
        lines,
        args.nperseg,
        args.overlap_fraction,
        args.fft_workers,
    )
    save_numerical_spectrum(
        spectrum,
        lines,
        data_path,
        pod_path,
        args.add_spatial_mean,
    )
    plot_spectrum(
        spectrum,
        lines,
        image_path,
        pod_path.parent.name,
        args.db_range,
        args.max_frequency_khz,
        args.max_wavenumber,
        args.add_spatial_mean,
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
