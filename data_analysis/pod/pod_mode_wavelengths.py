"""Estimate characteristic wavelengths of spatial POD modes using 2-D FFTs.

Each mode is optionally Hann-windowed, transformed in x and y, and collapsed
into a radial energy spectrum. If the exact constant-vector projection contains
most of a mode's energy, its wavelength is reported as N/A. Otherwise, the
dominant nonzero radial wavenumber is converted using lambda = 2*pi/k.
Spectral-centroid and RMS wavelengths are also reported because individual POD
modes need not be monochromatic.

Example
-------
    python pod_mode_wavelengths.py /path/to/Ca_ac_..._rep1
"""

from __future__ import annotations

import argparse
import csv
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

from data_analysis.psd.pod_center_psd import resolve_pod_file


WAVENUMBER_CONVERSION = 2.0 * np.pi * 1_000.0  # cycles/micron -> rad/mm


@dataclass(frozen=True)
class ModeWavelengthResults:
    mode_number: np.ndarray
    singular_value: np.ndarray
    energy_fraction: np.ndarray
    wavenumber: np.ndarray
    normalized_radial_power: np.ndarray
    zero_wavenumber_energy_fraction: np.ndarray
    dominant_wavenumber: np.ndarray
    dominant_wavelength: np.ndarray
    centroid_wavenumber: np.ndarray
    centroid_wavelength: np.ndarray
    rms_wavenumber: np.ndarray
    rms_wavelength: np.ndarray
    peak_energy_fraction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Reduced-data repetition folder or POD HDF5 file.",
    )
    parser.add_argument(
        "--pod-rank",
        type=int,
        default=1_000,
        help="Rank in the POD filename when input is a folder (default: 1000).",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of leading modes to analyze (default: all stored modes).",
    )
    parser.add_argument(
        "--zero-pad-factor",
        type=int,
        default=2,
        help="FFT zero-padding factor in x and y (default: 2).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the default two-dimensional Hann window.",
    )
    parser.add_argument(
        "--zero-mode-threshold",
        type=float,
        default=0.5,
        help=(
            "Return N/A when the constant-vector energy fraction is at least "
            "this value (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=4,
        help="Worker threads used by SciPy FFT operations (default: 4).",
    )
    parser.add_argument(
        "--max-wavenumber",
        type=float,
        default=500.0,
        help="Maximum displayed wavenumber in rad/mm (default: 500).",
    )
    parser.add_argument(
        "--db-range",
        type=float,
        default=50.0,
        help="Heatmap range below each mode's spectral peak (default: 50 dB).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: <repetition-folder>/pod_mode_wavelengths).",
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
        help="Replace existing output files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pod_rank < 1:
        raise ValueError("--pod-rank must be positive.")
    if args.count is not None and args.count < 1:
        raise ValueError("--count must be positive.")
    if args.zero_pad_factor < 1:
        raise ValueError("--zero-pad-factor must be at least 1.")
    if not 0 < args.zero_mode_threshold <= 1:
        raise ValueError("--zero-mode-threshold must lie in (0, 1].")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")
    if args.max_wavenumber <= 0:
        raise ValueError("--max-wavenumber must be positive.")
    if args.db_range <= 0:
        raise ValueError("--db-range must be positive.")
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
    variation = float(np.max(np.abs(differences - spacing)) / spacing)
    if variation > 1e-2:
        raise ValueError(
            f"{name} grid is not sufficiently uniform; relative spacing variation "
            f"is {variation:.3e}."
        )
    if variation > 1e-3:
        warnings.warn(
            f"{name} grid spacing varies by {variation:.3%}; using its mean "
            "spacing for the FFT.",
            stacklevel=2,
        )
    return spacing


def _require_micron_units(x_units: str, y_units: str) -> None:
    accepted = {"micron", "microns", "um"}
    normalized_x = x_units.strip().lower().replace("µ", "u")
    normalized_y = y_units.strip().lower().replace("µ", "u")
    if normalized_x not in accepted or normalized_y not in accepted:
        raise ValueError(
            "Expected x and y grid units to be microns for wavelength conversion; "
            f"found x={x_units!r}, y={y_units!r}."
        )


def _energy_fractions(
    handle: h5py.File, singular_values: np.ndarray, count: int
) -> np.ndarray:
    if "pod/cumulative_energy_fraction" in handle:
        cumulative = np.asarray(
            handle["pod/cumulative_energy_fraction"][:count], dtype=np.float64
        )
        if len(cumulative) == count and np.isfinite(cumulative).all():
            fractions = np.diff(np.concatenate(([0.0], cumulative)))
            # Tiny negative differences can occur when float storage rounds two
            # adjacent cumulative values to the same precision.
            return np.maximum(fractions, 0.0)
    squared = singular_values[:count].astype(np.float64) ** 2
    total = float(np.sum(squared))
    if total <= 0:
        raise ValueError("Stored singular values contain no positive energy.")
    return squared / total


def build_radial_bins(
    height: int,
    width: int,
    y_spacing: float,
    x_spacing: float,
    zero_pad_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    fft_height = height * zero_pad_factor
    fft_width = width * zero_pad_factor
    qx = fft.fftfreq(fft_width, d=x_spacing)
    qy = fft.fftfreq(fft_height, d=y_spacing)
    delta_qx = 1.0 / (fft_width * x_spacing)
    delta_qy = 1.0 / (fft_height * y_spacing)
    radial_width_q = min(delta_qx, delta_qy)
    inscribed_limit_q = min(0.5 / x_spacing, 0.5 / y_spacing)
    bin_count = max(
        2, int(np.floor(inscribed_limit_q / radial_width_q + 1e-9))
    )
    q_radius = np.hypot(qy[:, None], qx[None, :]).ravel()
    radial_indices = np.floor(q_radius / radial_width_q).astype(np.int64)
    valid = radial_indices < bin_count
    radial_centers_q = (np.arange(bin_count, dtype=np.float64) + 0.5) * radial_width_q
    wavenumber = radial_centers_q * WAVENUMBER_CONVERSION
    return radial_indices, valid, wavenumber, fft_height, fft_width


def _refined_peak_wavenumber(power: np.ndarray, wavenumber: np.ndarray) -> float:
    """Refine a discrete peak with a three-point log-parabolic interpolation."""
    peak_index = int(np.argmax(power))
    peak = float(wavenumber[peak_index])
    if peak_index == 0 or peak_index == len(power) - 1:
        return peak
    local = power[peak_index - 1 : peak_index + 2]
    if np.any(local <= 0) or not np.isfinite(local).all():
        return peak
    values = np.log(local)
    denominator = values[0] - 2.0 * values[1] + values[2]
    if denominator == 0:
        return peak
    offset = 0.5 * (values[0] - values[2]) / denominator
    offset = float(np.clip(offset, -0.5, 0.5))
    bin_width = float(wavenumber[1] - wavenumber[0])
    return peak + offset * bin_width


def _radial_shell_power(
    field: np.ndarray,
    fft_height: int,
    fft_width: int,
    fft_workers: int,
    valid: np.ndarray,
    valid_radial_indices: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    transformed = fft.fft2(
        field,
        s=(fft_height, fft_width),
        workers=fft_workers,
        overwrite_x=True,
    )
    flattened = transformed.ravel()
    pixel_power = flattened.real**2 + flattened.imag**2
    return np.bincount(
        valid_radial_indices,
        weights=pixel_power[valid],
        minlength=bin_count,
    ).astype(np.float64, copy=False)


def analyze_modes(
    modes: h5py.Dataset,
    singular_values: np.ndarray,
    energy_fractions: np.ndarray,
    count: int,
    y_spacing: float,
    x_spacing: float,
    zero_pad_factor: int,
    apply_window: bool,
    zero_mode_threshold: float,
    fft_workers: int,
) -> ModeWavelengthResults:
    height, width = modes.shape[1:]
    (
        radial_indices,
        valid,
        wavenumber,
        fft_height,
        fft_width,
    ) = build_radial_bins(
        height, width, y_spacing, x_spacing, zero_pad_factor
    )
    valid_indices = radial_indices[valid]
    radial_power = np.empty((count, len(wavenumber)), dtype=np.float32)
    dominant_wavenumber = np.empty(count, dtype=np.float64)
    centroid_wavenumber = np.empty(count, dtype=np.float64)
    rms_wavenumber = np.empty(count, dtype=np.float64)
    peak_energy_fraction = np.empty(count, dtype=np.float64)
    zero_wavenumber_energy_fraction = np.empty(count, dtype=np.float64)

    if apply_window:
        y_window = get_window("hann", height, fftbins=True).astype(np.float32)
        x_window = get_window("hann", width, fftbins=True).astype(np.float32)
        spatial_window = y_window[:, None] * x_window[None, :]
    else:
        spatial_window = np.ones((height, width), dtype=np.float32)
    window_sum = float(np.sum(spatial_window, dtype=np.float64))

    report_every = max(1, count // 10)
    started = time.perf_counter()
    for mode_index in range(count):
        mode = np.asarray(modes[mode_index], dtype=np.float32)
        if not np.isfinite(mode).all():
            raise ValueError(f"POD mode {mode_index + 1} contains nonfinite values.")

        mode_energy = float(np.sum(mode.astype(np.float64) ** 2))
        if mode_energy <= 0:
            raise ValueError(f"POD mode {mode_index + 1} is identically zero.")
        mode_sum = float(np.sum(mode, dtype=np.float64))
        zero_fraction = mode_sum**2 / (mode.size * mode_energy)
        zero_fraction = float(np.clip(zero_fraction, 0.0, 1.0))
        zero_wavenumber_energy_fraction[mode_index] = zero_fraction

        # Preserve the constant component in the saved radial spectrum so that
        # k=0-dominated modes are visible rather than silently centered away.
        shell_power = _radial_shell_power(
            mode * spatial_window,
            fft_height,
            fft_width,
            fft_workers,
            valid,
            valid_indices,
            len(wavenumber),
        )
        displayed_total = float(np.sum(shell_power))
        if displayed_total <= 0 or not np.isfinite(displayed_total):
            raise ValueError(f"POD mode {mode_index + 1} has no finite FFT power.")
        radial_power[mode_index] = (shell_power / displayed_total).astype(np.float32)

        if zero_fraction >= zero_mode_threshold:
            dominant_wavenumber[mode_index] = np.nan
            centroid_wavenumber[mode_index] = np.nan
            rms_wavenumber[mode_index] = np.nan
            peak_energy_fraction[mode_index] = np.nan
        else:
            # Remove only the constant component for estimation of the
            # remaining finite wavelength. Weighted centering makes the DC FFT
            # coefficient vanish after application of the spatial window.
            weighted_mean = float(
                np.sum(mode * spatial_window, dtype=np.float64) / window_sum
            )
            oscillatory_field = (mode - np.float32(weighted_mean)) * spatial_window
            oscillatory_power = _radial_shell_power(
                oscillatory_field,
                fft_height,
                fft_width,
                fft_workers,
                valid,
                valid_indices,
                len(wavenumber),
            )
            nonzero_power = oscillatory_power[1:]
            nonzero_wavenumber = wavenumber[1:]
            total = float(np.sum(nonzero_power))
            if total <= 0 or not np.isfinite(total):
                raise ValueError(
                    f"POD mode {mode_index + 1} has no finite "
                    "nonzero-wavenumber power."
                )
            dominant_wavenumber[mode_index] = _refined_peak_wavenumber(
                nonzero_power, nonzero_wavenumber
            )
            centroid_wavenumber[mode_index] = float(
                np.sum(nonzero_wavenumber * nonzero_power) / total
            )
            rms_wavenumber[mode_index] = float(
                np.sqrt(np.sum(nonzero_wavenumber**2 * nonzero_power) / total)
            )
            peak_energy_fraction[mode_index] = float(np.max(nonzero_power) / total)

        if (
            mode_index == 0
            or (mode_index + 1) % report_every == 0
            or mode_index + 1 == count
        ):
            print(f"Analyzed {mode_index + 1:,}/{count:,} modes", flush=True)

    print(
        f"Mode FFT analysis completed in {time.perf_counter() - started:.2f} s.",
        flush=True,
    )
    unavailable_count = int(
        np.count_nonzero(zero_wavenumber_energy_fraction >= zero_mode_threshold)
    )
    print(
        f"Marked {unavailable_count:,}/{count:,} wavelength(s) N/A because "
        f"k=0 energy was at least {zero_mode_threshold:.1%}.",
        flush=True,
    )
    wavelength_factor = 2.0 * np.pi * 1_000.0  # rad/mm -> microns
    return ModeWavelengthResults(
        mode_number=np.arange(1, count + 1, dtype=np.int64),
        singular_value=singular_values[:count],
        energy_fraction=energy_fractions,
        wavenumber=wavenumber,
        normalized_radial_power=radial_power,
        zero_wavenumber_energy_fraction=zero_wavenumber_energy_fraction,
        dominant_wavenumber=dominant_wavenumber,
        dominant_wavelength=wavelength_factor / dominant_wavenumber,
        centroid_wavenumber=centroid_wavenumber,
        centroid_wavelength=wavelength_factor / centroid_wavenumber,
        rms_wavenumber=rms_wavenumber,
        rms_wavelength=wavelength_factor / rms_wavenumber,
        peak_energy_fraction=peak_energy_fraction,
    )


def save_csv(results: ModeWavelengthResults, output_path: Path) -> None:
    columns = (
        "mode_number",
        "singular_value",
        "pod_energy_fraction",
        "zero_wavenumber_energy_fraction",
        "wavelength_status",
        "dominant_wavenumber_rad_per_mm",
        "dominant_wavelength_microns",
        "centroid_wavenumber_rad_per_mm",
        "centroid_wavelength_microns",
        "rms_wavenumber_rad_per_mm",
        "rms_wavelength_microns",
        "peak_radial_energy_fraction",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(len(results.mode_number)):
            available = np.isfinite(results.dominant_wavelength[index])

            def formatted(value: float) -> str:
                return f"{value:.12g}" if available else "N/A"

            writer.writerow(
                (
                    int(results.mode_number[index]),
                    f"{results.singular_value[index]:.12g}",
                    f"{results.energy_fraction[index]:.12g}",
                    f"{results.zero_wavenumber_energy_fraction[index]:.12g}",
                    "finite wavelength" if available else "N/A: k=0 dominant",
                    formatted(results.dominant_wavenumber[index]),
                    formatted(results.dominant_wavelength[index]),
                    formatted(results.centroid_wavenumber[index]),
                    formatted(results.centroid_wavelength[index]),
                    formatted(results.rms_wavenumber[index]),
                    formatted(results.rms_wavelength[index]),
                    formatted(results.peak_energy_fraction[index]),
                )
            )


def save_npz(
    results: ModeWavelengthResults,
    output_path: Path,
    source_path: Path,
    zero_pad_factor: int,
    window_applied: bool,
    zero_mode_threshold: float,
) -> None:
    np.savez_compressed(
        output_path,
        mode_number=results.mode_number,
        singular_value=results.singular_value,
        pod_energy_fraction=results.energy_fraction,
        wavenumber_rad_per_mm=results.wavenumber,
        normalized_radial_power=results.normalized_radial_power,
        zero_wavenumber_energy_fraction=results.zero_wavenumber_energy_fraction,
        dominant_wavenumber_rad_per_mm=results.dominant_wavenumber,
        dominant_wavelength_microns=results.dominant_wavelength,
        centroid_wavenumber_rad_per_mm=results.centroid_wavenumber,
        centroid_wavelength_microns=results.centroid_wavelength,
        rms_wavenumber_rad_per_mm=results.rms_wavenumber,
        rms_wavelength_microns=results.rms_wavelength,
        peak_radial_energy_fraction=results.peak_energy_fraction,
        zero_pad_factor=np.int64(zero_pad_factor),
        zero_mode_threshold=np.float64(zero_mode_threshold),
        spatial_window=np.asarray("hann" if window_applied else "none"),
        source_pod_file=np.asarray(str(source_path.resolve())),
    )


def plot_wavelengths(
    results: ModeWavelengthResults,
    output_path: Path,
    experiment_name: str,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.plot(
        results.mode_number,
        results.dominant_wavelength,
        color="tab:blue",
        linewidth=0.8,
        alpha=0.75,
        label="dominant spectral peak",
    )
    axis.scatter(
        results.mode_number,
        results.dominant_wavelength,
        s=8,
        color="tab:blue",
        alpha=0.8,
    )
    axis.plot(
        results.mode_number,
        results.centroid_wavelength,
        color="tab:orange",
        linewidth=1.0,
        alpha=0.9,
        label="spectral centroid",
    )
    axis.set_xlabel("POD mode number")
    axis.set_ylabel(r"Characteristic wavelength $\lambda$ [$\mu$m]")
    axis.set_title(f"Spatial wavelength of POD modes\n{experiment_name}")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    unavailable = int(np.count_nonzero(~np.isfinite(results.dominant_wavelength)))
    if unavailable:
        axis.text(
            0.985,
            0.02,
            f"{unavailable} k=0-dominated mode(s): N/A",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_spectral_heatmap(
    results: ModeWavelengthResults,
    output_path: Path,
    experiment_name: str,
    max_wavenumber: float,
    db_range: float,
    dpi: int,
) -> None:
    mask = results.wavenumber <= max_wavenumber
    if not np.any(mask):
        raise ValueError("--max-wavenumber excludes every FFT bin.")
    displayed_power = results.normalized_radial_power[:, mask].astype(np.float64)
    row_maxima = np.max(displayed_power, axis=1, keepdims=True)
    positive_maxima = row_maxima[row_maxima > 0]
    if not len(positive_maxima):
        raise ValueError("No positive radial power lies in the displayed range.")
    row_maxima[row_maxima <= 0] = 1.0
    floor = 10.0 ** (-db_range / 10.0)
    power_db = 10.0 * np.log10(
        np.maximum(displayed_power / row_maxima, floor)
    )
    displayed_wavenumber = results.wavenumber[mask]

    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    image = axis.imshow(
        power_db,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        extent=(
            0.0,
            float(displayed_wavenumber[-1]),
            float(results.mode_number[-1]) + 0.5,
            0.5,
        ),
        cmap="magma",
        vmin=-db_range,
        vmax=0.0,
    )
    axis.set_xlabel(r"Radial angular wavenumber $|\mathbf{k}|$ [rad/mm]")
    axis.set_ylabel("POD mode number")
    axis.set_title(f"Radial spatial FFT of POD modes\n{experiment_name}")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Power relative to each mode's peak [dB]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    validate_args(args)
    started = time.perf_counter()
    pod_path = resolve_pod_file(args.input, args.pod_rank)
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pod_path.parent / "pod_mode_wavelengths"
    )
    csv_path = output_directory / "mode_wavelengths.csv"
    data_path = output_directory / "mode_wavelengths.npz"
    wavelength_plot_path = output_directory / "mode_wavelengths.png"
    spectrum_plot_path = output_directory / "mode_radial_spectra.png"
    output_paths = (csv_path, data_path, wavelength_plot_path, spectrum_plot_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    with h5py.File(pod_path, "r") as handle:
        required = ("pod/modes", "pod/singular_values", "grid/x", "grid/y")
        for dataset_name in required:
            if dataset_name not in handle:
                raise KeyError(f"Missing dataset {dataset_name!r} in {pod_path}.")
        modes = handle["pod/modes"]
        singular_values = np.asarray(handle["pod/singular_values"], dtype=np.float64)
        x_dataset = handle["grid/x"]
        y_dataset = handle["grid/y"]
        x = np.asarray(x_dataset, dtype=np.float64)
        y = np.asarray(y_dataset, dtype=np.float64)
        if modes.ndim != 3 or modes.shape[1:] != (len(y), len(x)):
            raise ValueError(
                "pod/modes must have shape (mode, len(grid/y), len(grid/x)); "
                f"got {modes.shape}."
            )
        if singular_values.shape != (len(modes),):
            raise ValueError("pod/singular_values must contain one value per mode.")
        count = len(modes) if args.count is None else args.count
        if count > len(modes):
            raise ValueError(
                f"Requested {count:,} modes, but only {len(modes):,} are stored."
            )
        x_units = _attribute_text(x_dataset.attrs.get("units"))
        y_units = _attribute_text(y_dataset.attrs.get("units"))
        _require_micron_units(x_units, y_units)
        x_spacing = _uniform_spacing(x, "x")
        y_spacing = _uniform_spacing(y, "y")
        energy_fractions = _energy_fractions(handle, singular_values, count)

        print(
            f"Input: {pod_path}\n"
            f"Analyzing {count:,}/{len(modes):,} modes on a "
            f"{modes.shape[1]} x {modes.shape[2]} grid; "
            f"zero-padding factor: {args.zero_pad_factor}; "
            f"window: {'none' if args.no_window else '2-D Hann'}.",
            flush=True,
        )
        results = analyze_modes(
            modes,
            singular_values,
            energy_fractions,
            count,
            y_spacing,
            x_spacing,
            args.zero_pad_factor,
            not args.no_window,
            args.zero_mode_threshold,
            args.fft_workers,
        )

    save_csv(results, csv_path)
    save_npz(
        results,
        data_path,
        pod_path,
        args.zero_pad_factor,
        not args.no_window,
        args.zero_mode_threshold,
    )
    plot_wavelengths(results, wavelength_plot_path, pod_path.parent.name, args.dpi)
    plot_spectral_heatmap(
        results,
        spectrum_plot_path,
        pod_path.parent.name,
        args.max_wavenumber,
        args.db_range,
        args.dpi,
    )
    for output_path in output_paths:
        print(f"Saved {output_path.resolve()}")
    print(f"Finished in {time.perf_counter() - started:.2f} s.")
    return tuple(path.resolve() for path in output_paths)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
