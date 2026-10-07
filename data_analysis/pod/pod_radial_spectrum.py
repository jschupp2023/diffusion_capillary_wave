"""Compute a radial 2-D wavenumber-frequency spectrum from a POD result.

For each of several evenly spaced time clips, this script reconstructs the
complete two-dimensional POD field and computes its Fourier transform in time,
x, and y. The (kx, ky) power is then azimuthally integrated into shells of
constant |k| and averaged between clips. Processing one clip at a time avoids
ever constructing the complete space-time experiment in memory.

By default, eight 4,096-frame clips and all stored POD modes are used. The
instantaneous spatial mean is not restored.

Example
-------
    python pod_radial_spectrum.py /path/to/Ca_ac_..._rep1
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
from scipy import fft, sparse
from scipy.signal.windows import get_window

from data_analysis.psd.pod_center_psd import infer_sampling_frequency, resolve_pod_file


WAVENUMBER_CONVERSION = 2.0 * np.pi * 1_000.0  # cycles/micron -> rad/mm


@dataclass(frozen=True)
class PodMetadata:
    n_frames: int
    height: int
    width: int
    stored_rank: int
    reconstruction_rank: int
    x: np.ndarray
    y: np.ndarray
    x_spacing: float
    y_spacing: float
    sampling_frequency: float
    time_step_jitter: float
    signal_units: str
    spatial_mean_was_removed: bool


@dataclass(frozen=True)
class RadialSpectrum:
    frequency: np.ndarray
    wavenumber: np.ndarray
    power_density: np.ndarray
    segment_starts: np.ndarray
    segment_length: int


def frequency_normalized_power(
    spectrum: RadialSpectrum,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P(f|k) and the frequency-integrated power at every k."""
    if len(spectrum.frequency) < 2:
        raise ValueError("At least two frequency bins are required for normalization.")
    differences = np.diff(spectrum.frequency)
    frequency_step = float(np.mean(differences))
    if not np.allclose(differences, frequency_step, rtol=1e-10, atol=0.0):
        raise ValueError("Frequency bins must be uniformly spaced.")
    trapezoid_weights = np.ones(len(spectrum.frequency), dtype=np.float64)
    trapezoid_weights[[0, -1]] = 0.5
    integrated_power = frequency_step * np.sum(
        spectrum.power_density * trapezoid_weights[:, None], axis=0
    )
    normalized = np.zeros_like(spectrum.power_density, dtype=np.float64)
    np.divide(
        spectrum.power_density,
        integrated_power[None, :],
        out=normalized,
        where=integrated_power[None, :] > 0,
    )
    return normalized, integrated_power


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
        "--nperseg",
        type=int,
        default=4_096,
        help="Frames in each reconstructed 2-D clip (default: 4096).",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=8,
        help="Evenly spaced clips averaged into the spectrum (default: 8).",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        help="Sampling frequency in Hz (default: inferred from grid/time).",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=4,
        help="Worker threads used by SciPy FFT operations (default: 4).",
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
        "--column-min-power-db",
        type=float,
        default=-80.0,
        help=(
            "In the column-normalized plot, mask wavenumber columns whose "
            "total energy is this many dB below the strongest column "
            "(default: -80)."
        ),
    )
    parser.add_argument(
        "--max-frequency-khz",
        type=float,
        default=20.0,
        help="Maximum displayed frequency in kHz (default: 20).",
    )
    parser.add_argument(
        "--max-wavenumber",
        type=float,
        default=500.0,
        help="Maximum displayed |k| in rad/mm (default: 500).",
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
    if args.segments < 1:
        raise ValueError("--segments must be positive.")
    if args.sampling_frequency is not None and args.sampling_frequency <= 0:
        raise ValueError("--sampling-frequency must be positive.")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")
    if args.db_range <= 0:
        raise ValueError("--db-range must be positive.")
    if args.column_min_power_db > 0:
        raise ValueError("--column-min-power-db must be zero or negative.")
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


def _require_spatial_units(x_units: str, y_units: str) -> None:
    accepted = {"micron", "microns", "um"}
    normalized_x = x_units.strip().lower().replace("µ", "u")
    normalized_y = y_units.strip().lower().replace("µ", "u")
    if normalized_x not in accepted or normalized_y not in accepted:
        raise ValueError(
            "Expected x and y grid units to be microns for conversion to rad/mm; "
            f"found x={x_units!r}, y={y_units!r}."
        )


def read_metadata(
    handle: h5py.File,
    requested_rank: int | None,
    sampling_frequency_override: float | None,
) -> PodMetadata:
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
            raise KeyError(f"Missing dataset {dataset_name!r}.")

    modes = handle["pod/modes"]
    coefficients = handle["reduced/coefficients"]
    spatial_means = handle["preprocessing/frame_spatial_mean"]
    if modes.ndim != 3:
        raise ValueError(f"pod/modes must be 3-D; got shape {modes.shape}.")
    if coefficients.ndim != 2:
        raise ValueError(
            "reduced/coefficients must have shape (time, mode); got "
            f"{coefficients.shape}."
        )
    if coefficients.shape[1] != len(modes):
        raise ValueError("POD mode and coefficient counts do not match.")
    if len(coefficients) != len(spatial_means):
        raise ValueError("Coefficient and frame_spatial_mean lengths do not match.")

    time_values = np.asarray(handle["grid/time"], dtype=np.float64)
    x_dataset = handle["grid/x"]
    y_dataset = handle["grid/y"]
    x = np.asarray(x_dataset, dtype=np.float64)
    y = np.asarray(y_dataset, dtype=np.float64)
    if len(time_values) != len(coefficients):
        raise ValueError("Coefficient and time-grid lengths do not match.")
    if modes.shape[1:] != (len(y), len(x)):
        raise ValueError("Mode shape does not match the saved x and y grids.")

    x_units = _attribute_text(x_dataset.attrs.get("units"))
    y_units = _attribute_text(y_dataset.attrs.get("units"))
    _require_spatial_units(x_units, y_units)
    inferred_frequency, jitter = infer_sampling_frequency(time_values)
    rank = len(modes) if requested_rank is None else requested_rank
    if rank > len(modes):
        raise ValueError(
            f"Requested reconstruction rank {rank}, but only {len(modes)} modes "
            "are stored."
        )
    return PodMetadata(
        n_frames=len(coefficients),
        height=modes.shape[1],
        width=modes.shape[2],
        stored_rank=len(modes),
        reconstruction_rank=rank,
        x=x,
        y=y,
        x_spacing=_uniform_spacing(x, "x"),
        y_spacing=_uniform_spacing(y, "y"),
        sampling_frequency=(
            inferred_frequency
            if sampling_frequency_override is None
            else sampling_frequency_override
        ),
        time_step_jitter=jitter,
        signal_units=_attribute_text(coefficients.attrs.get("units")),
        spatial_mean_was_removed=bool(
            spatial_means.attrs.get(
                "subtracted_before_pod",
                handle.attrs.get("instantaneous_spatial_mean_removed", False),
            )
        ),
    )


def select_segment_starts(
    n_frames: int, segment_length: int, requested_segments: int
) -> np.ndarray:
    if segment_length > n_frames:
        raise ValueError(
            f"--nperseg is {segment_length:,}, but the result contains only "
            f"{n_frames:,} frames."
        )
    possible_starts = n_frames - segment_length + 1
    count = min(requested_segments, possible_starts)
    if count < requested_segments:
        warnings.warn(
            f"Only {count} distinct clips are available; reducing --segments from "
            f"{requested_segments} to {count}.",
            stacklevel=2,
        )
    if count == 1:
        return np.asarray([(n_frames - segment_length) // 2], dtype=np.int64)
    return np.rint(
        np.linspace(0, n_frames - segment_length, count)
    ).astype(np.int64)


def build_radial_reducer(
    metadata: PodMetadata,
) -> tuple[sparse.csr_matrix, np.ndarray, float, float]:
    """Build an operator that sums Cartesian spectral pixels by radial shell."""
    spatial_frequency_x = 1.0 / metadata.x_spacing
    spatial_frequency_y = 1.0 / metadata.y_spacing
    qx = fft.fftfreq(metadata.width, d=metadata.x_spacing)
    qy = fft.fftfreq(metadata.height, d=metadata.y_spacing)
    delta_qx = spatial_frequency_x / metadata.width
    delta_qy = spatial_frequency_y / metadata.height
    radial_bin_width_q = min(delta_qx, delta_qy)
    inscribed_limit_q = min(
        0.5 * spatial_frequency_x, 0.5 * spatial_frequency_y
    )
    bin_count = max(
        1, int(np.floor(inscribed_limit_q / radial_bin_width_q + 1e-9))
    )
    radial_edges_q = np.arange(bin_count + 1, dtype=np.float64) * radial_bin_width_q

    q_radius = np.hypot(qy[:, None], qx[None, :]).ravel()
    valid = q_radius < radial_edges_q[-1]
    spatial_indices = np.flatnonzero(valid)
    radial_indices = np.floor(
        q_radius[valid] / radial_bin_width_q
    ).astype(np.int64)
    reducer = sparse.csr_matrix(
        (
            np.ones(len(spatial_indices), dtype=np.float32),
            (radial_indices, spatial_indices),
        ),
        shape=(bin_count, metadata.height * metadata.width),
    )
    radial_centers_q = 0.5 * (radial_edges_q[:-1] + radial_edges_q[1:])
    radial_centers = radial_centers_q * WAVENUMBER_CONVERSION

    # A Cartesian density is integrated over each annulus and divided by its
    # radial width, yielding density per angular wavenumber (rad/mm).
    radial_integration_scale = (
        delta_qx
        * delta_qy
        / (radial_bin_width_q * WAVENUMBER_CONVERSION)
    )
    radial_limit = radial_edges_q[-1] * WAVENUMBER_CONVERSION
    return reducer, radial_centers, radial_integration_scale, radial_limit


def compute_radial_spectrum(
    handle: h5py.File,
    metadata: PodMetadata,
    segment_starts: np.ndarray,
    segment_length: int,
    add_spatial_mean: bool,
    fft_workers: int,
) -> RadialSpectrum:
    if add_spatial_mean and not metadata.spatial_mean_was_removed:
        raise ValueError(
            "Cannot add frame_spatial_mean because it was not removed before POD."
        )

    modes_dataset = handle["pod/modes"]
    coefficients_dataset = handle["reduced/coefficients"]
    spatial_mean_dataset = handle["preprocessing/frame_spatial_mean"]
    rank = metadata.reconstruction_rank
    print(f"Loading {rank:,} complete 2-D POD modes...", flush=True)
    modes = np.asarray(modes_dataset[:rank], dtype=np.float32).reshape(rank, -1)
    modes = np.ascontiguousarray(modes)

    reducer, wavenumber, radial_scale, radial_limit = build_radial_reducer(metadata)
    frequencies = fft.rfftfreq(
        segment_length, d=1.0 / metadata.sampling_frequency
    )
    accumulated = np.zeros((len(frequencies), len(wavenumber)), dtype=np.float64)

    time_window = get_window("hann", segment_length, fftbins=True).astype(np.float32)
    y_window = get_window("hann", metadata.height, fftbins=True).astype(np.float32)
    x_window = get_window("hann", metadata.width, fftbins=True).astype(np.float32)
    spatial_window = y_window[:, None] * x_window[None, :]
    temporal_window_energy = float(np.sum(time_window.astype(np.float64) ** 2))
    x_window_energy = float(np.sum(x_window.astype(np.float64) ** 2))
    y_window_energy = float(np.sum(y_window.astype(np.float64) ** 2))
    spatial_frequency_x = 1.0 / metadata.x_spacing
    spatial_frequency_y = 1.0 / metadata.y_spacing
    cartesian_density_scale = 1.0 / (
        metadata.sampling_frequency
        * spatial_frequency_x
        * spatial_frequency_y
        * temporal_window_energy
        * x_window_energy
        * y_window_energy
    )

    total_started = time.perf_counter()
    for segment_number, start in enumerate(segment_starts, start=1):
        segment_started = time.perf_counter()
        stop = int(start) + segment_length
        print(
            f"Clip {segment_number}/{len(segment_starts)}: frames "
            f"{int(start):,}:{stop:,}",
            flush=True,
        )
        coefficients = np.asarray(
            coefficients_dataset[int(start):stop, :rank], dtype=np.float32
        )
        field = (coefficients @ modes).reshape(
            segment_length, metadata.height, metadata.width
        )
        del coefficients
        if add_spatial_mean:
            frame_means = np.asarray(
                spatial_mean_dataset[int(start):stop], dtype=np.float32
            )
            field += frame_means[:, None, None]
            del frame_means

        # This removes the stationary temporal mean field. It has support only
        # at f=0 and would otherwise leak into nearby bins after windowing.
        field -= field.mean(axis=0, keepdims=True)
        field *= time_window[:, None, None]
        field *= spatial_window[None, :, :]

        transformed = fft.rfft(
            field,
            axis=0,
            workers=fft_workers,
            overwrite_x=True,
        )
        del field
        transformed = fft.fftn(
            transformed,
            axes=(1, 2),
            workers=fft_workers,
            overwrite_x=True,
        )

        flattened = transformed.reshape(len(frequencies), -1)
        frequency_chunk = 32
        for frequency_start in range(0, len(frequencies), frequency_chunk):
            frequency_stop = min(
                frequency_start + frequency_chunk, len(frequencies)
            )
            block = flattened[frequency_start:frequency_stop]
            block_power = block.real**2 + block.imag**2
            accumulated[frequency_start:frequency_stop] += (
                reducer @ block_power.T
            ).T
        del transformed, flattened
        print(
            f"Clip {segment_number}/{len(segment_starts)} completed in "
            f"{time.perf_counter() - segment_started:.2f} s.",
            flush=True,
        )

    accumulated *= (
        cartesian_density_scale * radial_scale / len(segment_starts)
    )
    if segment_length % 2 == 0:
        accumulated[1:-1] *= 2.0
    else:
        accumulated[1:] *= 2.0
    print(
        f"Radial spectrum completed in {time.perf_counter() - total_started:.2f} s; "
        f"usable isotropic range |k| < {radial_limit:.1f} rad/mm.",
        flush=True,
    )
    return RadialSpectrum(
        frequency=frequencies,
        wavenumber=wavenumber,
        power_density=accumulated,
        segment_starts=segment_starts,
        segment_length=segment_length,
    )


def save_numerical_spectrum(
    spectrum: RadialSpectrum,
    metadata: PodMetadata,
    output_path: Path,
    pod_path: Path,
    spatial_mean_added: bool,
) -> None:
    normalized_power, integrated_power = frequency_normalized_power(spectrum)
    np.savez_compressed(
        output_path,
        frequency_hz=spectrum.frequency,
        wavenumber_rad_per_mm=spectrum.wavenumber,
        radial_power_density=spectrum.power_density.astype(np.float32),
        frequency_normalized_power_density=normalized_power.astype(np.float32),
        wavenumber_integrated_power=integrated_power.astype(np.float32),
        segment_starts=spectrum.segment_starts,
        segment_length=np.int64(spectrum.segment_length),
        segment_count=np.int64(len(spectrum.segment_starts)),
        reconstruction_rank=np.int64(metadata.reconstruction_rank),
        sampling_frequency_hz=np.float64(metadata.sampling_frequency),
        spatial_mean_added=np.bool_(spatial_mean_added),
        radial_definition=np.asarray(
            "azimuthally integrated 2-D Cartesian power per radial wavenumber"
        ),
        source_pod_file=np.asarray(str(pod_path.resolve())),
    )


def plot_spectrum(
    spectrum: RadialSpectrum,
    metadata: PodMetadata,
    output_path: Path,
    experiment_name: str,
    db_range: float,
    max_frequency_khz: float | None,
    max_wavenumber: float | None,
    spatial_mean_added: bool,
    dpi: int,
    normalization: str = "global",
    column_min_power_db: float = -80.0,
) -> None:
    if normalization == "global":
        plotted_power = spectrum.power_density
        valid_columns = np.ones(len(spectrum.wavenumber), dtype=bool)
        title_prefix = "Radially integrated 2-D wavenumber–frequency spectrum"
        colorbar_label = "Power relative to global maximum [dB]"
    elif normalization == "frequency-column":
        plotted_power, integrated_power = frequency_normalized_power(spectrum)
        strongest_column = float(np.max(integrated_power))
        if strongest_column <= 0:
            raise ValueError("No wavenumber column contains positive power.")
        column_power_db = 10.0 * np.log10(
            np.maximum(integrated_power, np.finfo(np.float64).tiny)
            / strongest_column
        )
        valid_columns = column_power_db >= column_min_power_db
        title_prefix = "Frequency-normalized 2-D wavenumber–frequency spectrum"
        colorbar_label = r"Conditional power $P(f\mid k)$ [dB relative to maximum]"
    else:
        raise ValueError(f"Unknown plot normalization: {normalization!r}.")

    positive = plotted_power[plotted_power > 0]
    if not len(positive):
        raise ValueError("The radial spectrum has no positive power values.")
    maximum = float(np.max(positive))
    floor = maximum * 10.0 ** (-db_range / 10.0)
    power_db = 10.0 * np.log10(np.maximum(plotted_power, floor) / maximum)
    if normalization == "frequency-column":
        power_db[:, ~valid_columns] = np.nan

    frequency_khz = spectrum.frequency / 1_000.0
    frequency_mask = np.ones(len(frequency_khz), dtype=bool)
    wavenumber_mask = np.ones(len(spectrum.wavenumber), dtype=bool)
    if max_frequency_khz is not None:
        frequency_mask &= frequency_khz <= max_frequency_khz
    if max_wavenumber is not None:
        wavenumber_mask &= spectrum.wavenumber <= max_wavenumber
    if not np.any(frequency_mask) or not np.any(wavenumber_mask):
        raise ValueError("Requested plot limits exclude every spectral bin.")

    displayed = power_db[np.ix_(frequency_mask, wavenumber_mask)]
    displayed_frequency = frequency_khz[frequency_mask]
    displayed_wavenumber = spectrum.wavenumber[wavenumber_mask]
    figure, axis = plt.subplots(figsize=(8.4, 5.8))
    color_map = plt.get_cmap("magma").copy()
    color_map.set_bad(color="#b8b8b8")
    image = axis.imshow(
        displayed,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            0.0,
            float(displayed_wavenumber[-1]),
            float(displayed_frequency[0]),
            float(displayed_frequency[-1]),
        ),
        cmap=color_map,
        vmin=-db_range,
        vmax=0.0,
    )
    axis.set_xlabel(r"Radial angular wavenumber $|\mathbf{k}|$ [rad/mm]")
    axis.set_ylabel("Frequency [kHz]")
    mean_label = "spatial mean restored" if spatial_mean_added else "spatial mean removed"
    axis.set_title(
        f"{title_prefix}\n"
        f"{experiment_name} — {mean_label}"
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)
    details = (
        f"rank {metadata.reconstruction_rank:,}; "
        f"$f_s$={metadata.sampling_frequency:,.1f} Hz\n"
        f"{len(spectrum.segment_starts)} clips; "
        f"{spectrum.segment_length:,} frames/clip"
    )
    if normalization == "frequency-column":
        details += f"\ncolumns below {column_min_power_db:g} dB masked"
    axis.text(
        0.985,
        0.98,
        details,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "edgecolor": "white", "alpha": 0.55},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    validate_args(args)
    started = time.perf_counter()
    pod_path = resolve_pod_file(args.input, args.pod_rank)
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pod_path.parent / "pod_space_time"
    )
    stem = "radial_2d_k_frequency_spectrum"
    if args.add_spatial_mean:
        stem += "_with_spatial_mean"
    image_path = output_directory / f"{stem}.png"
    column_normalized_path = output_directory / f"{stem}_column_normalized.png"
    data_path = output_directory / f"{stem}.npz"
    output_paths = (image_path, column_normalized_path, data_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    with h5py.File(pod_path, "r") as handle:
        metadata = read_metadata(
            handle, args.reconstruction_rank, args.sampling_frequency
        )
        starts = select_segment_starts(
            metadata.n_frames, args.nperseg, args.segments
        )
        print(
            f"Input: {pod_path}\n"
            f"Field: {metadata.height} x {metadata.width}; "
            f"frames: {metadata.n_frames:,}; rank: {metadata.reconstruction_rank:,}\n"
            f"Sampling frequency: {metadata.sampling_frequency:,.6f} Hz; "
            f"time-grid maximum relative step variation: "
            f"{metadata.time_step_jitter:.3%}.",
            flush=True,
        )
        spectrum = compute_radial_spectrum(
            handle,
            metadata,
            starts,
            args.nperseg,
            args.add_spatial_mean,
            args.fft_workers,
        )

    save_numerical_spectrum(
        spectrum, metadata, data_path, pod_path, args.add_spatial_mean
    )
    plot_spectrum(
        spectrum,
        metadata,
        image_path,
        pod_path.parent.name,
        args.db_range,
        args.max_frequency_khz,
        args.max_wavenumber,
        args.add_spatial_mean,
        args.dpi,
        normalization="global",
        column_min_power_db=args.column_min_power_db,
    )
    plot_spectrum(
        spectrum,
        metadata,
        column_normalized_path,
        pod_path.parent.name,
        args.db_range,
        args.max_frequency_khz,
        args.max_wavenumber,
        args.add_spatial_mean,
        args.dpi,
        normalization="frequency-column",
        column_min_power_db=args.column_min_power_db,
    )
    print(f"Saved {image_path.resolve()}")
    print(f"Saved {column_normalized_path.resolve()}")
    print(f"Saved {data_path.resolve()}")
    print(f"Finished in {time.perf_counter() - started:.2f} s.")
    return (
        image_path.resolve(),
        column_normalized_path.resolve(),
        data_path.resolve(),
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
