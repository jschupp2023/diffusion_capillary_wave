"""Fit shared complex modes inside user-specified temporal frequency bands.

This is a read-only Python counterpart of the complex-band decomposition in
``matlab_analysis/Analysis6.m``.  It reuses a saved shared POD basis and its
saved local-to-shared projections; no HDF5 file is created or modified.

Example for the spectrum observed in the 0p08 data::

    python -m data_analysis.pod.complex_band_modes 0p08 \
        --rank 120 --downsample 40 \
        --bands 80:115 145:180 215:260 295:340 385:430 475:525 \
        --band-ranks 3 3 2 2 2 1 --save-coordinates

Band modes are fitted using training repetitions only.  Validation and test
repetitions are used only for reconstruction diagnostics.  The saved real
state is Yosc = [Re(z1), Im(z1), ...]; it is not demodulated by a prescribed
carrier, so its oscillation frequency remains in the trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT


@dataclass(frozen=True)
class BandDefinition:
    low_hz: float
    high_hz: float
    rank: int

    @property
    def center_hz(self) -> float:
        """Metadata/plotting center; it is not used to demodulate coordinates."""
        return 0.5 * (self.low_hz + self.high_hz)


def parse_band(text: str) -> tuple[float, float]:
    """Parse a ``LOW:HIGH`` frequency interval."""
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid band {text!r}; use LOW:HIGH in Hz, for example 80:115.")
    try:
        low, high = map(float, parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid numeric band {text!r}.") from error
    if not np.isfinite([low, high]).all() or low <= 0 or high <= low:
        raise argparse.ArgumentTypeError(
            f"Invalid band {text!r}; require 0 < LOW < HIGH.")
    return low, high


def make_bands(edges: Iterable[tuple[float, float]], ranks: Iterable[int],
               nyquist_hz: float) -> tuple[BandDefinition, ...]:
    edges = tuple((float(low), float(high)) for low, high in edges)
    ranks = tuple(int(rank) for rank in ranks)
    if len(ranks) == 1 and len(edges) > 1:
        ranks = ranks * len(edges)
    if not edges or len(ranks) != len(edges):
        raise ValueError("Supply one band rank, or one rank for every band.")
    if any(rank < 1 for rank in ranks):
        raise ValueError("Every band rank must be a positive integer.")
    bands = tuple(BandDefinition(*edge, rank) for edge, rank in zip(edges, ranks))
    bands = tuple(sorted(bands, key=lambda band: band.low_hz))
    if bands[-1].high_hz >= nyquist_hz:
        raise ValueError(
            f"Highest band edge {bands[-1].high_hz:g} Hz must be below the "
            f"post-downsampling Nyquist frequency {nyquist_hz:g} Hz.")
    for left, right in zip(bands[:-1], bands[1:]):
        if left.high_hz >= right.low_hz:
            raise ValueError(
                f"Bands [{left.low_hz:g}, {left.high_hz:g}] and "
                f"[{right.low_hz:g}, {right.high_hz:g}] overlap or touch.")
    return bands


def _filter_design(band: BandDefinition, fs: float, order: int) -> np.ndarray:
    return butter(order, (band.low_hz, band.high_hz), btype="bandpass",
                  fs=fs, output="sos")


def analytic_band(values: np.ndarray, sos: np.ndarray, trim_samples: int) -> np.ndarray:
    """Return the trimmed analytic signal of a zero-phase bandpass result."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 4:
        raise ValueError("Each trajectory must have shape (time, shared_POD_mode).")
    filtered = sosfiltfilt(sos, values, axis=0)
    analytic = hilbert(filtered, axis=0)
    if trim_samples:
        if 2 * trim_samples + 100 >= len(analytic):
            raise ValueError(
                "The edge trim leaves fewer than 100 samples; reduce "
                "--edge-trim-seconds or the downsampling factor.")
        analytic = analytic[trim_samples:-trim_samples]
    return analytic


def _fix_complex_phase(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.copy()
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        vectors[:, column] *= np.exp(-1j * np.angle(vectors[pivot, column]))
        # Remove the remaining signed-zero ambiguity at an almost-real pivot.
        if vectors[pivot, column].real < 0:
            vectors[:, column] *= -1
    return vectors


def fit_complex_band_modes(
        names: Iterable[str], read_values: Callable[[str], np.ndarray], fs: float,
        bands: Iterable[BandDefinition], *, filter_order: int = 4,
        trim_samples: int = 0) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Fit equal-repetition-weight complex covariances and leading eigenvectors.

    ``read_values`` permits the production CLI to stream one repetition at a
    time while keeping this numerical core directly testable.
    """
    names = tuple(names)
    bands = tuple(bands)
    if not names:
        raise ValueError("At least one training repetition is required.")
    designs = tuple(_filter_design(band, fs, filter_order) for band in bands)
    covariances = None
    for index, name in enumerate(names, start=1):
        values = np.asarray(read_values(name), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {name}.")
        if covariances is None:
            covariances = [np.zeros((values.shape[1], values.shape[1]),
                                    dtype=np.complex128) for _ in bands]
        elif values.shape[1] != covariances[0].shape[0]:
            raise ValueError("All trajectories must have the same coordinate count.")
        print(f"Fitting bands: training repetition {index}/{len(names)} ({name})", flush=True)
        for covariance, sos in zip(covariances, designs):
            analytic = analytic_band(values, sos, trim_samples)
            covariance += (analytic.conj().T @ analytic) / (len(names) * len(analytic))
    vectors, eigenvalues, captures = [], [], []
    for band, covariance in zip(bands, covariances):
        covariance = 0.5 * (covariance + covariance.conj().T)
        values, basis = np.linalg.eigh(covariance)
        order = np.argsort(values.real)[::-1]
        values = np.maximum(values.real[order], 0.)
        if band.rank > len(values):
            raise ValueError(
                f"Band rank {band.rank} exceeds shared POD rank {len(values)}.")
        vectors.append(_fix_complex_phase(basis[:, order[:band.rank]]))
        eigenvalues.append(values)
        captures.append(float(values[:band.rank].sum() / max(values.sum(), np.finfo(float).eps)))
    return np.concatenate(vectors, axis=1), eigenvalues, np.asarray(captures)


def transform_trajectory(values: np.ndarray, fs: float,
                         bands: Iterable[BandDefinition], vectors: np.ndarray, *,
                         filter_order: int = 4, trim_samples: int = 0):
    """Project one trajectory and return coordinates plus reconstruction errors."""
    bands = tuple(bands)
    target = None
    reconstruction = None
    coordinates = []
    per_band = []
    offset = 0
    for band in bands:
        analytic = analytic_band(
            values, _filter_design(band, fs, filter_order), trim_samples)
        basis = vectors[:, offset:offset + band.rank]
        z = analytic @ basis
        truth = analytic.real
        estimate = (z @ basis.conj().T).real
        denominator = max(np.linalg.norm(truth), np.finfo(float).eps)
        per_band.append(float(np.linalg.norm(truth - estimate) / denominator))
        target = truth.copy() if target is None else target + truth
        reconstruction = estimate.copy() if reconstruction is None else reconstruction + estimate
        coordinates.append(z)
        offset += band.rank
    z_all = np.concatenate(coordinates, axis=1)
    y_osc = np.empty((len(z_all), 2 * z_all.shape[1]), dtype=np.float64)
    y_osc[:, 0::2] = z_all.real
    y_osc[:, 1::2] = z_all.imag
    selected_error = float(np.linalg.norm(target - reconstruction) /
                           max(np.linalg.norm(target), np.finfo(float).eps))
    original = values[trim_samples:-trim_samples] if trim_samples else values
    full_error = float(np.linalg.norm(original - reconstruction) /
                       max(np.linalg.norm(original), np.finfo(float).eps))
    return dict(z_osc=z_all, y_osc=y_osc, selected_band_target=target,
                reconstruction=reconstruction, per_band_nrmse=per_band,
                selected_band_nrmse=selected_error, full_signal_nrmse=full_error)


def _decoder_diagnostics(vectors: np.ndarray) -> dict:
    complex_singular = np.linalg.svd(vectors, compute_uv=False)
    # real(Yosc) decoder: A = Re(z V^H) = Re(z) Re(V)^T + Im(z) Im(V)^T
    real_decoder = np.empty((len(vectors), 2 * vectors.shape[1]), dtype=np.float64)
    real_decoder[:, 0::2] = vectors.real
    real_decoder[:, 1::2] = vectors.imag
    real_singular = np.linalg.svd(real_decoder, compute_uv=False)

    def details(singular, rows, columns):
        tolerance = np.finfo(float).eps * max(rows, columns) * singular[0]
        effective = int(np.sum(singular > tolerance))
        condition = (float(singular[0] / singular[-1])
                     if effective == columns and singular[-1] > 0 else None)
        return singular, effective, condition, float(tolerance)

    cs, cr, cc, ct = details(complex_singular, *vectors.shape)
    rs, rr, rc, rt = details(real_singular, *real_decoder.shape)
    return dict(complex_singular_values=cs, complex_effective_rank=cr,
                complex_condition_number=cc, complex_rank_tolerance=ct,
                real_decoder=real_decoder, real_decoder_singular_values=rs,
                real_decoder_effective_rank=rr, real_decoder_condition_number=rc,
                real_decoder_rank_tolerance=rt)


def _pooled_welch(names, read_values, fs, nperseg, overlap, nfft=None):
    pooled = None
    segment = None
    noverlap = None
    frequency = None
    for name in names:
        values = read_values(name)
        segment = min(int(nperseg), len(values))
        noverlap = min(int(round(overlap * segment)), segment - 1)
        fft_length = segment if nfft is None else int(nfft)
        if fft_length < segment:
            raise ValueError(
                f"Welch nfft ({fft_length}) must be at least the effective "
                f"nperseg ({segment}).")
        frequency, density = welch(values, fs=fs, window="hann", nperseg=segment,
                                   noverlap=noverlap, nfft=fft_length, detrend="constant",
                                   scaling="density", axis=0)
        summed = density.sum(axis=1)
        pooled = summed / len(names) if pooled is None else pooled + summed / len(names)
    return frequency, pooled, segment, noverlap, fft_length


def _load_shared_spatial_modes(data: SharedPODTrainingData):
    with h5py.File(data.basis_path, "r") as source:
        modes = np.asarray(source["pod/modes"][:data.rank], dtype=np.float64)
        x = np.asarray(source["grid/x"], dtype=np.float64)
        y = np.asarray(source["grid/y"], dtype=np.float64)
        x_units = str(source["grid/x"].attrs.get("units", ""))
        y_units = str(source["grid/y"].attrs.get("units", ""))
    return modes, x, y, x_units, y_units


def _plot_psd(path, frequency, density, bands):
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    positive = density > 0
    axis.semilogy(frequency[positive], density[positive], lw=1.2, color="C0")
    colors = plt.get_cmap("tab10")
    for index, band in enumerate(bands):
        axis.axvspan(band.low_hz, band.high_hz, alpha=.18,
                    color=colors(index % 10), label=f"band {index + 1}, q={band.rank}")
    axis.set(xlabel="Frequency [Hz]", ylabel="Summed shared-POD PSD [signal²/Hz]",
             title="Training pooled Welch spectrum and selected bands")
    axis.grid(alpha=.25, which="both")
    axis.legend(ncol=min(3, len(bands)), fontsize=8, frameon=False)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_modes(path, complex_modes, x, y, max_modes):
    count = min(max_modes, len(complex_modes))
    rows = max(1, count)
    figure, axes = plt.subplots(rows, 2, figsize=(8.5, 3.1 * rows),
                                squeeze=False, constrained_layout=True)
    extent = [x[0], x[-1], y[0], y[-1]]
    for index in range(count):
        amplitude = axes[index, 0].imshow(np.abs(complex_modes[index]), origin="lower",
                                          extent=extent, aspect="equal", cmap="viridis")
        phase = axes[index, 1].imshow(np.angle(complex_modes[index]), origin="lower",
                                      extent=extent, aspect="equal", cmap="twilight",
                                      vmin=-np.pi, vmax=np.pi)
        axes[index, 0].set_title(f"Complex mode {index + 1}: amplitude")
        axes[index, 1].set_title(f"Complex mode {index + 1}: phase")
        figure.colorbar(amplitude, ax=axes[index, 0], shrink=.8)
        figure.colorbar(phase, ax=axes[index, 1], shrink=.8)
        for axis in axes[index]:
            axis.set(xlabel="x", ylabel="y")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_conditioning(path, vectors, diagnostics):
    gram = np.abs(vectors.conj().T @ vectors)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    image = axes[0].imshow(gram, vmin=0, vmax=1, origin="lower", cmap="magma")
    axes[0].set(title=r"$|V^H V|$ across all band modes", xlabel="mode", ylabel="mode")
    figure.colorbar(image, ax=axes[0])
    singular = diagnostics["complex_singular_values"]
    axes[1].semilogy(np.arange(1, len(singular) + 1), singular, "o-")
    axes[1].set(title="Global complex-basis singular values", xlabel="index",
                ylabel="singular value")
    axes[1].grid(alpha=.25, which="both")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_capture_and_error(path, bands, captures, split_metrics):
    labels = [f"{band.low_hz:g}–{band.high_hz:g}" for band in bands]
    x = np.arange(len(bands))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].bar(x, 100 * captures)
    axes[0].set(xticks=x, xticklabels=labels, xlabel="band [Hz]",
                ylabel="analytic band energy captured [%]", ylim=(0, 105))
    axes[0].tick_params(axis="x", rotation=35)
    split_names = [name for name in ("train", "validation", "test") if split_metrics[name]]
    selected = [np.mean([row["selected_band_nrmse"] for row in split_metrics[name]])
                for name in split_names]
    full = [np.mean([row["full_signal_nrmse"] for row in split_metrics[name]])
            for name in split_names]
    locations = np.arange(len(split_names))
    axes[1].bar(locations - .18, selected, .36, label="selected-band target")
    axes[1].bar(locations + .18, full, .36, label="full shared-POD signal")
    axes[1].set(xticks=locations, xticklabels=split_names, ylabel="relative Frobenius error",
                title="Reconstruction errors")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.2, axis="y")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _json_metric(row):
    return {key: ([float(value) for value in values] if key == "per_band_nrmse" else float(values))
            for key, values in row.items() if key != "name"} | {"repetition": row["name"]}


def analyze(experiment: str, rank: int, band_edges, band_ranks, *,
            data_root=DEFAULT_DATA_ROOT, shared_basis=None, validation_reps=None,
            test_reps=None, downsample=1, filter_order=4, edge_trim_seconds=.01,
            welch_nperseg=8192, welch_overlap=.5, welch_nfft=None,
            save_coordinates=False,
            max_plot_modes=8, cache_gib=2., projection_block_size=2048,
            output=None, overwrite=False):
    if (rank < 1 or downsample < 1 or filter_order < 1 or welch_nperseg < 4
            or (welch_nfft is not None and welch_nfft < 4)):
        raise ValueError("Rank, downsampling, filter order and Welch length must be positive.")
    if (not 0 <= welch_overlap < 1 or edge_trim_seconds < 0 or max_plot_modes < 1
            or not np.isfinite(cache_gib) or cache_gib < 0 or projection_block_size < 1):
        raise ValueError("Invalid Welch overlap, edge trim, cache, block size, or plot count.")
    data = SharedPODTrainingData(
        experiment, rank, data_root=data_root, shared_basis=shared_basis,
        validation_reps=validation_reps, test_reps=test_reps,
        include_spatial_mean=False, output_dtype="float64")
    fs_native = 1. / data.native_dt
    fs = fs_native / downsample
    bands = make_bands(band_edges, band_ranks, fs / 2.)
    trim_samples = int(round(edge_trim_seconds * fs))

    config_for_hash = dict(experiment=experiment, rank=rank, downsample=downsample,
                           bands=[[b.low_hz, b.high_hz, b.rank] for b in bands],
                           filter_order=filter_order, edge_trim_seconds=edge_trim_seconds,
                           validation_reps=list(data.validation_repetitions),
                           test_reps=list(data.test_repetitions))
    digest = hashlib.sha256(json.dumps(config_for_hash, sort_keys=True).encode()).hexdigest()[:10]
    out = (Path(output) if output is not None else
           Path(data.source_config["data_root"]) / "complex_band_modes" / experiment /
           f"shared_r{rank}_d{downsample}_{digest}")
    expected = ("summary.json", "complex_band_modes.npz", "pooled_psd_and_bands.png",
                "complex_spatial_modes.png", "basis_conditioning.png",
                "capture_and_reconstruction.png")
    existing = [out / name for name in expected if (out / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}. Use --overwrite or another --output.")
    out.mkdir(parents=True, exist_ok=True)
    if save_coordinates:
        (out / "coordinates").mkdir(exist_ok=True)

    def project_values(name):
        # Project in blocks: source files can contain 1000 local coefficients,
        # so reading the full local matrix at once would create a large temporary.
        values = np.empty((data.frame_counts[name], rank), dtype=np.float64)
        for start in range(0, len(values), projection_block_size):
            stop = min(start + projection_block_size, len(values))
            _, values[start:stop] = data.read_coordinates(name, start, stop)
        if downsample > 1:
            values = resample_poly(values, 1, downsample, axis=0)
        return np.asarray(values, dtype=np.float64)

    all_names = tuple(name for split in ("train", "validation", "test")
                      for name in data.splits[split])
    cache_bytes = sum(math.ceil(data.frame_counts[name] / downsample)
                      for name in all_names) * rank * 8
    trajectory_cache = {}
    if cache_bytes <= cache_gib * 2**30:
        print(f"Caching {cache_bytes / 2**30:.3f} GiB of projected/downsampled "
              "trajectories in RAM.", flush=True)
        for index, name in enumerate(all_names, start=1):
            print(f"Projecting repetition {index}/{len(all_names)} ({name})", flush=True)
            trajectory_cache[name] = project_values(name)
    else:
        print(f"Projected cache would require {cache_bytes / 2**30:.3f} GiB, above "
              f"--cache-gib {cache_gib:g}; trajectories will be reread.", flush=True)

    def read_values(name):
        return trajectory_cache[name] if name in trajectory_cache else project_values(name)

    vectors, eigenvalues, captures = fit_complex_band_modes(
        data.train_repetitions, read_values, fs, bands,
        filter_order=filter_order, trim_samples=trim_samples)
    diagnostics = _decoder_diagnostics(vectors)
    modes, x, y, x_units, y_units = _load_shared_spatial_modes(data)
    phi = modes.reshape(rank, -1).T
    complex_modes = (phi @ np.conj(vectors)).T.reshape(vectors.shape[1], *modes.shape[1:])
    mode_to_band = np.concatenate([
        np.full(band.rank, index, dtype=np.int64) for index, band in enumerate(bands)])

    frequency, pooled_psd, segment, noverlap, fft_length = _pooled_welch(
        data.train_repetitions, read_values, fs, welch_nperseg, welch_overlap,
        welch_nfft)
    split_metrics = {"train": [], "validation": [], "test": []}
    for split, names in data.splits.items():
        for index, name in enumerate(names, start=1):
            print(f"Evaluating {split} repetition {index}/{len(names)} ({name})", flush=True)
            values = read_values(name)
            result = transform_trajectory(
                values, fs, bands, vectors, filter_order=filter_order,
                trim_samples=trim_samples)
            row = dict(name=name, per_band_nrmse=result["per_band_nrmse"],
                       selected_band_nrmse=result["selected_band_nrmse"],
                       full_signal_nrmse=result["full_signal_nrmse"])
            split_metrics[split].append(row)
            if save_coordinates:
                valid_count = len(result["y_osc"])
                time = (np.arange(valid_count) + trim_samples) / fs
                np.savez_compressed(
                    out / "coordinates" / f"{name}.npz",
                    time_seconds=time, y_osc=result["y_osc"].astype(np.float32),
                    band_edges_hz=np.asarray([[b.low_hz, b.high_hz] for b in bands]),
                    band_ranks=np.asarray([b.rank for b in bands]),
                    mode_to_band=mode_to_band,
                    coordinate_order="Re(z1), Im(z1), Re(z2), Im(z2), ...",
                    demodulated=np.asarray(False))

    np.savez_compressed(
        out / "complex_band_modes.npz",
        band_edges_hz=np.asarray([[b.low_hz, b.high_hz] for b in bands]),
        band_centers_hz=np.asarray([b.center_hz for b in bands]),
        band_ranks=np.asarray([b.rank for b in bands], dtype=np.int64),
        mode_to_band=mode_to_band, complex_coordinate_vectors=vectors,
        complex_spatial_modes=complex_modes.astype(np.complex64),
        band_covariance_eigenvalues=np.stack(eigenvalues),
        band_energy_capture=captures,
        complex_basis_singular_values=diagnostics["complex_singular_values"],
        real_decoder_singular_values=diagnostics["real_decoder_singular_values"],
        real_coefficient_decoder=diagnostics["real_decoder"],
        grid_x=x, grid_y=y, frequency_hz=frequency, pooled_training_psd=pooled_psd)

    report = dict(
        purpose="training-fitted complex modes within explicit temporal frequency bands",
        experiment=experiment, shared_pod_rank=rank, shared_basis=str(data.basis_path),
        source_coordinate_definition="shared POD coefficients only; spatial-mean coordinate excluded",
        signal_units=data.signal_units, grid_units=dict(x=x_units, y=y_units),
        native_sampling_hz=float(fs_native), downsample_factor=int(downsample),
        effective_sampling_hz=float(fs), downsampling="scipy.signal.resample_poly with anti-aliasing",
        projected_trajectory_cache_gib=float(cache_bytes / 2**30),
        projected_trajectories_cached=bool(trajectory_cache),
        filter=f"zero-phase order-{filter_order} Butterworth bandpass (scipy sosfiltfilt)",
        edge_trim_seconds=float(edge_trim_seconds), edge_trim_samples=int(trim_samples),
        fit_weighting="equal repetition weights; each covariance divided by its valid sample count",
        fit_repetitions=list(data.train_repetitions),
        validation_repetitions=list(data.validation_repetitions),
        test_repetitions=list(data.test_repetitions),
        bands=[dict(index=index, low_hz=band.low_hz, high_hz=band.high_hz,
                    center_hz=band.center_hz, rank=band.rank,
                    energy_capture=float(captures[index]))
               for index, band in enumerate(bands)],
        complex_coordinate_count=int(vectors.shape[1]),
        real_state_count=int(2 * vectors.shape[1]),
        saved_state=("Yosc=[Re(z1), Im(z1), ...], retaining oscillation; no carrier "
                     "demodulation is applied"),
        reconstruction=("Per-band target is the real bandpass-filtered shared-POD trajectory. "
                        "Selected-band target is their sum. Full-signal error additionally "
                        "penalizes all frequencies outside the selected bands."),
        complex_basis_effective_rank=diagnostics["complex_effective_rank"],
        complex_basis_condition_number=diagnostics["complex_condition_number"],
        complex_basis_rank_tolerance=diagnostics["complex_rank_tolerance"],
        real_instantaneous_decoder_effective_rank=diagnostics["real_decoder_effective_rank"],
        real_instantaneous_decoder_condition_number=diagnostics["real_decoder_condition_number"],
        real_instantaneous_decoder_rank_tolerance=diagnostics["real_decoder_rank_tolerance"],
        decoder_note=("The complex-basis condition number measures independence of modes across "
                      "bands. The real instantaneous decoder can be rank-deficient because an "
                      "analytic signal contains temporal quadrature state not independently "
                      "observable from one real snapshot."),
        welch=dict(window="hann", detrend="constant", scaling="density",
                   nperseg=int(segment), noverlap=int(noverlap), nfft=int(fft_length),
                   training_repetition_average="equal weight after summing coordinate PSDs"),
        metrics={split: [_json_metric(row) for row in rows]
                 for split, rows in split_metrics.items()},
        shared_basis_leakage_note=("The complex band modes are fitted on training repetitions only. "
                                   "The supplied pre-existing shared spatial POD may itself have "
                                   "been constructed from a broader repetition population."),
        output_files=dict(modes="complex_band_modes.npz",
                          coordinates=("coordinates/*.npz" if save_coordinates else None)))
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _plot_psd(out / "pooled_psd_and_bands.png", frequency, pooled_psd, bands)
    _plot_modes(out / "complex_spatial_modes.png", complex_modes, x, y, max_plot_modes)
    _plot_conditioning(out / "basis_conditioning.png", vectors, diagnostics)
    _plot_capture_and_error(out / "capture_and_reconstruction.png", bands, captures, split_metrics)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", help="Experiment folder, e.g. 0p08.")
    parser.add_argument("--rank", type=int, required=True,
                        help="Number of shared POD coordinates used as input.")
    parser.add_argument("--bands", nargs="+", type=parse_band, required=True,
                        metavar="LOW:HIGH", help="Nonoverlapping frequency bands in Hz.")
    parser.add_argument("--band-ranks", nargs="+", type=int, default=[2], metavar="Q",
                        help="One rank for all bands, or one rank per band (default: 2).")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Antialiased integer downsampling factor (default: 1).")
    parser.add_argument("--filter-order", type=int, default=4,
                        help="Butterworth bandpass order (default: 4).")
    parser.add_argument("--edge-trim-seconds", type=float, default=.01,
                        help="Trim this duration from both filtered ends (default: 0.01).")
    parser.add_argument("--welch-nperseg", type=int, default=8192)
    parser.add_argument("--welch-overlap", type=float, default=.5)
    parser.add_argument("--welch-nfft", type=int,
                        help="FFT length; zero-padding is allowed (default: nperseg).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--validation-reps", nargs="*", type=int)
    parser.add_argument("--test-reps", nargs="*", type=int)
    parser.add_argument("--save-coordinates", action="store_true",
                        help="Also save one real Yosc trajectory NPZ per repetition.")
    parser.add_argument("--max-plot-modes", type=int, default=8)
    parser.add_argument("--cache-gib", type=float, default=2.,
                        help="RAM budget for projected/downsampled trajectories (default: 2).")
    parser.add_argument("--projection-block-size", type=int, default=2048,
                        help="Rows per local-to-shared projection block (default: 2048).")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    destination = analyze(
        args.experiment, args.rank, args.bands, args.band_ranks,
        data_root=args.data_root, shared_basis=args.shared_basis,
        validation_reps=args.validation_reps, test_reps=args.test_reps,
        downsample=args.downsample, filter_order=args.filter_order,
        edge_trim_seconds=args.edge_trim_seconds, welch_nperseg=args.welch_nperseg,
        welch_overlap=args.welch_overlap, welch_nfft=args.welch_nfft,
        save_coordinates=args.save_coordinates,
        max_plot_modes=args.max_plot_modes, cache_gib=args.cache_gib,
        projection_block_size=args.projection_block_size, output=args.output,
        overwrite=args.overwrite)
    print(destination)


if __name__ == "__main__":
    main()
