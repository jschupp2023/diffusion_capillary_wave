"""Process the full 0p20 experimental recording and compare PSD and capillary energy.

This script processes all 100,001 frames of the 0.20 Vpp recording:
1. Reconstructs each frame using gradient-domain Poisson restoration.
2. Extracts center-point and spatial-mean signals for raw vs cleaned fields.
3. Computes Welch PSDs for center-point and spatial-mean signals.
4. Computes exact and quadratic capillary surface energies using the physical
   water surface tension (gamma = 0.0728 N/m) and metric coordinates.
5. Saves comparative diagnostic plots, full time-series NPZ, and a summary JSON.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent BLAS thread oversubscription in worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.pod.phase_unwrap_gradient_poisson import (
    DEFAULT_Q,
    DEFAULT_TAU,
    restore_frame_poisson,
)
from data_analysis.energy.capillary_energy import field_energies

DEFAULT_INPUT = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
GAMMA_WATER = 0.0728  # N/m (surface tension of water at 20 deg C)


def _process_chunk_worker(
    input_path_str: str,
    start_idx: int,
    end_idx: int,
    q: float,
    tau: float,
    method: str = "integer",
) -> dict[str, np.ndarray]:
    """Worker process to restore and compute metrics for a contiguous frame chunk."""
    n_frames = end_idx - start_idx
    with h5py.File(input_path_str, "r") as h5:
        main_grp = h5["main"]
        x_m = np.asarray(h5["meta"]["x"], dtype=np.float64) * 1e-6
        y_m = np.asarray(h5["meta"]["y"], dtype=np.float64) * 1e-6

        raw_frames_m = np.empty((n_frames, len(y_m), len(x_m)), dtype=np.float64)
        cln_frames_m = np.empty((n_frames, len(y_m), len(x_m)), dtype=np.float64)

        raw_center = np.empty(n_frames, dtype=np.float32)
        cln_center = np.empty(n_frames, dtype=np.float32)
        raw_mean = np.empty(n_frames, dtype=np.float32)
        cln_mean = np.empty(n_frames, dtype=np.float32)

        raw_max_grad = np.empty(n_frames, dtype=np.float32)
        cln_max_grad = np.empty(n_frames, dtype=np.float32)
        was_modified = np.empty(n_frames, dtype=bool)
        pct_pixels_modified = np.empty(n_frames, dtype=np.float32)

        cy, cx = len(y_m) // 2, len(x_m) // 2

        for i in range(n_frames):
            frame_num = start_idx + i
            key = _frame_key(frame_num)
            frame_raw = np.asarray(main_grp[key], dtype=np.float32)

            res = restore_frame_poisson(frame_raw, q=q, tau=tau, method=method)
            frame_cln = res.restored

            raw_frames_m[i] = frame_raw.astype(np.float64) * 1e-6
            cln_frames_m[i] = frame_cln.astype(np.float64) * 1e-6

            raw_center[i] = frame_raw[cy, cx]
            cln_center[i] = frame_cln[cy, cx]
            raw_mean[i] = float(np.mean(frame_raw))
            cln_mean[i] = float(np.mean(frame_cln))

            raw_max_grad[i] = res.orig_max_dx
            cln_max_grad[i] = res.rest_max_dx
            was_modified[i] = res.was_modified
            pct_pixels_modified[i] = res.pct_pixels_modified

        # Vectorized batch energy computation on chunk
        quad_raw, exact_raw = field_energies(raw_frames_m, x_m, y_m, geometry="surface")
        quad_cln, exact_cln = field_energies(cln_frames_m, x_m, y_m, geometry="surface")

        raw_energy_quad = quad_raw * GAMMA_WATER
        raw_energy_exact = exact_raw * GAMMA_WATER
        cln_energy_quad = quad_cln * GAMMA_WATER
        cln_energy_exact = exact_cln * GAMMA_WATER

    return {
        "raw_center": raw_center,
        "cln_center": cln_center,
        "raw_mean": raw_mean,
        "cln_mean": cln_mean,
        "raw_max_grad": raw_max_grad,
        "cln_max_grad": cln_max_grad,
        "was_modified": was_modified,
        "pct_pixels_modified": pct_pixels_modified,
        "raw_energy_quad": raw_energy_quad,
        "raw_energy_exact": raw_energy_exact,
        "cln_energy_quad": cln_energy_quad,
        "cln_energy_exact": cln_energy_exact,
    }


def compute_welch_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 8192,
    overlap_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD with Hann window and mean detrending."""
    noverlap = int(round(overlap_fraction * nperseg))
    freq, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    return freq, psd


def plot_psd_analysis(
    freq: np.ndarray,
    raw_psd_center: np.ndarray,
    cln_psd_center: np.ndarray,
    raw_psd_mean: np.ndarray,
    cln_psd_mean: np.ndarray,
    output_path: Path,
    dpi: int = 200,
) -> None:
    """Generate high-resolution log-log PSD comparison figures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    pos = (freq > 0) & (raw_psd_center > 0) & (cln_psd_center > 0)
    f = freq[pos]
    p_raw_c = raw_psd_center[pos]
    p_cln_c = cln_psd_center[pos]

    # 1. Center-Point PSD
    axes[0, 0].loglog(f, p_raw_c, color="crimson", lw=1.2, label="Original Raw", alpha=0.85)
    axes[0, 0].loglog(f, p_cln_c, color="royalblue", lw=1.2, label="Cleaned (Poisson)", alpha=0.9)
    axes[0, 0].set_title("Center-Point Elevation PSD $S_{hh}(f)$", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Frequency [Hz]")
    axes[0, 0].set_ylabel(r"PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]")
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.35)
    axes[0, 0].legend(fontsize=10)

    # 2. Center-Point PSD Ratio
    ratio_c = p_cln_c / p_raw_c
    axes[0, 1].semilogx(f, ratio_c, color="darkgreen", lw=1.2)
    axes[0, 1].axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[0, 1].set_title("Center PSD Ratio: Cleaned / Original", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Frequency [Hz]")
    axes[0, 1].set_ylabel("Power Ratio")
    axes[0, 1].set_ylim(0.0, 1.5)
    axes[0, 1].grid(True, which="both", ls="--", alpha=0.35)

    # 3. Spatial-Mean PSD
    pos_m = (freq > 0) & (raw_psd_mean > 0) & (cln_psd_mean > 0)
    p_raw_m = raw_psd_mean[pos_m]
    p_cln_m = cln_psd_mean[pos_m]

    axes[1, 0].loglog(freq[pos_m], p_raw_m, color="crimson", lw=1.2, label="Original Raw", alpha=0.85)
    axes[1, 0].loglog(freq[pos_m], p_cln_m, color="royalblue", lw=1.2, label="Cleaned (Poisson)", alpha=0.9)
    axes[1, 0].set_title("Spatial-Mean Height PSD $S_{\\bar{h}\\bar{h}}(f)$", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel(r"PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]")
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.35)
    axes[1, 0].legend(fontsize=10)

    # 4. Zoomed Low-Frequency Peak Region (10 Hz to 2000 Hz)
    peak_band = (f >= 30) & (f <= 2500)
    axes[1, 1].loglog(f[peak_band], p_raw_c[peak_band], color="crimson", lw=1.3, label="Original Raw", alpha=0.85)
    axes[1, 1].loglog(f[peak_band], p_cln_c[peak_band], color="royalblue", lw=1.3, label="Cleaned", alpha=0.9)
    axes[1, 1].set_title("Turbulent Energy Cascade Band (30 Hz - 2.5 kHz)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].set_ylabel(r"PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]")
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.35)
    axes[1, 1].legend(fontsize=10)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_energy_analysis(
    time_s: np.ndarray,
    raw_energy: np.ndarray,
    cln_energy: np.ndarray,
    output_path: Path,
    dpi: int = 200,
) -> None:
    """Generate capillary energy time series and distribution comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # 1. Full Timeline
    axes[0, 0].plot(time_s, raw_energy * 1e12, color="crimson", lw=0.6, alpha=0.7, label="Original Raw")
    axes[0, 0].plot(time_s, cln_energy * 1e12, color="royalblue", lw=0.8, alpha=0.9, label="Cleaned (Poisson)")
    axes[0, 0].set_title("Capillary Energy Timeline $E_{\\mathrm{quad}}(t)$", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Capillary Energy [pJ]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)

    # 2. Zoomed Segment (e.g. turbulent burst episode around t ~ 0.013 - 0.017 s, frames 1500-2000)
    burst_mask = (time_s >= 0.012) & (time_s <= 0.018)
    axes[0, 1].plot(time_s[burst_mask] * 1e3, raw_energy[burst_mask] * 1e12, color="crimson", lw=1.0, alpha=0.8, label="Original Raw")
    axes[0, 1].plot(time_s[burst_mask] * 1e3, cln_energy[burst_mask] * 1e12, color="royalblue", lw=1.2, alpha=0.95, label="Cleaned")
    axes[0, 1].set_title("Turbulent Burst Episode (Frames ~1400 - 2100)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Time [ms]")
    axes[0, 1].set_ylabel("Capillary Energy [pJ]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)

    # 3. Log Energy Histogram
    log_raw = np.log10(np.maximum(raw_energy * 1e12, 1e-3))
    log_cln = np.log10(np.maximum(cln_energy * 1e12, 1e-3))
    bins = np.linspace(min(log_cln.min(), log_raw.min()), max(log_cln.max(), log_raw.max()), 100)

    axes[1, 0].hist(log_raw, bins=bins, density=True, color="crimson", alpha=0.5, label="Original Raw")
    axes[1, 0].hist(log_cln, bins=bins, density=True, color="royalblue", alpha=0.6, label="Cleaned")
    axes[1, 0].set_title("Energy Probability Distribution $\\log_{10}(E / \\mathrm{pJ})$", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel(r"$\log_{10}(E_{\mathrm{quad}} \,/\, \mathrm{pJ})$")
    axes[1, 0].set_ylabel("Probability Density")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)

    # 4. Energy Spike Elimination CDF / Exceedance
    sorted_raw = np.sort(raw_energy * 1e12)[::-1]
    sorted_cln = np.sort(cln_energy * 1e12)[::-1]
    prob = np.linspace(1 / len(sorted_raw), 1.0, len(sorted_raw))

    axes[1, 1].loglog(sorted_raw, prob, color="crimson", lw=1.5, label="Original Raw")
    axes[1, 1].loglog(sorted_cln, prob, color="royalblue", lw=1.5, label="Cleaned")
    axes[1, 1].set_title("Complementary CDF: $P(E > E_0)$ (Heavy Tail Collapse)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Energy Threshold $E_0$ [pJ]")
    axes[1, 1].set_ylabel("Exceedance Probability")
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.35)
    axes[1, 1].legend(fontsize=10)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_gradient_analysis(
    time_s: np.ndarray,
    raw_max_grad: np.ndarray,
    cln_max_grad: np.ndarray,
    output_path: Path,
    dpi: int = 200,
) -> None:
    """Plot spatial maximum gradient timeline and distributions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    axes[0].plot(time_s, raw_max_grad, color="crimson", lw=0.6, alpha=0.7, label="Original Raw")
    axes[0].plot(time_s, cln_max_grad, color="royalblue", lw=0.8, alpha=0.9, label="Cleaned")
    axes[0].axhline(DEFAULT_TAU, color="black", ls="--", lw=1.0, label=f"Threshold $\\tau = {DEFAULT_TAU} \\,\\mu\\mathrm{{m}}$")
    axes[0].set_title("Timeline of Maximum Spatial Gradient $\\max |\\nabla h|$", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel(r"$\max |\nabla h|$ [$\mu\mathrm{m}$]")
    axes[0].set_ylim(0, max(15.0, float(np.percentile(raw_max_grad, 99.5))))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Histogram
    bins = np.linspace(0, 15.0, 150)
    axes[1].hist(np.clip(raw_max_grad, 0, 15.0), bins=bins, density=True, color="crimson", alpha=0.5, label="Original Raw")
    axes[1].hist(np.clip(cln_max_grad, 0, 15.0), bins=bins, density=True, color="royalblue", alpha=0.6, label="Cleaned")
    axes[1].axvline(DEFAULT_TAU, color="black", ls="--", lw=1.0, label=f"$\\tau = {DEFAULT_TAU} \\,\\mu\\mathrm{{m}}$")
    axes[1].set_title("Distribution of Maximum Spatial Gradients", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(r"$\max |\nabla h|$ [$\mu\mathrm{m}$]")
    axes[1].set_ylabel("Probability Density")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def run_whole_comparison(
    input_path: Path,
    output_dir: Path,
    q: float = DEFAULT_Q,
    tau: float = DEFAULT_TAU,
    method: str = "integer",
    workers: int = 8,
    chunk_size: int = 500,
    nperseg: int = 8192,
) -> dict[str, Any]:
    """Execute complete 0p20 dataset reconstruction and comparative analysis."""
    started = time.perf_counter()
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        time_values = np.asarray(h5["meta"]["t"], dtype=np.float64)
        n_total = len(time_values)

    dt = float(np.median(np.diff(time_values)))
    fs = 1.0 / dt
    print(f"Dataset: {input_path.name}")
    print(f"Total frames: {n_total:,} | Sampling frequency: {fs:,.2f} Hz | dt: {dt*1e6:.3f} us")
    print(f"Parameters: q = {q:.6f} um | tau = {tau:.2f} um | method = {method} | workers = {workers} | chunk_size = {chunk_size}")

    chunks = [(i, min(i + chunk_size, n_total)) for i in range(0, n_total, chunk_size)]
    print(f"Processing in {len(chunks)} parallel chunks...", flush=True)

    t_proc_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_chunk_worker, str(input_path), s, e, q, tau, method)
            for s, e in chunks
        ]
        chunk_results = []
        for idx, f in enumerate(futures):
            res = f.result()
            chunk_results.append(res)
            if (idx + 1) % max(1, len(chunks) // 10) == 0 or idx + 1 == len(chunks):
                done_frames = min((idx + 1) * chunk_size, n_total)
                elapsed_s = time.perf_counter() - t_proc_start
                fps = done_frames / elapsed_s
                eta_s = (n_total - done_frames) / fps if fps > 0 else 0
                print(
                    f"  Chunk {idx+1}/{len(chunks)}: {done_frames:,}/{n_total:,} frames "
                    f"({100 * done_frames / n_total:.1f}%) | {fps:.1f} fps | ETA: {eta_s:.0f} s",
                    flush=True,
                )

    # Concatenate chunk outputs
    print("Aggregating time-series arrays...", flush=True)
    raw_center = np.concatenate([r["raw_center"] for r in chunk_results])
    cln_center = np.concatenate([r["cln_center"] for r in chunk_results])
    raw_mean = np.concatenate([r["raw_mean"] for r in chunk_results])
    cln_mean = np.concatenate([r["cln_mean"] for r in chunk_results])
    raw_max_grad = np.concatenate([r["raw_max_grad"] for r in chunk_results])
    cln_max_grad = np.concatenate([r["cln_max_grad"] for r in chunk_results])
    was_modified = np.concatenate([r["was_modified"] for r in chunk_results])
    pct_pixels_modified = np.concatenate([r["pct_pixels_modified"] for r in chunk_results])
    raw_energy_quad = np.concatenate([r["raw_energy_quad"] for r in chunk_results])
    raw_energy_exact = np.concatenate([r["raw_energy_exact"] for r in chunk_results])
    cln_energy_quad = np.concatenate([r["cln_energy_quad"] for r in chunk_results])
    cln_energy_exact = np.concatenate([r["cln_energy_exact"] for r in chunk_results])

    print("Computing Welch PSDs...", flush=True)
    freq, psd_raw_center = compute_welch_psd(raw_center, fs, nperseg=nperseg)
    _, psd_cln_center = compute_welch_psd(cln_center, fs, nperseg=nperseg)
    _, psd_raw_mean = compute_welch_psd(raw_mean, fs, nperseg=nperseg)
    _, psd_cln_mean = compute_welch_psd(cln_mean, fs, nperseg=nperseg)

    # Energy and Discontinuity Summary Metrics
    modified_count = int(np.sum(was_modified))
    modified_pct = float(100 * modified_count / n_total)

    stats = {
        "dataset": str(input_path),
        "total_frames": int(n_total),
        "sampling_frequency_hz": float(fs),
        "q_microns": float(q),
        "tau_microns": float(tau),
        "method": str(method),
        "modified_frames_count": modified_count,
        "modified_frames_pct": modified_pct,
        "untouched_frames_count": int(n_total - modified_count),
        "untouched_frames_pct": float(100 * (n_total - modified_count) / n_total),
        "pixels_modified": {
            "median_pct_in_modified_frames": float(np.median(pct_pixels_modified[was_modified])) if modified_count > 0 else 0.0,
            "p95_pct_in_modified_frames": float(np.percentile(pct_pixels_modified[was_modified], 95)) if modified_count > 0 else 0.0,
            "max_pct_in_modified_frames": float(np.max(pct_pixels_modified[was_modified])) if modified_count > 0 else 0.0,
            "overall_mean_pct_pixels_modified": float(np.mean(pct_pixels_modified)),
        },
        "processing_time_seconds": float(time.perf_counter() - started),
        "gradients": {
            "raw_max_spatial_grad_um": float(np.max(raw_max_grad)),
            "raw_p99_max_spatial_grad_um": float(np.percentile(raw_max_grad, 99)),
            "raw_median_max_spatial_grad_um": float(np.median(raw_max_grad)),
            "cln_max_spatial_grad_um": float(np.max(cln_max_grad)),
            "cln_p99_max_spatial_grad_um": float(np.percentile(cln_max_grad, 99)),
            "cln_median_max_spatial_grad_um": float(np.median(cln_max_grad)),
        },
        "capillary_energy_quad_pJ": {
            "raw_mean": float(np.mean(raw_energy_quad) * 1e12),
            "raw_median": float(np.median(raw_energy_quad) * 1e12),
            "raw_std": float(np.std(raw_energy_quad) * 1e12),
            "raw_p95": float(np.percentile(raw_energy_quad, 95) * 1e12),
            "raw_p99": float(np.percentile(raw_energy_quad, 99) * 1e12),
            "raw_max": float(np.max(raw_energy_quad) * 1e12),
            "cln_mean": float(np.mean(cln_energy_quad) * 1e12),
            "cln_median": float(np.median(cln_energy_quad) * 1e12),
            "cln_std": float(np.std(cln_energy_quad) * 1e12),
            "cln_p95": float(np.percentile(cln_energy_quad, 95) * 1e12),
            "cln_p99": float(np.percentile(cln_energy_quad, 99) * 1e12),
            "cln_max": float(np.max(cln_energy_quad) * 1e12),
            "mean_energy_reduction_pct": float(
                100 * (np.mean(raw_energy_quad) - np.mean(cln_energy_quad)) / np.mean(raw_energy_quad)
            ),
            "max_energy_reduction_pct": float(
                100 * (np.max(raw_energy_quad) - np.max(cln_energy_quad)) / np.max(raw_energy_quad)
            ),
        },
        "capillary_energy_exact_pJ": {
            "raw_mean": float(np.mean(raw_energy_exact) * 1e12),
            "cln_mean": float(np.mean(cln_energy_exact) * 1e12),
            "raw_max": float(np.max(raw_energy_exact) * 1e12),
            "cln_max": float(np.max(cln_energy_exact) * 1e12),
        },
        "center_point_elevation_um": {
            "raw_std": float(np.std(raw_center)),
            "cln_std": float(np.std(cln_center)),
            "raw_min": float(np.min(raw_center)),
            "raw_max": float(np.max(raw_center)),
            "cln_min": float(np.min(cln_center)),
            "cln_max": float(np.max(cln_center)),
        },
    }

    print("Generating diagnostic plots...", flush=True)
    plot_psd_analysis(
        freq,
        psd_raw_center,
        psd_cln_center,
        psd_raw_mean,
        psd_cln_mean,
        output_dir / "psd_comparison.png",
    )
    plot_energy_analysis(
        time_values,
        raw_energy_quad,
        cln_energy_quad,
        output_dir / "energy_comparison.png",
    )
    plot_gradient_analysis(
        time_values,
        raw_max_grad,
        cln_max_grad,
        output_dir / "spatial_gradient_comparison.png",
    )

    print("Saving compressed NPZ and summary JSON...", flush=True)
    np.savez_compressed(
        output_dir / "comparison_metrics.npz",
        time=time_values,
        raw_center=raw_center,
        cln_center=cln_center,
        raw_mean=raw_mean,
        cln_mean=cln_mean,
        raw_energy_quad=raw_energy_quad,
        cln_energy_quad=cln_energy_quad,
        raw_energy_exact=raw_energy_exact,
        cln_energy_exact=cln_energy_exact,
        raw_max_grad=raw_max_grad,
        cln_max_grad=cln_max_grad,
        was_modified=was_modified,
        pct_pixels_modified=pct_pixels_modified,
        freq=freq,
        psd_raw_center=psd_raw_center,
        psd_cln_center=psd_cln_center,
        psd_raw_mean=psd_raw_mean,
        psd_cln_mean=psd_cln_mean,
    )

    with (output_dir / "summary.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nFinished full recording analysis in {time.perf_counter() - started:.2f} s.")
    print(f"Saved artifacts to {output_dir}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison_0p20_integer_cleaned"))
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument(
        "--method",
        type=str,
        default="integer",
        choices=["integer", "poisson"],
        help="Restoration method ('integer' or 'poisson')",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--nperseg", type=int, default=8192)
    args = parser.parse_args()

    run_whole_comparison(
        input_path=args.raw_input,
        output_dir=args.output_dir,
        q=args.q,
        tau=args.tau,
        method=args.method,
        workers=args.workers,
        chunk_size=args.chunk_size,
        nperseg=args.nperseg,
    )


if __name__ == "__main__":
    main()
