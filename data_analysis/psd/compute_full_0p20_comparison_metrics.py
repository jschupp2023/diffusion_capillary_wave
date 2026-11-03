"""Compute and save exact comparison metrics between RAW and CLEANED 0p20 datasets.

Streams all 100,001 frames from:
  RAW:   /home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5
  CLEAN: /home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5

Computes:
1. Exact Center-point time series and Welch PSDs (raw vs cleaned)
2. Exact Spatial-mean time series and Welch PSDs (raw vs cleaned)
3. Capillary deformation energies (quadratic and exact)
4. Spatial maximum gradients (raw vs cleaned)
5. Summary metrics JSON and high-resolution comparison plots
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent BLAS thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from data_analysis.pod.pod_2d import _frame_key
from data_analysis.energy.capillary_energy import field_energies

RAW_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
CLEAN_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
OUTPUT_DIR = Path("runs/comparison_0p20_integer_cleaned")
GAMMA_WATER = 0.0728  # N/m


def compute_welch_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 8192,
    overlap_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Opening RAW and CLEANED datasets...")
    
    with h5py.File(RAW_H5, "r") as f_raw, h5py.File(CLEAN_H5, "r") as f_cln:
        raw_main = f_raw["main"]
        cln_main = f_cln["main"]
        
        t_arr = np.asarray(f_raw["meta"]["t"])
        x_m = np.asarray(f_raw["meta"]["x"], dtype=np.float64) * 1e-6
        y_m = np.asarray(f_raw["meta"]["y"], dtype=np.float64) * 1e-6
        
        n_frames = len(t_arr)
        dt = float(np.median(np.diff(t_arr)))
        fs = 1.0 / dt
        
        print(f"Total frames: {n_frames:,} | Grid: {len(y_m)} x {len(x_m)} | Sampling: {fs:.1f} Hz (dt = {dt*1e6:.2f} us)")
        
        raw_center = np.empty(n_frames, dtype=np.float32)
        cln_center = np.empty(n_frames, dtype=np.float32)
        raw_mean = np.empty(n_frames, dtype=np.float32)
        cln_mean = np.empty(n_frames, dtype=np.float32)
        
        raw_max_grad = np.empty(n_frames, dtype=np.float32)
        cln_max_grad = np.empty(n_frames, dtype=np.float32)
        
        raw_energy_quad = np.empty(n_frames, dtype=np.float64)
        raw_energy_exact = np.empty(n_frames, dtype=np.float64)
        cln_energy_quad = np.empty(n_frames, dtype=np.float64)
        cln_energy_exact = np.empty(n_frames, dtype=np.float64)
        
        cy, cx = len(y_m) // 2, len(x_m) // 2
        batch_size = 1000
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        print(f"Streaming and computing metrics in {n_batches} batches of {batch_size} frames...")
        
        for b in range(n_batches):
            b_start = b * batch_size
            b_end = min(b_start + batch_size, n_frames)
            b_len = b_end - b_start
            
            raw_block = np.empty((b_len, len(y_m), len(x_m)), dtype=np.float32)
            cln_block = np.empty((b_len, len(y_m), len(x_m)), dtype=np.float32)
            
            for i in range(b_len):
                fn = b_start + i
                k = _frame_key(fn)
                raw_block[i] = raw_main[k][:]
                cln_block[i] = cln_main[k][:]
                
            # Time series extractions
            raw_center[b_start:b_end] = raw_block[:, cy, cx]
            cln_center[b_start:b_end] = cln_block[:, cy, cx]
            raw_mean[b_start:b_end] = np.mean(raw_block, axis=(1, 2))
            cln_mean[b_start:b_end] = np.mean(cln_block, axis=(1, 2))
            
            # Max spatial gradients
            raw_dx = np.max(np.abs(np.diff(raw_block, axis=2)), axis=(1, 2))
            raw_dy = np.max(np.abs(np.diff(raw_block, axis=1)), axis=(1, 2))
            raw_max_grad[b_start:b_end] = np.maximum(raw_dx, raw_dy)
            
            cln_dx = np.max(np.abs(np.diff(cln_block, axis=2)), axis=(1, 2))
            cln_dy = np.max(np.abs(np.diff(cln_block, axis=1)), axis=(1, 2))
            cln_max_grad[b_start:b_end] = np.maximum(cln_dx, cln_dy)
            
            # Capillary energy (in Joules, converting to meters)
            raw_m = raw_block.astype(np.float64) * 1e-6
            cln_m = cln_block.astype(np.float64) * 1e-6
            
            q_r, e_r = field_energies(raw_m, x_m, y_m, geometry="surface")
            q_c, e_c = field_energies(cln_m, x_m, y_m, geometry="surface")
            
            raw_energy_quad[b_start:b_end] = q_r * GAMMA_WATER
            raw_energy_exact[b_start:b_end] = e_r * GAMMA_WATER
            cln_energy_quad[b_start:b_end] = q_c * GAMMA_WATER
            cln_energy_exact[b_start:b_end] = e_c * GAMMA_WATER
            
            if (b + 1) % 20 == 0 or (b + 1) == n_batches:
                elapsed = time.perf_counter() - started
                fps = b_end / elapsed
                print(f"  Processed {b_end:,}/{n_frames:,} frames ({(b_end/n_frames)*100:.1f}%) | {fps:.1f} fps")
                
    # Compute Welch PSDs
    print("\nComputing Welch PSDs (nperseg=8192, 50% overlap)...")
    freq, psd_raw_c = compute_welch_psd(raw_center, fs=fs, nperseg=8192)
    _, psd_cln_c = compute_welch_psd(cln_center, fs=fs, nperseg=8192)
    _, psd_raw_m = compute_welch_psd(raw_mean, fs=fs, nperseg=8192)
    _, psd_cln_m = compute_welch_psd(cln_mean, fs=fs, nperseg=8192)
    
    # Render PSD Comparison Figure
    print("Plotting PSD comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    pos_c = (freq > 0) & (psd_raw_c > 0) & (psd_cln_c > 0)
    fc = freq[pos_c]
    pr_c = psd_raw_c[pos_c]
    pc_c = psd_cln_c[pos_c]
    
    # 1. Center PSD
    axes[0, 0].loglog(fc, pr_c, color="crimson", lw=1.2, label="Original Raw", alpha=0.85)
    axes[0, 0].loglog(fc, pc_c, color="forestgreen", lw=1.2, label="Cleaned (Integer tau=4.0 + Temporal DC Locked)", alpha=0.9)
    axes[0, 0].set_title("Center-Point Elevation PSD $S_{hh}(f)$", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Frequency [Hz]")
    axes[0, 0].set_ylabel(r"PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]")
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.35)
    axes[0, 0].legend(fontsize=10)
    
    # 2. Ratio
    axes[0, 1].semilogx(fc, pc_c / pr_c, color="teal", lw=1.2)
    axes[0, 1].axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.7)
    axes[0, 1].set_title("Center PSD Ratio: Cleaned / Raw", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Frequency [Hz]")
    axes[0, 1].set_ylabel("Power Ratio")
    axes[0, 1].set_ylim(0.0, 1.5)
    axes[0, 1].grid(True, which="both", ls="--", alpha=0.35)
    
    # 3. Spatial Mean PSD
    pos_m = (freq > 0) & (psd_raw_m > 0) & (psd_cln_m > 0)
    axes[1, 0].loglog(freq[pos_m], psd_raw_m[pos_m], color="crimson", lw=1.2, label="Original Raw (Phase Slips)", alpha=0.85)
    axes[1, 0].loglog(freq[pos_m], psd_cln_m[pos_m], color="forestgreen", lw=1.2, label="Cleaned (Integer + Temporal DC Locked)", alpha=0.9)
    axes[1, 0].set_title("Spatial-Mean Height PSD $S_{\\bar{h}\\bar{h}}(f)$", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel(r"PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]")
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.35)
    axes[1, 0].legend(fontsize=10)
    
    # 4. Zoomed Turbulent Cascade
    peak = (fc >= 30) & (fc <= 2500)
    axes[1, 1].loglog(fc[peak], pr_c[peak], color="crimson", lw=1.3, label="Original Raw", alpha=0.85)
    axes[1, 1].loglog(fc[peak], pc_c[peak], color="forestgreen", lw=1.3, label="Cleaned", alpha=0.9)
    axes[1, 1].set_title("Turbulent Energy Cascade Band (30 Hz - 2.5 kHz)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].set_ylabel(r"PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]")
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.35)
    axes[1, 1].legend(fontsize=10)
    
    psd_fig_path = OUTPUT_DIR / "psd_comparison.png"
    fig.savefig(psd_fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved {psd_fig_path.name}")
    
    # Render Energy Comparison Figure
    print("Plotting energy comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    raw_e_pj = raw_energy_quad * 1e12
    cln_e_pj = cln_energy_quad * 1e12
    t_ms = t_arr * 1e3
    
    axes[0, 0].plot(t_ms, raw_e_pj, color="crimson", lw=0.6, alpha=0.7, label="Original Raw")
    axes[0, 0].plot(t_ms, cln_e_pj, color="forestgreen", lw=0.8, alpha=0.85, label="Cleaned")
    axes[0, 0].set_title("Full Capillary Energy Time Series (Quad)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Time [ms]")
    axes[0, 0].set_ylabel("Capillary Energy [pJ]")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    # Zoom
    w = slice(1500, 1600)
    axes[0, 1].plot(t_ms[w], raw_e_pj[w], color="crimson", lw=1.5, marker="o", ms=3, label="Original Raw")
    axes[0, 1].plot(t_ms[w], cln_e_pj[w], color="forestgreen", lw=1.5, marker="s", ms=3, label="Cleaned")
    axes[0, 1].set_title("Energy Zoom Around Frame 1553 Storm", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Time [ms]")
    axes[0, 1].set_ylabel("Capillary Energy [pJ]")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, which="both", ls="--", alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    
    # CDF
    bins = np.logspace(np.log10(max(1.0, cln_e_pj.min())), np.log10(raw_e_pj.max()), 100)
    axes[1, 0].hist(raw_e_pj, bins=bins, density=True, cumulative=True, histtype="step", color="crimson", lw=1.8, label="Raw")
    axes[1, 0].hist(cln_e_pj, bins=bins, density=True, cumulative=True, histtype="step", color="forestgreen", lw=1.8, label="Cleaned")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Capillary Energy CDF", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Energy [pJ]")
    axes[1, 0].set_ylabel("Cumulative Probability")
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    
    # Increment distributions of spatial mean
    d_raw_m = np.abs(np.diff(raw_mean))
    d_cln_m = np.abs(np.diff(clean_mean if "clean_mean" in locals() else cln_mean))
    bins_inc = np.linspace(0, 10, 100)
    axes[1, 1].hist(d_raw_m, bins=bins_inc, density=True, histtype="step", color="crimson", lw=1.8, label=f"Raw (Max={np.max(d_raw_m):.1f} um)")
    axes[1, 1].hist(d_cln_m, bins=bins_inc, density=True, histtype="step", color="forestgreen", lw=1.8, label=f"Cleaned (Max={np.max(d_cln_m):.1f} um)")
    axes[1, 1].axvline(1.9968, color="darkorange", ls="--", lw=1.2, label=r"Phase Wrap $q \approx 2.0\,\mu$m")
    axes[1, 1].set_title(r"Spatial Mean Frame-to-Frame Increment $|\Delta \langle h \rangle|$", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel(r"$|\Delta \langle h \rangle|$ [$\mu$m]")
    axes[1, 1].set_ylabel("Probability Density")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    
    energy_fig_path = OUTPUT_DIR / "energy_comparison.png"
    fig.savefig(energy_fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved {energy_fig_path.name}")
    
    # Save NPZ and JSON
    metrics_npz = OUTPUT_DIR / "comparison_metrics.npz"
    np.savez_compressed(
        metrics_npz,
        time=t_arr,
        raw_center=raw_center,
        cln_center=cln_center,
        raw_mean=raw_mean,
        cln_mean=cln_mean,
        raw_energy_quad=raw_energy_quad,
        raw_energy_exact=raw_energy_exact,
        cln_energy_quad=cln_energy_quad,
        cln_energy_exact=cln_energy_exact,
        raw_max_grad=raw_max_grad,
        cln_max_grad=cln_max_grad,
        freq=freq,
        psd_raw_center=psd_raw_c,
        psd_cln_center=psd_cln_c,
        psd_raw_mean=psd_raw_m,
        psd_cln_mean=psd_cln_m,
    )
    print(f"Saved {metrics_npz.name}")
    
    summary = {
        "dataset": str(RAW_H5),
        "total_frames": n_frames,
        "sampling_frequency_hz": float(fs),
        "tau_microns": 4.0,
        "method": "integer_tau_4.0_temporal_dc_locked",
        "spatial_mean_increment": {
            "raw_max_um": float(np.max(d_raw_m)),
            "cln_max_um": float(np.max(d_cln_m)),
            "raw_p99_9_um": float(np.percentile(d_raw_m, 99.9)),
            "cln_p99_9_um": float(np.percentile(d_cln_m, 99.9)),
            "cln_median_um": float(np.median(d_cln_m)),
            "max_increment_reduction_pct": float(100.0 * (1.0 - np.max(d_cln_m) / np.max(d_raw_m))),
        },
        "capillary_energy_quad_pJ": {
            "raw_mean": float(np.mean(raw_e_pj)),
            "raw_median": float(np.median(raw_e_pj)),
            "raw_max": float(np.max(raw_e_pj)),
            "cln_mean": float(np.mean(cln_e_pj)),
            "cln_median": float(np.median(cln_e_pj)),
            "cln_max": float(np.max(cln_e_pj)),
            "mean_energy_reduction_pct": float(100.0 * (1.0 - np.mean(cln_e_pj) / np.mean(raw_e_pj))),
        },
        "spatial_gradients_um": {
            "raw_max": float(np.max(raw_max_grad)),
            "raw_p99": float(np.percentile(raw_max_grad, 99)),
            "cln_max": float(np.max(cln_max_grad)),
            "cln_p99": float(np.percentile(cln_max_grad, 99)),
        },
    }
    
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path.name}")
    
    total_elapsed = time.perf_counter() - started
    print(f"\nAll metrics and plots computed successfully in {total_elapsed:.1f} s ({n_frames/total_elapsed:.1f} fps)!")


if __name__ == "__main__":
    main()

