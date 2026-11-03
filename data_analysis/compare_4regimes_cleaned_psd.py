"""Compute and compare high-resolution Power Spectral Densities (PSD) across all 4 cleaned regimes.

Regimes analyzed:
1. 0.04 Vpp (Weak forcing)
2. 0.08 Vpp (Intermediate forcing)
3. 0.20 Vpp (Turbulent forcing)
4. 0.35 Vpp (Strong turbulent forcing)

Computes:
- Surface elevation PSD (spatially averaged across 16 distributed probes)
- Center probe PSD
- Spatial mean PSD
- Turbulent inertial cascade power-law fitting S(f) ~ f^(-alpha) vs Zakharov-Filonenko theory (f^-2.83)
- Spectral energy scaling with forcing amplitude
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent BLAS thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

DATASETS = [
    {
        "label": "0.04 Vpp",
        "power": 0.04,
        "color": "#1f77b4",
        "path": Path("/home/jonas/ucsd_thesis/11222025_j_0.04vpp_data_roi-none_cal-true.hdf5"),
        "already_highpassed": False,
    },
    {
        "label": "0.08 Vpp",
        "power": 0.08,
        "color": "#2ca02c",
        "path": Path("/home/jonas/ucsd_thesis/11222025_h_0.08vpp_data_roi-none_cal-true.hdf5"),
        "already_highpassed": False,
    },
    {
        "label": "0.20 Vpp",
        "power": 0.20,
        "color": "#ff7f0e",
        "path": Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5"),
        "already_highpassed": True,
    },
    {
        "label": "0.35 Vpp",
        "power": 0.35,
        "color": "#d62728",
        "path": Path("/home/jonas/ucsd_thesis/11252025_e_0.35vpp_data_roi-none_cal-true_cleaned_tau4.hdf5"),
        "already_highpassed": False,
    },
]

OUTPUT_DIR = Path("runs/psd_4regimes_comparison")


def extract_spectral_probes(
    h5_path: Path,
    already_highpassed: bool = False,
    cutoff_hz: float = 50.0,
    grid_probes: int = 4,
) -> dict[str, Any]:
    """Extract spatial mean, center probe, and a 4x4 spatial probe grid for spectral analysis."""
    started = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        meta = f["meta"]
        time_arr = np.asarray(meta["t"], dtype=np.float64)
        n_frames = len(time_arr)
        dt = float(np.mean(np.diff(time_arr)))
        fs = 1.0 / dt
        
        main_grp = f["main"]
        first_frame = np.asarray(main_grp["00000"], dtype=np.float32)
        ny, nx = first_frame.shape
        
        # 16 distributed spatial probes: 4x4 grid in inner 75% of domain
        y_pts = np.linspace(25, ny - 25, grid_probes, dtype=int)
        x_pts = np.linspace(25, nx - 25, grid_probes, dtype=int)
        probe_coords = [(y, x) for y in y_pts for x in x_pts]
        n_probes = len(probe_coords)
        
        cx, cy = nx // 2, ny // 2
        
        print(f"Reading {h5_path.name} ({n_frames:,} frames, {n_probes} probes + mean)...")
        means = np.empty(n_frames, dtype=np.float32)
        center = np.empty(n_frames, dtype=np.float32)
        probes = np.empty((n_frames, n_probes), dtype=np.float32)
        
        report_step = n_frames // 4
        for i in range(n_frames):
            frame = np.asarray(main_grp[f"{i:05d}"], dtype=np.float32)
            means[i] = float(np.mean(frame))
            center[i] = frame[cy, cx]
            for p_idx, (py, px) in enumerate(probe_coords):
                probes[i, p_idx] = frame[py, px]
            if (i + 1) % report_step == 0 or (i + 1) == n_frames:
                print(f"  Read {i+1:,}/{n_frames:,} frames ({(i+1)/n_frames*100:.0f}%, {time.perf_counter()-started:.1f} s)")

    # 50 Hz high-pass filter if not already filtered
    if not already_highpassed:
        print(f"  Applying 50 Hz zero-phase high-pass filter...")
        sos = butter(4, cutoff_hz, btype="highpass", fs=fs, output="sos")
        means_hp = sosfiltfilt(sos, means.astype(np.float64)).astype(np.float32)
        center_hp = sosfiltfilt(sos, center.astype(np.float64)).astype(np.float32)
        probes_hp = sosfiltfilt(sos, probes.astype(np.float64), axis=0).astype(np.float32)
    else:
        print(f"  Dataset is already 50 Hz high-passed.")
        means_hp = means
        center_hp = center
        probes_hp = probes

    # Welch PSD computation
    nperseg = 16384  # high spectral resolution: df ~ 7 Hz
    f_psd, psd_mean = welch(means_hp, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_center = welch(center_hp, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    
    # Average PSD across all 16 spatial probes
    psd_probes = np.empty((n_probes, len(f_psd)), dtype=np.float64)
    for p_idx in range(n_probes):
        _, psd_probes[p_idx] = welch(probes_hp[:, p_idx], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    psd_surface_avg = np.mean(psd_probes, axis=0)
    
    # Integrate total wave variance: sigma^2 = int_{50}^{5000} S(f) df
    band = (f_psd >= 50.0) & (f_psd <= 5000.0)
    df = f_psd[1] - f_psd[0]
    total_var = float(np.sum(psd_surface_avg[band]) * df)
    rms_height = float(np.sqrt(total_var))
    
    # Fit inertial range power-law: S(f) ~ f^(-alpha) between 250 Hz and 1200 Hz
    fit_band = (f_psd >= 250.0) & (f_psd <= 1200.0)
    log_f = np.log10(f_psd[fit_band])
    log_s = np.log10(psd_surface_avg[fit_band])
    fit_coeffs = np.polyfit(log_f, log_s, 1)
    alpha = -fit_coeffs[0]
    
    print(f"  Extracted successfully in {time.perf_counter() - started:.1f} s:")
    print(f"    Total Wave RMS: {rms_height:.2f} um | Fitted Cascade Exponent alpha: {alpha:.2f}")

    return {
        "fs": fs,
        "f_psd": f_psd,
        "psd_mean": psd_mean,
        "psd_center": psd_center,
        "psd_surface_avg": psd_surface_avg,
        "total_var": total_var,
        "rms_height": rms_height,
        "cascade_alpha": alpha,
        "cascade_intercept": fit_coeffs[1],
    }


def plot_psd_comparison_suite(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # FIGURE 1: Cleaned Surface Elevation PSD Across 4 Regimes
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    
    # 1. Absolute Surface Elevation PSD (Space-Averaged)
    ax1 = axes1[0, 0]
    for r in results:
        f = r["f_psd"]
        s = r["psd_surface_avg"]
        ax1.loglog(f, s, color=r["color"], lw=1.8, label=f"{r['label']} (RMS = {r['rms_height']:.2f} $\\mu$m)")
    ax1.axvline(50, color="gray", ls=":", lw=1.2, label="50 Hz High-Pass Cutoff")
    ax1.set_xlim(30, 4000)
    ax1.set_ylim(1e-6, 1e5)
    ax1.set_title("A. Spatially-Averaged Surface Elevation PSD $\\langle S_\\eta(f) \\rangle$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
    ax1.set_ylabel("Power Spectral Density [$\\mu$m$^2$/Hz]", fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.35)
    ax1.legend(loc="lower left", fontsize=10.0)
    
    # 2. Normalized PSD (Divided by Total Variance)
    ax2 = axes1[0, 1]
    for r in results:
        f = r["f_psd"]
        s_norm = r["psd_surface_avg"] / r["total_var"]
        ax2.loglog(f, s_norm, color=r["color"], lw=1.8, label=f"{r['label']} ($\\alpha={r['cascade_alpha']:.2f}$)")
    # Overlay theoretical Zakharov-Filonenko slope: f^(-17/6) = f^(-2.83)
    f_theory = np.linspace(250, 1500, 100)
    s_theory = 1e-4 * (f_theory / 250.0)**(-17.0 / 6.0)
    ax2.loglog(f_theory, s_theory, "k--", lw=2.2, label="Theory: $f^{-17/6} \\approx f^{-2.83}$ (ZF)")
    ax2.set_xlim(30, 4000)
    ax2.set_ylim(1e-10, 1e-2)
    ax2.set_title("B. Normalized Spectra $S_\\eta(f) / \\sigma_\\eta^2$ & Universal Cascade Scaling", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
    ax2.set_ylabel("Normalized Spectral Density [1/Hz]", fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.35)
    ax2.legend(loc="lower left", fontsize=9.5)
    
    # 3. Spatial Mean PSD
    ax3 = axes1[1, 0]
    for r in results:
        f = r["f_psd"]
        s_m = r["psd_mean"]
        ax3.semilogy(f, s_m, color=r["color"], lw=1.8, label=f"{r['label']} Mean")
    ax3.axvline(140, color="purple", ls=":", lw=1.2, label="140 Hz Bulk Resonance")
    ax3.axvline(197, color="purple", ls=":", lw=1.2, label="197 Hz Resonance")
    ax3.set_xlim(30, 2000)
    ax3.set_ylim(1e-5, 1e5)
    ax3.set_title("C. Spatial Mean PSD $S_{\\langle h \\rangle}(f)$ — Pure Bulk Fluid Resonances", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
    ax3.set_ylabel("Power Spectral Density [$\\mu$m$^2$/Hz]", fontsize=11)
    ax3.grid(True, which="both", ls="--", alpha=0.35)
    ax3.legend(loc="upper right", fontsize=9.5)
    
    # 4. Total Turbulent Energy Scaling vs Forcing Amplitude
    ax4 = axes1[1, 1]
    powers = np.array([r["power"] for r in results])
    variances = np.array([r["total_var"] for r in results])
    rms_vals = np.array([r["rms_height"] for r in results])
    
    ax4.loglog(powers, variances, "ko-", lw=2.0, ms=8, label="Turbulent Surface Variance $\\sigma_\\eta^2$ [$\\mu$m$^2$]")
    # Fit power law scaling: sigma^2 ~ V_pp^beta
    log_p = np.log10(powers)
    log_v = np.log10(variances)
    slope, intercept = np.polyfit(log_p, log_v, 1)
    p_fit = np.linspace(0.035, 0.40, 50)
    v_fit = 10**intercept * p_fit**slope
    ax4.loglog(p_fit, v_fit, "r--", lw=1.8, label=f"Scaling Fit: $\\sigma_\\eta^2 \\propto V_{{\\mathrm{{pp}}}}^{{{slope:.2f}}}$")
    
    ax4.set_title("D. Turbulent Wave Energy Scaling with Forcing Amplitude", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Forcing Amplitude $V_{\\mathrm{pp}}$ [V]", fontsize=11)
    ax4.set_ylabel("Surface Elevation Variance $\\sigma_\\eta^2$ [$\\mu$m$^2$]", fontsize=11)
    ax4.grid(True, which="both", ls="--", alpha=0.35)
    ax4.legend(loc="lower right", fontsize=10.0)
    for i, r in enumerate(results):
        ax4.annotate(f"{r['label']}\n$\\sigma={rms_vals[i]:.1f}\\mu$m",
                     (powers[i], variances[i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9.5, fontweight="bold")
                     
    fig1_path = output_dir / "cleaned_psd_4regimes_surface_elevation.png"
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)
    print(f"Saved {fig1_path}")

    # =========================================================================
    # FIGURE 2: Compensated Wave Turbulence Cascade Spectrum f^(17/6) * S(f)
    # =========================================================================
    fig2, ax_comp = plt.subplots(figsize=(10, 7), constrained_layout=True)
    zf_power = 17.0 / 6.0  # 2.8333
    for r in results:
        f = r["f_psd"]
        s = r["psd_surface_avg"]
        mask = (f >= 100.0) & (f <= 2500.0)
        # Compensated spectrum: f^(17/6) * S(f)
        s_comp = (f[mask]**zf_power) * s[mask]
        ax_comp.loglog(f[mask], s_comp, color=r["color"], lw=2.0, label=f"{r['label']} (Fitted $\\alpha = {r['cascade_alpha']:.2f}$)")
        
    ax_comp.set_title("Compensated Capillary Wave Turbulence Cascade Spectrum $f^{17/6} S_\\eta(f)$", fontsize=13, fontweight="bold")
    ax_comp.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
    ax_comp.set_ylabel("Compensated Power $f^{17/6} S_\\eta(f)$ [$\\mu$m$^2$ Hz$^{11/6}$]", fontsize=11)
    ax_comp.grid(True, which="both", ls="--", alpha=0.35)
    ax_comp.legend(loc="upper left", fontsize=10.5)
    
    fig2_path = output_dir / "cleaned_psd_4regimes_cascade_scaling.png"
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)
    print(f"Saved {fig2_path}")


def main() -> None:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    for d in DATASETS:
        res = extract_spectral_probes(
            d["path"],
            already_highpassed=d["already_highpassed"],
            cutoff_hz=50.0,
            grid_probes=4,
        )
        res["label"] = d["label"]
        res["power"] = d["power"]
        res["color"] = d["color"]
        results.append(res)
        
    plot_psd_comparison_suite(results, OUTPUT_DIR)
    
    # Save metrics table
    csv_path = OUTPUT_DIR / "psd_4regimes_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["regime", "forcing_vpp", "rms_surface_um", "total_var_um2", "cascade_slope_alpha"])
        for r in results:
            writer.writerow([r["label"], r["power"], f"{r['rms_height']:.3f}", f"{r['total_var']:.3f}", f"{r['cascade_alpha']:.3f}"])
    print(f"Saved {csv_path}")
    print(f"All 4-regime PSD analyses complete in {time.perf_counter() - t_start:.1f} s.")


if __name__ == "__main__":
    main()

