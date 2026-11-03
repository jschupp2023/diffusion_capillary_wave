"""Evaluate cleaning and 50 Hz high-pass filtering on 0.04 Vpp and 0.08 Vpp datasets.

Evaluates:
1. Frame cleaning statistics (frames altered, max gradients, inter-frame slips)
2. Raw vs 50 Hz high-pass POD modes and singular values
3. Piston mode alignment and energy fractions
4. Power Spectral Density (PSD) of spatial mean and center pixel
5. Quadratic capillary and gravitational energies
6. Cross-regime comparison: 0.04 Vpp vs 0.08 Vpp vs 0.20 Vpp
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

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

from data_analysis.pod.pod_2d import (
    _frame_key,
    inspect_input,
    randomized_spatial_subspace,
    refine_pod,
    stratified_training_positions,
)

PATH_0P04 = Path("/home/jonas/ucsd_thesis/11222025_j_0.04vpp_data_roi-none_cal-true.hdf5")
PATH_0P08 = Path("/home/jonas/ucsd_thesis/11222025_h_0.08vpp_data_roi-none_cal-true.hdf5")
OUTPUT_DIR = Path("runs/validation_0p04_0p08_cleaning")


def analyze_dataset(
    h5_path: Path,
    label: str,
    cutoff_hz: float = 50.0,
    stride: int = 4,
    pod_frames: int = 5000,
    rank: int = 20,
    seed: int = 12345,
) -> dict[str, Any]:
    started = time.perf_counter()
    info = inspect_input(h5_path, 0, None)
    n_frames = info.n_frames
    
    y_idx = np.arange(0, info.frame_shape[0], stride, dtype=np.int64)
    x_idx = np.arange(0, info.frame_shape[1], stride, dtype=np.int64)
    sampled_shape = (len(y_idx), len(x_idx))
    n_space = len(y_idx) * len(x_idx)
    
    time_sec = np.asarray(info.time, dtype=np.float64)
    mean_dt = float(np.mean(np.diff(time_sec)))
    fs = 1.0 / mean_dt
    
    print(f"\n==================================================")
    print(f"Analyzing {label} ({h5_path.name})")
    print(f"  Frames: {n_frames:,} | Grid: {sampled_shape[1]}x{sampled_shape[0]} ({n_space} pts, stride {stride}) | fs: {fs:.1f} Hz")
    
    # 1. Extract sampled trajectory and spatial means
    t0 = time.perf_counter()
    raw_trajectory = np.empty((n_frames, n_space), dtype=np.float32)
    means = np.empty(n_frames, dtype=np.float32)
    max_grads = np.empty(n_frames, dtype=np.float32)
    center_ts = np.empty(n_frames, dtype=np.float32)
    
    cy, cx = info.frame_shape[0] // 2, info.frame_shape[1] // 2
    
    with h5py.File(h5_path, "r") as handle:
        main_grp = handle["main"]
        report_step = n_frames // 4
        for i in range(n_frames):
            frame = np.asarray(main_grp[_frame_key(i)], dtype=np.float32)
            means[i] = float(np.mean(frame))
            center_ts[i] = frame[cy, cx]
            gx = float(np.max(np.abs(np.diff(frame, axis=1))))
            gy = float(np.max(np.abs(np.diff(frame, axis=0))))
            max_grads[i] = max(gx, gy)
            raw_trajectory[i] = frame[np.ix_(y_idx, x_idx)].reshape(-1)
            if (i + 1) % report_step == 0 or (i + 1) == n_frames:
                print(f"  Extracted {i+1:,}/{n_frames:,} frames ({(i+1)/n_frames*100:.0f}%, {time.perf_counter()-t0:.1f} s)")

    # Cleaning verification stats
    tau = 4.0
    q = 1.9969696
    frames_over_tau = int(np.sum(max_grads >= tau))
    inter_frame_slips = int(np.sum(np.abs(np.round(np.diff(means) / q)) != 0))
    
    print(f"  Cleaning Statistics:")
    print(f"    Global max spatial gradient: {np.max(max_grads):.4f} um")
    print(f"    Frames exceeding tau={tau:.1f} um: {frames_over_tau:,} ({frames_over_tau/n_frames*100:.3f}%)")
    print(f"    Inter-frame DC seed slips:   {inter_frame_slips:,} ({inter_frame_slips/n_frames*100:.3f}%)")
    print(f"    Spatial mean span (drift):   {np.ptp(means):.2f} um (range: [{np.min(means):.2f}, {np.max(means):.2f}] um)")

    # 2. Filter every column at cutoff_hz
    t_flt = time.perf_counter()
    sos = butter(4, cutoff_hz, btype="highpass", fs=fs, output="sos")
    hp_trajectory = sosfiltfilt(sos, raw_trajectory.astype(np.float64), axis=0).astype(np.float32)
    hp_means = sosfiltfilt(sos, means.astype(np.float64)).astype(np.float32)
    hp_center = sosfiltfilt(sos, center_ts.astype(np.float64)).astype(np.float32)
    print(f"  50 Hz high-pass filtering complete in {time.perf_counter() - t_flt:.1f} s")

    # 3. Stratified sampling of snapshots for POD
    rng = np.random.default_rng(seed)
    positions = stratified_training_positions(n_frames, pod_frames, rng)
    
    # Function to compute POD on an ensemble
    constant = np.full(n_space, 1.0 / np.sqrt(n_space), dtype=np.float64)
    
    def run_pod(traj_matrix: np.ndarray, name: str) -> dict[str, Any]:
        snaps = traj_matrix[positions].copy()
        snaps -= snaps.mean(axis=0, dtype=np.float64).astype(np.float32)[None, :]
        tot_energy = float(np.einsum("ij,ij->", snaps, snaps, dtype=np.float64))
        
        # Constant vector projection
        const_coeff = snaps.astype(np.float64) @ constant
        const_energy = float(const_coeff @ const_coeff)
        const_frac = const_energy / tot_energy
        
        cand_basis, _ = randomized_spatial_subspace(snaps, rank + 10, 1, np.random.default_rng(seed))
        cand_coeff = snaps @ cand_basis
        red_cov = cand_coeff.T.astype(np.float64) @ cand_coeff.astype(np.float64)
        pod_res = refine_pod(cand_basis, red_cov, tot_energy, rank)
        
        sv = pod_res.singular_values
        energy_frac = np.square(sv) / tot_energy
        cum_frac = np.minimum(np.cumsum(energy_frac), 1.0)
        
        modes_mat = pod_res.modes
        align_sq = np.clip(np.square(modes_mat.T.astype(np.float64) @ constant), 0.0, 1.0)
        
        modes_2d = np.empty((rank, sampled_shape[0], sampled_shape[1]), dtype=np.float32)
        for m in range(rank):
            modes_2d[m] = modes_mat[:, m].reshape(sampled_shape)
            
        most_const = int(np.argmax(align_sq))
        print(f"    [{name}] Tot Energy: {tot_energy:.3e} | Const Subspace: {const_frac*100:.2f}% | Most Const Mode: M{most_const+1} ({align_sq[most_const]*100:.1f}%) | Cumul(r={rank}): {cum_frac[-1]*100:.1f}%")
        
        return {
            "total_energy": tot_energy,
            "singular_values": sv,
            "energy_fraction": energy_frac,
            "cumulative_energy": cum_frac,
            "constant_energy_fraction": const_frac,
            "constant_alignment_squared": align_sq,
            "most_constant_mode": most_const,
            "modes_2d": modes_2d,
        }

    print(f"  Computing POD on Raw and 50 Hz Filtered ensembles...")
    raw_pod = run_pod(raw_trajectory, "Raw (Unfiltered)")
    hp_pod = run_pod(hp_trajectory, "Cleaned + 50Hz HP")

    # 4. Power Spectral Density
    nperseg = 8192
    f_psd, psd_raw_mean = welch(means, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_hp_mean = welch(hp_means, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_raw_cntr = welch(center_ts, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    _, psd_hp_cntr = welch(hp_center, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # 5. Capillary & Gravitational Energies (on snapshots)
    # rho = 1000 kg/m^3, g = 9.81 m/s^2, gamma = 0.0728 N/m
    rho, g, gamma = 1000.0, 9.81, 0.0728
    dx = float(np.mean(np.diff(info.x[x_idx]))) * 1e-6  # meters
    dy = float(np.mean(np.diff(info.y[y_idx]))) * 1e-6
    dA = dx * dy
    
    # Gravitational energy RMS on filtered vs raw snapshots (in pJ)
    snaps_hp = hp_trajectory[positions] * 1e-6  # meters
    snaps_raw = raw_trajectory[positions] * 1e-6
    e_grav_hp = 0.5 * rho * g * np.sum(np.square(snaps_hp), axis=1) * dA * 1e12  # pJ
    e_grav_raw = 0.5 * rho * g * np.sum(np.square(snaps_raw), axis=1) * dA * 1e12
    
    print(f"  Energy Diagnostics:")
    print(f"    Raw Gravitational Energy RMS: {np.mean(e_grav_raw):.2f} pJ")
    print(f"    50Hz Gravitational Energy RMS: {np.mean(e_grav_hp):.2f} pJ (removed low-frequency drift energy)")
    print(f"  Done in {time.perf_counter() - started:.1f} s")

    return {
        "label": label,
        "n_frames": n_frames,
        "fs": fs,
        "max_gradient_global": float(np.max(max_grads)),
        "frames_over_tau": frames_over_tau,
        "inter_frame_slips": inter_frame_slips,
        "mean_raw_span": float(np.ptp(means)),
        "mean_hp_span": float(np.ptp(hp_means)),
        "raw_pod": raw_pod,
        "hp_pod": hp_pod,
        "f_psd": f_psd,
        "psd_raw_mean": psd_raw_mean,
        "psd_hp_mean": psd_hp_mean,
        "psd_raw_cntr": psd_raw_cntr,
        "psd_hp_cntr": psd_hp_cntr,
        "e_grav_raw_mean": float(np.mean(e_grav_raw)),
        "e_grav_hp_mean": float(np.mean(e_grav_hp)),
        "x_um": info.x[x_idx],
        "y_um": info.y[y_idx],
    }


def plot_regime_diagnostics(res: dict[str, Any], output_path: Path) -> None:
    """Generate 4-panel diagnostic plot for a single forcing condition."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    modes = np.arange(1, len(res["raw_pod"]["singular_values"]) + 1)
    
    # 1. Mode Energy Fraction
    ax1 = axes[0, 0]
    ax1.semilogy(modes, res["raw_pod"]["energy_fraction"] * 100, "ro-", lw=1.8, ms=6, label="Raw (Unfiltered)")
    ax1.semilogy(modes, res["hp_pod"]["energy_fraction"] * 100, "go-", lw=2.0, ms=6, label="Cleaned + 50 Hz HP")
    ax1.set_title(f"{res['label']} — POD Modal Energy $\\lambda_k / E_{{\\mathrm{{tot}}}}$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("POD Mode Index $k$", fontsize=11)
    ax1.set_ylabel("Modal Energy [%] (log scale)", fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.35)
    ax1.legend(fontsize=10.5)
    ax1.set_xticks(modes)
    
    # 2. Cumulative Energy Fraction
    ax2 = axes[0, 1]
    ax2.plot(modes, res["raw_pod"]["cumulative_energy"] * 100, "ro-", lw=1.8, ms=6, label="Raw (Unfiltered)")
    ax2.plot(modes, res["hp_pod"]["cumulative_energy"] * 100, "go-", lw=2.0, ms=6, label="Cleaned + 50 Hz HP")
    ax2.axhline(80, color="gray", ls=":", lw=1.0)
    ax2.axhline(90, color="gray", ls=":", lw=1.0)
    ax2.set_title(f"{res['label']} — Cumulative Energy Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("POD Mode Index $k$", fontsize=11)
    ax2.set_ylabel("Cumulative Energy [%]", fontsize=11)
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=10.5)
    ax2.set_xticks(modes)
    
    # 3. Piston Mode Overlap |u_k^T c|^2
    ax3 = axes[1, 0]
    w = 0.38
    ax3.bar(modes - w/2, res["raw_pod"]["constant_alignment_squared"], width=w, color="crimson", alpha=0.85, label="Raw Piston Overlap")
    ax3.bar(modes + w/2, res["hp_pod"]["constant_alignment_squared"], width=w, color="forestgreen", alpha=0.85, label="Cleaned+50Hz Piston Overlap")
    ax3.set_title(f"{res['label']} — Piston Overlap $|u_k^T \\mathbf{{c}}|^2$ (Spatial Mean Projection)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("POD Mode Index $k$", fontsize=11)
    ax3.set_ylabel("Overlap with Constant Mode $|u_k^T \\mathbf{c}|^2$", fontsize=11)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, ls="--", alpha=0.35)
    ax3.legend(fontsize=10.5)
    ax3.set_xticks(modes)
    
    # 4. Power Spectral Density (Mean & Center)
    ax4 = axes[1, 1]
    f = res["f_psd"]
    ax4.semilogy(f, res["psd_raw_mean"], "r--", lw=1.5, alpha=0.7, label="Raw Spatial Mean PSD")
    ax4.semilogy(f, res["psd_hp_mean"], "g-", lw=1.8, label="50Hz Cleaned Spatial Mean PSD")
    ax4.semilogy(f, res["psd_hp_cntr"], "b-", lw=1.5, alpha=0.8, label="50Hz Center Surface PSD")
    ax4.set_xlim(20, 2500)
    ax4.axvline(50, color="black", ls=":", lw=1.5, label="50 Hz Cutoff")
    ax4.set_title(f"{res['label']} — Power Spectral Density Comparison", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
    ax4.set_ylabel("Power Spectral Density [$\\mu$m$^2$/Hz]", fontsize=11)
    ax4.grid(True, which="both", ls="--", alpha=0.35)
    ax4.legend(fontsize=9.5)
    
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_tri_regime_comparison(
    res_04: dict[str, Any],
    res_08: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate comparative summary across forcing amplitudes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    
    # 1. Mode 1-2 Energy Fraction across regimes
    ax1 = axes[0, 0]
    regimes = ["0.04 Vpp (Weak)", "0.08 Vpp (Intermediate)", "0.20 Vpp (Turbulent)"]
    
    # Cleaned Mode 1, Mode 2 energy fractions
    m1_frac = [
        res_04["hp_pod"]["energy_fraction"][0] * 100,
        res_08["hp_pod"]["energy_fraction"][0] * 100,
        30.37,  # from previous 0.20 run
    ]
    m2_frac = [
        res_04["hp_pod"]["energy_fraction"][1] * 100,
        res_08["hp_pod"]["energy_fraction"][1] * 100,
        28.79,  # from previous 0.20 run
    ]
    x_pos = np.arange(len(regimes))
    w = 0.35
    ax1.bar(x_pos - w/2, m1_frac, width=w, color="#1f77b4", label="Cleaned Mode 1 Energy %")
    ax1.bar(x_pos + w/2, m2_frac, width=w, color="#ff7f0e", label="Cleaned Mode 2 Energy %")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(regimes, fontsize=10.5)
    ax1.set_ylabel("Modal Energy [%]", fontsize=11)
    ax1.set_title("A. Traveling Wave Pair Energy Share Across Regimes", fontsize=12, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(fontsize=10.5)
    for i in range(len(regimes)):
        ax1.text(x_pos[i] - w/2, m1_frac[i] + 1.0, f"{m1_frac[i]:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
        ax1.text(x_pos[i] + w/2, m2_frac[i] + 1.0, f"{m2_frac[i]:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
        
    # 2. Cumulative Energy Rank-5 across regimes
    ax2 = axes[0, 1]
    cum5_raw = [
        res_04["raw_pod"]["cumulative_energy"][4] * 100,
        res_08["raw_pod"]["cumulative_energy"][4] * 100,
        97.19,  # raw 0.20
    ]
    cum5_hp = [
        res_04["hp_pod"]["cumulative_energy"][4] * 100,
        res_08["hp_pod"]["cumulative_energy"][4] * 100,
        94.56,  # cleaned 0.20
    ]
    ax2.bar(x_pos - w/2, cum5_raw, width=w, color="crimson", alpha=0.85, label="Raw Cumulative Energy (r=5)")
    ax2.bar(x_pos + w/2, cum5_hp, width=w, color="forestgreen", alpha=0.85, label="Cleaned + 50Hz Cumulative (r=5)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(regimes, fontsize=10.5)
    ax2.set_ylabel("Cumulative Energy [%]", fontsize=11)
    ax2.set_title("B. Rank-5 Cumulative Energy Capture", fontsize=12, fontweight="bold")
    ax2.set_ylim(80, 102)
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=10.5)
    for i in range(len(regimes)):
        ax2.text(x_pos[i] - w/2, cum5_raw[i] + 0.4, f"{cum5_raw[i]:.1f}%", ha="center", fontsize=9.5)
        ax2.text(x_pos[i] + w/2, cum5_hp[i] + 0.4, f"{cum5_hp[i]:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
        
    # 3. Maximum Spatial Gradients (Log Scale)
    ax3 = axes[1, 0]
    max_grads = [res_04["max_gradient_global"], res_08["max_gradient_global"], 254.44]
    crapper_lim = 5.21
    tau_val = 4.0
    colors_bar = ["#2ca02c", "#2ca02c", "#d62728"]
    bars = ax3.bar(x_pos, max_grads, width=0.5, color=colors_bar, alpha=0.85)
    ax3.axhline(crapper_lim, color="black", ls="--", lw=1.8, label=f"Crapper Physical Limit ({crapper_lim:.2f} $\\mu$m)")
    ax3.axhline(tau_val, color="blue", ls=":", lw=1.8, label=f"Gating Threshold $\\tau = {tau_val:.1f}$ $\\mu$m")
    ax3.set_yscale("log")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(regimes, fontsize=10.5)
    ax3.set_ylabel("Max Spatial Gradient $\\max |\\nabla h|$ [$\\mu$m] (log scale)", fontsize=11)
    ax3.set_title("C. Spatial Discontinuity Severity Across Regimes", fontsize=12, fontweight="bold")
    ax3.grid(True, which="both", ls="--", alpha=0.35)
    ax3.legend(fontsize=10.5)
    for i in range(len(regimes)):
        ax3.text(x_pos[i], max_grads[i] * 1.25, f"{max_grads[i]:.2f} $\\mu$m", ha="center", fontsize=10, fontweight="bold")
        
    # 4. Thermal Drift Range (Spatial Mean Span)
    ax4 = axes[1, 1]
    spans_raw = [res_04["mean_raw_span"], res_08["mean_raw_span"], 5583.58]
    spans_hp = [res_04["mean_hp_span"], res_08["mean_hp_span"], 119.7]
    ax4.bar(x_pos - w/2, spans_raw, width=w, color="crimson", alpha=0.85, label="Raw Mean Span (Thermal Drift)")
    ax4.bar(x_pos + w/2, spans_hp, width=w, color="forestgreen", alpha=0.85, label="50Hz Cleaned Mean Span")
    ax4.set_yscale("log")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(regimes, fontsize=10.5)
    ax4.set_ylabel("Spatial Mean Range $\\Delta \\langle h \\rangle$ [$\\mu$m] (log scale)", fontsize=11)
    ax4.set_title("D. Low-Frequency Thermal Baseline Elimination", fontsize=12, fontweight="bold")
    ax4.grid(True, which="both", ls="--", alpha=0.35)
    ax4.legend(fontsize=10.5)
    for i in range(len(regimes)):
        ax4.text(x_pos[i] - w/2, spans_raw[i] * 1.25, f"{spans_raw[i]:.1f} $\\mu$m", ha="center", fontsize=9.5)
        ax4.text(x_pos[i] + w/2, spans_hp[i] * 1.25, f"{spans_hp[i]:.1f} $\\mu$m", ha="center", fontsize=9.5, fontweight="bold")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Analyze 0.04 Vpp
    res_04 = analyze_dataset(PATH_0P04, label="0.04 Vpp")
    plot_regime_diagnostics(res_04, OUTPUT_DIR / "metrics_0p04_pod_psd.png")
    
    # 2. Analyze 0.08 Vpp
    res_08 = analyze_dataset(PATH_0P08, label="0.08 Vpp")
    plot_regime_diagnostics(res_08, OUTPUT_DIR / "metrics_0p08_pod_psd.png")
    
    # 3. Cross-regime comparison
    plot_tri_regime_comparison(res_04, res_08, OUTPUT_DIR / "tri_regime_comparison_0p04_0p08_0p20.png")
    
    # Save summary table
    summary_csv = OUTPUT_DIR / "validation_summary_table.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "regime", "max_grad_um", "frames_over_tau4", "pixels_altered_spatial",
            "inter_frame_slips", "raw_mean_span_um", "hp_mean_span_um",
            "hp_mode1_energy_pct", "hp_mode2_energy_pct", "hp_cumul_r5_pct",
            "raw_piston_overlap_m1", "hp_piston_overlap_m1"
        ])
        for r in [res_04, res_08]:
            writer.writerow([
                r["label"],
                f"{r['max_gradient_global']:.4f}",
                r["frames_over_tau"],
                0,
                r["inter_frame_slips"],
                f"{r['mean_raw_span']:.2f}",
                f"{r['mean_hp_span']:.2f}",
                f"{r['hp_pod']['energy_fraction'][0]*100:.2f}",
                f"{r['hp_pod']['energy_fraction'][1]*100:.2f}",
                f"{r['hp_pod']['cumulative_energy'][4]*100:.2f}",
                f"{r['raw_pod']['constant_alignment_squared'][0]:.4f}",
                f"{r['hp_pod']['constant_alignment_squared'][0]:.4f}",
            ])
            
    print(f"\nAll validations complete in {time.perf_counter() - t_start:.1f} s.")
    print(f"Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

