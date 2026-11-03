"""Analyze and visualize 5-mode POD trajectories for the cleaned 0p20 dataset.

Focuses on dynamical systems and machine learning representation:
1. Trajectory time series at macro, meso, and micro scales
2. Phase space attractors (limit cycles, torus, invariant manifolds)
3. Dynamical learning targets: velocities, lag-1 increments, distributions, PSDs, autocorrelations
4. Direct comparison with raw projections around discontinuity episodes
"""

from __future__ import annotations

import csv
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

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from data_analysis.pod.pod_2d import _frame_key, inspect_input

RAW_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
CLEAN_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
POD_NPZ = Path("runs/pod_modes_comparison_50hz/pod_results.npz")
OUTPUT_DIR = Path("runs/pod_trajectories_cleaned_r5")


def load_pod_basis(pod_npz_path: Path) -> dict[str, np.ndarray]:
    """Load precomputed POD modes and singular values."""
    data = np.load(pod_npz_path)
    return {
        "cln_modes": data["cln_modes"],  # (20, 50, 50)
        "raw_modes": data["raw_modes"],  # (20, 50, 50)
        "cln_sv": data["cln_sv"],        # (20,)
        "raw_sv": data["raw_sv"],        # (20,)
    }


def extract_and_filter_trajectory(
    h5_path: Path,
    stride: int = 4,
    cutoff_hz: float = 50.0,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract downsampled spatial grid for frames and apply 50 Hz high-pass filter.
    
    Returns
    -------
    filtered_trajectory : np.ndarray, shape (n_frames, n_space)
    time_sec : np.ndarray, shape (n_frames,)
    fs : float
    """
    info = inspect_input(h5_path, 0, None)
    n_frames = info.n_frames if max_frames is None else min(info.n_frames, max_frames)
    
    y_idx = np.arange(0, info.frame_shape[0], stride, dtype=np.int64)
    x_idx = np.arange(0, info.frame_shape[1], stride, dtype=np.int64)
    n_space = len(y_idx) * len(x_idx)
    
    time_sec = np.asarray(info.time[:n_frames], dtype=np.float64)
    mean_dt = float(np.mean(np.diff(time_sec)))
    fs = 1.0 / mean_dt
    
    print(f"Reading {h5_path.name}: {n_frames:,} frames, grid {len(y_idx)}x{len(x_idx)} ({n_space} pts)...")
    t0 = time.perf_counter()
    raw_trajectory = np.empty((n_frames, n_space), dtype=np.float32)
    
    with h5py.File(h5_path, "r") as handle:
        main_grp = handle["main"]
        report_step = max(1, n_frames // 5)
        for i in range(n_frames):
            frame = np.asarray(main_grp[_frame_key(i)], dtype=np.float32)
            raw_trajectory[i] = frame[np.ix_(y_idx, x_idx)].reshape(-1)
            if (i + 1) % report_step == 0 or (i + 1) == n_frames:
                print(f"  Read {i+1:,}/{n_frames:,} frames ({(i+1)/n_frames*100:.0f}%, {time.perf_counter()-t0:.1f} s)")
                
    # Filter
    t_flt = time.perf_counter()
    sos = butter(4, cutoff_hz, btype="highpass", fs=fs, output="sos")
    filtered_trajectory = sosfiltfilt(sos, raw_trajectory.astype(np.float64), axis=0).astype(np.float32)
    print(f"  High-pass filtered in {time.perf_counter() - t_flt:.1f} s")
    
    return filtered_trajectory, time_sec, fs


def project_onto_modes(
    trajectory: np.ndarray,
    modes_2d: np.ndarray,
    rank: int = 5,
) -> np.ndarray:
    """Project trajectory (n_frames, n_space) onto top `rank` POD modes.
    
    modes_2d: (n_modes, ny, nx) where each flattened slice is orthonormal.
    Returns: coords (n_frames, rank) in physical units (um).
    """
    flat_basis = modes_2d[:rank].reshape(rank, -1).T  # (n_space, rank)
    coords = trajectory.astype(np.float64) @ flat_basis.astype(np.float64)
    return coords


def main() -> None:
    t_total = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load precomputed POD basis
    pod_data = load_pod_basis(POD_NPZ)
    cln_modes_5 = pod_data["cln_modes"][:5]  # (5, 50, 50)
    raw_modes_5 = pod_data["raw_modes"][:5]  # (5, 50, 50)
    
    # 2. Extract and filter trajectory for cleaned dataset
    # We use all 100,001 frames for full statistics and clean spectral analysis
    n_frames_to_process = 100001
    cln_traj, time_sec, fs = extract_and_filter_trajectory(CLEAN_H5, stride=4, cutoff_hz=50.0, max_frames=n_frames_to_process)
    
    # Also extract raw dataset for comparison around cliffs
    raw_traj, _, _ = extract_and_filter_trajectory(RAW_H5, stride=4, cutoff_hz=50.0, max_frames=n_frames_to_process)
    
    # 3. Project Cleaned trajectory onto Cleaned POD basis (the primary target)
    print("\nProjecting Cleaned dataset onto Cleaned POD Modes 1-5...")
    cln_coords = project_onto_modes(cln_traj, cln_modes_5, rank=5)  # (N, 5)
    
    # Also project Raw trajectory onto Cleaned POD basis and onto Raw POD basis for diagnostic comparison
    print("Projecting Raw dataset onto Cleaned POD Modes 1-5...")
    raw_on_cln_coords = project_onto_modes(raw_traj, cln_modes_5, rank=5)
    
    print("Projecting Raw dataset onto Raw POD Modes 1-5...")
    raw_coords = project_onto_modes(raw_traj, raw_modes_5, rank=5)
    
    dt = float(np.mean(np.diff(time_sec)))
    time_ms = time_sec * 1000.0  # in milliseconds
    
    # Standard deviations and standardizations
    std_cln = np.std(cln_coords, axis=0)
    std_raw = np.std(raw_coords, axis=0)
    z_cln = (cln_coords - np.mean(cln_coords, axis=0)) / std_cln
    z_raw = (raw_coords - np.mean(raw_coords, axis=0)) / std_raw
    
    print(f"\nCleaned POD Coordinate Standard Deviations (Physical units um):")
    for k in range(5):
        print(f"  Mode {k+1}: std = {std_cln[k]:.2f} um | RMS surface height equiv = {std_cln[k]/50.0:.2f} um")
        
    # Velocity field and increments
    diff_cln = np.diff(cln_coords, axis=0)  # (N-1, 5)
    vel_cln = diff_cln / dt                 # um/s
    diff_raw = np.diff(raw_on_cln_coords, axis=0)
    vel_raw = diff_raw / dt
    
    print(f"\nVelocity Statistics (um/s):")
    for k in range(5):
        print(f"  Mode {k+1} Cleaned: max = {np.max(np.abs(vel_cln[:, k])):.1f} um/s, std = {np.std(vel_cln[:, k]):.1f} um/s")
        print(f"  Mode {k+1} Raw:     max = {np.max(np.abs(vel_raw[:, k])):.1f} um/s, std = {np.std(vel_raw[:, k]):.1f} um/s")
        
    # =========================================================================
    # FIGURE 1: Multi-Scale Trajectory Overview
    # =========================================================================
    print("\nGenerating Figure 1: Multi-scale Trajectory Overview...")
    fig1 = plt.figure(figsize=(16, 13), constrained_layout=True)
    gs = fig1.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0])
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    labels = [
        "Mode 1 (Traveling wave A, $\\sigma_1=55.2\\mathrm{k}$)",
        "Mode 2 (Traveling wave B, $\\sigma_2=53.8\\mathrm{k}$)",
        "Mode 3 (Resonant breathing, $\\sigma_3=51.5\\mathrm{k}$)",
        "Mode 4 (Harmonic wave A, $\\sigma_4=21.8\\mathrm{k}$)",
        "Mode 5 (Harmonic wave B, $\\sigma_5=20.7\\mathrm{k}$)",
    ]
    
    # Panel A: Macro scale (10,000 frames = ~86.8 ms)
    ax_macro = fig1.add_subplot(gs[0])
    macro_slice = slice(1000, 11000)
    t_m = time_ms[macro_slice]
    for k in range(5):
        ax_macro.plot(t_m, cln_coords[macro_slice, k], color=colors[k], lw=1.0, alpha=0.85, label=labels[k])
    ax_macro.set_title("A. Macro-scale POD Coordinates $a_k(t)$ (10,000 frames $\\approx 86.8$ ms) — Turbulent Wavepacket Modulation", fontsize=12, fontweight="bold")
    ax_macro.set_ylabel("POD Amplitude $a_k(t)$ [$\\mu$m]", fontsize=11)
    ax_macro.grid(True, ls="--", alpha=0.35)
    ax_macro.legend(loc="upper right", ncol=3, fontsize=9.5, framealpha=0.9)
    ax_macro.set_xlim(t_m[0], t_m[-1])
    
    # Panel B: Meso scale around Frame 1553 cliff episode (frames 1100 to 2000 = 900 frames ~ 7.8 ms)
    ax_meso = fig1.add_subplot(gs[1])
    meso_slice = slice(1100, 2000)
    t_meso = time_ms[meso_slice]
    for k in range(5):
        ax_meso.plot(t_meso, cln_coords[meso_slice, k], color=colors[k], lw=1.5, alpha=0.9, label=f"Mode {k+1}")
    cliff_t = time_ms[1553]
    ax_meso.axvline(cliff_t, color="black", ls=":", lw=1.8, label="Frame 1553 (Raw Cliff Location)")
    ax_meso.annotate("Frame 1553\n(Cliff in Raw data\nsmoothly restored)", xy=(cliff_t, cln_coords[1553, 0]),
                     xytext=(cliff_t + 0.5, np.max(cln_coords[meso_slice, :2]) * 0.8),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                     fontsize=9.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.4))
    ax_meso.set_title("B. Meso-scale Window around Discontinuity Episode (Frames 1,100–2,000) — Continuous, Shock-Free Dynamics", fontsize=12, fontweight="bold")
    ax_meso.set_ylabel("POD Amplitude $a_k(t)$ [$\\mu$m]", fontsize=11)
    ax_meso.grid(True, ls="--", alpha=0.35)
    ax_meso.legend(loc="upper right", ncol=3, fontsize=9.5, framealpha=0.9)
    ax_meso.set_xlim(t_meso[0], t_meso[-1])
    
    # Panel C: Micro scale (frames 2200 to 2500 = 300 frames ~ 2.6 ms, ~1 wave period)
    ax_micro = fig1.add_subplot(gs[2])
    micro_slice = slice(2200, 2500)
    t_micro = time_ms[micro_slice]
    for k in range(5):
        ax_micro.plot(t_micro, cln_coords[micro_slice, k], color=colors[k], lw=1.8, label=f"Mode {k+1}")
    ax_micro.set_title("C. Micro-scale Waveform Resolution (Frames 2,200–2,500 $\\approx 2.6$ ms) — Smooth Wave Cycle without Pixel Discretization", fontsize=12, fontweight="bold")
    ax_micro.set_xlabel("Physical Time $t$ [ms]", fontsize=11)
    ax_micro.set_ylabel("POD Amplitude $a_k(t)$ [$\\mu$m]", fontsize=11)
    ax_micro.grid(True, ls="--", alpha=0.35)
    ax_micro.legend(loc="upper right", ncol=5, fontsize=9.5, framealpha=0.9)
    ax_micro.set_xlim(t_micro[0], t_micro[-1])
    
    fig1_path = OUTPUT_DIR / "cleaned_pod_trajectories_overview.png"
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)
    print(f"  Saved {fig1_path}")
    
    # =========================================================================
    # FIGURE 2: Phase Space Attractors & Manifold Geometry
    # =========================================================================
    print("Generating Figure 2: Phase Space Attractors & Manifold Geometry...")
    fig2 = plt.figure(figsize=(16, 12), constrained_layout=True)
    
    # We will plot 4 subplots: (a1, a2), (a4, a5), (a1, a3), and 3D (a1, a2, a3)
    # Use 3,000 consecutive frames (~26 ms) to show attractor orbits clearly
    attract_slice = slice(2000, 5000)
    a1 = cln_coords[attract_slice, 0]
    a2 = cln_coords[attract_slice, 1]
    a3 = cln_coords[attract_slice, 2]
    a4 = cln_coords[attract_slice, 3]
    a5 = cln_coords[attract_slice, 4]
    time_color = (time_ms[attract_slice] - time_ms[attract_slice][0])
    
    # 1. (a1, a2) Phase Portrait: Primary Traveling Wave Limit Cycle
    ax21 = fig2.add_subplot(2, 2, 1)
    sc1 = ax21.scatter(a1, a2, c=time_color, cmap="viridis", s=6, alpha=0.6)
    ax21.plot(a1[:200], a2[:200], color="navy", lw=1.2, alpha=0.8, label="Initial 200 steps")
    ax21.set_title("A. Primary Traveling Wave Pair $(a_1, a_2)$ Limit Cycle Attractor", fontsize=12, fontweight="bold")
    ax21.set_xlabel("Mode 1 Amplitude $a_1(t)$ [$\\mu$m]", fontsize=11)
    ax21.set_ylabel("Mode 2 Amplitude $a_2(t)$ [$\\mu$m]", fontsize=11)
    ax21.axis("equal")
    ax21.grid(True, ls="--", alpha=0.35)
    cbar1 = plt.colorbar(sc1, ax=ax21, pad=0.02)
    cbar1.set_label("Elapsed Time $\\Delta t$ [ms]", fontsize=9.5)
    ax21.legend(loc="upper left", fontsize=9.5)
    
    # 2. (a4, a5) Phase Portrait: Secondary Harmonic Traveling Wave
    ax22 = fig2.add_subplot(2, 2, 2)
    sc2 = ax22.scatter(a4, a5, c=time_color, cmap="plasma", s=6, alpha=0.6)
    ax22.plot(a4[:200], a5[:200], color="darkred", lw=1.2, alpha=0.8, label="Initial 200 steps")
    ax22.set_title("B. Harmonic Wave Pair $(a_4, a_5)$ Orbit Topology", fontsize=12, fontweight="bold")
    ax22.set_xlabel("Mode 4 Amplitude $a_4(t)$ [$\\mu$m]", fontsize=11)
    ax22.set_ylabel("Mode 5 Amplitude $a_5(t)$ [$\\mu$m]", fontsize=11)
    ax22.axis("equal")
    ax22.grid(True, ls="--", alpha=0.35)
    cbar2 = plt.colorbar(sc2, ax=ax22, pad=0.02)
    cbar2.set_label("Elapsed Time $\\Delta t$ [ms]", fontsize=9.5)
    ax22.legend(loc="upper left", fontsize=9.5)
    
    # 3. (a1, a3) Phase Portrait: Traveling Wave coupled with Breathing Mode
    ax23 = fig2.add_subplot(2, 2, 3)
    sc3 = ax23.scatter(a1, a3, c=time_color, cmap="cividis", s=6, alpha=0.6)
    ax23.set_title("C. Wave-Breathing Interaction $(a_1, a_3)$ Phase Plane", fontsize=12, fontweight="bold")
    ax23.set_xlabel("Mode 1 Amplitude $a_1(t)$ [$\\mu$m]", fontsize=11)
    ax23.set_ylabel("Mode 3 (Breathing) Amplitude $a_3(t)$ [$\\mu$m]", fontsize=11)
    ax23.grid(True, ls="--", alpha=0.35)
    cbar3 = plt.colorbar(sc3, ax=ax23, pad=0.02)
    cbar3.set_label("Elapsed Time $\\Delta t$ [ms]", fontsize=9.5)
    
    # 4. 3D Phase Space Trajectory (a1, a2, a3)
    ax24 = fig2.add_subplot(2, 2, 4, projection="3d")
    p3d = ax24.scatter(a1, a2, a3, c=time_color, cmap="viridis", s=4, alpha=0.5)
    ax24.plot(a1[:300], a2[:300], a3[:300], color="black", lw=1.5, alpha=0.9, label="Initial 300 steps")
    ax24.set_title("D. 3D Reduced State Attractor $(a_1, a_2, a_3)$", fontsize=12, fontweight="bold")
    ax24.set_xlabel("$a_1$ [$\\mu$m]", fontsize=10, labelpad=8)
    ax24.set_ylabel("$a_2$ [$\\mu$m]", fontsize=10, labelpad=8)
    ax24.set_zlabel("$a_3$ [$\\mu$m]", fontsize=10, labelpad=8)
    ax24.view_init(elev=28, azim=45)
    ax24.legend(loc="upper left", fontsize=9)
    
    fig2_path = OUTPUT_DIR / "cleaned_pod_phase_portractors_attractor.png"
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)
    print(f"  Saved {fig2_path}")
    
    # =========================================================================
    # FIGURE 3: What the Model Needs to Learn (Velocities, Increments, PSD, Autocorr)
    # =========================================================================
    print("Generating Figure 3: Dynamical Systems & Machine Learning Properties...")
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    
    # 1. Lag-1 Increments / Velocities Time Series: da_k/dt
    ax31 = axes3[0, 0]
    sub_slice = slice(2000, 3000)
    t_vel = time_ms[sub_slice]
    for k in range(3):
        ax31.plot(t_vel, vel_cln[sub_slice, k] / 1000.0, color=colors[k], lw=1.2, alpha=0.85, label=f"Mode {k+1} Velocity $\\dot{{a}}_{k}$")
    ax31.set_title("A. State Velocity $\\dot{a}_k(t) = \\Delta a_k / \\Delta t$ [mm/s] (Target for $\\dot{\\mathbf{a}} = f(\\mathbf{a})$)", fontsize=12, fontweight="bold")
    ax31.set_xlabel("Time $t$ [ms]", fontsize=11)
    ax31.set_ylabel("Coordinate Velocity $\\dot{a}_k$ [mm/s]", fontsize=11)
    ax31.grid(True, ls="--", alpha=0.35)
    ax31.legend(loc="upper right", fontsize=9.5)
    ax31.set_xlim(t_vel[0], t_vel[-1])
    
    # 2. Probability Density Functions (PDF) of Lag-1 Increments (Log scale)
    ax32 = axes3[0, 1]
    bins = np.linspace(-100, 100, 121)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for k in range(5):
        counts, _ = np.histogram(diff_cln[:, k], bins=bins, density=True)
        ax32.semilogy(bin_centers, counts, "o-", color=colors[k], ms=3.5, lw=1.2, label=f"Mode {k+1} ($\sigma={np.std(diff_cln[:, k]):.1f}$)")
    # Overlay theoretical Gaussian for Mode 1
    s1 = np.std(diff_cln[:, 0])
    gauss = (1.0 / (s1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (bin_centers / s1)**2)
    ax32.semilogy(bin_centers, gauss, "k--", lw=1.5, label=f"Gaussian fit ($M_1$)")
    ax32.set_title("B. Lag-1 Increment Distributions $P(\\Delta a_k)$ — Bounded & Stable for Training", fontsize=12, fontweight="bold")
    ax32.set_xlabel("Lag-1 Increment $\\Delta a_k$ [$\\mu$m / $\\Delta t$]", fontsize=11)
    ax32.set_ylabel("Probability Density (log scale)", fontsize=11)
    ax32.set_ylim(1e-5, 0.1)
    ax32.grid(True, which="both", ls="--", alpha=0.35)
    ax32.legend(loc="upper right", fontsize=9.0)
    
    # 3. Autocorrelation Functions R_k(tau)
    ax33 = axes3[1, 0]
    max_lag = 1200  # ~10.4 ms
    lags = np.arange(max_lag)
    lags_ms = lags * dt * 1000.0
    for k in range(5):
        ak = cln_coords[:, k] - np.mean(cln_coords[:, k])
        # FFT based autocorrelation
        n = len(ak)
        ak_pad = np.pad(ak, (0, n))
        fft_ak = np.fft.fft(ak_pad)
        corr = np.fft.ifft(fft_ak * np.conj(fft_ak)).real[:max_lag]
        corr /= corr[0]
        ax33.plot(lags_ms, corr, color=colors[k], lw=1.5, label=f"Mode {k+1}")
    ax33.axhline(0, color="gray", ls=":", lw=1.0)
    ax33.set_title("C. Coordinate Autocorrelation $R_k(\\tau) = \\langle a_k(t) a_k(t+\\tau) \\rangle$ — Wave Period & Decorrelation", fontsize=12, fontweight="bold")
    ax33.set_xlabel("Time Lag $\\tau$ [ms]", fontsize=11)
    ax33.set_ylabel("Normalized Autocorrelation", fontsize=11)
    ax33.grid(True, ls="--", alpha=0.35)
    ax33.legend(loc="upper right", fontsize=9.5)
    ax33.set_xlim(0, lags_ms[-1])
    
    # 4. Power Spectral Density (PSD) of POD Coordinates
    ax34 = axes3[1, 1]
    nperseg = 8192
    for k in range(5):
        f_psd, psd_k = welch(cln_coords[:, k], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        ax34.semilogy(f_psd, psd_k, color=colors[k], lw=1.5, label=f"Mode {k+1}")
    ax34.set_xlim(50, 2000)
    ax34.set_ylim(1e-2, 1e8)
    ax34.axvline(140, color="gray", ls=":", lw=1.0)
    ax34.axvline(197, color="gray", ls=":", lw=1.0)
    ax34.axvline(365, color="gray", ls=":", lw=1.0)
    ax34.set_title("D. Power Spectral Density $S_k(f)$ of POD Coordinates — Resonant Carrier Frequencies", fontsize=12, fontweight="bold")
    ax34.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
    ax34.set_ylabel("Power Spectral Density [$\\mu$m$^2$/Hz]", fontsize=11)
    ax34.grid(True, which="both", ls="--", alpha=0.35)
    ax34.legend(loc="upper right", fontsize=9.5)
    
    fig3_path = OUTPUT_DIR / "cleaned_pod_learning_diagnostics.png"
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)
    print(f"  Saved {fig3_path}")
    
    # =========================================================================
    # FIGURE 4: Raw vs Cleaned Trajectory Contrast at Discontinuity Episodes
    # =========================================================================
    print("Generating Figure 4: Raw vs Cleaned Trajectory Comparison...")
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    
    # Episode 1: Frame 1553 Cliff (Raw has huge jump, Cleaned is continuous)
    # Left column: Mode 1 and Mode 2 for Raw vs Cleaned
    ep1_slice = slice(1450, 1650)
    t_ep1 = time_ms[ep1_slice]
    
    # Raw on Cleaned POD modes
    ax41 = axes4[0, 0]
    ax41.plot(t_ep1, raw_on_cln_coords[ep1_slice, 0], "r--", lw=1.8, label="Raw data on Cleaned Mode 1")
    ax41.plot(t_ep1, cln_coords[ep1_slice, 0], "b-", lw=2.0, label="Cleaned data on Cleaned Mode 1")
    ax41.axvline(time_ms[1553], color="black", ls=":", lw=1.5, label="Discontinuity Frame 1553")
    ax41.set_title("A. Discontinuity Episode (Frame 1553) — Mode 1 Trajectory", fontsize=12, fontweight="bold")
    ax41.set_ylabel("Amplitude $a_1(t)$ [$\\mu$m]", fontsize=11)
    ax41.grid(True, ls="--", alpha=0.35)
    ax41.legend(loc="upper right", fontsize=9.5)
    
    ax42 = axes4[1, 0]
    ax42.plot(t_ep1, raw_on_cln_coords[ep1_slice, 1], "r--", lw=1.8, label="Raw data on Cleaned Mode 2")
    ax42.plot(t_ep1, cln_coords[ep1_slice, 1], "g-", lw=2.0, label="Cleaned data on Cleaned Mode 2")
    ax42.axvline(time_ms[1553], color="black", ls=":", lw=1.5, label="Discontinuity Frame 1553")
    ax42.set_title("B. Discontinuity Episode (Frame 1553) — Mode 2 Trajectory", fontsize=12, fontweight="bold")
    ax42.set_xlabel("Time $t$ [ms]", fontsize=11)
    ax42.set_ylabel("Amplitude $a_2(t)$ [$\\mu$m]", fontsize=11)
    ax42.grid(True, ls="--", alpha=0.35)
    ax42.legend(loc="upper right", fontsize=9.5)
    
    # Episode 2: Phase Portrait (a1, a2) comparison around Cliff episode
    # Shows the catastrophic scar in Raw vs the smooth circle in Cleaned
    ax43 = axes4[0, 1]
    ax43.plot(raw_on_cln_coords[ep1_slice, 0], raw_on_cln_coords[ep1_slice, 1], "r.-", lw=1.2, ms=4, label="Raw Trajectory (Shock jump scar!)")
    ax43.plot(cln_coords[ep1_slice, 0], cln_coords[ep1_slice, 1], "b.-", lw=1.5, ms=4, label="Cleaned Trajectory (Smooth orbit)")
    ax43.set_title("C. Phase Portrait $(a_1, a_2)$ around Frame 1553 Episode", fontsize=12, fontweight="bold")
    ax43.set_xlabel("Mode 1 Amplitude $a_1(t)$ [$\\mu$m]", fontsize=11)
    ax43.set_ylabel("Mode 2 Amplitude $a_2(t)$ [$\\mu$m]", fontsize=11)
    ax43.axis("equal")
    ax43.grid(True, ls="--", alpha=0.35)
    ax43.legend(loc="upper left", fontsize=9.5)
    
    # Episode 3: Clean Episode Frame 2300 (Bit-for-bit identical)
    ep2_slice = slice(2200, 2400)
    t_ep2 = time_ms[ep2_slice]
    ax44 = axes4[1, 1]
    diff_2300 = np.abs(cln_coords[ep2_slice, 0] - raw_on_cln_coords[ep2_slice, 0])
    ax44.plot(t_ep2, raw_on_cln_coords[ep2_slice, 0], "r--", lw=2.2, label="Raw data")
    ax44.plot(t_ep2, cln_coords[ep2_slice, 0], "b-", lw=1.2, label="Cleaned data")
    ax44.set_title(f"D. Clean Episode (Frame 2300) — Bit-for-Bit Preservation (max $|\\Delta a_1| = {np.max(diff_2300):.4f}$ $\\mu$m)", fontsize=12, fontweight="bold")
    ax44.set_xlabel("Time $t$ [ms]", fontsize=11)
    ax44.set_ylabel("Amplitude $a_1(t)$ [$\\mu$m]", fontsize=11)
    ax44.grid(True, ls="--", alpha=0.35)
    ax44.legend(loc="upper right", fontsize=9.5)
    
    fig4_path = OUTPUT_DIR / "raw_vs_cleaned_pod_trajectories_comparison.png"
    fig4.savefig(fig4_path, dpi=200)
    plt.close(fig4)
    print(f"  Saved {fig4_path}")
    
    # =========================================================================
    # Save Key Metrics to CSV & NPZ
    # =========================================================================
    csv_path = OUTPUT_DIR / "pod_trajectories_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "std_um", "rms_equiv_height_um", "max_vel_mms", "std_vel_mms", "dominant_freq_hz"])
        for k in range(5):
            f_psd, psd_k = welch(cln_coords[:, k], fs=fs, nperseg=8192)
            dom_f = f_psd[np.argmax(psd_k[f_psd >= 50])]
            writer.writerow([
                k + 1,
                f"{std_cln[k]:.3f}",
                f"{std_cln[k]/50.0:.3f}",
                f"{np.max(np.abs(vel_cln[:, k]))/1000.0:.3f}",
                f"{np.std(vel_cln[:, k])/1000.0:.3f}",
                f"{dom_f:.1f}",
            ])
    print(f"  Saved {csv_path}")
    
    # Save coordinate arrays for future modeling use
    npz_path = OUTPUT_DIR / "cleaned_pod_r5_coordinates.npz"
    np.savez_compressed(
        npz_path,
        time_sec=time_sec,
        cln_coords=cln_coords.astype(np.float32),
        std_cln=std_cln,
        vel_cln=vel_cln.astype(np.float32),
        cln_modes=cln_modes_5,
    )
    print(f"  Saved {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Done in {time.perf_counter() - t_total:.1f} s")


if __name__ == "__main__":
    main()

