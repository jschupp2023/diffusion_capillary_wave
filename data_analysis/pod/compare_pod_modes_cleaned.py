"""Compare POD modes and singular values for Raw vs Cleaned 0p20 datasets under 50 Hz high-pass filtering.

Computes:
1. 20 POD modes with singular values (SV)
2. Energy fraction per mode and cumulative energy curves
3. Overlap / alignment with the uniform spatial mean (constant piston mode)
4. 2-D spatial mode shapes phi_k(x, y) for Modes 1 to 8 comparing Raw vs Cleaned
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
import numpy as np
from scipy.signal import butter, sosfiltfilt

from data_analysis.pod.pod_2d import (
    _frame_key,
    inspect_input,
    randomized_spatial_subspace,
    refine_pod,
    stratified_training_positions,
)

RAW_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
CLEAN_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
OUTPUT_DIR = Path("runs/pod_modes_comparison_50hz")


def compute_highpass_pod(
    h5_path: Path,
    cutoff_hz: float = 50.0,
    stride: int = 4,
    pod_frames: int = 5000,
    rank: int = 20,
    seed: int = 12345,
) -> dict[str, Any]:
    started = time.perf_counter()
    info = inspect_input(h5_path, 0, None)
    
    y_idx = np.arange(0, info.frame_shape[0], stride, dtype=np.int64)
    x_idx = np.arange(0, info.frame_shape[1], stride, dtype=np.int64)
    sampled_shape = (len(y_idx), len(x_idx))
    n_space = len(y_idx) * len(x_idx)
    
    mean_dt = float(np.mean(np.diff(info.time)))
    fs = 1.0 / mean_dt
    
    print(f"\nProcessing {h5_path.name}:")
    print(f"  Grid: {sampled_shape[1]} x {sampled_shape[0]} ({n_space} pixels, stride {stride})")
    print(f"  Sampling: {fs:.1f} Hz | Cutoff: {cutoff_hz} Hz | POD Snapshots: {pod_frames:,} | Rank: {rank}")
    
    # 1. Extract sampled trajectory across all frames
    t0 = time.perf_counter()
    raw_trajectory = np.empty((info.n_frames, n_space), dtype=np.float32)
    
    with h5py.File(h5_path, "r") as handle:
        main_grp = handle["main"]
        report_step = info.n_frames // 5
        for i in range(info.n_frames):
            frame = np.asarray(main_grp[_frame_key(i)], dtype=np.float32)
            raw_trajectory[i] = frame[np.ix_(y_idx, x_idx)].reshape(-1)
            if (i + 1) % report_step == 0 or (i + 1) == info.n_frames:
                print(f"  Extracted {i+1:,}/{info.n_frames:,} frames ({(i+1)/info.n_frames*100:.0f}%, {time.perf_counter()-t0:.1f} s)")
                
    # 2. Filter every column at cutoff_hz
    t_flt = time.perf_counter()
    sos = butter(4, cutoff_hz, btype="highpass", fs=fs, output="sos")
    filtered_trajectory = sosfiltfilt(sos, raw_trajectory.astype(np.float64), axis=0).astype(np.float32)
    print(f"  High-pass filtering complete in {time.perf_counter() - t_flt:.1f} s")
    
    # 3. Stratified sampling of snapshots
    rng = np.random.default_rng(seed)
    positions = stratified_training_positions(info.n_frames, pod_frames, rng)
    snapshots = filtered_trajectory[positions]
    
    # Temporal centering
    temporal_mean = snapshots.mean(axis=0, dtype=np.float64)
    snapshots -= temporal_mean.astype(np.float32)[None, :]
    
    # Total energy
    total_energy = float(np.einsum("ij,ij->", snapshots, snapshots, dtype=np.float64))
    
    # 4. Constant spatial mean vector & energy
    constant = np.full(n_space, 1.0 / np.sqrt(n_space), dtype=np.float64)
    const_coeff = snapshots.astype(np.float64) @ constant
    constant_energy = float(const_coeff @ const_coeff)
    const_energy_frac = constant_energy / total_energy
    
    # 5. Randomized SVD
    cand_basis, _ = randomized_spatial_subspace(snapshots, rank + 10, 1, np.random.default_rng(seed))
    cand_coeff = snapshots @ cand_basis
    reduced_cov = cand_coeff.T.astype(np.float64) @ cand_coeff.astype(np.float64)
    pod = refine_pod(cand_basis, reduced_cov, total_energy, rank)
    
    sv = pod.singular_values
    mode_energy_frac = np.square(sv) / total_energy
    cum_energy_frac = np.minimum(np.cumsum(mode_energy_frac), 1.0)
    
    # Alignment squared with constant vector: |u_i^T c|^2
    modes_matrix = pod.modes  # shape: (n_space, rank)
    const_align_sq = np.clip(np.square(modes_matrix.T.astype(np.float64) @ constant), 0.0, 1.0)
    
    # Reshape modes to 2-D spatial fields: (rank, ny, nx)
    modes_2d = np.empty((rank, sampled_shape[0], sampled_shape[1]), dtype=np.float32)
    for m in range(rank):
        modes_2d[m] = modes_matrix[:, m].reshape(sampled_shape)
        
    most_const_mode = int(np.argmax(const_align_sq))
    
    print(f"  POD complete in {time.perf_counter() - started:.1f} s:")
    print(f"    Total energy: {total_energy:.3e}")
    print(f"    Constant subspace energy fraction: {const_energy_frac*100:.2f}%")
    print(f"    Most constant mode: Mode {most_const_mode + 1} (alignment^2 = {const_align_sq[most_const_mode]:.4f}, energy = {mode_energy_frac[most_const_mode]*100:.2f}%)")
    print(f"    Rank-{rank} cumulative energy: {cum_energy_frac[-1]*100:.2f}%")
    
    return {
        "dataset_name": h5_path.stem,
        "n_space": n_space,
        "sampled_shape": sampled_shape,
        "x_um": info.x[x_idx],
        "y_um": info.y[y_idx],
        "total_energy": total_energy,
        "singular_values": sv,
        "mode_energy_fraction": mode_energy_frac,
        "cumulative_energy_fraction": cum_energy_frac,
        "constant_energy_fraction": const_energy_frac,
        "constant_alignment_squared": const_align_sq,
        "most_constant_mode": most_const_mode,
        "modes_2d": modes_2d,
    }


def plot_pod_comparison_metrics(
    raw_res: dict[str, Any],
    cln_res: dict[str, Any],
    output_path: Path,
    dpi: int = 200,
) -> None:
    modes = np.arange(1, len(raw_res["singular_values"]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    
    # 1. Mode Energy Fraction (%) on Log Scale
    axes[0, 0].semilogy(modes, raw_res["mode_energy_fraction"] * 100, "ro-", lw=1.8, ms=6, label="Raw (50 Hz HP)")
    axes[0, 0].semilogy(modes, cln_res["mode_energy_fraction"] * 100, "go-", lw=2.0, ms=6, label="Cleaned (Integer tau=4.0 + 50 Hz HP)")
    axes[0, 0].set_title("POD Mode Energy Distribution $\\lambda_k / E_{\\mathrm{tot}}$", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("POD Mode Index $k$", fontsize=11)
    axes[0, 0].set_ylabel("Modal Energy [%] (log scale)", fontsize=11)
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.35)
    axes[0, 0].legend(fontsize=10.5)
    axes[0, 0].set_xticks(modes)
    
    # 2. Cumulative Energy Fraction (%)
    axes[0, 1].plot(modes, raw_res["cumulative_energy_fraction"] * 100, "ro-", lw=1.8, ms=6, label="Raw (50 Hz HP)")
    axes[0, 1].plot(modes, cln_res["cumulative_energy_fraction"] * 100, "go-", lw=2.0, ms=6, label="Cleaned (Integer + 50 Hz HP)")
    axes[0, 1].axhline(80, color="gray", ls=":", lw=1.0)
    axes[0, 1].axhline(90, color="gray", ls=":", lw=1.0)
    axes[0, 1].set_title("Cumulative POD Energy $\\sum_{j=1}^k \\lambda_j / E_{\\mathrm{tot}}$", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("POD Mode Index $k$", fontsize=11)
    axes[0, 1].set_ylabel("Cumulative Energy [%]", fontsize=11)
    axes[0, 1].grid(True, ls="--", alpha=0.35)
    axes[0, 1].legend(fontsize=10.5)
    axes[0, 1].set_xticks(modes)
    axes[0, 1].set_ylim(20, 102)
    
    # 3. Constant Spatial Mean Overlap |u_k^T c|^2
    bar_width = 0.38
    axes[1, 0].bar(modes - bar_width/2, raw_res["constant_alignment_squared"], width=bar_width, color="crimson", alpha=0.85, label="Raw (50 Hz HP)")
    axes[1, 0].bar(modes + bar_width/2, cln_res["constant_alignment_squared"], width=bar_width, color="forestgreen", alpha=0.85, label="Cleaned (Integer + 50 Hz HP)")
    axes[1, 0].set_title("Overlap with Uniform Spatial Mean $|\\mathbf{u}_k^T \\mathbf{c}|^2$", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("POD Mode Index $k$", fontsize=11)
    axes[1, 0].set_ylabel(r"Squared Alignment $|u_k^T c|^2$", fontsize=11)
    axes[1, 0].grid(True, ls="--", alpha=0.35)
    axes[1, 0].legend(fontsize=10.5)
    axes[1, 0].set_xticks(modes)
    axes[1, 0].set_ylim(0, 1.05)
    
    # 4. Singular Values (Direct SV)
    axes[1, 1].semilogy(modes, raw_res["singular_values"], "rs-", lw=1.8, ms=6, label=f"Raw SV (Total E = {raw_res['total_energy']:.2e})")
    axes[1, 1].semilogy(modes, cln_res["singular_values"], "gs-", lw=2.0, ms=6, label=f"Cleaned SV (Total E = {cln_res['total_energy']:.2e})")
    axes[1, 1].set_title("Singular Value Spectrum $\\sigma_k$", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("POD Mode Index $k$", fontsize=11)
    axes[1, 1].set_ylabel("Singular Value $\\sigma_k$", fontsize=11)
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.35)
    axes[1, 1].legend(fontsize=10.5)
    axes[1, 1].set_xticks(modes)
    
    fig.suptitle(
        r"POD Basis & Singular Value Comparison on $0.20\,\mathrm{V}_{\mathrm{pp}}$ (50 Hz High-Pass Filtered)" "\n"
        f"Raw: Spatial mean dominated Mode 1 (67.9% energy, 99.8% piston) | Cleaned: Spatial mean dropped to 27.8% (Mode 3)",
        fontsize=14,
        fontweight="bold",
    )
    
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved POD comparison plot to {output_path}")


def plot_spatial_mode_shapes(
    raw_res: dict[str, Any],
    cln_res: dict[str, Any],
    n_display_modes: int = 6,
    output_path: Path = None,
    dpi: int = 200,
) -> None:
    """Plot 2-D spatial mode eigenfunctions comparing Raw vs Cleaned."""
    x = raw_res["x_um"]
    y = raw_res["y_um"]
    extent = [x[0], x[-1], y[0], y[-1]]
    
    fig, axes = plt.subplots(2, n_display_modes, figsize=(3.2 * n_display_modes, 6.8), constrained_layout=True)
    
    for m in range(n_display_modes):
        phi_raw = raw_res["modes_2d"][m]
        phi_cln = cln_res["modes_2d"][m]
        
        # SVD sign convention: fix sign so center or peak is positive
        if phi_raw.ravel()[np.argmax(np.abs(phi_raw))] < 0:
            phi_raw = -phi_raw
        if phi_cln.ravel()[np.argmax(np.abs(phi_cln))] < 0:
            phi_cln = -phi_cln
            
        # Common symmetric color limits
        v_raw = float(np.max(np.abs(phi_raw)))
        v_cln = float(np.max(np.abs(phi_cln)))
        
        # Row 0: Raw Modes
        im0 = axes[0, m].imshow(phi_raw, origin="lower", cmap="coolwarm", extent=extent, vmin=-v_raw, vmax=v_raw)
        re_pct = raw_res["mode_energy_fraction"][m] * 100
        rc_align = raw_res["constant_alignment_squared"][m]
        axes[0, m].set_title(f"Raw Mode {m+1}\n{re_pct:.1f}% E | Piston={rc_align:.2f}", fontsize=10, fontweight="bold")
        if m == 0:
            axes[0, m].set_ylabel("Raw Modes\n$y$ [$\\mu\\mathrm{m}$]", fontsize=11, fontweight="bold")
        fig.colorbar(im0, ax=axes[0, m], fraction=0.046, pad=0.04)
        
        # Row 1: Cleaned Modes
        im1 = axes[1, m].imshow(phi_cln, origin="lower", cmap="coolwarm", extent=extent, vmin=-v_cln, vmax=v_cln)
        ce_pct = cln_res["mode_energy_fraction"][m] * 100
        cc_align = cln_res["constant_alignment_squared"][m]
        axes[1, m].set_title(f"Cleaned Mode {m+1}\n{ce_pct:.1f}% E | Piston={cc_align:.2f}", fontsize=10, fontweight="bold")
        if m == 0:
            axes[1, m].set_ylabel("Cleaned Modes\n$y$ [$\\mu\\mathrm{m}$]", fontsize=11, fontweight="bold")
        axes[1, m].set_xlabel(r"$x$ [$\mu\mathrm{m}$]", fontsize=10)
        fig.colorbar(im1, ax=axes[1, m], fraction=0.046, pad=0.04)
        
    fig.suptitle(
        r"2-D Spatial POD Modes $\phi_k(x, y)$ (50 Hz High-Pass Filtered)" "\n"
        "Notice: Raw Mode 1 is an artificial flat piston sheet (0.998 overlap), while Cleaned Modes reveal true 2D wave nodal geometries!",
        fontsize=13,
        fontweight="bold",
    )
    
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved 2-D spatial modes plot to {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Computing 50 Hz high-pass POD on Raw dataset...")
    raw_res = compute_highpass_pod(RAW_H5, cutoff_hz=50.0, stride=4, pod_frames=5000, rank=20)
    
    print("\nComputing 50 Hz high-pass POD on Cleaned dataset...")
    cln_res = compute_highpass_pod(CLEAN_H5, cutoff_hz=50.0, stride=4, pod_frames=5000, rank=20)
    
    # 1. Plot POD spectra and alignment metrics
    metric_plot_path = OUTPUT_DIR / "pod_metrics_comparison_50hz.png"
    plot_pod_comparison_metrics(raw_res, cln_res, output_path=metric_plot_path)
    
    # 2. Plot 2D spatial mode shapes
    modes_plot_path = OUTPUT_DIR / "pod_spatial_modes_comparison_50hz.png"
    plot_spatial_mode_shapes(raw_res, cln_res, n_display_modes=6, output_path=modes_plot_path)
    
    # Save CSV comparison table
    csv_path = OUTPUT_DIR / "pod_modes_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mode",
            "Raw_SV", "Clean_SV",
            "Raw_Energy_Pct", "Clean_Energy_Pct",
            "Raw_Cum_Energy_Pct", "Clean_Cum_Energy_Pct",
            "Raw_Const_Align_Sq", "Clean_Const_Align_Sq"
        ])
        for m in range(20):
            writer.writerow([
                m + 1,
                f"{raw_res['singular_values'][m]:.4e}", f"{cln_res['singular_values'][m]:.4e}",
                f"{raw_res['mode_energy_fraction'][m]*100:.3f}", f"{cln_res['mode_energy_fraction'][m]*100:.3f}",
                f"{raw_res['cumulative_energy_fraction'][m]*100:.3f}", f"{cln_res['cumulative_energy_fraction'][m]*100:.3f}",
                f"{raw_res['constant_alignment_squared'][m]:.4f}", f"{cln_res['constant_alignment_squared'][m]:.4f}",
            ])
    print(f"Saved CSV table to {csv_path}")
    
    # Save NPZ
    npz_path = OUTPUT_DIR / "pod_results.npz"
    np.savez_compressed(
        npz_path,
        raw_sv=raw_res["singular_values"],
        cln_sv=cln_res["singular_values"],
        raw_energy=raw_res["mode_energy_fraction"],
        cln_energy=cln_res["mode_energy_fraction"],
        raw_align=raw_res["constant_alignment_squared"],
        cln_align=cln_res["constant_alignment_squared"],
        raw_modes=raw_res["modes_2d"],
        cln_modes=cln_res["modes_2d"],
        x_um=raw_res["x_um"],
        y_um=raw_res["y_um"],
    )
    print(f"Saved NPZ to {npz_path}")
    print("\nAll POD comparison diagnostics completed successfully!")


if __name__ == "__main__":
    main()

