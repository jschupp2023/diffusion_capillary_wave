"""Compare shared-POD trajectories of raw vs cleaned datasets around discontinuities.

Projects raw and Poisson-restored height fields onto the shared 0p20 POD basis,
applies the training standardization, and generates:
1. Stacked coordinate trajectory overlays (a_k(t))
2. Stacked lag-1 increment comparisons (Delta a_k(t))
3. 2-D Phase portrait orbits (a_1 vs a_2, a_3 vs a_4, a_1 vs a_20)
4. Mahalanobis increment anomaly distance metrics

Matches the conventions of data_analysis.plot_normalized_coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.pod.pod_2d import _frame_key
from data_analysis.pod.phase_unwrap_gradient_poisson import DEFAULT_Q, DEFAULT_TAU, restore_frame_poisson
from data_analysis.plot_normalized_coordinates import increment_mahalanobis
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT

DEFAULT_RAW = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
DEFAULT_CLN = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
DEFAULT_SHARED_BASIS = Path("/home/jonas/ucsd_thesis/reduced_data/0p20/shared_pod/shared_pod_r1000.h5")
DEFAULT_REP13_POD = Path("/home/jonas/ucsd_thesis/reduced_data/0p20/Ca_ac_0p001660_rep13/pod_2d_r1000.h5")
DEFAULT_SCALER_CACHE = Path("runs/comparison_0p20_cleaned/standardization_scaler_r20.json")
DEFAULT_MODES = [0, 1, 2, 3, 4, 10, 20]


def load_or_fit_standardization(rank: int = 20, scaler_cache: Path | None = DEFAULT_SCALER_CACHE) -> tuple[np.ndarray, np.ndarray]:
    """Load cached standardization scaler or compute from training repetitions."""
    if scaler_cache is not None and scaler_cache.is_file():
        with open(scaler_cache, "r") as f:
            data = json.load(f)
        mean = np.asarray(data["mean"][:rank + 1], dtype=np.float64)
        scale = np.asarray(data["scale"][:rank + 1], dtype=np.float64)
        return mean, scale

    print(f"Fitting standardization across 0p20 training repetitions (rank {rank})...")
    sp_data = SharedPODTrainingData("0p20", rank).fit_standardization()
    mean = np.asarray(sp_data.mean, dtype=np.float64)
    scale = np.asarray(sp_data.scale, dtype=np.float64)
    if scaler_cache is not None:
        scaler_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_cache, "w") as f:
            json.dump({"mean": mean.tolist(), "scale": scale.tolist()}, f, indent=2)
    return mean, scale


def project_frames_window(
    h5_file: Path,
    start_step: int,
    n_steps: int,
    shared_modes: np.ndarray,
    temporal_mean_flat: np.ndarray,
    restore: bool = False,
    q: float = DEFAULT_Q,
    tau: float = DEFAULT_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a window of frames from an HDF5 dataset onto shared POD modes.
    
    Returns
    -------
    time_sec : np.ndarray, shape (n_steps,)
    coords : np.ndarray, shape (n_steps, rank + 1)
        Column 0 is sqrt(N_space) * frame_spatial_mean.
        Columns 1..rank are shared POD projections.
    """
    n_space = temporal_mean_flat.size
    rank = len(shared_modes)
    coords = np.empty((n_steps, rank + 1), dtype=np.float64)

    with h5py.File(h5_file, "r") as f:
        main_grp = f["main"]
        time_arr = np.asarray(f["meta"]["t"][start_step:start_step + n_steps])

        # Batch read frames
        frames = np.empty((n_steps, n_space), dtype=np.float64)
        for i in range(n_steps):
            frame_num = start_step + i
            key = _frame_key(frame_num)
            f_arr = np.asarray(main_grp[key], dtype=np.float32)
            if restore:
                res = restore_frame_poisson(f_arr, q=q, tau=tau)
                f_arr = res.restored
            frames[i] = f_arr.astype(np.float64).reshape(-1)

    # 1. Spatial means (Mode 0)
    spatial_means = np.mean(frames, axis=1)
    coords[:, 0] = np.sqrt(n_space) * spatial_means

    # 2. Subtract instantaneous spatial mean and temporal mean field
    fluctuations = (frames - spatial_means[:, None]) - temporal_mean_flat[None, :]

    # 3. Inner product with shared modes: (n_steps, n_space) @ (n_space, rank)
    coords[:, 1:] = fluctuations @ shared_modes.T

    return time_arr, coords


def plot_trajectory_overlay(
    time_rel_ms: np.ndarray,
    raw_norm: np.ndarray,
    cln_norm: np.ndarray,
    modes: list[int],
    output_path: Path,
    title: str,
    disc_frame_ms: float | None = None,
) -> None:
    """Plot stacked time-series comparing raw vs cleaned trajectories."""
    n_modes = len(modes)
    fig, axes = plt.subplots(
        n_modes, 1, figsize=(13, max(4, 2.2 * n_modes)), sharex=True, constrained_layout=True
    )
    if n_modes == 1:
        axes = np.array([axes])

    for ax, mode in zip(axes, modes):
        raw_s = raw_norm[:, mode]
        cln_s = cln_norm[:, mode]

        ax.plot(time_rel_ms, raw_s, color="crimson", lw=1.2, ls="--", alpha=0.85, label="Raw (with phase jumps)")
        ax.plot(time_rel_ms, cln_s, color="navy", lw=1.4, ls="-", label="Cleaned (Poisson restored)")

        # Highlight difference
        diff = np.abs(raw_s - cln_s)
        max_diff = np.max(diff)

        if disc_frame_ms is not None:
            ax.axvline(disc_frame_ms, color="darkorange", ls=":", lw=1.5, alpha=0.8, label="Discontinuity cliff" if mode == modes[0] else None)

        ax.axhline(0, color="0.6", lw=0.6, ls="-")
        label = "√N × mean (Mode 0)" if mode == 0 else f"Shared POD {mode}"
        ax.set_ylabel(f"{label}\n[$\\sigma$ units]", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

        # Annotate peak difference
        if max_diff > 0.05:
            ax.text(
                0.985, 0.88,
                f"Max |Δ| = {max_diff:.2f} σ",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
            )

    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=3)
    axes[-1].set_xlabel("Elapsed Time [ms]", fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_increments_comparison(
    time_rel_ms: np.ndarray,
    raw_norm: np.ndarray,
    cln_norm: np.ndarray,
    modes: list[int],
    output_path: Path,
    title: str,
    lag: int = 1,
    disc_frame_ms: float | None = None,
) -> None:
    """Plot stacked increments Delta a_k = a_k(t+lag) - a_k(t) comparing raw vs cleaned."""
    n_modes = len(modes)
    fig, axes = plt.subplots(
        n_modes, 1, figsize=(13, max(4, 2.2 * n_modes)), sharex=True, constrained_layout=True
    )
    if n_modes == 1:
        axes = np.array([axes])

    raw_inc = raw_norm[lag:] - raw_norm[:-lag]
    cln_inc = cln_norm[lag:] - cln_norm[:-lag]
    t_inc = time_rel_ms[lag:]

    for ax, mode in zip(axes, modes):
        r_i = raw_inc[:, mode]
        c_i = cln_inc[:, mode]

        ax.plot(t_inc, r_i, color="crimson", lw=1.2, ls="--", alpha=0.85, label=f"Raw lag-{lag} Δ")
        ax.plot(t_inc, c_i, color="teal", lw=1.4, ls="-", label=f"Cleaned lag-{lag} Δ")

        if disc_frame_ms is not None:
            ax.axvline(disc_frame_ms, color="darkorange", ls=":", lw=1.5, alpha=0.8)

        ax.axhline(0, color="0.6", lw=0.6, ls="-")
        label = "√N × mean (0)" if mode == 0 else f"POD {mode}"
        ax.set_ylabel(f"Δ {label}\n[$\\sigma$ / step]", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

        max_spike_raw = np.max(np.abs(r_i))
        max_spike_cln = np.max(np.abs(c_i))
        ax.text(
            0.985, 0.88,
            f"Peak step: Raw {max_spike_raw:.2f} σ | Cln {max_spike_cln:.2f} σ",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
        )

    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=3)
    axes[-1].set_xlabel(f"Elapsed Time [ms] (lag-{lag} step end)", fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_phase_portraits(
    raw_norm: np.ndarray,
    cln_norm: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot 2D phase space orbits: (a_1, a_2), (a_3, a_4), (a_1, a_20)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

    pairs = [(1, 2, "Fundamental Wave Conjugate Pair (Modes 1 vs 2)"),
             (3, 4, "First Harmonic Conjugate Pair (Modes 3 vs 4)"),
             (1, 20, "Energy Cascade Coupling (Modes 1 vs 20)")]

    for ax, (m1, m2, pair_title) in zip(axes, pairs):
        # Raw trajectory
        ax.plot(raw_norm[:, m1], raw_norm[:, m2], color="crimson", lw=1.0, ls="--", alpha=0.7, label="Raw trajectory")
        ax.scatter(raw_norm[0, m1], raw_norm[0, m2], color="crimson", marker="o", s=40, zorder=4)

        # Cleaned trajectory
        ax.plot(cln_norm[:, m1], cln_norm[:, m2], color="navy", lw=1.3, ls="-", alpha=0.9, label="Cleaned trajectory")
        ax.scatter(cln_norm[0, m1], cln_norm[0, m2], color="navy", marker="s", s=40, zorder=4)

        ax.set_xlabel(f"POD Mode {m1} [$\\sigma$]", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"POD Mode {m2} [$\\sigma$]", fontsize=11, fontweight="bold")
        ax.set_title(pair_title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_anomaly_scores(
    time_rel_ms: np.ndarray,
    raw_norm: np.ndarray,
    cln_norm: np.ndarray,
    output_path: Path,
    title: str,
    lag: int = 1,
    disc_frame_ms: float | None = None,
) -> None:
    """Plot increment Mahalanobis anomaly distance comparing raw vs cleaned."""
    # Compute Mahalanobis metric on both
    raw_res = increment_mahalanobis(raw_norm, lag=lag, percentile=95.0)
    cln_res = increment_mahalanobis(cln_norm, lag=lag, percentile=95.0)

    t_inc = time_rel_ms[lag:]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)

    # Top: Raw Mahalanobis score
    ax_top.plot(t_inc, raw_res["distance"], color="crimson", lw=1.2, label=f"Raw lag-{lag} Mahalanobis Distance")
    ax_top.axhline(raw_res["threshold"], color="darkred", ls="--", lw=1.2, label=f"95th percentile threshold = {raw_res['threshold']:.2f}")
    raw_exceed = raw_res["distance"] > raw_res["threshold"]
    ax_top.scatter(t_inc[raw_exceed], raw_res["distance"][raw_exceed], color="firebrick", s=18, zorder=3, label="Exceedance Anomaly")

    if disc_frame_ms is not None:
        ax_top.axvline(disc_frame_ms, color="darkorange", ls=":", lw=1.8, label="Discontinuity cliff")

    ax_top.set_ylabel("Mahalanobis Distance", fontsize=11, fontweight="bold")
    ax_top.set_title("Raw Trajectory: Unphysical Spurious Shock Anomalies", fontsize=12, fontweight="bold", color="darkred")
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Bottom: Cleaned Mahalanobis score
    ax_bot.plot(t_inc, cln_res["distance"], color="teal", lw=1.2, label=f"Cleaned lag-{lag} Mahalanobis Distance")
    ax_bot.axhline(cln_res["threshold"], color="darkgreen", ls="--", lw=1.2, label=f"95th percentile threshold = {cln_res['threshold']:.2f}")
    cln_exceed = cln_res["distance"] > cln_res["threshold"]
    ax_bot.scatter(t_inc[cln_exceed], cln_res["distance"][cln_exceed], color="seagreen", s=18, zorder=3, label="Exceedance")

    if disc_frame_ms is not None:
        ax_bot.axvline(disc_frame_ms, color="darkorange", ls=":", lw=1.8)

    ax_bot.set_ylabel("Mahalanobis Distance", fontsize=11, fontweight="bold")
    ax_bot.set_xlabel("Elapsed Time [ms]", fontsize=11, fontweight="bold")
    ax_bot.set_title("Cleaned Trajectory: Stationary Physical Wave Increments", fontsize=12, fontweight="bold", color="darkgreen")
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_trajectory_comparison(
    start_step: int,
    timesteps: int,
    output_dir: Path,
    modes: list[int] | None = None,
    raw_path: Path = DEFAULT_RAW,
    cln_path: Path | None = DEFAULT_CLN,
    shared_basis_path: Path = DEFAULT_SHARED_BASIS,
    rep_pod_path: Path = DEFAULT_REP13_POD,
    scaler_cache: Path | None = DEFAULT_SCALER_CACHE,
    disc_frame: int | None = None,
    title_tag: str = "0p20 Rep 13",
) -> dict[str, Any]:
    """Execute projection, standardization, and plotting for a given frame window."""
    if modes is None:
        modes = DEFAULT_MODES

    output_dir.mkdir(parents=True, exist_ok=True)
    last_step = start_step + timesteps - 1
    print(f"\nProcessing window: frames [{start_step}, {last_step}] ({timesteps} steps)...")

    # 1. Load basis and temporal mean field
    with h5py.File(shared_basis_path, "r") as f_sh, h5py.File(rep_pod_path, "r") as f_rep:
        rank = 20
        shared_modes = np.asarray(f_sh["pod/modes"][:rank], dtype=np.float64).reshape(rank, -1)
        t_mean = np.asarray(f_rep["preprocessing/temporal_mean_field"], dtype=np.float64).reshape(-1)

    # 2. Project raw and cleaned windows
    t0 = time.perf_counter()
    time_raw, raw_coords = project_frames_window(raw_path, start_step, timesteps, shared_modes, t_mean, restore=False)
    if cln_path is not None and cln_path.is_file():
        time_cln, cln_coords = project_frames_window(cln_path, start_step, timesteps, shared_modes, t_mean, restore=False)
    else:
        # Restore on-the-fly directly from raw frames
        time_cln, cln_coords = project_frames_window(raw_path, start_step, timesteps, shared_modes, t_mean, restore=True)
    print(f"Projected {timesteps} frames in {time.perf_counter() - t0:.3f} s.")

    # 3. Standardize coordinates using fitted scaler
    mean, scale = load_or_fit_standardization(rank=rank, scaler_cache=scaler_cache)
    raw_norm = (raw_coords - mean) / scale
    cln_norm = (cln_coords - mean) / scale

    # Time relative in milliseconds
    time_rel_ms = (time_raw - time_raw[0]) * 1e3
    disc_ms = ((time_raw[disc_frame - start_step] - time_raw[0]) * 1e3) if (disc_frame is not None and start_step <= disc_frame <= last_step) else None

    # Title prefix
    title_prefix = f"{title_tag}: Frames {start_step}–{last_step}"
    if disc_frame is not None:
        title_prefix += f" (Discontinuity @ Frame {disc_frame})"

    # 4. Generate plots
    print("Generating trajectory overlay plot...")
    plot_trajectory_overlay(
        time_rel_ms,
        raw_norm,
        cln_norm,
        modes,
        output_dir / "pod_trajectories_overlay.png",
        f"{title_prefix} — Normalized State Trajectories a_k(t)",
        disc_frame_ms=disc_ms,
    )

    print("Generating increments comparison plot...")
    plot_increments_comparison(
        time_rel_ms,
        raw_norm,
        cln_norm,
        modes,
        output_dir / "pod_increments_comparison.png",
        f"{title_prefix} — Lag-1 State Increments Δa_k(t)",
        lag=1,
        disc_frame_ms=disc_ms,
    )

    print("Generating phase portraits plot...")
    plot_phase_portraits(
        raw_norm,
        cln_norm,
        output_dir / "pod_phase_portraits.png",
        f"{title_prefix} — 2D Phase Space Orbits",
    )

    print("Generating Mahalanobis anomaly distance plot...")
    plot_anomaly_scores(
        time_rel_ms,
        raw_norm,
        cln_norm,
        output_dir / "pod_mahalanobis_anomaly.png",
        f"{title_prefix} — Increment Mahalanobis Distance",
        lag=1,
        disc_frame_ms=disc_ms,
    )

    # 5. Save NPZ arrays
    np.savez_compressed(
        output_dir / "pod_trajectories.npz",
        frames=np.arange(start_step, start_step + timesteps),
        time_sec=time_raw,
        time_rel_ms=time_rel_ms,
        raw_coords=raw_coords,
        cln_coords=cln_coords,
        raw_norm=raw_norm,
        cln_norm=cln_norm,
        modes=np.asarray(modes),
        mean=mean,
        scale=scale,
    )

    # Compute key stats for summary
    max_norm_diff = {f"mode_{m}": float(np.max(np.abs(raw_norm[:, m] - cln_norm[:, m]))) for m in modes}
    raw_lag1 = np.abs(raw_norm[1:] - raw_norm[:-1])
    cln_lag1 = np.abs(cln_norm[1:] - cln_norm[:-1])
    max_step_raw = {f"mode_{m}": float(np.max(raw_lag1[:, m])) for m in modes}
    max_step_cln = {f"mode_{m}": float(np.max(cln_lag1[:, m])) for m in modes}

    stats = {
        "start_step": start_step,
        "last_step": last_step,
        "timesteps": timesteps,
        "discontinuity_frame": disc_frame,
        "max_trajectory_diff_sigma": max_norm_diff,
        "max_lag1_step_raw_sigma": max_step_raw,
        "max_lag1_step_cln_sigma": max_step_cln,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Finished window [{start_step}, {last_step}]! Saved to {output_dir}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1500, help="Start frame index")
    parser.add_argument("--timesteps", type=int, default=150, help="Number of frames to plot")
    parser.add_argument("--disc-frame", type=int, default=1553, help="Known discontinuity frame to annotate")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison_0p20_cleaned/trajectories_frame1553"))
    parser.add_argument("--modes", type=int, nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cln-path", type=Path, default=DEFAULT_CLN)
    parser.add_argument("--shared-basis-path", type=Path, default=DEFAULT_SHARED_BASIS)
    parser.add_argument("--rep-pod-path", type=Path, default=DEFAULT_REP13_POD)
    parser.add_argument("--scaler-cache", type=Path, default=DEFAULT_SCALER_CACHE)
    parser.add_argument("--title-tag", type=str, default="0p20 Rep 13")
    args = parser.parse_args()

    run_trajectory_comparison(
        start_step=args.start,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
        modes=args.modes,
        raw_path=args.raw_path,
        cln_path=args.cln_path,
        shared_basis_path=args.shared_basis_path,
        rep_pod_path=args.rep_pod_path,
        scaler_cache=args.scaler_cache,
        disc_frame=args.disc_frame,
        title_tag=args.title_tag,
    )


if __name__ == "__main__":
    main()

