"""Compute and plot 50 Hz high-passed Autocorrelation Functions (ACF) for all coordinates.
Compares generated vs reference trajectories for Spatial Mean (coord 0) and POD Modes 1-10.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.fft import rfft, irfft, next_fast_len

def compute_mean_path_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute path-averaged normalized autocorrelation function.
    x: shape (num_paths, time_steps, num_coords)
    Returns: shape (max_lag + 1, num_coords)
    """
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=1, keepdims=True)
    n = centered.shape[1]
    nfft = next_fast_len(2 * n - 1)
    spec = rfft(centered, n=nfft, axis=1)
    sums = irfft(spec * spec.conj(), n=nfft, axis=1)[:, :max_lag + 1]
    energy = np.sum(centered**2, axis=1)
    acf = np.divide(sums, energy[:, None, :], out=np.full_like(sums, np.nan), where=energy[:, None, :] > 0)
    return np.nanmean(acf, axis=0)

def find_crossing(acf: np.ndarray, threshold: float, t_ms: np.ndarray) -> float:
    idx = np.where(acf < threshold)[0]
    return float(t_ms[idx[0]]) if len(idx) > 0 else np.nan

def compute_integral_timescale(acf: np.ndarray, t_ms: np.ndarray) -> float:
    idx0 = np.where(acf <= 0)[0]
    cutoff = idx0[0] if len(idx0) > 0 else len(acf)
    dt = t_ms[1] - t_ms[0]
    return float(np.sum(acf[:cutoff]) * dt)

def generate_acf_diagnostics(rollout_npz: Path, output_dir: Path, title_prefix: str, fs: float = 115200.00439453142):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(rollout_npz)
    ref = data['reference'] # (num_paths, time, coords)
    gen = data['trajectories'] # (num_paths, 1, time, coords) or (num_paths, time, coords)
    if gen.ndim == 4:
        gen = gen[:, 0, :, :]
    
    num_paths, num_time, num_coords = ref.shape
    coord_names = ['Spatial Mean'] + [f'POD Mode {i}' for i in range(1, num_coords)]
    
    # 4th order zero-phase 50 Hz high-pass filter
    sos50 = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
    ref_hp = sosfiltfilt(sos50, ref, axis=1)
    gen_hp = sosfiltfilt(sos50, gen, axis=1)
    
    # 50 ms max lag
    max_lag = int(round(fs * 0.050))
    t_ms = np.arange(max_lag + 1) / fs * 1000.0
    
    acf_ref = compute_mean_path_acf(ref_hp, max_lag)
    acf_gen = compute_mean_path_acf(gen_hp, max_lag)
    
    # Quantitative summary
    metrics = {
        'fs': fs,
        'highpass_cutoff_hz': 50.0,
        'max_lag_ms': 50.0,
        'coordinates': {}
    }
    
    tau_1e_ref = []
    tau_1e_gen = []
    tau_zero_ref = []
    tau_zero_gen = []
    rmse_list = []
    
    for i, name in enumerate(coord_names):
        t1e_r = find_crossing(acf_ref[:, i], 1.0 / np.e, t_ms)
        t1e_g = find_crossing(acf_gen[:, i], 1.0 / np.e, t_ms)
        t0_r = find_crossing(acf_ref[:, i], 0.0, t_ms)
        t0_g = find_crossing(acf_gen[:, i], 0.0, t_ms)
        tint_r = compute_integral_timescale(acf_ref[:, i], t_ms)
        tint_g = compute_integral_timescale(acf_gen[:, i], t_ms)
        # RMSE over first 15 ms
        idx15 = int(round(fs * 0.015))
        rmse = float(np.sqrt(np.mean((acf_gen[:idx15, i] - acf_ref[:idx15, i])**2)))
        
        tau_1e_ref.append(t1e_r)
        tau_1e_gen.append(t1e_g)
        tau_zero_ref.append(t0_r)
        tau_zero_gen.append(t0_g)
        rmse_list.append(rmse)
        
        metrics['coordinates'][name] = {
            'tau_1e_ref_ms': t1e_r,
            'tau_1e_gen_ms': t1e_g,
            'tau_1e_ratio': t1e_g / t1e_r if not np.isnan(t1e_r) and t1e_r > 0 else None,
            'tau_zero_ref_ms': t0_r,
            'tau_zero_gen_ms': t0_g,
            'tau_zero_ratio': t0_g / t0_r if not np.isnan(t0_r) and t0_r > 0 else None,
            'tau_integral_ref_ms': tint_r,
            'tau_integral_gen_ms': tint_g,
            'acf_rmse_15ms': rmse
        }
    
    with open(output_dir / 'acf_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    np.savez_compressed(
        output_dir / 'acf_curves.npz',
        t_ms=t_ms,
        acf_ref=acf_ref,
        acf_gen=acf_gen,
        coord_names=coord_names
    )
    
    # -------------------------------------------------------------
    # Plot 1: Full 11 Coordinates Overview (0 to 30 ms) (4x3 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=False)
    axes_flat = axes.flatten()
    
    idx30 = int(round(fs * 0.030))
    t30 = t_ms[:idx30]
    
    for i in range(num_coords):
        ax = axes_flat[i]
        ax.plot(t30, acf_ref[:idx30, i], color='#2c3e50', linewidth=2.0, label='Reference (Ground Truth)')
        ax.plot(t30, acf_gen[:idx30, i], color='#e74c3c', linewidth=2.0, linestyle='--', label='Generated (Model)')
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axhline(1.0 / np.e, color='#27ae60', linestyle='-.', linewidth=0.8, alpha=0.6, label='1/e threshold')
        
        t1e_r = tau_1e_ref[i]
        t1e_g = tau_1e_gen[i]
        info_str = f"$\\tau_{{1/e}}$ Ref: {t1e_r:.2f} ms | Gen: {t1e_g:.2f} ms"
        ax.set_title(f"{coord_names[i]}\n{info_str}", fontsize=11, fontweight='bold', pad=4)
        ax.set_ylabel("Normalized ACF", fontsize=9)
        ax.set_xlabel("Lag [ms]", fontsize=9)
        ax.set_xlim([0, 30.0])
        ax.set_ylim([-0.35, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
            
    # Subplot 11: Summary Bar Chart of tau_1e
    ax_bar = axes_flat[11]
    x_indices = np.arange(num_coords)
    width = 0.38
    ax_bar.bar(x_indices - width/2, tau_1e_ref, width, label='Reference', color='#2c3e50', alpha=0.85)
    ax_bar.bar(x_indices + width/2, tau_1e_gen, width, label='Generated', color='#e74c3c', alpha=0.85)
    ax_bar.set_xticks(x_indices)
    labels = ['Mean'] + [f'P{i}' for i in range(1, num_coords)]
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_ylabel(r"Decorrelation Time $\tau_{1/e}$ [ms]", fontsize=9)
    ax_bar.set_title(r"Timescale Comparison ($\tau_{1/e}$)", fontsize=11, fontweight='bold', pad=4)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    fig.suptitle(f"{title_prefix}: Autocorrelation Functions (50 Hz High-Passed)", fontsize=15, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / 'acf_all_coordinates_30ms.png', dpi=160)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Plot 2: Fast Turbulent Decorrelation Zoom (0 to 6 ms)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=False)
    axes_flat = axes.flatten()
    
    idx6 = int(round(fs * 0.006))
    t6 = t_ms[:idx6]
    
    for i in range(num_coords):
        ax = axes_flat[i]
        ax.plot(t6, acf_ref[:idx6, i], color='#2c3e50', linewidth=2.0, label='Reference')
        ax.plot(t6, acf_gen[:idx6, i], color='#e74c3c', linewidth=2.0, linestyle='--', label='Generated')
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.axhline(1.0 / np.e, color='#27ae60', linestyle='-.', linewidth=0.8, alpha=0.6, label='1/e')
        
        t1e_r = tau_1e_ref[i]
        t1e_g = tau_1e_gen[i]
        ax.set_title(f"{coord_names[i]} (Zoom 0-6 ms)", fontsize=11, fontweight='bold', pad=4)
        ax.set_ylabel("Normalized ACF", fontsize=9)
        ax.set_xlabel("Lag [ms]", fontsize=9)
        ax.set_xlim([0, 6.0])
        ax.set_ylim([-0.35, 1.05])
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
            
    # Subplot 11: First Zero-Crossing Bar Chart
    ax_bar = axes_flat[11]
    ax_bar.bar(x_indices - width/2, tau_zero_ref, width, label='Reference', color='#2c3e50', alpha=0.85)
    ax_bar.bar(x_indices + width/2, tau_zero_gen, width, label='Generated', color='#e74c3c', alpha=0.85)
    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_ylabel(r"First Zero-Crossing $\tau_0$ [ms]", fontsize=9)
    ax_bar.set_title(r"First Zero-Crossing Comparison ($\tau_0$)", fontsize=11, fontweight='bold', pad=4)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    fig.suptitle(f"{title_prefix}: Short-Lag Turbulent Decorrelation (0 to 6 ms)", fontsize=15, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / 'acf_all_coordinates_zoom_6ms.png', dpi=160)
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Plot 3: Focused 4-Coordinate Dual Window (Spatial Mean, POD 1-3)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    focus_indices = [0, 1, 2, 3]
    
    for row, idx in enumerate(focus_indices):
        name = coord_names[idx]
        # Left: Zoom 0 to 6 ms
        ax_left = axes[row, 0]
        ax_left.plot(t6, acf_ref[:idx6, idx], color='#2c3e50', linewidth=2.2, label='Reference')
        ax_left.plot(t6, acf_gen[:idx6, idx], color='#e74c3c', linewidth=2.2, linestyle='--', label='Generated')
        ax_left.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)
        ax_left.axhline(1.0 / np.e, color='#27ae60', linestyle='-.', linewidth=0.8, label='1/e threshold')
        ax_left.set_title(f"{name} — Fast Decorrelation (0–6 ms)", fontsize=11, fontweight='bold')
        ax_left.set_ylabel("Normalized ACF", fontsize=10)
        ax_left.set_xlabel("Lag [ms]", fontsize=10)
        ax_left.set_xlim([0, 6.0])
        ax_left.set_ylim([-0.35, 1.05])
        ax_left.grid(True, linestyle='--', alpha=0.3)
        if row == 0:
            ax_left.legend(fontsize=8, loc='upper right')
            
        # Right: Extended 0 to 50 ms
        ax_right = axes[row, 1]
        ax_right.plot(t_ms, acf_ref[:, idx], color='#2c3e50', linewidth=2.0, label='Reference')
        ax_right.plot(t_ms, acf_gen[:, idx], color='#e74c3c', linewidth=2.0, linestyle='--', label='Generated')
        ax_right.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)
        ax_right.set_title(f"{name} — Wave Cycle Envelope (0–50 ms)", fontsize=11, fontweight='bold')
        ax_right.set_ylabel("Normalized ACF", fontsize=10)
        ax_right.set_xlabel("Lag [ms]", fontsize=10)
        ax_right.set_xlim([0, 50.0])
        ax_right.set_ylim([-0.35, 1.05])
        ax_right.grid(True, linestyle='--', alpha=0.3)
        if row == 0:
            ax_right.legend(fontsize=8, loc='upper right')
            
    fig.suptitle(f"{title_prefix}: Key Mode Decorrelation Dynamics (Spatial Mean & POD 1–3)", fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / 'acf_focus_key_modes.png', dpi=160)
    plt.close(fig)

    print(f"Generated ACF diagnostics successfully in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout', type=Path, required=True, help='Path to rollout.npz')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory for plots and metrics')
    parser.add_argument('--title', type=str, default='Hybrid Model', help='Title prefix for plots')
    args = parser.parse_args()
    generate_acf_diagnostics(args.rollout, args.output_dir, args.title)

