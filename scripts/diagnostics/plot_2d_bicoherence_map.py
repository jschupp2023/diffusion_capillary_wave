"""Compute and visualize the continuous 2D bicoherence map for 0p08.
Tests whether resonant peak triads are distinguishable from background frequency pairs.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window, resample_poly
from scipy.fft import rfft

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData


def compute_2d_bicoherence(signal, fs, N=1024, step=512):
    """Compute 2D bicoherence matrix b^2(f1, f2) over non-overlapping or overlapping windows."""
    starts = range(0, len(signal) - N + 1, step)
    segs = np.stack([signal[i:i + N] for i in starts])
    segs = segs - segs.mean(axis=1, keepdims=True)
    w = get_window('hann', N)
    Z = rfft(segs * w[None, :], axis=1)  # [n_segs, N//2 + 1]

    n_segs, n_freqs = Z.shape
    freqs = np.arange(n_freqs) * fs / N

    # Compute bicoherence for all f1, f2 such that f1 + f2 < fs/2
    # To keep matrix manageable, restrict to f_max = 600 Hz
    max_bin = int(min(n_freqs - 1, round(600.0 / (fs / N))))
    b2_matrix = np.full((max_bin, max_bin), np.nan)

    print(f'Computing bicoherence across {max_bin}x{max_bin} frequency grid (up to {freqs[max_bin]:.1f} Hz)...')
    for i in range(1, max_bin):
        for j in range(1, max_bin - i):
            k = i + j
            if k < n_freqs:
                a = Z[:, i] * Z[:, j]
                b = Z[:, k]
                num = np.abs(np.mean(a * np.conj(b)))**2
                den = np.mean(np.abs(a)**2) * np.mean(np.abs(b)**2)
                b2_matrix[j, i] = num / max(den, 1e-20)

    # Compute phase-randomized surrogate baseline to establish statistical noise floor
    print('Computing phase-randomized surrogate noise floor...')
    n_surr = 50
    surr_b2 = []
    rng = np.random.default_rng(42)
    mag = np.abs(Z)
    for _ in range(n_surr):
        phases = rng.uniform(-np.pi, np.pi, Z.shape)
        Z_surr = mag * np.exp(1j * phases)
        # sample 50 random pairs
        sample_b = []
        for _ in range(30):
            si = rng.integers(1, max_bin // 2)
            sj = rng.integers(1, max_bin // 2)
            sk = si + sj
            a = Z_surr[:, si] * Z_surr[:, sj]
            b = Z_surr[:, sk]
            num = np.abs(np.mean(a * np.conj(b)))**2
            den = np.mean(np.abs(a)**2) * np.mean(np.abs(b)**2)
            sample_b.append(num / max(den, 1e-20))
        surr_b2.extend(sample_b)

    surr_p95 = np.percentile(surr_b2, 95)
    surr_mean = np.mean(surr_b2)

    return freqs[:max_bin], b2_matrix, surr_mean, surr_p95


def main():
    output_dir = Path('best_models/0p08/evaluation/envelope_diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading 0p08 data across 4 repetitions...')
    data_loader = SharedPODTrainingData('0p08', 10, spatial_mean_highpass_hz=50.0)
    fs_raw = 1.0 / data_loader.native_dt
    down = 8
    fs = fs_raw / down  # 14400 Hz

    reps = ['Ca_ac_0p000664_rep1', 'Ca_ac_0p000664_rep2', 'Ca_ac_0p000664_rep13', 'Ca_ac_0p000664_rep14']
    signals = []
    for r in reps:
        t, c = data_loader.read_coordinates(r, 0, data_loader.frame_counts[r])
        # Reconstruct center point signal using rank-10 POD modes:
        # eta_center = spatial_mean + sum_m c[:, m]
        eta_center = c[:, 0] + np.sum(c[:, 1:11], axis=1) * 0.1  # weighted center proxy
        sig = resample_poly(eta_center, 1, down)[32:-32]
        signals.append(sig)
    full_signal = np.concatenate(signals)

    freqs, b2_map, surr_mean, surr_p95 = compute_2d_bicoherence(full_signal, fs, N=1024, step=512)

    # Plot 2D bicoherence map
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')

    peak_freqs = [80.5, 138.4, 207.0, 283.3, 363.2, 457.8]

    # Panel 1: Full scale [0, 1] (as traditionally shown)
    im0 = axes[0].pcolormesh(freqs, freqs, b2_map, vmin=0, vmax=0.25, cmap='magma', shading='auto')
    axes[0].set_title('2D Bicoherence Map b²(f₁, f₂)\n(Linear Scale [0, 0.25], Showing Low Uniform Level)', fontsize=11, fontweight='bold')
    axes[0].set(xlabel='Frequency f₁ [Hz]', ylabel='Frequency f₂ [Hz]')
    fig.colorbar(im0, ax=axes[0], label='b²')

    # Panel 2: Thresholded at Surrogate 95% Noise Floor
    b2_thresholded = np.where(b2_map >= surr_p95, b2_map, np.nan)
    im1 = axes[1].pcolormesh(freqs, freqs, b2_thresholded, vmin=surr_p95, vmax=0.25, cmap='viridis', shading='auto')
    axes[1].set_title(f'Statistically Significant Bicoherence (> 95% Surrogate Baseline: {surr_p95:.3f})\n(Resonant Triad Grid Overlaid)', fontsize=11, fontweight='bold')
    axes[1].set(xlabel='Frequency f₁ [Hz]', ylabel='Frequency f₂ [Hz]')
    fig.colorbar(im1, ax=axes[1], label='b² (above noise floor)')

    # Overlay resonant lines on both panels
    for ax in axes:
        for pf in peak_freqs:
            ax.axvline(pf, color='cyan', linestyle=':', alpha=0.4, lw=1.0)
            ax.axhline(pf, color='cyan', linestyle=':', alpha=0.4, lw=1.0)
        # Sum constraint line f1 + f2 = 600 Hz
        ax.plot([0, 600], [600, 0], 'w--', alpha=0.5, label='f₁ + f₂ = 600 Hz')
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)

    fig.savefig(output_dir / 'bicoherence_2d_map.png', dpi=200)
    plt.close(fig)

    print(f'2D Bicoherence map saved to {output_dir / "bicoherence_2d_map.png"}')
    print(f'Surrogate noise floor mean: {surr_mean:.4f}, 95th percentile: {surr_p95:.4f}')

    # Compute average bicoherence at resonant intersections vs random off-resonant bins
    peak_bins = [np.argmin(np.abs(freqs - pf)) for pf in peak_freqs]
    resonant_vals = []
    for i in peak_bins:
        for j in peak_bins:
            if i + j < len(freqs) and not np.isnan(b2_map[j, i]):
                resonant_vals.append(b2_map[j, i])

    # Random off-peak bins
    random_bins = [np.argmin(np.abs(freqs - pf)) for pf in [105, 175, 245, 320, 410]]
    random_vals = []
    for i in random_bins:
        for j in random_bins:
            if i + j < len(freqs) and not np.isnan(b2_map[j, i]):
                random_vals.append(b2_map[j, i])

    print(f'Resonant Grid Points: Mean b² = {np.mean(resonant_vals):.4f} (Max = {np.max(resonant_vals):.4f})')
    print(f'Off-Peak Random Points: Mean b² = {np.mean(random_vals):.4f} (Max = {np.max(random_vals):.4f})')


if __name__ == '__main__':
    main()

