"""Comprehensive diagnostic analysis of latent oscillator envelopes in capillary wave turbulence.
Analyzes:
1. Envelope cross-correlation matrix (synchronous co-activation).
2. Time-lagged cross-correlation curves (causal energy cascades).
3. Amplitude probability density functions and non-Gaussian statistics (saturation vs intermittency).
4. Synchronized time-domain burst traces.
5. Three-wave triad bicoherence (phase locking).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank


def extract_envelopes_multi_rep(data_loader, reps, models_bank, fs, edge):
    """Extract complex envelopes z_k(t) and amplitudes A_k(t) across multiple repetitions."""
    print(f'Preloading {len(reps)} repetitions into memory...')
    reps_coords = {}
    for r in reps:
        _, c = data_loader.read_coordinates(r, 0, data_loader.frame_counts[r])
        reps_coords[r] = c

    all_envelopes = []
    all_amps = []
    all_basebands = []
    centers = []

    for idx, m in enumerate(models_bank):
        low, high = np.asarray(m['edges_hz'], dtype=float)
        order = int(m.get('filter_order', 4))
        sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
        modes = np.asarray(m['modes'])
        f_center = float(m.get('frequency_hz', (low + high) / 2))
        centers.append(f_center)
        omega_k = 2 * np.pi * f_center

        band_z = []
        band_amp = []
        band_base = []

        for r in reps:
            coords = reps_coords[r]
            filt = sosfiltfilt(sos, coords, axis=0, padlen=edge)
            analytic = hilbert(filt, axis=0)
            z_k = (analytic @ modes.conj())[edge:-edge, 0]
            t_crop = np.arange(len(z_k)) / fs
            z_base = z_k * np.exp(-1j * omega_k * t_crop)

            band_z.append(z_k)
            band_amp.append(np.abs(z_k))
            band_base.append(z_base)

        all_envelopes.append(np.concatenate(band_z, axis=0))
        all_amps.append(np.concatenate(band_amp, axis=0))
        all_basebands.append(np.concatenate(band_base, axis=0))

    return (np.stack(all_envelopes, axis=1),
            np.stack(all_amps, axis=1),
            np.stack(all_basebands, axis=1),
            np.array(centers))


def run_diagnostics(experiment, osc_dir, output_dir, n_reps=4, downsample_hz=1000.0):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f'Initializing SharedPODTrainingData for {experiment}...')
    data_loader = SharedPODTrainingData(experiment, 10, spatial_mean_highpass_hz=50.0)
    fs = 1.0 / data_loader.native_dt
    edge = int(round(0.1 * fs))

    # Select representative repetitions (mix of training and validation)
    reps = list(data_loader.train_repetitions[:n_reps // 2]) + list(data_loader.validation_repetitions[:n_reps // 2])
    if len(reps) < n_reps:
        reps = list(data_loader.train_repetitions[:n_reps])
    print(f'Selected repetitions: {reps}')

    print(f'Loading oscillator bank from {osc_dir}...')
    _, models_bank = load_bank(osc_dir)
    n_bands = len(models_bank)

    print(f'Extracting empirical envelopes across {n_bands} bands over {len(reps)} repetitions...')
    envelopes, amps, baseband, centers = extract_envelopes_multi_rep(data_loader, reps, models_bank, fs, edge)

    # Downsample amplitudes for correlation and envelope analysis
    step = int(round(fs / downsample_hz))
    amps_sub = amps[::step]
    t_sub = (np.arange(len(amps_sub)) * step) / fs
    T_sec = len(amps) / fs
    print(f'Extracted {len(amps)} raw frames ({T_sec:.2f} s total duration).')
    print(f'Downsampled to {downsample_hz} Hz ({len(amps_sub)} frames).')

    # Simulate independent linear oscillator bank as baseline
    print('Simulating independent linear oscillator baseline...')
    sim_amps = []
    sim_len = min(len(amps), 300000)
    for idx, m in enumerate(models_bank):
        raw = simulate_complex_band(m, sim_len + 2 * edge, paths=1, seed=1041 + idx)[0, edge:-edge, 0]
        sim_amps.append(np.abs(raw)[::step])
    sim_amps = np.stack(sim_amps, axis=1)

    # ----------------------------------------------------
    # 1. Amplitude Cross-Correlation Matrices (Empirical vs Simulated)
    # ----------------------------------------------------
    corr_emp = np.corrcoef(amps_sub, rowvar=False)
    corr_sim = np.corrcoef(sim_amps, rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    band_labels = [f'{i}: {c:.0f}Hz' for i, c in enumerate(centers)]

    im0 = axes[0].imshow(corr_emp, vmin=-0.4, vmax=0.6, cmap='coolwarm', origin='lower')
    axes[0].set_title(f'{experiment} Empirical Data: Envelope Correlations\n(Notice Strong Inter-Band Coupling)', fontsize=11, fontweight='bold')
    axes[0].set_xticks(range(n_bands))
    axes[0].set_xticklabels(band_labels, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(n_bands))
    axes[0].set_yticklabels(band_labels, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Annotate significant correlations (> 0.20 or < -0.20)
    for i in range(n_bands):
        for j in range(n_bands):
            if i != j and abs(corr_emp[i, j]) >= 0.20:
                axes[0].text(j, i, f'{corr_emp[i, j]:.2f}', ha='center', va='center',
                             fontsize=7.5, fontweight='bold', color='black')

    im1 = axes[1].imshow(corr_sim, vmin=-0.4, vmax=0.6, cmap='coolwarm', origin='lower')
    axes[1].set_title('Independent Linear Oscillators (Current Model)\n(Zero Coupling: Off-Diagonals ~ 0)', fontsize=11, fontweight='bold')
    axes[1].set_xticks(range(n_bands))
    axes[1].set_xticklabels(band_labels, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(range(n_bands))
    axes[1].set_yticklabels(band_labels, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(output_path / 'envelope_correlation_matrix.png', dpi=200)
    plt.close(fig)

    # ----------------------------------------------------
    # 2. Time-Lagged Cross-Correlation (Causal Cascade Propagation)
    # ----------------------------------------------------
    print('Computing time-lagged cross-correlations...')
    max_lag_ms = 150
    lags = np.arange(-max_lag_ms, max_lag_ms + 1)

    pairs_to_test = [
        (0, 1, 'B0 (81 Hz Heave) -> B1 (137 Hz Slosh)'),
        (1, 8, 'B1 (137 Hz Slosh) -> B8 (752 Hz Capillary Dipole)'),
        (0, 8, 'B0 (81 Hz Heave) -> B8 (752 Hz Capillary Dipole)'),
        (8, 9, 'B8 (752 Hz Dipole) -> B9 (861 Hz Quadrupole)'),
        (0, 2, 'B0 (81 Hz Heave) -> B2 (207 Hz Slosh)'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    lag_results = {}

    for i, j, label in pairs_to_test:
        ai = (amps_sub[:, i] - np.mean(amps_sub[:, i])) / np.std(amps_sub[:, i])
        aj = (amps_sub[:, j] - np.mean(amps_sub[:, j])) / np.std(amps_sub[:, j])
        N = len(ai)

        xcorrs = []
        for lag in lags:
            if lag >= 0:
                c = np.mean(ai[:N - lag] * aj[lag:])
            else:
                c = np.mean(ai[-lag:] * aj[:N + lag])
            xcorrs.append(c)
        xcorrs = np.array(xcorrs)

        peak_idx = np.argmax(xcorrs)
        peak_lag = lags[peak_idx]
        peak_val = xcorrs[peak_idx]
        lag_results[label] = dict(peak_lag_ms=int(peak_lag), peak_correlation=float(peak_val), lag_zero_corr=float(xcorrs[max_lag_ms]))

        line, = ax.plot(lags, xcorrs, lw=1.8, label=f'{label} [Peak: {peak_val:.2f} at +{peak_lag}ms]')
        ax.plot(peak_lag, peak_val, 'o', color=line.get_color(), markersize=6)

    ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Zero Lag (Simultaneous)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set(xlabel='Time Lag [ms] (Positive = Low Frequency Leads High Frequency)',
           ylabel='Normalized Envelope Cross-Correlation',
           title=f'{experiment} Directional Energy Cascade: Causal Propagation Across Envelopes')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    fig.savefig(output_path / 'envelope_lag_cascade.png', dpi=200)
    plt.close(fig)

    # ----------------------------------------------------
    # 3. Probability Density Functions: Limit Cycle Saturation vs Bursts
    # ----------------------------------------------------
    print('Computing envelope distributions and statistics...')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    axes = axes.flatten()

    selected_bands = [0, 1, 6, 10]
    stats_dict = {}

    r_x = np.linspace(0, 3.5, 300)
    s_ray = np.sqrt(2 / np.pi)
    rayleigh_pdf = (r_x / s_ray**2) * np.exp(-r_x**2 / (2 * s_ray**2))

    for ax_idx, b_idx in enumerate(selected_bands):
        ax = axes[ax_idx]
        amp = amps_sub[:, b_idx]
        norm_amp = amp / np.mean(amp)

        sk = skew(amp)
        kt = kurtosis(amp)
        stats_dict[f'Band_{b_idx}'] = dict(frequency_hz=float(centers[b_idx]),
                                           mean_amp=float(np.mean(amp)),
                                           skewness=float(sk),
                                           excess_kurtosis=float(kt))

        counts, bins, _ = ax.hist(norm_amp, bins=60, density=True, alpha=0.6,
                                  color='tab:blue', label='Empirical Envelope')
        ax.plot(r_x, rayleigh_pdf, 'r--', lw=2.0, label='Linear OU (Rayleigh: Skew=0.63, Kurt=0.25)')

        behavior = 'Saturated Limit Cycle (Kurtosis < 0)' if kt < 0 else ('Intermittent Burst Spikes (Kurtosis > 2)' if kt > 1.5 else 'Linear-like')
        ax.set_title(f'Band {b_idx} ({centers[b_idx]:.1f} Hz) - {behavior}\nSkewness: {sk:.2f} | Excess Kurtosis: {kt:.2f}', fontsize=10)
        ax.set(xlabel='Normalized Amplitude (A / <A>)', ylabel='Probability Density', xlim=[0, 3.5])
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.savefig(output_path / 'envelope_distributions.png', dpi=200)
    plt.close(fig)

    # ----------------------------------------------------
    # 4. Synchronized Envelope Time-Series Snippet (1.0 second zoom)
    # ----------------------------------------------------
    print('Plotting synchronized burst traces...')
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, layout='constrained')

    # Zoom window from 0.2s to 1.2s (1.0 second window)
    t_mask = (t_sub >= 0.2) & (t_sub <= 1.2)
    t_plot = (t_sub[t_mask] - 0.2) * 1000 # ms

    plot_bands = [
        (0, 'B0 (81 Hz Heave)', 'tab:blue'),
        (1, 'B1 (137 Hz Slosh)', 'tab:orange'),
        (8, 'B8 (752 Hz Capillary Dipole)', 'tab:green'),
        (10, 'B10 (987 Hz High Ripple)', 'tab:purple')
    ]

    for ax_idx, (b_idx, name, color) in enumerate(plot_bands):
        ax = axes[ax_idx]
        norm_trace = amps_sub[t_mask, b_idx] / np.mean(amps_sub[:, b_idx])
        ax.plot(t_plot, norm_trace, color=color, lw=1.5, label=name)
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
        ax.set_ylabel('Amp / <Amp>', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time [ms]', fontsize=10)
    axes[0].set_title(f'{experiment} Synchronized Envelope Waveforms: Observable Energy Bursts Across Frequencies', fontsize=12)
    fig.savefig(output_path / 'envelope_time_traces.png', dpi=200)
    plt.close(fig)

    # ----------------------------------------------------
    # 5. Three-Wave Triad Phase Coupling (Bicoherence)
    # ----------------------------------------------------
    print('Computing three-wave triad bicoherence...')
    triads = [
        (0, 1, 2, '81 + 138 ≈ 207 Hz (B0 + B1 ≈ B2)'),
        (0, 3, 4, '81 + 283 ≈ 363 Hz (B0 + B3 ≈ B4)'),
        (1, 4, 6, '138 + 363 ≈ 550 Hz (B1 + B4 ≈ B6)'),
        (1, 6, 8, '138 + 550 ≈ 754 Hz (B1 + B6 ≈ B8)'),
        (3, 5, 8, '283 + 458 ≈ 754 Hz (B3 + B5 ≈ B8)'),
    ]

    triad_results = []
    for i, j, k, name in triads:
        zi = envelopes[:, i]
        zj = envelopes[:, j]
        zk = envelopes[:, k]

        bispec = np.mean(zi * zj * np.conj(zk))
        denom = np.sqrt(np.mean(np.abs(zi * zj)**2) * np.mean(np.abs(zk)**2))
        bicoherence = float(np.abs(bispec) / max(denom, 1e-20))
        triad_results.append(dict(name=name, bicoherence=bicoherence,
                                  f_sum_hz=float(centers[i] + centers[j]),
                                  f_target_hz=float(centers[k])))
        print(f'Triad {name}: Bicoherence = {bicoherence:.4f}')

    summary = dict(
        experiment=experiment,
        n_repetitions=len(reps),
        total_duration_sec=float(T_sec),
        downsample_hz=downsample_hz,
        centers_hz=centers.tolist(),
        correlation_matrix_empirical=corr_emp.tolist(),
        correlation_matrix_simulated_independent=corr_sim.tolist(),
        lag_cascade_results=lag_results,
        envelope_statistics=stats_dict,
        three_wave_triads=triad_results
    )

    with open(output_path / 'envelope_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nDiagnostics successfully generated and saved to {output_path}!')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='0p08')
    parser.add_argument('--oscillator-dir', type=Path, default='best_models/0p08/oscillators')
    parser.add_argument('--output-dir', type=Path, default='best_models/0p08/evaluation/envelope_diagnostics')
    parser.add_argument('--n-reps', type=int, default=4)
    args = parser.parse_args()

    run_diagnostics(args.experiment, args.oscillator_dir, args.output_dir, n_reps=args.n_reps)
