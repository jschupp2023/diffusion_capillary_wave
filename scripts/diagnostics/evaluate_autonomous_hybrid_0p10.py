"""Autonomous evaluation of self-regulating hybrid models on 0p10.

Evaluates WITHOUT any manual scaling hacks:
- Strictly unscaled diffusion: s_vec = 1.0 (no post-hoc tweaking)
- Strictly unscaled oscillator bank: bank_scales = 1.0 (no post-hoc tweaking)
- Zero digital noise shaping: pure white Gaussian noise (no high-shelf filter)

Computes:
1. Capillary deformation energy ratio (>50 Hz physical waves)
2. Modal variance ratios across all 11 coordinates
3. Spatial mean power ratios across frequency sub-bands
4. High-resolution Welch PSD and 11-peak contrast MAE (dB)
5. 3-Way comparison against Training Ensemble Mean (reps 1-12) and Validation rep 13
6. Comprehensive diagnostic plots
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import load_model
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData


def run_autonomous_rollout_0p10(model: NeuralSDE, init: torch.Tensor, hist: torch.Tensor,
                                horizon=50000, seed=41, device='cuda'):
    """Autonomous rollout for 0p10 model with strictly s_vec = 1.0 and unshaped noise."""
    generator = torch.Generator(device=device).manual_seed(seed)
    state = init[:, None, :].expand(-1, 1, -1).clone()
    history = hist[:, None, :model.config.history_steps].expand(-1, 1, -1, -1).clone()
    velocity = model._frame_velocity(state, history)
    step_size = model.config.lag_steps
    tcn_cache = model.new_tcn_cache()
    
    states = [state]
    for _ in range(horizon):
        base_mean, noise_factor = model.velocity_base_step_distribution(
            state, history, velocity, tcn_cache=tcn_cache)
        noise = torch.randn(state.shape, device=device, dtype=state.dtype, generator=generator)
        stochastic = torch.matmul(noise, noise_factor.T)
        velocity = base_mean + stochastic
        state = state + step_size * velocity
        history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
        states.append(state)
    return torch.stack(states, dim=2)


def evaluate_model_0p10(checkpoint_path: Path, osc_dir: Path, output_dir: Path,
                        horizon=50000, seed=41, device='cuda'):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Loading checkpoint {checkpoint_path}...')
    model, _ = load_model(str(checkpoint_path), device)
    model.eval()
    
    # Load 0p10 reference data
    data_dir = Path('runs/adaptive/neural_sde_0p10_local_excess_low77_psd_refinement_v1/data')
    manifest_path = data_dir / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    edge = int(manifest.get('edge_samples', 12564))
    fs = float(manifest.get('fs', 115200.0))
    scored_len = horizon - 2 * edge
    
    rem_rep13 = np.load(data_dir / 'Ca_ac_0p000830_rep13_remainder.npy')
    rem_rep14 = np.load(data_dir / 'Ca_ac_0p000830_rep14_remainder.npy')
    full_rep13 = np.load(data_dir / 'Ca_ac_0p000830_rep13_full.npy')
    full_rep14 = np.load(data_dir / 'Ca_ac_0p000830_rep14_full.npy')
    
    # Unscaled linear oscillator bank simulation (strictly bank_scales = 1.0)
    _, models_bank = load_bank(osc_dir)
    assert len(models_bank) == 11, f'Expected 11 bands, got {len(models_bank)}'
    
    overlap = 0.0578
    filters_bank = []
    for idx, m in enumerate(models_bank):
        low, high = np.asarray(m['edges_hz'], dtype=float)
        width = high - low
        new_low = max(10.0, low - overlap * width)
        new_high = high + overlap * width
        sos = butter(int(m.get('filter_order', 4)), [new_low, new_high], btype='bandpass', fs=fs, output='sos')
        filters_bank.append(sos)

    print('Simulating 11 0p10 linear complex oscillators (strictly bank_scales = 1.0)...')
    bank_draw = np.zeros((2, 1, scored_len, 11), dtype=np.float64)
    for idx, (m, sos) in enumerate(zip(models_bank, filters_bank)):
        raw = simulate_complex_band(m, scored_len + 2 * edge, paths=2, seed=1041 + idx)
        filt = sosfiltfilt(sos, raw, axis=1, padlen=edge)[:, edge:-edge, :]
        bank_draw += filt[:, None, :, :]

    h_steps = model.config.history_steps
    init = torch.tensor(np.stack([rem_rep13[edge + h_steps], rem_rep14[edge + h_steps]]), dtype=torch.float32, device=device)
    hist = torch.tensor(np.stack([rem_rep13[edge:edge + h_steps][::-1].copy(), rem_rep14[edge:edge + h_steps][::-1].copy()]), dtype=torch.float32, device=device)
    
    print('Running autonomous unscaled rollout for 0p10...')
    with torch.no_grad():
        gen_rem = run_autonomous_rollout_0p10(
            model, init, hist, horizon=scored_len, seed=seed, device=device
        ).cpu().numpy()

    # gen_rem: [2, 1, scored_len + 1, 11]
    gen_rem_cropped = gen_rem[:, :, 1:, :] # match scored_len
    hybrid = gen_rem_cropped + bank_draw # [2, 1, scored_len, 11]
    
    # Ground truth reference cropped to scored_len
    ref_full = np.stack([full_rep13[edge:edge + scored_len], full_rep14[edge:edge + scored_len]])
    
    # Capillary deformation energy calculation (>50 Hz waves)
    basis_path = '/home/jonas/ucsd_thesis/reduced_data/0p10/shared_pod/shared_pod_r1000.h5'
    with h5py.File(basis_path, 'r') as h5:
        pod_modes = h5['spatial_modes'][:10]
        x_m = h5['x_coords_m'][:]
        y_m = h5['y_coords_m'][:]
        
    gamma = 0.10 * 9.81
    opts = {'surface_tension': 0.0728, 'fluid_density': 1000.0}
    
    E_ref_raw = float(np.mean(compute_energies(ref_full, pod_modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']))
    E_gen_raw = float(np.mean(compute_energies(hybrid[:, 0], pod_modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']))
    
    sos50 = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
    ref_full_hp = ref_full.copy()
    ref_full_hp[..., 1:] = sosfiltfilt(sos50, ref_full_hp[..., 1:], axis=-2)
    hybrid_hp = hybrid.copy()
    hybrid_hp[..., 1:] = sosfiltfilt(sos50, hybrid_hp[..., 1:], axis=-2)
    
    E_ref_hp = float(np.mean(compute_energies(ref_full_hp, pod_modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']))
    E_gen_hp = float(np.mean(compute_energies(hybrid_hp[:, 0], pod_modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']))
    
    # Modal variance ratios
    var_ref = np.var(ref_full_hp.reshape(-1, 11), axis=0)
    var_gen = np.var(hybrid_hp[:, 0].reshape(-1, 11), axis=0)
    var_ratios = var_gen / var_ref
    
    # Welch PSD of Spatial Mean (coord 0)
    f_w, p_ref = welch(ref_full[0, :, 0], fs=fs, nperseg=8192, noverlap=4096)
    _, p_gen = welch(hybrid[0, 0, :, 0], fs=fs, nperseg=8192, noverlap=4096)
    
    # Sub-band power ratios for Spatial Mean
    bands = [
        (50.0, 100.0), (100.0, 300.0), (300.0, 1000.0), (1000.0, 1400.0),
        (1400.0, 2500.0), (2500.0, 5000.0), (5000.0, 20000.0)
    ]
    subband_ratios = []
    for low, high in bands:
        idx = (f_w >= low) & (f_w < high)
        p_r = np.sum(p_ref[idx])
        p_g = np.sum(p_gen[idx])
        subband_ratios.append(p_g / p_r)
        
    # 11 Resonant Peaks for 0p10
    peak_freqs = [77.13, 132.60, 206.83, 282.58, 364.30, 455.86, 547.13, 650.96, 756.86, 867.80, 982.36]
    peak_diffs_db = []
    peak_ratios = []
    for pf in peak_freqs:
        idx_p = np.argmin(np.abs(f_w - pf))
        ref_peak = p_ref[idx_p]
        gen_peak = p_gen[idx_p]
        diff_db = 10 * np.log10(gen_peak / ref_peak)
        peak_diffs_db.append(diff_db)
        peak_ratios.append(gen_peak / ref_peak)
        
    mae_db = float(np.mean(np.abs(peak_diffs_db)))
    max_db = float(np.max(np.abs(peak_diffs_db)))
    
    # Also compute Training Ensemble Mean PSD across reps 1-12
    data_prep = SharedPODTrainingData('0p10', 10, spatial_mean_highpass_hz=50.0, output_dtype='float64')
    train_psds = []
    for r_name in data_prep.train_repetitions:
        raw_train = data_prep.read_coordinates(r_name, edge, edge + scored_len)[1]
        _, p_tr = welch(raw_train[:, 0], fs=fs, nperseg=8192, noverlap=4096)
        train_psds.append(p_tr)
    mean_train_psd = np.mean(train_psds, axis=0)
    
    # Scorecard Output
    print('\n================== 0p10 AUTONOMOUS EVALUATION SCORECARD ==================')
    print('--- Capillary Deformation Energy ---')
    print(f'Raw Energy: Ref = {E_ref_raw:.3e} J | Gen = {E_gen_raw:.3e} J | Ratio = {E_gen_raw/E_ref_raw:.4f}')
    print(f'>50 Hz Physical Waves: Ref = {E_ref_hp:.3e} J | Gen = {E_gen_hp:.3e} J | Ratio = {E_gen_hp/E_ref_hp:.4f}')

    print('\n--- Modal Variance Ratios (Gen / Ref) ---')
    labels = ['Mean'] + [f'Mode {i}' for i in range(1, 11)]
    for name, vr in zip(labels, var_ratios):
        print(f'{name:>7s}: Ratio = {vr:.4f}')
    print(f'Mean Ratio across Modes 1-10: {np.mean(var_ratios[1:]):.4f}')

    print('\n--- Spatial Mean Power Ratios ---')
    for (low, high), ratio in zip(bands, subband_ratios):
        print(f'{low:6.0f} - {high:6.0f} Hz: Power Ratio = {ratio:.4f}')

    print(f'\n--- Resonant Peak Fit vs Val Rep 13 (MAE: {mae_db:.2f} dB, Max: {max_db:.2f} dB) ---')
    for pf, diff, rat in zip(peak_freqs, peak_diffs_db, peak_ratios):
        print(f'Peak {pf:6.1f} Hz: Ratio = {rat:.4f} ({diff:+.2f} dB)')

    print(f'\n--- Resonant Peak Fit vs Training Ensemble Mean ---')
    for pf in peak_freqs:
        idx_p = np.argmin(np.abs(f_w - pf))
        pt = mean_train_psd[idx_p]
        pg = p_gen[idx_p]
        diff_train_db = 10 * np.log10(pg / pt)
        print(f'Peak {pf:6.1f} Hz: Train={pt:10.2e} | Gen={pg:10.2e} | Ratio={pg/pt:.2f}x ({diff_train_db:+.2f} dB)')

    # High-frequency turbulent cascade
    print('\n--- High-Frequency Turbulent Cascade vs Train Mean ---')
    for f_check in [1200.0, 1500.0, 2000.0, 3000.0, 5000.0, 8000.0]:
        idx_c = np.argmin(np.abs(f_w - f_check))
        pt = mean_train_psd[idx_c]
        pg = p_gen[idx_c]
        print(f'Freq {f_check:6.0f} Hz: Train={pt:10.2e} | Gen={pg:10.2e} | Ratio={pg/pt:.2f}x')

    # Save summary JSON
    summary = {
        'checkpoint': str(checkpoint_path),
        'horizon': horizon,
        'scored_len': scored_len,
        'energy_ratio_hp': E_gen_hp / E_ref_hp,
        'energy_ratio_raw': E_gen_raw / E_ref_raw,
        'modal_variance_ratios': var_ratios.tolist(),
        'mean_pod_variance_ratio': float(np.mean(var_ratios[1:])),
        'subband_power_ratios': subband_ratios,
        'peak_diffs_db': peak_diffs_db,
        'peak_ratios': peak_ratios,
        'peak_mae_db': mae_db,
        'peak_max_db': max_db
    }
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot 1: Standard PSD (Model vs Val Rep 13)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_w, p_ref, label='Reference (Rep 13)', color='black', alpha=0.8, lw=1.5)
    ax.semilogy(f_w, p_gen, label='Autonomous Generated Hybrid (Unscaled)', color='crimson', lw=1.5)
    ax.set_xlim(30, 20000)
    ax.set_ylim(1e-3, 1e7)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power Spectral Density (µm²/Hz)', fontsize=11)
    ax.set_title('0p10 Autonomous Hybrid: Spatial Mean PSD vs Val Rep 13', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path / 'psd.png', dpi=200)
    plt.close(fig)

    # Plot 2: 3-Way Ensemble Comparison (Train Mean vs Model Gen vs Val Rep 13)
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = (f_w >= 40) & (f_w <= 20000)
    ax.loglog(f_w[mask], mean_train_psd[mask], color='black', lw=2.0, label='Train Mean (Reps 1-12)')
    ax.loglog(f_w[mask], p_gen[mask], color='tab:blue', lw=1.8, label='Autonomous Model (Unscaled)')
    ax.loglog(f_w[mask], p_ref[mask], color='tab:orange', lw=1.2, ls='--', alpha=0.7, label='Validation Rep 13')
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_ylabel('Center PSD [µm²/Hz]', fontsize=12)
    ax.set_title('0p10 Autonomous Model PSD vs Physical Ensemble', fontsize=13, fontweight='bold')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path / 'psd_ensemble_comparison.png', dpi=200)
    plt.close(fig)

    # Plot 3: Modal Variances
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(var_ratios))
    bars = ax.bar(x, var_ratios, color='royalblue', alpha=0.85, edgecolor='black', lw=0.7)
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Target Ratio 1.0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Generated / Reference Variance Ratio', fontsize=11)
    ax.set_title('0p10 Autonomous Hybrid: Modal Variance Ratios (>50 Hz)', fontsize=12, fontweight='bold')
    ax.grid(True, ls=':', alpha=0.5, axis='y')
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path / 'modal_variances.png', dpi=200)
    plt.close(fig)

    print(f'Evaluation complete! Results and plots saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--osc-dir', type=Path, default=Path('best_models/0p10/oscillators'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_model_0p10(args.checkpoint, args.osc_dir, args.output_dir,
                       horizon=args.horizon, seed=args.seed, device=device)
