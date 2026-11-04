"""Autonomous evaluation of self-regulating hybrid models on 0p08 held-out validation.

Evaluates WITHOUT any manual scaling hacks:
- Strictly unscaled diffusion: s_vec = 1.0 (no post-hoc tweaking)
- Strictly unscaled oscillator bank: bank_scales = 1.0 (no post-hoc tweaking)

Computes:
1. Capillary deformation energy ratio (>50 Hz physical waves)
2. Modal variance ratios across all 11 coordinates
3. Spatial mean power ratios across 7 frequency bands
4. High-resolution Welch PSD and 11-peak contrast MAE (dB)
5. Comprehensive diagnostic plots
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import load_model
from scripts.training.train_hybrid_aware_0p08 import OscillatorCoupledNeuralSDE


def run_autonomous_rollout_a(model: NeuralSDE, init: torch.Tensor, hist: torch.Tensor,
                             horizon=50000, seed=41, device='cuda'):
    """Autonomous rollout for Model A with strictly s_vec = 1.0."""
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


def run_autonomous_rollout_coupled(model: OscillatorCoupledNeuralSDE, init: torch.Tensor, hist: torch.Tensor,
                                   osc_init: torch.Tensor, osc_hist: torch.Tensor,
                                   osc_seq: torch.Tensor, horizon=50000, seed=41, device='cuda'):
    """Autonomous rollout for OscillatorCoupled Model B/C with strictly s_vec = 1.0."""
    generator = torch.Generator(device=device).manual_seed(seed)
    state = init[:, None, :].expand(-1, 1, -1).clone()
    history = hist[:, None, :model.config.history_steps].expand(-1, 1, -1, -1).clone()
    velocity = model.base._frame_velocity(state, history)
    
    osc_state = osc_init[:, None, :].expand(-1, 1, -1).clone()
    osc_history = osc_hist[:, None, :model.config.history_steps].expand(-1, 1, -1, -1).clone()
    osc_velocity = osc_state - osc_history[..., 0, :]
    
    step_size = model.config.lag_steps
    factor = math.sqrt(step_size) * model.diffusion_matrix()
    tcn_cache = model.new_tcn_cache()
    
    states = [state]
    for step in range(horizon):
        drift = model.drift(state, history, velocity, osc_state, osc_velocity, tcn_cache)
        mean = velocity + step_size * drift
        noise = torch.randn(state.shape, device=device, dtype=state.dtype, generator=generator)
        stochastic = torch.matmul(noise, factor.T)
        velocity = mean + stochastic
        state = state + step_size * velocity
        history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
        states.append(state)
        
        # Advance oscillator state from pre-simulated sequence
        next_osc = osc_seq[:, step:step+1]
        osc_velocity = next_osc - osc_state
        osc_history = torch.cat((osc_state.unsqueeze(-2), osc_history[..., :-1, :]), dim=-2)
        osc_state = next_osc
        
    return torch.stack(states, dim=2)


def evaluate_model(checkpoint_path: Path, osc_dir: Path, output_dir: Path,
                   horizon=50000, seed=41, device='cuda'):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Loading checkpoint {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load 0p08 validation reference data
    data_dir = Path('runs/adaptive/neural_sde_0p08_local_excess_remainder_v1/data')
    manifest_path = data_dir / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    edge = int(manifest.get('edge_samples', 12986))
    fs = float(manifest.get('fs', 115200.0))
    scored_len = horizon - 2 * edge
    
    rem_rep13 = np.load(data_dir / 'Ca_ac_0p000664_rep13_remainder.npy')
    rem_rep14 = np.load(data_dir / 'Ca_ac_0p000664_rep14_remainder.npy')
    full_rep13 = np.load(data_dir / 'Ca_ac_0p000664_rep13_full.npy')
    full_rep14 = np.load(data_dir / 'Ca_ac_0p000664_rep14_full.npy')
    
    # Unscaled linear oscillator bank simulation (strictly bank_scales = 1.0)
    _, models_bank = load_bank(osc_dir)
    assert len(models_bank) == 11, f'Expected 11 bands, got {len(models_bank)}'
    
    h_steps = 15
    print('Simulating 11 0p08 linear complex oscillators (strictly bank_scales = 1.0)...')
    bank_draw = np.zeros((2, 1, scored_len, 11), dtype=np.float64)
    full_osc_sim = np.zeros((2, horizon + h_steps + 1, 11), dtype=np.float64)
    
    for idx, m in enumerate(models_bank):
        sos = butter(int(m.get('filter_order', 4)), m['edges_hz'], btype='bandpass', fs=fs, output='sos')
        raw = simulate_complex_band(m, horizon + h_steps + 1 + 2 * edge, paths=2, seed=1041 + idx)
        filt = sosfiltfilt(sos, raw, axis=1, padlen=edge)[:, edge:-edge, :]
        full_osc_sim += filt
        bank_draw += filt[:, None, edge:edge + scored_len, :]

    # Determine model type and rollout
    if 'mode' in checkpoint and checkpoint['mode'] in {'model_b', 'model_c'}:
        print(f"Detected oscillator-coupled model ({checkpoint['mode']})...")
        base_model, _ = load_model(checkpoint['base_checkpoint'], device)
        model = OscillatorCoupledNeuralSDE(base_model).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        h_steps = 15
        init = torch.tensor(np.stack([rem_rep13[h_steps], rem_rep14[h_steps]]), dtype=torch.float32, device=device)
        hist = torch.tensor(np.stack([rem_rep13[h_steps - 1::-1].copy(), rem_rep14[h_steps - 1::-1].copy()]), dtype=torch.float32, device=device)
        
        osc_init = torch.tensor(full_osc_sim[:, h_steps], dtype=torch.float32, device=device)
        osc_hist = torch.tensor(full_osc_sim[:, h_steps - 1::-1].copy(), dtype=torch.float32, device=device)
        osc_seq_t = torch.tensor(full_osc_sim[:, h_steps + 1 : h_steps + horizon + 1], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            gen_rem = run_autonomous_rollout_coupled(
                model, init, hist, osc_init, osc_hist, osc_seq_t, horizon=horizon, seed=seed, device=device
            ).cpu().numpy()
    else:
        print("Detected standard Neural SDE model (Model A or baseline)...")
        if 'model_state' in checkpoint and 'base_checkpoint' in checkpoint:
            base_model, _ = load_model(checkpoint['base_checkpoint'], device)
            base_model.load_state_dict(checkpoint['model_state'])
            model = base_model
        else:
            model, _ = load_model(checkpoint_path, device)
        model.eval()
        
        h_steps = model.config.history_steps
        init = torch.tensor(np.stack([rem_rep13[h_steps], rem_rep14[h_steps]]), dtype=torch.float32, device=device)
        hist = torch.tensor(np.stack([rem_rep13[h_steps - 1::-1].copy(), rem_rep14[h_steps - 1::-1].copy()]), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            gen_rem = run_autonomous_rollout_a(
                model, init, hist, horizon=horizon, seed=seed, device=device
            ).cpu().numpy()

    gen_rem_cropped = gen_rem[:, :, edge:edge + scored_len, :]
    hybrid = gen_rem_cropped + bank_draw

    # Reference full signal cropped to match
    ref_full = np.stack([full_rep13[edge:edge + scored_len], full_rep14[edge:edge + scored_len]])

    # Physical Capillary Deformation Energy
    basis_path = '/home/jonas/ucsd_thesis/reduced_data/0p08/shared_pod/shared_pod_r1000.h5'
    with h5py.File(basis_path, 'r') as h5:
        pod_modes = h5['pod']['modes'][:10]
        x_m = h5['grid']['x'][:] * length_scale('mm')
        y_m = h5['grid']['y'][:] * length_scale('mm')
    gamma = 0.0728
    stiffness, _ = capillary_stiffness(pod_modes, x_m, y_m, gamma=gamma)
    opts = dict(exact=False, batch_size=256, stiffness=stiffness, include_spatial_mean=True)

    E_ref_raw = float(np.mean(compute_energies(ref_full, pod_modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']))
    E_gen_raw = float(np.mean(compute_energies(hybrid[:, 0], pod_modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']))

    # 50 Hz high-passed physical wave energy
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

    # Welch PSD
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

    # Resonant Peaks Check
    peak_freqs = [80.9, 137.1, 207.4, 277.7, 365.6, 457.0, 552.0, 646.9, 752.3, 861.3, 977.3]
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

    # Print scorecard
    print('\n================== 0p08 AUTONOMOUS EVALUATION SCORECARD ==================')
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

    print(f'\n--- Resonant Peak Fit (MAE: {mae_db:.2f} dB, Max: {max_db:.2f} dB) ---')
    for pf, diff, rat in zip(peak_freqs, peak_diffs_db, peak_ratios):
        print(f'Peak {pf:6.1f} Hz: Ratio = {rat:.4f} ({diff:+.2f} dB)')

    # Plots
    # 1. PSD
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_w, p_ref, label='Reference (Rep 13)', color='black', alpha=0.8, lw=1.5)
    ax.semilogy(f_w, p_gen, label='Autonomous Generated Hybrid', color='crimson', lw=1.5)
    ax.set_xlim(30, 2000)
    ax.set_ylim(1e-1, 1e6)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (Spatial Mean)')
    ax.set_title(f'0p08 Autonomous PSD Comparison (E_ratio={E_gen_hp/E_ref_hp:.3f}, Peak MAE={mae_db:.2f} dB)')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path / 'psd.png', dpi=200)
    plt.close(fig)

    # 2. Modal Variances Bar Chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, var_ratios, color=['teal'] + ['royalblue']*10, alpha=0.85, edgecolor='black')
    ax.axhline(1.0, color='crimson', linestyle='--', lw=1.5, label='Target (1.0)')
    ax.axhspan(0.8, 1.2, color='green', alpha=0.15, label='Acceptable Range [0.8, 1.2]')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Variance Ratio (Gen / Ref)')
    ax.set_title(f'0p08 Modal Variance Ratios (Mean = {np.mean(var_ratios[1:]):.3f})')
    ax.set_ylim(0, max(2.5, float(np.max(var_ratios)) * 1.15))
    ax.grid(True, axis='y', ls='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / 'modal_variances.png', dpi=200)
    plt.close(fig)

    # Save summary json
    summary = {
        'checkpoint': str(checkpoint_path),
        'horizon': horizon,
        'energy_ratio_hp': float(E_gen_hp / E_ref_hp),
        'energy_ratio_raw': float(E_gen_raw / E_ref_raw),
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

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--osc-dir', type=Path, default='runs/best_models/0p08/oscillators')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.osc_dir, args.output_dir,
                   horizon=args.horizon, seed=args.seed)
