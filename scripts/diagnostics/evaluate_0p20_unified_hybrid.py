"""Evaluation script for the 0p20 unified single-stage compact hybrid model.
Tests:
1. 1-stage compact (32-channel) Neural SDE trained on 0p20 remainder.
2. 11 linear complex oscillators (74-981 Hz) matching 0p20 resonance with 5.8% overlap.
3. Evaluates >50 Hz capillary deformation energy, modal variances, and high-res Welch PSD.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
import torch
import h5py

from modelling.neural_sde.train import load_model
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank
from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale

def make_highshelf(fz, fp, fs):
    wz = 2 * np.pi * fz
    wp = 2 * np.pi * fp
    b, a = signal.bilinear([wp/wz, wp], [1, wp], fs=fs)
    return b, a

def run_boosted_rollout(model, init, hist, s_vec=None, b_mean=None, a_mean=None, horizon=60000, seed=41, device='cuda'):
    generator = torch.Generator(device=device).manual_seed(seed)
    state = init[:, None, :].expand(-1, 1, -1).clone()
    history = hist[:, None, :model.config.history_steps].expand(-1, 1, -1, -1).clone()
    velocity = model._frame_velocity(state, history)
    step_size = model.config.lag_steps * (1. if model.config.state_variable == 'velocity' else model.config.dt)
    tcn_cache = model.new_tcn_cache()
    
    if s_vec is not None:
        s_t = torch.tensor(s_vec, dtype=state.dtype, device=device)
    else:
        s_t = torch.ones(state.shape[-1], dtype=state.dtype, device=device)
        
    use_shaping = (b_mean is not None and a_mean is not None)
    if use_shaping:
        b_t = torch.tensor(b_mean, dtype=state.dtype, device=device)
        a_t = torch.tensor(a_mean, dtype=state.dtype, device=device)
        B = state.shape[0] * state.shape[1]
        x_buf = torch.zeros(B, 1, dtype=state.dtype, device=device)
        y_buf = torch.zeros(B, 1, dtype=state.dtype, device=device)
    
    states = [state]
    for _ in range(horizon):
        raw_noise = torch.randn(state.shape, device=device, dtype=state.dtype, generator=generator)
        if use_shaping:
            x0 = raw_noise[..., 0].reshape(-1)
            y0 = b_t[0] * x0 + b_t[1] * x_buf.squeeze(1) - a_t[1] * y_buf.squeeze(1)
            x_buf[:, 0] = x0
            y_buf[:, 0] = y0
            noise0 = y0.reshape(state.shape[:-1])
            noise = torch.cat([noise0.unsqueeze(-1), raw_noise[..., 1:]], dim=-1)
        else:
            noise = raw_noise
            
        noise = noise * s_t
        
        base_mean, noise_factor = model.velocity_base_step_distribution(
            state, history, velocity, tcn_cache=tcn_cache)
        stochastic = torch.matmul(noise, noise_factor.T)
        velocity = base_mean + stochastic
        following = state + step_size * velocity
        history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
        state = following
        states.append(state)
    return torch.stack(states, dim=2)

def evaluate_0p20(checkpoint_path, osc_dir, output_dir, s_vec=None, bank_scales=None, use_shaping=False, horizon=60000, seed=41):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading checkpoint {checkpoint_path}...')
    model, _ = load_model(str(checkpoint_path), device)
    model.eval()

    fs = 115000.0  # Native sampling frequency
    # Load 0p20 validation remainder & full references
    data_dir = Path('runs/adaptive/neural_sde_0p20_local_excess_remainder_v1/data')
    manifest_path = data_dir / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    edge = int(manifest['edge_samples'])
    fs = float(manifest['fs'])

    rem_rep13 = np.load(data_dir / 'Ca_ac_0p001660_rep13_remainder.npy')
    rem_rep14 = np.load(data_dir / 'Ca_ac_0p001660_rep14_remainder.npy')
    full_rep13 = np.load(data_dir / 'Ca_ac_0p001660_rep13_full.npy')
    full_rep14 = np.load(data_dir / 'Ca_ac_0p001660_rep14_full.npy')

    # Initial condition from history_steps
    h_steps = model.config.history_steps
    init = torch.tensor(np.stack([rem_rep13[h_steps], rem_rep14[h_steps]]), dtype=torch.float32, device=device)
    hist = torch.tensor(np.stack([rem_rep13[h_steps - 1::-1].copy(), rem_rep14[h_steps - 1::-1].copy()]), dtype=torch.float32, device=device)

    b_mean, a_mean = None, None
    if use_shaping:
        b_mean, a_mean = make_highshelf(1200.0, 1600.0, fs)

    print('Rolling out Neural SDE remainder...')
    with torch.no_grad():
        gen_rem = run_boosted_rollout(model, init, hist, s_vec=s_vec, b_mean=b_mean, a_mean=a_mean,
                                      horizon=horizon, seed=seed, device=device).cpu().numpy()

    scored_len = horizon - 2 * edge
    gen_rem_cropped = gen_rem[:, :, edge:edge + scored_len, :]

    # Load 0p20 oscillators
    _, models_bank = load_bank(osc_dir)
    assert len(models_bank) == 11

    if bank_scales is None:
        bank_scales = np.ones(11, dtype=np.float64)
    else:
        bank_scales = np.asarray(bank_scales, dtype=np.float64)

    overlap = 0.0578
    filters_bank = []
    models_cal = []
    for idx, m in enumerate(models_bank):
        low, high = np.asarray(m['edges_hz'], dtype=float)
        width = high - low
        new_low = max(10.0, low - overlap * width)
        new_high = high + overlap * width
        sos = butter(int(m.get('filter_order', 4)), [new_low, new_high], btype='bandpass', fs=fs, output='sos')
        filters_bank.append(sos)

        m_c = dict(m)
        m_c['coefficient_factor'] = m['coefficient_factor'] * bank_scales[idx]
        models_cal.append(m_c)

    print('Simulating 11 0p20 linear complex oscillators...')
    bank_draw = np.zeros((2, 1, scored_len, 11), dtype=np.float64)
    for idx, (m, sos) in enumerate(zip(models_cal, filters_bank)):
        raw = simulate_complex_band(m, scored_len + 2 * edge, paths=2, seed=1041 + idx)
        filt = sosfiltfilt(sos, raw, axis=1, padlen=edge)[:, edge:-edge, :]
        bank_draw += filt[:, None, :, :]

    # Hybrid rollout
    hybrid = gen_rem_cropped + bank_draw

    # Reference full signal cropped to match
    ref_full = np.stack([full_rep13[edge:edge + scored_len], full_rep14[edge:edge + scored_len]])

    # Capillary Deformation Energy for 0p20
    basis_path = '/home/jonas/ucsd_thesis/reduced_data/0p20/shared_pod/shared_pod_r1000.h5'
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

    # Welch PSD (nperseg=8192)
    f_w, p_ref_mean0 = signal.welch(ref_full[0, :, 0], fs=fs, nperseg=8192, noverlap=4096)
    p_ref_m = np.mean([signal.welch(ref_full[c, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_gen_m = np.mean([signal.welch(hybrid[c, 0, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)

    p_ref_pod = np.mean([sum(signal.welch(ref_full[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)
    p_gen_pod = np.mean([sum(signal.welch(hybrid[c, 0, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)

    # Print scorecard
    print('\n================== 0p20 UNIFIED HYBRID EVALUATION ==================')
    print('--- Capillary Deformation Energy ---')
    print(f'Raw Energy: Ref = {E_ref_raw:.3e} J | Gen = {E_gen_raw:.3e} J | Ratio = {E_gen_raw/E_ref_raw:.4f}')
    print(f'>50 Hz Physical Waves: Ref = {E_ref_hp:.3e} J | Gen = {E_gen_hp:.3e} J | Ratio = {E_gen_hp/E_ref_hp:.4f}')

    print('\n--- Modal Variance Ratios (Gen / Ref) ---')
    print(f'Mean:   Ref Var = {var_ref[0]:10.1f} | Gen Var = {var_gen[0]:10.1f} | Ratio = {var_ratios[0]:.4f}')
    for m in range(1, 11):
        print(f'Mode {m:2d}: Ref Var = {var_ref[m]:10.1f} | Gen Var = {var_gen[m]:10.1f} | Ratio = {var_ratios[m]:.4f}')

    bands_eval = [(50, 100), (100, 300), (300, 1000), (1000, 1400), (1400, 2500), (2500, 5000), (5000, 20000)]
    print('\n--- Spatial Mean Power Ratios ---')
    band_power_ratios = {}
    for lo, hi in bands_eval:
        mask = (f_w >= lo) & (f_w < hi)
        ratio = float(np.sum(p_gen_m[mask]) / np.sum(p_ref_m[mask]))
        band_power_ratios[f'{lo}_{hi}'] = ratio
        print(f'{lo:5d} - {hi:5d} Hz: Power Ratio = {ratio:.4f}')

    peak_locs = [73.8, 126.6, 200.4, 274.2, 362.1, 450.0, 537.9, 636.3, 741.8, 857.8, 980.9]
    print('\n--- Peak Fit at 0p20 Resonant Frequencies ---')
    peak_ratios = {}
    for pf in peak_locs:
        k = np.argmin(np.abs(f_w - pf))
        r = float(p_gen_m[k] / p_ref_m[k])
        peak_ratios[f'{pf:.1f}'] = r
        print(f'Peak {pf:6.1f} Hz: Ref = {p_ref_m[k]:10.2e} | Gen = {p_gen_m[k]:10.2e} | Ratio = {r:.4f}')

    # Save summary json
    summary = dict(
        raw_energy_ref=E_ref_raw,
        raw_energy_gen=E_gen_raw,
        raw_energy_ratio=E_gen_raw / E_ref_raw,
        physical_energy_hp50_ref=E_ref_hp,
        physical_energy_hp50_gen=E_gen_hp,
        physical_energy_hp50_ratio=E_gen_hp / E_ref_hp,
        modal_variance_ratios=var_ratios.tolist(),
        band_power_ratios=band_power_ratios,
        peak_ratios=peak_ratios
    )
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save rollout npz
    np.savez_compressed(
        output_path / 'rollout.npz',
        trajectories=hybrid,
        reference=ref_full,
        reference_time=np.stack([np.arange(scored_len) / fs, np.arange(scored_len) / fs]),
        repetition=np.array(['Ca_ac_0p001660_rep13', 'Ca_ac_0p001660_rep14']),
        start_index=np.array([edge, edge])
    )

    # 1. Full PSD Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    m_plot = (f_w >= 50) & (f_w <= 20000)
    axes[0].loglog(f_w[m_plot], p_ref_m[m_plot], color='black', label='0p20 Ground Truth Reference', lw=1.5)
    axes[0].loglog(f_w[m_plot], p_gen_m[m_plot], color='tab:blue', label='0p20 Unified Compact Hybrid', lw=1.2, alpha=0.9)
    axes[0].set(ylabel='Spatial Mean PSD [µm²/Hz]', title='0p20 Unified Hybrid: Spatial Mean Spectrum')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].legend()

    axes[1].loglog(f_w[m_plot], p_ref_pod[m_plot], color='black', label='0p20 Ground Truth Reference (Sum POD 1-10)', lw=1.5)
    axes[1].loglog(f_w[m_plot], p_gen_pod[m_plot], color='tab:blue', label='0p20 Unified Compact Hybrid (Sum POD 1-10)', lw=1.2, alpha=0.9)
    axes[1].set(xlabel='Frequency [Hz]', ylabel='POD 1-10 Sum PSD [µm²/Hz]', title='0p20 Unified Hybrid: POD Modes 1-10 Spectrum')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend()
    fig.savefig(output_path / 'psd.png', dpi=200)
    plt.close(fig)

    # 2. Peak Resolution Zoom
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    m_peak = (f_w >= 50) & (f_w <= 1200)
    ax.semilogy(f_w[m_peak], p_ref_m[m_peak], color='black', label='Reference (0p20 Validation)', lw=1.5)
    ax.semilogy(f_w[m_peak], p_gen_m[m_peak], color='tab:blue', label='0p20 Unified Compact Hybrid', lw=1.2)
    for pf in peak_locs:
        ax.axvline(pf, color='gray', linestyle=':', alpha=0.5)
    ax.set(xlabel='Frequency [Hz]', ylabel='PSD [µm²/Hz]', title='0p20 Resonant Peak Structure (50 - 1200 Hz)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.savefig(output_path / 'psd_peaks.png', dpi=200)
    plt.close(fig)

    # 3. Energy Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    bars = ax.bar(['Raw Energy', '>50 Hz Physical Energy'],
                  [E_gen_raw / E_ref_raw, E_gen_hp / E_ref_hp],
                  color=['tab:blue', 'tab:green'], alpha=0.85, width=0.5)
    ax.axhline(1.0, color='black', linestyle='--', lw=1.5, label='Target 1.0x')
    ax.axhspan(0.90, 1.10, color='gray', alpha=0.2, label='Acceptable Range [0.90, 1.10]')
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.02, f'{h:.3f}x', ha='center', va='bottom', fontweight='bold')
    ax.set(ylabel='Generated / Reference Energy Ratio', title='0p20 Capillary Deformation Energy Scorecard', ylim=[0, 1.3])
    ax.legend()
    fig.savefig(output_path / 'energy.png', dpi=200)
    plt.close(fig)

    # 4. Modal Variance Bar Chart
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    coords = ['Mean'] + [f'POD {i}' for i in range(1, 11)]
    bars = ax.bar(coords, var_ratios, color='tab:blue', alpha=0.85)
    ax.axhline(1.0, color='black', linestyle='--', lw=1.5, label='Target 1.0x')
    ax.axhspan(0.80, 1.20, color='gray', alpha=0.2, label='Target Band [0.80, 1.20]')
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set(ylabel='Variance Ratio (Gen / Ref)', title='0p20 Modal Variances (>50 Hz)')
    ax.legend()
    fig.savefig(output_path / 'modal_variances.png', dpi=200)
    plt.close(fig)

    # 5. Trajectory Snippets
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout='constrained')
    dt = 1.0 / fs
    t_snip = np.arange(2000) * dt * 1000 # ms
    axes[0].plot(t_snip, ref_full[0, :2000, 0], color='black', label='Ref Mean', lw=1.2)
    axes[0].plot(t_snip, hybrid[0, 0, :2000, 0], color='tab:blue', label='Gen Mean', lw=1.0, alpha=0.8)
    axes[0].set(ylabel='Mean [µm]'); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(t_snip, ref_full[0, :2000, 1], color='black', label='Ref POD 1', lw=1.2)
    axes[1].plot(t_snip, hybrid[0, 0, :2000, 1], color='tab:blue', label='Gen POD 1', lw=1.0, alpha=0.8)
    axes[1].set(ylabel='POD 1 [µm]'); axes[1].grid(alpha=0.3); axes[1].legend()

    axes[2].plot(t_snip, ref_full[0, :2000, 3], color='black', label='Ref POD 3', lw=1.2)
    axes[2].plot(t_snip, hybrid[0, 0, :2000, 3], color='tab:blue', label='Gen POD 3', lw=1.0, alpha=0.8)
    axes[2].set(xlabel='Time [ms]', ylabel='POD 3 [µm]'); axes[2].grid(alpha=0.3); axes[2].legend()
    fig.savefig(output_path / 'trajectories.png', dpi=200)
    plt.close(fig)

    print(f'Evaluation saved to {output_path}')
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default='runs/adaptive/neural_sde_0p20_unified_single_stage_sde_v1/best.pt')
    parser.add_argument('--oscillator-dir', type=Path, default='runs/diagnostics/frequency_hybrid_0p20_local_excess_rank1_v1/oscillators')
    parser.add_argument('--output-dir', type=Path, default='runs/diagnostics/unified_hybrid_0p20_v1')
    parser.add_argument('--s-vec', nargs='+', type=float, default=None)
    parser.add_argument('--bank-scales', nargs='+', type=float, default=None)
    parser.add_argument('--use-shaping', action='store_true')
    args = parser.parse_args()
    evaluate_0p20(args.checkpoint, args.oscillator_dir, args.output_dir,
                  s_vec=args.s_vec, bank_scales=args.bank_scales, use_shaping=args.use_shaping)

