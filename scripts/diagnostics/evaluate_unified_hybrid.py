"""Unified hybrid model evaluation script across any forcing amplitude.

Works seamlessly for:
- Embedded spatial mean (e.g., 0p20 clean, 10 POD coordinates).
- Separate spatial mean (e.g., 0p10 / 0p08 legacy, 11 coordinates).

Produces:
1. Physical capillary wave deformation energy (>50 Hz) and energy ratio.
2. Modal variance breakdown across all coordinates.
3. Broadband Welch PSD (Center physical probe and modal trace across 4 decades).
4. Resonant peak fitting zoom (50-1200 Hz).
5. Comprehensive metrics summary JSON.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def run_rollout(model, init, hist, s_vec=None, horizon=60000, seed=41, device='cuda'):
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
    
    states = [state]
    for _ in range(horizon):
        raw_noise = torch.randn(state.shape, device=device, dtype=state.dtype, generator=generator)
        noise = raw_noise * s_t
        
        base_mean, noise_factor = model.velocity_base_step_distribution(
            state, history, velocity, tcn_cache=tcn_cache)
        stochastic = torch.matmul(noise, noise_factor.T)
        velocity = base_mean + stochastic
        following = state + step_size * velocity
        history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
        state = following
        states.append(state)
    return torch.stack(states, dim=2)


def evaluate_unified_hybrid(
    checkpoint_path,
    osc_dir,
    output_dir,
    manifest_path,
    shared_basis,
    include_spatial_mean=None,
    s_vec=None,
    bank_scales=None,
    overlap=0.0578,
    horizon=60000,
    seed=41
):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading checkpoint {checkpoint_path} on {device}...')
    model, _ = load_model(str(checkpoint_path), device)
    model.eval()

    manifest_p = Path(manifest_path).resolve()
    data_dir = manifest_p.parent
    with open(manifest_p) as f:
        manifest = json.load(f)
    edge = int(manifest['edge_samples'])
    fs = float(manifest['fs'])

    # Find validation repetition files
    rep13_matches = sorted(list(data_dir.glob('*_rep13_remainder.npy')))
    rep14_matches = sorted(list(data_dir.glob('*_rep14_remainder.npy')))
    if not rep13_matches or not rep14_matches:
        raise RuntimeError(f"Could not find rep13 and rep14 in {data_dir}")

    rem_rep13 = np.load(rep13_matches[0])
    rem_rep14 = np.load(rep14_matches[0])
    full_rep13 = np.load(str(rep13_matches[0]).replace('_remainder.npy', '_full.npy'))
    full_rep14 = np.load(str(rep14_matches[0]).replace('_remainder.npy', '_full.npy'))

    n_coords = rem_rep13.shape[1]
    print(f'Detected {n_coords} coordinates in data.')

    # Auto-detect include_spatial_mean if not specified
    if include_spatial_mean is None:
        include_spatial_mean = (n_coords == 11)
    print(f'Include separate spatial mean: {include_spatial_mean}')

    # Initial condition from validation history
    h_steps = model.config.history_steps
    init = torch.tensor(np.stack([rem_rep13[h_steps], rem_rep14[h_steps]]), dtype=torch.float32, device=device)
    hist = torch.tensor(np.stack([rem_rep13[h_steps - 1::-1].copy(), rem_rep14[h_steps - 1::-1].copy()]), dtype=torch.float32, device=device)

    scored_len = horizon
    
    print(f'Simulating Neural SDE remainder rollout for {scored_len} steps...')
    with torch.no_grad():
        rem_gen_t = run_rollout(model, init, hist, s_vec=s_vec, horizon=scored_len, seed=seed, device=device)
    # rem_gen_t shape: [2, 1, scored_len+1, n_coords]
    rem_gen = rem_gen_t[:, :, :scored_len, :].cpu().numpy()

    print(f'Simulating {osc_dir} oscillator bank...')
    _, models_bank = load_bank(osc_dir)
    filters_bank = []
    models_cal = []
    for b_idx, m in enumerate(models_bank):
        low, high = np.asarray(m['edges_hz'], dtype=float)
        width = high - low
        new_low = max(10.0, low - overlap * width)
        new_high = high + overlap * width
        sos = butter(int(m.get('filter_order', 4)), [new_low, new_high], btype='bandpass', fs=fs, output='sos')
        filters_bank.append(sos)
        
        m_c = dict(m)
        s = float(bank_scales[b_idx]) if bank_scales is not None else 1.0
        m_c['coefficient_factor'] = m['coefficient_factor'] * s
        models_cal.append(m_c)

    bank_draw = np.zeros((2, 1, scored_len, n_coords), dtype=np.float64)
    for b_idx, (m, sos) in enumerate(zip(models_cal, filters_bank)):
        # Simulate with edge padding to eliminate filter transient
        raw = simulate_complex_band(m, scored_len + 2 * edge, paths=2, seed=seed + 100 + b_idx)
        filt = sosfiltfilt(sos, raw, axis=1)[:, edge:edge + scored_len, :]
        bank_draw += filt[:, None, :, :]

    # Combine into hybrid simulation: [2, scored_len, n_coords]
    hybrid_gen = (rem_gen + bank_draw)[:, 0, :, :]
    ref_full = np.stack([full_rep13[h_steps:h_steps + scored_len],
                         full_rep14[h_steps:h_steps + scored_len]])

    # Load spatial basis and compute capillary stiffness
    with h5py.File(shared_basis, 'r') as h5:
        # Load up to 10 POD modes
        n_pod = 10
        pod_modes = h5['pod']['modes'][:n_pod]
        x_m = h5['grid']['x'][:] * length_scale('mm')
        y_m = h5['grid']['y'][:] * length_scale('mm')

    stiffness, _ = capillary_stiffness(pod_modes, x_m, y_m, gamma=0.0728)
    cy, cx = pod_modes.shape[1] // 2, pod_modes.shape[2] // 2
    center_weights = pod_modes[:, cy, cx]

    # Evaluate capillary energy
    options = dict(stiffness=stiffness, include_spatial_mean=include_spatial_mean)
    e_gen = compute_energies(hybrid_gen, pod_modes, x_m, y_m, 1e-6, 0.0728, **options)['quadratic']
    e_ref = compute_energies(ref_full, pod_modes, x_m, y_m, 1e-6, 0.0728, **options)['quadratic']

    raw_ratio = float(np.mean(e_gen) / np.mean(e_ref))

    # High-pass filter >50 Hz for physical wave energy
    sos_hp = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
    pod_start = 1 if include_spatial_mean else 0
    hybrid_pod_hp = sosfiltfilt(sos_hp, hybrid_gen[..., pod_start:], axis=1)
    ref_pod_hp = sosfiltfilt(sos_hp, ref_full[..., pod_start:], axis=1)

    opt_hp = dict(stiffness=stiffness, include_spatial_mean=False)
    e_gen_hp = compute_energies(hybrid_pod_hp, pod_modes, x_m, y_m, 1e-6, 0.0728, **opt_hp)['quadratic']
    e_ref_hp = compute_energies(ref_pod_hp, pod_modes, x_m, y_m, 1e-6, 0.0728, **opt_hp)['quadratic']
    hp50_ratio = float(np.mean(e_gen_hp) / np.mean(e_ref_hp))

    # Modal variance ratios
    var_gen = np.var(hybrid_gen.reshape(-1, n_coords), axis=0)
    var_ref = np.var(ref_full.reshape(-1, n_coords), axis=0)
    var_ratios = var_gen / var_ref

    # Center probe surface elevation
    if include_spatial_mean:
        eta_gen = hybrid_gen[..., 0] + hybrid_gen[..., 1:] @ center_weights
        eta_ref = ref_full[..., 0] + ref_full[..., 1:] @ center_weights
    else:
        eta_gen = hybrid_gen @ center_weights
        eta_ref = ref_full @ center_weights

    # Welch PSD
    nperseg = 8192
    freqs, psd_ref = signal.welch(eta_ref.flatten(), fs=fs, nperseg=nperseg)
    _, psd_gen = signal.welch(eta_gen.flatten(), fs=fs, nperseg=nperseg)

    # Subband power analysis
    bands = [
        (50, 100), (100, 300), (300, 1000), (1000, 1400),
        (1400, 2500), (2500, 5000), (5000, 20000)
    ]
    subband_ratios = {}
    for low, high in bands:
        idx = (freqs >= low) & (freqs < high)
        p_ref = np.trapezoid(psd_ref[idx], freqs[idx])
        p_gen = np.trapezoid(psd_gen[idx], freqs[idx])
        subband_ratios[f'{low}_{high}'] = float(p_gen / p_ref)

    # Resonant peak ratios
    osc_summary_path = Path(osc_dir) / 'summary.json'
    peak_ratios = {}
    if osc_summary_path.exists():
        with open(osc_summary_path) as f:
            osc_meta = json.load(f)
        for b in osc_meta['bands']:
            f_peak = b.get('fitted_frequency_hz', b.get('frequency_hz', b.get('nominal_center_hz')))
            idx = np.argmin(np.abs(freqs - f_peak))
            ratio = float(psd_gen[idx] / psd_ref[idx])
            peak_ratios[f'{f_peak:.1f}'] = ratio

    # Print scorecard
    print('\n' + '='*60)
    print('AUTONOMOUS SCORECARD (ZERO ROLLOUT FACTORS)')
    print('='*60)
    print(f'Raw Deformation Energy Ratio:        {raw_ratio:.4f}x')
    print(f'>50 Hz Physical Capillary Ratio:     {hp50_ratio:.4f}x  (Target: 0.90 - 1.10)')
    print('\nModal Variance Ratios:')
    for m in range(n_coords):
        label = "Spatial Mean" if (include_spatial_mean and m == 0) else f"POD Mode {m if include_spatial_mean else m + 1}"
        print(f'  {label:15s}: {var_ratios[m]:.4f}x ({100*(var_ratios[m]-1):+.2f}%)')
    
    print('\nResonant Peak Contrast Ratios:')
    for f_str, r in peak_ratios.items():
        db = 10 * np.log10(r)
        print(f'  {f_str:6s} Hz: {r:.3f}x ({db:+.2f} dB)')
    print('='*60 + '\n')

    # Plot 1: Broadband PSD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.loglog(freqs, psd_ref, 'k-', alpha=0.85, label='Ground Truth Reference')
    ax1.loglog(freqs, psd_gen, 'r-', alpha=0.85, label='Autonomous Generated Hybrid')
    ax1.set_ylabel(r'Center Probe PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]')
    ax1.set_title(f'Broadband PSD Comparison (>50 Hz Energy Ratio: {hp50_ratio:.4f}x)')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(frameon=True)
    ax1.set_xlim(20, fs / 2)

    # Modal sum PSD
    _, psd_sum_ref = signal.welch(np.sum(ref_full[..., pod_start:]**2, axis=-1).flatten(), fs=fs, nperseg=nperseg)
    _, psd_sum_gen = signal.welch(np.sum(hybrid_gen[..., pod_start:]**2, axis=-1).flatten(), fs=fs, nperseg=nperseg)
    ax2.loglog(freqs, psd_sum_ref, 'k-', alpha=0.85, label='Reference POD Modes Energy')
    ax2.loglog(freqs, psd_sum_gen, 'r-', alpha=0.85, label='Generated POD Modes Energy')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'POD Modes PSD [$\mu\mathrm{m}^4/\mathrm{Hz}$]')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path / 'psd.png', dpi=200)
    plt.close()

    # Plot 2: Peak zoom (50 - 1200 Hz)
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd_ref, 'k-', alpha=0.85, label='Reference')
    plt.semilogy(freqs, psd_gen, 'r-', alpha=0.85, label='Autonomous Hybrid')
    plt.xlim(50, 1200)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD [$\mu\mathrm{m}^2/\mathrm{Hz}$]')
    plt.title('Resonant Carrier Peaks (50 - 1200 Hz)')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path / 'psd_peaks.png', dpi=200)
    plt.close()

    # Plot 3: Modal variances
    plt.figure(figsize=(9, 4.5))
    x_idx = np.arange(n_coords)
    plt.bar(x_idx - 0.18, var_ref, width=0.36, color='black', alpha=0.7, label='Reference')
    plt.bar(x_idx + 0.18, var_gen, width=0.36, color='crimson', alpha=0.85, label='Generated')
    labels = ["Mean"] + [f"P{i}" for i in range(1, 11)] if include_spatial_mean else [f"P{i}" for i in range(1, n_coords + 1)]
    plt.xticks(x_idx, labels)
    plt.yscale('log')
    plt.ylabel(r'Variance [$\mu\mathrm{m}^2$]')
    plt.title('Modal Variance Distribution')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'modal_variances.png', dpi=200)
    plt.close()

    # Plot 4: Energy evolution
    plt.figure(figsize=(10, 4))
    t_ms = np.arange(len(e_ref[0])) / fs * 1000
    plt.plot(t_ms, e_ref[0] * 1e10, 'k-', alpha=0.7, label='Reference Rep 13')
    plt.plot(t_ms, e_gen[0] * 1e10, 'r-', alpha=0.85, label='Generated Hybrid')
    plt.xlabel('Physical Time [ms]')
    plt.ylabel(r'Deformation Energy [$10^{-10}\ \mathrm{J}$]')
    plt.title(f'Capillary Deformation Energy Evolution (Ratio: {raw_ratio:.4f}x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'energy.png', dpi=200)
    plt.close()

    # Save summary json
    summary = {
        'raw_energy_ref': float(np.mean(e_ref)),
        'raw_energy_gen': float(np.mean(e_gen)),
        'raw_energy_ratio': raw_ratio,
        'physical_energy_hp50_ref': float(np.mean(e_ref_hp)),
        'physical_energy_hp50_gen': float(np.mean(e_gen_hp)),
        'physical_energy_hp50_ratio': hp50_ratio,
        'modal_variance_ratios': [float(x) for x in var_ratios],
        'band_power_ratios': subband_ratios,
        'peak_ratios': peak_ratios,
    }
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save rollout npz
    np.savez_compressed(
        output_path / 'rollout.npz',
        generated=hybrid_gen,
        reference=ref_full,
        fs=fs,
        coordinates=n_coords,
        include_spatial_mean=include_spatial_mean
    )
    print(f'Evaluation complete! Results saved to {output_path}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--oscillator-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--shared-basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-spatial-mean', action='store_true', default=None)
    parser.add_argument('--horizon', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()

    evaluate_unified_hybrid(
        checkpoint_path=args.checkpoint,
        osc_dir=args.oscillator_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        shared_basis=args.shared_basis,
        include_spatial_mean=args.include_spatial_mean,
        horizon=args.horizon,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
