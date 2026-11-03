"""Benchmark evaluation script for the simplified 4-oscillator hybrid model.
Tests:
1. Only 4 linear oscillators (77, 133, 207, 283 Hz) with 5.8% overlap.
2. Neural SDE trained directly on the 4-band remainder (untouched continuum above 300 Hz).
3. Evaluates >50 Hz capillary deformation energy, modal variances, and PSD.
"""
from pathlib import Path
import json
import math
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

def run_boosted_shaped_rollout(model, init, hist, b, a, s_vec, horizon=60000, seed=41, device='cuda'):
    generator = torch.Generator(device=device).manual_seed(seed)
    state = init[:, None, :].expand(-1, 1, -1).clone()
    history = hist[:, None, :model.config.history_steps].expand(-1, 1, -1, -1).clone()
    velocity = model._frame_velocity(state, history)
    step_size = model.config.lag_steps * (1. if model.config.state_variable == 'velocity' else model.config.dt)
    tcn_cache = model.new_tcn_cache()
    
    b_t = torch.tensor(b, dtype=state.dtype, device=device)
    a_t = torch.tensor(a, dtype=state.dtype, device=device)
    s_t = torch.tensor(s_vec, dtype=state.dtype, device=device)
    
    B = state.shape[0] * state.shape[1]
    x_buf = torch.zeros(B, 1, dtype=state.dtype, device=device)
    y_buf = torch.zeros(B, 1, dtype=state.dtype, device=device)
    
    states = [state]
    for _ in range(horizon):
        raw_noise = torch.randn(state.shape, device=device, dtype=state.dtype, generator=generator)
        x0 = raw_noise[..., 0].reshape(-1)
        y0 = b_t[0] * x0 + b_t[1] * x_buf.squeeze(1) - a_t[1] * y_buf.squeeze(1)
        x_buf[:, 0] = x0
        y_buf[:, 0] = y0
        noise0 = y0.reshape(state.shape[:-1])
        noise = torch.cat([noise0.unsqueeze(-1), raw_noise[..., 1:]], dim=-1)
        
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='runs/adaptive/simplified_hybrid_4band_sde_v1/best.pt')
    parser.add_argument('--oscillator-dir', type=str,
                        default='runs/diagnostics/simplified_4band_oscillators')
    parser.add_argument('--remainder-manifest', type=str,
                        default='runs/adaptive/simplified_hybrid_4band_remainder_v1/data/manifest.json')
    parser.add_argument('--output-dir', type=str,
                        default='runs/diagnostics/simplified_4band_hybrid_v1')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = Path(args.checkpoint)
    print(f'Loading checkpoint {ckpt_path}...')
    model, _ = load_model(str(ckpt_path), device)
    model.eval()

    from modelling.frequency_hybrid.study import component_data
    # Read 4-band remainder data
    manifest_data = json.loads(Path(args.remainder_manifest).read_text())
    edge = int(manifest_data['edge_samples'])
    fs = float(manifest_data['fs'])

    # Read validation reps
    val_reps = ['Ca_ac_0p000830_rep13', 'Ca_ac_0p000830_rep14']
    full_data = component_data('runs/adaptive/neural_sde_0p10_local_excess_low77_psd_refinement_v1', 'full')

    # Load initial states from 4-band remainder
    rem_target = Path(args.remainder_manifest).parent
    rem_arrays = [np.load(rem_target / f'{name}_remainder.npy') for name in val_reps]
    
    # Init and hist from first 64 steps
    init = torch.tensor(np.stack([arr[63] for arr in rem_arrays]), dtype=torch.float32, device=device)
    hist = torch.tensor(np.stack([arr[:63] for arr in rem_arrays]), dtype=torch.float32, device=device)

    # 1. Roll out Neural SDE Remainder with Shaped Noise & Modal Boost
    b_mean, a_mean = make_highshelf(750.0, 4000.0, fs)
    # Mode 3 boost = 1.70, Modes 4-5 = 1.15, Modes 6-10 = 1.10
    s_vec = np.array([1.0, 1.05, 1.0, 1.70, 1.15, 1.15, 1.10, 1.10, 1.10, 1.10, 1.10], dtype=np.float32)
    print('Generating shaped & modal-boosted remainder rollout...')
    with torch.no_grad():
        gen_rem = run_boosted_shaped_rollout(model, init, hist, b_mean, a_mean, s_vec, horizon=60000, seed=41, device=device).cpu().numpy()

    scored_len = gen_rem.shape[2] - 2 * edge
    gen_rem_cropped = gen_rem[:, :, edge:-edge, :]

    # 2. Simulate 4-band Linear Oscillator Bank
    _, models_bank = load_bank(args.oscillator_dir)
    assert len(models_bank) == 4, f'Expected 4 bands, got {len(models_bank)}'

    bank_scales = [1.0, 0.65, 0.90, 0.85]
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

    bank_draw = np.zeros((2, 1, scored_len, 11), dtype=np.float64)
    for idx, (m, sos) in enumerate(zip(models_cal, filters_bank)):
        raw = simulate_complex_band(m, scored_len + 2 * edge, paths=2, seed=1041 + idx)
        filt = sosfiltfilt(sos, raw, axis=1, padlen=edge)[:, edge:-edge, :]
        bank_draw += filt[:, None, :, :]

    # 3. Hybrid Trajectory (ONLY 4 oscillators + SDE remainder!)
    simplified_hybrid = gen_rem_cropped + bank_draw

    # Read ground truth reference
    ref_full = np.stack([full_data.read_coordinates(name, 63 + edge, 63 + edge + scored_len)[1]
                         for name in val_reps]) # [2, scored_len, 11]

    # Energy calculations
    basis_path = '/home/jonas/ucsd_thesis/reduced_data/0p10/shared_pod/shared_pod_r1000.h5'
    with h5py.File(basis_path, 'r') as h5:
        modes = h5['pod']['modes'][:10]
        x_m = h5['grid']['x'][:] * length_scale('mm')
        y_m = h5['grid']['y'][:] * length_scale('mm')
    gamma = 0.0728
    stiffness, _ = capillary_stiffness(modes, x_m, y_m, gamma=gamma)
    opts = dict(exact=False, batch_size=256, stiffness=stiffness, include_spatial_mean=True)

    E_ref_raw = np.mean(compute_energies(ref_full, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])
    E_gen_raw = np.mean(compute_energies(simplified_hybrid[:, 0], modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])

    # 50 Hz high-passed energy
    sos50 = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
    ref_full_hp = ref_full.copy()
    ref_full_hp[..., 1:] = sosfiltfilt(sos50, ref_full_hp[..., 1:], axis=-2)
    simplified_hp = simplified_hybrid.copy()
    simplified_hp[..., 1:] = sosfiltfilt(sos50, simplified_hp[..., 1:], axis=-2)

    E_ref_hp = np.mean(compute_energies(ref_full_hp, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])
    E_gen_hp = np.mean(compute_energies(simplified_hp[:, 0], modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])

    # Modal variance ratios
    var_ref = np.var(ref_full_hp.reshape(-1, 11), axis=0)
    var_gen = np.var(simplified_hp[:, 0].reshape(-1, 11), axis=0)
    var_ratios = var_gen / var_ref

    # Spectral comparisons
    f_w, p_ref_mean = signal.welch(ref_full[0, :, 0], fs=fs, nperseg=8192, noverlap=4096)
    p_ref_m = np.mean([signal.welch(ref_full[c, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_gen_m = np.mean([signal.welch(simplified_hybrid[c, 0, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)

    p_ref_pod = np.mean([sum(signal.welch(ref_full[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)
    p_gen_pod = np.mean([sum(signal.welch(simplified_hybrid[c, 0, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)

    bands = [(50, 100), (100, 300), (300, 1000), (1000, 1400), (1150, 1320), (1400, 2500), (2500, 5000), (5000, 20000)]

    print('\n================== SIMPLIFIED 4-BAND HYBRID RESULTS ==================')
    print('--- Capillary Deformation Energy ---')
    print(f'Raw Energy: Ref = {E_ref_raw:.2e} J, Gen = {E_gen_raw:.2e} J (Ratio = {E_gen_raw/E_ref_raw:.4f})')
    print(f'50 Hz HP Energy (>50 Hz Physical Waves): Ref = {E_ref_hp:.2e} J, Gen = {E_gen_hp:.2e} J (Ratio = {E_gen_hp/E_ref_hp:.4f})')

    print('\n--- Modal Variance Ratios (Gen / Ref) ---')
    for m in range(1, 11):
        print(f'Mode {m:2d}: Ref Var = {var_ref[m]:10.1f} | Gen Var = {var_gen[m]:10.1f} | Ratio = {var_ratios[m]:.4f}')

    print('\n--- Spatial Mean Power Ratios (Gen / Ref) ---')
    for lo, hi in bands:
        m = (f_w >= lo) & (f_w < hi)
        r = np.sum(p_gen_m[m]) / np.sum(p_ref_m[m])
        print(f'{lo:5d} - {hi:5d} Hz:  Power Ratio = {r:.4f}')

    print('\n--- Peaks & Valleys Match in Spatial Mean ---')
    peak_locs = [77.1, 132.6, 206.8, 282.6, 364.3, 455.9, 547.1, 651.0, 756.9, 867.8, 982.4, 1232.0]
    for pf in peak_locs:
        k = np.argmin(np.abs(f_w - pf))
        print(f'Peak {pf:6.1f} Hz: Ref = {p_ref_m[k]:10.2e} | Gen = {p_gen_m[k]:10.2e} | Ratio = {p_gen_m[k]/p_ref_m[k]:.4f}')

    # Save complete rollout directory and format
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'rollout.npz',
                        reference=ref_full,
                        trajectories=simplified_hybrid,
                        repetition=np.array(val_reps),
                        start_index=np.array([63, 63]),
                        reference_time=np.stack([np.arange(scored_len)/fs, np.arange(scored_len)/fs], axis=0))

    # Save metrics JSON
    metrics = {
        'rank': 10,
        'include_spatial_mean': True,
        'source_shared_basis': basis_path,
        'time_units': 's',
        'signal_units': 'microns',
        'fs': fs,
        'physical_lag': 1.0 / fs,
        'lag_steps': 1,
        'spatial_mean_highpass_hz': 50.0,
        'checkpoint': str(ckpt_path),
        'energy': {
            'raw': {
                'reference': float(E_ref_raw),
                'generated': float(E_gen_raw),
                'ratio': float(E_gen_raw / E_ref_raw)
            },
            'highpass_50hz': {
                'reference': float(E_ref_hp),
                'generated': float(E_gen_hp),
                'ratio': float(E_gen_hp / E_ref_hp)
            }
        },
        'modal_variance_ratios': {f'mode_{m}': float(var_ratios[m]) for m in range(1, 11)},
        'spatial_mean_power_ratios': {f'{lo}_{hi}_hz': float(np.sum(p_gen_m[(f_w>=lo)&(f_w<hi)]) / np.sum(p_ref_m[(f_w>=lo)&(f_w<hi)])) for lo, hi in bands}
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nFull rollout saved to {out_dir / "rollout.npz"}')
    print(f'Metrics saved to {out_dir / "metrics.json"}')

if __name__ == '__main__':
    main()
