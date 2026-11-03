"""Evaluate all improvements on the hybrid model:
1. Short-horizon multistep trained Neural SDE remainder
2. Shaped noise diffusion (Filter 1 on spatial mean)
3. Added 1232 Hz harmonic oscillator mode
4. 50 Hz high-pass filtered physical deformation energy
"""
from pathlib import Path
import json
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
import torch

from modelling.neural_sde.train import load_model
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.data_preparation.frequency_components import settling_samples
from data_analysis.rollout import Rollout
from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale

def make_highshelf(fz, fp, fs):
    wz = 2 * np.pi * fz
    wp = 2 * np.pi * fp
    b, a = signal.bilinear([wp/wz, wp], [1, wp], fs=fs)
    return b, a

def run_shaped_rollout(model, init, hist, b, a, horizon=60000, seed=41, device='cuda'):
    generator = torch.Generator(device=device).manual_seed(seed)
    state = init[:, None, :].expand(-1, 1, -1).clone()
    history = hist[:, None, :model.config.history_steps].expand(-1, 1, -1, -1).clone()
    velocity = model._frame_velocity(state, history)
    step_size = model.config.lag_steps * (1. if model.config.state_variable == 'velocity' else model.config.dt)
    tcn_cache = model.new_tcn_cache()
    
    b_t = torch.tensor(b, dtype=state.dtype, device=device)
    a_t = torch.tensor(a, dtype=state.dtype, device=device)
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
                        default='runs/adaptive/neural_sde_0p10_local_excess_low77_psd_refinement_v1/trials/mean_variance_w0p03_e2/best.pt')
    parser.add_argument('--output-dir', type=str,
                        default='runs/diagnostics/final_improved_hybrid_v1')
    args = parser.parse_args()

    device = 'cuda'
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f'Waiting for checkpoint {ckpt_path}...')
        return
    print(f'Loading checkpoint {ckpt_path}...')
    model, _ = load_model(str(ckpt_path), device)
    model.eval()

    # Load validation data reference
    ref_rollout_file = 'runs/adaptive/neural_sde_0p10_local_excess_low77_psd_refinement_v1/trials/mean_variance_w0p03_e2/evaluation_scale1p0/remainder_raw/rollout.npz'
    with np.load(ref_rollout_file) as z:
        ref_remainder = z['reference'] # [2, 60001, 11]
        reps = z['repetition']
        starts = z['start_index']
        times = z['reference_time']

    # Read full validation data reference
    from modelling.frequency_hybrid.study import component_data
    full_data = component_data('runs/adaptive/neural_sde_0p10_local_excess_low77_psd_refinement_v1', 'full')
    edge = int(full_data.manifest['edge_samples'])
    fs = float(full_data.manifest['fs'])

    init = torch.tensor(ref_remainder[:, 63, :], dtype=torch.float32, device=device)
    hist = torch.tensor(ref_remainder[:, :63, :], dtype=torch.float32, device=device)

    # 1. Generate Shaped Noise SDE remainder
    b_mean, a_mean = make_highshelf(800.0, 4000.0, fs)
    print('Generating shaped noise remainder rollout...')
    with torch.no_grad():
        gen_rem = run_shaped_rollout(model, init, hist, b_mean, a_mean, horizon=60000, seed=41, device=device).cpu().numpy()

    # 2. Simulate standard 11-peak oscillator bank
    from modelling.frequency_hybrid.local_excess_remainder import load_bank, bank_filters, simulate_bank
    osc_dir = full_data.manifest['local_excess']['oscillator_directory']
    _, models_bank = load_bank(osc_dir)
    filters_bank = bank_filters(models_bank, fs)
    
    scored_len = gen_rem.shape[2] - 2 * edge
    bank_draw = simulate_bank(models_bank, filters_bank, edge, scored_len, (2, 1, scored_len, 11), seed=1041)

    # Hybrid without extra oscillator
    gen_rem_cropped = gen_rem[:, :, edge:-edge, :]
    hybrid_base = gen_rem_cropped + bank_draw

    # 3. Simulate extra 1232 Hz oscillator
    extra_osc_path = 'runs/diagnostics/extra_oscillators/oscillator_1232.npz'
    with np.load(extra_osc_path) as saved:
        extra_model = dict(saved)
    sos_extra = butter(int(extra_model['filter_order']), np.asarray(extra_model['edges_hz'], dtype=float),
                       btype='bandpass', fs=fs, output='sos')
    edge_extra = max(int(np.ceil(0.1 * fs)), settling_samples(sos_extra, fs))
    added = simulate_complex_band(extra_model, scored_len + 2 * edge_extra, paths=2, seed=1551)
    added = sosfiltfilt(sos_extra, added, axis=-2, padlen=edge_extra)[..., edge_extra:-edge_extra, :].reshape(2, 1, scored_len, 11)

    hybrid_augmented = hybrid_base + added

    # Full reference cropped
    ref_full = np.stack([full_data.read_coordinates(name, int(start) + edge, int(start) + edge + scored_len)[1]
                         for name, start in zip(reps, starts)]) # [2, scored_len, 11]

    # Energy calculations
    basis_path = '/home/jonas/ucsd_thesis/reduced_data/0p10/shared_pod/shared_pod_r1000.h5'
    import h5py
    with h5py.File(basis_path, 'r') as h5:
        modes = h5['pod']['modes'][:10]
        x_m = h5['grid']['x'][:] * length_scale('mm')
        y_m = h5['grid']['y'][:] * length_scale('mm')
    gamma = 0.0728
    stiffness, _ = capillary_stiffness(modes, x_m, y_m, gamma=gamma)
    opts = dict(exact=False, batch_size=256, stiffness=stiffness, include_spatial_mean=True)

    E_ref_raw = np.mean(compute_energies(ref_full, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])
    E_base_raw = np.mean(compute_energies(hybrid_base[:, 0], modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])
    E_aug_raw = np.mean(compute_energies(hybrid_augmented[:, 0], modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])

    # 50 Hz high-passed energy
    sos50 = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
    ref_full_hp = ref_full.copy()
    ref_full_hp[..., 1:] = sosfiltfilt(sos50, ref_full_hp[..., 1:], axis=-2)
    hybrid_base_hp = hybrid_base.copy()
    hybrid_base_hp[..., 1:] = sosfiltfilt(sos50, hybrid_base_hp[..., 1:], axis=-2)
    hybrid_aug_hp = hybrid_augmented.copy()
    hybrid_aug_hp[..., 1:] = sosfiltfilt(sos50, hybrid_aug_hp[..., 1:], axis=-2)

    E_ref_hp = np.mean(compute_energies(ref_full_hp, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])
    E_base_hp = np.mean(compute_energies(hybrid_base_hp[:, 0], modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])
    E_aug_hp = np.mean(compute_energies(hybrid_aug_hp[:, 0], modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic'])

    # Spectral comparisons
    f_w, p_ref_mean = signal.welch(ref_full[0, :, 0], fs=fs, nperseg=8192, noverlap=4096)
    p_ref_m = np.mean([signal.welch(ref_full[c, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_base_m = np.mean([signal.welch(hybrid_base[c, 0, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_aug_m = np.mean([signal.welch(hybrid_augmented[c, 0, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)

    p_ref_pod = np.mean([sum(signal.welch(ref_full[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)
    p_base_pod = np.mean([sum(signal.welch(hybrid_base[c, 0, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)
    p_aug_pod = np.mean([sum(signal.welch(hybrid_augmented[c, 0, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)

    bands = [(50, 100), (100, 300), (300, 1000), (1000, 1400), (1150, 1320), (1400, 2500), (2500, 5000), (5000, 20000)]

    print('\n================== SUMMARY RESULTS ==================')
    print('--- Capillary Deformation Energy ---')
    print(f'Raw Energy: Ref = {E_ref_raw:.2e} J')
    print(f'  Base Hybrid (Shaped Noise):  {E_base_raw:.2e} J (Ratio = {E_base_raw/E_ref_raw:.4f})')
    print(f'  Augmented (+1232Hz mode):   {E_aug_raw:.2e} J (Ratio = {E_aug_raw/E_ref_raw:.4f})')
    print(f'50 Hz HP Energy (Physical Waves, sloshing removed): Ref = {E_ref_hp:.2e} J')
    print(f'  Base Hybrid (Shaped Noise):  {E_base_hp:.2e} J (Ratio = {E_base_hp/E_ref_hp:.4f})')
    print(f'  Augmented (+1232Hz mode):   {E_aug_hp:.2e} J (Ratio = {E_aug_hp/E_ref_hp:.4f})')

    print('\n--- Spatial Mean Power Ratios (Gen / Ref) ---')
    for lo, hi in bands:
        m = (f_w >= lo) & (f_w < hi)
        r_b = np.sum(p_base_m[m]) / np.sum(p_ref_m[m])
        r_a = np.sum(p_aug_m[m]) / np.sum(p_ref_m[m])
        print(f'{lo:5d} - {hi:5d} Hz:  Shaped = {r_b:.4f}  |  Augmented = {r_a:.4f}')

    print('\n--- POD Trace Power Ratios (Gen / Ref) ---')
    for lo, hi in bands:
        m = (f_w >= lo) & (f_w < hi)
        r_b = np.sum(p_base_pod[m]) / np.sum(p_ref_pod[m])
        r_a = np.sum(p_aug_pod[m]) / np.sum(p_ref_pod[m])
        print(f'{lo:5d} - {hi:5d} Hz:  Shaped = {r_b:.4f}  |  Augmented = {r_a:.4f}')

    # Save output artifacts
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'augmented_hybrid_rollout.npz',
                        reference=ref_full,
                        trajectories=hybrid_augmented,
                        base_trajectories=hybrid_base)
    out_file = out_dir / "augmented_hybrid_rollout.npz"
    print(f'\nArtifact saved to {out_file}')

if __name__ == '__main__':
    main()
