"""Find optimal bank scales and s_vec for clean 0p20 hybrid model."""
import json
import sys
from pathlib import Path
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
import torch
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelling.neural_sde.train import load_model
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank
from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale


def evaluate_candidate(model, init, hist, models_bank, filters_bank, ref_full, pod_modes, stiffness, center_weights, fs, edge, s_vec, bank_scales, horizon=30000, device='cuda'):
    scored_len = horizon - 2 * edge
    # Rollout SDE
    s_t = torch.tensor(s_vec, dtype=init.dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(41)
    state = init[:, None, :].expand(-1, 1, -1).clone()
    history = hist[:, None, :model.config.history_steps].expand(-1, 1, -1, -1).clone()
    velocity = model._frame_velocity(state, history)
    step_size = model.config.lag_steps * (1. if model.config.state_variable == 'velocity' else model.config.dt)
    tcn_cache = model.new_tcn_cache()

    states = [state]
    with torch.no_grad():
        for _ in range(horizon):
            raw_noise = torch.randn(state.shape, device=device, dtype=state.dtype, generator=generator)
            noise = raw_noise * s_t
            base_mean, noise_factor = model.velocity_base_step_distribution(state, history, velocity, tcn_cache=tcn_cache)
            velocity = base_mean + torch.matmul(noise, noise_factor.T)
            state = state + step_size * velocity
            history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
            states.append(state)
    gen_rem = torch.stack(states, dim=2).cpu().numpy()[:, :, edge:edge + scored_len, :]

    # Sim oscillators
    bank_draw = np.zeros((2, 1, scored_len, 10), dtype=np.float64)
    for idx, (m, sos) in enumerate(zip(models_bank, filters_bank)):
        m_c = dict(m)
        m_c['coefficient_factor'] = m['coefficient_factor'] * bank_scales[idx]
        raw = simulate_complex_band(m_c, scored_len + 2 * edge, paths=2, seed=1041 + idx)
        filt = sosfiltfilt(sos, raw, axis=1, padlen=edge)[:, edge:-edge, :]
        bank_draw += filt[:, None, :, :]

    hybrid = gen_rem + bank_draw
    ref_crop = ref_full[:, edge:edge + scored_len]

    sos50 = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
    ref_hp = sosfiltfilt(sos50, ref_crop, axis=-2)
    hyb_hp = sosfiltfilt(sos50, hybrid[:, 0], axis=-2)

    opts = dict(exact=False, batch_size=256, stiffness=stiffness, include_spatial_mean=False)
    x_m = np.zeros(200) # dummy, exact=False uses stiffness directly
    y_m = np.zeros(200)
    E_ref_hp = float(np.mean(compute_energies(ref_hp, pod_modes, x_m, y_m, length_scale('microns'), 0.0728, **opts)['quadratic']))
    E_gen_hp = float(np.mean(compute_energies(hyb_hp, pod_modes, x_m, y_m, length_scale('microns'), 0.0728, **opts)['quadratic']))
    energy_ratio = E_gen_hp / E_ref_hp

    var_ref = np.var(ref_hp.reshape(-1, 10), axis=0)
    var_gen = np.var(hyb_hp.reshape(-1, 10), axis=0)
    var_ratios = var_gen / var_ref

    # Center probe PSD
    ref_center = np.einsum('cri,i->cr', ref_crop, center_weights)
    gen_center = np.einsum('cri,i->cr', hybrid[:, 0], center_weights)
    f_w, p_ref_c0 = signal.welch(ref_center[0], fs=fs, nperseg=8192, noverlap=4096)
    p_ref_c = np.mean([signal.welch(ref_center[c], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_gen_c = np.mean([signal.welch(gen_center[c], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)

    peak_locs = [73.8, 126.6, 200.4, 274.2, 362.1, 450.0, 537.9, 636.3, 741.8, 857.8, 980.9]
    peak_ratios = []
    for pf in peak_locs:
        k = np.argmin(np.abs(f_w - pf))
        peak_ratios.append(float(p_gen_c[k] / p_ref_c[k]))

    return energy_ratio, var_ratios, peak_ratios


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'runs/adaptive/neural_sde_0p20_clean_tcn15_es_v1/best.pt'
    osc_dir = 'runs/diagnostics/frequency_hybrid_0p20_clean_production_v1/oscillators'
    manifest_path = 'runs/adaptive/neural_sde_0p20_clean_local_excess_remainder_v1/data/manifest.json'
    shared_basis = '/home/jonas/ucsd_thesis/clean_red_data/0p20/shared_pod/shared_pod_r200.h5'

    model, _ = load_model(checkpoint_path, device)
    model.eval()

    manifest_p = Path(manifest_path).resolve()
    data_dir = manifest_p.parent
    with open(manifest_p) as f:
        manifest = json.load(f)
    edge = int(manifest['edge_samples'])
    fs = float(manifest['fs'])

    rem_rep13 = np.load(data_dir / 'Ca_ac_0p001660_rep13_remainder.npy')
    rem_rep14 = np.load(data_dir / 'Ca_ac_0p001660_rep14_remainder.npy')
    full_rep13 = np.load(data_dir / 'Ca_ac_0p001660_rep13_full.npy')
    full_rep14 = np.load(data_dir / 'Ca_ac_0p001660_rep14_full.npy')
    ref_full = np.stack([full_rep13, full_rep14])

    h_steps = model.config.history_steps
    init = torch.tensor(np.stack([rem_rep13[h_steps], rem_rep14[h_steps]]), dtype=torch.float32, device=device)
    hist = torch.tensor(np.stack([rem_rep13[h_steps - 1::-1].copy(), rem_rep14[h_steps - 1::-1].copy()]), dtype=torch.float32, device=device)

    _, models_bank = load_bank(osc_dir)
    overlap = 0.0578
    filters_bank = []
    for m in models_bank:
        low, high = np.asarray(m['edges_hz'], dtype=float)
        width = high - low
        new_low = max(10.0, low - overlap * width)
        new_high = high + overlap * width
        sos = butter(int(m.get('filter_order', 4)), [new_low, new_high], btype='bandpass', fs=fs, output='sos')
        filters_bank.append(sos)

    with h5py.File(shared_basis, 'r') as h5:
        pod_modes = h5['pod']['modes'][:10]
        x_m = h5['grid']['x'][:] * length_scale('mm')
        y_m = h5['grid']['y'][:] * length_scale('mm')
    cy, cx = pod_modes.shape[1] // 2, pod_modes.shape[2] // 2
    center_weights = pod_modes[:, cy, cx]
    stiffness, _ = capillary_stiffness(pod_modes, x_m, y_m, gamma=0.0728)

    # Base candidate: raw model
    e_base, v_base, p_base = evaluate_candidate(model, init, hist, models_bank, filters_bank, ref_full, pod_modes, stiffness, center_weights, fs, edge,
                                                s_vec=np.ones(10), bank_scales=np.ones(11), horizon=20000, device=device)
    print(f"Base: Energy={e_base:.4f} | V1={v_base[0]:.3f}, V2={v_base[1]:.3f}, V3={v_base[2]:.3f}, V5={v_base[4]:.3f}")

    # Fine-tuned candidate:
    # 1. Bank scales:
    # Bank 0 (73.8 Hz): was 0.51 peak ratio -> scale by 1.15
    # Bank 1 (126.6 Hz): was 1.35 peak ratio -> scale by 0.90
    # Bank 2 (200.4 Hz): was 1.72 peak ratio -> scale by 0.80
    # Bank 3 (274.2 Hz): was 1.26 peak ratio -> scale by 0.90
    # Bank 4 (362.1 Hz): was 0.74 peak ratio -> scale by 1.15
    # Bank 5 (450.0 Hz): was 2.08 peak ratio -> scale by 0.70
    # Bank 6 (537.9 Hz): was 1.44 peak ratio -> scale by 0.85
    # Bank 7 (636.3 Hz): was 3.12 peak ratio -> scale by 0.58
    # Bank 8 (741.8 Hz): was 1.58 peak ratio -> scale by 0.80
    # Bank 9 (857.8 Hz): was 1.22 peak ratio -> scale by 0.90
    # Bank 10 (980.9 Hz): was 1.00 peak ratio -> scale by 1.00
    bank_scales = np.array([1.18, 0.90, 0.80, 0.90, 1.15, 0.70, 0.85, 0.58, 0.80, 0.90, 1.00])

    # 2. s_vec for remainder:
    # Mode 1: 1.0
    # Mode 2: 0.97
    # Mode 3: 0.94
    # Mode 4: 1.12
    # Mode 5: 0.76 (controls excess Mode 5 variance!)
    # Mode 6: 0.96
    # Mode 7: 1.14
    # Mode 8: 1.05
    # Mode 9: 1.12
    # Mode 10: 1.12
    s_vec = np.array([1.00, 0.97, 0.94, 1.12, 0.76, 0.96, 1.14, 1.05, 1.12, 1.12])

    e_tuned, v_tuned, p_tuned = evaluate_candidate(model, init, hist, models_bank, filters_bank, ref_full, pod_modes, stiffness, center_weights, fs, edge,
                                                   s_vec=s_vec, bank_scales=bank_scales, horizon=20000, device=device)
    print(f"\nTuned Candidate 1: Energy={e_tuned:.4f} | V1={v_tuned[0]:.3f}, V2={v_tuned[1]:.3f}, V3={v_tuned[2]:.3f}, V5={v_tuned[4]:.3f}")
    print("Variance ratios:", [f"{x:.3f}" for x in v_tuned])
    print("Peak ratios:", [f"{x:.3f}" for x in p_tuned])


if __name__ == '__main__':
    main()

