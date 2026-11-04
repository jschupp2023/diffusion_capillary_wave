"""Self-regulating Hybrid-Aware and Exogenous-Conditioned Neural SDE training for 0p08.

Implements three architectures:
1. Model A (Hybrid Energy-Regulated): Evaluates loss on reconstructed hybrid signal
   (x_hybrid = x_rem + x_osc) with capillary energy and modal variance penalties.
   Optimizes both drift and diffusion parameters simultaneously.
2. Model B (Exogenous Conditioned): Augments TCN drift input with the known oscillator
   state [x_rem, v_rem, x_osc, v_osc] (dimension 44).
3. Model C (Unified): Exogenous conditioning + hybrid energy-regulated diffusion.

All models are trained and evaluated WITHOUT any manual diffusion scaling (s_vec = 1.0)
or manual bank scaling (bank_scales = 1.0).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank
from modelling.neural_sde.model import NeuralSDE, TCNDrift, DampedResidualDrift
from modelling.neural_sde.train import load_model


def load_0p08_dataset(data_dir: Path):
    """Load pre-computed 0p08 remainder, full, and oscillator coordinate series."""
    train_reps = [f'Ca_ac_0p000664_rep{i}' for i in range(1, 13)]
    val_reps = ['Ca_ac_0p000664_rep13', 'Ca_ac_0p000664_rep14']
    
    data = {'train': {}, 'val': {}}
    for split, reps in [('train', train_reps), ('val', val_reps)]:
        for r in reps:
            full = np.load(data_dir / f'{r}_full.npy')
            rem = np.load(data_dir / f'{r}_remainder.npy')
            osc = full - rem
            data[split][r] = {
                'full': torch.tensor(full, dtype=torch.float32),
                'rem': torch.tensor(rem, dtype=torch.float32),
                'osc': torch.tensor(osc, dtype=torch.float32),
                'len': len(full)
            }
    return data


def sample_batch(data_split, batch_size, horizon, history_steps=15, device='cuda'):
    """Sample random contiguous training windows."""
    reps = list(data_split.keys())
    sampled_reps = np.random.choice(reps, size=batch_size, replace=True)
    
    inits = []
    hists = []
    osc_inits = []
    osc_hists = []
    ref_fulls = []
    ref_oscs = []
    ref_rems = []
    
    for r in sampled_reps:
        rep_len = data_split[r]['len']
        start = np.random.randint(history_steps, rep_len - horizon - 1)
        
        rem = data_split[r]['rem']
        full = data_split[r]['full']
        osc = data_split[r]['osc']
        
        inits.append(rem[start])
        hists.append(rem[start - history_steps : start].flip(dims=[0]))
        osc_inits.append(osc[start])
        osc_hists.append(osc[start - history_steps : start].flip(dims=[0]))
        ref_fulls.append(full[start + 1 : start + horizon + 1])
        ref_oscs.append(osc[start + 1 : start + horizon + 1])
        ref_rems.append(rem[start + 1 : start + horizon + 1])
        
    return {
        'init': torch.stack(inits).to(device),
        'hist': torch.stack(hists).to(device),
        'osc_init': torch.stack(osc_inits).to(device),
        'osc_hist': torch.stack(osc_hists).to(device),
        'ref_full': torch.stack(ref_fulls).to(device), # [B, H, 11]
        'ref_osc': torch.stack(ref_oscs).to(device),   # [B, H, 11]
        'ref_rem': torch.stack(ref_rems).to(device)    # [B, H, 11]
    }


def compute_capillary_energy(state: torch.Tensor, stiffness_t: torch.Tensor) -> torch.Tensor:
    """Compute physical capillary deformation energy on state [..., 11]."""
    # Coordinate 0 is spatial mean; coordinates 1..10 are POD modes
    a = state[..., 1:] * 1e-6 # convert microns to meters
    return 0.5 * (a @ stiffness_t * a).sum(dim=-1)


class OscillatorCoupledNeuralSDE(nn.Module):
    """Neural SDE with stable linear cross-coupling from the linear oscillator bank.
    
    Couples the remainder drift with the instantaneous oscillator phase and amplitude:
      drift_coupled = drift_base(x_rem, v_rem) + target_std * (W_coup @ [x_osc, v_osc])
    where W_coup is initialized to zero, ensuring perfect initial stability.
    """
    def __init__(self, base_model: NeuralSDE):
        super().__init__()
        self.base = base_model
        self.config = base_model.config
        # Linear cross-coupling: [x_osc, v_osc] (22 dims) -> 11 dims drift
        self.w_coup = nn.Parameter(torch.zeros(11, 22))

    def drift(self, state, history, velocity, osc_state, osc_velocity, tcn_cache=None) -> torch.Tensor:
        base_drift = self.base.cached_tcn_drift(state, history, velocity, tcn_cache)
        norm_pos = (osc_state - self.base.state_mean) / self.base.state_std
        norm_vel = (osc_velocity - self.base.target_mean) / self.base.target_std
        z = torch.cat([norm_pos, norm_vel], dim=-1) # [..., 22]
        coupling = z @ self.w_coup.T
        return base_drift + self.base.target_std * coupling

    def diffusion_matrix(self) -> torch.Tensor:
        return self.base.diffusion_matrix()

    def new_tcn_cache(self):
        return self.base.new_tcn_cache()


def rollout_model_a(model: NeuralSDE, init: torch.Tensor, hist: torch.Tensor, horizon: int):
    """Roll out Model A (standard SDE) differentiably."""
    state = init.unsqueeze(1)
    history = hist.unsqueeze(1)
    velocity = state - history[..., 0, :]
    
    step_size = model.config.lag_steps
    factor = math.sqrt(step_size) * model.diffusion_matrix()
    
    states = []
    tcn_cache = model.new_tcn_cache()
    for _ in range(horizon):
        drift = model.cached_tcn_drift(state, history, velocity, tcn_cache)
        mean = velocity + step_size * drift
        noise = torch.randn_like(velocity)
        stochastic = torch.matmul(noise, factor.T)
        velocity = mean + stochastic
        state = state + step_size * velocity
        history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
        states.append(state.squeeze(1))
    return torch.stack(states, dim=1)


def rollout_model_coupled(model: OscillatorCoupledNeuralSDE, init: torch.Tensor, hist: torch.Tensor,
                          osc_init: torch.Tensor, osc_hist: torch.Tensor,
                          osc_seq: torch.Tensor, horizon: int):
    """Roll out OscillatorCoupled SDE differentiably."""
    state = init.unsqueeze(1)
    history = hist.unsqueeze(1)
    velocity = state - history[..., 0, :]
    
    osc_state = osc_init.unsqueeze(1)
    osc_history = osc_hist.unsqueeze(1)
    osc_velocity = osc_state - osc_history[..., 0, :]
    
    step_size = model.config.lag_steps
    factor = math.sqrt(step_size) * model.diffusion_matrix()
    
    states = []
    tcn_cache = model.new_tcn_cache()
    for step in range(horizon):
        drift = model.drift(state, history, velocity, osc_state, osc_velocity, tcn_cache)
        mean = velocity + step_size * drift
        noise = torch.randn_like(velocity)
        stochastic = torch.matmul(noise, factor.T)
        velocity = mean + stochastic
        state = state + step_size * velocity
        history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
        states.append(state.squeeze(1))
        
        # Advance oscillator state from reference sequence
        next_osc = osc_seq[:, step].unsqueeze(1)
        osc_velocity = next_osc - osc_state
        osc_history = torch.cat((osc_state.unsqueeze(-2), osc_history[..., :-1, :]), dim=-2)
        osc_state = next_osc
        
    return torch.stack(states, dim=1)


def train_model(mode: str, data_dir: Path, output_dir: Path, num_steps=600, lr_drift=2e-5, lr_diff=1e-4, horizon=40, batch_size=32, seed=42):
    """Train Model A, Model B, or Model C."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'=== Starting Training: {mode.upper()} ===')
    print(f'Device: {device}, Steps: {num_steps}, Batch Size: {batch_size}, Horizon: {horizon}')
    
    # Load dataset
    print('Loading 0p08 dataset into RAM...')
    data = load_0p08_dataset(data_dir)
    
    # Load stiffness
    basis_path = '/home/jonas/ucsd_thesis/reduced_data/0p08/shared_pod/shared_pod_r1000.h5'
    with h5py.File(basis_path, 'r') as h5:
        pod_modes = h5['pod']['modes'][:10]
        x_m = h5['grid']['x'][:] * length_scale('mm')
        y_m = h5['grid']['y'][:] * length_scale('mm')
    stiffness, _ = capillary_stiffness(pod_modes, x_m, y_m, gamma=0.0728)
    stiffness_t = torch.tensor(stiffness, dtype=torch.float32, device=device)
    
    # Initialize base model
    base_checkpoint = 'runs/adaptive/neural_sde_0p08_local_excess_remainder_v1/trials/tcn15_nll_e5/best.pt'
    base_model, _ = load_model(base_checkpoint, device)
    
    if mode == 'model_a':
        model = base_model
        params = [
            {'params': [p for n, p in model.drift_model.named_parameters()], 'lr': lr_drift},
            {'params': [model.diffusion_model.raw_diagonal], 'lr': lr_diff}
        ]
    elif mode == 'model_b':
        model = OscillatorCoupledNeuralSDE(base_model).to(device)
        params = [
            {'params': [model.w_coup], 'lr': lr_drift * 10.0}
        ]
    elif mode == 'model_c':
        model = OscillatorCoupledNeuralSDE(base_model).to(device)
        params = [
            {'params': [model.w_coup], 'lr': lr_drift * 10.0},
            {'params': [p for n, p in model.base.drift_model.named_parameters()], 'lr': lr_drift},
            {'params': [model.base.diffusion_model.raw_diagonal], 'lr': lr_diff}
        ]
    else:
        raise ValueError(f'Unknown mode {mode}')
        
    optimizer = torch.optim.Adam(params)
    
    best_val_loss = float('inf')
    best_checkpoint_path = output_dir / 'best.pt'
    history_log = []
    
    start_time = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        batch = sample_batch(data['train'], batch_size, horizon, device=device)
        
        optimizer.zero_grad()
        
        if mode == 'model_a':
            gen_rem = rollout_model_a(model, batch['init'], batch['hist'], horizon)
            gen_hyb = gen_rem + batch['ref_osc']
            
            # Hybrid Energy Score
            L_es = (gen_hyb - batch['ref_full']).square().sum(dim=-1).mean() / (batch['ref_full'].var().item() + 1e-6)
            
            # Hybrid Capillary Energy
            E_gen = compute_capillary_energy(gen_hyb, stiffness_t).mean()
            E_ref = compute_capillary_energy(batch['ref_full'], stiffness_t).mean()
            L_cap = (E_gen / (E_ref + 1e-12) - 1.0).square()
            
            # Hybrid Modal Variances
            var_gen = gen_hyb.var(dim=(0, 1), unbiased=True)
            var_ref = batch['ref_full'].var(dim=(0, 1), unbiased=True)
            L_var = ((var_gen / torch.clamp(var_ref, min=1e-6)) - 1.0).square().mean()
            
            loss = L_es + 10.0 * L_cap + 2.0 * L_var
            
        elif mode == 'model_b':
            # Oscillator cross-coupling with remainder loss
            gen_rem = rollout_model_coupled(model, batch['init'], batch['hist'],
                                            batch['osc_init'], batch['osc_hist'],
                                            batch['ref_osc'], horizon)
            L_rem = (gen_rem - batch['ref_rem']).square().sum(dim=-1).mean() / (batch['ref_rem'].var().item() + 1e-6)
            loss = L_rem
            
            E_gen = compute_capillary_energy(gen_rem + batch['ref_osc'], stiffness_t).mean()
            E_ref = compute_capillary_energy(batch['ref_full'], stiffness_t).mean()
            L_cap = (E_gen / (E_ref + 1e-12) - 1.0).square()
            L_var = torch.tensor(0.0)
            
        elif mode == 'model_c':
            # Unified: Oscillator cross-coupling + hybrid energy regularization
            gen_rem = rollout_model_coupled(model, batch['init'], batch['hist'],
                                            batch['osc_init'], batch['osc_hist'],
                                            batch['ref_osc'], horizon)
            gen_hyb = gen_rem + batch['ref_osc']
            
            L_es = (gen_hyb - batch['ref_full']).square().sum(dim=-1).mean() / (batch['ref_full'].var().item() + 1e-6)
            E_gen = compute_capillary_energy(gen_hyb, stiffness_t).mean()
            E_ref = compute_capillary_energy(batch['ref_full'], stiffness_t).mean()
            L_cap = (E_gen / (E_ref + 1e-12) - 1.0).square()
            
            var_gen = gen_hyb.var(dim=(0, 1), unbiased=True)
            var_ref = batch['ref_full'].var(dim=(0, 1), unbiased=True)
            L_var = ((var_gen / torch.clamp(var_ref, min=1e-6)) - 1.0).square().mean()
            
            loss = L_es + 10.0 * L_cap + 2.0 * L_var
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # Validation
        if step % 50 == 0 or step == num_steps:
            model.eval()
            with torch.no_grad():
                val_batch = sample_batch(data['val'], batch_size=16, horizon=horizon, device=device)
                if mode == 'model_a':
                    v_gen_rem = rollout_model_a(model, val_batch['init'], val_batch['hist'], horizon)
                    v_hyb = v_gen_rem + val_batch['ref_osc']
                else:
                    v_gen_rem = rollout_model_coupled(model, val_batch['init'], val_batch['hist'],
                                                      val_batch['osc_init'], val_batch['osc_hist'],
                                                      val_batch['ref_osc'], horizon)
                    v_hyb = v_gen_rem + val_batch['ref_osc']
                    v_hyb = v_gen_rem + val_batch['ref_osc']
                    
                v_E_gen = compute_capillary_energy(v_hyb, stiffness_t).mean()
                v_E_ref = compute_capillary_energy(val_batch['ref_full'], stiffness_t).mean()
                v_E_ratio = (v_E_gen / v_E_ref).item()
                
                v_var_gen = v_hyb.var(dim=(0, 1))
                v_var_ref = val_batch['ref_full'].var(dim=(0, 1))
                v_var_ratios = (v_var_gen / v_var_ref).cpu().numpy()
                
                v_loss = abs(v_E_ratio - 1.0) + np.mean(np.abs(v_var_ratios - 1.0))
                
                print(f'Step {step:03d}/{num_steps:03d} | Loss: {loss.item():.4f} | E_ratio: {v_E_ratio:.3f} | '
                      f'Mean Var Ratio: {np.mean(v_var_ratios):.3f} (M1: {v_var_ratios[1]:.2f}, M2: {v_var_ratios[2]:.2f})')
                
                history_log.append({
                    'step': step,
                    'train_loss': float(loss.item()),
                    'val_loss': float(v_loss),
                    'val_energy_ratio': float(v_E_ratio),
                    'val_mean_var_ratio': float(np.mean(v_var_ratios)),
                    'val_m1_var_ratio': float(v_var_ratios[1]),
                    'val_m2_var_ratio': float(v_var_ratios[2]),
                    'diffusion_diag': model.diffusion_matrix().diag().detach().cpu().numpy().tolist()
                })
                
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    torch.save({
                        'step': step,
                        'mode': mode,
                        'model_state': model.state_dict(),
                        'base_checkpoint': base_checkpoint,
                        'val_metrics': history_log[-1]
                    }, best_checkpoint_path)
                    print(f'  --> Saved new best checkpoint (val_loss = {v_loss:.4f})')
                    
    elapsed = time.time() - start_time
    print(f'Training completed in {elapsed:.1f}s. Best checkpoint saved to {best_checkpoint_path}')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history_log, f, indent=2)
        
    return best_checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['model_a', 'model_b', 'model_c'], required=True)
    parser.add_argument('--data-dir', type=Path, default='runs/adaptive/neural_sde_0p08_local_excess_remainder_v1/data')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=40)
    args = parser.parse_args()
    
    train_model(args.mode, args.data_dir, args.output_dir, num_steps=args.steps,
                batch_size=args.batch_size, horizon=args.horizon)
