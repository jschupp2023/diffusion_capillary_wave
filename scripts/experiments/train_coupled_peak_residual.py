"""Train and evaluate a Coupled Peak-to-Residual Hybrid Model.

In this model, the coherent linear oscillator bank informs the neural SDE residual:
The neural SDE drift evolves the remainder coordinates x_rem conditioned on BOTH
the remainder history AND the instantaneous/past states of the resonant peaks x_osc.

Compares:
1. 'decoupled': Standard production baseline (Neural SDE receives only remainder history).
2. 'coupled': Peak-to-residual coupled model (Neural SDE receives remainder history + peak state history + linear cross-modal coupling).

Usage:
    python scripts/experiments/train_coupled_peak_residual.py --experiment 0p07 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank

PYTHON_BIN = "/home/jonas/miniforge3/envs/diff-model-mit/bin/python"
DATA_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data/clean_red_data")

CENTERS_GRID = [
    73.83, 126.56, 200.39, 274.22, 362.11,
    450.00, 537.89, 636.33, 741.80, 857.81, 980.86
]

BENCHMARK_EXACT_PJ = {
    "0p07": [5.95, 7.23, 24.10],
    "0p08": [8.65, 15.18, 38.01],
    "0p10": [42.34, 53.59, 65.56],
    "0p15": [186.47, 213.36, 232.83],
    "0p20": [337.47, 381.23, 390.27],
    "0p25": [410.87, 448.71, 500.85],
}


class SPSDOperator(nn.Module):
    """Symmetric positive semi-definite matrix: L L^T + eps * I."""
    def __init__(self, dimension: int, init_scale: float = 0.01):
        super().__init__()
        self.dimension = dimension
        raw_diag = math.sqrt(max(init_scale, 1e-4))
        entries = torch.zeros(dimension, dimension)
        entries.diagonal().fill_(raw_diag)
        self.entries = nn.Parameter(entries)

    def forward(self) -> torch.Tensor:
        L = torch.tril(self.entries)
        return L @ L.T + 1e-6 * torch.eye(self.dimension, device=self.entries.device)


class CausalTemporalBlock(nn.Module):
    """Causal dilated 1D convolutional residual block."""
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        x_pad = F.pad(x, (self.padding, 0))
        out = self.act(self.conv1(x_pad))
        out = self.conv2(out)
        return self.act(out + residual)


class CausalTCN(nn.Module):
    """Causal TCN encoder mapping sequence [B, T, D_in] -> last token [B, D_out]."""
    def __init__(self, in_features: int, out_features: int, channels: int = 32,
                 kernel_size: int = 3, dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, channels, 1)
        self.blocks = nn.ModuleList([
            CausalTemporalBlock(channels, kernel_size, d) for d in dilations
        ])
        self.out_proj = nn.Conv1d(channels, out_features, 1)
        # Initialize output projection to zero so model starts strictly as linear base
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, D_in]
        x = seq.transpose(1, 2)  # [B, D_in, T]
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        out = self.out_proj(x)   # [B, D_out, T]
        return out[..., -1]      # Return newest frame [B, D_out]


class CoupledPeakResidualSDE(nn.Module):
    """Hybrid Neural SDE where coherent peaks inform the broadband remainder drift."""
    def __init__(self, dimension: int = 10, context_frames: int = 15,
                 coupled: bool = True, channels: int = 32):
        super().__init__()
        self.dimension = dimension
        self.context_frames = context_frames
        self.coupled = coupled

        # Stable linear operators for remainder
        self.stiffness = SPSDOperator(dimension, init_scale=0.01)
        self.damping = SPSDOperator(dimension, init_scale=0.01)

        # Cross-modal linear coupling from peaks into remainder
        if coupled:
            self.cross_stiffness = nn.Parameter(torch.zeros(dimension, dimension))
            self.cross_damping = nn.Parameter(torch.zeros(dimension, dimension))
            in_features = 2 * dimension + 2 * dimension  # [x_rem, v_rem, x_osc, v_osc]
        else:
            self.register_parameter("cross_stiffness", None)
            self.register_parameter("cross_damping", None)
            in_features = 2 * dimension                  # [x_rem, v_rem]

        # Nonlinear TCN residual
        self.tcn = CausalTCN(in_features=in_features, out_features=dimension,
                            channels=channels, kernel_size=3, dilations=(1, 2, 4))

        # Diagonal diffusion (constant)
        self.raw_diagonal = nn.Parameter(torch.full((dimension,), math.log(math.exp(0.1) - 1.0)))

        # Normalization buffers
        for name in ["x_rem_mean", "x_rem_std", "v_rem_mean", "v_rem_std",
                     "x_osc_mean", "x_osc_std", "v_osc_mean", "v_osc_std"]:
            self.register_buffer(name, torch.zeros(dimension))

    @property
    def diffusion_std(self) -> torch.Tensor:
        return F.softplus(self.raw_diagonal)

    def set_normalization(self, stats: dict[str, np.ndarray]):
        for k, v in stats.items():
            getattr(self, k).copy_(torch.as_tensor(v, dtype=torch.float32))

    def _drift(self, x_rem_seq: torch.Tensor, v_rem_seq: torch.Tensor,
               x_osc_seq: torch.Tensor | None = None,
               v_osc_seq: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized drift for the newest frame."""
        # Current (newest) standardized state
        x_rem_curr = (x_rem_seq[:, -1, :] - self.x_rem_mean) / self.x_rem_std
        v_rem_curr = (v_rem_seq[:, -1, :] - self.v_rem_mean) / self.v_rem_std

        # Dissipative SPSD linear base
        linear = -x_rem_curr @ self.stiffness() - v_rem_curr @ self.damping()

        # Cross-modal linear coupling from peak state
        if self.coupled and x_osc_seq is not None and v_osc_seq is not None:
            x_osc_curr = (x_osc_seq[:, -1, :] - self.x_osc_mean) / self.x_osc_std
            v_osc_curr = (v_osc_seq[:, -1, :] - self.v_osc_mean) / self.v_osc_std
            linear = linear - x_osc_curr @ self.cross_stiffness - v_osc_curr @ self.cross_damping

        # Standardized TCN sequence input
        norm_x_rem = (x_rem_seq - self.x_rem_mean) / self.x_rem_std
        norm_v_rem = (v_rem_seq - self.v_rem_mean) / self.v_rem_std

        if self.coupled and x_osc_seq is not None and v_osc_seq is not None:
            norm_x_osc = (x_osc_seq - self.x_osc_mean) / self.x_osc_std
            norm_v_osc = (v_osc_seq - self.v_osc_mean) / self.v_osc_std
            seq_in = torch.cat([norm_x_rem, norm_v_rem, norm_x_osc, norm_v_osc], dim=-1)
        else:
            seq_in = torch.cat([norm_x_rem, norm_v_rem], dim=-1)

        residual = self.tcn(seq_in)
        return linear + residual

    def forward_step(self, x_rem_seq: torch.Tensor, v_rem_seq: torch.Tensor,
                     x_osc_seq: torch.Tensor | None = None,
                     v_osc_seq: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute predicted velocity mean and physical noise std for the next frame."""
        norm_drift = self._drift(x_rem_seq, v_rem_seq, x_osc_seq, v_osc_seq)
        v_curr = v_rem_seq[:, -1, :]
        v_mean = v_curr + self.v_rem_std * norm_drift
        v_noise_std = (self.v_rem_std * self.diffusion_std).unsqueeze(0).expand(v_mean.shape[0], -1)
        return v_mean, v_noise_std


def load_dataset_sequences(manifest_path: Path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    data_dir = manifest_path.parent

    records = manifest["records"]
    train_reps, val_reps = [], []

    # Map keys to repetition numbers
    rep_map = {}
    for k in records.keys():
        m = re.search(r"(?:_rep|^rep)(\d+)$", k)
        if m:
            r = int(m.group(1))
            rep_map[r] = k

    train_data = []
    for r in range(1, 13):
        if r in rep_map:
            k = rep_map[r]
            rem_p = data_dir / records[k]["files"]["remainder"]["path"]
            full_p = data_dir / records[k]["files"]["full"]["path"]
            rem = np.load(rem_p)
            full = np.load(full_p)
            osc = full - rem
            train_data.append({"rem": rem, "osc": osc, "full": full})

    val_data = []
    for r in [13, 14]:
        if r in rep_map:
            k = rep_map[r]
            rem_p = data_dir / records[k]["files"]["remainder"]["path"]
            full_p = data_dir / records[k]["files"]["full"]["path"]
            rem = np.load(rem_p)
            full = np.load(full_p)
            osc = full - rem
            val_data.append({"rem": rem, "osc": osc, "full": full})

    return train_data, val_data, manifest


def compute_normalization_stats(train_data: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    all_x_rem = np.concatenate([d["rem"] for d in train_data], axis=0)
    all_v_rem = np.concatenate([d["rem"][1:] - d["rem"][:-1] for d in train_data], axis=0)
    all_x_osc = np.concatenate([d["osc"] for d in train_data], axis=0)
    all_v_osc = np.concatenate([d["osc"][1:] - d["osc"][:-1] for d in train_data], axis=0)

    stats = {
        "x_rem_mean": np.mean(all_x_rem, axis=0),
        "x_rem_std": np.std(all_x_rem, axis=0) + 1e-6,
        "v_rem_mean": np.mean(all_v_rem, axis=0),
        "v_rem_std": np.std(all_v_rem, axis=0) + 1e-6,
        "x_osc_mean": np.mean(all_x_osc, axis=0),
        "x_osc_std": np.std(all_x_osc, axis=0) + 1e-6,
        "v_osc_mean": np.mean(all_v_osc, axis=0),
        "v_osc_std": np.std(all_v_osc, axis=0) + 1e-6,
    }
    return stats


def sample_training_batch(train_data: list[dict[str, np.ndarray]], batch_size: int,
                          context_frames: int = 15, device: str = "cuda"):
    """Sample random windows [B, context_frames, D] for remainder and peaks."""
    B = batch_size
    H = context_frames

    x_rem_list, v_rem_list = [], []
    x_osc_list, v_osc_list = [], []
    v_target_list = []

    for _ in range(B):
        rec = train_data[np.random.randint(len(train_data))]
        rem = rec["rem"]
        osc = rec["osc"]
        T = len(rem)

        # We need frames t-H to t+1: history ends at t, target is v_rem at t+1
        # Remainder velocity at step s is rem[s] - rem[s-1]
        t = np.random.randint(H + 1, T - 1)

        x_rem_win = rem[t - H + 1:t + 1]  # length H
        v_rem_win = rem[t - H + 1:t + 1] - rem[t - H:t]  # length H

        x_osc_win = osc[t - H + 1:t + 1]
        v_osc_win = osc[t - H + 1:t + 1] - osc[t - H:t]

        # Target velocity for next step t+1
        v_target = rem[t + 1] - rem[t]

        x_rem_list.append(x_rem_win)
        v_rem_list.append(v_rem_win)
        x_osc_list.append(x_osc_win)
        v_osc_list.append(v_osc_win)
        v_target_list.append(v_target)

    return (
        torch.tensor(np.stack(x_rem_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(v_rem_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(x_osc_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(v_osc_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(v_target_list), dtype=torch.float32, device=device),
    )


def energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Energy Score: E||y - Y|| - 0.5 E||Y - Y'||."""
    # samples: [B, P, D], target: [B, D]
    diff_target = torch.linalg.norm(samples - target.unsqueeze(1), dim=-1).mean(dim=1)  # [B]
    P = samples.shape[1]
    if P > 1:
        diff_pair = torch.linalg.norm(
            samples.unsqueeze(2) - samples.unsqueeze(1), dim=-1
        ).sum(dim=(1, 2)) / (P * (P - 1))  # [B]
    else:
        diff_pair = 0.0
    return (diff_target - 0.5 * diff_pair).mean()


def train_model(train_data: list[dict[str, np.ndarray]], stats: dict[str, np.ndarray],
                coupled: bool = True, epochs: int = 5, batch_size: int = 256,
                device: str = "cuda") -> CoupledPeakResidualSDE:
    model = CoupledPeakResidualSDE(dimension=10, context_frames=15, coupled=coupled).to(device)
    model.set_normalization(stats)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    batches_per_epoch = 1500  # fast & comprehensive

    print(f"--- Training {'COUPLED' if coupled else 'DECOUPLED'} Model ({epochs} epochs, {batches_per_epoch} batches/ep) ---")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_nll, total_es = 0.0, 0.0, 0.0

        for b in range(batches_per_epoch):
            x_rem, v_rem, x_osc, v_osc, v_target = sample_training_batch(
                train_data, batch_size, context_frames=15, device=device
            )

            # 1. One-step Gaussian NLL
            v_mean, v_noise_std = model.forward_step(
                x_rem, v_rem, x_osc if coupled else None, v_osc if coupled else None
            )
            nll = (torch.log(v_noise_std) + 0.5 * ((v_target - v_mean) / v_noise_std) ** 2).sum(dim=-1).mean()

            # 2. Multi-step Energy Score on short horizon (horizon=5, particles=4)
            if b % 4 == 0:
                P = 4
                x_curr = x_rem[:, -1, :].unsqueeze(1).expand(-1, P, -1).clone()
                v_curr = v_rem[:, -1, :].unsqueeze(1).expand(-1, P, -1).clone()
                x_seq = x_rem.unsqueeze(1).expand(-1, P, -1, -1).clone()
                v_seq = v_rem.unsqueeze(1).expand(-1, P, -1, -1).clone()

                # For multi-step rollout, expand oscillator conditioning
                if coupled:
                    x_osc_seq = x_osc.unsqueeze(1).expand(-1, P, -1, -1)
                    v_osc_seq = v_osc.unsqueeze(1).expand(-1, P, -1, -1)
                else:
                    x_osc_seq, v_osc_seq = None, None

                for step in range(5):
                    # Flatten batch and particle dims for model forward
                    v_m, v_s = model.forward_step(
                        x_seq.reshape(-1, 15, 10), v_seq.reshape(-1, 15, 10),
                        x_osc_seq.reshape(-1, 15, 10) if coupled else None,
                        v_osc_seq.reshape(-1, 15, 10) if coupled else None
                    )
                    v_m = v_m.reshape(batch_size, P, 10)
                    v_s = v_s.reshape(batch_size, P, 10)

                    noise = torch.randn_like(v_m)
                    v_next = v_m + v_s * noise
                    x_next = x_curr + v_next

                    x_curr = x_next
                    v_curr = v_next
                    x_seq = torch.cat([x_seq[..., 1:, :], x_curr.unsqueeze(-2)], dim=-2)
                    v_seq = torch.cat([v_seq[..., 1:, :], v_curr.unsqueeze(-2)], dim=-2)

                es_loss = energy_score(v_curr, v_target)
                loss = nll + 0.03 * es_loss
            else:
                loss = nll

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()

        print(f"  Epoch {epoch:02d}/{epochs:02d}: NLL = {total_nll/batches_per_epoch:.4f}, total_loss = {total_loss/batches_per_epoch:.4f} ({time.time() - t0:.1f}s)")

    return model


def finetune_diffusion(model: CoupledPeakResidualSDE, train_data: list[dict[str, np.ndarray]],
                       iterations: int = 100, device: str = "cuda") -> CoupledPeakResidualSDE:
    """Fine-tune diffusion on multi-step displacement variance with frozen drift."""
    print("Fine-tuning diffusion variance on multi-step displacement horizons [20, 50, 100, 200]...")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.raw_diagonal.requires_grad = True
    optimizer = optim.Adam([model.raw_diagonal], lr=0.01)

    train_seqs_rem = [torch.tensor(d["rem"], dtype=torch.float32) for d in train_data]
    train_seqs_osc = [torch.tensor(d["osc"], dtype=torch.float32) for d in train_data]
    horizons = [20, 50, 100, 200]
    target_vars = {}
    for H in horizons:
        diffs = torch.cat([seq[H:] - seq[:-H] for seq in train_seqs_rem], dim=0)
        target_vars[H] = diffs.var(dim=0).to(device)

    max_h = max(horizons)
    h_set = set(horizons)

    best_loss = float("inf")
    best_diag = model.raw_diagonal.clone().detach()

    for it in range(1, iterations + 1):
        batch_x, batch_v = [], []
        batch_x_osc, batch_v_osc = [], []
        for _ in range(64):
            idx = np.random.randint(len(train_seqs_rem))
            seq_rem = train_seqs_rem[idx]
            seq_osc = train_seqs_osc[idx]
            t_start = np.random.randint(15, len(seq_rem) - max_h - 1)

            x_win = seq_rem[t_start - 14:t_start + 1]
            v_win = seq_rem[t_start - 14:t_start + 1] - seq_rem[t_start - 15:t_start]
            batch_x.append(x_win)
            batch_v.append(v_win)

            if model.coupled:
                # We need oscillator frames up to t_start + max_h
                x_osc_full = seq_osc[t_start - 14:t_start + max_h + 1]
                v_osc_full = seq_osc[t_start - 14:t_start + max_h + 1] - seq_osc[t_start - 15:t_start + max_h]
                batch_x_osc.append(x_osc_full)
                batch_v_osc.append(v_osc_full)

        x_seq = torch.stack(batch_x).to(device)
        v_seq = torch.stack(batch_v).to(device)
        x0 = x_seq[:, -1, :].clone()

        if model.coupled:
            x_osc_full_t = torch.stack(batch_x_osc).to(device)
            v_osc_full_t = torch.stack(batch_v_osc).to(device)
        else:
            x_osc_full_t, v_osc_full_t = None, None

        horizon_losses = []
        for step in range(1, max_h + 1):
            if model.coupled:
                curr_x_osc = x_osc_full_t[:, step - 1:step + 14, :]
                curr_v_osc = v_osc_full_t[:, step - 1:step + 14, :]
            else:
                curr_x_osc, curr_v_osc = None, None

            v_m, v_s = model.forward_step(x_seq, v_seq, curr_x_osc, curr_v_osc)
            noise = torch.randn_like(v_m)
            v_next = v_m + v_s * noise
            x_next = x_seq[:, -1, :] + v_next

            x_seq = torch.cat([x_seq[:, 1:, :], x_next.unsqueeze(1)], dim=1)
            v_seq = torch.cat([v_seq[:, 1:, :], v_next.unsqueeze(1)], dim=1)

            if step in h_set:
                gen_disp_var = (x_next - x0).var(dim=0)
                target_var = target_vars[step]
                h_loss = ((torch.log(gen_disp_var + 1e-8) - torch.log(target_var + 1e-8)) ** 2).mean()
                horizon_losses.append(h_loss)

        loss = torch.stack(horizon_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_diag = model.raw_diagonal.clone().detach()

        if it % 25 == 0 or it == iterations:
            print(f"  Diffusion step {it:03d}/{iterations}: loss = {loss.item():.4f}")

    model.raw_diagonal.data.copy_(best_diag)
    return model


def evaluate_rollout(model: CoupledPeakResidualSDE, exp: str, osc_dir: Path,
                     manifest_path: Path, val_data: list[dict[str, np.ndarray]],
                     horizon: int = 50000, device: str = "cuda") -> dict:
    model.eval()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    edge = int(manifest["edge_samples"])
    fs = float(manifest["fs"])
    scored_len = horizon - 2 * edge

    # 1. Simulate 11-band oscillator bank for scored_len + 2*edge
    print(f"Simulating 11-band oscillator bank ({scored_len} steps, bank_scales=1.0)...")
    _, models_bank = load_bank(osc_dir)
    n_coords = model.dimension
    bank_draw = np.zeros((2, scored_len + 2 * edge, n_coords), dtype=np.float64)

    overlap = 0.0578
    for idx, m in enumerate(models_bank):
        low, high = np.asarray(m["edges_hz"], dtype=float)
        width = high - low
        sos = butter(int(m.get("filter_order", 4)), [max(10.0, low - overlap * width), high + overlap * width],
                     btype="bandpass", fs=fs, output="sos")
        raw = simulate_complex_band(m, scored_len + 2 * edge, paths=2, seed=1041 + idx)
        filt = sosfiltfilt(sos, raw, axis=1, padlen=edge)
        bank_draw += filt

    # Cut to scored length
    bank_crop = bank_draw[:, edge:edge + scored_len, :]

    # 2. Rollout Remainder Neural SDE
    print(f"Generating autonomous Remainder rollout ({scored_len} steps, coupled={model.coupled})...")
    # Initial condition from validation sequences (reps 13 & 14)
    init_x = np.stack([val_data[i]["rem"][edge - 14:edge + 1] for i in range(2)])
    init_v = np.stack([val_data[i]["rem"][edge - 14:edge + 1] - val_data[i]["rem"][edge - 15:edge] for i in range(2)])

    x_seq = torch.tensor(init_x, dtype=torch.float32, device=device)  # [2, 15, 10]
    v_seq = torch.tensor(init_v, dtype=torch.float32, device=device)  # [2, 15, 10]

    bank_tensor = torch.tensor(bank_draw, dtype=torch.float32, device=device)
    v_bank_tensor = torch.zeros_like(bank_tensor)
    v_bank_tensor[:, 1:, :] = bank_tensor[:, 1:, :] - bank_tensor[:, :-1, :]

    gen_rem_states = []
    generator = torch.Generator(device=device).manual_seed(41)

    with torch.no_grad():
        for t in range(scored_len):
            current_idx = edge + t
            if model.coupled:
                x_osc_seq = bank_tensor[:, current_idx - 14:current_idx + 1, :]
                v_osc_seq = v_bank_tensor[:, current_idx - 14:current_idx + 1, :]
            else:
                x_osc_seq, v_osc_seq = None, None

            v_mean, v_noise_std = model.forward_step(x_seq, v_seq, x_osc_seq, v_osc_seq)
            noise = torch.randn(v_mean.shape, device=device, dtype=v_mean.dtype, generator=generator)
            v_next = v_mean + v_noise_std * noise
            x_next = x_seq[:, -1, :] + v_next

            x_seq = torch.cat([x_seq[:, 1:, :], x_next.unsqueeze(1)], dim=1)
            v_seq = torch.cat([v_seq[:, 1:, :], v_next.unsqueeze(1)], dim=1)
            gen_rem_states.append(x_next.cpu().numpy())

    gen_rem = np.stack(gen_rem_states, axis=1)  # [2, scored_len, n_coords]
    hybrid = gen_rem + bank_crop                # [2, scored_len, n_coords]

    # Reference full signal
    ref_crop = np.stack([val_data[i]["full"][edge:edge + scored_len] for i in range(2)])

    # Physical highpass > 50 Hz
    sos50 = butter(4, 50.0, btype="highpass", fs=fs, output="sos")
    ref_hp = sosfiltfilt(sos50, ref_crop, axis=1)
    hyb_hp = sosfiltfilt(sos50, hybrid, axis=1)

    # Capillary energy
    shared_basis = DATA_ROOT / exp / "shared_pod/shared_pod_r200.h5"
    with h5py.File(shared_basis, "r") as h5:
        pod_modes = h5["pod"]["modes"][:n_coords]
        x_m = h5["grid"]["x"][:] * length_scale("mm")
        y_m = h5["grid"]["y"][:] * length_scale("mm")
    cy, cx = pod_modes.shape[1] // 2, pod_modes.shape[2] // 2
    center_weights = pod_modes[:, cy, cx]
    stiffness, _ = capillary_stiffness(pod_modes, x_m, y_m, gamma=0.0728)

    opts = dict(exact=False, batch_size=256, stiffness=stiffness, include_spatial_mean=False)
    x_dummy, y_dummy = np.zeros(200), np.zeros(200)

    E_ref_hp = float(np.mean(compute_energies(ref_hp, pod_modes, x_dummy, y_dummy, length_scale("microns"), 0.0728, **opts)["quadratic"]))
    E_gen_hp = float(np.mean(compute_energies(hyb_hp, pod_modes, x_dummy, y_dummy, length_scale("microns"), 0.0728, **opts)["quadratic"]))
    gen_exact_pJ = E_gen_hp * 1e12

    # Modal variances
    var_ref = np.var(ref_hp.reshape(-1, n_coords), axis=0)
    var_gen = np.var(hyb_hp.reshape(-1, n_coords), axis=0)
    var_ratios = (var_gen / var_ref).tolist()

    # Peak contrast and PSD
    ref_center = np.einsum("cri,i->cr", ref_crop, center_weights)
    gen_center = np.einsum("cri,i->cr", hybrid, center_weights)
    f_w, p_ref_c = signal.welch(ref_center[0], fs=fs, nperseg=8192, noverlap=4096)
    p_ref_c = np.mean([signal.welch(ref_center[c], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_gen_c = np.mean([signal.welch(gen_center[c], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)

    peak_ratios = {}
    peak_errors_db = []
    for pf in CENTERS_GRID:
        k = np.argmin(np.abs(f_w - pf))
        r = float(p_gen_c[k] / p_ref_c[k])
        peak_ratios[f"{pf:.1f}"] = r
        peak_errors_db.append(abs(10.0 * np.log10(max(r, 1e-6))))
    mean_peak_mae_db = float(np.mean(peak_errors_db))

    # Cross-peak envelope correlation between 868 and 982 Hz
    # Bandpass filter around 868 and 982 Hz
    sos_868 = butter(4, [830.0, 905.0], btype="bandpass", fs=fs, output="sos")
    sos_982 = butter(4, [940.0, 1025.0], btype="bandpass", fs=fs, output="sos")

    env_corr_list = []
    for c in range(2):
        sig_868 = sosfiltfilt(sos_868, hybrid[c, :, 0])
        sig_982 = sosfiltfilt(sos_982, hybrid[c, :, 0])
        env_868 = np.abs(signal.hilbert(sig_868))
        env_982 = np.abs(signal.hilbert(sig_982))
        r = float(np.corrcoef(env_868, env_982)[0, 1])
        env_corr_list.append(r)
    mean_env_corr_868_982 = float(np.mean(env_corr_list))

    benchmark = BENCHMARK_EXACT_PJ.get(exp, [np.nan, np.nan, np.nan])

    metrics = {
        "coupled": model.coupled,
        "physical_energy_hp50_ref_pJ": E_ref_hp * 1e12,
        "physical_energy_hp50_gen_pJ": gen_exact_pJ,
        "energy_ratio": E_gen_hp / E_ref_hp,
        "benchmark_pJ": benchmark,
        "within_benchmark": bool(benchmark[0] <= gen_exact_pJ <= benchmark[2]),
        "modal_variance_ratios": var_ratios,
        "mean_modal_variance_ratio": float(np.mean(var_ratios)),
        "mean_peak_mae_db": mean_peak_mae_db,
        "peak_ratios": peak_ratios,
        "mean_env_corr_868_982": mean_env_corr_868_982,
        "f_w": f_w.tolist(),
        "p_ref": p_ref_c.tolist(),
        "p_gen": p_gen_c.tolist(),
    }
    return metrics, hybrid, ref_crop


def plot_comparison(results: dict[str, dict], out_dir: Path, exp: str):
    """Publication-quality 4-panel comparison of Decoupled vs Coupled Peak-to-Residual Models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")

    # (a) Surface Probe PSD Comparison
    ax = axes[0, 0]
    f = np.asarray(results["coupled"]["f_w"])
    p_ref = np.asarray(results["coupled"]["p_ref"])
    p_dec = np.asarray(results["decoupled"]["p_gen"])
    p_coup = np.asarray(results["coupled"]["p_gen"])

    ax.loglog(f, p_ref, "k-", lw=2.0, label="Experiment (Reference)", alpha=0.85)
    ax.loglog(f, p_dec, "C0--", lw=1.8, label="Decoupled Baseline", alpha=0.9)
    ax.loglog(f, p_coup, "C3-", lw=2.0, label="Coupled Peak-to-Residual", alpha=0.9)
    ax.set_xlim(50, 2000)
    ax.set_ylim(1e-6, 1e2)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Surface PSD (μm²/Hz)", fontsize=11)
    ax.set_title(f"(a) Central Surface Elevation PSD ({exp})", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")

    # (b) Modal Variance Ratios
    ax = axes[0, 1]
    modes = np.arange(1, 11)
    vr_dec = np.asarray(results["decoupled"]["modal_variance_ratios"])
    vr_coup = np.asarray(results["coupled"]["modal_variance_ratios"])

    ax.plot(modes, vr_dec, "o--", color="C0", lw=1.8, ms=7, label=f"Decoupled (Mean = {np.mean(vr_dec):.2f})")
    ax.plot(modes, vr_coup, "s-", color="C3", lw=2.0, ms=7, label=f"Coupled (Mean = {np.mean(vr_coup):.2f})")
    ax.axhline(1.0, color="k", ls=":", lw=1.5, label="Perfect Match (1.0)")
    ax.axhspan(0.8, 1.2, color="gray", alpha=0.15, label="±20% Tolerance Band")
    ax.set_xticks(modes)
    ax.set_xlabel("POD Mode Index", fontsize=11)
    ax.set_ylabel("Variance Ratio (Gen / Ref)", fontsize=11)
    ax.set_title("(b) Modal Variance Reproduction Across POD Modes", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 1.6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # (c) Resonant Peak Ratios Across the 11 Bands
    ax = axes[1, 0]
    p_dec_r = [results["decoupled"]["peak_ratios"][f"{c:.1f}"] for c in CENTERS_GRID]
    p_coup_r = [results["coupled"]["peak_ratios"][f"{c:.1f}"] for c in CENTERS_GRID]
    x_pos = np.arange(len(CENTERS_GRID))
    width = 0.35

    ax.bar(x_pos - width / 2, p_dec_r, width, color="C0", alpha=0.7, label=f"Decoupled (MAE = {results['decoupled']['mean_peak_mae_db']:.2f} dB)")
    ax.bar(x_pos + width / 2, p_coup_r, width, color="C3", alpha=0.8, label=f"Coupled (MAE = {results['coupled']['mean_peak_mae_db']:.2f} dB)")
    ax.axhline(1.0, color="k", ls="--", lw=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c:.0f}" for c in CENTERS_GRID], rotation=45, fontsize=9)
    ax.set_xlabel("Nominal Resonant Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Peak Power Ratio (Gen / Ref)", fontsize=11)
    ax.set_title("(c) Discrete Resonant Peak Contrast Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 3.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper left")

    # (d) Cross-Peak Envelope Correlation & Energy
    ax = axes[1, 1]
    categories = ["Envelope Corr\n(868 / 982 Hz)", "Deform. Energy\nRatio (Gen/Ref)"]
    dec_vals = [results["decoupled"]["mean_env_corr_868_982"], results["decoupled"]["energy_ratio"]]
    coup_vals = [results["coupled"]["mean_env_corr_868_982"], results["coupled"]["energy_ratio"]]
    # Ground truth reference target: r=0.33 for 0p07, energy ratio=1.0
    ref_vals = [0.3325 if exp == "0p07" else 0.1799, 1.0]

    x_c = np.arange(len(categories))
    w = 0.25
    ax.bar(x_c - w, dec_vals, w, color="C0", alpha=0.7, label="Decoupled Baseline")
    ax.bar(x_c, coup_vals, w, color="C3", alpha=0.8, label="Coupled Peak-to-Residual")
    ax.bar(x_c + w, ref_vals, w, color="black", alpha=0.8, label="Target / Experiment")

    for i in range(len(categories)):
        ax.text(x_c[i] - w, dec_vals[i] + 0.02, f"{dec_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x_c[i], coup_vals[i] + 0.02, f"{coup_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold", color="darkred")
        ax.text(x_c[i] + w, ref_vals[i] + 0.02, f"{ref_vals[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x_c)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("(d) Key Diagnostic Comparison", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")

    plt.savefig(out_dir / f"coupled_vs_decoupled_{exp}_comparison.png", dpi=200)
    plt.close()
    print(f"Saved comparison figure to {out_dir / f'coupled_vs_decoupled_{exp}_comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="0p07", choices=["0p07", "0p10", "0p15", "0p20"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/experiments/coupled_peak_residual")
    args = parser.parse_args()

    exp = args.experiment
    out_dir = Path(args.out_dir) / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"Coupled Peak-to-Residual Experiment: Condition {exp}")
    print(f"============================================================")

    # Paths
    osc_dir = Path(f"runs/clean_production/{exp}/oscillators")
    manifest_path = Path(f"runs/clean_production/{exp}/remainder_data/data/manifest.json")

    # Load data
    train_data, val_data, manifest = load_dataset_sequences(manifest_path)
    stats = compute_normalization_stats(train_data)
    print(f"Loaded {len(train_data)} training sequences and {len(val_data)} validation sequences.")

    results = {}

    # 1. Train Decoupled Model (Baseline)
    print("\n>>> STEP 1: Training Decoupled Baseline Model <<<")
    dec_model = train_model(train_data, stats, coupled=False, epochs=args.epochs, device=args.device)
    dec_model = finetune_diffusion(dec_model, train_data, iterations=100, device=args.device)
    dec_metrics, _, _ = evaluate_rollout(dec_model, exp, osc_dir, manifest_path, val_data, horizon=50000, device=args.device)
    results["decoupled"] = dec_metrics

    # 2. Train Coupled Model (Peaks -> Remainder)
    print("\n>>> STEP 2: Training Coupled Peak-to-Residual Model <<<")
    coup_model = train_model(train_data, stats, coupled=True, epochs=args.epochs, device=args.device)
    coup_model = finetune_diffusion(coup_model, train_data, iterations=100, device=args.device)
    coup_metrics, _, _ = evaluate_rollout(coup_model, exp, osc_dir, manifest_path, val_data, horizon=50000, device=args.device)
    results["coupled"] = coup_metrics

    # Save summary
    with open(out_dir / "comparison_metrics.json", "w") as f:
        # Strip long arrays for compact summary
        compact_res = {
            k: {sk: sv for sk, sv in v.items() if sk not in ["f_w", "p_ref", "p_gen"]}
            for k, v in results.items()
        }
        json.dump(compact_res, f, indent=2)

    # Plot
    plot_comparison(results, out_dir, exp)

    # Print summary
    print("\n============================================================")
    print("FINAL EXPERIMENT RESULTS:")
    print("============================================================")
    print(f"Metric                           | Decoupled Baseline | Coupled Peak-to-Residual | Target / Ref")
    print(f"---------------------------------+--------------------+--------------------------+-------------")
    print(f"Physical Energy (pJ)             | {results['decoupled']['physical_energy_hp50_gen_pJ']:18.2f} | {results['coupled']['physical_energy_hp50_gen_pJ']:24.2f} | {results['coupled']['physical_energy_hp50_ref_pJ']:11.2f} (Target: {BENCHMARK_EXACT_PJ[exp]})")
    print(f"Physical Energy Ratio            | {results['decoupled']['energy_ratio']:18.2f} | {results['coupled']['energy_ratio']:24.2f} |        1.00")
    print(f"Mean Modal Variance Ratio        | {results['decoupled']['mean_modal_variance_ratio']:18.2f} | {results['coupled']['mean_modal_variance_ratio']:24.2f} |        1.00")
    print(f"Peak Contrast MAE (dB)           | {results['decoupled']['mean_peak_mae_db']:18.2f} | {results['coupled']['mean_peak_mae_db']:24.2f} |        0.00")
    print(f"868/982 Envelope Correlation     | {results['decoupled']['mean_env_corr_868_982']:18.3f} | {results['coupled']['mean_env_corr_868_982']:24.3f} | {0.3325 if exp == '0p07' else 0.1799:11.3f}")
    print("============================================================")


if __name__ == "__main__":
    main()
