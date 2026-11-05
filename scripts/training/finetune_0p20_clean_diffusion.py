"""Fine-tune clean 0p20 Neural SDE diffusion with frozen drift to match rollout variance.

Targeted, stable optimization:
- Drift is 100% frozen (guaranteeing exact drift stability and dynamics).
- Only the 10 diagonal diffusion parameters are optimized.
- Loss is the log-variance discrepancy over multi-step rollout horizons on remainder data.
- Uses TCN cache for high GPU throughput.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modelling.neural_sde.train import load_model


def load_dataset(data_dir: Path, reps: list[int]):
    sequences = []
    for r in reps:
        matches = sorted(list(data_dir.glob(f"*_rep{r}_remainder.npy")))
        if matches:
            arr = np.load(matches[0])
            sequences.append(torch.tensor(arr, dtype=torch.float32))
        else:
            print(f"Warning: rep{r} not found in {data_dir}")
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default="runs/adaptive/neural_sde_0p20_clean_tcn15_es_v1/best.pt")
    parser.add_argument("--output-checkpoint", type=Path,
                        default="runs/adaptive/neural_sde_0p20_clean_tcn15_es_v1/best_intrinsic.pt")
    parser.add_argument("--data-dir", type=Path,
                        default="runs/adaptive/neural_sde_0p20_clean_local_excess_remainder_v1/data")
    parser.add_argument("--horizons", type=int, nargs="+", default=[20, 50, 100, 200],
                        help="Rollout horizons to score variance")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of start states per batch")
    parser.add_argument("--iterations", type=int, default=120,
                        help="Number of gradient steps")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for diffusion parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_ckpt = Path(args.output_checkpoint).resolve()
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(str(args.checkpoint), device=args.device)
    model.eval()

    # Freeze ALL drift parameters
    for p in model.drift_model.parameters():
        p.requires_grad = False
    
    # Train ONLY diffusion raw diagonal
    model.diffusion_model.raw_diagonal.requires_grad = True
    trainable_params = [model.diffusion_model.raw_diagonal]

    print("Trainable parameters:", [name for name, p in model.named_parameters() if p.requires_grad])

    # Load training data
    train_reps = list(range(1, 13))
    train_seqs = load_dataset(args.data_dir, train_reps)
    val_reps = [13, 14]
    val_seqs = load_dataset(args.data_dir, val_reps)

    n_coords = train_seqs[0].shape[-1]
    print(f"Detected {n_coords} coordinates in training sequences.")

    # Precompute true data displacement variance at each horizon
    target_disp_vars = {}
    for H in args.horizons:
        diffs = torch.cat([seq[H:] - seq[:-H] for seq in train_seqs], dim=0)
        target_disp_vars[H] = diffs.var(dim=0).to(args.device)
        print(f"H={H:3d} data displacement variance (first 3 coords):",
              target_disp_vars[H][:3].cpu().numpy())

    init_diff = torch.diag(model.diffusion_matrix()).detach().cpu().numpy()
    print("Initial diffusion diagonal:", init_diff)

    optimizer = optim.Adam(trainable_params, lr=args.lr)

    h_steps = model.config.history_steps
    step_size = model.config.lag_steps * (1. if model.config.state_variable == 'velocity' else model.config.dt)
    max_horizon = max(args.horizons)
    horizons_set = set(args.horizons)

    best_loss = float("inf")
    best_weights = None
    history_log = []

    print(f"\nStarting diffusion displacement-variance tuning for {args.iterations} iterations across horizons {args.horizons}...")
    for it in range(1, args.iterations + 1):
        # Sample batch of starts
        batch_z0 = []
        batch_hist = []
        for _ in range(args.batch_size):
            seq = train_seqs[np.random.randint(len(train_seqs))]
            t_start = np.random.randint(h_steps, len(seq) - max_horizon - 1)
            batch_z0.append(seq[t_start])
            batch_hist.append(seq[t_start - h_steps : t_start].flip(0))
        
        z0 = torch.stack(batch_z0).unsqueeze(1).to(args.device)  # [B, 1, n_coords]
        hist = torch.stack(batch_hist).unsqueeze(1).to(args.device)  # [B, 1, h_steps, n_coords]
        velocity = model._frame_velocity(z0, hist)

        # Roll out with TCN cache
        state = z0
        history = hist
        tcn_cache = model.new_tcn_cache()

        horizon_losses = []
        horizon_ratios = {}

        for step in range(1, max_horizon + 1):
            base_mean, noise_factor = model.velocity_base_step_distribution(state, history, velocity, tcn_cache=tcn_cache)
            noise = torch.randn(state.shape, device=args.device, dtype=state.dtype)
            velocity = base_mean + torch.matmul(noise, noise_factor.T)
            state = state + step_size * velocity
            history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)

            if step in horizons_set:
                gen_disp_var = (state - z0).squeeze(1).var(dim=0)  # [n_coords]
                target_var_h = target_disp_vars[step]
                log_gen = torch.log(gen_disp_var + 1e-8)
                log_target = torch.log(target_var_h + 1e-8)
                # Weight coordinate 5 (index 4) slightly more to control excess energy
                coord_weights = torch.ones(n_coords, device=args.device)
                coord_weights[4] = 2.0
                h_loss = (coord_weights * ((log_gen - log_target) ** 2)).mean()
                horizon_losses.append(h_loss)
                horizon_ratios[step] = (gen_disp_var / target_var_h).detach().cpu().numpy()

        loss = torch.stack(horizon_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        cur_diff = torch.diag(model.diffusion_matrix()).detach().cpu().numpy()
        loss_val = loss.item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = model.state_dict().copy()

        if it % 10 == 0 or it == 1 or it == args.iterations:
            r_max = horizon_ratios[max_horizon]
            print(f"Iter {it:3d}/{args.iterations}: loss={loss_val:.4f} | "
                  f"H={max_horizon} disp ratios: mean={r_max.mean():.3f}, "
                  f"M1={r_max[0]:.3f}, M2={r_max[1]:.3f}, M3={r_max[2]:.3f}, M5={r_max[4]:.3f}")

        history_log.append({
            "iteration": it,
            "loss": loss_val,
            "diffusion_diagonal": cur_diff.tolist(),
            "horizon_ratios": {h: r.tolist() for h, r in horizon_ratios.items()}
        })

    # Restore best weights and save
    print(f"\nOptimization complete! Best loss: {best_loss:.4f}")
    model.load_state_dict(best_weights)
    final_diff = torch.diag(model.diffusion_matrix()).detach().cpu().numpy()
    print("Final tuned diffusion diagonal:", final_diff)

    # Save checkpoint
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_data["model_state"] = model.state_dict()
    if "training" not in ckpt_data:
        ckpt_data["training"] = {}
    ckpt_data["training"]["diffusion_variance_finetuned"] = True
    ckpt_data["training"]["diffusion_tuning_loss"] = best_loss

    torch.save(ckpt_data, output_ckpt)
    with open(output_ckpt.parent / "tuning_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"Saved fine-tuned checkpoint to: {output_ckpt}")


if __name__ == "__main__":
    main()

