"""Universal fine-tuning script for Neural SDE diffusion with frozen drift.

Matches multi-step rollout displacement variance across any forcing amplitude.
- Drift is 100% frozen (preserving exact stability and SPSD dissipative dynamics).
- Only the diagonal diffusion parameters are optimized via Adam.
- Loss is the log-variance discrepancy over configured multi-step horizons on remainder data.
- Uses TCN rollout cache for high GPU throughput.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
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
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Input Neural SDE checkpoint (e.g., best.pt)")
    parser.add_argument("--output-checkpoint", type=Path, required=True,
                        help="Output path for intrinsic fine-tuned checkpoint")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path containing *_rep{r}_remainder.npy files")
    parser.add_argument("--horizons", type=int, nargs="+", default=[20, 50, 100, 200],
                        help="Rollout horizons to score displacement variance")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of start states per batch")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of gradient steps")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for diffusion parameters")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_ckpt = Path(args.output_checkpoint).resolve()
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint} on {args.device}...")
    model, config = load_model(str(args.checkpoint), device=args.device)
    model.eval()

    # Freeze 100% of drift parameters
    for p in model.drift_model.parameters():
        p.requires_grad = False
    
    # Train ONLY diffusion raw diagonal
    model.diffusion_model.raw_diagonal.requires_grad = True
    trainable_params = [model.diffusion_model.raw_diagonal]

    print("Trainable parameters:", [name for name, p in model.named_parameters() if p.requires_grad])

    # Load training data
    train_reps = list(range(1, 13))
    train_seqs = load_dataset(args.data_dir, train_reps)
    if not train_seqs:
        raise RuntimeError(f"No remainder sequences found in {args.data_dir}")

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
                coord_weights = torch.ones(n_coords, device=args.device)
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
            best_weights = model.diffusion_model.raw_diagonal.detach().clone()

        if it % 10 == 0 or it == 1 or it == args.iterations:
            h200_str = " ".join([f"{r:.2f}" for r in horizon_ratios[max_horizon][:min(5, n_coords)]])
            print(f"Iter {it:3d}/{args.iterations:3d} | Loss: {loss_val:.4f} (best: {best_loss:.4f}) | H={max_horizon} ratios: [{h200_str}]")

    print(f"\nOptimization complete! Best loss: {best_loss:.4f}")
    with torch.no_grad():
        model.diffusion_model.raw_diagonal.copy_(best_weights)

    final_diff = torch.diag(model.diffusion_matrix()).detach().cpu().numpy()
    print("Final tuned diffusion diagonal:", final_diff)

    # Save intrinsic checkpoint with full metadata
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_ckpt["model_state"]["diffusion_model.raw_diagonal"] = best_weights.cpu()
    raw_ckpt["training"]["fine_tuned_diffusion"] = {
        "best_loss": float(best_loss),
        "horizons": list(args.horizons),
        "iterations": int(args.iterations),
        "lr": float(args.lr),
        "final_diagonal": [float(x) for x in final_diff]
    }
    torch.save(raw_ckpt, output_ckpt)
    print(f"Saved fine-tuned checkpoint to: {output_ckpt}")

    # Write tuning history
    tuning_log = output_ckpt.parent / "tuning_history.json"
    tuning_log.write_text(json.dumps(raw_ckpt["training"]["fine_tuned_diffusion"], indent=2) + "\n")
    print(f"Wrote tuning history to: {tuning_log}")


if __name__ == "__main__":
    main()
