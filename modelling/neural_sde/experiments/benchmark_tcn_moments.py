"""Bounded cached/reference BPTT benchmark for the existing moment losses.

Measured validation histories/endpoints, no parameter updates, no checkpoint
writes. Windows are untrimmed: this measures execution, not scientific scores.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.neural_sde.losses import StationaryMeanLoss
from modelling.neural_sde.train import load_model, _sde_transition_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[20, 100, 500])
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile-cache", action="store_true")
    args = parser.parse_args()
    if (args.output.exists() or not 2 <= args.trajectories <= 128
            or not 1 <= args.repeats <= 10 or min(args.horizons) < 1
            or max(args.horizons) > 1000):
        parser.error("Need new output, 2..128 starts, 1..10 repeats, horizons 1..1000")
    model, checkpoint = load_model(args.checkpoint, "cuda")
    # Match arithmetic precision for the forward/gradient comparison. Legacy
    # cuDNN permits TF32 by default but the cached Linear path uses IEEE FP32.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model.train()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    data = checkpoint_data(checkpoint)
    rows = []
    for horizon in args.horizons:
        batch = next(_sde_transition_batches(data, model.config, "validation",
                                             args.trajectories, multistep_horizons=(horizon,)))
        current = batch["current_state"].cuda()
        history = batch["history_states"].cuda()
        endpoint = (batch["current_state"] + batch["target_displacements"][:, 0]).cuda()
        objective = StationaryMeanLoss(horizon)
        comparisons = []
        for cached in (False, True):
            model.tcn_rollout_cache = cached
            model.tcn_rollout_compile = cached and args.compile_cache
            times = []
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(args.repeats + 1):
                model.zero_grad(set_to_none=True)
                rng = torch.Generator(device="cuda").manual_seed(123)
                torch.cuda.synchronize()
                start = time.perf_counter()
                mean, _, variance, _ = objective.statistics(
                    model, current, endpoint, history_states=history, generator=rng)
                loss = mean + .1 * variance
                loss.backward()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if iteration:
                    times.append(elapsed)
            gradient = torch.cat([p.grad.flatten() for p in model.parameters()
                                  if p.grad is not None]).detach().cpu()
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite benchmark at horizon {horizon}, cached={cached}")
            comparisons.append((loss.item(), gradient))
            row = dict(horizon=horizon, starts=len(current), branches=2, cached=cached,
                       compiled=model.tcn_rollout_compile,
                       forward_backward_median_ms=statistics.median(times),
                       peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
                       loss=loss.item())
            rows.append(row)
            print(json.dumps(row), flush=True)
        ref, fast = comparisons
        error = dict(horizon=horizon, loss_relative_error=abs(ref[0]-fast[0])/max(abs(ref[0]), 1e-12),
                     gradient_relative_l2=(fast[1]-ref[1]).norm().item()/max(ref[1].norm().item(), 1e-12))
        rows.append(error)
        print(json.dumps(error), flush=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value.detach().cpu(), initial[name]), name
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint), device=torch.cuda.get_device_name(),
        dtype="float32", cudnn_tf32=False, matmul_tf32=False,
        rows=rows, repetitions=args.repeats,
        caveats=["Untrimmed measured windows; compute benchmark, not validation scores.",
                 "Identical seeded noise; no optimizer steps or parameter writes.",
                 "Includes moment forward/backward only, not clean NLL or data loading."]), indent=2) + "\n")


if __name__ == "__main__":
    main()
