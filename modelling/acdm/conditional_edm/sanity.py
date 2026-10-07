from __future__ import annotations

import argparse
import json
import math
import time

import torch

from .config import EDMConfig
from .data import NormalizationStats
from .model import ConditionalEDM
from .train import resolve_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _linear_case(
    *,
    residual_slope: float,
    intercept: float,
    innovation_std: float,
    conditions: torch.Tensor,
    num_samples: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = EDMConfig(
        reduced_dim=1,
        hidden_dim=32,
        num_blocks=1,
        noise_embedding_dim=8,
        sigma_min=0.002,
        sigma_max=20.0,
        num_sampling_steps=24,
    )
    model = ConditionalEDM(config)
    model.set_normalization(
        NormalizationStats(
            state_mean=torch.zeros(1),
            state_std=torch.full((1,), 1 / math.sqrt(3)),
            target_mean=torch.full((1,), intercept),
            target_std=torch.full(
                (1,), math.sqrt(residual_slope**2 / 3 + innovation_std**2)
            ),
        )
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    data_generator = torch.Generator(device=device).manual_seed(seed + 1_000)
    diffusion_generator = torch.Generator(device=device).manual_seed(seed + 2_000)
    recent_losses: list[float] = []

    _synchronize(device)
    start = time.perf_counter()
    for step in range(1_000):
        current = 2 * torch.rand(128, 1, device=device, generator=data_generator) - 1
        residual = residual_slope * current + intercept
        if innovation_std:
            residual += innovation_std * torch.randn(
                current.shape, device=device, generator=data_generator
            )
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(
            {"current_state": current, "next_state": current + residual},
            generator=diffusion_generator,
        )
        loss.backward()
        optimizer.step()
        if step >= 900:
            recent_losses.append(float(loss.detach()))

    conditions = conditions.to(device)
    samples = model.sample_next(conditions, num_samples=num_samples, seed=seed + 3_000)[..., 0]
    _synchronize(device)
    elapsed_seconds = time.perf_counter() - start
    predicted_mean = samples.mean(dim=1)
    predicted_variance = samples.var(dim=1, correction=0)
    exact_mean = (1 + residual_slope) * conditions[:, 0] + intercept
    exact_variance = torch.full_like(exact_mean, innovation_std**2)
    return {
        "conditions": conditions[:, 0].cpu().tolist(),
        "predicted_mean": predicted_mean.cpu().tolist(),
        "exact_mean": exact_mean.cpu().tolist(),
        "predicted_variance": predicted_variance.cpu().tolist(),
        "exact_variance": exact_variance.cpu().tolist(),
        "max_mean_error": float((predicted_mean - exact_mean).abs().max()),
        "max_variance_error": float((predicted_variance - exact_variance).abs().max()),
        "mean_recent_loss": sum(recent_losses) / len(recent_losses),
        "elapsed_seconds": elapsed_seconds,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m modelling.acdm.conditional_edm sanity",
        description="Train learned deterministic-linear and linear-Gaussian sanity cases",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args(argv)
    if args.num_samples is not None and args.num_samples < 100:
        parser.error("--num-samples must be at least 100")
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    deterministic = _linear_case(
        residual_slope=0.65,
        intercept=0.2,
        innovation_std=0.0,
        conditions=torch.tensor([[-0.75], [0.0], [0.8]]),
        num_samples=2_000 if args.num_samples is None else args.num_samples,
        seed=args.seed,
        device=device,
    )
    gaussian = _linear_case(
        residual_slope=-0.55,
        intercept=0.15,
        innovation_std=0.4,
        conditions=torch.zeros(1, 1),
        num_samples=4_000 if args.num_samples is None else args.num_samples,
        seed=args.seed,
        device=device,
    )
    passed = (
        deterministic["max_mean_error"] < 0.04
        and deterministic["max_variance_error"] < 0.02
        and gaussian["max_mean_error"] < 0.04
        and gaussian["max_variance_error"] < 0.04
    )
    report = {"deterministic_linear": deterministic, "linear_gaussian": gaussian, "passed": passed}
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("learned sanity tolerances were not met")


if __name__ == "__main__":
    main()
