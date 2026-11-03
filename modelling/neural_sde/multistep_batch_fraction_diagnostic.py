"""Diagnose multistep-gradient noise from sparse full-batch sampling."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.acdm.conditional_edm.train import resolve_device

from .multistep_gradient_diagnostic import (
    _gradient_sample, _gradient_set_summary, _objective, _parameter_groups,
    _resolve_objective_name, _validate_distinct_integers)
from .multistep_particle_diagnostic import _resolve_checkpoint_path, _resolve_settings
from .train import _capped_multistep_batches, load_model


def _nested_group_summary(samples: list[list[np.ndarray]]) -> dict:
    """Separate particle noise within a batch from batch-selection noise."""
    batch_means = [np.stack(batch).mean(axis=0) for batch in samples]
    grand_mean = np.stack(batch_means).mean(axis=0)
    signal = float(np.linalg.norm(grand_mean))
    flattened = [value for batch in samples for value in batch]
    within_squared = np.mean([
        np.mean(np.sum((np.stack(batch) - mean) ** 2, axis=1))
        for batch, mean in zip(samples, batch_means)
    ])
    return {
        "single_scored_update": _gradient_set_summary(flattened, grand_mean),
        "particle_averaged_batch": _gradient_set_summary(batch_means, grand_mean),
        "within_batch_particle_noise_over_signal": (
            float(np.sqrt(within_squared) / signal) if signal > 0 else None),
        "empirical_grand_mean_norm": signal,
    }


def _default_fractions(saved_fraction: float | None) -> tuple[float, ...]:
    values = {0.005, 0.01, 0.02, 0.05, 0.1}
    if saved_fraction is not None:
        values.add(float(saved_fraction))
    return tuple(sorted(value for value in values if 0 < value <= 1))


def _fraction_projection(fractions: tuple[float, ...], *, epoch_batch_count: int | None,
                         multistep_weight: float | None,
                         group_summaries: dict[str, dict]) -> list[dict]:
    result = []
    for fraction in fractions:
        selected = (None if epoch_batch_count is None
                    else float(fraction * epoch_batch_count))
        item = {
            "fraction": fraction,
            "inverse_probability_scale": 1. / fraction,
            "expected_scored_batches_per_epoch": selected,
            "selected_update_multistep_coefficient": (
                None if multistep_weight is None else multistep_weight / fraction),
            "projected_noise_over_signal": {},
        }
        if selected is not None and selected > 0:
            for group, summary in group_summaries.items():
                total = summary["single_scored_update"][
                    "rms_seed_deviation_over_mean_gradient_norm"]
                window = summary["particle_averaged_batch"][
                    "rms_seed_deviation_over_mean_gradient_norm"]
                item["projected_noise_over_signal"][group] = {
                    "particle_and_batch": float(total / math.sqrt(selected)),
                    "batch_after_particle_averaging": float(window / math.sqrt(selected)),
                }
        result.append(item)
    return result


def _write_quick_summary(path: Path, report: dict) -> None:
    context = report["saved_training_context"]
    lines = [
        "# Multistep batch-fraction gradient noise",
        "",
        f"Objective: **`{report['objective_name']}`** on frozen `{report['split']}` "
        f"windows. Each sampled batch contains **{report['windows_per_batch']} retained "
        f"windows**. The table uses {len(report['window_seeds'])} independently shuffled "
        f"batches and {len(report['particle_seeds'])} particle draws per batch.",
        "",
        "## Directly measured single-batch variability",
        "",
        "| group | update pairwise cosine | update noise/signal | "
        "batch-only pairwise cosine | batch-only noise/signal | particle noise/signal |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", "drift", "diffusion"):
        summary = report["gradient_groups"][group]
        total = summary["single_scored_update"]
        batch = summary["particle_averaged_batch"]
        lines.append(
            f"| `{group}` | {total['pairwise_seed_cosine']['mean']:.3f} | "
            f"{total['rms_seed_deviation_over_mean_gradient_norm']:.3f} | "
            f"{batch['pairwise_seed_cosine']['mean']:.3f} | "
            f"{batch['rms_seed_deviation_over_mean_gradient_norm']:.3f} | "
            f"{summary['within_batch_particle_noise_over_signal']:.3f} |")
    lines.extend((
        "", "## Connection to batch fraction", "",
        "| fraction | expected scored batches/epoch | loss scale `1/f` | "
        "selected-update coefficient `weight/f` | projected full-gradient noise/signal |",
        "|---:|---:|---:|---:|---:|",
    ))
    for item in report["fraction_projection"]:
        selected = item["expected_scored_batches_per_epoch"]
        coefficient = item["selected_update_multistep_coefficient"]
        projected = item["projected_noise_over_signal"].get("all")
        selected_text = "n/a" if selected is None else f"{selected:.1f}"
        coefficient_text = "n/a" if coefficient is None else f"{coefficient:.3g}"
        projected_text = "n/a" if projected is None else f"{projected['particle_and_batch']:.3f}"
        lines.append(
            f"| {item['fraction']:.4g} | {selected_text} | "
            f"{item['inverse_probability_scale']:.1f} | {coefficient_text} | "
            f"{projected_text} |")
    lines.extend((
        "",
        "The direct rows measure frozen-model gradients. `batch-only` first averages the "
        "particle draws for each selected batch, so its remaining variation is primarily "
        "window selection plus residual finite-particle noise. Fraction projections assume "
        "independent selected batches and scale measured single-update noise by "
        "`1/sqrt(expected batches)`. They describe an epoch-average estimator, not Adam's "
        "trajectory under rare, rescaled updates.",
        "",
        f"Saved training fraction: `{context.get('multistep_window_fraction')}`; saved "
        f"multistep weight: `{context.get('multistep_weight')}`. With full-batch sampling, "
        "training multiplies every selected multistep loss by `1/f`.",
        "",
    ))
    path.write_text("\n".join(lines))


def diagnose_batch_fraction(
        checkpoint_path: Path, output: Path, *, split: str = "train",
        windows_per_batch: int | None = None, microbatch_size: int | None = None,
        window_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        particle_seeds: tuple[int, ...] = (100, 101, 102),
        particle_multiplier: int = 1, fractions: tuple[float, ...] | None = None,
        device: torch.device | str = "cpu", data_root: Path | None = None,
        shared_basis: Path | None = None, objective: str | None = None) -> Path:
    """Measure nested particle and full-batch sampling variability."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation, or test")
    window_seeds = _validate_distinct_integers(window_seeds, "Window seeds")
    particle_seeds = _validate_distinct_integers(particle_seeds, "Particle seeds")
    if len(window_seeds) < 2 or len(particle_seeds) < 2:
        raise ValueError("At least two window seeds and particle seeds are required")
    if (not isinstance(particle_multiplier, int) or isinstance(particle_multiplier, bool)
            or particle_multiplier < 1):
        raise ValueError("Particle multiplier must be a positive integer")

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    device = torch.device(device)
    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()
    if model.config.state_variable != "velocity":
        raise ValueError("Multistep batch diagnostics require a velocity checkpoint")
    data = checkpoint_data(checkpoint, data_root=data_root, shared_basis=shared_basis)
    training = checkpoint.get("training", {})
    objective_name = _resolve_objective_name(training, objective)
    horizons, target, mode_counts, base_counts = _resolve_settings(
        checkpoint, data.rank, horizons=None, target=None, mode_counts=None,
        particles=None, particle_counts=None)
    if windows_per_batch is None:
        windows_per_batch = int(training.get("batch_size") or data.batch_size)
    if windows_per_batch < 1:
        raise ValueError("Windows per batch must be positive")
    if microbatch_size is None:
        microbatch_size = windows_per_batch
    if microbatch_size < 1:
        raise ValueError("Microbatch size must be positive")

    counts = tuple(particle_multiplier * value for value in base_counts)
    loss_objective = _objective(
        objective_name, horizons, counts, target=target, mode_counts=mode_counts,
        pod_coordinate_offset=int(data.include_spatial_mean))
    parameters, group_indices, parameter_counts = _parameter_groups(model)
    trim_cutoff = training.get("train_trim_norm_cutoff")
    trim_state_std = training.get("train_trim_state_std")
    target_key = "target_displacements" if target == "displacement" else "target_velocities"

    nested = {group: [] for group in group_indices}
    losses, retained_counts = [], []
    stored = {
        "window_seeds": np.asarray(window_seeds),
        "particle_seeds": np.asarray(particle_seeds),
    }
    for window_seed in window_seeds:
        batches = list(_capped_multistep_batches(
            data, model, split, windows_per_batch, horizons, windows_per_batch,
            window_seed, trim_cutoff, trim_state_std))
        if not batches:
            raise ValueError(f"No retained multistep windows for seed {window_seed}")
        current = torch.cat(tuple(batch["current_state"] for batch in batches))
        history = torch.cat(tuple(batch["history_states"] for batch in batches))
        targets = torch.cat(tuple(batch[target_key] for batch in batches))
        retained_counts.append(len(current))
        batch_results = [
            _gradient_sample(
                model, loss_objective, current, history, targets, parameters,
                group_indices, seed=particle_seed, device=device,
                microbatch_size=microbatch_size)
            for particle_seed in particle_seeds
        ]
        losses.append([result[0] for result in batch_results])
        for group in group_indices:
            values = [result[2][group] for result in batch_results]
            nested[group].append(values)
            stored[f"window{window_seed}_{group}"] = np.stack(values)

    group_summaries = {
        group: _nested_group_summary(samples) for group, samples in nested.items()}
    saved_fraction = training.get("multistep_window_fraction")
    if fractions is None:
        fractions = _default_fractions(saved_fraction)
    else:
        fractions = tuple(sorted(set(float(value) for value in fractions)))
        if not fractions or any(not math.isfinite(value) or not 0 < value <= 1
                                for value in fractions):
            raise ValueError("Fractions must lie in (0, 1]")
    available = training.get("normalization_retained_transition_count")
    epoch_batch_count = (None if available is None
                         else math.ceil(available / windows_per_batch))
    multistep_weight = training.get("multistep_weight")
    projections = _fraction_projection(
        fractions, epoch_batch_count=epoch_batch_count,
        multistep_weight=multistep_weight, group_summaries=group_summaries)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "gradients.npz", **stored)
    report = {
        "checkpoint": str(checkpoint_path.resolve()), "split": split,
        "objective_name": objective_name, "horizons": list(horizons),
        "particle_counts": list(counts), "windows_per_batch": windows_per_batch,
        "retained_window_counts": retained_counts,
        "gradient_microbatch_size": microbatch_size,
        "window_seeds": list(window_seeds), "particle_seeds": list(particle_seeds),
        "loss": {"mean": float(np.mean(losses)),
                 "standard_deviation": float(np.std(losses))},
        "parameter_counts": parameter_counts, "gradient_groups": group_summaries,
        "saved_training_context": {
            "batch_size": training.get("batch_size"),
            "normalization_retained_transition_count": available,
            "estimated_epoch_batch_count": epoch_batch_count,
            "multistep_window_fraction": saved_fraction,
            "multistep_window_sampling": training.get("multistep_window_sampling"),
            "multistep_weight": multistep_weight,
        },
        "fraction_projection": projections,
        "interpretation": {
            "reference": "Empirical grand mean over diagnostic batch and particle means.",
            "projection": (
                "Epoch-average projection assumes independent batches and 1/sqrt(K) "
                "averaging; sparse optimizer dynamics are not reproduced."),
        },
        "array_file": "gradients.npz", "quick_summary_file": "quick_summary.md",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_quick_summary(output / "quick_summary.md", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--windows-per-batch", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--window-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    parser.add_argument("--particle-seeds", type=int, nargs="+", default=(100, 101, 102))
    parser.add_argument("--particle-multiplier", type=int, default=1)
    parser.add_argument("--fractions", type=float, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--objective", choices=("gmm_nll", "energy_score"))
    args = parser.parse_args(argv)
    output = diagnose_batch_fraction(
        args.checkpoint, args.output, split=args.split,
        windows_per_batch=args.windows_per_batch, microbatch_size=args.microbatch_size,
        window_seeds=tuple(args.window_seeds), particle_seeds=tuple(args.particle_seeds),
        particle_multiplier=args.particle_multiplier,
        fractions=None if args.fractions is None else tuple(args.fractions),
        device=resolve_device(args.device), data_root=args.data_root,
        shared_basis=args.shared_basis, objective=args.objective)
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
