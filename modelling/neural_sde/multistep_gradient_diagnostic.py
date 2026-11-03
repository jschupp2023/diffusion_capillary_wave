"""Measure finite-particle convergence of a frozen multistep-loss gradient."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.acdm.conditional_edm.train import resolve_device

from .losses import MonteCarloMultistepNLL, MultistepVelocityEnergyScore
from .multistep_particle_diagnostic import (
    _distribution_summary, _resolve_checkpoint_path, _resolve_settings)
from .train import _capped_multistep_batches, load_model


def _validate_distinct_integers(values: tuple[int, ...], name: str,
                                *, positive: bool = False) -> tuple[int, ...]:
    values = tuple(values)
    if (not values or len(set(values)) != len(values)
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or (positive and value < 1) for value in values)):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a nonempty sequence of distinct {qualifier}integers")
    return values


def _parameter_groups(model: torch.nn.Module) -> tuple[
        tuple[tuple[str, torch.nn.Parameter], ...], dict[str, np.ndarray], dict[str, int]]:
    parameters = tuple((name, parameter) for name, parameter in model.named_parameters()
                       if parameter.requires_grad)
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    sizes = np.asarray([parameter.numel() for _, parameter in parameters], dtype=np.int64)
    group_indices = {
        "all": np.arange(len(parameters), dtype=np.int64),
        "drift": np.asarray([
            index for index, (name, _) in enumerate(parameters)
            if name.startswith("drift_model.")], dtype=np.int64),
        "diffusion": np.asarray([
            index for index, (name, _) in enumerate(parameters)
            if name.startswith("diffusion_model.")], dtype=np.int64),
    }
    optional_prefixes = {
        "drift_stiffness": (
            "drift_model.raw_stiffness",
            "drift_model.stiffness_factor.",
            "drift_model.raw_mean_stiffness",
        ),
        "drift_damping": (
            "drift_model.raw_damping",
            "drift_model.damping_factor.",
        ),
        "drift_mass": ("drift_model.raw_mass_multiplier",),
        "drift_residual": ("drift_model.residual.",),
    }
    for group, prefixes in optional_prefixes.items():
        indices = np.asarray([
            index for index, (name, _) in enumerate(parameters)
            if name.startswith(prefixes)], dtype=np.int64)
        if len(indices):
            group_indices[group] = indices
    residual = set(group_indices.get("drift_residual", ()))
    structured = np.asarray([
        index for index in group_indices["drift"] if index not in residual],
        dtype=np.int64)
    if len(structured):
        group_indices["drift_structured"] = structured
    for name, indices in group_indices.items():
        if not len(indices):
            raise ValueError(f"Model has no parameters in the {name} group")
    parameter_counts = {
        name: int(sizes[indices].sum()) for name, indices in group_indices.items()}
    return parameters, group_indices, parameter_counts


def _gradient_sample(model: torch.nn.Module, objective: MonteCarloMultistepNLL,
                     current: torch.Tensor, history: torch.Tensor,
                     targets: torch.Tensor, parameters, group_indices, *,
                     seed: int, device: torch.device, microbatch_size: int) -> tuple[
                         float, dict[int, float], dict[str, np.ndarray]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    sample_count = len(current)
    mean_loss = 0.
    mean_horizon_losses = {horizon: 0. for horizon in objective.horizons}
    grouped = {
        name: np.zeros(sum(parameters[index][1].numel() for index in indices),
                       dtype=np.float64)
        for name, indices in group_indices.items()
    }
    for start in range(0, sample_count, microbatch_size):
        stop = min(start + microbatch_size, sample_count)
        weight = (stop - start) / sample_count
        horizon_losses = objective.per_horizon(
            model, current[start:stop].to(device), targets[start:stop].to(device),
            history_states=history[start:stop].to(device), generator=generator)
        # This is exactly the aggregation used by train.py before applying the
        # scalar multistep weight and adding the one-step objective.
        loss = torch.stack(tuple(horizon_losses.values())).mean()
        gradients = torch.autograd.grad(
            loss, tuple(parameter for _, parameter in parameters), allow_unused=True)
        pieces = tuple(
            torch.zeros_like(parameter).reshape(-1) if gradient is None
            else gradient.detach().reshape(-1)
            for (_, parameter), gradient in zip(parameters, gradients))
        for name, indices in group_indices.items():
            vector = torch.cat(tuple(pieces[index] for index in indices))
            grouped[name] += weight * vector.double().cpu().numpy()
        mean_loss += weight * float(loss.detach())
        for horizon, value in horizon_losses.items():
            mean_horizon_losses[horizon] += weight * float(value.detach())
    return mean_loss, mean_horizon_losses, grouped


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else np.nan


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    return (float(np.linalg.norm(value - reference) / denominator)
            if denominator > 0 else np.nan)


def _gradient_set_summary(samples: list[np.ndarray], reference: np.ndarray) -> dict:
    matrix = np.stack(samples)
    mean = matrix.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    rms_deviation = float(np.sqrt(np.mean(np.sum((matrix - mean) ** 2, axis=1))))
    pairwise = np.asarray([
        _cosine(matrix[left], matrix[right])
        for left, right in itertools.combinations(range(len(matrix)), 2)
    ], dtype=np.float64)
    return {
        "gradient_norm": _distribution_summary(np.linalg.norm(matrix, axis=1)),
        "cosine_to_reference": _distribution_summary(np.asarray([
            _cosine(value, reference) for value in matrix])),
        "norm_ratio_to_reference": _distribution_summary(np.asarray([
            np.linalg.norm(value) / np.linalg.norm(reference)
            if np.linalg.norm(reference) > 0 else np.nan for value in matrix])),
        "relative_l2_error_to_reference": _distribution_summary(np.asarray([
            _relative_error(value, reference) for value in matrix])),
        "pairwise_seed_cosine": (
            _distribution_summary(pairwise) if pairwise.size else None),
        "mean_gradient_norm": float(mean_norm),
        "mean_gradient_cosine_to_reference": _cosine(mean, reference),
        "mean_gradient_norm_ratio_to_reference": (
            float(mean_norm / np.linalg.norm(reference))
            if np.linalg.norm(reference) > 0 else None),
        "rms_seed_deviation_over_mean_gradient_norm": (
            float(rms_deviation / mean_norm) if mean_norm > 0 else None),
    }


def _resolve_objective_name(training: dict, objective: str | None) -> str:
    """Resolve an explicit objective or inherit it with legacy-NLL fallback."""
    name = training.get("multistep_objective", "gmm_nll") if objective is None else objective
    if name not in {"gmm_nll", "energy_score"}:
        raise ValueError("Objective must be gmm_nll or energy_score")
    return name


def _objective(
        name: str, horizons: tuple[int, ...], counts: tuple[int, ...], *,
        target: str, mode_counts: tuple[int, ...] | None,
        pod_coordinate_offset: int) -> MonteCarloMultistepNLL:
    if name == "energy_score":
        if target != "velocity":
            raise ValueError("Energy-score gradient diagnostics require target velocity")
        return MultistepVelocityEnergyScore(
            horizons, counts[0], mode_counts=mode_counts,
            pod_coordinate_offset=pod_coordinate_offset, particle_counts=counts)
    return MonteCarloMultistepNLL(
        horizons, counts[0], target=target, mode_counts=mode_counts,
        pod_coordinate_offset=pod_coordinate_offset, particle_counts=counts)


def _write_quick_summary(path: Path, report: dict) -> None:
    lines = [
        "# Multistep gradient convergence",
        "",
        f"Objective: **`{report['objective_name']}`**. ",
        "",
        f"Fixed `{report['split']}` minibatch: **{report['window_count']} windows**. "
        f"Gradient microbatch: **{report['gradient_microbatch_size']} windows**. "
        f"Base particle schedule: `{report['base_particle_counts']}`. "
        f"Reference: **{report['reference']['multiplier']}x**, averaged over "
        f"{len(report['reference']['seeds'])} independent seeds.",
        "",
        "| scale | particle schedule | loss mean +/- seed SD | full cosine | "
        "mean-grad cosine | norm ratio | pairwise seed cosine | noise/signal | "
        "diffusion cosine |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for multiplier in report["particle_multipliers"]:
        item = report["by_particle_multiplier"][str(multiplier)]
        all_gradient = item["gradient_groups"]["all"]
        diffusion = item["gradient_groups"]["diffusion"]
        lines.append(
            f"| {multiplier}x | `{item['particle_counts']}` | "
            f"{item['loss']['mean']:.4f} +/- {item['loss']['standard_deviation']:.4f} | "
            f"{all_gradient['cosine_to_reference']['mean']:.3f} | "
            f"{all_gradient['mean_gradient_cosine_to_reference']:.3f} | "
            f"{all_gradient['norm_ratio_to_reference']['mean']:.3f} | "
            f"{all_gradient['pairwise_seed_cosine']['mean']:.3f} | "
            f"{all_gradient['rms_seed_deviation_over_mean_gradient_norm']:.3f} | "
            f"{diffusion['cosine_to_reference']['mean']:.3f} |")
    decomposed_groups = tuple(
        group for group in (
            "drift_structured", "drift_stiffness", "drift_damping",
            "drift_mass", "drift_residual")
        if group in report["reference"]["gradient_groups"])
    if decomposed_groups:
        lines.extend((
            "",
            "## Drift decomposition",
            "",
            "| scale | group | parameters | cosine | mean-grad cosine | norm ratio | "
            "pairwise seed cosine | noise/signal |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|",
        ))
        for multiplier in report["particle_multipliers"]:
            item = report["by_particle_multiplier"][str(multiplier)]
            for group in decomposed_groups:
                gradient = item["gradient_groups"][group]
                lines.append(
                    f"| {multiplier}x | `{group}` | {report['parameter_counts'][group]} | "
                    f"{gradient['cosine_to_reference']['mean']:.3f} | "
                    f"{gradient['mean_gradient_cosine_to_reference']:.3f} | "
                    f"{gradient['norm_ratio_to_reference']['mean']:.3f} | "
                    f"{gradient['pairwise_seed_cosine']['mean']:.3f} | "
                    f"{gradient['rms_seed_deviation_over_mean_gradient_norm']:.3f} |")
    lines.extend((
        "",
        "`full cosine` is the mean per-seed cosine to the independent reference; "
        "`mean-grad cosine` first averages tested seeds. A cosine of 1, norm ratio "
        "of 1, and noise/signal of 0 are ideal. The reference is finite-particle, "
        "not exact. The gradients exclude the scalar multistep weight and one-step "
        "loss; both leave standalone multistep direction unchanged, but the one-step "
        "term changes the actual combined optimizer update.",
        "",
    ))
    path.write_text("\n".join(lines))


def diagnose_gradients(
        checkpoint_path: Path, output: Path, *, split: str = "validation",
        windows: int = 16, microbatch_size: int | None = None, window_seed: int = 0,
        seeds: tuple[int, ...] = (0, 1, 2),
        particle_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        reference_multiplier: int | None = None,
        reference_seeds: tuple[int, ...] = (100, 101, 102),
        device: torch.device | str = "cpu", data_root: Path | None = None,
        shared_basis: Path | None = None,
        objective: str | None = None,
        horizons: tuple[int, ...] | None = None, target: str | None = None,
        mode_counts: tuple[int, ...] | None = None,
        particles: int | None = None,
        particle_counts: tuple[int, ...] | None = None) -> Path:
    """Compare fixed-minibatch gradients against an independent high-M mean."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation, or test")
    if windows < 1:
        raise ValueError("Windows must be positive")
    if microbatch_size is None:
        microbatch_size = windows
    if microbatch_size < 1:
        raise ValueError("Microbatch size must be positive")
    seeds = _validate_distinct_integers(seeds, "Seeds")
    reference_seeds = _validate_distinct_integers(reference_seeds, "Reference seeds")
    if set(seeds) & set(reference_seeds):
        raise ValueError("Seeds and reference seeds must be disjoint")
    particle_multipliers = _validate_distinct_integers(
        particle_multipliers, "Particle multipliers", positive=True)
    if reference_multiplier is None:
        reference_multiplier = max(particle_multipliers)
    if (not isinstance(reference_multiplier, int)
            or isinstance(reference_multiplier, bool) or reference_multiplier < 1):
        raise ValueError("Reference multiplier must be a positive integer")

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    device = torch.device(device)
    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()
    if model.config.state_variable != "velocity":
        raise ValueError("Multistep gradient diagnostics require a velocity checkpoint")
    data = checkpoint_data(checkpoint, data_root=data_root, shared_basis=shared_basis)
    training = checkpoint.get("training", {})
    objective_name = _resolve_objective_name(training, objective)
    horizons, target, mode_counts, base_counts = _resolve_settings(
        checkpoint, data.rank, horizons=horizons, target=target,
        mode_counts=mode_counts, particles=particles, particle_counts=particle_counts)
    target_key = "target_displacements" if target == "displacement" else "target_velocities"
    trim_cutoff = training.get("train_trim_norm_cutoff")
    trim_state_std = training.get("train_trim_state_std")
    batches = list(_capped_multistep_batches(
        data, model, split, windows, horizons, windows, window_seed,
        trim_cutoff, trim_state_std))
    if not batches:
        raise ValueError("No retained multistep windows were available")
    current = torch.cat(tuple(batch["current_state"] for batch in batches))
    history = torch.cat(tuple(batch["history_states"] for batch in batches))
    targets = torch.cat(tuple(batch[target_key] for batch in batches))

    parameters, group_indices, parameter_counts = _parameter_groups(model)
    schedules = {
        multiplier: tuple(multiplier * count for count in base_counts)
        for multiplier in set(particle_multipliers) | {reference_multiplier}
    }

    def collect(multiplier: int, sample_seeds: tuple[int, ...]):
        counts = schedules[multiplier]
        loss_objective = _objective(
            objective_name, horizons, counts, target=target,
            mode_counts=mode_counts,
            pod_coordinate_offset=int(data.include_spatial_mean))
        results = []
        for seed in sample_seeds:
            results.append(_gradient_sample(
                model, loss_objective, current, history, targets, parameters,
                group_indices, seed=seed, device=device,
                microbatch_size=microbatch_size))
        return results

    reference_results = collect(reference_multiplier, reference_seeds)
    reference_gradients = {
        group: np.stack([result[2][group] for result in reference_results]).mean(axis=0)
        for group in group_indices
    }
    results_by_multiplier = {
        multiplier: collect(multiplier, seeds)
        for multiplier in particle_multipliers
    }

    stored = {
        "current_state": current.detach().cpu().numpy(),
        "history_states": history.detach().cpu().numpy(),
        "targets": targets.detach().cpu().numpy(),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "reference_seeds": np.asarray(reference_seeds, dtype=np.int64),
    }
    reference_report = {
        "multiplier": reference_multiplier,
        "particle_counts": list(schedules[reference_multiplier]),
        "seeds": list(reference_seeds),
        "loss": _distribution_summary(np.asarray([
            result[0] for result in reference_results])),
        "per_horizon_loss": {
            str(horizon): _distribution_summary(np.asarray([
                result[1][horizon] for result in reference_results]))
            for horizon in horizons
        },
        "gradient_groups": {},
    }
    for group, reference in reference_gradients.items():
        samples = [result[2][group] for result in reference_results]
        stored[f"reference_mean_{group}"] = reference
        stored[f"reference_samples_{group}"] = np.stack(samples)
        reference_report["gradient_groups"][group] = _gradient_set_summary(
            samples, reference)

    multiplier_report = {}
    for multiplier, results in results_by_multiplier.items():
        item = {
            "particle_counts": list(schedules[multiplier]),
            "loss": _distribution_summary(np.asarray([result[0] for result in results])),
            "per_horizon_loss": {
                str(horizon): _distribution_summary(np.asarray([
                    result[1][horizon] for result in results]))
                for horizon in horizons
            },
            "gradient_groups": {},
        }
        for group, reference in reference_gradients.items():
            samples = [result[2][group] for result in results]
            stored[f"m{multiplier}_samples_{group}"] = np.stack(samples)
            item["gradient_groups"][group] = _gradient_set_summary(samples, reference)
        multiplier_report[str(multiplier)] = item

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "gradients.npz", **stored)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "split": split,
        "window_count": int(len(current)),
        "gradient_microbatch_size": microbatch_size,
        "window_seed": window_seed,
        "particle_seeds": list(seeds),
        "objective_name": objective_name,
        "target": target,
        "horizons": list(horizons),
        "mode_counts": None if mode_counts is None else list(mode_counts),
        "base_particle_counts": list(base_counts),
        "particle_multipliers": list(particle_multipliers),
        "parameter_counts": parameter_counts,
        "saved_training_context": {
            "batch_size": training.get("batch_size"),
            "multistep_objective": training.get("multistep_objective", "gmm_nll"),
            "multistep_weight": training.get("multistep_weight"),
            "multistep_window_fraction": training.get("multistep_window_fraction"),
            "multistep_window_sampling": training.get("multistep_window_sampling"),
            "training_trim_applied": trim_cutoff is not None,
            "normalized_increment_norm_cutoff": trim_cutoff,
        },
        "objective": (
            "mean of per-horizon conditional velocity energy scores, matching train.py "
            "before its scalar multistep weight and addition to one-step NLL"
            if objective_name == "energy_score" else
            "mean of per-horizon multistep NLLs, matching train.py before its scalar "
            "multistep weight and addition to one-step NLL"),
        "frozen_model": True,
        "reference": reference_report,
        "by_particle_multiplier": multiplier_report,
        "interpretation": {
            "cosine_to_reference": "1 means the same update direction; 0 means orthogonal.",
            "norm_ratio_to_reference": "1 means the same Euclidean gradient magnitude.",
            "relative_l2_error_to_reference": "0 means an identical gradient vector.",
            "pairwise_seed_cosine": "Seed-to-seed directional repeatability at fixed M.",
            "rms_seed_deviation_over_mean_gradient_norm": (
                "Monte Carlo gradient noise relative to the mean-gradient signal; lower is better."),
            "caveat": (
                "The reference is an independent finite-particle seed average, not an exact "
                "infinite-particle gradient. Raw norms are parameterization-dependent. "
                "Structured drift groups are selected from model parameter names; the "
                "residual group contains the residual MLP parameters."),
        },
        "array_file": "gradients.npz",
        "quick_summary_file": "quick_summary.md",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_quick_summary(output / "quick_summary.md", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint file or run directory containing best.pt.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument(
        "--microbatch-size", type=int,
        help="Accumulate the exact window-mean gradient in memory-bounded chunks.")
    parser.add_argument("--window-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--particle-multipliers", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--reference-multiplier", type=int)
    parser.add_argument("--reference-seeds", type=int, nargs="+", default=(100, 101, 102))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument(
        "--objective", choices=("gmm_nll", "energy_score"),
        help="Override the checkpoint multistep objective (legacy default gmm_nll).")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--target", choices=("velocity", "displacement"))
    parser.add_argument("--mode-counts", type=int, nargs="+")
    parser.add_argument("--particles", type=int)
    parser.add_argument("--particle-counts", type=int, nargs="+")
    args = parser.parse_args(argv)
    output = diagnose_gradients(
        args.checkpoint, args.output, split=args.split, windows=args.windows,
        microbatch_size=args.microbatch_size,
        window_seed=args.window_seed, seeds=tuple(args.seeds),
        particle_multipliers=tuple(args.particle_multipliers),
        reference_multiplier=args.reference_multiplier,
        reference_seeds=tuple(args.reference_seeds),
        device=resolve_device(args.device), data_root=args.data_root,
        shared_basis=args.shared_basis,
        objective=args.objective,
        horizons=None if args.horizons is None else tuple(args.horizons),
        target=args.target,
        mode_counts=None if args.mode_counts is None else tuple(args.mode_counts),
        particles=args.particles,
        particle_counts=(None if args.particle_counts is None
                         else tuple(args.particle_counts)),
    )
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
