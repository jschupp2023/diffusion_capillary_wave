"""Measure the relative gradients of Neural-SDE training objectives.

The diagnostic freezes a checkpoint and evaluates one-step NLL, the configured
multistep objective, and endpoint mean/variance objectives on fixed retained
validation samples.  It reports raw, epoch-expected weighted, and selected-
update gradient norms, along with objective cosines.  The distinction matters
for sparsely sampled objectives: inverse-probability scaling changes individual
Adam updates, while cancelling in the expected epoch gradient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.acdm.conditional_edm.train import resolve_device

from .losses import EulerMaruyamaNLL, MultistepVelocityEnergyScore, StationaryMeanLoss
from .multistep_gradient_diagnostic import _cosine, _parameter_groups
from .multistep_particle_diagnostic import _distribution_summary, _resolve_checkpoint_path
from .train import (_capped_multistep_batches,
                    _globally_mixed_retained_velocity_batches, load_model)


def _window_identifiers(batch: dict, data, split: str) -> list[dict]:
    """Return stable source/start records for auditing a sampled population."""
    sources = tuple(data.splits[split])
    source_indices = batch["window_source_index"].detach().cpu().long().tolist()
    starts = batch["window_start_index"].detach().cpu().long().tolist()
    return [{"source": sources[source], "start_index": start}
            for source, start in zip(source_indices, starts, strict=True)]


def _identifier_digest(records: list[dict]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gradient_vector(loss: torch.Tensor, parameters, *, retain_graph: bool = False) -> np.ndarray:
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError("Nonfinite objective loss in frozen diagnostic")
    gradients = torch.autograd.grad(
        loss, tuple(parameter for _, parameter in parameters), allow_unused=True,
        retain_graph=retain_graph)
    vector = np.concatenate([
        (torch.zeros_like(parameter) if gradient is None else gradient)
        .detach().reshape(-1).double().cpu().numpy()
        for (_, parameter), gradient in zip(parameters, gradients, strict=True)
    ])
    if not np.isfinite(vector).all():
        raise FloatingPointError("Nonfinite objective gradient in frozen diagnostic")
    return vector


def _group_slices(parameters, group_indices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    offsets = np.cumsum([0] + [parameter.numel() for _, parameter in parameters])
    return {
        group: np.concatenate([
            np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
            for index in indices
        ])
        for group, indices in group_indices.items()
    }


def _route(vector: np.ndarray, indices: np.ndarray, size: int) -> np.ndarray:
    routed = np.zeros(size, dtype=np.float64)
    routed[indices] = vector[indices]
    return routed


def _objective_summary(losses: list[float], vectors: list[np.ndarray], *,
                       weight: float, fraction: float,
                       route_indices: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    matrix = np.stack(vectors)
    routed = np.stack([_route(value, route_indices, matrix.shape[1]) for value in matrix])
    mean = routed.mean(axis=0)
    rms_noise = float(np.sqrt(np.mean(np.sum((routed - mean) ** 2, axis=1))))
    selected_scale = weight / fraction
    report = {
        "loss": _distribution_summary(np.asarray(losses)),
        "weight": weight,
        "sampling_fraction": fraction,
        "selected_update_coefficient": selected_scale,
        "routed_parameter_count": int(len(route_indices)),
        "groups": {},
    }
    for group, indices in groups.items():
        raw_values = matrix[:, indices]
        routed_values = routed[:, indices]
        group_mean = routed_values.mean(axis=0)
        group_noise = float(np.sqrt(np.mean(np.sum(
            (routed_values - group_mean) ** 2, axis=1))))
        report["groups"][group] = {
            "raw_gradient_norm": _distribution_summary(
                np.linalg.norm(raw_values, axis=1)),
            "routed_mean_gradient_norm": float(np.linalg.norm(group_mean)),
            "expected_weighted_gradient_norm": float(weight * np.linalg.norm(group_mean)),
            "selected_update_weighted_gradient_norm": float(
                selected_scale * np.linalg.norm(group_mean)),
            "gradient_noise_over_mean": (
                float(group_noise / np.linalg.norm(group_mean))
                if np.linalg.norm(group_mean) > 0 else None),
        }
    report["mean_routed_gradient"] = mean
    report["gradient_noise_over_mean"] = (
        float(rms_noise / np.linalg.norm(mean)) if np.linalg.norm(mean) > 0 else None)
    return report


def _write_summary(path: Path, report: dict) -> None:
    objectives = report["objectives"]
    lines = [
        "# Objective gradient balance", "",
        f"Frozen-checkpoint gradients on fixed retained {report['split']} samples. Expected weighted "
        "norms use the configured scalar coefficient; selected-update norms also include the "
        "inverse sampling fraction used on batches that are actually scored.", "",
        "| objective | weight | fraction | selected coefficient | drift expected | "
        "diffusion expected | drift selected | diffusion selected | noise / mean |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("one_step_nll", "multistep", "stationary_mean", "stationary_variance"):
        item = objectives[name]
        drift = item["groups"]["drift"]
        diffusion = item["groups"]["diffusion"]
        noise = item["gradient_noise_over_mean"]
        lines.append(
            f"| `{name}` | {item['weight']:.4g} | {item['sampling_fraction']:.4g} | "
            f"{item['selected_update_coefficient']:.4g} | "
            f"{drift['expected_weighted_gradient_norm']:.4g} | "
            f"{diffusion['expected_weighted_gradient_norm']:.4g} | "
            f"{drift['selected_update_weighted_gradient_norm']:.4g} | "
            f"{diffusion['selected_update_weighted_gradient_norm']:.4g} | "
            f"{('n/a' if noise is None else f'{noise:.3g}')} |")
    lines.extend(("", "## Expected-gradient cosine", "",
                  "| objective A | objective B | all | drift | diffusion |",
                  "|:---|:---|---:|---:|---:|"))
    for row in report["expected_gradient_cosines"]:
        def value(group):
            result = row[group]
            return "n/a" if result is None else f"{result:.3f}"
        lines.append(
            f"| `{row['left']}` | `{row['right']}` | {value('all')} | "
            f"{value('drift')} | {value('diffusion')} |")
    lines.extend((
        "", "Cosines below zero indicate local conflict at this checkpoint. Norms compare "
        "the immediate Euclidean gradient in parameter coordinates; they do not predict "
        "Adam step sizes or rollout quality by themselves.", ""))
    lines.extend(("## Unweighted routed-gradient cosine", "",
                  "Directions remain available when an objective weight is zero.", "",
                  "| A | B | drift | diffusion |", "|:---|:---|---:|---:|"))
    for row in report.get("unweighted_routed_gradient_cosines", []):
        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"
        lines.append(f"| {row['left']} | {row['right']} | {fmt(row['drift'])} | {fmt(row['diffusion'])} |")
    lines.extend(("", report.get("population_note", ""), report.get("noise_note", "")))
    horizon_reports = report.get("multistep_horizon_gradients", {})
    if horizon_reports:
        lines.extend(("", "## Per-horizon multistep gradients", "",
                      "Each row is the configured horizon contribution, including its normalized "
                      "horizon weight and the overall multistep coefficient.", "",
                      "| horizon | horizon weight | effective weight | drift expected | "
                      "diffusion expected | drift selected | noise / mean |",
                      "|---:|---:|---:|---:|---:|---:|---:|"))
        for horizon in sorted(horizon_reports, key=int):
            item = horizon_reports[horizon]
            drift = item["groups"]["drift"]
            diffusion = item["groups"]["diffusion"]
            noise = item["gradient_noise_over_mean"]
            lines.append(
                f"| {horizon} | {item['horizon_weight']:.4g} | "
                f"{item['effective_weight']:.4g} | "
                f"{drift['expected_weighted_gradient_norm']:.4g} | "
                f"{diffusion['expected_weighted_gradient_norm']:.4g} | "
                f"{drift['selected_update_weighted_gradient_norm']:.4g} | "
                f"{('n/a' if noise is None else f'{noise:.3g}')} |")
        lines.extend(("", "## Per-horizon routed-gradient cosine", "",
                      "| A | B | drift | diffusion |",
                      "|:---|:---|---:|---:|"))
        for row in report.get("multistep_horizon_gradient_cosines", []):
            def fmt_horizon(value):
                return "n/a" if value is None else f"{value:.3f}"
            lines.append(
                f"| {row['left']} | {row['right']} | "
                f"{fmt_horizon(row['drift'])} | {fmt_horizon(row['diffusion'])} |")
    path.write_text("\n".join(lines))


def _diagnose_once(checkpoint_path: Path, output: Path, *, split: str = "validation",
             windows: int = 64, moment_trajectories: int = 32,
             seeds: tuple[int, ...] = (0, 1), window_seed: int = 0,
             moment_horizons: tuple[int, ...] | None = None,
             device: torch.device | str = "cpu", min_history: int = 0,
             mean_weight: float | None = None, variance_weight: float | None = None,
             variance_pod_mode_count: int | None = None,
             moment_fraction: float | None = None,
             multistep_horizons: tuple[int, ...] | None = None,
             multistep_horizon_weights: tuple[float, ...] | None = None,
             multistep_particle_counts: tuple[int, ...] | None = None,
             multistep_objective: str | None = None,
             multistep_target: str | None = None,
             multistep_position_weight: float | None = None,
             multistep_velocity_weight: float | None = None,
             multistep_weight: float | None = None,
             multistep_fraction: float | None = None,
             multistep_gradient_groups: tuple[str, ...] | None = None) -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if windows < 2 or moment_trajectories < 2:
        raise ValueError("windows and moment trajectories must be at least two")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty distinct sequence")
    if any(value is not None and (not np.isfinite(value) or value < 0)
           for value in (mean_weight, variance_weight)):
        raise ValueError("Moment weights must be nonnegative and finite")
    if (variance_pod_mode_count is not None
            and (not isinstance(variance_pod_mode_count, int)
                 or isinstance(variance_pod_mode_count, bool)
                 or variance_pod_mode_count < 0)):
        raise ValueError("variance POD mode count must be nonnegative")
    if moment_fraction is not None and not 0 < moment_fraction <= 1:
        raise ValueError("Moment fraction must be in (0, 1]")
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    device = torch.device(device)
    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()
    if not isinstance(min_history, int) or isinstance(min_history, bool) or min_history < 0:
        raise ValueError("min_history must be a nonnegative integer")
    model_history_steps = model.config.history_steps
    eligibility_history_steps = max(model_history_steps, min_history)
    if model.config.state_variable != "velocity":
        raise ValueError("objective balance diagnostic requires a velocity model")
    data = checkpoint_data(checkpoint)
    if (variance_pod_mode_count is not None
            and (variance_pod_mode_count > data.rank
                 or (variance_pod_mode_count == 0
                     and not data.include_spatial_mean))):
        raise ValueError("variance POD mode selection is incompatible with the data")
    training = dict(checkpoint.get("training", {}))
    for key, value in (("stationary_mean_weight", mean_weight),
                       ("stationary_variance_weight", variance_weight),
                       ("stationary_mean_batch_fraction", moment_fraction)):
        if value is not None:
            training[key] = value
    if variance_pod_mode_count is not None:
        training["stationary_variance_pod_mode_count"] = variance_pod_mode_count
    diagnostic_overrides = {
        "multistep_horizons": multistep_horizons,
        "multistep_horizon_weights": multistep_horizon_weights,
        "multistep_particle_counts": multistep_particle_counts,
        "multistep_objective": multistep_objective,
        "multistep_target": multistep_target,
        "multistep_position_weight": multistep_position_weight,
        "multistep_velocity_weight": multistep_velocity_weight,
        "multistep_weight": multistep_weight,
        "multistep_window_fraction": multistep_fraction,
        "multistep_gradient_groups": multistep_gradient_groups,
    }
    for key, value in diagnostic_overrides.items():
        if value is not None:
            training[key] = list(value) if isinstance(value, tuple) else value
    if multistep_weight is not None and (not np.isfinite(multistep_weight)
                                         or multistep_weight < 0):
        raise ValueError("multistep weight must be nonnegative and finite")
    if multistep_fraction is not None and not 0 < multistep_fraction <= 1:
        raise ValueError("multistep fraction must be in (0, 1]")
    horizons = tuple(training.get("multistep_horizons") or ())
    if not horizons:
        raise ValueError("checkpoint has no configured multistep horizons")
    target = training.get("multistep_target", "velocity")
    if training.get("multistep_objective", "gmm_nll") != "energy_score":
        raise ValueError("objective balance diagnostic currently requires energy_score")
    counts = tuple(training.get("multistep_particle_counts") or
                   [training.get("multistep_particles", 8)] * len(horizons))
    configured_horizon_weights = tuple(training.get("multistep_horizon_weights") or
                                       [1.] * len(horizons))
    if len(counts) != len(horizons) or len(configured_horizon_weights) != len(horizons):
        raise ValueError("particle counts and horizon weights need one value per horizon")
    trim_cutoff = training.get("train_trim_norm_cutoff")
    trim_state_std = training.get("train_trim_state_std")
    batches = list(_capped_multistep_batches(
        data, model, split, windows, horizons, windows, window_seed,
        trim_cutoff, trim_state_std,
        eligibility_history_steps=eligibility_history_steps,
        include_window_ids=True))
    if not batches:
        raise ValueError("no retained multistep windows")
    batch = {key: torch.cat([part[key] for part in batches]) for key in batches[0]}
    selected_windows = _window_identifiers(batch, data, split)

    if moment_horizons is None:
        moment_horizons = tuple(training.get("stationary_mean_horizons") or (100, 200, 500))
    moment_batches = _globally_mixed_retained_velocity_batches(
        data, model, split, min(256, data.batch_size), moment_trajectories,
        window_seed, trim_cutoff, trim_state_std, moment_horizons,
        eligibility_history_steps=eligibility_history_steps,
        include_window_ids=True)
    moment_batch = next(moment_batches, None)
    if moment_batch is None:
        raise ValueError("no retained stationary-moment ensemble")
    selected_moment_windows = _window_identifiers(moment_batch, data, split)

    parameters, group_indices, parameter_counts = _parameter_groups(model)
    groups = _group_slices(parameters, group_indices)
    full_size = sum(parameter.numel() for _, parameter in parameters)
    all_indices = np.arange(full_size, dtype=np.int64)
    drift_indices = groups["drift"]
    diffusion_indices = groups["diffusion"]
    drift_diffusion_indices = np.sort(np.concatenate((drift_indices, diffusion_indices)))
    multistep_groups = training.get("multistep_gradient_groups", ["drift"])
    multistep_route = np.sort(np.concatenate([
        drift_indices if group == "drift" else diffusion_indices
        for group in multistep_groups]))

    one_step = EulerMaruyamaNLL()
    energy_score = MultistepVelocityEnergyScore(
        horizons, counts[0], particle_counts=counts, target=target,
        pod_coordinate_offset=int(data.include_spatial_mean),
        position_weight=float(training.get("multistep_position_weight", 1.)),
        velocity_weight=float(training.get("multistep_velocity_weight", 1.)),
        horizon_weights=configured_horizon_weights)
    moments = StationaryMeanLoss(
        moment_horizons, pod_coordinate_offset=int(data.include_spatial_mean),
        variance_pod_mode_count=training.get(
            "stationary_variance_pod_mode_count")).to(device)

    current = batch["current_state"].to(device)
    history = batch["history_states"].to(device)
    following = batch["next_state"].to(device)
    following_history = batch["next_history_states"].to(device)
    if target == "joint":
        targets = torch.cat((batch["target_displacements"],
                             batch["target_velocities"]), dim=-1).to(device)
    elif target == "velocity":
        targets = batch["target_velocities"].to(device)
    else:
        raise ValueError("energy-score target must be velocity or joint")
    moment_current = moment_batch["current_state"].to(device)
    moment_history = moment_batch["history_states"].to(device)
    moment_endpoints = moment_batch["data_endpoints"].to(device)

    samples: dict[str, tuple[list[float], list[np.ndarray]]] = {
        name: ([], []) for name in (
            "one_step_nll", "multistep", "stationary_mean", "stationary_variance")}
    horizon_samples = {horizon: ([], []) for horizon in horizons}
    # The one-step objective is deterministic; retaining one sample avoids
    # falsely reporting zero stochastic noise as repeated evidence.
    loss = one_step(model, current, following, history_states=history,
                    following_history_states=following_history)
    samples["one_step_nll"][0].append(float(loss.detach()))
    samples["one_step_nll"][1].append(_gradient_vector(loss, parameters))
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        horizon_losses = energy_score.per_horizon(
            model, current, targets, history_states=history, generator=generator)
        for horizon in horizons:
            horizon_loss = horizon_losses[horizon]
            horizon_samples[horizon][0].append(float(horizon_loss.detach()))
            horizon_samples[horizon][1].append(_gradient_vector(
                horizon_loss, parameters, retain_graph=True))
        loss = energy_score.aggregate(horizon_losses)
        samples["multistep"][0].append(float(loss.detach()))
        samples["multistep"][1].append(_gradient_vector(loss, parameters))

        generator = torch.Generator(device=device).manual_seed(seed + 10_000)
        statistics = moments.per_horizon_statistics(
            model, moment_current, moment_endpoints,
            history_states=moment_history, generator=generator)
        mean_loss = torch.stack([value[0] for value in statistics.values()]).mean()
        variance_loss = torch.stack([value[2] for value in statistics.values()]).mean()
        samples["stationary_mean"][0].append(float(mean_loss.detach()))
        samples["stationary_mean"][1].append(_gradient_vector(
            mean_loss, parameters))
        # Recompute because autograd.grad above releases the shared rollout graph.
        generator = torch.Generator(device=device).manual_seed(seed + 10_000)
        statistics = moments.per_horizon_statistics(
            model, moment_current, moment_endpoints,
            history_states=moment_history, generator=generator)
        variance_loss = torch.stack([value[2] for value in statistics.values()]).mean()
        samples["stationary_variance"][0].append(float(variance_loss.detach()))
        samples["stationary_variance"][1].append(_gradient_vector(
            variance_loss, parameters))

    specifications = {
        "one_step_nll": (1., 1., all_indices),
        "multistep": (float(training.get("multistep_weight", 0.)),
                      float(training.get("multistep_window_fraction", 1.)),
                      multistep_route),
        "stationary_mean": (float(training.get("stationary_mean_weight", 0.)),
                            float(training.get("stationary_mean_batch_fraction", 1.)),
                            drift_indices),
        "stationary_variance": (float(training.get("stationary_variance_weight", 0.)),
                                float(training.get("stationary_mean_batch_fraction", 1.)),
                                drift_diffusion_indices),
    }
    objective_reports = {}
    expected_vectors = {}
    raw_vectors = {}
    for name, (weight, fraction, route_indices) in specifications.items():
        losses, vectors = samples[name]
        item = _objective_summary(
            losses, vectors, weight=weight, fraction=fraction,
            route_indices=route_indices, groups=groups)
        raw_vectors[name] = item.pop("mean_routed_gradient")
        expected_vectors[name] = weight * raw_vectors[name]
        objective_reports[name] = item

    names = tuple(objective_reports)
    cosines = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            row = {"left": left, "right": right}
            for group, indices in groups.items():
                value = _cosine(expected_vectors[left][indices],
                                expected_vectors[right][indices])
                row[group] = None if not np.isfinite(value) else value
            cosines.append(row)
    raw_cosines = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            row = {"left": left, "right": right}
            for group, indices in groups.items():
                value = _cosine(raw_vectors[left][indices], raw_vectors[right][indices])
                row[group] = None if not np.isfinite(value) else value
            raw_cosines.append(row)
    horizon_reports = {}
    horizon_vectors = {"one_step_nll": raw_vectors["one_step_nll"]}
    multistep_weight_value = specifications["multistep"][0]
    multistep_fraction_value = specifications["multistep"][1]
    for horizon, horizon_weight in zip(
            horizons, energy_score.horizon_weights, strict=True):
        losses, vectors = horizon_samples[horizon]
        effective_weight = multistep_weight_value * horizon_weight
        item = _objective_summary(
            losses, vectors, weight=effective_weight,
            fraction=multistep_fraction_value,
            route_indices=multistep_route, groups=groups)
        horizon_vectors[f"h{horizon}"] = item.pop("mean_routed_gradient")
        item.update(horizon_weight=horizon_weight,
                    effective_weight=effective_weight)
        horizon_reports[str(horizon)] = item
    horizon_cosines = []
    horizon_names = tuple(horizon_vectors)
    for left_index, left in enumerate(horizon_names):
        for right in horizon_names[left_index + 1:]:
            row = {"left": left, "right": right}
            for group, indices in groups.items():
                value = _cosine(horizon_vectors[left][indices],
                                horizon_vectors[right][indices])
                row[group] = None if not np.isfinite(value) else value
            horizon_cosines.append(row)
    combined = sum(expected_vectors.values())
    report = {
        "checkpoint": str(checkpoint_path.resolve()), "split": split,
        "requested_eligibility_history_steps": min_history,
        "eligibility_history_steps": eligibility_history_steps,
        "model_conditioning_history_steps": model_history_steps,
        "selected_window_identifiers": selected_windows,
        "selected_window_identifiers_sha256": _identifier_digest(selected_windows),
        "selected_moment_window_identifiers": selected_moment_windows,
        "selected_moment_window_identifiers_sha256": _identifier_digest(
            selected_moment_windows),
        "population_note": "NLL here uses ES-eligible windows, not all one-step training windows; moment windows require their full longer stencil.",
        "window_count": int(len(current)),
        "moment_trajectory_count": int(len(moment_current)),
        "multistep_horizons": list(horizons),
        "moment_horizons": list(moment_horizons), "seeds": list(seeds),
        "stationary_variance_pod_mode_count": training.get(
            "stationary_variance_pod_mode_count"),
        "parameter_counts": parameter_counts,
        "multistep_gradient_groups": list(multistep_groups),
        "objectives": objective_reports,
        "expected_gradient_cosines": cosines,
        "unweighted_routed_gradient_cosines": raw_cosines,
        "multistep_horizon_gradients": horizon_reports,
        "multistep_horizon_gradient_cosines": horizon_cosines,
        "multistep_diagnostic_overrides": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in diagnostic_overrides.items() if value is not None},
        "combined_expected_gradient_norms": {
            group: float(np.linalg.norm(combined[indices]))
            for group, indices in groups.items()},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stored = {f"{name}_samples": np.stack(vectors)
              for name, (_, vectors) in samples.items()}
    stored.update({f"multistep_h{horizon}_samples": np.stack(vectors)
                   for horizon, (_, vectors) in horizon_samples.items()})
    stored.update({f"group_{name}": indices for name, indices in groups.items()})
    stored.update({f"route_{name}": route for name, (_, _, route) in specifications.items()})
    np.savez_compressed(output / "gradient_samples.npz", **stored)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_summary(output / "quick_summary.md", report)
    return output


def diagnose(checkpoint_path: Path, output: Path, *, window_seeds=None, **kwargs) -> Path:
    """Repeat on independently selected batches; retain per-batch evidence."""
    if window_seeds is None:
        return _diagnose_once(checkpoint_path, output, **kwargs)
    window_seeds = tuple(window_seeds)
    if not window_seeds or len(set(window_seeds)) != len(window_seeds):
        raise ValueError("window seeds must be nonempty and distinct")
    reports, arrays = [], []
    output = Path(output)
    for seed in window_seeds:
        local = dict(kwargs, window_seed=seed)
        part = _diagnose_once(checkpoint_path, output / f"batch_{seed}", **local)
        reports.append(json.loads((part / "summary.json").read_text()))
        with np.load(part / "gradient_samples.npz") as values:
            arrays.append({key: values[key] for key in values.files})
    report = dict(reports[0])
    report["window_seeds"] = list(window_seeds)
    report["batch_reports"] = [str(output / f"batch_{seed}" / "summary.json")
                                for seed in window_seeds]
    report["selected_window_identifiers_by_batch"] = {
        str(seed): {
            "identifiers": current["selected_window_identifiers"],
            "sha256": current["selected_window_identifiers_sha256"],
            "moment_identifiers": current["selected_moment_window_identifiers"],
            "moment_sha256": current["selected_moment_window_identifiers_sha256"],
        }
        for seed, current in zip(window_seeds, reports, strict=True)
    }
    report["noise_note"] = "Across batches and particle seeds; one-step variation is batch selection only. Finite sample estimates, not Adam update predictions."
    groups = {key[6:]: value for key, value in arrays[0].items() if key.startswith("group_")}
    raw, weighted = {}, {}
    for name, settings in report["objectives"].items():
        vectors = np.concatenate([a[f"{name}_samples"] for a in arrays])
        # Recover per-sample losses from the stored diagnostic distributions' mean;
        # gradient statistics below use every actual vector, not mean norms.
        losses = [r["objectives"][name]["loss"]["mean"] for r in reports]
        item = _objective_summary(losses, list(vectors), weight=settings["weight"],
                                  fraction=settings["sampling_fraction"],
                                  route_indices=arrays[0][f"route_{name}"], groups=groups)
        item["loss_summary_unit"] = "batch-mean loss (particle-averaged)"
        raw[name] = item.pop("mean_routed_gradient")
        weighted[name] = settings["weight"] * raw[name]
        report["objectives"][name] = item
    horizon_raw = {"one_step_nll": raw["one_step_nll"]}
    for horizon, settings in report.get("multistep_horizon_gradients", {}).items():
        vectors = np.concatenate([
            a[f"multistep_h{horizon}_samples"] for a in arrays])
        losses = [r["multistep_horizon_gradients"][horizon]["loss"]["mean"]
                  for r in reports]
        item = _objective_summary(
            losses, list(vectors), weight=settings["effective_weight"],
            fraction=settings["sampling_fraction"],
            route_indices=arrays[0]["route_multistep"], groups=groups)
        item["loss_summary_unit"] = "batch-mean loss (particle-averaged)"
        horizon_raw[f"h{horizon}"] = item.pop("mean_routed_gradient")
        item.update(horizon_weight=settings["horizon_weight"],
                    effective_weight=settings["effective_weight"])
        report["multistep_horizon_gradients"][horizon] = item
    names = tuple(raw)
    for key, vectors in (("expected_gradient_cosines", weighted),
                         ("unweighted_routed_gradient_cosines", raw)):
        rows = []
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                row = {"left": left, "right": right}
                for group, indices in groups.items():
                    value = _cosine(vectors[left][indices], vectors[right][indices])
                    row[group] = float(value) if np.isfinite(value) else None
                rows.append(row)
        report[key] = rows
    combined = sum(weighted.values())
    report["combined_expected_gradient_norms"] = {
        name: float(np.linalg.norm(combined[indices])) for name, indices in groups.items()}
    rows = []
    horizon_names = tuple(horizon_raw)
    for i, left in enumerate(horizon_names):
        for right in horizon_names[i + 1:]:
            row = {"left": left, "right": right}
            for group, indices in groups.items():
                value = _cosine(horizon_raw[left][indices], horizon_raw[right][indices])
                row[group] = float(value) if np.isfinite(value) else None
            rows.append(row)
    report["multistep_horizon_gradient_cosines"] = rows
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    _write_summary(output / "quick_summary.md", report)
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--moment-trajectories", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--window-seed", type=int, default=0)
    parser.add_argument("--moment-horizons", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-seeds", type=int, nargs="+")
    parser.add_argument("--min-history", type=int, default=0)
    parser.add_argument("--mean-weight", type=float)
    parser.add_argument("--variance-weight", type=float)
    parser.add_argument("--variance-pod-mode-count", type=int)
    parser.add_argument("--moment-fraction", type=float)
    parser.add_argument("--multistep-horizons", type=int, nargs="+")
    parser.add_argument("--multistep-horizon-weights", type=float, nargs="+")
    parser.add_argument("--multistep-particle-counts", type=int, nargs="+")
    parser.add_argument("--multistep-objective", choices=("energy_score",))
    parser.add_argument("--multistep-target", choices=("velocity", "joint"))
    parser.add_argument("--multistep-position-weight", type=float)
    parser.add_argument("--multistep-velocity-weight", type=float)
    parser.add_argument("--multistep-weight", type=float)
    parser.add_argument("--multistep-fraction", type=float)
    parser.add_argument("--multistep-gradient-groups", nargs="+",
                        choices=("drift", "diffusion"))
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    path = diagnose(
        args.checkpoint, args.output, split=args.split, windows=args.windows,
        moment_trajectories=args.moment_trajectories, seeds=tuple(args.seeds),
        window_seed=args.window_seed, window_seeds=args.window_seeds,
        min_history=args.min_history, mean_weight=args.mean_weight,
        variance_weight=args.variance_weight,
        variance_pod_mode_count=args.variance_pod_mode_count,
        moment_fraction=args.moment_fraction,
        multistep_horizons=(None if args.multistep_horizons is None
                            else tuple(args.multistep_horizons)),
        multistep_horizon_weights=(None if args.multistep_horizon_weights is None
                                  else tuple(args.multistep_horizon_weights)),
        multistep_particle_counts=(None if args.multistep_particle_counts is None
                                  else tuple(args.multistep_particle_counts)),
        multistep_objective=args.multistep_objective,
        multistep_target=args.multistep_target,
        multistep_position_weight=args.multistep_position_weight,
        multistep_velocity_weight=args.multistep_velocity_weight,
        multistep_weight=args.multistep_weight,
        multistep_fraction=args.multistep_fraction,
        multistep_gradient_groups=(None if args.multistep_gradient_groups is None
                                   else tuple(args.multistep_gradient_groups)),
        moment_horizons=(None if args.moment_horizons is None
                         else tuple(args.moment_horizons)), device=device)
    print(path)


if __name__ == "__main__":
    main()
