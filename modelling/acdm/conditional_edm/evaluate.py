from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def one_step_metrics(samples: Tensor, next_state: Tensor) -> dict[str, float]:
    """Basic ensemble diagnostics for samples [batch, ensemble, mode]."""

    if samples.ndim != 3 or next_state.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("incompatible sample and next-state shapes")
    ensemble_mean = samples.mean(dim=1)
    return {
        "mean_prediction_mse": float((ensemble_mean - next_state).square().mean()),
        "mean_sample_variance": float(samples.var(dim=1, correction=0).mean()),
        "mean_sample_rmse": float(
            (samples - next_state[:, None, :]).square().mean(dim=-1).sqrt().mean()
        ),
    }


def rollout_statistics(trajectories: Tensor) -> dict[str, Tensor]:
    """Ensemble moments over [batch, ensemble, physical time, mode]."""

    if trajectories.ndim != 4:
        raise ValueError("trajectories must have shape [batch, ensemble, physical time, mode]")
    mean = trajectories.mean(dim=1)
    centered = trajectories - mean[:, None]
    covariance = torch.einsum("bekr,beks->bkrs", centered, centered) / trajectories.shape[1]
    return {
        "mean": mean,
        "variance": covariance.diagonal(dim1=-2, dim2=-1),
        "covariance": covariance,
    }


def _covariance(samples: Tensor) -> Tensor:
    centered = samples - samples.mean(dim=0)
    return centered.T @ centered / samples.shape[0]


def _relative_frobenius(predicted: Tensor, reference: Tensor) -> float:
    return float((predicted - reference).norm() / reference.norm().clamp_min(1.0e-12))


def segmented_reference_ensemble(
    states: Tensor,
    trajectory_indices: list[int] | tuple[int, ...] | np.ndarray,
    *,
    segment_length: int,
    lag_steps: int = 1,
    history_steps: int = 0,
) -> dict[str, Tensor | int | None]:
    """Build exchangeable, non-overlapping reference windows.

    ``segment_length`` is the number of native data samples in each window. The
    returned reference is sampled at ``lag_steps`` to match one model rollout
    transition. Incomplete final windows are discarded. If history is needed,
    the first window of an experiment is omitted because it has no pre-window
    observations with which to condition the model.
    """

    if states.ndim != 3:
        raise ValueError("states must have shape [experiment, physical time, mode]")
    if segment_length < 2:
        raise ValueError("segment_length must be at least 2")
    if lag_steps < 1:
        raise ValueError("lag_steps must be positive")
    if history_steps < 0:
        raise ValueError("history_steps must be nonnegative")

    experiment_ids = tuple(int(index) for index in trajectory_indices)
    if not experiment_ids:
        raise ValueError("trajectory_indices cannot be empty")
    if min(experiment_ids) < 0 or max(experiment_ids) >= states.shape[0]:
        raise IndexError("trajectory index outside states")

    run_ids: list[int] = []
    start_indices: list[int] = []
    skipped_missing_history = 0
    minimum_start = history_steps * lag_steps
    for run_id in experiment_ids:
        for start in range(0, states.shape[1] - segment_length + 1, segment_length):
            if start < minimum_start:
                skipped_missing_history += 1
                continue
            run_ids.append(run_id)
            start_indices.append(start)
    if not run_ids:
        raise ValueError("no complete segments have the history required by the model")

    runs = torch.tensor(run_ids, dtype=torch.long, device=states.device)
    starts = torch.tensor(start_indices, dtype=torch.long, device=states.device)
    relative_steps = torch.arange(
        0,
        segment_length,
        lag_steps,
        dtype=torch.long,
        device=states.device,
    )
    reference_indices = starts[:, None] + relative_steps[None]
    reference = states[runs[:, None], reference_indices]
    history = None
    if history_steps:
        history_offsets = lag_steps * torch.arange(
            1,
            history_steps + 1,
            dtype=torch.long,
            device=states.device,
        )
        history = states[runs[:, None], starts[:, None] - history_offsets[None]]
    return {
        "reference": reference,
        "initial_state": reference[:, 0],
        "history_states": history,
        "run_ids": runs,
        "start_indices": starts,
        "relative_steps": relative_steps,
        "skipped_missing_history": skipped_missing_history,
    }


def segmented_ensemble_metrics(
    trajectories: Tensor,
    reference: Tensor,
) -> dict[str, object]:
    """Compare moments across generated and experimental window ensembles.

    Both inputs have shape ``[segment, relative time, mode]``. Each generated
    path starts from the corresponding experimental window's initial state, but
    only the ensemble moments are compared; window ordering has no effect.
    """

    if trajectories.ndim != 3 or reference.ndim != 3:
        raise ValueError("expected generated and reference [segment, time, mode]")
    if trajectories.shape != reference.shape:
        raise ValueError("generated and reference segment shapes must match")
    if trajectories.shape[0] < 2:
        raise ValueError("at least two segments are required to estimate covariance")

    generated_mean = trajectories.mean(dim=0)
    reference_mean = reference.mean(dim=0)
    generated_centered = trajectories - generated_mean[None]
    reference_centered = reference - reference_mean[None]
    covariance_denominator = trajectories.shape[0] - 1
    generated_covariance = torch.einsum(
        "lti,ltj->tij", generated_centered, generated_centered
    ) / covariance_denominator
    reference_covariance = torch.einsum(
        "lti,ltj->tij", reference_centered, reference_centered
    ) / covariance_denominator

    mean_difference = generated_mean - reference_mean
    covariance_difference = generated_covariance - reference_covariance
    mean_relative_error_over_time = (
        mean_difference.norm(dim=-1) / reference_mean.norm(dim=-1).clamp_min(1.0e-12)
    )
    covariance_relative_error_over_time = (
        covariance_difference.flatten(start_dim=1).norm(dim=-1)
        / reference_covariance.flatten(start_dim=1).norm(dim=-1).clamp_min(1.0e-12)
    )

    return {
        "mean_relative_l2_error": _relative_frobenius(generated_mean, reference_mean),
        "mean_relative_error_average": float(mean_relative_error_over_time.mean()),
        "mean_relative_error_final": float(mean_relative_error_over_time[-1]),
        "mean_relative_error_over_time": mean_relative_error_over_time.tolist(),
        "covariance_relative_frobenius_error": _relative_frobenius(
            generated_covariance, reference_covariance
        ),
        "covariance_relative_error_average": float(
            covariance_relative_error_over_time.mean()
        ),
        "covariance_relative_error_final": float(covariance_relative_error_over_time[-1]),
        "covariance_relative_error_over_time": covariance_relative_error_over_time.tolist(),
    }


def _increment_correlation(trajectories: Tensor) -> Tensor:
    """Per-mode correlation of consecutive increments for [path, time, mode]."""

    increments = trajectories[:, 1:] - trajectories[:, :-1]
    first = increments[:, :-1].reshape(-1, trajectories.shape[-1])
    second = increments[:, 1:].reshape(-1, trajectories.shape[-1])
    first = first - first.mean(dim=0)
    second = second - second.mean(dim=0)
    numerator = (first * second).mean(dim=0)
    denominator = (first.square().mean(dim=0) * second.square().mean(dim=0)).sqrt()
    return numerator / denominator.clamp_min(1.0e-12)


def stochastic_rollout_metrics(
    trajectories: Tensor,
    reference: Tensor,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
) -> dict[str, object]:
    """Metrics for generated [condition, ensemble, time, mode] and reference [condition, time, mode]."""

    if trajectories.ndim != 4 or reference.ndim != 3:
        raise ValueError("expected generated [condition, ensemble, time, mode] and reference [condition, time, mode]")
    if trajectories.shape[0] != reference.shape[0] or trajectories.shape[2:] != reference.shape[1:]:
        raise ValueError("generated and reference rollout shapes are incompatible")
    if trajectories.shape[2] < 3:
        raise ValueError("at least two rollout transitions are required")

    generated_increment = (trajectories[:, :, 1] - trajectories[:, :, 0] - target_mean) / target_std
    reference_increment = (reference[:, 1] - reference[:, 0] - target_mean) / target_std
    predicted_mean = generated_increment.mean(dim=1)
    mean_mse = (predicted_mean - reference_increment).square().mean()
    zero_increment = -target_mean / target_std
    zero_mse = (reference_increment - zero_increment).square().mean()

    centered = generated_increment - predicted_mean[:, None]
    predicted_conditional_covariance = torch.einsum(
        "bem,ben->mn", centered, centered
    ) / (trajectories.shape[0] * trajectories.shape[1])
    innovations = reference_increment - predicted_mean
    empirical_innovation_covariance = _covariance(innovations)

    generated_flat = generated_increment.reshape(-1, generated_increment.shape[-1])
    generated_marginal_covariance = _covariance(generated_flat)
    reference_marginal_covariance = _covariance(reference_increment)
    marginal_mean_error = (
        generated_flat.mean(dim=0) - reference_increment.mean(dim=0)
    ).square().mean().sqrt()

    distances_to_observation = (generated_increment - reference_increment[:, None]).norm(dim=-1)
    pairwise_distances = torch.cdist(generated_increment, generated_increment)
    energy_score = distances_to_observation.mean() - 0.5 * pairwise_distances.mean()
    lower = generated_increment.quantile(0.05, dim=1)
    upper = generated_increment.quantile(0.95, dim=1)
    coverage = ((reference_increment >= lower) & (reference_increment <= upper)).float().mean()

    generated_state = (trajectories - state_mean) / state_std
    reference_state = (reference - state_mean) / state_std
    normalized_generated_mean = generated_state.mean(dim=(0, 1))
    normalized_reference_mean = reference_state.mean(dim=0)
    normalized_mean_difference = normalized_generated_mean - normalized_reference_mean
    mean_rmse_over_time = normalized_mean_difference.square().mean(dim=-1).sqrt()

    raw_generated_mean = trajectories.mean(dim=(0, 1))
    raw_reference_mean = reference.mean(dim=0)
    raw_mean_difference = raw_generated_mean - raw_reference_mean
    mean_relative_l2_error_over_time = (
        raw_mean_difference.norm(dim=-1)
        / raw_reference_mean.norm(dim=-1).clamp_min(1.0e-12)
    )
    covariance_error_over_time = []
    normalized_covariance_error_over_time = []
    for time_index in range(trajectories.shape[2]):
        normalized_generated_covariance = _covariance(
            generated_state[:, :, time_index].reshape(-1, trajectories.shape[-1])
        )
        normalized_reference_covariance = _covariance(reference_state[:, time_index])
        normalized_covariance_error_over_time.append(
            _relative_frobenius(
                normalized_generated_covariance,
                normalized_reference_covariance,
            )
        )
        raw_generated_covariance = _covariance(
            trajectories[:, :, time_index].reshape(-1, trajectories.shape[-1])
        )
        raw_reference_covariance = _covariance(reference[:, time_index])
        covariance_error_over_time.append(
            _relative_frobenius(raw_generated_covariance, raw_reference_covariance)
        )
    generated_energy = generated_state.square().mean(dim=(0, 1, 3))
    reference_energy = reference_state.square().mean(dim=(0, 2))

    generated_paths = generated_state.reshape(-1, generated_state.shape[2], generated_state.shape[3])
    generated_correlation = _increment_correlation(generated_paths)
    reference_correlation = _increment_correlation(reference_state)
    correlation_error = (generated_correlation - reference_correlation).square().mean().sqrt()

    covariance_error = _relative_frobenius(
        predicted_conditional_covariance, empirical_innovation_covariance
    )
    return {
        "one_step": {
            "conditional_mean_rmse": float(mean_mse.sqrt()),
            "zero_increment_rmse": float(zero_mse.sqrt()),
            "conditional_mean_skill": float(1 - mean_mse / zero_mse.clamp_min(1.0e-12)),
            "conditional_covariance_relative_error": covariance_error,
            "predicted_conditional_variance_mean": float(
                predicted_conditional_covariance.diagonal().mean()
            ),
            "empirical_innovation_variance_mean": float(
                empirical_innovation_covariance.diagonal().mean()
            ),
            "marginal_mean_rmse": float(marginal_mean_error),
            "marginal_covariance_relative_error": _relative_frobenius(
                generated_marginal_covariance, reference_marginal_covariance
            ),
            "energy_score": float(energy_score),
            "central_90_percent_coverage": float(coverage),
            "coverage_absolute_error": float(abs(coverage - 0.9)),
        },
        "rollout": {
            "raw_generated_mean_over_time": raw_generated_mean.tolist(),
            "raw_reference_mean_over_time": raw_reference_mean.tolist(),
            "mean_rmse_over_time": mean_rmse_over_time.tolist(),
            "mean_rmse_average": float(mean_rmse_over_time.mean()),
            "mean_rmse_final": float(mean_rmse_over_time[-1]),
            "mean_rmse_coordinate_system": "normalized reduced state",
            "mean_relative_l2_error_over_time": mean_relative_l2_error_over_time.tolist(),
            "mean_relative_l2_error_average": float(mean_relative_l2_error_over_time.mean()),
            "mean_relative_l2_error_final": float(mean_relative_l2_error_over_time[-1]),
            "mean_relative_l2_error_global": _relative_frobenius(
                raw_generated_mean, raw_reference_mean
            ),
            "mean_relative_l2_error_coordinate_system": "raw reduced state",
            "covariance_relative_error_over_time": covariance_error_over_time,
            "covariance_relative_error_average": float(np.mean(covariance_error_over_time)),
            "covariance_relative_error_final": covariance_error_over_time[-1],
            "covariance_relative_error_coordinate_system": "raw reduced state",
            "normalized_covariance_relative_error_over_time": (
                normalized_covariance_error_over_time
            ),
            "normalized_covariance_relative_error_average": float(
                np.mean(normalized_covariance_error_over_time)
            ),
            "normalized_covariance_relative_error_final": (
                normalized_covariance_error_over_time[-1]
            ),
            "normalized_energy_relative_error_average": float(
                ((generated_energy - reference_energy).abs() / reference_energy.clamp_min(1.0e-12)).mean()
            ),
            "consecutive_increment_correlation_rmse": float(correlation_error),
            "generated_increment_correlation_per_mode": generated_correlation.tolist(),
            "reference_increment_correlation_per_mode": reference_correlation.tolist(),
        },
    }


def reference_windows(data, config, split, num_conditions, horizon, seed, *,
                      increment_norm_cutoff=None, state_std=None, filter_report=None):
    """Fixed, uniformly sampled windows with true preceding history."""
    names = data.splits.get(split)
    if not names:
        raise ValueError(f"No repetitions in split {split!r}.")
    lag = config.lag_steps
    history_steps = config.history_steps if config.history_conditioning else 0
    velocity_state = getattr(config, "state_variable", None) == "velocity"
    history_lag = 1 if velocity_state else lag
    first = history_steps * history_lag
    counts = np.array([max(0, data.frame_counts[n] - horizon * lag - first) for n in names])
    if not counts.sum():
        raise ValueError("No reference window is long enough for this history and horizon.")
    # Uniform over valid initial conditions, sampled reproducibly with replacement.
    rng = np.random.default_rng(seed)
    if increment_norm_cutoff is None:
        draws = rng.integers(int(counts.sum()), size=num_conditions)
        cumulative = np.cumsum(counts)
        selected_runs = np.searchsorted(cumulative, draws, side="right")
        selected_starts = first + draws - np.r_[0, cumulative[:-1]][selected_runs]
    else:
        scale = np.asarray(state_std, dtype=np.float32)
        if (not np.isfinite(increment_norm_cutoff) or increment_norm_cutoff <= 0
                or scale.shape != (data.n_coordinates,) or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise ValueError("Reference-window trim requires a positive cutoff and valid state scale.")
        eligible_runs, eligible_starts = [], []
        for run, (name, count) in enumerate(zip(names, counts)):
            if not count:
                continue
            _, values = data.read_coordinates(name, 0, data.frame_counts[name])
            values = np.asarray(values, dtype=np.float32)
            cutoff_lag = 1 if velocity_state else lag
            increment_norm = np.linalg.norm(
                (values[cutoff_lag:] - values[:-cutoff_lag]) / scale, axis=1)
            good = increment_norm <= increment_norm_cutoff
            starts_for_run = first + np.arange(count)
            valid = np.ones(count, dtype=bool)
            offsets = (range(-first, horizon * lag) if velocity_state
                       else range(0, horizon * lag, lag))
            for offset in offsets:
                valid &= good[starts_for_run + offset]
            eligible_starts.extend(starts_for_run[valid])
            eligible_runs.extend([run] * int(valid.sum()))
        if not eligible_starts:
            raise ValueError("No reference rollout window remains after applying the training trim cutoff.")
        choice = rng.integers(len(eligible_starts), size=num_conditions)
        selected_runs = np.asarray(eligible_runs)[choice]
        selected_starts = np.asarray(eligible_starts)[choice]
        if filter_report is not None:
            filter_report.update(
                method="all_window_transitions_below_training_increment_norm_cutoff",
                normalized_increment_norm_cutoff=float(increment_norm_cutoff),
                candidate_windows_before_trim=int(counts.sum()),
                candidate_windows_after_trim=len(eligible_starts),
                rejected_window_percent=100 * (1 - len(eligible_starts) / int(counts.sum())),
                generated_trajectories_trimmed=False,
            )
    references, histories, times, selected_names, starts = [], [], [], [], []
    for run, start in zip(selected_runs, selected_starts):
        run, start = int(run), int(start)
        name = names[run]
        time, values = data.read_coordinates(name, start - first, start + horizon * lag + 1)
        indices = first + lag * np.arange(horizon + 1)
        references.append(values[indices])
        histories.append(values[first - history_lag * np.arange(1, history_steps + 1)])
        times.append(time[indices])
        selected_names.append(name)
        starts.append(start)
    reference = torch.as_tensor(np.stack(references), dtype=torch.float32)
    history = torch.as_tensor(np.stack(histories), dtype=torch.float32) if history_steps else None
    return reference, history, np.stack(times), np.asarray(selected_names), np.asarray(starts)


def evaluate_model(model, data, *, split="test", num_conditions=32, ensemble_size=16,
                   horizon=50, batch_size=8, sampling_steps=None, seed=0, metrics=True,
                   reference_increment_norm_cutoff=None,
                   reference_increment_norm_scale=None,
                   metric_target_mean=None, metric_target_std=None,
                   rollout_options=None):
    """Sample physical-coordinate trajectories and matching reference windows."""
    if min(num_conditions, ensemble_size, batch_size, horizon) < 1 or (metrics and horizon < 2):
        raise ValueError("Counts must be positive; metrics require horizon >= 2.")
    if metrics and not data.include_spatial_mean:
        raise ValueError("Aggregate rollout metrics require spatial mean; use --no-metrics for POD-only rollouts.")
    filter_report = {}
    reference, history, times, selected_names, starts = reference_windows(
        data, model.config, split, num_conditions, horizon, seed,
        increment_norm_cutoff=reference_increment_norm_cutoff,
        state_std=(model.state_std.detach().cpu()
                   if reference_increment_norm_scale is None
                   else reference_increment_norm_scale),
        filter_report=filter_report)
    lag = model.config.lag_steps
    device = next(model.parameters()).device
    rollout_options = {} if rollout_options is None else dict(rollout_options)
    generated = []
    for start in range(0, num_conditions, batch_size):
        generated.append(model.rollout(
            reference[start:start + batch_size, 0].to(device),
            history_states=None if history is None else history[start:start + batch_size].to(device),
            horizon=horizon, num_trajectories=ensemble_size, num_steps=sampling_steps,
            seed=seed + start,
            **rollout_options,
        ).cpu())
    trajectories = torch.cat(generated)
    result = dict(trajectories=trajectories.numpy(), reference=reference.numpy(),
                  reference_time=np.stack(times), repetition=np.asarray(selected_names),
                  start_index=np.asarray(starts))
    report = dict(split=split, conditions=num_conditions, ensemble_size=ensemble_size,
                  horizon=horizon, lag_steps=lag, native_dt=data.native_dt,
                  time_units=data.time_units, physical_lag=lag * data.native_dt,
                  coordinate_order=("sqrt(N)*spatial_mean, shared POD coefficients" if data.include_spatial_mean
                                    else "shared POD coefficients only"),
                  include_spatial_mean=data.include_spatial_mean,
                  spatial_mean_highpass_hz=data.spatial_mean_highpass_hz)
    if filter_report:
        report["reference_window_filter"] = filter_report
    if metrics:
        target_mean = (model.target_mean.cpu() if metric_target_mean is None
                       else metric_target_mean.cpu())
        target_std = (model.target_std.cpu() if metric_target_std is None
                      else metric_target_std.cpu())
        report.update(stochastic_rollout_metrics(
            trajectories, reference, state_mean=model.state_mean.cpu(), state_std=model.state_std.cpu(),
            target_mean=target_mean, target_std=target_std))
    return result, report


def main(argv=None, *, metrics=True):
    from .checkpoint import load_model
    from .data import checkpoint_data
    from .train import resolve_device

    parser = argparse.ArgumentParser(description="Evaluate shared-POD EDM rollouts using checkpoint data and splits.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Optional source tree relocation.")
    parser.add_argument("--shared-basis", type=Path, help="Optional basis relocation; identity must match.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--num-conditions", type=int, default=32)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Save trajectories without aggregate rollout metrics (useful for long visual diagnostics).")
    parser.add_argument("--output", type=Path, required=True, help="Directory for rollout.npz and metrics.json.")
    args = parser.parse_args(argv)
    model, checkpoint = load_model(args.checkpoint, resolve_device(args.device))
    data = checkpoint_data(checkpoint, data_root=args.data_root, shared_basis=args.shared_basis)
    result, report = evaluate_model(
        model, data, split=args.split, num_conditions=args.num_conditions,
        ensemble_size=args.ensemble_size, horizon=args.horizon, batch_size=args.batch_size,
        sampling_steps=args.sampling_steps, seed=args.seed, metrics=metrics and not args.no_metrics)
    report["source_shared_basis"] = str(data.basis_path)
    report["rank"] = data.rank
    report["checkpoint"] = str(args.checkpoint.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "rollout.npz", **result)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
