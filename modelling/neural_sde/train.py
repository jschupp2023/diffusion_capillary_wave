"""Train/evaluate the SDE on the same shared-POD source and rollout metrics as EDM."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from data_analysis.energy.capillary_energy import (capillary_stiffness, length_scale,
                                                   matrix_diagnostics)
from modelling.acdm.conditional_edm.checkpoint import load_checkpoint, save_checkpoint
from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.data import (NormalizationStats, checkpoint_data,
                                                 fit_normalization, transition_batches)
from modelling.acdm.conditional_edm.evaluate import evaluate_model
from modelling.acdm.conditional_edm.train import resolve_device, seed_everything
from modelling.data_preparation.prepare_shared_pod_training import RunningMoments, SharedPODTrainingData

from .config import SDEConfig
from .losses import EulerMaruyamaNLL, MonteCarloMultistepNLL, StationaryMeanLoss
from .model import NeuralSDE
from .stability import (DiscreteLyapunovRegularizer, StabilityConfig,
                        fit_stability_regularizer)


def _validated_multistep_mode_counts(mode_counts, horizons, rank, target):
    """Validate optional first-POD-mode counts aligned with the horizons."""
    if mode_counts is None:
        return None
    counts = tuple(mode_counts)
    if (target != "velocity" or len(counts) != len(horizons)
            or any(not isinstance(count, int) or isinstance(count, bool)
                   or not 1 <= count <= rank for count in counts)):
        raise ValueError(
            "Multistep mode counts require one integer in [1, rank] per velocity horizon")
    return counts


def _validated_multistep_particle_counts(particle_counts, horizons, particles):
    """Return one nonincreasing particle count per horizon."""
    if (not isinstance(particles, int) or isinstance(particles, bool) or particles < 1):
        raise ValueError("Multistep particles must be a positive integer")
    if particle_counts is None:
        return (particles,) * len(horizons)
    counts = tuple(particle_counts)
    if (len(counts) != len(horizons)
            or any(not isinstance(count, int) or isinstance(count, bool) or count < 1
                   for count in counts)
            or any(following > current for current, following in zip(counts, counts[1:]))):
        raise ValueError(
            "Multistep particle counts require one nonincreasing positive integer per horizon")
    return counts


def _edm_transition_config(dimension: int, lag_steps: int = 1,
                           history_steps: int = 0) -> EDMConfig:
    """Reuse EDM's physical lagged batches and training-only increment scale."""
    return EDMConfig(reduced_dim=dimension, lag_steps=lag_steps, target_mode="increment",
                     history_conditioning=history_steps > 0,
                     history_steps=max(1, history_steps))


def _fit_sde_normalization(data, config: SDEConfig) -> NormalizationStats:
    """Fit state scaling, plus native per-frame velocity scaling when requested."""
    base = fit_normalization(data, _edm_transition_config(
        data.n_coordinates, config.lag_steps,
        0 if config.state_variable == "velocity" else config.history_steps))
    if config.state_variable != "velocity":
        return base
    velocities = RunningMoments(data.n_coordinates)
    for _, batch in data.iter_transitions("train", lag_steps=1, history_steps=1,
                                           batch_size=data.batch_size):
        history = batch["history_states"]
        velocity = batch["current_state"] - history[:, 0]
        velocities.update(velocity)
    mean, _, scale, _ = velocities.finish()
    return NormalizationStats(base.state_mean, base.state_std,
                              torch.as_tensor(mean, dtype=torch.float32),
                              torch.as_tensor(scale, dtype=torch.float32))


def _configure_capillary_energy(model: NeuralSDE, data: SharedPODTrainingData):
    """Fix Q in standardized position coordinates from the shared POD modes."""
    if model.config.drift_type != "capillary_energy":
        return None
    if model.config.capillary_mean_mode != data.include_spatial_mean:
        raise ValueError("capillary mean-mode configuration disagrees with prepared data")
    with h5py.File(data.basis_path, "r") as basis:
        modes = np.asarray(basis["pod/modes"][:data.rank], dtype=np.float64)
        x = np.asarray(basis["grid/x"], dtype=np.float64)
        y = np.asarray(basis["grid/y"], dtype=np.float64)
        x_units = str(basis["grid/x"].attrs["units"])
        y_units = str(basis["grid/y"].attrs["units"])
    x_m = x * length_scale(x_units)
    y_m = y * length_scale(y_units)
    physical, construction = capillary_stiffness(
        modes, x_m, y_m, gamma=model.config.surface_tension)
    offset = int(data.include_spatial_mean)
    state_scale_m = (np.asarray(model.state_std.detach().cpu(), dtype=np.float64)[offset:]
                     * length_scale(data.signal_units))
    normalized_pod = state_scale_m[:, None] * physical * state_scale_m[None, :]
    normalized = np.zeros((data.n_coordinates, data.n_coordinates), dtype=np.float64)
    normalized[offset:, offset:] = normalized_pod
    diagnostics = matrix_diagnostics(normalized)
    model.drift_model.set_energy_matrix(torch.as_tensor(
        normalized, dtype=model.drift_model.energy_matrix.dtype,
        device=model.drift_model.energy_matrix.device))
    return dict(
        definition="0.5 * q_standardized.T @ Q @ q_standardized",
        coordinate_space="training-standardized shared-POD position",
        energy_units="J",
        surface_tension_N_per_m=model.config.surface_tension,
        spatial_geometry="surface",
        state_scale_m=state_scale_m.tolist(),
        spatial_mean_coordinate=0 if data.include_spatial_mean else None,
        spatial_mean_Q_row_and_column_zero=bool(data.include_spatial_mean),
        matrix_diagnostics=diagnostics,
        quadrature_measure_m2=construction["measure"],
    )


def _velocity_transition_arrays(data, config, split, batch_size, seed=None,
                                multistep_horizons: tuple[int, ...] | None = None):
    """Lagged position pairs with native one-frame velocities at both endpoints."""
    maximum_horizon = max(multistep_horizons or (1,))
    for _, batch in data.iter_transitions(
            split, lag_steps=config.lag_steps * maximum_horizon,
            history_steps=config.history_steps,
            history_spacing_steps=1,
            include_next_history=True, include_state_path=True,
            batch_size=batch_size, seed=seed):
        if multistep_horizons is not None:
            path = batch["state_path"]
            current_index = config.history_steps
            one_step_index = current_index + config.lag_steps
            batch["next_state"] = path[:, one_step_index]
            history_offsets = np.arange(1, config.history_steps + 1)
            batch["next_history_states"] = path[
                :, one_step_index - history_offsets]
            endpoint_indices = current_index + config.lag_steps * np.asarray(
                multistep_horizons)
            batch["target_velocities"] = (
                path[:, endpoint_indices] - path[:, endpoint_indices - 1])
            batch["target_displacements"] = (
                path[:, endpoint_indices] - path[:, current_index, None, :])
        yield batch


def _sde_transition_batches(data, config, split, batch_size, seed=None,
                            multistep_horizons: tuple[int, ...] | None = None):
    if config.state_variable != "velocity":
        if multistep_horizons is not None:
            raise ValueError("multistep likelihood requires state_variable velocity")
        yield from transition_batches(data, _edm_transition_config(
            data.n_coordinates, config.lag_steps, config.history_steps),
                                      split, batch_size, seed=seed)
        return
    for batch in _velocity_transition_arrays(
            data, config, split, batch_size, seed, multistep_horizons):
        yield {key: torch.from_numpy(value).float() for key, value in batch.items()
               if key in ("current_state", "next_state", "history_states",
                          "next_history_states", "state_path", "target_velocities",
                          "target_displacements")}


def _trim_keep(batch, model, cutoff, state_std=None):
    """Mask transitions whose complete modeled stencil contains a large state jump."""
    scale = (model.state_std.detach().cpu() if state_std is None
             else torch.as_tensor(state_std, dtype=batch["current_state"].dtype))
    if model.config.state_variable == "velocity":
        path = batch["state_path"]
        normalized_deltas = (path[:, 1:] - path[:, :-1]) / scale
        norms = torch.linalg.vector_norm(normalized_deltas, dim=2)
    else:
        delta = (batch["next_state"] - batch["current_state"]) / scale
        norms = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    return (norms <= cutoff).all(dim=1)


def _split_trim_counts(data, model, split, batch_size, cutoff, state_std=None,
                       multistep_horizons: tuple[int, ...] | None = None):
    total = removed = 0
    batches = _sde_transition_batches(
        data, model.config, split, batch_size,
        multistep_horizons=multistep_horizons)
    for batch in batches:
        keep = _trim_keep(batch, model, cutoff, state_std)
        total += len(keep)
        removed += int((~keep).sum())
    return total, removed


def _globally_mixed_retained_velocity_batches(
        data, model, split, source_batch_size, trajectory_count, seed,
        trim_cutoff=None, trim_state_std=None, horizon=100):
    """Build endpoint ensembles with one clean horizon window per source block."""
    generator = torch.Generator().manual_seed(seed)
    current, history, endpoint = [], [], []
    for batch in _sde_transition_batches(
            data, model.config, split, source_batch_size, seed=seed,
            multistep_horizons=(horizon,)):
        if trim_cutoff is not None:
            keep = _trim_keep(batch, model, trim_cutoff, trim_state_std)
            if not bool(keep.any()):
                continue
            batch = {key: value[keep] for key, value in batch.items()}
        available = len(batch["current_state"])
        if not available:
            continue
        selected = int(torch.randint(available, (), generator=generator))
        current.append(batch["current_state"][selected:selected + 1])
        history.append(batch["history_states"][selected:selected + 1])
        endpoint.append(
            batch["current_state"][selected:selected + 1]
            + batch["target_displacements"][selected:selected + 1, 0])
        if len(current) == trajectory_count:
            yield {"current_state": torch.cat(current),
                   "history_states": torch.cat(history),
                   "data_endpoint": torch.cat(endpoint)}
            current, history, endpoint = [], [], []


def _capped_multistep_batches(data, model, split, batch_size, horizons, max_windows,
                              window_seed, trim_cutoff=None, trim_state_std=None):
    """Select a reproducibly shuffled subset of retained multistep windows.

    Selection uses the data source's native batch size, rather than the scoring
    batch size, so changing GPU memory settings does not change the sampled
    windows. The selected CPU tensors are then rebatched for scoring.
    """
    selected = []
    count = 0
    for batch in _sde_transition_batches(
            data, model.config, split, data.batch_size, seed=window_seed,
            multistep_horizons=horizons):
        if trim_cutoff is not None:
            keep = _trim_keep(batch, model, trim_cutoff, trim_state_std)
            if not bool(keep.any()):
                continue
            batch = {key: value[keep] for key, value in batch.items()}
        take = min(len(batch["current_state"]), max_windows - count)
        if take:
            selected.append({key: value[:take] for key, value in batch.items()})
            count += take
        if count == max_windows:
            break
    if not selected:
        return
    merged = {key: torch.cat([batch[key] for batch in selected])
              for key in selected[0]}
    for start in range(0, count, batch_size):
        yield {key: value[start:start + batch_size] for key, value in merged.items()}


def set_trainable_components(model: NeuralSDE, train_mode: str):
    if train_mode not in {"joint", "diffusion", "drift"}:
        raise ValueError("train_mode must be joint, diffusion, or drift")
    for parameter in model.drift_model.parameters():
        parameter.requires_grad_(train_mode in {"joint", "drift"})
    for parameter in model.diffusion_model.parameters():
        parameter.requires_grad_(train_mode in {"joint", "diffusion"})
    for status in (True, False):
        names = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad == status]
        print(f"{'Trainable' if status else 'Frozen'} parameters: {', '.join(names) or 'none'}",
              flush=True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _epoch(model, data, split, batch_size, device, objective, optimizer=None, seed=None,
           train_mode="joint", train_trim_cutoff=None, train_trim_state_std=None,
           multistep_objective: MonteCarloMultistepNLL | None = None,
           multistep_weight: float = 0., multistep_noise_seed: int | None = None,
           metric_output: dict | None = None, multistep_max_windows: int | None = None,
           multistep_window_seed: int = 0,
           multistep_window_fraction: float = 1.,
           multistep_window_sampling: str = "examples",
           stationary_mean_objective: StationaryMeanLoss | None = None,
           stationary_mean_weight: float = 0.,
           stationary_variance_weight: float = 0.,
           stationary_mean_trajectories: int = 30,
           stationary_mean_batch_fraction: float = .05,
           stationary_mean_noise_seed: int = 0,
           stationary_mean_window_seed: int = 0,
           stability_objective: DiscreteLyapunovRegularizer | None = None,
           stability_noise_seed: int = 0):
    model.train(optimizer is not None)
    use_multistep = multistep_weight > 0
    use_stationary_mean = stationary_mean_weight > 0
    use_stationary_variance = stationary_variance_weight > 0
    use_stationary_moments = use_stationary_mean or use_stationary_variance
    use_stability = stability_objective is not None and stability_objective.config.weight > 0
    stability_totals = {}
    stability_generator = (torch.Generator(device=torch.device(device)).manual_seed(
        stability_noise_seed) if use_stability else None)
    if use_multistep and multistep_objective is None:
        raise ValueError("positive multistep_weight requires a multistep objective")
    if (not math.isfinite(multistep_window_fraction)
            or not 0 < multistep_window_fraction <= 1):
        raise ValueError("multistep_window_fraction must lie in (0, 1]")
    if multistep_window_sampling not in {"examples", "batches"}:
        raise ValueError("multistep_window_sampling must be examples or batches")
    if use_stationary_moments and stationary_mean_objective is None:
        raise ValueError("positive stationary moment weight requires its objective")
    if (not math.isfinite(stationary_mean_weight) or stationary_mean_weight < 0
            or not math.isfinite(stationary_variance_weight)
            or stationary_variance_weight < 0
            or not isinstance(stationary_mean_trajectories, int)
            or isinstance(stationary_mean_trajectories, bool)
            or stationary_mean_trajectories < 2
            or not math.isfinite(stationary_mean_batch_fraction)
            or not 0 < stationary_mean_batch_fraction <= 1
            or not isinstance(stationary_mean_noise_seed, int)
            or isinstance(stationary_mean_noise_seed, bool)
            or not isinstance(stationary_mean_window_seed, int)
            or isinstance(stationary_mean_window_seed, bool)):
        raise ValueError("invalid stationary moment weight, trajectory count, fraction, or seed")
    if multistep_max_windows is not None:
        if (not use_multistep or not isinstance(multistep_max_windows, int)
                or isinstance(multistep_max_windows, bool) or multistep_max_windows < 1
                or not isinstance(multistep_window_seed, int)
                or isinstance(multistep_window_seed, bool)):
            raise ValueError("multistep window cap requires a positive count and integer seed")
    horizons = multistep_objective.horizons if use_multistep else None
    total = one_step_total = 0.
    multistep_totals = ({horizon: 0. for horizon in horizons}
                        if horizons is not None else {})
    count = multistep_count = 0
    stationary_mean_total = 0.
    stationary_mean_count = stationary_mean_batch_count = 0
    stationary_mean_error_sum = None
    stationary_mean_weighted_total = 0.
    stationary_variance_total = 0.
    stationary_variance_error_sum = None
    stationary_variance_weighted_total = 0.
    windows_pretrimmed = multistep_max_windows is not None
    if windows_pretrimmed:
        batches = _capped_multistep_batches(
            data, model, split, batch_size, horizons, multistep_max_windows,
            multistep_window_seed, train_trim_cutoff, train_trim_state_std)
    else:
        batches = _sde_transition_batches(
            data, model.config, split, batch_size, seed=seed,
            multistep_horizons=horizons)
    generator = None
    if use_multistep and multistep_noise_seed is not None:
        generator = torch.Generator(device=torch.device(device)).manual_seed(
            multistep_noise_seed)
    window_generator = None
    if use_multistep and multistep_window_fraction < 1:
        window_generator = torch.Generator().manual_seed(multistep_window_seed)
    batch_sampling_accumulator = (
        float(torch.rand((), generator=window_generator))
        if window_generator is not None and multistep_window_sampling == "batches" else 0.)
    stationary_mean_batches = (iter(_globally_mixed_retained_velocity_batches(
        data, model, split, min(256, data.batch_size, batch_size),
        stationary_mean_trajectories,
        stationary_mean_window_seed, train_trim_cutoff,
        train_trim_state_std, stationary_mean_objective.horizon))
        if use_stationary_moments else None)
    stationary_mean_sampler_cycle = 0
    stationary_window_generator = (
        torch.Generator().manual_seed(stationary_mean_window_seed)
        if use_stationary_moments else None)
    stationary_noise_generator = (
        torch.Generator(device=torch.device(device)).manual_seed(
            stationary_mean_noise_seed) if use_stationary_moments else None)
    stationary_batch_accumulator = (
        float(torch.rand((), generator=stationary_window_generator))
        if use_stationary_moments and stationary_mean_batch_fraction < 1 else 0.)
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in batches:
            if train_trim_cutoff is not None and not windows_pretrimmed:
                keep = _trim_keep(batch, model, train_trim_cutoff, train_trim_state_std)
                if not bool(keep.any()):
                    continue
                batch = {key: value[keep] for key, value in batch.items()}
            current = batch["current_state"].to(device)
            following = batch["next_state"].to(device)
            history = batch.get("history_states")
            options = dict(zero_drift=train_mode == "diffusion")
            if history is not None:
                options["history_states"] = history.to(device)
            if "next_history_states" in batch:
                options["following_history_states"] = batch["next_history_states"].to(device)
            one_step = objective(model, current, following, **options)
            horizon_losses = {}
            loss = one_step
            if use_multistep:
                multistep_scale = 1.
                score_multistep = True
                if (multistep_window_sampling == "batches"
                        and multistep_window_fraction < 1):
                    batch_sampling_accumulator += multistep_window_fraction
                    score_multistep = batch_sampling_accumulator >= 1.
                    if score_multistep:
                        batch_sampling_accumulator -= 1.
                        multistep_scale = 1. / multistep_window_fraction
                if score_multistep and multistep_window_sampling == "examples":
                    selected_count = max(1, math.ceil(
                        multistep_window_fraction * len(current)))
                else:
                    selected_count = len(current)
                if not score_multistep:
                    selected_cpu = selected_device = None
                elif selected_count == len(current):
                    selected_cpu = selected_device = None
                else:
                    selected_cpu = torch.randperm(
                        len(current), generator=window_generator)[:selected_count].sort().values
                    selected_device = selected_cpu.to(device)
                if score_multistep:
                    target_key = ("target_displacements"
                                  if multistep_objective.target == "displacement" else
                                  "target_velocities")
                    multistep_current = (current if selected_device is None else
                                         current.index_select(0, selected_device))
                    multistep_history = (options["history_states"]
                                         if selected_device is None else
                                         options["history_states"].index_select(
                                             0, selected_device))
                    multistep_targets = batch[target_key]
                    if selected_cpu is not None:
                        multistep_targets = multistep_targets.index_select(0, selected_cpu)
                    horizon_losses = multistep_objective.per_horizon(
                        model, multistep_current, multistep_targets.to(device),
                        history_states=multistep_history, generator=generator,
                        zero_drift=options["zero_drift"])
                    multistep = torch.stack(tuple(horizon_losses.values())).mean()
                    loss = one_step + multistep_weight * multistep_scale * multistep
                    multistep_count += selected_count
            stationary_loss = stationary_weighted = stationary_mean_error = None
            stationary_variance_loss = stationary_variance_weighted = None
            stationary_variance_error = None
            if use_stationary_moments:
                stationary_batch_accumulator += stationary_mean_batch_fraction
                score_stationary_mean = stationary_batch_accumulator >= 1.
                if score_stationary_mean:
                    stationary_batch_accumulator -= 1.
                    stationary_batch = None
                    # Restart exhausted global sampling with a deterministic new
                    # block shuffle. This retains broad temporal/repetition mixing
                    # when the requested ensembles outnumber one source pass.
                    for _ in range(2):
                        for candidate in stationary_mean_batches:
                            if len(candidate["current_state"]) >= 2:
                                stationary_batch = candidate
                                break
                        if stationary_batch is not None:
                            break
                        stationary_mean_sampler_cycle += 1
                        stationary_mean_batches = iter(
                            _globally_mixed_retained_velocity_batches(
                            data, model, split, min(256, data.batch_size, batch_size),
                            stationary_mean_trajectories,
                            stationary_mean_window_seed + stationary_mean_sampler_cycle,
                            train_trim_cutoff, train_trim_state_std,
                            stationary_mean_objective.horizon))
                    if stationary_batch is None:
                        raise ValueError(
                            f"stationary mean sampler found fewer than two {split} trajectories")
                    stationary_selected_count = len(stationary_batch["current_state"])
                    stationary_current = stationary_batch["current_state"]
                    stationary_history = stationary_batch["history_states"]
                    stationary_endpoint = stationary_batch["data_endpoint"]
                    (stationary_loss, stationary_mean_error,
                     stationary_variance_loss, stationary_variance_error) = (
                        stationary_mean_objective.statistics(
                            model, stationary_current.to(device),
                            stationary_endpoint.to(device),
                            history_states=stationary_history.to(device),
                            generator=stationary_noise_generator))
                    if use_stationary_mean:
                        stationary_weighted = (
                            stationary_mean_weight / stationary_mean_batch_fraction
                            * stationary_loss)
                        stationary_mean_total += (
                            float(stationary_loss.detach()) * stationary_selected_count)
                        contribution = (stationary_mean_error.detach().double().cpu()
                                        * stationary_selected_count)
                        stationary_mean_error_sum = (
                            contribution if stationary_mean_error_sum is None else
                            stationary_mean_error_sum + contribution)
                    if use_stationary_variance:
                        stationary_variance_weighted = (
                            stationary_variance_weight / stationary_mean_batch_fraction
                            * stationary_variance_loss)
                        stationary_variance_total += (
                            float(stationary_variance_loss.detach())
                            * stationary_selected_count)
                        contribution = (stationary_variance_error.detach().double().cpu()
                                        * stationary_selected_count)
                        stationary_variance_error_sum = (
                            contribution if stationary_variance_error_sum is None else
                            stationary_variance_error_sum + contribution)
                    stationary_mean_count += stationary_selected_count
                    stationary_mean_batch_count += 1
            if use_stability:
                stability_metrics = stability_objective(
                    model, current, history_states=options["history_states"],
                    generator=stability_generator, zero_drift=options["zero_drift"])
                loss = loss + stability_metrics["weighted_penalty"]
                for key, value in stability_metrics.items():
                    scalar = float(value.detach())
                    stability_totals[key] = (
                        max(stability_totals.get(key, -math.inf), scalar)
                        if key.endswith("max_excess") else
                        stability_totals.get(key, 0.) + scalar * len(current))
            reported_loss = loss
            if stationary_weighted is not None:
                reported_loss = reported_loss + stationary_weighted
                stationary_mean_weighted_total += (
                    float(stationary_weighted.detach()) * len(current))
            if stationary_variance_weighted is not None:
                reported_loss = reported_loss + stationary_variance_weighted
                stationary_variance_weighted_total += (
                    float(stationary_variance_weighted.detach()) * len(current))
            if not torch.isfinite(reported_loss):
                raise FloatingPointError(f"nonfinite {split} Neural-SDE loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                drift_parameters = tuple(
                    parameter for parameter in model.drift_model.parameters()
                    if parameter.requires_grad)
                diffusion_parameters = tuple(
                    parameter for parameter in model.diffusion_model.parameters()
                    if parameter.requires_grad)
                if drift_parameters and (
                        stationary_weighted is not None
                        or stationary_variance_weighted is not None):
                    drift_auxiliary = sum(
                        value for value in (stationary_weighted,
                                            stationary_variance_weighted)
                        if value is not None)
                    drift_gradients = torch.autograd.grad(
                        drift_auxiliary, drift_parameters,
                        retain_graph=bool(diffusion_parameters
                                          and stationary_variance_weighted is not None))
                    for parameter, gradient in zip(
                            drift_parameters, drift_gradients, strict=True):
                        if parameter.grad is None:
                            parameter.grad = gradient
                        else:
                            parameter.grad.add_(gradient)
                if stationary_variance_weighted is not None and diffusion_parameters:
                    variance_gradients = torch.autograd.grad(
                        stationary_variance_weighted, diffusion_parameters)
                    for parameter, gradient in zip(
                            diffusion_parameters, variance_gradients, strict=True):
                        if parameter.grad is None:
                            parameter.grad = gradient
                        else:
                            parameter.grad.add_(gradient)
                optimizer.step()
            total += float(reported_loss.detach()) * len(current)
            one_step_total += float(one_step.detach()) * len(current)
            for horizon, horizon_loss in horizon_losses.items():
                multistep_totals[horizon] += float(horizon_loss.detach()) * selected_count
            count += len(current)
    if not count:
        raise ValueError(f"no {split} transitions")
    if use_multistep and not multistep_count:
        raise ValueError(
            f"multistep sampling selected no {split} windows; increase its fraction")
    if use_stationary_moments and not stationary_mean_count:
        raise ValueError(
            f"stationary mean sampling selected no {split} trajectories; increase its fraction")
    if metric_output is not None:
        metric_output.update(total_loss=total / count,
                             one_step_nll=one_step_total / count,
                             window_count=count,
                             multistep_window_count=multistep_count)
        metric_output.update({f"multistep_nll_h{horizon}": value / multistep_count
                              for horizon, value in multistep_totals.items()})
        if use_stationary_moments:
            metric_output.update(
                stationary_mean_trajectory_count=stationary_mean_count,
                stationary_mean_batch_count=stationary_mean_batch_count)
        if use_stationary_mean:
            aggregate = stationary_mean_error_sum / stationary_mean_count
            metric_output.update(
                stationary_mean_loss=stationary_mean_total / stationary_mean_count,
                stationary_mean_weighted_contribution=(
                    stationary_mean_weighted_total / count),
                stationary_mean_normalized_error=aggregate.tolist(),
                stationary_mean_normalized_error_rmse=float(
                    torch.sqrt(torch.mean(aggregate.square()))))
        if use_stationary_variance:
            aggregate = stationary_variance_error_sum / stationary_mean_count
            metric_output.update(
                stationary_variance_loss=(
                    stationary_variance_total / stationary_mean_count),
                stationary_variance_weighted_contribution=(
                    stationary_variance_weighted_total / count),
                stationary_variance_normalized_error=aggregate.tolist(),
                stationary_variance_normalized_error_rmse=float(
                    torch.sqrt(torch.mean(aggregate.square()))))
        metric_output.update({f"stability_{key}": (
            value if key.endswith("max_excess") else value / count)
            for key, value in stability_totals.items()})
    return total / count


def _split_increment_norms(data, model, split, batch_size, state_std=None):
    # Match the float32 transition tensors consumed by the NLL.
    scale = np.asarray(
        model.state_std.detach().cpu() if state_std is None else state_std,
        dtype=np.float32)
    norms = []
    history_steps = 0 if model.config.state_variable == "velocity" else model.config.history_steps
    lag_steps = 1 if model.config.state_variable == "velocity" else model.config.lag_steps
    for _, batch in data.iter_transitions(split, lag_steps=lag_steps,
                                           history_steps=history_steps,
                                           batch_size=batch_size):
        current = np.asarray(batch["current_state"], dtype=np.float32)
        following = np.asarray(batch["next_state"], dtype=np.float32)
        delta = (following - current) / scale
        norms.append(np.linalg.norm(delta, axis=1))
    if not norms:
        raise ValueError(f"No {split} transitions for increment trimming.")
    return np.concatenate(norms)


def compute_training_trim_cutoff(data, model, trim_percent, batch_size):
    """Global training-only upper-tail cutoff in state-normalized increment norm."""
    if not 0 < trim_percent < 100:
        raise ValueError("Training trim percentage must lie in (0, 100).")
    norms = _split_increment_norms(data, model, "train", batch_size)
    cutoff = float(np.percentile(norms, 100 - trim_percent))
    removed = norms > cutoff
    total_transitions, removed_transitions = _split_trim_counts(
        data, model, "train", batch_size, cutoff)
    return cutoff, dict(
        train_trim_increment_count=len(norms),
        train_trim_removed_count=int(removed.sum()),
        train_trim_removed_percent=100 * float(removed.mean()),
        train_trim_transition_count=total_transitions,
        train_trim_removed_transition_count=removed_transitions,
        train_trim_removed_transition_percent=100 * removed_transitions / total_transitions,
        train_trim_norm_cutoff=cutoff)


def _fit_trimmed_sde_normalization(data, model, cutoff, state_std, batch_size):
    """Refit normalization on transitions retained by a frozen trim rule.

    State moments use retained current states. Target moments retain the
    existing SDE convention: lagged displacements for state/increment models
    and native one-frame velocities for velocity models.
    """
    states = RunningMoments(data.n_coordinates)
    targets = RunningMoments(data.n_coordinates)
    retained = 0
    for batch in _sde_transition_batches(data, model.config, "train", batch_size):
        keep = _trim_keep(batch, model, cutoff, state_std)
        if not bool(keep.any()):
            continue
        current = np.asarray(batch["current_state"][keep], dtype=np.float64)
        following = np.asarray(batch["next_state"][keep], dtype=np.float64)
        states.update(current)
        if model.config.state_variable == "velocity":
            history = np.asarray(batch["history_states"][keep, 0], dtype=np.float64)
            targets.update(current - history)
        else:
            targets.update(following - current)
        retained += len(current)
    state_mean, _, state_std, _ = states.finish()
    target_mean, _, target_std, _ = targets.finish()
    stats = NormalizationStats(*(torch.as_tensor(value, dtype=torch.float32)
                                 for value in (state_mean, state_std,
                                               target_mean, target_std)))
    return stats, retained


def _validation_change_batches(data, model, batch_size):
    """Yield physical SDE-variable changes and their conditioning batches."""
    lag = model.config.lag_steps
    if model.config.state_variable == "velocity":
        batches = _velocity_transition_arrays(data, model.config, "validation", batch_size)
    else:
        batches = (batch for _, batch in data.iter_transitions(
            "validation", lag_steps=lag, history_steps=model.config.history_steps,
            batch_size=batch_size))
    for batch in batches:
        delta = batch["next_state"] - batch["current_state"]
        if model.config.state_variable == "increment":
            delta = delta - (batch["current_state"] - batch["history_states"][:, 0])
        elif model.config.state_variable == "velocity":
            history = batch["history_states"]
            current_velocity = batch["current_state"] - history[:, 0]
            next_history = batch["next_history_states"]
            next_velocity = batch["next_state"] - next_history[:, 0]
            delta = next_velocity - current_velocity
        yield batch, delta


def compute_trimmed_increment_covariance(data, model, trim_percent, batch_size,
                                         coordinate_space="normalized"):
    """Validation covariance of SDE-variable changes, trimming only largest norms."""
    if not 0 <= trim_percent < 100 or coordinate_space not in {"normalized", "physical"}:
        raise ValueError("Invalid trim percentage or covariance coordinate space.")
    lag = model.config.lag_steps
    scale = (np.asarray(model._sde_scale().detach().cpu(), dtype=np.float64)
             if coordinate_space == "normalized" else 1.)

    def increments():
        for _, delta in _validation_change_batches(data, model, batch_size):
            yield delta / scale

    cutoff = None
    removed_energy_fraction = 0.
    if trim_percent:
        norms = np.concatenate([np.linalg.norm(delta, axis=1) for delta in increments()])
        cutoff = float(np.percentile(norms, 100 - trim_percent))
        removed = norms > cutoff
        total_energy = float(norms @ norms)
        removed_energy_fraction = (float(norms[removed] @ norms[removed]) / total_energy
                                   if total_energy > 0 else 0.)
    gram = np.zeros((data.n_coordinates, data.n_coordinates), dtype=np.float64)
    total_count = retained_count = 0
    for delta in increments():
        total_count += len(delta)
        if cutoff is not None:
            delta = delta[np.linalg.norm(delta, axis=1) <= cutoff]
        retained_count += len(delta)
        gram += delta.T @ delta
    if not retained_count:
        raise ValueError("No validation increments remain after diffusion trimming.")
    time_unit = 1. if model.config.state_variable == "velocity" else data.native_dt
    covariance = gram / (retained_count * lag * time_unit)
    removed_count = total_count - retained_count
    return covariance, dict(diffusion_validation_increment_count=total_count,
                            diffusion_trim_retained_count=retained_count,
                            diffusion_trim_removed_count=removed_count,
                            diffusion_trim_removed_percent=100 * removed_count / total_count,
                            diffusion_trim_removed_l2_energy_fraction=removed_energy_fraction,
                            diffusion_trim_norm_cutoff=cutoff)


def compute_diffusion_validation_metric(model, data, trim_percent, batch_size,
                                        coordinate_space="normalized"):
    """Compare mean learned G(x)G(x)^T with validation Q_trim."""
    empirical, report = compute_trimmed_increment_covariance(
        data, model, trim_percent, batch_size, coordinate_space)
    with torch.no_grad():
        time_unit = 1. if model.config.state_variable == "velocity" else data.native_dt
        if model.config.diffusion_type in {"state_diagonal", "bounded_state_diagonal"}:
            scale = (np.asarray(model._sde_scale().detach().cpu(), dtype=np.float64)
                     if coordinate_space == "normalized" else 1.)
            cutoff = report["diffusion_trim_norm_cutoff"]
            device = next(model.parameters()).device
            covariance_sum = torch.zeros((data.n_coordinates, data.n_coordinates),
                                         dtype=torch.float64, device=device)
            count = 0
            for batch, physical_delta in _validation_change_batches(data, model, batch_size):
                keep = np.ones(len(physical_delta), dtype=bool)
                if cutoff is not None:
                    keep = np.linalg.norm(physical_delta / scale, axis=1) <= cutoff
                if not keep.any():
                    continue
                current = torch.as_tensor(batch["current_state"][keep], dtype=torch.float32,
                                          device=device)
                history = batch.get("history_states")
                if history is not None:
                    history = torch.as_tensor(history[keep], dtype=torch.float32, device=device)
                factor = (model.normalized_diffusion_matrix(current, history)
                          if coordinate_space == "normalized" else
                          model.diffusion_matrix(current, history))
                covariance_sum += torch.einsum(
                    "bij,bkj->ik", factor.double(), factor.double())
                count += len(current)
            denominator = count * time_unit if coordinate_space == "normalized" else count
            model_covariance = np.asarray((covariance_sum / denominator).cpu())
        else:
            if coordinate_space == "normalized":
                g = np.asarray(model.normalized_diffusion_matrix().detach().cpu(), dtype=np.float64)
                model_covariance = g @ g.T / time_unit
            else:
                g = np.asarray(model.diffusion_matrix().detach().cpu(), dtype=np.float64)
                model_covariance = g @ g.T
    reference_norm = np.linalg.norm(empirical)
    report.update(diffusion_covariance_space=coordinate_space,
                  diffusion_covariance_target=(
                      "increment_change" if model.config.state_variable == "increment" else
                      "velocity_change" if model.config.state_variable == "velocity" else
                      "state_change"),
                  diffusion_covariance_relative_error=(float(np.linalg.norm(model_covariance - empirical)
                                                             / reference_norm) if reference_norm > 0 else None),
                  diffusion_model_trace=float(np.trace(model_covariance)),
                  diffusion_empirical_trace=float(np.trace(empirical)))
    return report


def train(data: SharedPODTrainingData, output: Path, config: SDEConfig, *,
          learning_rate: float = 1e-3, batch_size: int = 256, epochs: int = 20,
          seed: int = 0, device: torch.device | str = "cpu", train_mode: str = "joint",
          diffusion_trim_percent: float = 0., train_trim_percent: float = 0.,
          covariance_space: str = "normalized",
          multistep_weight: float = 0., multistep_horizons: tuple[int, ...] = (2, 4),
          multistep_particles: int = 100, multistep_validation_seed: int = 0,
          multistep_target: str = "velocity",
          multistep_mode_counts: tuple[int, ...] | None = None,
          multistep_particle_counts: tuple[int, ...] | None = None,
          multistep_window_fraction: float = 1.,
          multistep_window_sampling: str = "examples",
          stationary_mean_weight: float = 0.,
          stationary_variance_weight: float = 0.,
          stationary_mean_horizon: int = 100,
          stationary_mean_trajectories: int = 30,
          stationary_mean_batch_fraction: float = .05,
          stationary_mean_validation_seed: int = 0,
          stability: StabilityConfig | None = None,
          initial_checkpoint=None, initial_checkpoint_path: Path | None = None) -> Path:
    """Fit selected components; save best/latest by the validation objective."""
    if config.reduced_dim != data.n_coordinates or not math.isclose(config.dt, data.native_dt, rel_tol=1e-4):
        raise ValueError("SDE dimension and dt must match the prepared coordinates and native interval")
    if not data.splits["validation"]:
        raise ValueError("at least one validation repetition is required")
    if batch_size < 1 or epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive")
    if (train_mode not in {"joint", "diffusion", "drift"}
            or not 0 <= diffusion_trim_percent < 100 or not 0 <= train_trim_percent < 100):
        raise ValueError("Invalid train mode or diffusion trim percentage")
    if covariance_space not in {"normalized", "physical"}:
        raise ValueError("Invalid diffusion covariance space")
    multistep_horizons = tuple(multistep_horizons)
    if multistep_target not in {"velocity", "displacement"}:
        raise ValueError("Multistep target must be velocity or displacement")
    custom_particle_counts = multistep_particle_counts is not None
    multistep_mode_counts = _validated_multistep_mode_counts(
        multistep_mode_counts, multistep_horizons, data.rank, multistep_target)
    multistep_particle_counts = _validated_multistep_particle_counts(
        multistep_particle_counts, multistep_horizons, multistep_particles)
    if multistep_mode_counts is not None and multistep_weight <= 0:
        raise ValueError("Multistep mode counts require a positive multistep weight")
    if custom_particle_counts and multistep_weight <= 0:
        raise ValueError("Multistep particle counts require a positive multistep weight")
    if multistep_window_fraction < 1 and multistep_weight <= 0:
        raise ValueError("Multistep window subsampling requires a positive multistep weight")
    if multistep_window_sampling not in {"examples", "batches"}:
        raise ValueError("Multistep window sampling must be examples or batches")
    if (not math.isfinite(multistep_weight) or multistep_weight < 0
            or not multistep_horizons
            or any(not isinstance(horizon, int) or isinstance(horizon, bool)
                   or horizon < 1 for horizon in multistep_horizons)
            or tuple(sorted(set(multistep_horizons))) != multistep_horizons
            or not isinstance(multistep_particles, int) or isinstance(multistep_particles, bool)
            or multistep_particles < 1
            or not isinstance(multistep_validation_seed, int)
            or isinstance(multistep_validation_seed, bool)
            or not math.isfinite(multistep_window_fraction)
            or not 0 < multistep_window_fraction <= 1):
        raise ValueError("Invalid multistep weight, horizons, particles, or validation seed")
    if multistep_weight > 0 and config.state_variable != "velocity":
        raise ValueError("Multistep likelihood requires state_variable velocity")
    if (not math.isfinite(stationary_mean_weight) or stationary_mean_weight < 0
            or not math.isfinite(stationary_variance_weight)
            or stationary_variance_weight < 0
            or not isinstance(stationary_mean_horizon, int)
            or isinstance(stationary_mean_horizon, bool) or stationary_mean_horizon < 1
            or not isinstance(stationary_mean_trajectories, int)
            or isinstance(stationary_mean_trajectories, bool)
            or stationary_mean_trajectories < 2
            or not math.isfinite(stationary_mean_batch_fraction)
            or not 0 < stationary_mean_batch_fraction <= 1
            or not isinstance(stationary_mean_validation_seed, int)
            or isinstance(stationary_mean_validation_seed, bool)):
        raise ValueError("Invalid stationary moment weight, horizon, trajectories, fraction, or seed")
    if ((stationary_mean_weight > 0 or stationary_variance_weight > 0)
            and config.state_variable != "velocity"):
        raise ValueError("Stationary moment loss requires state_variable velocity")
    if stationary_mean_weight > 0 and train_mode == "diffusion":
        raise ValueError("Stationary mean loss requires trainable drift")
    stability = stability or StabilityConfig()
    if stability.weight > 0 and config.state_variable != "velocity":
        raise ValueError("Stability regularization requires state_variable velocity")
    if train_mode == "diffusion" and initial_checkpoint is not None:
        raise ValueError("Diffusion mode starts without a checkpoint")
    seed_everything(seed)
    device = torch.device(device)
    output = Path(output)
    if initial_checkpoint_path is not None and output.resolve() == Path(initial_checkpoint_path).resolve().parent:
        raise ValueError("Stage 2 output must differ from the Stage 1 checkpoint directory")
    output.mkdir(parents=True, exist_ok=True)
    data.cache_coordinates(2.)
    signature = data.signature()
    model = NeuralSDE(config)
    if initial_checkpoint is not None:
        previous = SDEConfig.from_dict(initial_checkpoint["model_config"])
        required_previous_mode = "diffusion" if train_mode == "drift" else "joint"
        if (initial_checkpoint.get("training", {}).get("train_mode") != required_previous_mode
                or config != replace(previous, lag_steps=config.lag_steps)
                or (config.lag_steps != previous.lag_steps
                    if config.state_variable == "increment"
                    else config.lag_steps < previous.lag_steps)
                or signature != initial_checkpoint["data_signature"]):
            raise ValueError(f"{train_mode} warm start requires a matching "
                             f"{required_previous_mode} checkpoint and "
                             f"{'the same lag' if config.state_variable == 'increment' else 'the same or a larger lag'}")
        model.load_state_dict(initial_checkpoint["model_state"])
        # Keep the state scaling used by both learned components; refresh only
        # lag-dependent increment statistics used by rollout diagnostics.
        stats = _fit_sde_normalization(data, config)
        model.target_mean.copy_(stats.target_mean)
        model.target_std.copy_(stats.target_std)
    else:
        stats = _fit_sde_normalization(data, config)
        model.set_normalization(stats)
    model.to(device)
    trim_cutoff = trim_report = trim_state_std = None
    if train_trim_percent:
        # Freeze the preliminary scale together with the cutoff. Normalization
        # is refitted below, but must not silently change which pairs are kept.
        previous_training = (initial_checkpoint or {}).get("training", {})
        reuse_trim = (
            initial_checkpoint is not None
            and math.isclose(float(previous_training.get("train_trim_percent", 0.)),
                             train_trim_percent)
            and previous_training.get("train_trim_norm_cutoff") is not None
            and previous_training.get("train_trim_state_std") is not None)
        if reuse_trim:
            trim_cutoff = float(previous_training["train_trim_norm_cutoff"])
            trim_state_std = torch.as_tensor(
                previous_training["train_trim_state_std"], dtype=torch.float32).clone()
            training_norms = _split_increment_norms(
                data, model, "train", batch_size, trim_state_std)
            training_removed = training_norms > trim_cutoff
            training_transitions, training_removed_transitions = _split_trim_counts(
                data, model, "train", batch_size, trim_cutoff, trim_state_std)
            trim_report = dict(
                train_trim_increment_count=len(training_norms),
                train_trim_removed_count=int(training_removed.sum()),
                train_trim_removed_percent=100 * float(training_removed.mean()),
                train_trim_transition_count=training_transitions,
                train_trim_removed_transition_count=training_removed_transitions,
                train_trim_removed_transition_percent=(
                    100 * training_removed_transitions / training_transitions),
                train_trim_norm_cutoff=trim_cutoff,
                train_trim_rule_source="checkpoint")
        else:
            trim_state_std = model.state_std.detach().cpu().clone()
            trim_cutoff, trim_report = compute_training_trim_cutoff(
                data, model, train_trim_percent, batch_size)
            trim_report["train_trim_rule_source"] = "recomputed"
        validation_norms = _split_increment_norms(
            data, model, "validation", batch_size, trim_state_std)
        validation_removed = validation_norms > trim_cutoff
        validation_transitions, validation_removed_transitions = _split_trim_counts(
            data, model, "validation", batch_size, trim_cutoff, trim_state_std)
        trim_report.update(
            validation_trim_increment_count=len(validation_norms),
            validation_trim_removed_count=int(validation_removed.sum()),
            validation_trim_removed_percent=100 * float(validation_removed.mean()),
            validation_trim_transition_count=validation_transitions,
            validation_trim_removed_transition_count=validation_removed_transitions,
            validation_trim_removed_transition_percent=(
                100 * validation_removed_transitions / validation_transitions),
        )
        trimmed_stats, retained = _fit_trimmed_sde_normalization(
            data, model, trim_cutoff, trim_state_std, batch_size)
        model.set_normalization(trimmed_stats)
        trim_report.update(
            train_trim_state_std=trim_state_std.tolist(),
            normalization_fit="retained_training_transitions",
            normalization_retained_transition_count=retained,
        )
        print(f"Training increment trim: removed {trim_report['train_trim_removed_count']:,}/"
              f"{trim_report['train_trim_increment_count']:,} pairs "
              f"({trim_report['train_trim_removed_percent']:.3g}%), "
              f"normalized norm > {trim_cutoff:.6g}", flush=True)
        print(f"Validation increment trim: removed {trim_report['validation_trim_removed_count']:,}/"
              f"{trim_report['validation_trim_increment_count']:,} pairs "
              f"({trim_report['validation_trim_removed_percent']:.3g}%) using the training cutoff",
              flush=True)
        if config.state_variable == "velocity":
            print(f"Velocity-path trim: removed "
                  f"{trim_report['train_trim_removed_transition_count']:,}/"
                  f"{trim_report['train_trim_transition_count']:,} training pairs and "
                  f"{trim_report['validation_trim_removed_transition_count']:,}/"
                  f"{trim_report['validation_trim_transition_count']:,} validation pairs",
                  flush=True)
        if multistep_weight > 0:
            for split_name in ("train", "validation"):
                window_count, removed_window_count = _split_trim_counts(
                    data, model, split_name, batch_size, trim_cutoff, trim_state_std,
                    multistep_horizons)
                trim_report.update({
                    f"{split_name}_multistep_window_count": window_count,
                    f"{split_name}_multistep_removed_window_count": removed_window_count,
                    f"{split_name}_multistep_removed_window_percent": (
                        100 * removed_window_count / window_count),
                })
    capillary_report = _configure_capillary_energy(model, data)
    if (config.drift_type == "capillary_energy" and initial_checkpoint is not None
            and initial_checkpoint.get("training", {}).get("train_mode") == "diffusion"):
        model.drift_model.reset_structured_terms(config.stiffness_init, config.damping_init)
    if train_mode == "diffusion":
        # The saved Stage 1 checkpoint should also roll out with zero drift.
        model.drift_model.zero_output()
    model_step = (f"{config.lag_steps:g} frames" if config.state_variable == "velocity" else
                  f"{config.lag_steps * config.dt:.9g} s")
    print(f"train_mode={train_mode} state_variable={config.state_variable} "
          f"lag_steps={config.lag_steps} h={model_step}", flush=True)
    trainable = set_trainable_components(model, train_mode)
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    objective = EulerMaruyamaNLL()
    multistep_objective = (MonteCarloMultistepNLL(
        multistep_horizons, multistep_particles, target=multistep_target,
        mode_counts=multistep_mode_counts,
        pod_coordinate_offset=int(data.include_spatial_mean),
        particle_counts=multistep_particle_counts)
        if multistep_weight > 0 else None)
    if stationary_mean_weight > 0 or stationary_variance_weight > 0:
        stationary_mean_objective = StationaryMeanLoss(stationary_mean_horizon).to(device)
        print(
            f"Endpoint moment target: matching observed h={stationary_mean_horizon} "
            "endpoints; each retained start produces two independently noised model "
            "paths, with at most one window per shuffled source block.", flush=True)
    else:
        stationary_mean_objective = None
    stability_objective = None
    if stability.weight > 0:
        def retained_training_states():
            for batch in _sde_transition_batches(
                    data, config, "train", batch_size,
                    multistep_horizons=(multistep_horizons if multistep_weight > 0 else None)):
                if trim_cutoff is not None:
                    keep = _trim_keep(batch, model, trim_cutoff, trim_state_std)
                    batch = {key: value[keep] for key, value in batch.items()}
                if len(batch["current_state"]):
                    yield batch["current_state"], batch["history_states"]

        stability_objective = fit_stability_regularizer(
            model, stability, retained_training_states())
        print(f"Stability metric={stability_objective.reference['kind']} "
              f"initial-core spectral radius={stability_objective.reference.get('spectral_radius')} "
              f"training mean V=1; weight={stability.weight:g}, a={stability.a:g}, b={stability.b:g}. "
              "Sampled penalties do not prove global stability; V omits older history.", flush=True)
    metadata = dict(data=data.source_config, model=config.to_dict(),
                    training=dict(learning_rate=learning_rate, batch_size=batch_size,
                                  epochs=epochs, seed=seed, train_mode=train_mode,
                                  diffusion_trim_percent=diffusion_trim_percent,
                                  train_trim_percent=train_trim_percent,
                                  multistep_weight=multistep_weight,
                                  multistep_target=multistep_target,
                                  multistep_horizons=list(multistep_horizons),
                                  multistep_mode_counts=(
                                      None if multistep_mode_counts is None else
                                      list(multistep_mode_counts)),
                                  multistep_particles=multistep_particles,
                                  multistep_particle_counts=list(multistep_particle_counts),
                                  multistep_window_fraction=multistep_window_fraction,
                                  multistep_window_sampling=multistep_window_sampling,
                                  multistep_validation_seed=multistep_validation_seed,
                                  stationary_mean_weight=stationary_mean_weight,
                                  stationary_variance_weight=stationary_variance_weight,
                                  stationary_mean_horizon=stationary_mean_horizon,
                                  stationary_mean_trajectories=stationary_mean_trajectories,
                                  stationary_mean_batch_fraction=(
                                      stationary_mean_batch_fraction),
                                  stationary_mean_validation_seed=(
                                      stationary_mean_validation_seed),
                                  stationary_mean_gradient_parameters=(
                                      "mean: drift only; variance: every trainable drift and "
                                      "diffusion parameter"),
                                  stationary_mean_start_population=(
                                      "globally mixed velocity windows whose complete measured "
                                      "path through the endpoint passes training trimming; at most "
                                      "one window per shuffled source block"),
                                  stationary_mean_target_population=(
                                      "matching measured endpoint at the configured horizon"),
                                  stationary_mean_generated_paths_per_start=2,
                                  stationary_mean_source_block_size=min(
                                      256, data.batch_size, batch_size),
                                  stationary_mean_definition=(
                                      "coordinate mean of the cross-product between two independent "
                                      "normalized generated-minus-measured endpoint mean errors; "
                                      "unbiased over rollout noise for the sampled endpoint batch and "
                                      "finite-batch values may be negative"),
                                  stationary_variance_definition=(
                                      "coordinate mean of the cross-product between two independent "
                                      "generated-minus-measured normalized endpoint sample-variance "
                                      "errors; sample variances use N-1 and finite-batch values may "
                                      "be negative"),
                                  **(trim_report or {}),
                                  diffusion_covariance_space=covariance_space,
                                  capillary_energy=capillary_report,
                                  initial_checkpoint=(str(Path(initial_checkpoint_path).resolve())
                                                      if initial_checkpoint_path else None)),
                    split_labels={k: list(v) for k, v in data.splits.items()})
    if stability_objective is not None:
        metadata["training"]["stability"] = stability_objective.to_metadata()
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best = float("inf")
    for epoch in range(epochs):
        train_metrics, validation_metrics = {}, {}
        train_loss = _epoch(
            model, data, "train", batch_size, device, objective, optimizer,
            seed=seed + epoch, train_mode=train_mode,
            train_trim_cutoff=trim_cutoff,
            train_trim_state_std=trim_state_std,
            multistep_objective=multistep_objective,
            multistep_weight=multistep_weight,
            multistep_window_fraction=multistep_window_fraction,
            multistep_window_sampling=multistep_window_sampling,
            multistep_window_seed=seed + epoch,
            stationary_mean_objective=stationary_mean_objective,
            stationary_mean_weight=stationary_mean_weight,
            stationary_variance_weight=stationary_variance_weight,
            stationary_mean_trajectories=stationary_mean_trajectories,
            stationary_mean_batch_fraction=stationary_mean_batch_fraction,
            stationary_mean_noise_seed=seed + epoch,
            stationary_mean_window_seed=seed + epoch,
            stability_objective=stability_objective,
            stability_noise_seed=seed + epoch,
            metric_output=train_metrics)
        validation_loss = _epoch(
            model, data, "validation", batch_size, device, objective,
            train_mode=train_mode, train_trim_cutoff=trim_cutoff,
            train_trim_state_std=trim_state_std,
            multistep_objective=multistep_objective,
            multistep_weight=multistep_weight,
            multistep_noise_seed=(multistep_validation_seed
                                  if multistep_weight > 0 else None),
            multistep_window_fraction=multistep_window_fraction,
            multistep_window_sampling=multistep_window_sampling,
            multistep_window_seed=multistep_validation_seed,
            stationary_mean_objective=stationary_mean_objective,
            stationary_mean_weight=stationary_mean_weight,
            stationary_variance_weight=stationary_variance_weight,
            stationary_mean_trajectories=stationary_mean_trajectories,
            stationary_mean_batch_fraction=stationary_mean_batch_fraction,
            stationary_mean_noise_seed=stationary_mean_validation_seed,
            stationary_mean_window_seed=stationary_mean_validation_seed,
            stability_objective=stability_objective,
            stability_noise_seed=stability.validation_seed,
            metric_output=validation_metrics)
        record = dict(epoch=epoch + 1,
                      train_nll=train_metrics["one_step_nll"],
                      validation_nll=validation_metrics["one_step_nll"])
        if stability_objective is not None:
            record.update(train_total_loss=train_loss, validation_total_loss=validation_loss)
            for prefix, values in (("train", train_metrics), ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value for key, value in values.items()
                               if key.startswith("stability_")})
        if multistep_weight > 0:
            record.update(train_total_loss=train_loss,
                          validation_total_loss=validation_loss,
                          multistep_target=multistep_target,
                          multistep_mode_counts=(
                              None if multistep_mode_counts is None else
                              list(multistep_mode_counts)),
                          multistep_particle_counts=list(multistep_particle_counts),
                          multistep_window_fraction=multistep_window_fraction,
                          multistep_window_sampling=multistep_window_sampling,
                          train_multistep_windows_available=train_metrics["window_count"],
                          validation_multistep_windows_available=(
                              validation_metrics["window_count"]),
                          train_multistep_windows_scored=(
                              train_metrics["multistep_window_count"]),
                          validation_multistep_windows_scored=(
                              validation_metrics["multistep_window_count"]))
            record.update({f"train_{key}": value for key, value in train_metrics.items()
                           if key.startswith("multistep_nll_")})
            record.update({f"validation_{key}": value
                           for key, value in validation_metrics.items()
                           if key.startswith("multistep_nll_")})
        if stationary_mean_weight > 0 or stationary_variance_weight > 0:
            record.update(
                train_total_loss=train_loss,
                validation_total_loss=validation_loss,
                stationary_mean_weight=stationary_mean_weight,
                stationary_variance_weight=stationary_variance_weight,
                stationary_mean_horizon=stationary_mean_horizon,
                stationary_mean_trajectories=stationary_mean_trajectories,
                stationary_mean_batch_fraction=stationary_mean_batch_fraction,
            )
            for prefix, values in (("train", train_metrics),
                                   ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value for key, value in values.items()
                               if (key.startswith("stationary_mean_")
                                   or key.startswith("stationary_variance_"))})
        if train_mode == "diffusion":
            record.update(compute_diffusion_validation_metric(
                model, data, diffusion_trim_percent, batch_size, covariance_space))
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        message = (f"epoch={epoch + 1} train_nll={record['train_nll']:.5g} "
                   f"validation_nll={record['validation_nll']:.5g}")
        if stability_objective is not None:
            for prefix in ("train", "validation"):
                message += (f" {prefix}_stability={record[f'{prefix}_stability_penalty']:.5g}"
                            f" weighted={record[f'{prefix}_stability_weighted_penalty']:.5g}"
                            f" violations={record[f'{prefix}_stability_violation_fraction']:.2%}")
        if multistep_weight > 0:
            message += (f" train_total={train_loss:.5g}"
                        f" validation_total={validation_loss:.5g}")
            for horizon in multistep_horizons:
                message += (f" train_h{horizon}="
                            f"{record[f'train_multistep_nll_h{horizon}']:.5g}"
                            f" validation_h{horizon}="
                            f"{record[f'validation_multistep_nll_h{horizon}']:.5g}")
        if stationary_mean_weight > 0:
            message += (
                f" train_mean_h{stationary_mean_horizon}="
                f"{record['train_stationary_mean_loss']:.5g}"
                f" validation_mean_h{stationary_mean_horizon}="
                f"{record['validation_stationary_mean_loss']:.5g}"
                f" train_mean_weighted="
                f"{record['train_stationary_mean_weighted_contribution']:.5g}"
                f" validation_mean_weighted="
                f"{record['validation_stationary_mean_weighted_contribution']:.5g}"
                f" train_mean_rmse="
                f"{record['train_stationary_mean_normalized_error_rmse']:.5g}"
                f" validation_mean_rmse="
                f"{record['validation_stationary_mean_normalized_error_rmse']:.5g}")
        if stationary_variance_weight > 0:
            message += (
                f" train_var_h{stationary_mean_horizon}="
                f"{record['train_stationary_variance_loss']:.5g}"
                f" validation_var_h{stationary_mean_horizon}="
                f"{record['validation_stationary_variance_loss']:.5g}"
                f" train_var_weighted="
                f"{record['train_stationary_variance_weighted_contribution']:.5g}"
                f" validation_var_weighted="
                f"{record['validation_stationary_variance_weighted_contribution']:.5g}"
                f" train_var_rmse="
                f"{record['train_stationary_variance_normalized_error_rmse']:.5g}"
                f" validation_var_rmse="
                f"{record['validation_stationary_variance_normalized_error_rmse']:.5g}")
        if train_mode == "diffusion":
            message += (f" E_diff={record['diffusion_covariance_relative_error']:.5g}"
                        if record["diffusion_covariance_relative_error"] is not None else " E_diff=undefined")
            message += (f" tr_A={record['diffusion_model_trace']:.5g}"
                        f" tr_Q={record['diffusion_empirical_trace']:.5g}"
                        f" cut={record['diffusion_trim_removed_count']}"
                        f" ({record['diffusion_trim_removed_percent']:.2f}%)"
                        f" cut_L2_energy={record['diffusion_trim_removed_l2_energy_fraction']:.2%}")
        print(message, flush=True)
        payload = dict(model_config=config.to_dict(), data_config=data.source_config,
                       data_signature=signature, model_state=model.state_dict(),
                       epoch=epoch + 1, training=metadata["training"])
        save_checkpoint(output / "latest.pt", payload)
        if validation_loss < best:
            best = validation_loss
            save_checkpoint(output / "best.pt", payload)
    return output / "best.pt"


def load_model(path: Path, device: torch.device | str = "cpu"):
    checkpoint = load_checkpoint(path, device)
    model = NeuralSDE(SDEConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def evaluate(checkpoint_path: Path, output: Path, *, split="test", num_conditions=32,
             ensemble_size=16, horizon=50, batch_size=8, seed=0, device="cpu",
             data_root=None, shared_basis=None, metrics=True, rollout_lag=None,
             enforce_train_trim_on_reference_windows=False, diffusion_scale=1.,
             multistep_horizons: tuple[int, ...] | None = None,
             multistep_particles: int = 100, multistep_seed: int = 0,
             multistep_max_windows: int | None = None,
             multistep_window_seed: int = 0,
             multistep_target: str | None = None,
             multistep_mode_counts: tuple[int, ...] | None = None,
             multistep_particle_counts: tuple[int, ...] | None = None):
    """Save the standard EDM-compatible rollout.npz and metrics.json."""
    if not math.isfinite(diffusion_scale) or diffusion_scale < 0:
        raise ValueError("diffusion_scale must be finite and nonnegative")
    if multistep_horizons is not None:
        multistep_horizons = tuple(multistep_horizons)
        if (not multistep_horizons
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                       for value in multistep_horizons)
                or tuple(sorted(set(multistep_horizons))) != multistep_horizons
                or not isinstance(multistep_particles, int)
                or isinstance(multistep_particles, bool) or multistep_particles < 1
                or not isinstance(multistep_seed, int) or isinstance(multistep_seed, bool)):
            raise ValueError("Invalid multistep evaluation horizons, particles, or seed")
    elif multistep_mode_counts is not None:
        raise ValueError("Multistep mode counts require multistep horizons")
    if multistep_horizons is None and multistep_particle_counts is not None:
        raise ValueError("Multistep particle counts require multistep horizons")
    if (multistep_max_windows is not None
            and (multistep_horizons is None
                 or not isinstance(multistep_max_windows, int)
                 or isinstance(multistep_max_windows, bool)
                 or multistep_max_windows < 1)):
        raise ValueError("Multistep max windows requires a positive count and horizons")
    if not isinstance(multistep_window_seed, int) or isinstance(multistep_window_seed, bool):
        raise ValueError("Multistep window seed must be an integer")
    model, checkpoint = load_model(checkpoint_path, device)
    if multistep_target is None:
        multistep_target = checkpoint.get("training", {}).get("multistep_target", "velocity")
    if multistep_target not in {"velocity", "displacement"}:
        raise ValueError("Multistep target must be velocity or displacement")
    training_lag = model.config.lag_steps
    if rollout_lag is not None:
        if rollout_lag < 1:
            raise ValueError("rollout_lag must be positive")
        if model.config.state_variable == "increment" and rollout_lag != training_lag:
            raise ValueError("Increment-state rollouts require the checkpoint training lag")
        model.config = replace(model.config, lag_steps=rollout_lag)
    data = checkpoint_data(checkpoint, data_root=data_root, shared_basis=shared_basis)
    if not math.isclose(model.config.dt, data.native_dt, rel_tol=1e-4):
        raise ValueError("checkpoint dt differs from source timestamps")
    metric_stats = None
    if metrics and model.config.state_variable == "velocity":
        metric_stats = fit_normalization(data, _edm_transition_config(
            data.n_coordinates, model.config.lag_steps, 0))
    elif metrics and model.config.lag_steps != training_lag:
        stats = fit_normalization(data, _edm_transition_config(
            data.n_coordinates, model.config.lag_steps, model.config.history_steps))
        model.target_mean.copy_(stats.target_mean)
        model.target_std.copy_(stats.target_std)
    training = checkpoint.get("training", {})
    if multistep_horizons is not None:
        multistep_particle_counts = _validated_multistep_particle_counts(
            multistep_particle_counts, multistep_horizons, multistep_particles)
        if multistep_mode_counts is None:
            saved_counts = training.get("multistep_mode_counts")
            if saved_counts is not None:
                if list(multistep_horizons) != training.get("multistep_horizons"):
                    raise ValueError(
                        "Checkpoint mode counts can only be inherited at its saved horizons; "
                        "supply --multistep-mode-counts for different horizons")
                multistep_mode_counts = tuple(saved_counts)
        multistep_mode_counts = _validated_multistep_mode_counts(
            multistep_mode_counts, multistep_horizons, data.rank, multistep_target)
    trim_cutoff = training.get("train_trim_norm_cutoff")
    trim_state_std = training.get("train_trim_state_std")
    if enforce_train_trim_on_reference_windows and trim_cutoff is None:
        raise ValueError("Reference-window trim enforcement requires a training-trimmed checkpoint")
    if (enforce_train_trim_on_reference_windows and trim_cutoff is not None
            and model.config.lag_steps != training_lag
            and model.config.state_variable != "velocity"):
        raise ValueError("Training-trimmed rollout evaluation requires the checkpoint training lag")
    applied_cutoff = trim_cutoff if enforce_train_trim_on_reference_windows else None
    if trim_cutoff is not None and not enforce_train_trim_on_reference_windows:
        print("WARNING: reference windows may contain increments above the saved training cutoff; "
              "energy and PSD comparisons require caution", file=sys.stderr, flush=True)
    result, report = evaluate_model(model, data, split=split, num_conditions=num_conditions,
                                    ensemble_size=ensemble_size, horizon=horizon,
                                    batch_size=batch_size, seed=seed, metrics=metrics,
                                    reference_increment_norm_cutoff=applied_cutoff,
                                    reference_increment_norm_scale=trim_state_std,
                                    metric_target_mean=(None if metric_stats is None
                                                        else metric_stats.target_mean),
                                    metric_target_std=(None if metric_stats is None
                                                       else metric_stats.target_std),
                                    rollout_options={"diffusion_scale": diffusion_scale})
    report["reference_window_trim_policy"] = dict(
        training_trim_available=trim_cutoff is not None,
        enforced=bool(enforce_train_trim_on_reference_windows),
        train_trim_percent=float(training.get("train_trim_percent", 0.)),
        normalized_increment_norm_cutoff=trim_cutoff,
        normalization_scale_source=(None if trim_cutoff is None else
                                    "saved_pretrim_state_std"
                                    if trim_state_std is not None else
                                    "checkpoint_state_std_legacy"),
        generated_trajectories_trimmed=False,
        caution=(None if enforce_train_trim_on_reference_windows or trim_cutoff is None else
                 "reference windows may contain increments excluded from training"),
    )
    report.update(source_shared_basis=str(data.basis_path), rank=data.rank,
                  spatial_mean_highpass_hz=data.spatial_mean_highpass_hz,
                  training_lag_steps=training_lag,
                  state_variable=model.config.state_variable,
                  drift_type=model.config.drift_type,
                  damped_operator=(model.config.damped_operator
                                   if model.config.drift_type == "damped_residual" else None),
                  diffusion_type=model.config.diffusion_type,
                  diffusion_scale=diffusion_scale,
                  diffusion_log_range=(model.config.diffusion_log_range
                                       if model.config.diffusion_type == "bounded_state_diagonal"
                                       else None),
                  history_steps=model.config.history_steps,
                  history_offsets=list(model.config.effective_history_offsets),
                  train_trim_percent=training.get("train_trim_percent", 0.),
                  train_trim_norm_cutoff=trim_cutoff,
                  checkpoint=str(Path(checkpoint_path).resolve()), model_type="neural_sde")
    if multistep_horizons is not None:
        if model.config.state_variable != "velocity":
            raise ValueError("Multistep likelihood evaluation requires a velocity checkpoint")
        likelihood_metrics = {}
        multistep_objective = MonteCarloMultistepNLL(
            multistep_horizons, multistep_particles, target=multistep_target,
            mode_counts=multistep_mode_counts,
            pod_coordinate_offset=int(data.include_spatial_mean),
            particle_counts=multistep_particle_counts)
        _epoch(
            model, data, split, batch_size, device, EulerMaruyamaNLL(),
            train_trim_cutoff=trim_cutoff,
            train_trim_state_std=trim_state_std,
            multistep_objective=multistep_objective, multistep_weight=1.,
            multistep_noise_seed=multistep_seed,
            metric_output=likelihood_metrics,
            multistep_max_windows=multistep_max_windows,
            multistep_window_seed=multistep_window_seed)
        report["multistep_likelihood"] = dict(
            split=split, horizons=list(multistep_horizons),
            target=multistep_target,
            mode_counts=(None if multistep_mode_counts is None else
                         list(multistep_mode_counts)),
            coordinate_selection=(
                "all reduced coordinates" if multistep_mode_counts is None else
                "first N shared-POD modes at each horizon; spatial mean excluded"),
            target_definition=("a[n + H*lag_steps] - a[n]" if multistep_target == "displacement"
                               else "a[n + H*lag_steps] - a[n + H*lag_steps - 1]"),
            particles=multistep_particles, seed=multistep_seed,
            particle_counts=list(multistep_particle_counts),
            max_windows=multistep_max_windows,
            window_seed=multistep_window_seed,
            windows_scored=likelihood_metrics["window_count"],
            lag_steps=model.config.lag_steps,
            training_trim_applied=trim_cutoff is not None,
            normalized_increment_norm_cutoff=trim_cutoff,
            trim_rule=("every native increment from required prehistory through the farthest endpoint"
                       if trim_cutoff is not None else None),
            trim_interpretation=("heuristic amplitude cutoff; not a discontinuity detector"
                                 if trim_cutoff is not None else "untrimmed checkpoint"),
            one_step_nll=likelihood_metrics["one_step_nll"],
            mean_multistep_nll=float(np.mean([
                likelihood_metrics[f"multistep_nll_h{value}"]
                for value in multistep_horizons])),
            per_horizon_nll={str(value): likelihood_metrics[f"multistep_nll_h{value}"]
                             for value in multistep_horizons},
        )
    if model.config.drift_type == "damped_residual":
        report.update(
            stiffness_diagonal=model.drift_model.stiffness_diagonal().detach().cpu().tolist(),
            damping_diagonal=model.drift_model.damping_diagonal().detach().cpu().tolist(),
            structured_drift_coordinate_space="standardized_position_and_velocity",
        )
        if model.config.damped_operator == "spsd":
            stiffness = model.drift_model.stiffness_matrix().detach().cpu()
            damping = model.drift_model.damping_matrix().detach().cpu()
            report.update(
                stiffness_matrix=stiffness.tolist(),
                damping_matrix=damping.tolist(),
                stiffness_eigenvalues=torch.linalg.eigvalsh(stiffness).tolist(),
                damping_eigenvalues=torch.linalg.eigvalsh(damping).tolist(),
            )
    elif model.config.drift_type == "capillary_energy":
        energy_matrix = model.drift_model.energy_matrix.detach().cpu()
        damping = model.drift_model.damping_matrix().detach().cpu()
        mass = model.drift_model.mass_diagonal().detach().cpu()
        report.update(
            capillary_energy_matrix=energy_matrix.tolist(),
            capillary_energy_matrix_eigenvalues=torch.linalg.eigvalsh(energy_matrix).tolist(),
            effective_mass_diagonal=mass.tolist(),
            damping_matrix=damping.tolist(),
            damping_eigenvalues=torch.linalg.eigvalsh(damping).tolist(),
            mean_mode_stiffness=(float(model.drift_model.mean_stiffness().detach().cpu())
                                 if model.config.capillary_mean_mode else None),
            surface_tension_N_per_m=model.config.surface_tension,
            structured_drift_coordinate_space="standardized position and native-frame increment",
            structured_drift_time_unit="one native frame",
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "rollout.npz", **result)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m modelling.neural_sde")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="fit joint, diffusion-only, or drift-only Euler-Maruyama NLL")
    training.add_argument("--experiment", help="Required when starting without a checkpoint.")
    training.add_argument("--rank", type=int, help="Required when starting without a checkpoint.")
    training.add_argument("--train-mode", choices=("joint", "diffusion", "drift"), default="joint")
    training.add_argument("--lag", "--lag-steps", dest="lag_steps", type=int, default=1,
                          help="Native sampling steps per model transition (default 1).")
    training.add_argument("--state-variable", choices=("state", "increment", "velocity"),
                          help="SDE variable: state (default), fixed-lag increment, or native-frame velocity.")
    history = training.add_mutually_exclusive_group()
    history.add_argument("--history", type=int,
                         help="Contiguous preceding states (default 0 for state; increment and velocity default to 1).")
    history.add_argument(
        "--history-offsets", type=int, nargs="+",
        help="Sparse preceding-state offsets in model transitions, for example 1 2 3 5 10.",
    )
    training.add_argument("--checkpoint", type=Path,
                          help="Prior run directory or best.pt; optional for joint or drift training.")
    training.add_argument("--diffusion-trim-percent", type=float, default=0.,
                          help="Remove this largest percentage of validation increment norms for E_diff (default 0).")
    training.add_argument("--train-trim-percent", type=float, default=0.,
                          help="Exclude this largest percentage of training increment norms from NLL (default 0).")
    training.add_argument(
        "--multistep-weight", type=float, default=0.,
        help="Weight of the Monte Carlo multistep NLL (default 0: disabled).",
    )
    training.add_argument(
        "--multistep-target", choices=("velocity", "displacement"), default="velocity",
        help="Score endpoint velocity (default) or displacement from the initial position; "
             "used only when --multistep-weight is positive.",
    )
    training.add_argument(
        "--multistep-horizons", type=int, nargs="+", default=(2, 4),
        help="Strictly increasing endpoint horizons in model steps (default 2 4).",
    )
    training.add_argument(
        "--multistep-mode-counts", type=int, nargs="+",
        help="First shared-POD modes scored at each matching multistep horizon; "
             "the spatial-mean coordinate is excluded. Default: score all coordinates.",
    )
    training.add_argument(
        "--multistep-particles", type=int, default=100,
        help="Reparameterized latent paths per observed starting state (default 100).",
    )
    training.add_argument(
        "--multistep-particle-counts", type=int, nargs="+",
        help="Optional nonincreasing particle count at each matching horizon; paths are "
             "pruned after a scored horizon. Overrides the scalar count schedule.",
    )
    training.add_argument(
        "--multistep-window-fraction", type=float, default=1.,
        help="Uniform fraction of every retained batch receiving the multistep term; "
             "the one-step NLL still uses the full batch (default 1).",
    )
    training.add_argument(
        "--multistep-window-sampling", choices=("examples", "batches"), default="examples",
        help="Subsample examples inside every batch, or score full batches less often with "
             "an unbiased inverse-fraction correction (default examples).",
    )
    training.add_argument(
        "--multistep-validation-seed", type=int, default=0,
        help="Fixed Monte Carlo seed used for every validation epoch (default 0).",
    )
    training.add_argument(
        "--stationary-mean-weight", type=float, default=0.,
        help="Weight of the independent rollout-endpoint mean match to trimmed training data "
             "(default 0: disabled).",
    )
    training.add_argument(
        "--stationary-variance-weight", type=float, default=0.,
        help="Weight of the independent diagonal endpoint-variance match to measured "
             "data (default 0: disabled).",
    )
    training.add_argument(
        "--stationary-mean-horizon", type=int, default=100,
        help="Model steps in each stationary-mean rollout (default 100).",
    )
    training.add_argument(
        "--stationary-mean-trajectories", type=int, default=30,
        help="Horizon-clean observed starts per selected batch; two independently "
             "noised model paths are generated per start (default 30).",
    )
    training.add_argument(
        "--stationary-mean-batch-fraction", type=float, default=.05,
        help="Fraction of batches receiving the stationary-mean term; selected "
             "updates use inverse-fraction correction (default 0.05).",
    )
    training.add_argument(
        "--stationary-mean-validation-seed", type=int, default=0,
        help="Fixed start/noise seed for stationary-mean validation (default 0).",
    )
    training.add_argument("--stability-weight", type=float, default=0.,
                          help="Discrete Lyapunov penalty weight; 0 disables all stability work.")
    training.add_argument("--stability-a", type=float, default=.01,
                          help="Fixed nonnegative allowance in Delta V <= a-b V (default .01).")
    training.add_argument("--stability-b", type=float, default=.01,
                          help="Fixed contraction fraction, strictly between 0 and 1 (default .01).")
    training.add_argument("--stability-metric", choices=("auto", "lyapunov", "data_scaled"),
                          default="auto", help="Freeze initial-core Lyapunov P, or empirical data-scaled P.")
    training.add_argument("--stability-perturb-scale", type=float, default=0.,
                          help="Add a history-consistent perturbed copy, in training std units.")
    training.add_argument("--stability-generated-steps", type=int, default=0,
                          help="Also penalize each of this many generated steps from observed states.")
    training.add_argument("--stability-validation-seed", type=int, default=0,
                          help="Fixed validation state-sampling seed (independent of NLL sampling).")
    training.add_argument("--diffusion-covariance-space", choices=("normalized", "physical"),
                          default="normalized", help="Coordinate space for E_diff (default normalized).")
    training.add_argument("--data-root", type=Path)
    training.add_argument("--shared-basis", type=Path)
    training.add_argument("--validation-reps", type=int, nargs="+")
    training.add_argument("--test-reps", type=int, nargs="*")
    training.add_argument("--no-spatial-mean", action="store_true")
    training.add_argument(
        "--spatial-mean-highpass-hz", type=float,
        help="Use the matching high-pass spatial-mean trajectory stored in each POD file; "
             "evaluation inherits this choice from the checkpoint.",
    )
    training.add_argument("--hidden-layers", type=int, nargs="+", help="Default 64 64 for a new model.")
    training.add_argument("--activation", choices=("silu", "gelu", "relu", "tanh"),
                          help="Default silu for a new model.")
    training.add_argument(
        "--drift-type", choices=("mlp", "damped_residual", "capillary_energy"),
        help="Drift architecture: unconstrained MLP (default), learned damped residual, "
             "or fixed capillary-energy restoring operator with learned mass/damping.",
    )
    training.add_argument(
        "--stiffness-init", type=float,
        help="Initial normalized-coordinate restoring rate for a structured drift (default 0.01).",
    )
    training.add_argument(
        "--damping-init", type=float,
        help="Initial normalized-coordinate damping rate for a structured drift (default 0.01).",
    )
    training.add_argument(
        "--damped-operator", choices=("diagonal", "spsd"),
        help="Linear stiffness/damping structure for damped_residual: positive diagonal "
             "(default) or symmetric positive semidefinite full matrices.",
    )
    training.add_argument(
        "--surface-tension", type=float,
        help="Surface tension in N/m for capillary_energy drift (default 0.0728).",
    )
    training.add_argument("--diffusion-init", type=float,
                          help="Initial per-step standard deviation in standardized coordinates; fixed for drift-only training without a checkpoint.")
    training.add_argument("--diffusion-type", choices=("diagonal", "full", "state_diagonal",
                                                        "bounded_state_diagonal"),
                          help="Diffusion structure: constant diagonal (default), constant full, "
                               "unbounded state-conditioned diagonal, or smoothly bounded "
                               "state-conditioned diagonal MLP.")
    training.add_argument(
        "--diffusion-log-range", type=float,
        help="For bounded_state_diagonal, maximum absolute log modulation around its "
             "trainable per-mode baseline (default 2, giving factors exp(-2) to exp(2)).",
    )
    training.add_argument("--dt", type=float, help="Physical timestep; defaults to source interval and must match it.")
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="auto")
    training.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="sample and score standard physical rollouts")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data-root", type=Path)
    evaluation.add_argument("--shared-basis", type=Path)
    evaluation.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluation.add_argument("--num-conditions", type=int, default=32)
    evaluation.add_argument("--ensemble-size", type=int, default=16)
    evaluation.add_argument("--horizon", type=int, default=50)
    evaluation.add_argument("--rollout-lag", type=int,
                            help="Native steps per rollout transition; defaults to the checkpoint training lag.")
    evaluation.add_argument(
        "--diffusion-scale", type=float, default=1.,
        help="Multiply the trained diffusion factor during rollout only (default 1; 0 is deterministic).",
    )
    evaluation.add_argument("--enforce-train-trim-on-reference-windows", action="store_true",
                            help="Reject complete reference windows containing an increment above the saved "
                                 "training cutoff; default: allow them for long contiguous rollouts.")
    evaluation.add_argument("--batch-size", type=int, default=8)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument(
        "--multistep-horizons", type=int, nargs="+",
        help="Also estimate multistep NLL at these model-step horizons.",
    )
    evaluation.add_argument(
        "--multistep-mode-counts", type=int, nargs="+",
        help="First shared-POD modes scored at each matching evaluation horizon; "
             "defaults to matching counts saved by training, otherwise all coordinates.",
    )
    evaluation.add_argument(
        "--multistep-target", choices=("velocity", "displacement"),
        help="Override the saved multistep target (legacy checkpoints default to velocity).",
    )
    evaluation.add_argument(
        "--multistep-particles", type=int, default=100,
        help="Particles for optional multistep likelihood evaluation (default 100).",
    )
    evaluation.add_argument(
        "--multistep-particle-counts", type=int, nargs="+",
        help="Optional nonincreasing particle count at each matching evaluation horizon.",
    )
    evaluation.add_argument(
        "--multistep-seed", type=int, default=0,
        help="Reproducible noise seed for optional multistep likelihood evaluation.",
    )
    evaluation.add_argument(
        "--multistep-max-windows", type=int,
        help="Score at most this many reproducibly sampled retained multistep windows.",
    )
    evaluation.add_argument(
        "--multistep-window-seed", type=int, default=0,
        help="Window-sampling seed used with --multistep-max-windows (default 0).",
    )
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--no-metrics", action="store_true")
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        history_offsets = (None if args.history_offsets is None
                           else tuple(args.history_offsets))
        if (args.lag_steps < 1 or (args.history is not None and args.history < 0)
                or (history_offsets is not None
                    and (any(offset < 1 for offset in history_offsets)
                         or tuple(sorted(set(history_offsets))) != history_offsets))
                or not 0 <= args.diffusion_trim_percent < 100
                or not 0 <= args.train_trim_percent < 100
                or not math.isfinite(args.multistep_weight)
                or args.multistep_weight < 0
                or args.multistep_particles < 1
                or not math.isfinite(args.multistep_window_fraction)
                or not 0 < args.multistep_window_fraction <= 1
                or (not args.multistep_horizons
                    or any(horizon < 1 for horizon in args.multistep_horizons)
                    or tuple(sorted(set(args.multistep_horizons)))
                    != tuple(args.multistep_horizons))
                or (args.multistep_mode_counts is not None
                    and (args.multistep_weight <= 0
                         or args.multistep_target != "velocity"
                         or len(args.multistep_mode_counts) != len(args.multistep_horizons)
                         or any(count < 1 for count in args.multistep_mode_counts)))
                or (args.multistep_particle_counts is not None
                    and (args.multistep_weight <= 0
                         or len(args.multistep_particle_counts) != len(args.multistep_horizons)
                         or any(count < 1 for count in args.multistep_particle_counts)
                         or any(following > current for current, following in zip(
                             args.multistep_particle_counts,
                             args.multistep_particle_counts[1:]))))
                or (args.multistep_window_fraction < 1 and args.multistep_weight <= 0)
                or not math.isfinite(args.stationary_mean_weight)
                or args.stationary_mean_weight < 0
                or not math.isfinite(args.stationary_variance_weight)
                or args.stationary_variance_weight < 0
                or args.stationary_mean_horizon < 1
                or args.stationary_mean_trajectories < 2
                or not math.isfinite(args.stationary_mean_batch_fraction)
                or not 0 < args.stationary_mean_batch_fraction <= 1
                or (args.diffusion_log_range is not None
                    and (not math.isfinite(args.diffusion_log_range)
                         or args.diffusion_log_range <= 0))
                or (args.stiffness_init is not None
                    and (not math.isfinite(args.stiffness_init) or args.stiffness_init <= 0))
                or (args.damping_init is not None
                    and (not math.isfinite(args.damping_init) or args.damping_init <= 0))
                or (args.surface_tension is not None
                    and (not math.isfinite(args.surface_tension) or args.surface_tension <= 0))
                or (args.spatial_mean_highpass_hz is not None
                    and (not math.isfinite(args.spatial_mean_highpass_hz)
                         or args.spatial_mean_highpass_hz <= 0))):
            parser.error("Training requires positive --lag, nonnegative --history, strictly "
                         "increasing positive history and multistep horizons, positive particles, "
                         "a nonnegative multistep weight, valid mode/particle schedules and window "
                         "fraction, valid stationary-moment settings, and trim percentages in [0, 100).")
        if args.no_spatial_mean and args.spatial_mean_highpass_hz is not None:
            parser.error("--spatial-mean-highpass-hz cannot be combined with --no-spatial-mean.")
        if (args.checkpoint is None and args.diffusion_log_range is not None
                and args.diffusion_type != "bounded_state_diagonal"):
            parser.error("--diffusion-log-range requires --diffusion-type bounded_state_diagonal.")
        if (args.checkpoint is None
                and (args.stiffness_init is not None or args.damping_init is not None)
                and args.drift_type not in {"damped_residual", "capillary_energy"}):
            parser.error("--stiffness-init and --damping-init require a structured drift type.")
        if (args.checkpoint is None and args.damped_operator is not None
                and args.drift_type != "damped_residual"):
            parser.error("--damped-operator requires --drift-type damped_residual.")
        if (args.checkpoint is None and args.surface_tension is not None
                and args.drift_type != "capillary_energy"):
            parser.error("--surface-tension requires --drift-type capillary_energy.")
        if args.train_mode != "diffusion" and args.diffusion_trim_percent:
            parser.error("--diffusion-trim-percent is used only with --train-mode diffusion.")
        initial_checkpoint = None
        checkpoint_path = None
        if args.checkpoint is not None:
            if args.train_mode == "diffusion":
                parser.error("--train-mode diffusion starts without --checkpoint.")
            checkpoint_path = args.checkpoint / "best.pt" if args.checkpoint.is_dir() else args.checkpoint
            initial_checkpoint = load_checkpoint(checkpoint_path, "cpu")
            saved = SDEConfig.from_dict(initial_checkpoint["model_config"])
            required_mode = "diffusion" if args.train_mode == "drift" else "joint"
            if initial_checkpoint.get("training", {}).get("train_mode") != required_mode:
                parser.error(f"--train-mode {args.train_mode} requires a {required_mode} checkpoint.")
            if saved.state_variable == "increment":
                if args.lag_steps != saved.lag_steps:
                    parser.error(f"{saved.state_variable.capitalize()}-state checkpoint training "
                                 "requires the saved lag.")
            elif args.lag_steps < saved.lag_steps:
                parser.error("Checkpoint training requires the saved lag or a larger lag.")
            data = checkpoint_data(initial_checkpoint, data_root=args.data_root,
                                   shared_basis=args.shared_basis)
            requested_offsets = (history_offsets if history_offsets is not None else
                                 tuple(range(1, args.history + 1))
                                 if args.history is not None else None)
            if ((args.experiment is not None and args.experiment != data.experiment)
                    or (args.rank is not None and args.rank != data.rank)
                    or (args.dt is not None and not math.isclose(args.dt, saved.dt, rel_tol=1e-4))
                    or (args.hidden_layers is not None and tuple(args.hidden_layers) != saved.hidden_layers)
                    or (args.activation is not None and args.activation != saved.activation)
                    or (args.drift_type is not None and args.drift_type != saved.drift_type)
                    or (args.stiffness_init is not None
                        and args.stiffness_init != saved.stiffness_init)
                    or (args.damping_init is not None
                        and args.damping_init != saved.damping_init)
                    or (args.damped_operator is not None
                        and args.damped_operator != saved.damped_operator)
                    or (args.surface_tension is not None
                        and args.surface_tension != saved.surface_tension)
                    or (args.diffusion_init is not None and args.diffusion_init != saved.diffusion_init)
                    or (args.diffusion_type is not None and args.diffusion_type != saved.diffusion_type)
                    or (args.diffusion_log_range is not None
                        and args.diffusion_log_range != saved.diffusion_log_range)
                    or (requested_offsets is not None
                        and requested_offsets != saved.effective_history_offsets)
                    or (args.state_variable is not None and args.state_variable != saved.state_variable)
                    or (args.no_spatial_mean and data.include_spatial_mean)
                    or (args.spatial_mean_highpass_hz is not None
                        and args.spatial_mean_highpass_hz != data.spatial_mean_highpass_hz)
                    or args.validation_reps is not None or args.test_reps is not None):
                parser.error("Warm-start source, split, and architecture come from the checkpoint; supplied options differ.")
            config = replace(saved, lag_steps=args.lag_steps)
        else:
            if args.experiment is None or args.rank is None:
                parser.error("Training without --checkpoint requires --experiment and --rank.")
            state_variable = args.state_variable or "state"
            drift_type = args.drift_type or "mlp"
            default_history = {"state": 0, "increment": 1, "velocity": 1}[state_variable]
            history_steps = (history_offsets[-1] if history_offsets is not None else
                             args.history if args.history is not None else default_history)
            if state_variable == "increment" and history_steps < 1:
                parser.error("--state-variable increment requires --history of at least 1.")
            if (state_variable == "increment" and history_offsets is not None
                    and history_offsets[0] != 1):
                parser.error("Increment-state --history-offsets must begin with 1.")
            if state_variable == "velocity" and history_steps < 1:
                parser.error("--state-variable velocity requires at least one history frame.")
            if (state_variable == "velocity" and history_offsets is not None
                    and history_offsets[0] != 1):
                parser.error("Velocity-state --history-offsets must begin with 1.")
            if state_variable == "velocity" and history_steps > 1 and args.lag_steps != 1:
                parser.error("Velocity-state additional history is supported only with --lag 1.")
            if drift_type in {"damped_residual", "capillary_energy"} and state_variable != "velocity":
                parser.error(f"--drift-type {drift_type} requires --state-variable velocity.")
            options = dict(data_root=args.data_root, shared_basis=args.shared_basis,
                           validation_reps=args.validation_reps, test_reps=args.test_reps,
                           include_spatial_mean=not args.no_spatial_mean,
                           spatial_mean_highpass_hz=args.spatial_mean_highpass_hz)
            options = {key: value for key, value in options.items() if value is not None}
            data = SharedPODTrainingData(args.experiment, args.rank, **options)
            config = SDEConfig(data.n_coordinates, dt=data.native_dt if args.dt is None else args.dt,
                               hidden_layers=tuple(args.hidden_layers or (64, 64)),
                               activation=args.activation or "silu",
                               drift_type=drift_type,
                               stiffness_init=(args.stiffness_init
                                               if args.stiffness_init is not None else .01),
                               damping_init=(args.damping_init
                                             if args.damping_init is not None else .01),
                               damped_operator=args.damped_operator or "diagonal",
                               surface_tension=(args.surface_tension
                                                if args.surface_tension is not None else .0728),
                               capillary_mean_mode=(data.include_spatial_mean
                                                    if drift_type == "capillary_energy" else True),
                               diffusion_init=args.diffusion_init if args.diffusion_init is not None else .1,
                               diffusion_type=args.diffusion_type or "diagonal",
                               diffusion_log_range=(args.diffusion_log_range
                                                    if args.diffusion_log_range is not None else 2.),
                               lag_steps=args.lag_steps, history_steps=history_steps,
                               history_offsets=history_offsets,
                               state_variable=state_variable)
        if args.multistep_weight > 0 and config.state_variable != "velocity":
            parser.error("--multistep-weight requires --state-variable velocity.")
        if ((args.stationary_mean_weight > 0 or args.stationary_variance_weight > 0)
                and config.state_variable != "velocity"):
            parser.error("Stationary moment weights require --state-variable velocity.")
        if args.stationary_mean_weight > 0 and args.train_mode == "diffusion":
            parser.error("--stationary-mean-weight requires trainable drift.")
        try:
            stability = StabilityConfig(
                weight=args.stability_weight, a=args.stability_a, b=args.stability_b,
                metric=args.stability_metric, perturb_scale=args.stability_perturb_scale,
                generated_steps=args.stability_generated_steps,
                validation_seed=args.stability_validation_seed)
        except ValueError as error:
            parser.error(str(error))
        if stability.weight > 0 and config.state_variable != "velocity":
            parser.error("--stability-weight requires --state-variable velocity.")
        if (args.multistep_mode_counts is not None
                and any(count > data.rank for count in args.multistep_mode_counts)):
            parser.error("--multistep-mode-counts cannot exceed the shared-POD rank.")
        path = train(data, args.output, config, learning_rate=args.learning_rate,
                     batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
                     device=resolve_device(args.device), train_mode=args.train_mode,
                     diffusion_trim_percent=args.diffusion_trim_percent,
                     train_trim_percent=args.train_trim_percent,
                     covariance_space=args.diffusion_covariance_space,
                     multistep_weight=args.multistep_weight,
                     multistep_target=args.multistep_target,
                     multistep_horizons=tuple(args.multistep_horizons),
                     multistep_mode_counts=(
                         None if args.multistep_mode_counts is None else
                         tuple(args.multistep_mode_counts)),
                     multistep_particles=args.multistep_particles,
                     multistep_particle_counts=(
                         None if args.multistep_particle_counts is None else
                         tuple(args.multistep_particle_counts)),
                     multistep_window_fraction=args.multistep_window_fraction,
                     multistep_window_sampling=args.multistep_window_sampling,
                     multistep_validation_seed=args.multistep_validation_seed,
                     stationary_mean_weight=args.stationary_mean_weight,
                     stationary_variance_weight=args.stationary_variance_weight,
                     stationary_mean_horizon=args.stationary_mean_horizon,
                     stationary_mean_trajectories=args.stationary_mean_trajectories,
                     stationary_mean_batch_fraction=args.stationary_mean_batch_fraction,
                     stationary_mean_validation_seed=(
                         args.stationary_mean_validation_seed),
                     stability=stability,
                     initial_checkpoint=initial_checkpoint, initial_checkpoint_path=checkpoint_path)
    else:
        if args.rollout_lag is not None and args.rollout_lag < 1:
            parser.error("--rollout-lag must be positive.")
        if not math.isfinite(args.diffusion_scale) or args.diffusion_scale < 0:
            parser.error("--diffusion-scale must be finite and nonnegative.")
        if (args.multistep_horizons is not None
                and (args.multistep_particles < 1
                     or any(value < 1 for value in args.multistep_horizons)
                     or tuple(sorted(set(args.multistep_horizons)))
                     != tuple(args.multistep_horizons))):
            parser.error("Multistep evaluation requires increasing positive horizons and particles.")
        if (args.multistep_mode_counts is not None
                and (args.multistep_horizons is None
                     or args.multistep_target == "displacement"
                     or len(args.multistep_mode_counts) != len(args.multistep_horizons)
                     or any(count < 1 for count in args.multistep_mode_counts))):
            parser.error("--multistep-mode-counts require one positive count per velocity horizon.")
        if (args.multistep_particle_counts is not None
                and (args.multistep_horizons is None
                     or len(args.multistep_particle_counts) != len(args.multistep_horizons)
                     or any(count < 1 for count in args.multistep_particle_counts)
                     or any(following > current for current, following in zip(
                         args.multistep_particle_counts,
                         args.multistep_particle_counts[1:])))):
            parser.error(
                "--multistep-particle-counts require one nonincreasing positive count per horizon.")
        if (args.multistep_max_windows is not None
                and (args.multistep_horizons is None or args.multistep_max_windows < 1)):
            parser.error("--multistep-max-windows requires multistep horizons and a positive count.")
        path = evaluate(args.checkpoint, args.output, split=args.split,
                        num_conditions=args.num_conditions, ensemble_size=args.ensemble_size,
                        horizon=args.horizon, batch_size=args.batch_size, seed=args.seed,
                        device=resolve_device(args.device), data_root=args.data_root,
                        shared_basis=args.shared_basis, metrics=not args.no_metrics,
                        rollout_lag=args.rollout_lag,
                        diffusion_scale=args.diffusion_scale,
                        multistep_horizons=(None if args.multistep_horizons is None else
                                            tuple(args.multistep_horizons)),
                        multistep_particles=args.multistep_particles,
                        multistep_target=args.multistep_target,
                        multistep_mode_counts=(
                            None if args.multistep_mode_counts is None else
                            tuple(args.multistep_mode_counts)),
                        multistep_particle_counts=(
                            None if args.multistep_particle_counts is None else
                            tuple(args.multistep_particle_counts)),
                        multistep_seed=args.multistep_seed,
                        multistep_max_windows=args.multistep_max_windows,
                        multistep_window_seed=args.multistep_window_seed,
                        enforce_train_trim_on_reference_windows=(
                            args.enforce_train_trim_on_reference_windows))
    print(path.resolve())
