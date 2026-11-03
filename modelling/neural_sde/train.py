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
from .batches import indexed_velocity_batches
from .losses import (EulerMaruyamaNLL, MonteCarloMultistepNLL,
                     MultistepVelocityEnergyScore, StationaryMeanLoss)
from .model import NeuralSDE
from .history_noise import noisy_history_drift_loss, validate_history_noise
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


def _validated_multistep_horizon_weights(weights, horizons):
    """Return normalized nonnegative weights aligned with the horizons."""
    if weights is None:
        return (1 / len(horizons),) * len(horizons)
    weights = tuple(weights)
    if (len(weights) != len(horizons)
            or any(not math.isfinite(weight) or weight < 0 for weight in weights)
            or sum(weights) <= 0):
        raise ValueError(
            "Multistep horizon weights require one nonnegative finite value per horizon "
            "and a positive sum")
    total = sum(weights)
    return tuple(weight / total for weight in weights)


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
                                multistep_horizons: tuple[int, ...] | None = None,
                                eligibility_history_steps: int | None = None,
                                include_window_ids: bool = False):
    """Lagged position pairs with native one-frame velocities at both endpoints."""
    model_history_steps = config.history_steps
    if eligibility_history_steps is None:
        eligibility_history_steps = model_history_steps
    if (not isinstance(eligibility_history_steps, int)
            or isinstance(eligibility_history_steps, bool)
            or eligibility_history_steps < model_history_steps):
        raise ValueError(
            "eligibility_history_steps must be an integer at least as large as model history")
    sampling_config = config
    if eligibility_history_steps > model_history_steps:
        # Window eligibility and complete-stencil trimming may require more
        # observed frames than the checkpoint consumes. Dense offsets are used
        # only to construct the data window; model inputs are cropped below.
        sampling_config = replace(
            config, history_steps=eligibility_history_steps,
            history_offsets=tuple(range(1, eligibility_history_steps + 1)))
    maximum_horizon = max(multistep_horizons or (1,))
    source_indices = {name: index for index, name in enumerate(data.splits[split])}
    for source_name, batch in data.iter_transitions(
            split, lag_steps=sampling_config.lag_steps * maximum_horizon,
            history_steps=sampling_config.history_steps,
            history_spacing_steps=1,
            include_next_history=True, include_state_path=True,
            batch_size=batch_size, seed=seed):
        if multistep_horizons is not None:
            path = batch["state_path"]
            current_index = sampling_config.history_steps
            one_step_index = current_index + sampling_config.lag_steps
            batch["next_state"] = path[:, one_step_index]
            history_offsets = np.arange(1, model_history_steps + 1)
            batch["next_history_states"] = path[
                :, one_step_index - history_offsets]
            endpoint_indices = current_index + sampling_config.lag_steps * np.asarray(
                multistep_horizons)
            batch["target_velocities"] = (
                path[:, endpoint_indices] - path[:, endpoint_indices - 1])
            batch["target_displacements"] = (
                path[:, endpoint_indices] - path[:, current_index, None, :])
        # Preserve exactly the history consumed by the saved model. The older
        # frames remain in state_path solely for eligibility and trim checks.
        batch["history_states"] = batch["history_states"][:, :model_history_steps]
        if config.forcing_filter == "modal_biquad":
            cache = getattr(data, "_modal_forcing_cache", None)
            if cache is None or source_name not in cache:
                raise RuntimeError("modal forcing cache must be prepared before batching")
            indices = np.asarray(batch["start_index"], dtype=np.int64)
            offsets = np.arange(1, config.tcn_context_frames + 1, dtype=np.int64)
            forcing_history = cache[source_name][indices[:, None] - offsets]
            raw_target = cache[source_name][indices]
            keep = (np.isfinite(raw_target).all(axis=1)
                    & np.isfinite(forcing_history).all(axis=(1, 2)))
            if not keep.any():
                continue
            count = len(indices)
            batch = {key: (value[keep] if isinstance(value, np.ndarray)
                           and value.shape[:1] == (count,) else value)
                     for key, value in batch.items()}
            batch["forcing_history"] = forcing_history[keep]
            batch["raw_forcing_target"] = raw_target[keep]
        if include_window_ids:
            batch["window_start_index"] = batch["start_index"]
            batch["window_source_index"] = np.full(
                len(batch["start_index"]), source_indices[source_name], dtype=np.int64)
        yield batch


def _sde_transition_batches(data, config, split, batch_size, seed=None,
                            multistep_horizons: tuple[int, ...] | None = None,
                            eligibility_history_steps: int | None = None,
                            include_window_ids: bool = False):
    if config.state_variable != "velocity":
        if multistep_horizons is not None:
            raise ValueError("multistep likelihood requires state_variable velocity")
        yield from transition_batches(data, _edm_transition_config(
            data.n_coordinates, config.lag_steps, config.history_steps),
                                      split, batch_size, seed=seed)
        return
    for batch in _velocity_transition_arrays(
            data, config, split, batch_size, seed, multistep_horizons,
            eligibility_history_steps, include_window_ids):
        yield {key: torch.from_numpy(value).float() for key, value in batch.items()
               if key in ("current_state", "next_state", "history_states",
                          "next_history_states", "state_path", "target_velocities",
                          "target_displacements", "window_start_index",
                          "window_source_index", "forcing_history",
                          "raw_forcing_target")}


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
        trim_cutoff=None, trim_state_std=None, horizon=100,
        eligibility_history_steps: int | None = None,
        include_window_ids: bool = False):
    """Build endpoint ensembles with one clean max-horizon window per source block."""
    horizons = (horizon,) if isinstance(horizon, int) else tuple(horizon)
    generator = torch.Generator().manual_seed(seed)
    current, history, endpoints = [], [], []
    identifiers = {"window_start_index": [], "window_source_index": []}
    for batch in _retained_multistep_batches(
            data, model, split, source_batch_size, horizons, seed,
            trim_cutoff, trim_state_std, eligibility_history_steps,
            include_window_ids):
        available = len(batch["current_state"])
        if not available:
            continue
        selected = int(torch.randint(available, (), generator=generator))
        current.append(batch["current_state"][selected:selected + 1])
        history.append(batch["history_states"][selected:selected + 1])
        endpoints.append(
            batch["current_state"][selected:selected + 1, None, :]
            + batch["target_displacements"][selected:selected + 1])
        if include_window_ids:
            for key in identifiers:
                identifiers[key].append(batch[key][selected:selected + 1])
        if len(current) == trajectory_count:
            data_endpoints = torch.cat(endpoints)
            result = {"current_state": torch.cat(current),
                      "history_states": torch.cat(history),
                      "data_endpoints": data_endpoints}
            if len(horizons) == 1:
                result["data_endpoint"] = data_endpoints[:, 0]
            if include_window_ids:
                result.update({key: torch.cat(value)
                               for key, value in identifiers.items()})
            yield result
            current, history, endpoints = [], [], []
            identifiers = {"window_start_index": [], "window_source_index": []}


def _capped_multistep_batches(data, model, split, batch_size, horizons, max_windows,
                              window_seed, trim_cutoff=None, trim_state_std=None,
                              eligibility_history_steps: int | None = None,
                              include_window_ids: bool = False):
    """Select a reproducibly shuffled subset of retained multistep windows.

    Selection uses the data source's native batch size, rather than the scoring
    batch size, so changing GPU memory settings does not change the sampled
    windows. The selected CPU tensors are then rebatched for scoring.
    """
    selected = []
    count = 0
    for batch in _sde_transition_batches(
            data, model.config, split, data.batch_size, seed=window_seed,
            multistep_horizons=horizons,
            eligibility_history_steps=eligibility_history_steps,
            include_window_ids=include_window_ids):
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


def _retained_multistep_batches(data, model, split, batch_size, horizons, seed=None,
                                trim_cutoff=None, trim_state_std=None,
                                eligibility_history_steps: int | None = None,
                                include_window_ids: bool = False):
    """Yield long windows after applying the trim rule to their complete paths."""
    if (getattr(model, "indexed_training_batches", False)
            and isinstance(data, SharedPODTrainingData)):
        yield from indexed_velocity_batches(
            data, model.config, split, batch_size, seed, horizons, trim_cutoff,
            model.state_std.detach().cpu() if trim_state_std is None else trim_state_std,
            eligibility_history_steps=eligibility_history_steps,
            include_window_ids=include_window_ids)
        return
    for batch in _sde_transition_batches(
            data, model.config, split, batch_size, seed=seed,
            multistep_horizons=horizons,
            eligibility_history_steps=eligibility_history_steps,
            include_window_ids=include_window_ids):
        if trim_cutoff is not None:
            keep = _trim_keep(batch, model, trim_cutoff, trim_state_std)
            if not bool(keep.any()):
                continue
            batch = {key: value[keep] for key, value in batch.items()}
        yield batch


def set_trainable_components(model: NeuralSDE, train_mode: str):
    if train_mode not in {"joint", "diffusion", "drift", "memory"}:
        raise ValueError("train_mode must be joint, diffusion, drift, or memory")
    memory_parameters = model.memory_parameters()
    if train_mode == "memory" and not memory_parameters:
        raise ValueError("memory training requires a velocity-memory extension")
    for parameter in model.drift_model.parameters():
        parameter.requires_grad_(train_mode in {"joint", "drift"})
    for parameter in model.diffusion_model.parameters():
        parameter.requires_grad_(train_mode in {"joint", "diffusion"})
    for parameter in memory_parameters:
        parameter.requires_grad_(train_mode in {"joint", "drift", "memory"})
    for status in (True, False):
        names = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad == status]
        print(f"{'Trainable' if status else 'Frozen'} parameters: {', '.join(names) or 'none'}",
              flush=True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


_MULTISTEP_GRADIENT_GROUPS = (
    "drift", "structured", "stiffness", "damping", "mass", "residual",
    "memory", "diffusion")


def _validated_multistep_gradient_groups(groups) -> tuple[str, ...]:
    groups = tuple(groups)
    if (not groups or len(set(groups)) != len(groups)
            or any(group not in _MULTISTEP_GRADIENT_GROUPS for group in groups)):
        raise ValueError(
            "Multistep gradient groups must be distinct values from "
            + ", ".join(_MULTISTEP_GRADIENT_GROUPS))
    return groups


def _multistep_gradient_parameters(
        model: NeuralSDE, groups: tuple[str, ...]) -> tuple[
            tuple[torch.nn.Parameter, ...], tuple[str, ...]]:
    """Resolve the trainable leaves receiving the multistep loss gradient."""
    groups = _validated_multistep_gradient_groups(groups)
    memory_ids = {id(parameter) for parameter in model.memory_parameters()}
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_drift = name.startswith("drift_model.") or id(parameter) in memory_ids
        is_stiffness = name.startswith((
            "drift_model.raw_stiffness",
            "drift_model.stiffness_factor.",
            "drift_model.raw_mean_stiffness",
        ))
        is_damping = name.startswith((
            "drift_model.raw_damping",
            "drift_model.damping_factor.",
        ))
        is_mass = name.startswith("drift_model.raw_mass_multiplier")
        is_residual = (name.startswith("drift_model.residual.")
                       or (model.config.drift_type == "mlp"
                           and name.startswith("drift_model.")))
        matches = {
            "drift": is_drift,
            "structured": is_stiffness or is_damping or is_mass,
            "stiffness": is_stiffness,
            "damping": is_damping,
            "mass": is_mass,
            "residual": is_residual,
            "memory": id(parameter) in memory_ids,
            "diffusion": name.startswith("diffusion_model."),
        }
        if any(matches[group] for group in groups):
            selected.append((name, parameter))
    if not selected:
        raise ValueError(
            "The requested multistep gradient groups select no trainable parameters: "
            + ", ".join(groups))
    return (tuple(parameter for _, parameter in selected),
            tuple(name for name, _ in selected))


def _fit_restoring_velocity_adjustment(
        data, model: NeuralSDE, batch_size: int, train_trim_cutoff=None,
        train_trim_state_std=None) -> tuple[dict[str, torch.Tensor], dict]:
    """Fit both centered z-on-v and v-on-z regressions on retained training states."""
    if model.config.state_variable != "velocity":
        raise ValueError("velocity-adjusted restoring orthogonality requires velocity state")
    dimension = model.config.reduced_dim
    count = 0
    sum_position = torch.zeros(dimension, dtype=torch.float64)
    sum_velocity = torch.zeros(dimension, dtype=torch.float64)
    sum_position_position = torch.zeros((dimension, dimension), dtype=torch.float64)
    sum_position_velocity = torch.zeros((dimension, dimension), dtype=torch.float64)
    sum_velocity_velocity = torch.zeros((dimension, dimension), dtype=torch.float64)
    state_mean = model.state_mean.detach().cpu().double()
    state_std = model.state_std.detach().cpu().double()
    target_mean = model.target_mean.detach().cpu().double()
    target_std = model.target_std.detach().cpu().double()
    for batch in _sde_transition_batches(
            data, model.config, "train", batch_size):
        if train_trim_cutoff is not None:
            keep = _trim_keep(
                batch, model, train_trim_cutoff, train_trim_state_std)
            batch = {key: value[keep] for key, value in batch.items()}
        if not len(batch["current_state"]):
            continue
        current = batch["current_state"].double()
        previous = batch["history_states"][:, 0].double()
        position = (current - state_mean) / state_std
        velocity = (current - previous - target_mean) / target_std
        count += len(position)
        sum_position += position.sum(0)
        sum_velocity += velocity.sum(0)
        sum_position_position += position.T @ position
        sum_position_velocity += position.T @ velocity
        sum_velocity_velocity += velocity.T @ velocity
    if count <= dimension:
        raise ValueError("too few retained training states to fit velocity adjustment")
    position_mean = sum_position / count
    velocity_mean = sum_velocity / count
    position_covariance = (
        sum_position_position / count
        - torch.outer(position_mean, position_mean))
    cross_covariance = (
        sum_position_velocity / count
        - torch.outer(position_mean, velocity_mean))
    velocity_covariance = (
        sum_velocity_velocity / count
        - torch.outer(velocity_mean, velocity_mean))
    position_covariance = (position_covariance + position_covariance.T) / 2
    velocity_covariance = (velocity_covariance + velocity_covariance.T) / 2
    relative_tolerance = dimension * torch.finfo(torch.float64).eps
    velocity_inverse = torch.linalg.pinv(
        velocity_covariance, rtol=relative_tolerance, hermitian=True)
    position_inverse = torch.linalg.pinv(
        position_covariance, rtol=relative_tolerance, hermitian=True)
    matrix = cross_covariance @ velocity_inverse
    velocity_matrix = cross_covariance.T @ position_inverse
    adjusted_position_velocity_covariance = (
        cross_covariance - matrix @ velocity_covariance)
    adjusted_position_covariance = (
        position_covariance
        - matrix @ cross_covariance.T
        - cross_covariance @ matrix.T
        + matrix @ velocity_covariance @ matrix.T)
    adjusted_position_covariance = (
        adjusted_position_covariance + adjusted_position_covariance.T) / 2
    adjusted_velocity_position_covariance = (
        cross_covariance.T - velocity_matrix @ position_covariance)
    adjusted_velocity_covariance = (
        velocity_covariance
        - velocity_matrix @ cross_covariance
        - cross_covariance.T @ velocity_matrix.T
        + velocity_matrix @ position_covariance @ velocity_matrix.T)
    adjusted_velocity_covariance = (
        adjusted_velocity_covariance + adjusted_velocity_covariance.T) / 2
    position_variance = torch.diagonal(position_covariance)
    velocity_variance = torch.diagonal(velocity_covariance)
    adjusted_position_variance = torch.diagonal(
        adjusted_position_covariance).clamp_min(0.)
    adjusted_velocity_variance = torch.diagonal(
        adjusted_velocity_covariance).clamp_min(0.)
    position_velocity_correlation_denominator = torch.sqrt(
        adjusted_position_variance[:, None]
        * velocity_variance.clamp_min(0.)[None, :])
    adjusted_position_velocity_correlation = torch.where(
        position_velocity_correlation_denominator > 0,
        adjusted_position_velocity_covariance
        / position_velocity_correlation_denominator,
        torch.zeros_like(adjusted_position_velocity_covariance))
    velocity_position_correlation_denominator = torch.sqrt(
        adjusted_velocity_variance[:, None]
        * position_variance.clamp_min(0.)[None, :])
    adjusted_velocity_position_correlation = torch.where(
        velocity_position_correlation_denominator > 0,
        adjusted_velocity_position_covariance
        / velocity_position_correlation_denominator,
        torch.zeros_like(adjusted_velocity_position_covariance))
    position_explained_fraction = torch.where(
        position_variance > 0,
        1 - adjusted_position_variance / position_variance,
        torch.zeros_like(position_variance))
    velocity_explained_fraction = torch.where(
        velocity_variance > 0,
        1 - adjusted_velocity_variance / velocity_variance,
        torch.zeros_like(velocity_variance))

    def covariance_diagnostics(covariance):
        eigenvalues = torch.linalg.eigvalsh(covariance)
        tolerance = relative_tolerance * eigenvalues.abs().max()
        positive = eigenvalues[eigenvalues > tolerance]
        rank = int((eigenvalues > tolerance).sum())
        condition_number = (
            float(eigenvalues.max() / positive.min())
            if rank == dimension else None)
        return rank, condition_number

    velocity_rank, velocity_condition_number = covariance_diagnostics(
        velocity_covariance)
    position_rank, position_condition_number = covariance_diagnostics(
        position_covariance)
    adjustment = {
        "matrix": matrix.float(),
        "velocity_matrix": velocity_matrix.float(),
        "position_mean": position_mean.float(),
        "velocity_mean": velocity_mean.float(),
    }
    report = {
        "method": "symmetric training-population centered covariance pseudoinverses",
        "sample_count": count,
        "matrix": matrix.tolist(),
        "position_from_velocity_matrix": matrix.tolist(),
        "velocity_from_position_matrix": velocity_matrix.tolist(),
        "position_mean": position_mean.tolist(),
        "velocity_mean": velocity_mean.tolist(),
        "velocity_covariance_rank": velocity_rank,
        "velocity_covariance_condition_number": velocity_condition_number,
        "position_covariance_rank": position_rank,
        "position_covariance_condition_number": position_condition_number,
        "pseudoinverse_relative_tolerance": relative_tolerance,
        "position_variance": position_variance.tolist(),
        "velocity_variance": velocity_variance.tolist(),
        "adjusted_position_variance": adjusted_position_variance.tolist(),
        "adjusted_velocity_variance": adjusted_velocity_variance.tolist(),
        "position_explained_fraction": position_explained_fraction.tolist(),
        "velocity_explained_fraction": velocity_explained_fraction.tolist(),
        "max_abs_adjusted_position_velocity_correlation": float(
            adjusted_position_velocity_correlation.abs().max()),
        "max_abs_adjusted_velocity_position_correlation": float(
            adjusted_velocity_position_correlation.abs().max()),
    }
    return adjustment, report


def _restoring_residual_orthogonality(
        model: NeuralSDE, current: torch.Tensor,
        history_states: torch.Tensor | None, epsilon: float,
        velocity_adjustment: dict[str, torch.Tensor] | None = None,
        normalized_input: torch.Tensor | None = None,
        residual: torch.Tensor | None = None) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch projection energy of the residual onto diagonal restoring fields."""
    if (model.config.drift_type != "damped_residual"
            or model.config.damped_operator != "diagonal"):
        raise ValueError(
            "Restoring-residual orthogonality requires a diagonal damped_residual drift")
    if normalized_input is None:
        normalized_input = model._drift_input(current, history_states)
    position = normalized_input[..., :model.config.reduced_dim]
    if velocity_adjustment is not None:
        velocity = normalized_input[...,
                                    model.config.reduced_dim:2 * model.config.reduced_dim]
        position = (
            position - velocity_adjustment["position_mean"]
            - (velocity - velocity_adjustment["velocity_mean"])
            @ velocity_adjustment["matrix"].T)
    if residual is None:
        residual = model.nonlinear_drift(
            current, history_states, normalized_input=normalized_input)
    cross_moment = torch.mean(position * residual, dim=0)
    position_second_moment = torch.mean(position.square(), dim=0)
    projection_coefficients = cross_moment / (position_second_moment + epsilon)
    per_coordinate = cross_moment.square() / (position_second_moment + epsilon)
    return per_coordinate.sum(), per_coordinate, projection_coefficients


def _damping_residual_orthogonality(
        model: NeuralSDE, current: torch.Tensor,
        history_states: torch.Tensor | None, epsilon: float,
        velocity_adjustment: dict[str, torch.Tensor] | None = None,
        normalized_input: torch.Tensor | None = None,
        residual: torch.Tensor | None = None) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch projection energy of the residual onto diagonal damping fields."""
    if (model.config.drift_type != "damped_residual"
            or model.config.damped_operator != "diagonal"
            or not model.config.structured_damping):
        raise ValueError(
            "Damping-residual orthogonality requires a structured diagonal "
            "damped_residual drift")
    if normalized_input is None:
        normalized_input = model._drift_input(current, history_states)
    start = model.config.reduced_dim
    position = normalized_input[..., :start]
    velocity = normalized_input[..., start:2 * start]
    if velocity_adjustment is not None:
        if "velocity_matrix" not in velocity_adjustment:
            raise ValueError(
                "symmetric damping adjustment requires velocity_matrix")
        velocity = (
            velocity - velocity_adjustment["velocity_mean"]
            - (position - velocity_adjustment["position_mean"])
            @ velocity_adjustment["velocity_matrix"].T)
    if residual is None:
        residual = model.nonlinear_drift(
            current, history_states, normalized_input=normalized_input)
    cross_moment = torch.mean(velocity * residual, dim=0)
    velocity_second_moment = torch.mean(velocity.square(), dim=0)
    projection_coefficients = cross_moment / (velocity_second_moment + epsilon)
    per_coordinate = cross_moment.square() / (velocity_second_moment + epsilon)
    return per_coordinate.sum(), per_coordinate, projection_coefficients


def _diagonal_linear_residual_orthogonality(
        model: NeuralSDE, current: torch.Tensor,
        history_states: torch.Tensor | None, epsilon: float,
        normalized_input: torch.Tensor | None = None,
        residual: torch.Tensor | None = None) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projection energy onto each coordinate's joint ``span(z_i, v_i)``.

    Position, velocity, and residual are centered within the minibatch.  The
    model inputs are already standardized with retained-training statistics;
    this additional centering makes the finite-batch covariance estimate
    insensitive to a residual intercept.
    """
    if (model.config.drift_type != "damped_residual"
            or model.config.damped_operator != "diagonal"
            or not model.config.structured_damping):
        raise ValueError(
            "Joint diagonal linear orthogonality requires a structured "
            "diagonal damped_residual drift")
    if normalized_input is None:
        normalized_input = model._drift_input(current, history_states)
    dimension = model.config.reduced_dim
    position = normalized_input[..., :dimension]
    velocity = normalized_input[..., dimension:2 * dimension]
    if residual is None:
        residual = model.nonlinear_drift(
            current, history_states, normalized_input=normalized_input)
    position = position - position.mean(dim=0)
    velocity = velocity - velocity.mean(dim=0)
    residual = residual - residual.mean(dim=0)

    position_second_moment = torch.mean(position.square(), dim=0)
    velocity_second_moment = torch.mean(velocity.square(), dim=0)
    cross_moment = torch.mean(position * velocity, dim=0)
    residual_moments = torch.stack((
        torch.mean(position * residual, dim=0),
        torch.mean(velocity * residual, dim=0),
    ), dim=-1)
    covariance = torch.stack((
        torch.stack((position_second_moment, cross_moment), dim=-1),
        torch.stack((cross_moment, velocity_second_moment), dim=-1),
    ), dim=-2)
    identity = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    coefficients = torch.linalg.solve(
        covariance + epsilon * identity, residual_moments.unsqueeze(-1)
    ).squeeze(-1)
    per_coordinate = torch.sum(residual_moments * coefficients, dim=-1)
    return per_coordinate.sum(), per_coordinate, coefficients


def _epoch(model, data, split, batch_size, device, objective, optimizer=None, seed=None,
           train_mode="joint", train_trim_cutoff=None, train_trim_state_std=None,
           multistep_objective: MonteCarloMultistepNLL | None = None,
           multistep_weight: float = 0., multistep_noise_seed: int | None = None,
           metric_output: dict | None = None, multistep_max_windows: int | None = None,
           multistep_window_seed: int = 0,
           multistep_window_fraction: float = 1.,
           multistep_window_sampling: str = "examples",
           multistep_gradient_groups: tuple[str, ...] = ("drift",),
           stationary_mean_objective: StationaryMeanLoss | None = None,
           stationary_mean_weight: float = 0.,
           stationary_variance_weight: float = 0.,
           stationary_lagged_covariance_weight: float = 0.,
           stationary_mean_trajectories: int = 30,
           stationary_mean_batch_fraction: float = .05,
           stationary_mean_noise_seed: int = 0,
           stationary_mean_window_seed: int = 0,
           stability_objective: DiscreteLyapunovRegularizer | None = None,
           stability_noise_seed: int = 0,
           restoring_orthogonality_weight: float = 0.,
           damping_orthogonality_weight: float = 0.,
           restoring_orthogonality_epsilon: float = 1e-8,
           diagonal_linear_orthogonality_weight: float = 0.,
           diagonal_linear_orthogonality_epsilon: float = 1e-6,
           restoring_velocity_adjustment: dict[str, torch.Tensor] | None = None,
           history_noise_weight: float = 0., history_noise_std: float = .05,
           history_noise_seed: int = 0):
    model.train(optimizer is not None)
    validate_history_noise(model.config, history_noise_weight, history_noise_std,
                           history_noise_seed, train_mode)
    history_noise_generator = (torch.Generator(device=torch.device(device)).manual_seed(
        history_noise_seed) if history_noise_weight > 0 else None)
    history_noise_total = 0.
    use_multistep = multistep_weight > 0
    use_stationary_mean = stationary_mean_weight > 0
    use_stationary_variance = stationary_variance_weight > 0
    use_stationary_lagged_covariance = stationary_lagged_covariance_weight > 0
    use_stationary_moments = (use_stationary_mean or use_stationary_variance
                              or use_stationary_lagged_covariance)
    use_stability = stability_objective is not None and stability_objective.config.weight > 0
    use_restoring_orthogonality = restoring_orthogonality_weight > 0
    use_damping_orthogonality = damping_orthogonality_weight > 0
    use_diagonal_linear_orthogonality = diagonal_linear_orthogonality_weight > 0
    stability_totals = {}
    stability_generator = (torch.Generator(device=torch.device(device)).manual_seed(
        stability_noise_seed) if use_stability else None)
    if use_multistep and multistep_objective is None:
        raise ValueError("positive multistep_weight requires a multistep objective")
    if (not math.isfinite(restoring_orthogonality_weight)
            or restoring_orthogonality_weight < 0
            or not math.isfinite(damping_orthogonality_weight)
            or damping_orthogonality_weight < 0
            or not math.isfinite(restoring_orthogonality_epsilon)
            or restoring_orthogonality_epsilon <= 0
            or not math.isfinite(diagonal_linear_orthogonality_weight)
            or diagonal_linear_orthogonality_weight < 0
            or not math.isfinite(diagonal_linear_orthogonality_epsilon)
            or diagonal_linear_orthogonality_epsilon <= 0):
        raise ValueError("invalid residual orthogonality weight or epsilon")
    if (not math.isfinite(multistep_window_fraction)
            or not 0 < multistep_window_fraction <= 1):
        raise ValueError("multistep_window_fraction must lie in (0, 1]")
    if multistep_window_sampling not in {"examples", "batches"}:
        raise ValueError("multistep_window_sampling must be examples or batches")
    multistep_gradient_groups = _validated_multistep_gradient_groups(
        multistep_gradient_groups)
    multistep_gradient_parameters = (
        _multistep_gradient_parameters(model, multistep_gradient_groups)[0]
        if use_multistep and optimizer is not None else ())
    if use_stationary_moments and stationary_mean_objective is None:
        raise ValueError("positive stationary moment weight requires its objective")
    if (not math.isfinite(stationary_mean_weight) or stationary_mean_weight < 0
            or not math.isfinite(stationary_variance_weight)
            or stationary_variance_weight < 0
            or not math.isfinite(stationary_lagged_covariance_weight)
            or stationary_lagged_covariance_weight < 0
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
    one_step_total = 0.
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
    stationary_lagged_covariance_total = 0.
    stationary_lagged_covariance_error_sum = None
    stationary_lagged_covariance_weighted_total = 0.
    stationary_horizons = (stationary_mean_objective.horizons
                           if use_stationary_moments else ())
    stationary_mean_horizon_totals = {horizon: 0. for horizon in stationary_horizons}
    stationary_mean_horizon_error_sums = {horizon: None for horizon in stationary_horizons}
    stationary_variance_horizon_totals = {horizon: 0. for horizon in stationary_horizons}
    stationary_variance_horizon_error_sums = {
        horizon: None for horizon in stationary_horizons}
    stationary_lagged_covariance_horizon_totals = {
        horizon: 0. for horizon in stationary_horizons}
    stationary_lagged_covariance_horizon_error_sums = {
        horizon: None for horizon in stationary_horizons}
    restoring_orthogonality_total = 0.
    restoring_orthogonality_per_coordinate_total = None
    restoring_projection_coefficient_total = None
    damping_orthogonality_total = 0.
    damping_orthogonality_per_coordinate_total = None
    damping_projection_coefficient_total = None
    diagonal_linear_orthogonality_total = 0.
    diagonal_linear_orthogonality_per_coordinate_total = None
    diagonal_linear_projection_coefficient_total = None
    windows_pretrimmed = use_multistep and multistep_max_windows is not None
    indexed_batches = (getattr(model, "indexed_training_batches", False)
                       and model.config.state_variable == "velocity"
                       and isinstance(data, SharedPODTrainingData))
    if windows_pretrimmed:
        batches = _capped_multistep_batches(
            data, model, split, batch_size, horizons, multistep_max_windows,
            multistep_window_seed, train_trim_cutoff, train_trim_state_std)
    elif indexed_batches:
        batches = indexed_velocity_batches(
            data, model.config, split, batch_size, seed, cutoff=train_trim_cutoff,
            state_std=(model.state_std.detach().cpu() if train_trim_state_std is None
                       else train_trim_state_std))
    else:
        # Keep the one-step population independent of the largest requested
        # multistep horizon. Long, completely clean paths are drawn below from
        # their own iterator only when the auxiliary likelihood is evaluated.
        batches = _sde_transition_batches(
            data, model.config, split, batch_size, seed=seed)
    multistep_batches = (
        iter(_retained_multistep_batches(
            data, model, split, batch_size, horizons, multistep_window_seed,
            train_trim_cutoff, train_trim_state_std))
        if use_multistep and not windows_pretrimmed else None)
    generator = None
    if use_multistep and multistep_noise_seed is not None:
        generator = torch.Generator(device=torch.device(device)).manual_seed(
            multistep_noise_seed)
    window_generator = None
    if use_multistep and multistep_window_fraction < 1:
        window_generator = torch.Generator().manual_seed(multistep_window_seed)
    multistep_preselected = (use_multistep and indexed_batches and not windows_pretrimmed
                            and multistep_window_sampling == "examples")
    if multistep_preselected:
        # Select the SAME retained examples before materializing history and
        # endpoint tensors, not after constructing an entire long-window batch.
        multistep_batches = iter(indexed_velocity_batches(
            data, model.config, split, batch_size, multistep_window_seed, horizons,
            train_trim_cutoff, model.state_std.detach().cpu()
            if train_trim_state_std is None else train_trim_state_std,
            multistep_window_fraction, window_generator))
    batch_sampling_accumulator = (
        float(torch.rand((), generator=window_generator))
        if window_generator is not None and multistep_window_sampling == "batches" else 0.)
    stationary_mean_batches = (iter(_globally_mixed_retained_velocity_batches(
        data, model, split, min(256, data.batch_size, batch_size),
        stationary_mean_trajectories,
        stationary_mean_window_seed, train_trim_cutoff,
        train_trim_state_std, stationary_horizons))
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
    component_metrics = {}
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in batches:
            if train_trim_cutoff is not None and not windows_pretrimmed and not indexed_batches:
                keep = _trim_keep(batch, model, train_trim_cutoff, train_trim_state_std)
                if not bool(keep.any()):
                    continue
                batch = {key: value[keep] for key, value in batch.items()}
            current = batch["current_state"].to(device)
            following = batch["next_state"].to(device)
            history = batch.get("history_states")
            options = dict(zero_drift=(train_mode == "diffusion" and
                model.config.drift_type != "imported_oscillator_residual"))
            if history is not None:
                options["history_states"] = history.to(device)
            if "next_history_states" in batch:
                options["following_history_states"] = batch["next_history_states"].to(device)
            if "forcing_history" in batch:
                options["forcing_history"] = batch["forcing_history"].to(device)
                options["raw_forcing_target"] = batch["raw_forcing_target"].to(device)
            if model.config.drift_type == "imported_oscillator_residual" and not component_metrics:
                with torch.no_grad():
                    fixed = model.imported_oscillator_drift(current, options.get("history_states"))
                    if model.has_forcing_filter:
                        # The physical filtered mean also depends on the two
                        # observed past filtered values. Report the normalized
                        # learned raw-forcing mean here instead.
                        residual = model.raw_forcing_mean(options["forcing_history"])
                    else:
                        residual = model.drift(current, options.get("history_states")) - fixed
                    for name, value in (("fixed", fixed), ("residual", residual)):
                        component_metrics[f"imported_{name}_normalized_rms_first_batch"] = float(
                            (value / model._sde_scale()).square().mean().sqrt())
            one_step = objective(model, current, following, **options)
            horizon_losses = {}
            loss = one_step
            history_noise_weighted = None
            if history_noise_weight > 0:
                history_noise_loss = noisy_history_drift_loss(
                    model, current, following, options["history_states"],
                    history_noise_std, generator=history_noise_generator)
                history_noise_weighted = history_noise_weight * history_noise_loss
                history_noise_total += float(history_noise_loss.detach()) * len(current)
            orthogonality_input = orthogonality_residual = None
            if (use_restoring_orthogonality or use_damping_orthogonality
                    or use_diagonal_linear_orthogonality):
                orthogonality_input = model._drift_input(
                    current, options.get("history_states"))
                orthogonality_residual = model.nonlinear_drift(
                    current, options.get("history_states"),
                    normalized_input=orthogonality_input)
            if use_restoring_orthogonality:
                (restoring_orthogonality, restoring_per_coordinate,
                 restoring_projection_coefficients) = _restoring_residual_orthogonality(
                    model, current, options.get("history_states"),
                    restoring_orthogonality_epsilon,
                    restoring_velocity_adjustment,
                    orthogonality_input, orthogonality_residual)
                restoring_weighted = (
                    restoring_orthogonality_weight * restoring_orthogonality)
                loss = loss + restoring_weighted
                restoring_orthogonality_total += (
                    float(restoring_orthogonality.detach()) * len(current))
                contribution = restoring_per_coordinate.detach().double().cpu() * len(current)
                restoring_orthogonality_per_coordinate_total = (
                    contribution if restoring_orthogonality_per_coordinate_total is None else
                    restoring_orthogonality_per_coordinate_total + contribution)
                contribution = (
                    restoring_projection_coefficients.detach().double().cpu() * len(current))
                restoring_projection_coefficient_total = (
                    contribution if restoring_projection_coefficient_total is None else
                    restoring_projection_coefficient_total + contribution)
            if use_damping_orthogonality:
                (damping_orthogonality, damping_per_coordinate,
                 damping_projection_coefficients) = _damping_residual_orthogonality(
                    model, current, options.get("history_states"),
                    restoring_orthogonality_epsilon,
                    restoring_velocity_adjustment,
                    orthogonality_input, orthogonality_residual)
                damping_weighted = damping_orthogonality_weight * damping_orthogonality
                loss = loss + damping_weighted
                damping_orthogonality_total += (
                    float(damping_orthogonality.detach()) * len(current))
                contribution = damping_per_coordinate.detach().double().cpu() * len(current)
                damping_orthogonality_per_coordinate_total = (
                    contribution if damping_orthogonality_per_coordinate_total is None else
                    damping_orthogonality_per_coordinate_total + contribution)
                contribution = (
                    damping_projection_coefficients.detach().double().cpu() * len(current))
                damping_projection_coefficient_total = (
                    contribution if damping_projection_coefficient_total is None else
                    damping_projection_coefficient_total + contribution)
            if use_diagonal_linear_orthogonality:
                (diagonal_linear_orthogonality, diagonal_linear_per_coordinate,
                 diagonal_linear_projection_coefficients) = (
                    _diagonal_linear_residual_orthogonality(
                        model, current, options.get("history_states"),
                        diagonal_linear_orthogonality_epsilon,
                        orthogonality_input, orthogonality_residual))
                diagonal_linear_weighted = (
                    diagonal_linear_orthogonality_weight
                    * diagonal_linear_orthogonality)
                loss = loss + diagonal_linear_weighted
                diagonal_linear_orthogonality_total += (
                    float(diagonal_linear_orthogonality.detach()) * len(current))
                contribution = (
                    diagonal_linear_per_coordinate.detach().double().cpu()
                    * len(current))
                diagonal_linear_orthogonality_per_coordinate_total = (
                    contribution
                    if diagonal_linear_orthogonality_per_coordinate_total is None else
                    diagonal_linear_orthogonality_per_coordinate_total + contribution)
                contribution = (
                    diagonal_linear_projection_coefficients.detach().double().cpu()
                    * len(current))
                diagonal_linear_projection_coefficient_total = (
                    contribution
                    if diagonal_linear_projection_coefficient_total is None else
                    diagonal_linear_projection_coefficient_total + contribution)
            multistep_weighted = None
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
                multistep_batch = batch
                if score_multistep and multistep_batches is not None:
                    multistep_batch = next(multistep_batches, None)
                    if multistep_batch is None:
                        score_multistep = False
                candidate_count = (len(multistep_batch["current_state"])
                                   if score_multistep else 0)
                if (score_multistep and multistep_window_sampling == "examples"
                        and not multistep_preselected):
                    selected_count = max(1, math.ceil(
                        multistep_window_fraction * candidate_count))
                else:
                    selected_count = candidate_count
                if not score_multistep:
                    selected_cpu = None
                elif selected_count == candidate_count:
                    selected_cpu = None
                else:
                    selected_cpu = torch.randperm(
                        candidate_count,
                        generator=window_generator)[:selected_count].sort().values
                if score_multistep:
                    multistep_current = multistep_batch["current_state"]
                    multistep_history = multistep_batch["history_states"]
                    if selected_cpu is not None:
                        multistep_current = multistep_current.index_select(
                            0, selected_cpu)
                        multistep_history = multistep_history.index_select(
                            0, selected_cpu)
                    multistep_current = multistep_current.to(device)
                    multistep_history = multistep_history.to(device)
                    if multistep_objective.target == "joint":
                        multistep_targets = torch.cat((
                            multistep_batch["target_displacements"],
                            multistep_batch["target_velocities"]), dim=-1)
                    else:
                        target_key = ("target_displacements"
                                      if multistep_objective.target == "displacement" else
                                      "target_velocities")
                        multistep_targets = multistep_batch[target_key]
                    if selected_cpu is not None:
                        multistep_targets = multistep_targets.index_select(0, selected_cpu)
                    horizon_losses = multistep_objective.per_horizon(
                        model, multistep_current, multistep_targets.to(device),
                        history_states=multistep_history, generator=generator,
                        zero_drift=options["zero_drift"])
                    multistep = multistep_objective.aggregate(horizon_losses)
                    multistep_weighted = multistep_weight * multistep_scale * multistep
                    multistep_count += selected_count
            stationary_loss = stationary_weighted = stationary_mean_error = None
            stationary_variance_loss = stationary_variance_weighted = None
            stationary_variance_error = None
            stationary_lagged_covariance_loss = None
            stationary_lagged_covariance_weighted = None
            stationary_lagged_covariance_error = None
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
                            stationary_horizons))
                    if stationary_batch is None:
                        raise ValueError(
                            f"stationary mean sampler found fewer than two {split} trajectories")
                    stationary_selected_count = len(stationary_batch["current_state"])
                    stationary_current = stationary_batch["current_state"]
                    stationary_history = stationary_batch["history_states"]
                    stationary_endpoint = stationary_batch["data_endpoints"]
                    statistics_method = (
                        stationary_mean_objective
                        .per_horizon_statistics_with_lagged_covariance
                        if use_stationary_lagged_covariance else
                        stationary_mean_objective.per_horizon_statistics)
                    stationary_statistics = statistics_method(
                        model, stationary_current.to(device),
                        stationary_endpoint.to(device),
                        history_states=stationary_history.to(device),
                        generator=stationary_noise_generator)
                    aggregates = tuple(torch.stack([
                        stationary_statistics[h][column] for h in (
                            stationary_mean_objective.endpoint_horizons if column<4 else
                            stationary_mean_objective.lagged_covariance_horizons)]).mean(dim=0)
                        for column in range(6 if use_stationary_lagged_covariance else 4))
                    (stationary_loss, stationary_mean_error,
                     stationary_variance_loss, stationary_variance_error) = aggregates[:4]
                    if use_stationary_lagged_covariance:
                        (stationary_lagged_covariance_loss,
                         stationary_lagged_covariance_error) = aggregates[4:]
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
                        for horizon, statistics in stationary_statistics.items():
                            horizon_loss, horizon_error = statistics[:2]
                            stationary_mean_horizon_totals[horizon] += (
                                float(horizon_loss.detach()) * stationary_selected_count)
                            contribution = (horizon_error.detach().double().cpu()
                                            * stationary_selected_count)
                            previous = stationary_mean_horizon_error_sums[horizon]
                            stationary_mean_horizon_error_sums[horizon] = (
                                contribution if previous is None else previous + contribution)
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
                        for horizon, statistics in stationary_statistics.items():
                            horizon_loss, horizon_error = statistics[2:4]
                            stationary_variance_horizon_totals[horizon] += (
                                float(horizon_loss.detach()) * stationary_selected_count)
                            contribution = (horizon_error.detach().double().cpu()
                                            * stationary_selected_count)
                            previous = stationary_variance_horizon_error_sums[horizon]
                            stationary_variance_horizon_error_sums[horizon] = (
                                contribution if previous is None else previous + contribution)
                    if use_stationary_lagged_covariance:
                        stationary_lagged_covariance_weighted = (
                            stationary_lagged_covariance_weight
                            / stationary_mean_batch_fraction
                            * stationary_lagged_covariance_loss)
                        stationary_lagged_covariance_total += (
                            float(stationary_lagged_covariance_loss.detach())
                            * stationary_selected_count)
                        contribution = (
                            stationary_lagged_covariance_error.detach().double().cpu()
                            * stationary_selected_count)
                        stationary_lagged_covariance_error_sum = (
                            contribution
                            if stationary_lagged_covariance_error_sum is None else
                            stationary_lagged_covariance_error_sum + contribution)
                        for horizon, statistics in stationary_statistics.items():
                            horizon_loss, horizon_error = statistics[4:6]
                            stationary_lagged_covariance_horizon_totals[horizon] += (
                                float(horizon_loss.detach()) * stationary_selected_count)
                            contribution = (horizon_error.detach().double().cpu()
                                            * stationary_selected_count)
                            previous = (
                                stationary_lagged_covariance_horizon_error_sums[horizon])
                            stationary_lagged_covariance_horizon_error_sums[horizon] = (
                                contribution if previous is None else previous + contribution)
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
            if history_noise_weighted is not None:
                reported_loss = reported_loss + history_noise_weighted
            if multistep_weighted is not None:
                reported_loss = reported_loss + multistep_weighted
            if stationary_weighted is not None:
                reported_loss = reported_loss + stationary_weighted
                stationary_mean_weighted_total += (
                    float(stationary_weighted.detach()) * len(current))
            if stationary_variance_weighted is not None:
                reported_loss = reported_loss + stationary_variance_weighted
                stationary_variance_weighted_total += (
                    float(stationary_variance_weighted.detach()) * len(current))
            if stationary_lagged_covariance_weighted is not None:
                reported_loss = reported_loss + stationary_lagged_covariance_weighted
                stationary_lagged_covariance_weighted_total += (
                    float(stationary_lagged_covariance_weighted.detach()) * len(current))
            if not torch.isfinite(reported_loss):
                raise FloatingPointError(f"nonfinite {split} Neural-SDE loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if history_noise_weighted is not None:
                    parameters = tuple(p for p in model.drift_model.parameters()
                                       if p.requires_grad)
                    gradients = torch.autograd.grad(history_noise_weighted, parameters,
                                                    allow_unused=True)
                    for parameter, gradient in zip(parameters, gradients, strict=True):
                        if gradient is not None:
                            if parameter.grad is None:
                                parameter.grad = gradient
                            else:
                                parameter.grad.add_(gradient)
                if multistep_weighted is not None:
                    multistep_gradients = torch.autograd.grad(
                        multistep_weighted, multistep_gradient_parameters,
                        allow_unused=True)
                    for parameter, gradient in zip(
                            multistep_gradient_parameters, multistep_gradients,
                            strict=True):
                        if gradient is None:
                            continue
                        if parameter.grad is None:
                            parameter.grad = gradient
                        else:
                            parameter.grad.add_(gradient)
                drift_parameters = tuple(
                    parameter for parameter in model.drift_model.parameters()
                    if parameter.requires_grad)
                diffusion_parameters = tuple(
                    parameter for parameter in model.diffusion_model.parameters()
                    if parameter.requires_grad)
                if drift_parameters and (
                        stationary_weighted is not None
                        or stationary_variance_weighted is not None
                        or stationary_lagged_covariance_weighted is not None):
                    drift_auxiliary = sum(
                        value for value in (stationary_weighted,
                                            stationary_variance_weighted,
                                            stationary_lagged_covariance_weighted)
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
                model.validate_velocity_memory()
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
    epoch_total = one_step_total / count
    if use_multistep:
        epoch_total += multistep_weight * sum(
            weight * multistep_totals[horizon] / multistep_count
            for horizon, weight in zip(
                horizons, multistep_objective.horizon_weights, strict=True))
    if use_stability:
        epoch_total += stability_totals["weighted_penalty"] / count
    epoch_total += stationary_mean_weighted_total / count
    if history_noise_weight > 0:
        epoch_total += history_noise_weight * history_noise_total / count
    epoch_total += stationary_variance_weighted_total / count
    epoch_total += stationary_lagged_covariance_weighted_total / count
    if use_restoring_orthogonality:
        epoch_total += (restoring_orthogonality_weight
                        * restoring_orthogonality_total / count)
    if use_damping_orthogonality:
        epoch_total += (damping_orthogonality_weight
                        * damping_orthogonality_total / count)
    if use_diagonal_linear_orthogonality:
        epoch_total += (diagonal_linear_orthogonality_weight
                        * diagonal_linear_orthogonality_total / count)
    if metric_output is not None:
        metric_output.update(component_metrics)
        metric_output.update(total_loss=epoch_total,
                             one_step_nll=one_step_total / count,
                             window_count=count,
                             multistep_window_count=multistep_count)
        if history_noise_weight > 0:
            metric_output.update(
                history_noise_loss=history_noise_total / count,
                history_noise_weighted_contribution=(
                    history_noise_weight * history_noise_total / count))
        if use_multistep:
            values = {horizon: value / multistep_count
                      for horizon, value in multistep_totals.items()}
            metric_output.update({f"multistep_loss_h{horizon}": value
                                  for horizon, value in values.items()})
            metric_output.update({
                f"multistep_{multistep_objective.metric_name}_h{horizon}": value
                for horizon, value in values.items()})
            metric_output.update({
                f"multistep_weighted_loss_h{horizon}": (
                    multistep_weight * weight * values[horizon])
                for horizon, weight in zip(
                    horizons, multistep_objective.horizon_weights, strict=True)})
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
            for horizon in stationary_horizons:
                horizon_error = (
                    stationary_mean_horizon_error_sums[horizon] / stationary_mean_count)
                metric_output.update({
                    f"stationary_mean_loss_h{horizon}": (
                        stationary_mean_horizon_totals[horizon] / stationary_mean_count),
                    f"stationary_mean_normalized_error_h{horizon}": (
                        horizon_error.tolist()),
                    f"stationary_mean_normalized_error_rmse_h{horizon}": float(
                        torch.sqrt(torch.mean(horizon_error.square()))),
                })
        if use_stationary_variance:
            aggregate = stationary_variance_error_sum / stationary_mean_count
            selected_aggregate = stationary_mean_objective.selected_variance_error(
                aggregate)
            metric_output.update(
                stationary_variance_loss=(
                    stationary_variance_total / stationary_mean_count),
                stationary_variance_weighted_contribution=(
                    stationary_variance_weighted_total / count),
                stationary_variance_normalized_error=aggregate.tolist(),
                stationary_variance_normalized_error_rmse=float(
                    torch.sqrt(torch.mean(aggregate.square()))),
                stationary_variance_selected_normalized_error_rmse=float(
                    torch.sqrt(torch.mean(selected_aggregate.square()))))
            for horizon in stationary_horizons:
                horizon_error = (
                    stationary_variance_horizon_error_sums[horizon] / stationary_mean_count)
                selected_horizon_error = (
                    stationary_mean_objective.selected_variance_error(horizon_error))
                metric_output.update({
                    f"stationary_variance_loss_h{horizon}": (
                        stationary_variance_horizon_totals[horizon]
                        / stationary_mean_count),
                    f"stationary_variance_normalized_error_h{horizon}": (
                        horizon_error.tolist()),
                    f"stationary_variance_normalized_error_rmse_h{horizon}": float(
                        torch.sqrt(torch.mean(horizon_error.square()))),
                    f"stationary_variance_selected_normalized_error_rmse_h{horizon}": float(
                        torch.sqrt(torch.mean(selected_horizon_error.square()))),
                })
        if use_stationary_lagged_covariance:
            aggregate = (
                stationary_lagged_covariance_error_sum / stationary_mean_count)
            metric_output.update(
                stationary_lagged_covariance_loss=(
                    stationary_lagged_covariance_total / stationary_mean_count),
                stationary_lagged_covariance_weighted_contribution=(
                    stationary_lagged_covariance_weighted_total / count),
                stationary_lagged_covariance_normalized_error=aggregate.tolist(),
                stationary_lagged_covariance_normalized_error_rmse=float(
                    torch.sqrt(torch.mean(aggregate.square()))))
            for horizon in stationary_horizons:
                horizon_error = (
                    stationary_lagged_covariance_horizon_error_sums[horizon]
                    / stationary_mean_count)
                metric_output.update({
                    f"stationary_lagged_covariance_loss_h{horizon}": (
                        stationary_lagged_covariance_horizon_totals[horizon]
                        / stationary_mean_count),
                    f"stationary_lagged_covariance_normalized_error_h{horizon}": (
                        horizon_error.tolist()),
                    f"stationary_lagged_covariance_normalized_error_rmse_h{horizon}": float(
                        torch.sqrt(torch.mean(horizon_error.square()))),
                })
        if use_restoring_orthogonality:
            metric_output.update(
                restoring_orthogonality_loss=(
                    restoring_orthogonality_total / count),
                restoring_orthogonality_weighted_contribution=(
                    restoring_orthogonality_weight
                    * restoring_orthogonality_total / count),
                restoring_orthogonality_per_coordinate=(
                    restoring_orthogonality_per_coordinate_total / count).tolist(),
                restoring_residual_projection_coefficients=(
                    restoring_projection_coefficient_total / count).tolist())
        if use_damping_orthogonality:
            metric_output.update(
                damping_orthogonality_loss=(
                    damping_orthogonality_total / count),
                damping_orthogonality_weighted_contribution=(
                    damping_orthogonality_weight
                    * damping_orthogonality_total / count),
                damping_orthogonality_per_coordinate=(
                    damping_orthogonality_per_coordinate_total / count).tolist(),
                damping_residual_projection_coefficients=(
                    damping_projection_coefficient_total / count).tolist())
        if use_diagonal_linear_orthogonality:
            coefficients = diagonal_linear_projection_coefficient_total / count
            metric_output.update(
                diagonal_linear_orthogonality_loss=(
                    diagonal_linear_orthogonality_total / count),
                diagonal_linear_orthogonality_weighted_contribution=(
                    diagonal_linear_orthogonality_weight
                    * diagonal_linear_orthogonality_total / count),
                diagonal_linear_orthogonality_per_coordinate=(
                    diagonal_linear_orthogonality_per_coordinate_total / count).tolist(),
                diagonal_linear_residual_projection_coefficients=(
                    coefficients.tolist()),
                diagonal_linear_residual_position_coefficients=(
                    coefficients[:, 0].tolist()),
                diagonal_linear_residual_velocity_coefficients=(
                    coefficients[:, 1].tolist()))
        metric_output.update({f"stability_{key}": (
            value if key.endswith("max_excess") else value / count)
            for key, value in stability_totals.items()})
    return epoch_total


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
    """Yield changes (minus the fixed imported prior, when present) and batches."""
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
        if model.config.drift_type == "imported_oscillator_residual":
            device = model.state_mean.device
            with torch.no_grad():
                fixed = model.imported_oscillator_drift(
                    torch.as_tensor(batch["current_state"], dtype=torch.float32, device=device),
                    torch.as_tensor(batch["history_states"], dtype=torch.float32, device=device))
            delta = delta - fixed.cpu().numpy()
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
                      "velocity_change_minus_fixed_oscillator" if model.config.drift_type == "imported_oscillator_residual" else
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
          multistep_objective: str = "gmm_nll",
          multistep_target: str = "velocity",
          multistep_mode_counts: tuple[int, ...] | None = None,
          multistep_particle_counts: tuple[int, ...] | None = None,
          multistep_horizon_weights: tuple[float, ...] | None = None,
          multistep_position_weight: float = 1.,
          multistep_velocity_weight: float = 1.,
          multistep_window_fraction: float = 1.,
          multistep_window_sampling: str = "examples",
          multistep_gradient_groups: tuple[str, ...] = ("drift",),
          stationary_mean_weight: float = 0.,
          stationary_variance_weight: float = 0.,
          stationary_variance_pod_mode_count: int | None = None,
          stationary_lagged_covariance_weight: float = 0.,
          stationary_lagged_covariance_mode_count: int | None = None,
          stationary_lagged_covariance_epsilon: float = 1e-6,
          stationary_lagged_covariance_coordinates: tuple[int, ...] | None = None,
          stationary_lagged_covariance_normalization: str = "batch",
          stationary_lagged_covariance_horizons: tuple[int, ...] | None = None,
          stationary_mean_horizon: int = 100,
          stationary_mean_horizons: tuple[int, ...] | None = None,
          stationary_mean_trajectories: int = 30,
          stationary_mean_batch_fraction: float = .05,
          stationary_mean_validation_seed: int = 0,
          restoring_orthogonality_weight: float = 0.,
          damping_orthogonality_weight: float = 0.,
          restoring_orthogonality_epsilon: float = 1e-8,
          restoring_orthogonality_adjust_velocity: bool = False,
          diagonal_linear_orthogonality_weight: float = 0.,
          diagonal_linear_orthogonality_epsilon: float = 1e-6,
          stability: StabilityConfig | None = None,
          initial_checkpoint=None, initial_checkpoint_path: Path | None = None,
          history_noise_weight: float = 0., history_noise_std: float = .05,
          history_noise_validation_seed: int = 0,
          tcn_rollout_cache: bool = False,
          indexed_training_batches: bool = False,
          tcn_rollout_compile: bool = False,
          oscillator_artifact: Path | None = None,
          forcing_filter_artifact: Path | None = None) -> Path:
    """Fit selected components; save best/latest by the validation objective."""
    if config.drift_type == "imported_oscillator_residual":
        if initial_checkpoint is None and oscillator_artifact is None:
            raise ValueError("Fresh imported oscillator training requires oscillator_artifact")
        if initial_checkpoint is not None and oscillator_artifact is not None:
            raise ValueError("Warm starts use embedded operators; do not supply oscillator_artifact")
    elif oscillator_artifact is not None:
        raise ValueError("oscillator_artifact requires imported_oscillator_residual drift")
    if config.forcing_filter == "modal_biquad":
        if initial_checkpoint is None and forcing_filter_artifact is None:
            raise ValueError("Fresh filtered-forcing training requires forcing_filter_artifact")
        if initial_checkpoint is not None and forcing_filter_artifact is not None:
            raise ValueError("Warm starts use the embedded forcing filter")
        unsupported = []
        if multistep_weight > 0: unsupported.append("multistep")
        if any(weight > 0 for weight in (
                stationary_mean_weight, stationary_variance_weight,
                stationary_lagged_covariance_weight)): unsupported.append("stationary moments")
        if stability is not None and stability.weight > 0: unsupported.append("stability")
        if history_noise_weight > 0: unsupported.append("history noise")
        if any(weight > 0 for weight in (
                restoring_orthogonality_weight, damping_orthogonality_weight,
                diagonal_linear_orthogonality_weight)): unsupported.append("orthogonality")
        if indexed_training_batches: unsupported.append("indexed batches")
        if tcn_rollout_cache or tcn_rollout_compile: unsupported.append("TCN rollout cache")
        if unsupported:
            raise ValueError("modal forcing filter currently supports one-step training only; disable "
                             + ", ".join(unsupported))
    elif forcing_filter_artifact is not None:
        raise ValueError("forcing_filter_artifact requires forcing_filter modal_biquad")
    validate_history_noise(config, history_noise_weight, history_noise_std,
                           history_noise_validation_seed, train_mode)
    if tcn_rollout_compile and not tcn_rollout_cache:
        raise ValueError("tcn_rollout_compile requires tcn_rollout_cache")
    if config.fixed_memory_frequencies is not None:
        raise ValueError("Fit fixed memory with modelling.neural_sde.fit_fixed_memory; generic training would unfreeze the parent")
    if stationary_lagged_covariance_normalization not in {"batch", "training"}:
        raise ValueError("lagged covariance normalization must be batch or training")
    if stationary_lagged_covariance_horizons is not None:
        StationaryMeanLoss(lagged_covariance_horizons=stationary_lagged_covariance_horizons)
        if stationary_lagged_covariance_weight <= 0:
            raise ValueError("independent correlation horizons require a positive correlation weight")
    if stationary_lagged_covariance_coordinates is not None:
        StationaryMeanLoss(lagged_covariance_coordinates=stationary_lagged_covariance_coordinates,
                           lagged_covariance_mode_count=stationary_lagged_covariance_mode_count)
        if max(stationary_lagged_covariance_coordinates) >= config.reduced_dim:
            raise ValueError("lagged covariance coordinate exceeds reduced dimension")
    if config.reduced_dim != data.n_coordinates or not math.isclose(config.dt, data.native_dt, rel_tol=1e-4):
        raise ValueError("SDE dimension and dt must match the prepared coordinates and native interval")
    if not data.splits["validation"]:
        raise ValueError("at least one validation repetition is required")
    if batch_size < 1 or epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive")
    if (train_mode not in {"joint", "diffusion", "drift", "memory"}
            or not 0 <= diffusion_trim_percent < 100 or not 0 <= train_trim_percent < 100):
        raise ValueError("Invalid train mode or diffusion trim percentage")
    if covariance_space not in {"normalized", "physical"}:
        raise ValueError("Invalid diffusion covariance space")
    multistep_horizons = tuple(multistep_horizons)
    if multistep_objective not in {"gmm_nll", "energy_score"}:
        raise ValueError("Multistep objective must be gmm_nll or energy_score")
    if multistep_target not in {"velocity", "displacement", "joint"}:
        raise ValueError("Multistep target must be velocity, displacement, or joint")
    if multistep_objective == "energy_score" and multistep_target == "displacement":
        raise ValueError("Multistep energy score supports velocity or joint target")
    if multistep_objective == "gmm_nll" and multistep_target == "joint":
        raise ValueError("Multistep GMM NLL does not support a joint target")
    if (multistep_target != "joint"
            and (multistep_position_weight != 1 or multistep_velocity_weight != 1)):
        raise ValueError(
            "Multistep position/velocity block weights require target joint")
    custom_particle_counts = multistep_particle_counts is not None
    multistep_mode_counts = _validated_multistep_mode_counts(
        multistep_mode_counts, multistep_horizons, data.rank, multistep_target)
    multistep_particle_counts = _validated_multistep_particle_counts(
        multistep_particle_counts, multistep_horizons, multistep_particles)
    multistep_horizon_weights = _validated_multistep_horizon_weights(
        multistep_horizon_weights, multistep_horizons)
    if (multistep_objective == "energy_score"
            and any(count < 2 for count in multistep_particle_counts)):
        raise ValueError("Multistep energy score requires at least two particles per horizon")
    if multistep_mode_counts is not None and multistep_weight <= 0:
        raise ValueError("Multistep mode counts require a positive multistep weight")
    if custom_particle_counts and multistep_weight <= 0:
        raise ValueError("Multistep particle counts require a positive multistep weight")
    if multistep_window_fraction < 1 and multistep_weight <= 0:
        raise ValueError("Multistep window subsampling requires a positive multistep weight")
    if multistep_window_sampling not in {"examples", "batches"}:
        raise ValueError("Multistep window sampling must be examples or batches")
    multistep_gradient_groups = _validated_multistep_gradient_groups(
        multistep_gradient_groups)
    if (not math.isfinite(multistep_weight) or multistep_weight < 0
            or not math.isfinite(multistep_position_weight)
            or multistep_position_weight <= 0
            or not math.isfinite(multistep_velocity_weight)
            or multistep_velocity_weight <= 0
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
        raise ValueError("Multistep objective requires state_variable velocity")
    stationary_mean_horizons = (
        (stationary_mean_horizon,) if stationary_mean_horizons is None
        else tuple(stationary_mean_horizons))
    if (not math.isfinite(stationary_mean_weight) or stationary_mean_weight < 0
            or not math.isfinite(stationary_variance_weight)
            or stationary_variance_weight < 0
            or (stationary_variance_pod_mode_count is not None
                and (not isinstance(stationary_variance_pod_mode_count, int)
                     or isinstance(stationary_variance_pod_mode_count, bool)
                     or stationary_variance_pod_mode_count < 0
                     or stationary_variance_pod_mode_count > data.rank
                     or (stationary_variance_pod_mode_count == 0
                         and not data.include_spatial_mean)))
            or not math.isfinite(stationary_lagged_covariance_weight)
            or stationary_lagged_covariance_weight < 0
            or (stationary_lagged_covariance_mode_count is not None
                and (not isinstance(stationary_lagged_covariance_mode_count, int)
                     or isinstance(stationary_lagged_covariance_mode_count, bool)
                     or stationary_lagged_covariance_mode_count < 1
                     or stationary_lagged_covariance_mode_count > data.rank))
            or not math.isfinite(stationary_lagged_covariance_epsilon)
            or stationary_lagged_covariance_epsilon <= 0
            or not isinstance(stationary_mean_horizon, int)
            or isinstance(stationary_mean_horizon, bool) or stationary_mean_horizon < 1
            or not stationary_mean_horizons
            or any(not isinstance(horizon, int) or isinstance(horizon, bool)
                   or horizon < 1 for horizon in stationary_mean_horizons)
            or tuple(sorted(set(stationary_mean_horizons))) != stationary_mean_horizons
            or not isinstance(stationary_mean_trajectories, int)
            or isinstance(stationary_mean_trajectories, bool)
            or stationary_mean_trajectories < 2
            or not math.isfinite(stationary_mean_batch_fraction)
            or not 0 < stationary_mean_batch_fraction <= 1
            or not isinstance(stationary_mean_validation_seed, int)
            or isinstance(stationary_mean_validation_seed, bool)):
        raise ValueError("Invalid stationary moment weight, horizon, trajectories, fraction, or seed")
    if ((stationary_mean_weight > 0 or stationary_variance_weight > 0
         or stationary_lagged_covariance_weight > 0)
            and config.state_variable != "velocity"):
        raise ValueError("Stationary moment loss requires state_variable velocity")
    if ((stationary_mean_weight > 0 or stationary_lagged_covariance_weight > 0)
            and train_mode == "diffusion"):
        raise ValueError("Stationary mean/lagged-covariance loss requires trainable drift")
    if (not math.isfinite(restoring_orthogonality_weight)
            or restoring_orthogonality_weight < 0
            or not math.isfinite(damping_orthogonality_weight)
            or damping_orthogonality_weight < 0
            or not math.isfinite(restoring_orthogonality_epsilon)
            or restoring_orthogonality_epsilon <= 0
            or not math.isfinite(diagonal_linear_orthogonality_weight)
            or diagonal_linear_orthogonality_weight < 0
            or not math.isfinite(diagonal_linear_orthogonality_epsilon)
            or diagonal_linear_orthogonality_epsilon <= 0):
        raise ValueError("Invalid residual orthogonality weight or epsilon")
    if restoring_orthogonality_weight > 0:
        if (config.drift_type != "damped_residual"
                or config.damped_operator != "diagonal"):
            raise ValueError(
                "Restoring orthogonality requires --drift-type damped_residual, "
                "and --damped-operator diagonal")
        if train_mode in {"diffusion", "memory"}:
            raise ValueError("Restoring orthogonality requires directly trainable drift")
    if damping_orthogonality_weight > 0:
        if (config.drift_type != "damped_residual"
                or config.damped_operator != "diagonal"
                or not config.structured_damping):
            raise ValueError(
                "Damping orthogonality requires --drift-type damped_residual, "
                "--damped-operator diagonal, and structured damping")
        if train_mode in {"diffusion", "memory"}:
            raise ValueError("Damping orthogonality requires directly trainable drift")
    if diagonal_linear_orthogonality_weight > 0:
        if (config.drift_type != "damped_residual"
                or config.damped_operator != "diagonal"
                or not config.structured_damping):
            raise ValueError(
                "Joint diagonal linear orthogonality requires --drift-type "
                "damped_residual, --damped-operator diagonal, and structured damping")
        if train_mode in {"diffusion", "memory"}:
            raise ValueError(
                "Joint diagonal linear orthogonality requires directly trainable drift")
        if (restoring_orthogonality_weight > 0
                or damping_orthogonality_weight > 0
                or restoring_orthogonality_adjust_velocity):
            raise ValueError(
                "Joint diagonal linear orthogonality cannot be combined with the legacy "
                "restoring/damping orthogonality options")
    if (restoring_orthogonality_adjust_velocity
            and restoring_orthogonality_weight <= 0
            and damping_orthogonality_weight <= 0):
        raise ValueError(
            "Cross adjustment requires a positive residual orthogonality weight")
    stability = stability or StabilityConfig()
    if stability.weight > 0 and config.state_variable != "velocity":
        raise ValueError("Stability regularization requires state_variable velocity")
    if train_mode == "diffusion" and initial_checkpoint is not None:
        raise ValueError("Diffusion mode starts without a checkpoint")
    if train_mode == "memory" and initial_checkpoint is None:
        raise ValueError("Memory mode requires a warm-start checkpoint")
    seed_everything(seed)
    device = torch.device(device)
    output = Path(output)
    if initial_checkpoint_path is not None and output.resolve() == Path(initial_checkpoint_path).resolve().parent:
        raise ValueError("Stage 2 output must differ from the Stage 1 checkpoint directory")
    output.mkdir(parents=True, exist_ok=True)
    data.cache_coordinates(2.)
    signature = data.signature()
    model = NeuralSDE(config)
    model.tcn_rollout_cache = tcn_rollout_cache
    model.tcn_rollout_compile = tcn_rollout_compile
    model.indexed_training_batches = indexed_training_batches
    if initial_checkpoint is not None:
        previous = SDEConfig.from_dict(initial_checkpoint["model_config"])
        required_previous_mode = "diffusion" if train_mode == "drift" else "joint"
        removing_structured_damping = (
            previous.drift_type == "damped_residual"
            and previous.structured_damping
            and not config.structured_damping)
        if train_mode == "memory":
            compatible_config = replace(
                config, velocity_memory_offsets=None, velocity_memory_rank=None,
                velocity_memory_norm_cap=previous.velocity_memory_norm_cap,
                velocity_oscillator_periods=None,
                velocity_oscillator_burnin=previous.velocity_oscillator_burnin,
                velocity_oscillator_radius_init=previous.velocity_oscillator_radius_init,
                velocity_oscillator_norm_cap=previous.velocity_oscillator_norm_cap,
                velocity_innovation_offsets=None, velocity_innovation_rank=None,
                velocity_innovation_norm_cap=previous.velocity_innovation_norm_cap,
                history_steps=previous.history_steps,
                history_offsets=previous.history_offsets)
        elif removing_structured_damping:
            compatible_config = replace(config, structured_damping=True)
        else:
            compatible_config = config
        if (initial_checkpoint.get("training", {}).get("train_mode") != required_previous_mode
                or compatible_config != replace(previous, lag_steps=config.lag_steps)
                or (config.lag_steps != previous.lag_steps
                    if config.state_variable == "increment"
                    else config.lag_steps < previous.lag_steps)
                or signature != initial_checkpoint["data_signature"]):
            raise ValueError(f"{train_mode} warm start requires a matching "
                             f"{required_previous_mode} checkpoint and "
                             f"{'the same lag' if config.state_variable == 'increment' else 'the same or a larger lag'}")
        if train_mode == "memory":
            incompatible = model.load_state_dict(
                initial_checkpoint["model_state"], strict=False)
            if config.velocity_innovation_offsets is not None:
                expected_missing = {
                    "velocity_innovation_left", "velocity_innovation_right",
                    "velocity_innovation_coefficients"}
            elif config.velocity_oscillator_periods is not None:
                expected_missing = {
                    "velocity_oscillator_input", "velocity_oscillator_output",
                    "velocity_oscillator_coefficients",
                    "velocity_oscillator_raw_radius", "velocity_oscillator_raw_phase"}
            else:
                expected_missing = ({"velocity_memory"}
                                    if config.velocity_memory_rank is None else
                                    {"velocity_memory_left", "velocity_memory_right",
                                     "velocity_memory_coefficients"})
            if (set(incompatible.missing_keys) != expected_missing
                    or incompatible.unexpected_keys):
                raise ValueError(
                    "Memory warm start may add only the configured velocity-memory parameters")
        elif removing_structured_damping:
            incompatible = model.load_state_dict(
                initial_checkpoint["model_state"], strict=False)
            expected_unexpected = ({"drift_model.damping_factor.entries"}
                                   if config.damped_operator == "spsd" else
                                   {"drift_model.raw_damping"})
            if incompatible.missing_keys or set(incompatible.unexpected_keys) != expected_unexpected:
                raise ValueError(
                    "Restoring-only warm start may drop only the structured damping parameter")
        else:
            model.load_state_dict(initial_checkpoint["model_state"])
        if train_mode != "memory":
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
        if train_mode != "memory":
            model.set_normalization(trimmed_stats)
        trim_report.update(
            train_trim_state_std=trim_state_std.tolist(),
            normalization_fit=("checkpoint" if train_mode == "memory"
                               else "retained_training_transitions"),
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
    restoring_velocity_adjustment = None
    restoring_velocity_adjustment_report = None
    if restoring_orthogonality_adjust_velocity:
        (restoring_velocity_adjustment,
         restoring_velocity_adjustment_report) = _fit_restoring_velocity_adjustment(
            data, model, max(batch_size, data.batch_size),
            trim_cutoff, trim_state_std)
        restoring_velocity_adjustment = {
            key: value.to(device) for key, value in
            restoring_velocity_adjustment.items()}
        velocity_condition_number = restoring_velocity_adjustment_report[
            "velocity_covariance_condition_number"]
        position_condition_number = restoring_velocity_adjustment_report[
            "position_covariance_condition_number"]
        velocity_condition_text = (
            f"{velocity_condition_number:.5g}"
            if velocity_condition_number is not None else "singular")
        position_condition_text = (
            f"{position_condition_number:.5g}"
            if position_condition_number is not None else "singular")
        print(
            "Symmetric orthogonality cross adjustment: "
            f"velocity rank={restoring_velocity_adjustment_report['velocity_covariance_rank']}/"
            f"{config.reduced_dim}, condition={velocity_condition_text}; "
            f"position rank={restoring_velocity_adjustment_report['position_covariance_rank']}/"
            f"{config.reduced_dim}, condition={position_condition_text}; "
            "max |corr(z_tilde, velocity)|="
            f"{restoring_velocity_adjustment_report['max_abs_adjusted_position_velocity_correlation']:.5g}, "
            "max |corr(v_tilde, position)|="
            f"{restoring_velocity_adjustment_report['max_abs_adjusted_velocity_position_correlation']:.5g}",
            flush=True)
    oscillator_report = None
    forcing_filter_report = None
    if config.drift_type == "imported_oscillator_residual":
        if initial_checkpoint is None:
            from .imported_oscillator import configure_imported_oscillator
            oscillator_report = configure_imported_oscillator(model, data, oscillator_artifact)
        else:
            if not model.drift_model.ready:
                raise ValueError("Checkpoint is missing configured oscillator operators")
            model.drift_model.refresh_normalization(model)
            oscillator_report = initial_checkpoint.get("training", {}).get("imported_oscillator")
            if oscillator_report is None:
                raise ValueError("Checkpoint is missing imported oscillator provenance")
        if config.forcing_filter == "modal_biquad":
            from .imported_oscillator import (
                configure_modal_forcing_filter, prepare_modal_forcing_cache,
            )
            if initial_checkpoint is None:
                forcing_filter_report = configure_modal_forcing_filter(
                    model, forcing_filter_artifact, oscillator_report)
            else:
                if not model.drift_model.filter_ready:
                    raise ValueError("Checkpoint is missing configured modal forcing filter")
                forcing_filter_report = initial_checkpoint.get("training", {}).get(
                    "forcing_filter")
                if forcing_filter_report is None:
                    raise ValueError("Checkpoint is missing forcing-filter provenance")
            forcing_filter_report = dict(
                forcing_filter_report,
                cache=prepare_modal_forcing_cache(data, model))
            print(
                "Modal forcing cache: retained "
                f"{forcing_filter_report['cache']['retained_fraction']:.3%}; "
                "inverse reconstruction relative error "
                f"{forcing_filter_report['cache']['inverse_reconstruction_relative_error']:.3g}",
                flush=True)
    capillary_report = _configure_capillary_energy(model, data)
    if (config.drift_type == "capillary_energy" and initial_checkpoint is not None
            and initial_checkpoint.get("training", {}).get("train_mode") == "diffusion"):
        model.drift_model.reset_structured_terms(config.stiffness_init, config.damping_init)
    if train_mode == "diffusion":
        # Zero the trainable drift; imported oscillator buffers remain active.
        model.drift_model.zero_output()
    model_step = (f"{config.lag_steps:g} frames" if config.state_variable == "velocity" else
                  f"{config.lag_steps * config.dt:.9g} s")
    print(f"train_mode={train_mode} state_variable={config.state_variable} "
          f"lag_steps={config.lag_steps} h={model_step}", flush=True)
    trainable = set_trainable_components(model, train_mode)
    if multistep_weight > 0:
        _, multistep_gradient_parameter_names = _multistep_gradient_parameters(
            model, multistep_gradient_groups)
        print(
            "Multistep gradient parameters "
            f"({', '.join(multistep_gradient_groups)}): "
            + ", ".join(multistep_gradient_parameter_names), flush=True)
    else:
        multistep_gradient_parameter_names = ()
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    objective = EulerMaruyamaNLL()
    if multistep_weight <= 0:
        multistep_loss_objective = None
    elif multistep_objective == "energy_score":
        multistep_loss_objective = MultistepVelocityEnergyScore(
            multistep_horizons, multistep_particles,
            mode_counts=multistep_mode_counts,
            pod_coordinate_offset=int(data.include_spatial_mean),
            particle_counts=multistep_particle_counts,
            target=multistep_target,
            position_weight=multistep_position_weight,
            velocity_weight=multistep_velocity_weight,
            horizon_weights=multistep_horizon_weights)
    else:
        multistep_loss_objective = MonteCarloMultistepNLL(
            multistep_horizons, multistep_particles, target=multistep_target,
            mode_counts=multistep_mode_counts,
            pod_coordinate_offset=int(data.include_spatial_mean),
            particle_counts=multistep_particle_counts,
            horizon_weights=multistep_horizon_weights)
    if multistep_loss_objective is not None:
        print(
            f"Multistep objective={multistep_objective} target={multistep_target} "
            f"horizons={multistep_horizons} horizon_weights={multistep_horizon_weights} "
            f"particles={multistep_particle_counts} position_weight={multistep_position_weight:g} "
            f"velocity_weight={multistep_velocity_weight:g}",
            flush=True)
    correlation_reference = None
    correlation_reference_report = None
    if stationary_lagged_covariance_weight > 0 and stationary_lagged_covariance_normalization == "training":
        from .correlation import fit_correlation_reference
        correlation_reference, correlation_reference_report = fit_correlation_reference(
            data, model, tuple(sorted(set(stationary_mean_horizons +
                tuple(stationary_lagged_covariance_horizons or ())))), trim_cutoff, trim_state_std)
    if (stationary_mean_weight > 0 or stationary_variance_weight > 0
            or stationary_lagged_covariance_weight > 0):
        stationary_mean_objective = StationaryMeanLoss(
            stationary_mean_horizons,
            pod_coordinate_offset=int(data.include_spatial_mean),
            variance_pod_mode_count=stationary_variance_pod_mode_count,
            lagged_covariance_mode_count=stationary_lagged_covariance_mode_count,
            lagged_covariance_epsilon=stationary_lagged_covariance_epsilon,
            lagged_covariance_coordinates=stationary_lagged_covariance_coordinates,
            lagged_covariance_reference_variance=correlation_reference,
            lagged_covariance_horizons=stationary_lagged_covariance_horizons).to(device)
        print(
            f"Endpoint moment target: matching observed h={stationary_mean_horizons} "
            "endpoints; each retained start produces two independently noised model "
            "paths shared through the maximum horizon, with at most one window per "
            "shuffled source block.", flush=True)
    else:
        stationary_mean_objective = None
    stability_objective = None
    if stability.weight > 0:
        def retained_training_states():
            for batch in _sde_transition_batches(
                    data, config, "train", batch_size):
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
                                  multistep_objective=multistep_objective,
                                  multistep_target=multistep_target,
                                  multistep_horizons=list(multistep_horizons),
                                  multistep_horizon_weights=list(
                                      multistep_horizon_weights),
                                  multistep_position_weight=multistep_position_weight,
                                  multistep_velocity_weight=multistep_velocity_weight,
                                  multistep_mode_counts=(
                                      None if multistep_mode_counts is None else
                                      list(multistep_mode_counts)),
                                  multistep_particles=multistep_particles,
                                  multistep_particle_counts=list(multistep_particle_counts),
                                  multistep_window_fraction=multistep_window_fraction,
                                  multistep_window_sampling=multistep_window_sampling,
                                  multistep_gradient_groups=list(multistep_gradient_groups),
                                  multistep_gradient_parameter_names=list(
                                      multistep_gradient_parameter_names),
                                  multistep_sampler=(
                                      "independent complete-path-trimmed long-window batches"),
                                  multistep_objective_definition=(
                                      (("Per-start conditional joint displacement-velocity "
                                        "energy score in state_std and target_std coordinates "
                                        "with configured block weights, ordinary Euclidean "
                                        "distances, and all distinct particle pairs")
                                       if multistep_target == "joint" else
                                       ("Per-start conditional velocity energy score in "
                                        "target_std coordinates with ordinary Euclidean distances "
                                        "and all distinct particle pairs"))
                                      if multistep_objective == "energy_score" else
                                      "Monte Carlo Gaussian-mixture endpoint negative "
                                      "log likelihood"),
                                  one_step_population=(
                                      "all retained one-step transitions, independent of "
                                      "multistep horizons"),
                                  multistep_validation_seed=multistep_validation_seed,
                                  tcn_rollout_cache=tcn_rollout_cache,
                                  tcn_rollout_compile=tcn_rollout_compile,
                                  indexed_training_batches=indexed_training_batches,
                                  history_noise_weight=history_noise_weight,
                                  history_noise_std=history_noise_std,
                                  history_noise_validation_seed=history_noise_validation_seed,
                                  history_noise_definition=(
                                      "Drift-only coordinate-mean velocity MSE scaled by saved "
                                      "target_std; IID velocity perturbations integrated backwards "
                                      "with current position anchored; clean targets and trim unchanged"),
                                  stationary_mean_weight=stationary_mean_weight,
                                  stationary_variance_weight=stationary_variance_weight,
                                  stationary_variance_pod_mode_count=(
                                      stationary_variance_pod_mode_count),
                                  stationary_variance_coordinates=(
                                      (["spatial_mean"] if data.include_spatial_mean else [])
                                      + [f"POD {mode}" for mode in range(
                                          1,
                                          (data.rank if stationary_variance_pod_mode_count is None
                                           else stationary_variance_pod_mode_count) + 1)]),
                                  stationary_lagged_covariance_weight=(
                                      stationary_lagged_covariance_weight),
                                  stationary_lagged_covariance_mode_count=(
                                      stationary_lagged_covariance_mode_count),
                                  stationary_lagged_covariance_epsilon=(
                                      stationary_lagged_covariance_epsilon),
                                  stationary_mean_horizon=stationary_mean_horizons[-1],
                                  stationary_mean_horizons=list(stationary_mean_horizons),
                                  stationary_mean_trajectories=stationary_mean_trajectories,
                                  stationary_mean_batch_fraction=(
                                      stationary_mean_batch_fraction),
                                  stationary_mean_validation_seed=(
                                      stationary_mean_validation_seed),
                                  restoring_orthogonality_weight=(
                                      restoring_orthogonality_weight),
                                  damping_orthogonality_weight=(
                                      damping_orthogonality_weight),
                                  restoring_orthogonality_epsilon=(
                                      restoring_orthogonality_epsilon),
                                  restoring_orthogonality_adjust_velocity=(
                                      restoring_orthogonality_adjust_velocity),
                                  diagonal_linear_orthogonality_weight=(
                                      diagonal_linear_orthogonality_weight),
                                  diagonal_linear_orthogonality_epsilon=(
                                      diagonal_linear_orthogonality_epsilon),
                                  diagonal_linear_orthogonality_coordinates=(
                                      "each normalized output coordinate projected jointly "
                                      "onto its centered normalized position and velocity"),
                                  diagonal_linear_orthogonality_definition=(
                                      "sum_i b_i^T (C_i + epsilon I)^-1 b_i, evaluated "
                                      "per minibatch after centering normalized position, "
                                      "velocity, and residual; C_i is the 2x2 covariance "
                                      "of (z_i, v_i) and b_i its cross-covariance with "
                                      "residual_i"),
                                  orthogonality_symmetric_cross_adjustment=(
                                      restoring_orthogonality_adjust_velocity),
                                  restoring_orthogonality_coordinates=(
                                      "all normalized position coordinates owned by the "
                                      "diagonal restoring operator, including spatial mean "
                                      "when present"),
                                  restoring_orthogonality_definition=(
                                      "sum_i mean(z_tilde_i * residual_i)^2 / "
                                      "(mean(z_tilde_i^2) + epsilon), evaluated per minibatch; "
                                      "z_tilde is velocity-adjusted when configured and otherwise z"),
                                  damping_orthogonality_coordinates=(
                                      "all normalized velocity coordinates owned by the "
                                      "diagonal damping operator"),
                                  damping_orthogonality_definition=(
                                      "sum_i mean(v_tilde_i * residual_i)^2 / "
                                      "(mean(v_tilde_i^2) + epsilon), evaluated per minibatch; "
                                      "v_tilde is position-adjusted when configured and otherwise v"),
                                  stationary_mean_gradient_parameters=(
                                      "mean and lagged covariance: drift only; variance: every "
                                      "trainable drift and diffusion parameter"),
                                  stationary_mean_start_population=(
                                      "globally mixed velocity windows whose complete measured "
                                      "path through the endpoint passes training trimming; at most "
                                      "one window per shuffled source block"),
                                  stationary_mean_target_population=(
                                      "matching measured endpoints at the configured horizons"),
                                  stationary_mean_generated_paths_per_start=2,
                                  stationary_mean_source_block_size=min(
                                      256, data.batch_size, batch_size),
                                  stationary_mean_definition=(
                                      "coordinate mean of the cross-product between two independent "
                                      "normalized generated-minus-measured endpoint mean errors, "
                                      "averaged equally across configured horizons; "
                                      "unbiased over rollout noise for the sampled endpoint batch and "
                                      "finite-batch values may be negative"),
                                  stationary_variance_definition=(
                                      "coordinate mean of the cross-product between two independent "
                                      "generated-minus-measured normalized endpoint sample-variance "
                                      "errors, averaged equally across configured horizons; sample "
                                      "variances use N-1 and finite-batch values may be negative"),
                                  stationary_lagged_covariance_coordinates=(
                                      list(stationary_lagged_covariance_coordinates)
                                      if stationary_lagged_covariance_coordinates is not None else None),
                                  stationary_lagged_covariance_coordinate_selection=(
                                      "explicit state coordinates; coordinate 0 is spatial mean when present"
                                      if stationary_lagged_covariance_coordinates is not None else
                                      "first configured shared-POD modes; spatial mean excluded"),
                                  stationary_lagged_covariance_normalization=stationary_lagged_covariance_normalization,
                                  stationary_lagged_covariance_horizons=(list(stationary_lagged_covariance_horizons)
                                      if stationary_lagged_covariance_horizons is not None else list(stationary_mean_horizons)),
                                  stationary_lagged_covariance_reference=correlation_reference_report,
                                  stationary_lagged_covariance_definition=(
                                      "mode mean of the cross-product between two independent "
                                      "generated-minus-measured centered start-to-endpoint sample "
                                      "covariance errors, divided by the configured batch or fixed training variance "
                                      "plus epsilon and averaged equally across horizons; covariance "
                                      "uses N-1 and finite-batch values may be negative"),
                                  **(trim_report or {}),
                                  diffusion_covariance_space=covariance_space,
                                  capillary_energy=capillary_report,
                                  imported_oscillator=oscillator_report,
                                  forcing_filter=forcing_filter_report,
                                  initial_checkpoint=(str(Path(initial_checkpoint_path).resolve())
                                                      if initial_checkpoint_path else None)),
                    split_labels={k: list(v) for k, v in data.splits.items()})
    if stability_objective is not None:
        metadata["training"]["stability"] = stability_objective.to_metadata()
    if restoring_velocity_adjustment_report is not None:
        metadata["training"]["restoring_orthogonality_velocity_adjustment"] = (
            restoring_velocity_adjustment_report)
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best = float("inf")
    for epoch in range(epochs):
        train_metrics, validation_metrics = {}, {}
        train_loss = _epoch(
            model, data, "train", batch_size, device, objective, optimizer,
            seed=seed + epoch, train_mode=train_mode,
            history_noise_weight=history_noise_weight,
            history_noise_std=history_noise_std,
            history_noise_seed=seed + epoch,
            train_trim_cutoff=trim_cutoff,
            train_trim_state_std=trim_state_std,
            multistep_objective=multistep_loss_objective,
            multistep_weight=multistep_weight,
            multistep_window_fraction=multistep_window_fraction,
            multistep_window_sampling=multistep_window_sampling,
            multistep_gradient_groups=multistep_gradient_groups,
            multistep_window_seed=seed + epoch,
            stationary_mean_objective=stationary_mean_objective,
            stationary_mean_weight=stationary_mean_weight,
            stationary_variance_weight=stationary_variance_weight,
            stationary_lagged_covariance_weight=(
                stationary_lagged_covariance_weight),
            stationary_mean_trajectories=stationary_mean_trajectories,
            stationary_mean_batch_fraction=stationary_mean_batch_fraction,
            stationary_mean_noise_seed=seed + epoch,
            stationary_mean_window_seed=seed + epoch,
            stability_objective=stability_objective,
            stability_noise_seed=seed + epoch,
            restoring_orthogonality_weight=restoring_orthogonality_weight,
            damping_orthogonality_weight=damping_orthogonality_weight,
            restoring_orthogonality_epsilon=restoring_orthogonality_epsilon,
            diagonal_linear_orthogonality_weight=(
                diagonal_linear_orthogonality_weight),
            diagonal_linear_orthogonality_epsilon=(
                diagonal_linear_orthogonality_epsilon),
            restoring_velocity_adjustment=restoring_velocity_adjustment,
            metric_output=train_metrics)
        validation_loss = _epoch(
            model, data, "validation", batch_size, device, objective,
            train_mode=train_mode, train_trim_cutoff=trim_cutoff,
            history_noise_weight=history_noise_weight,
            history_noise_std=history_noise_std,
            history_noise_seed=history_noise_validation_seed,
            train_trim_state_std=trim_state_std,
            multistep_objective=multistep_loss_objective,
            multistep_weight=multistep_weight,
            multistep_noise_seed=(multistep_validation_seed
                                  if multistep_weight > 0 else None),
            multistep_window_fraction=multistep_window_fraction,
            multistep_window_sampling=multistep_window_sampling,
            multistep_gradient_groups=multistep_gradient_groups,
            multistep_window_seed=multistep_validation_seed,
            stationary_mean_objective=stationary_mean_objective,
            stationary_mean_weight=stationary_mean_weight,
            stationary_variance_weight=stationary_variance_weight,
            stationary_lagged_covariance_weight=(
                stationary_lagged_covariance_weight),
            stationary_mean_trajectories=stationary_mean_trajectories,
            stationary_mean_batch_fraction=stationary_mean_batch_fraction,
            stationary_mean_noise_seed=stationary_mean_validation_seed,
            stationary_mean_window_seed=stationary_mean_validation_seed,
            stability_objective=stability_objective,
            stability_noise_seed=stability.validation_seed,
            restoring_orthogonality_weight=restoring_orthogonality_weight,
            damping_orthogonality_weight=damping_orthogonality_weight,
            restoring_orthogonality_epsilon=restoring_orthogonality_epsilon,
            diagonal_linear_orthogonality_weight=(
                diagonal_linear_orthogonality_weight),
            diagonal_linear_orthogonality_epsilon=(
                diagonal_linear_orthogonality_epsilon),
            restoring_velocity_adjustment=restoring_velocity_adjustment,
            metric_output=validation_metrics)
        record = dict(epoch=epoch + 1,
                      train_nll=train_metrics["one_step_nll"],
                      validation_nll=validation_metrics["one_step_nll"])
        if history_noise_weight > 0:
            record.update(train_total_loss=train_loss, validation_total_loss=validation_loss,
                          history_noise_weight=history_noise_weight,
                          history_noise_std=history_noise_std)
            for prefix, values in (("train", train_metrics), ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value for key, value in values.items()
                               if key.startswith("history_noise_")})
        if stability_objective is not None:
            record.update(train_total_loss=train_loss, validation_total_loss=validation_loss)
            for prefix, values in (("train", train_metrics), ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value for key, value in values.items()
                               if key.startswith("stability_")})
        if multistep_weight > 0:
            record.update(train_total_loss=train_loss,
                          validation_total_loss=validation_loss,
                          multistep_objective=multistep_objective,
                          multistep_target=multistep_target,
                          multistep_horizon_weights=list(multistep_horizon_weights),
                          multistep_position_weight=multistep_position_weight,
                          multistep_velocity_weight=multistep_velocity_weight,
                          multistep_mode_counts=(
                              None if multistep_mode_counts is None else
                              list(multistep_mode_counts)),
                          multistep_particle_counts=list(multistep_particle_counts),
                          multistep_window_fraction=multistep_window_fraction,
                          multistep_window_sampling=multistep_window_sampling,
                          multistep_gradient_groups=list(multistep_gradient_groups),
                          train_multistep_windows_available=train_metrics["window_count"],
                          validation_multistep_windows_available=(
                              validation_metrics["window_count"]),
                          train_multistep_windows_scored=(
                              train_metrics["multistep_window_count"]),
                          validation_multistep_windows_scored=(
                              validation_metrics["multistep_window_count"]))
            record.update({f"train_{key}": value for key, value in train_metrics.items()
                           if key.startswith(("multistep_loss_", "multistep_nll_",
                                              "multistep_energy_score_",
                                              "multistep_weighted_loss_"))})
            record.update({f"validation_{key}": value
                           for key, value in validation_metrics.items()
                           if key.startswith(("multistep_loss_", "multistep_nll_",
                                              "multistep_energy_score_",
                                              "multistep_weighted_loss_"))})
        if (stationary_mean_weight > 0 or stationary_variance_weight > 0
                or stationary_lagged_covariance_weight > 0):
            record.update(
                train_total_loss=train_loss,
                validation_total_loss=validation_loss,
                stationary_mean_weight=stationary_mean_weight,
                stationary_variance_weight=stationary_variance_weight,
                stationary_variance_pod_mode_count=(
                    stationary_variance_pod_mode_count),
                stationary_lagged_covariance_weight=(
                    stationary_lagged_covariance_weight),
                stationary_lagged_covariance_mode_count=(
                    stationary_lagged_covariance_mode_count),
                stationary_lagged_covariance_epsilon=(
                    stationary_lagged_covariance_epsilon),
                stationary_lagged_covariance_coordinates=(list(stationary_lagged_covariance_coordinates)
                    if stationary_lagged_covariance_coordinates is not None else None),
                stationary_lagged_covariance_normalization=stationary_lagged_covariance_normalization,
                stationary_lagged_covariance_horizons=list(stationary_lagged_covariance_horizons or stationary_mean_horizons),
                stationary_mean_horizon=stationary_mean_horizons[-1],
                stationary_mean_horizons=list(stationary_mean_horizons),
                stationary_mean_trajectories=stationary_mean_trajectories,
                stationary_mean_batch_fraction=stationary_mean_batch_fraction,
            )
            for prefix, values in (("train", train_metrics),
                                   ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value for key, value in values.items()
                               if (key.startswith("stationary_mean_")
                                   or key.startswith("stationary_variance_")
                                   or key.startswith(
                                       "stationary_lagged_covariance_"))})
        if restoring_orthogonality_weight > 0:
            record.update(
                train_total_loss=train_loss,
                validation_total_loss=validation_loss,
                restoring_orthogonality_weight=restoring_orthogonality_weight,
                restoring_orthogonality_epsilon=restoring_orthogonality_epsilon,
                restoring_orthogonality_adjust_velocity=(
                    restoring_orthogonality_adjust_velocity))
            for prefix, values in (("train", train_metrics),
                                   ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value
                               for key, value in values.items()
                               if (key.startswith("restoring_orthogonality_")
                                   or key == "restoring_residual_projection_coefficients")})
        if damping_orthogonality_weight > 0:
            record.update(
                train_total_loss=train_loss,
                validation_total_loss=validation_loss,
                damping_orthogonality_weight=damping_orthogonality_weight,
                restoring_orthogonality_epsilon=restoring_orthogonality_epsilon)
            for prefix, values in (("train", train_metrics),
                                   ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value
                               for key, value in values.items()
                               if (key.startswith("damping_orthogonality_")
                                   or key == "damping_residual_projection_coefficients")})
        if diagonal_linear_orthogonality_weight > 0:
            record.update(
                train_total_loss=train_loss,
                validation_total_loss=validation_loss,
                diagonal_linear_orthogonality_weight=(
                    diagonal_linear_orthogonality_weight),
                diagonal_linear_orthogonality_epsilon=(
                    diagonal_linear_orthogonality_epsilon))
            for prefix, values in (("train", train_metrics),
                                   ("validation", validation_metrics)):
                record.update({f"{prefix}_{key}": value
                               for key, value in values.items()
                               if (key.startswith("diagonal_linear_orthogonality_")
                                   or key.startswith(
                                       "diagonal_linear_residual_"))})
        for prefix, values in (("train", train_metrics), ("validation", validation_metrics)):
            record.update({f"{prefix}_{key}": value for key, value in values.items()
                           if key.startswith("imported_")})
        if train_mode == "diffusion":
            record.update(compute_diffusion_validation_metric(
                model, data, diffusion_trim_percent, batch_size, covariance_space))
        if model.memory_parameters():
            model.validate_velocity_memory()
            if model.config.velocity_memory_offsets is not None:
                norms = model.velocity_memory_kernel_norms()
                record.update(velocity_memory_kernel_norms=norms,
                              velocity_memory_max_kernel_norm=max(norms))
            if model.config.velocity_oscillator_periods is not None:
                record.update(
                    velocity_oscillator_radii=(
                        model.velocity_oscillator_radius().detach().cpu().tolist()),
                    velocity_oscillator_periods_learned=(
                        (2 * math.pi / model.velocity_oscillator_phase())
                        .detach().cpu().tolist()),
                    velocity_oscillator_coefficients=(
                        (model.config.velocity_oscillator_norm_cap
                         * torch.tanh(model.velocity_oscillator_coefficients))
                        .detach().cpu().tolist()))
            if model.config.velocity_innovation_offsets is not None:
                norms = model.velocity_innovation_kernel_norms()
                record.update(
                    velocity_innovation_kernel_norms=norms,
                    velocity_innovation_kernel_norm_sum=sum(norms),
                    velocity_innovation_coefficients=(
                        model.velocity_innovation_coefficients_bounded()
                        .detach().cpu().tolist()))
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        message = (f"epoch={epoch + 1} train_nll={record['train_nll']:.5g} "
                   f"validation_nll={record['validation_nll']:.5g}")
        if history_noise_weight > 0:
            message += (f" train_history_noise={train_metrics['history_noise_loss']:.5g}"
                        f" validation_history_noise={validation_metrics['history_noise_loss']:.5g}"
                        f" train_total={train_loss:.5g} validation_total={validation_loss:.5g}")
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
                            f"{record[f'train_multistep_loss_h{horizon}']:.5g}"
                            f" validation_h{horizon}="
                            f"{record[f'validation_multistep_loss_h{horizon}']:.5g}")
        if stationary_mean_weight > 0:
            for horizon in stationary_mean_horizons:
                message += (
                    f" train_mean_h{horizon}="
                    f"{record[f'train_stationary_mean_loss_h{horizon}']:.5g}"
                    f" validation_mean_h{horizon}="
                    f"{record[f'validation_stationary_mean_loss_h{horizon}']:.5g}")
            message += (
                f" train_mean_weighted="
                f"{record['train_stationary_mean_weighted_contribution']:.5g}"
                f" validation_mean_weighted="
                f"{record['validation_stationary_mean_weighted_contribution']:.5g}"
                f" train_mean_rmse="
                f"{record['train_stationary_mean_normalized_error_rmse']:.5g}"
                f" validation_mean_rmse="
                f"{record['validation_stationary_mean_normalized_error_rmse']:.5g}")
        if stationary_variance_weight > 0:
            for horizon in stationary_mean_horizons:
                message += (
                    f" train_var_h{horizon}="
                    f"{record[f'train_stationary_variance_loss_h{horizon}']:.5g}"
                    f" validation_var_h{horizon}="
                    f"{record[f'validation_stationary_variance_loss_h{horizon}']:.5g}")
            message += (
                f" train_var_weighted="
                f"{record['train_stationary_variance_weighted_contribution']:.5g}"
                f" validation_var_weighted="
                f"{record['validation_stationary_variance_weighted_contribution']:.5g}"
                f" train_var_rmse="
                f"{record['train_stationary_variance_selected_normalized_error_rmse']:.5g}"
                f" validation_var_rmse="
                f"{record['validation_stationary_variance_selected_normalized_error_rmse']:.5g}")
        if stationary_lagged_covariance_weight > 0:
            for horizon in stationary_lagged_covariance_horizons or stationary_mean_horizons:
                message += (
                    f" train_lagcov_h{horizon}="
                    f"{record[f'train_stationary_lagged_covariance_loss_h{horizon}']:.5g}"
                    f" validation_lagcov_h{horizon}="
                    f"{record[f'validation_stationary_lagged_covariance_loss_h{horizon}']:.5g}")
            message += (
                f" train_lagcov_weighted="
                f"{record['train_stationary_lagged_covariance_weighted_contribution']:.5g}"
                f" validation_lagcov_weighted="
                f"{record['validation_stationary_lagged_covariance_weighted_contribution']:.5g}"
                f" train_lagcov_rmse="
                f"{record['train_stationary_lagged_covariance_normalized_error_rmse']:.5g}"
                f" validation_lagcov_rmse="
                f"{record['validation_stationary_lagged_covariance_normalized_error_rmse']:.5g}")
        if restoring_orthogonality_weight > 0:
            message += (
                f" train_orth={record['train_restoring_orthogonality_loss']:.5g}"
                f" validation_orth="
                f"{record['validation_restoring_orthogonality_loss']:.5g}"
                f" train_orth_weighted="
                f"{record['train_restoring_orthogonality_weighted_contribution']:.5g}"
                f" validation_orth_weighted="
                f"{record['validation_restoring_orthogonality_weighted_contribution']:.5g}")
        if damping_orthogonality_weight > 0:
            message += (
                f" train_damping_orth={record['train_damping_orthogonality_loss']:.5g}"
                f" validation_damping_orth="
                f"{record['validation_damping_orthogonality_loss']:.5g}"
                f" train_damping_orth_weighted="
                f"{record['train_damping_orthogonality_weighted_contribution']:.5g}"
                f" validation_damping_orth_weighted="
                f"{record['validation_damping_orthogonality_weighted_contribution']:.5g}")
        if diagonal_linear_orthogonality_weight > 0:
            message += (
                f" train_linear_orth="
                f"{record['train_diagonal_linear_orthogonality_loss']:.5g}"
                f" validation_linear_orth="
                f"{record['validation_diagonal_linear_orthogonality_loss']:.5g}"
                f" train_linear_orth_weighted="
                f"{record['train_diagonal_linear_orthogonality_weighted_contribution']:.5g}"
                f" validation_linear_orth_weighted="
                f"{record['validation_diagonal_linear_orthogonality_weighted_contribution']:.5g}")
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
    if model.fixed_memory is not None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("fixed_memory.readout."))
    model.tcn_rollout_cache = checkpoint.get("training", {}).get("tcn_rollout_cache", False)
    # Compilation is training-execution policy, not required to load/evaluate
    # a checkpoint. Avoid an unexpected compiler startup in diagnostics.
    model.tcn_rollout_compile = False
    model.indexed_training_batches = checkpoint.get("training", {}).get("indexed_training_batches", False)
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
    if multistep_target not in {"velocity", "displacement", "joint"}:
        raise ValueError("Multistep target must be velocity, displacement, or joint")
    training_lag = model.config.lag_steps
    if rollout_lag is not None:
        if rollout_lag < 1:
            raise ValueError("rollout_lag must be positive")
        if model.config.state_variable == "increment" and rollout_lag != training_lag:
            raise ValueError("Increment-state rollouts require the checkpoint training lag")
        if model.fixed_memory is not None and rollout_lag != training_lag:
            raise ValueError("Fixed memory requires the checkpoint native sampling interval")
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
                  temporal_preprocessing=data.temporal_preprocessing(),
                  spatial_mean_highpass_hz=data.spatial_mean_highpass_hz,
                  training_lag_steps=training_lag,
                  state_variable=model.config.state_variable,
                  drift_type=model.config.drift_type,
                  residual_architecture=model.config.residual_architecture,
                  residual_input=model.config.residual_input,
                  tcn_context_frames=(model.config.tcn_context_frames
                                      if model.config.residual_architecture == "tcn" else None),
                  tcn_receptive_field=(model.config.tcn_receptive_field
                                       if model.config.residual_architecture == "tcn" else None),
                  tcn_rollout_cache_requested=model.tcn_rollout_cache,
                  tcn_rollout_cache_active=model.new_tcn_cache() is not None,
                  damped_operator=(model.config.damped_operator
                                   if model.config.drift_type == "damped_residual" else None),
                  structured_damping=(model.config.structured_damping
                                      if model.config.drift_type == "damped_residual" else None),
                  diffusion_type=model.config.diffusion_type,
                  diffusion_scale=diffusion_scale,
                  diffusion_log_range=(model.config.diffusion_log_range
                                       if model.config.diffusion_type == "bounded_state_diagonal"
                                       else None),
                  history_steps=model.config.history_steps,
                  history_offsets=list(model.config.effective_history_offsets),
                  velocity_memory_offsets=(
                      None if model.config.velocity_memory_offsets is None else
                      list(model.config.velocity_memory_offsets)),
                  velocity_memory_rank=model.config.velocity_memory_rank,
                  velocity_memory_norm_cap=(
                      model.config.velocity_memory_norm_cap
                      if model.config.velocity_memory_rank is not None else None),
                  velocity_memory_kernel_norms=model.velocity_memory_kernel_norms(),
                  velocity_oscillator_periods=(
                      None if model.config.velocity_oscillator_periods is None else
                      list(model.config.velocity_oscillator_periods)),
                  velocity_oscillator_burnin=(
                      model.config.velocity_oscillator_burnin
                      if model.config.velocity_oscillator_periods is not None else None),
                  velocity_oscillator_radii=(
                      model.velocity_oscillator_radius().detach().cpu().tolist()),
                  velocity_oscillator_periods_learned=(
                      [] if model.config.velocity_oscillator_periods is None else
                      (2 * math.pi / model.velocity_oscillator_phase())
                      .detach().cpu().tolist()),
                  velocity_innovation_offsets=(
                      None if model.config.velocity_innovation_offsets is None else
                      list(model.config.velocity_innovation_offsets)),
                  velocity_innovation_rank=model.config.velocity_innovation_rank,
                  velocity_innovation_norm_cap=(
                      model.config.velocity_innovation_norm_cap
                      if model.config.velocity_innovation_offsets is not None else None),
                  velocity_innovation_kernel_norms=(
                      model.velocity_innovation_kernel_norms()),
                  train_trim_percent=training.get("train_trim_percent", 0.),
                  train_trim_norm_cutoff=trim_cutoff,
                  checkpoint=str(Path(checkpoint_path).resolve()), model_type="neural_sde")
    if multistep_horizons is not None:
        if model.config.state_variable != "velocity":
            raise ValueError("Multistep likelihood evaluation requires a velocity checkpoint")
        if multistep_target == "joint":
            raise ValueError(
                "Optional multistep likelihood evaluation does not support joint energy-score "
                "targets; use the saved training metrics")
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
    elif model.config.drift_type == "imported_oscillator_residual":
        core = model.drift_model
        report.update(
            imported_oscillator=checkpoint.get("training", {}).get("imported_oscillator"),
            forcing_filter=checkpoint.get("training", {}).get("forcing_filter"),
            physical_stiffness_matrix=core.physical_stiffness.cpu().tolist(),
            physical_damping_matrix=core.physical_damping.cpu().tolist(),
            physical_equilibrium=core.equilibrium.cpu().tolist(),
            normalized_stiffness_matrix=core.stiffness.cpu().tolist(),
            normalized_damping_matrix=core.damping.cpu().tolist(),
            normalized_drift_bias=core.bias.cpu().tolist(),
            structured_drift_coordinate_space="affine standardized position and native-frame velocity",
            structured_drift_time_unit="one native frame",
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
    training = commands.add_parser(
        "train", help="fit joint, diffusion-only, drift-only, or velocity-memory NLL")
    training.add_argument("--experiment", help="Required when starting without a checkpoint.")
    training.add_argument("--rank", type=int, help="Required when starting without a checkpoint.")
    training.add_argument("--temporal-component-manifest", type=Path,
                          help="Prepared offline temporal-component manifest for fresh training.")
    training.add_argument("--temporal-component",
                          help="Component in --temporal-component-manifest, for example remainder.")
    training.add_argument(
        "--train-mode", choices=("joint", "diffusion", "drift", "memory"),
        default="joint")
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
    training.add_argument(
        "--velocity-memory-offsets", type=int, nargs="+",
        help="Previous native-frame velocity offsets for a zero-initialized diagonal "
             "drift-memory correction, for example 1 2 5 10 20 50.",
    )
    training.add_argument(
        "--velocity-memory-rank", type=int,
        help="Shared cross-coordinate velocity-memory rank; unset keeps diagonal memory.",
    )
    training.add_argument(
        "--velocity-memory-norm-cap", type=float,
        help="Per-offset induced-norm cap for low-rank velocity memory (default 0.25).",
    )
    training.add_argument(
        "--velocity-oscillator-periods", type=float, nargs="+",
        help="Initial periods in native frames for stable two-state recurrent oscillators.",
    )
    training.add_argument(
        "--velocity-oscillator-burnin", type=int,
        help="Observed native velocities used to initialize recurrent memory (default 256).",
    )
    training.add_argument(
        "--velocity-oscillator-radius-init", type=float,
        help="Initial recurrent decay radius in (0, 0.9999), default 0.98.",
    )
    training.add_argument(
        "--velocity-oscillator-norm-cap", type=float,
        help="Induced-norm cap for the recurrent drift correction (default 0.25).",
    )
    training.add_argument(
        "--velocity-innovation-offsets", type=int, nargs="+",
        help="Past parent-innovation offsets for causal autoregressive forcing.",
    )
    training.add_argument(
        "--velocity-innovation-rank", type=int,
        help="Low rank of the autoregressive innovation kernels.",
    )
    training.add_argument(
        "--velocity-innovation-norm-cap", type=float,
        help="Cap in (0,1) on the summed innovation-kernel norm (default 0.95).",
    )
    training.add_argument("--checkpoint", type=Path,
                          help="Prior run directory or best.pt; required for memory training.")
    training.add_argument("--diffusion-trim-percent", type=float, default=0.,
                          help="Remove this largest percentage of validation increment norms for E_diff (default 0).")
    training.add_argument("--train-trim-percent", type=float, default=0.,
                          help="Exclude this largest percentage of training increment norms from NLL (default 0).")
    training.add_argument(
        "--multistep-weight", type=float, default=0.,
        help="Weight of the selected conditional multistep objective (default 0: disabled).",
    )
    training.add_argument(
        "--multistep-objective", choices=("gmm_nll", "energy_score"), default="gmm_nll",
        help="Conditional endpoint objective: Gaussian-mixture NLL (default) or the "
             "sample energy score.",
    )
    training.add_argument(
        "--multistep-target", choices=("velocity", "displacement", "joint"), default="velocity",
        help="Score endpoint velocity (default), displacement for GMM NLL, or the joint "
             "standardized displacement-velocity endpoint for energy score; "
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
        "--multistep-horizon-weights", type=float, nargs="+",
        help="Optional nonnegative weight per matching horizon. Values are normalized to "
             "sum to one; default: equal horizon weights.",
    )
    training.add_argument(
        "--multistep-position-weight", type=float, default=1.,
        help="Positive squared-metric weight for the standardized displacement block of a "
             "joint energy score (default 1).",
    )
    training.add_argument(
        "--multistep-velocity-weight", type=float, default=1.,
        help="Positive squared-metric weight for the standardized velocity block of a joint "
             "energy score (default 1).",
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
        "--multistep-gradient-groups", nargs="+",
        choices=_MULTISTEP_GRADIENT_GROUPS, default=("drift",),
        help="Parameter groups receiving MC multistep gradients. The one-step loss still "
             "updates every normally trainable parameter. Default: drift (diffusion excluded).",
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
        "--stationary-variance-pod-mode-count", type=int, default=None,
        help="Restrict endpoint-variance matching to the spatial mean, when present, "
             "plus the first N shared-POD modes. N=0 selects only the spatial mean. "
             "Omitting this option preserves all-coordinate matching.",
    )
    training.add_argument(
        "--stationary-lagged-covariance-weight", type=float, default=0.,
        help="Weight of the centered start-to-endpoint covariance match for selected "
             "POD modes (default 0: disabled).",
    )
    training.add_argument(
        "--stationary-lagged-covariance-mode-count", type=int, default=None,
        help="Number of leading shared-POD modes in the lagged-covariance match; the "
             "spatial mean is excluded (default: all POD modes).",
    )
    training.add_argument(
        "--stationary-lagged-covariance-horizons", type=int, nargs="+",
        help="Independent correlation lags in model steps; leaves endpoint mean/variance horizons unchanged.",
    )
    training.add_argument(
        "--stationary-lagged-covariance-coordinates", type=int, nargs="+",
        help="Explicit increasing state coordinate indices; 0 includes the spatial mean when present. Mutually exclusive with mode count.",
    )
    training.add_argument(
        "--stationary-lagged-covariance-normalization", choices=("batch", "training"), default="batch",
        help="Use legacy batch variance or a frozen training-only variance over complete eligible stencils.",
    )
    training.add_argument(
        "--stationary-lagged-covariance-epsilon", type=float, default=1e-6,
        help="Positive start-variance denominator regularizer for lagged covariance "
             "(default 1e-6).",
    )
    training.add_argument(
        "--stationary-mean-horizon", type=int, default=100,
        help="Legacy single endpoint-moment horizon (default 100).",
    )
    training.add_argument(
        "--stationary-mean-horizons", type=int, nargs="+", default=None,
        help="Strictly increasing endpoint-moment horizons evaluated along one rollout; "
             "overrides --stationary-mean-horizon.",
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
    training.add_argument(
        "--restoring-orthogonality-weight", type=float, default=0.,
        help="Weight of the residual projection penalty onto coordinatewise diagonal "
             "restoring fields. Requires diagonal damped_residual; "
             "0 disables all associated work (default 0).",
    )
    training.add_argument(
        "--damping-orthogonality-weight", type=float, default=0.,
        help="Weight of the residual projection penalty onto coordinatewise diagonal "
             "velocity-damping fields. Requires structured diagonal damped_residual; "
             "0 disables all associated work (default 0).",
    )
    training.add_argument(
        "--restoring-orthogonality-epsilon", type=float, default=1e-8,
        help="Positive denominator regularizer for restoring orthogonality (default 1e-8).",
    )
    training.add_argument(
        "--restoring-orthogonality-adjust-velocity", action="store_true",
        help="Symmetrically partial out retained-training linear cross-predictability: "
             "position from normalized velocity for the restoring term and velocity "
             "from normalized position for the damping term. Requires a positive "
             "orthogonality weight.",
    )
    training.add_argument(
        "--diagonal-linear-orthogonality-weight", type=float, default=0.,
        help="Weight of the joint residual projection penalty onto each "
             "coordinate's normalized span(position_i, velocity_i). Requires "
             "structured diagonal damped_residual and cannot be combined with "
             "the legacy orthogonality options; 0 disables it (default 0).",
    )
    training.add_argument(
        "--diagonal-linear-orthogonality-epsilon", type=float, default=1e-6,
        help="Positive ridge added to each centered 2x2 position/velocity "
             "covariance in the joint diagonal projection (default 1e-6).",
    )
    training.add_argument("--tcn-rollout-cache", action=argparse.BooleanOptionalAction,
                          default=False, help="Use differentiable streaming TCN execution for full-receptive-field rollouts; shorter contexts fall back to the reference path.")
    training.add_argument("--indexed-training-batches", action=argparse.BooleanOptionalAction,
                          default=False, help="Use indexed velocity windows and cached trim masks without changing example order or eligibility.")
    training.add_argument("--tcn-rollout-compile", action=argparse.BooleanOptionalAction,
                          default=False, help="Compile the cached TCN kernels; requires --tcn-rollout-cache and incurs first-use compilation.")
    training.add_argument("--history-noise-weight", type=float, default=0.,
                          help="Drift-only noisy-history velocity-MSE weight; 0 disables augmentation.")
    training.add_argument("--history-noise-std", type=float, default=.05,
                          help="Velocity perturbation SD relative to saved target_std; current position stays fixed.")
    training.add_argument("--history-noise-validation-seed", type=int, default=0,
                          help="Fixed auxiliary validation noise seed; clean NLL remains unperturbed.")
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
    training.add_argument("--downsample-factor", type=int,
                          help="Anti-alias filter and decimate every coordinate per repetition (new model default 1). "
                               "Keep --lag 1 on the resulting prepared grid; checkpoints retain this choice.")
    training.add_argument("--downsample-edge-trim", type=int,
                          help="Exclude this many decimated samples at each repetition edge (default 32; minimum 10).")
    training.add_argument(
        "--spatial-mean-highpass-hz", type=float,
        help="Use the matching high-pass spatial-mean trajectory stored in each POD file; "
             "evaluation inherits this choice from the checkpoint.",
    )
    training.add_argument("--hidden-layers", type=int, nargs="+", help="Default 64 64 for a new model.")
    training.add_argument("--activation", choices=("silu", "gelu", "relu", "tanh"),
                          help="Default silu for a new model.")
    training.add_argument(
        "--residual-architecture", choices=("mlp", "tcn"),
        help="Nonlinear drift network (default mlp). TCN replaces the residual, or "
             "the entire unstructured drift; requires native-lag velocity mode.")
    training.add_argument("--tcn-context-frames", type=int,
                          help="Number of dense state or core-innovation tokens (default 63).")
    training.add_argument("--residual-input", choices=("state", "core_innovation"),
                          help="TCN inputs: state (default), or past forcing increments after "
                               "subtracting the fixed oscillator; requires imported oscillator "
                               "TCN and constant diffusion. Fresh training required to switch.")
    training.add_argument("--tcn-channels", type=int, help="TCN channel width (default 64).")
    training.add_argument("--tcn-kernel-size", type=int, help="TCN kernel size (default 3).")
    training.add_argument("--tcn-dilations", type=int, nargs="+",
                          help="One causal convolution per dilation (default 1 2 4 8 16).")
    training.add_argument("--oscillator-artifact", type=Path,
                          help="Peak-fit model.npz for a fresh imported oscillator drift.")
    training.add_argument("--forcing-filter-artifact", type=Path,
                          help="Fixed modal biquad notch-filter artifact for fresh training; "
                               "enables inverse-filtered broadband forcing targets.")
    training.add_argument("--forcing-filter-burnin", type=int,
                          help="Observed frames discarded after inverse-filter initialization "
                               "(default 32768).")
    training.add_argument(
        "--drift-type", choices=("mlp", "damped_residual", "capillary_energy", "imported_oscillator_residual"),
        help="Drift structure: unconstrained neural network (mlp, default), learned damped residual, "
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
        "--no-structured-damping", action="store_true",
        help="For damped_residual, omit the explicit damping operator and train only "
             "the restoring operator plus residual MLP. A damped checkpoint may be "
             "warm-started; only its damping parameter is dropped.",
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
        "--multistep-target", choices=("velocity", "displacement", "joint"),
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
        velocity_memory_offsets = (
            None if args.velocity_memory_offsets is None
            else tuple(args.velocity_memory_offsets))
        velocity_oscillator_periods = (
            None if args.velocity_oscillator_periods is None
            else tuple(args.velocity_oscillator_periods))
        velocity_innovation_offsets = (
            None if args.velocity_innovation_offsets is None
            else tuple(args.velocity_innovation_offsets))
        if (args.lag_steps < 1 or (args.history is not None and args.history < 0)
                or (history_offsets is not None
                    and (any(offset < 1 for offset in history_offsets)
                         or tuple(sorted(set(history_offsets))) != history_offsets))
                or (velocity_memory_offsets is not None
                    and (any(offset < 1 for offset in velocity_memory_offsets)
                         or tuple(sorted(set(velocity_memory_offsets)))
                         != velocity_memory_offsets))
                or (args.velocity_memory_rank is not None
                    and args.velocity_memory_rank < 1)
                or (args.velocity_memory_norm_cap is not None
                    and (not math.isfinite(args.velocity_memory_norm_cap)
                         or args.velocity_memory_norm_cap <= 0))
                or (velocity_oscillator_periods is not None
                    and (not velocity_oscillator_periods
                         or any(not math.isfinite(period) or period <= 2
                                for period in velocity_oscillator_periods)
                         or tuple(sorted(set(velocity_oscillator_periods)))
                         != velocity_oscillator_periods))
                or (args.velocity_oscillator_burnin is not None
                    and args.velocity_oscillator_burnin < 1)
                or (args.velocity_oscillator_radius_init is not None
                    and (not math.isfinite(args.velocity_oscillator_radius_init)
                         or not 0 < args.velocity_oscillator_radius_init < .9999))
                or (args.velocity_oscillator_norm_cap is not None
                    and (not math.isfinite(args.velocity_oscillator_norm_cap)
                         or args.velocity_oscillator_norm_cap <= 0))
                or (velocity_innovation_offsets is not None
                    and (any(offset < 1 for offset in velocity_innovation_offsets)
                         or tuple(sorted(set(velocity_innovation_offsets)))
                         != velocity_innovation_offsets))
                or (args.velocity_innovation_rank is not None
                    and args.velocity_innovation_rank < 1)
                or (args.velocity_innovation_norm_cap is not None
                    and (not math.isfinite(args.velocity_innovation_norm_cap)
                         or not 0 < args.velocity_innovation_norm_cap < 1))
                or not 0 <= args.diffusion_trim_percent < 100
                or not 0 <= args.train_trim_percent < 100
                or not math.isfinite(args.multistep_weight)
                or args.multistep_weight < 0
                or args.multistep_particles < 1
                or (args.multistep_objective == "energy_score"
                    and (args.multistep_target not in {"velocity", "joint"}
                         or args.multistep_particles < 2))
                or (args.multistep_objective == "gmm_nll"
                    and args.multistep_target == "joint")
                or not math.isfinite(args.multistep_position_weight)
                or args.multistep_position_weight <= 0
                or not math.isfinite(args.multistep_velocity_weight)
                or args.multistep_velocity_weight <= 0
                or (args.multistep_target != "joint"
                    and (args.multistep_position_weight != 1
                         or args.multistep_velocity_weight != 1))
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
                         or (args.multistep_objective == "energy_score"
                             and any(count < 2
                                     for count in args.multistep_particle_counts))
                         or any(following > current for current, following in zip(
                             args.multistep_particle_counts,
                             args.multistep_particle_counts[1:]))))
                or (args.multistep_horizon_weights is not None
                    and (args.multistep_weight <= 0
                         or len(args.multistep_horizon_weights)
                         != len(args.multistep_horizons)
                         or any(not math.isfinite(weight) or weight < 0
                                for weight in args.multistep_horizon_weights)
                         or sum(args.multistep_horizon_weights) <= 0))
                or (args.multistep_window_fraction < 1 and args.multistep_weight <= 0)
                or not math.isfinite(args.stationary_mean_weight)
                or args.stationary_mean_weight < 0
                or not math.isfinite(args.stationary_variance_weight)
                or args.stationary_variance_weight < 0
                or (args.stationary_variance_pod_mode_count is not None
                    and args.stationary_variance_pod_mode_count < 0)
                or not math.isfinite(args.stationary_lagged_covariance_weight)
                or args.stationary_lagged_covariance_weight < 0
                or (args.stationary_lagged_covariance_mode_count is not None
                    and args.stationary_lagged_covariance_mode_count < 1)
                or not math.isfinite(args.stationary_lagged_covariance_epsilon)
                or args.stationary_lagged_covariance_epsilon <= 0
                or args.stationary_mean_horizon < 1
                or (args.stationary_mean_horizons is not None
                    and (not args.stationary_mean_horizons
                         or any(horizon < 1 for horizon in args.stationary_mean_horizons)
                         or tuple(sorted(set(args.stationary_mean_horizons)))
                         != tuple(args.stationary_mean_horizons)))
                or args.stationary_mean_trajectories < 2
                or not math.isfinite(args.stationary_mean_batch_fraction)
                or not 0 < args.stationary_mean_batch_fraction <= 1
                or not math.isfinite(args.restoring_orthogonality_weight)
                or args.restoring_orthogonality_weight < 0
                or not math.isfinite(args.damping_orthogonality_weight)
                or args.damping_orthogonality_weight < 0
                or not math.isfinite(args.restoring_orthogonality_epsilon)
                or args.restoring_orthogonality_epsilon <= 0
                or not math.isfinite(args.diagonal_linear_orthogonality_weight)
                or args.diagonal_linear_orthogonality_weight < 0
                or not math.isfinite(args.diagonal_linear_orthogonality_epsilon)
                or args.diagonal_linear_orthogonality_epsilon <= 0
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
        if args.train_mode == "memory" and args.checkpoint is None:
            parser.error("--train-mode memory requires --checkpoint.")
        if (args.train_mode == "memory"
                and sum(value is not None for value in (
                    velocity_memory_offsets, velocity_oscillator_periods,
                    velocity_innovation_offsets)) != 1):
            parser.error(
                "--train-mode memory requires exactly one velocity-memory option.")
        if sum(value is not None for value in (
                velocity_memory_offsets, velocity_oscillator_periods,
                velocity_innovation_offsets)) > 1:
            parser.error("Velocity-memory options are mutually exclusive.")
        if args.velocity_memory_rank is not None and velocity_memory_offsets is None:
            parser.error("--velocity-memory-rank requires --velocity-memory-offsets.")
        if (args.velocity_memory_norm_cap is not None
                and args.velocity_memory_rank is None):
            parser.error("--velocity-memory-norm-cap requires --velocity-memory-rank.")
        if ((args.velocity_oscillator_burnin is not None
             or args.velocity_oscillator_radius_init is not None
             or args.velocity_oscillator_norm_cap is not None)
                and velocity_oscillator_periods is None):
            parser.error("Oscillator settings require --velocity-oscillator-periods.")
        if args.velocity_innovation_rank is not None and velocity_innovation_offsets is None:
            parser.error("--velocity-innovation-rank requires --velocity-innovation-offsets.")
        if velocity_innovation_offsets is not None and args.velocity_innovation_rank is None:
            parser.error("--velocity-innovation-offsets requires --velocity-innovation-rank.")
        if (args.velocity_innovation_norm_cap is not None
                and velocity_innovation_offsets is None):
            parser.error(
                "--velocity-innovation-norm-cap requires --velocity-innovation-offsets.")
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
        if (args.checkpoint is None and args.no_structured_damping
                and args.drift_type != "damped_residual"):
            parser.error("--no-structured-damping requires --drift-type damped_residual.")
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
            if args.no_structured_damping and saved.drift_type != "damped_residual":
                parser.error(
                    "--no-structured-damping requires a damped_residual checkpoint.")
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
                    or any(getattr(args, name) is not None
                           and getattr(args, name) != getattr(saved, name)
                           for name in ("residual_architecture", "residual_input", "tcn_context_frames",
                                        "tcn_channels", "tcn_kernel_size"))
                    or (args.tcn_dilations is not None
                        and tuple(args.tcn_dilations) != saved.tcn_dilations)
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
                    or (args.train_mode != "memory"
                        and velocity_memory_offsets is not None
                        and velocity_memory_offsets != saved.velocity_memory_offsets)
                    or (args.train_mode != "memory"
                        and args.velocity_memory_rank is not None
                        and args.velocity_memory_rank != saved.velocity_memory_rank)
                    or (args.train_mode != "memory"
                        and velocity_oscillator_periods is not None
                        and velocity_oscillator_periods
                        != saved.velocity_oscillator_periods)
                    or (args.train_mode != "memory"
                        and velocity_innovation_offsets is not None
                        and velocity_innovation_offsets
                        != saved.velocity_innovation_offsets)
                    or (args.state_variable is not None and args.state_variable != saved.state_variable)
                    or (args.no_spatial_mean and data.include_spatial_mean)
                    or (args.spatial_mean_highpass_hz is not None
                        and args.spatial_mean_highpass_hz != data.spatial_mean_highpass_hz)
                    or (args.downsample_factor is not None
                        and args.downsample_factor != data.downsample_factor)
                    or (args.downsample_edge_trim is not None
                        and args.downsample_edge_trim != data.downsample_edge_trim)
                    or args.validation_reps is not None or args.test_reps is not None):
                parser.error("Warm-start source, split, and architecture come from the checkpoint; supplied options differ.")
            if (args.train_mode == "memory"
                    and (saved.velocity_memory_offsets is not None
                         or saved.velocity_oscillator_periods is not None
                         or saved.velocity_innovation_offsets is not None)):
                parser.error("--train-mode memory must start from a memory-free joint checkpoint.")
            config = replace(
                saved, lag_steps=args.lag_steps,
                structured_damping=(False if args.no_structured_damping
                                    else saved.structured_damping),
                velocity_memory_offsets=(velocity_memory_offsets
                                         if args.train_mode == "memory"
                                         else saved.velocity_memory_offsets),
                velocity_memory_rank=(args.velocity_memory_rank
                                      if args.train_mode == "memory"
                                      else saved.velocity_memory_rank),
                velocity_memory_norm_cap=(args.velocity_memory_norm_cap
                                          if args.velocity_memory_norm_cap is not None
                                          else saved.velocity_memory_norm_cap),
                velocity_oscillator_periods=(
                    velocity_oscillator_periods if args.train_mode == "memory"
                    else saved.velocity_oscillator_periods),
                velocity_oscillator_burnin=(
                    args.velocity_oscillator_burnin
                    if args.velocity_oscillator_burnin is not None else
                    saved.velocity_oscillator_burnin),
                velocity_oscillator_radius_init=(
                    args.velocity_oscillator_radius_init
                    if args.velocity_oscillator_radius_init is not None else
                    saved.velocity_oscillator_radius_init),
                velocity_oscillator_norm_cap=(
                    args.velocity_oscillator_norm_cap
                    if args.velocity_oscillator_norm_cap is not None else
                    saved.velocity_oscillator_norm_cap),
                velocity_innovation_offsets=(
                    velocity_innovation_offsets if args.train_mode == "memory"
                    else saved.velocity_innovation_offsets),
                velocity_innovation_rank=(
                    args.velocity_innovation_rank if args.train_mode == "memory"
                    else saved.velocity_innovation_rank),
                velocity_innovation_norm_cap=(
                    args.velocity_innovation_norm_cap
                    if args.velocity_innovation_norm_cap is not None else
                    saved.velocity_innovation_norm_cap))
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
            if (velocity_memory_offsets is not None
                    and (state_variable != "velocity" or args.lag_steps != 1)):
                parser.error(
                    "--velocity-memory-offsets requires --state-variable velocity --lag 1.")
            if (velocity_oscillator_periods is not None
                    and (state_variable != "velocity" or args.lag_steps != 1)):
                parser.error(
                    "--velocity-oscillator-periods requires --state-variable velocity --lag 1.")
            if (velocity_innovation_offsets is not None
                    and (state_variable != "velocity" or args.lag_steps != 1)):
                parser.error(
                    "--velocity-innovation-offsets requires --state-variable velocity --lag 1.")
            if drift_type in {"damped_residual", "capillary_energy", "imported_oscillator_residual"} and state_variable != "velocity":
                parser.error(f"--drift-type {drift_type} requires --state-variable velocity.")
            if ((args.temporal_component_manifest is None)
                    != (args.temporal_component is None)):
                parser.error("Temporal component manifest and component must be supplied together.")
            options = dict(data_root=args.data_root, shared_basis=args.shared_basis,
                           validation_reps=args.validation_reps, test_reps=args.test_reps,
                           include_spatial_mean=not args.no_spatial_mean,
                           downsample_factor=args.downsample_factor,
                           downsample_edge_trim=args.downsample_edge_trim,
                           spatial_mean_highpass_hz=args.spatial_mean_highpass_hz,
                           temporal_component_manifest=args.temporal_component_manifest,
                           temporal_component=args.temporal_component)
            options = {key: value for key, value in options.items() if value is not None}
            if args.temporal_component_manifest is not None:
                from modelling.data_preparation.frequency_components import FrequencyComponentData
                data = FrequencyComponentData(experiment=args.experiment, rank=args.rank, **options)
            else:
                data = SharedPODTrainingData(args.experiment, args.rank, **options)
            config = SDEConfig(data.n_coordinates, dt=data.native_dt if args.dt is None else args.dt,
                               hidden_layers=tuple(args.hidden_layers or (64, 64)),
                               activation=args.activation or "silu",
                               residual_architecture=args.residual_architecture or "mlp",
                               residual_input=args.residual_input or "state",
                               forcing_filter=("modal_biquad" if args.forcing_filter_artifact
                                               is not None else "none"),
                               forcing_filter_burnin=(args.forcing_filter_burnin
                                                      if args.forcing_filter_burnin is not None
                                                      else 32768),
                               tcn_context_frames=(args.tcn_context_frames
                                                   if args.tcn_context_frames is not None else 63),
                               tcn_channels=(args.tcn_channels
                                             if args.tcn_channels is not None else 64),
                               tcn_kernel_size=(args.tcn_kernel_size
                                                if args.tcn_kernel_size is not None else 3),
                               tcn_dilations=tuple(args.tcn_dilations or (1, 2, 4, 8, 16)),
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
                               structured_damping=not args.no_structured_damping,
                               diffusion_init=args.diffusion_init if args.diffusion_init is not None else .1,
                               diffusion_type=args.diffusion_type or "diagonal",
                               diffusion_log_range=(args.diffusion_log_range
                                                    if args.diffusion_log_range is not None else 2.),
                               lag_steps=args.lag_steps, history_steps=history_steps,
                               history_offsets=history_offsets,
                               velocity_memory_offsets=velocity_memory_offsets,
                               velocity_memory_rank=args.velocity_memory_rank,
                               velocity_memory_norm_cap=(
                                   args.velocity_memory_norm_cap
                                   if args.velocity_memory_norm_cap is not None else .25),
                               velocity_oscillator_periods=velocity_oscillator_periods,
                               velocity_oscillator_burnin=(
                                   args.velocity_oscillator_burnin
                                   if args.velocity_oscillator_burnin is not None else 256),
                               velocity_oscillator_radius_init=(
                                   args.velocity_oscillator_radius_init
                                   if args.velocity_oscillator_radius_init is not None else .98),
                               velocity_oscillator_norm_cap=(
                                   args.velocity_oscillator_norm_cap
                                   if args.velocity_oscillator_norm_cap is not None else .25),
                               velocity_innovation_offsets=velocity_innovation_offsets,
                               velocity_innovation_rank=args.velocity_innovation_rank,
                               velocity_innovation_norm_cap=(
                                   args.velocity_innovation_norm_cap
                                   if args.velocity_innovation_norm_cap is not None else .95),
                               state_variable=state_variable)
        if args.multistep_weight > 0 and config.state_variable != "velocity":
            parser.error("--multistep-weight requires --state-variable velocity.")
        if ((args.stationary_mean_weight > 0 or args.stationary_variance_weight > 0
             or args.stationary_lagged_covariance_weight > 0)
                and config.state_variable != "velocity"):
            parser.error("Stationary moment weights require --state-variable velocity.")
        if ((args.stationary_mean_weight > 0
             or args.stationary_lagged_covariance_weight > 0)
                and args.train_mode == "diffusion"):
            parser.error(
                "Stationary mean/lagged-covariance weights require trainable drift.")
        if (args.stationary_lagged_covariance_mode_count is not None
                and args.stationary_lagged_covariance_mode_count > data.rank):
            parser.error(
                "--stationary-lagged-covariance-mode-count cannot exceed the POD rank.")
        if (args.stationary_variance_pod_mode_count is not None
                and args.stationary_variance_pod_mode_count > data.rank):
            parser.error(
                "--stationary-variance-pod-mode-count cannot exceed the POD rank.")
        if (args.stationary_variance_pod_mode_count == 0
                and not data.include_spatial_mean):
            parser.error(
                "--stationary-variance-pod-mode-count 0 requires the spatial mean.")
        if args.restoring_orthogonality_weight > 0:
            if (config.drift_type != "damped_residual"
                    or config.damped_operator != "diagonal"):
                parser.error(
                    "--restoring-orthogonality-weight requires a diagonal "
                    "--drift-type damped_residual.")
            if args.train_mode in {"diffusion", "memory"}:
                parser.error(
                    "--restoring-orthogonality-weight requires directly trainable drift.")
        if args.damping_orthogonality_weight > 0:
            if (config.drift_type != "damped_residual"
                    or config.damped_operator != "diagonal"
                    or not config.structured_damping):
                parser.error(
                    "--damping-orthogonality-weight requires a structured diagonal "
                    "--drift-type damped_residual.")
            if args.train_mode in {"diffusion", "memory"}:
                parser.error(
                    "--damping-orthogonality-weight requires directly trainable drift.")
        if args.diagonal_linear_orthogonality_weight > 0:
            if (config.drift_type != "damped_residual"
                    or config.damped_operator != "diagonal"
                    or not config.structured_damping):
                parser.error(
                    "--diagonal-linear-orthogonality-weight requires a structured "
                    "diagonal --drift-type damped_residual.")
            if args.train_mode in {"diffusion", "memory"}:
                parser.error(
                    "--diagonal-linear-orthogonality-weight requires directly "
                    "trainable drift.")
            if (args.restoring_orthogonality_weight > 0
                    or args.damping_orthogonality_weight > 0
                    or args.restoring_orthogonality_adjust_velocity):
                parser.error(
                    "--diagonal-linear-orthogonality-weight cannot be combined with "
                    "the legacy orthogonality options.")
        if (args.restoring_orthogonality_adjust_velocity
                and args.restoring_orthogonality_weight <= 0
                and args.damping_orthogonality_weight <= 0):
            parser.error(
                "--restoring-orthogonality-adjust-velocity requires a positive "
                "residual orthogonality weight.")
        try:
            validate_history_noise(config, args.history_noise_weight,
                                   args.history_noise_std,
                                   args.history_noise_validation_seed, args.train_mode)
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
        path = train(data, args.output, config, oscillator_artifact=args.oscillator_artifact,
                     forcing_filter_artifact=args.forcing_filter_artifact,
                     learning_rate=args.learning_rate,
                     batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
                     device=resolve_device(args.device), train_mode=args.train_mode,
                     diffusion_trim_percent=args.diffusion_trim_percent,
                     train_trim_percent=args.train_trim_percent,
                     covariance_space=args.diffusion_covariance_space,
                     multistep_weight=args.multistep_weight,
                     multistep_objective=args.multistep_objective,
                     multistep_target=args.multistep_target,
                     multistep_horizons=tuple(args.multistep_horizons),
                     multistep_mode_counts=(
                         None if args.multistep_mode_counts is None else
                         tuple(args.multistep_mode_counts)),
                     multistep_particles=args.multistep_particles,
                     multistep_particle_counts=(
                         None if args.multistep_particle_counts is None else
                         tuple(args.multistep_particle_counts)),
                     multistep_horizon_weights=(
                         None if args.multistep_horizon_weights is None else
                         tuple(args.multistep_horizon_weights)),
                     multistep_position_weight=args.multistep_position_weight,
                     multistep_velocity_weight=args.multistep_velocity_weight,
                     multistep_window_fraction=args.multistep_window_fraction,
                     multistep_window_sampling=args.multistep_window_sampling,
                     multistep_gradient_groups=tuple(args.multistep_gradient_groups),
                     multistep_validation_seed=args.multistep_validation_seed,
                     tcn_rollout_cache=args.tcn_rollout_cache,
                     tcn_rollout_compile=args.tcn_rollout_compile,
                     indexed_training_batches=args.indexed_training_batches,
                     history_noise_weight=args.history_noise_weight,
                     history_noise_std=args.history_noise_std,
                     history_noise_validation_seed=args.history_noise_validation_seed,
                     stationary_mean_weight=args.stationary_mean_weight,
                     stationary_variance_weight=args.stationary_variance_weight,
                     stationary_variance_pod_mode_count=(
                         args.stationary_variance_pod_mode_count),
                     stationary_lagged_covariance_weight=(
                         args.stationary_lagged_covariance_weight),
                     stationary_lagged_covariance_mode_count=(
                         args.stationary_lagged_covariance_mode_count),
                     stationary_lagged_covariance_epsilon=(
                         args.stationary_lagged_covariance_epsilon),
                     stationary_lagged_covariance_coordinates=(None if args.stationary_lagged_covariance_coordinates is None
                         else tuple(args.stationary_lagged_covariance_coordinates)),
                     stationary_lagged_covariance_normalization=args.stationary_lagged_covariance_normalization,
                     stationary_lagged_covariance_horizons=(None if args.stationary_lagged_covariance_horizons is None
                         else tuple(args.stationary_lagged_covariance_horizons)),
                     stationary_mean_horizon=args.stationary_mean_horizon,
                     stationary_mean_horizons=(
                         None if args.stationary_mean_horizons is None else
                         tuple(args.stationary_mean_horizons)),
                     stationary_mean_trajectories=args.stationary_mean_trajectories,
                     stationary_mean_batch_fraction=args.stationary_mean_batch_fraction,
                     stationary_mean_validation_seed=(
                         args.stationary_mean_validation_seed),
                     restoring_orthogonality_weight=(
                         args.restoring_orthogonality_weight),
                     damping_orthogonality_weight=(
                         args.damping_orthogonality_weight),
                     restoring_orthogonality_epsilon=(
                         args.restoring_orthogonality_epsilon),
                     restoring_orthogonality_adjust_velocity=(
                         args.restoring_orthogonality_adjust_velocity),
                     diagonal_linear_orthogonality_weight=(
                         args.diagonal_linear_orthogonality_weight),
                     diagonal_linear_orthogonality_epsilon=(
                         args.diagonal_linear_orthogonality_epsilon),
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
