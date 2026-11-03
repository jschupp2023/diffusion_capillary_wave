"""Replaceable transition losses; no objective is embedded in NeuralSDE."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .model import NeuralSDE


def _gaussian_log_prob(target: Tensor, mean: Tensor, scale_tril: Tensor) -> Tensor:
    """Joint Gaussian log density with broadcast batch/particle dimensions."""
    if target.shape != mean.shape:
        raise ValueError("target and mean must have matching shapes")
    dimension = target.shape[-1]
    expected = (*target.shape[:-1], dimension, dimension)
    if scale_tril.ndim == 2:
        scale_tril = scale_tril.expand(expected)
    elif scale_tril.shape != expected:
        raise ValueError(f"scale_tril must have shape {expected}")
    residual = target - mean
    whitened = torch.linalg.solve_triangular(
        scale_tril, residual.unsqueeze(-1), upper=False).squeeze(-1)
    log_determinant = torch.log(
        torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(-1)
    return (-.5 * whitened.square().sum(dim=-1) - log_determinant
            - .5 * dimension * math.log(2 * math.pi))


def _gaussian_marginal(mean: Tensor, scale_tril: Tensor,
                       indices: Tensor) -> tuple[Tensor, Tensor]:
    """Select a Gaussian marginal from a mean and full Cholesky factor."""
    selected_mean = mean.index_select(-1, indices)
    selected_rows = scale_tril.index_select(-2, indices)
    covariance = selected_rows @ selected_rows.transpose(-1, -2)
    return selected_mean, torch.linalg.cholesky(covariance)


def _velocity_distribution(model: NeuralSDE, state: Tensor, history_states: Tensor,
                           velocity: Tensor, *, zero_drift: bool = False,
                           oscillator_state: Tensor | None = None,
                           innovation_state: Tensor | None = None
                           ) -> tuple[Tensor, Tensor]:
    """Euler--Maruyama conditional mean/factor for native-frame velocity."""
    return model.velocity_step_distribution(
        state, history_states, velocity, zero_drift=zero_drift,
        oscillator_state=oscillator_state, innovation_state=innovation_state)


class EulerMaruyamaNLL(nn.Module):
    """Mean physical-coordinate Gaussian NLL for the configured SDE variable."""

    def forward(self, model: NeuralSDE, current: Tensor, following: Tensor,
                *, history_states: Tensor | None = None,
                following_history_states: Tensor | None = None,
                forcing_history: Tensor | None = None,
                raw_forcing_target: Tensor | None = None,
                zero_drift: bool = False) -> Tensor:
        return self.per_example(
            model, current, following, history_states=history_states,
            following_history_states=following_history_states,
            forcing_history=forcing_history,
            raw_forcing_target=raw_forcing_target,
            zero_drift=zero_drift).mean()

    def per_example(self, model: NeuralSDE, current: Tensor, following: Tensor,
                    *, history_states: Tensor | None = None,
                    following_history_states: Tensor | None = None,
                    forcing_history: Tensor | None = None,
                    raw_forcing_target: Tensor | None = None,
                    zero_drift: bool = False) -> Tensor:
        """Return the existing joint-coordinate NLL before batch averaging."""
        if current.ndim != 2 or current.shape != following.shape:
            raise ValueError("current and following must be matching [batch, reduced_dim] tensors")
        if model.config.state_variable == "velocity":
            if model.has_forcing_filter:
                if forcing_history is None or raw_forcing_target is None:
                    raise ValueError(
                        "filtered forcing loss requires forcing_history and raw_forcing_target")
                mean, factor = model.raw_forcing_distribution(forcing_history)
                if zero_drift:
                    mean = torch.zeros_like(mean)
                # The likelihood is evaluated in the exactly invertible raw
                # variable. Add the constant Jacobian so reported NLL remains
                # the physical filtered-forcing density.
                return (-_gaussian_log_prob(raw_forcing_target, mean, factor)
                        + model.forcing_filter_logabsdet())
            current_velocity = model._frame_velocity(current, history_states)
            if following_history_states is None:
                if model.config.lag_steps != 1:
                    raise ValueError("following_history_states are required for lagged velocity loss")
                following_history_states = current.unsqueeze(-2)
            next_velocity = model._frame_velocity(following, following_history_states)
            mean, factor = _velocity_distribution(
                model, current, history_states, current_velocity, zero_drift=zero_drift)
            return -_gaussian_log_prob(next_velocity, mean, factor)
        drift_input = model._drift_input(current, history_states)
        # Drift and diffusion are parameterized per native data step.
        next_increment = following - current
        if model.config.state_variable == "increment":
            current_increment = current - history_states[..., 0, :]
            residual = (next_increment - current_increment) / model.target_std
        else:
            residual = next_increment / model.state_std
        if not zero_drift:
            residual = residual - model.config.lag_steps * model.drift_model(drift_input)
        factor = math.sqrt(model.config.lag_steps) * model.normalized_diffusion_matrix(
            normalized_input=drift_input)
        if factor.ndim == 2:
            factor = factor.expand(len(residual), -1, -1)
        whitened = torch.linalg.solve_triangular(
            factor, residual.unsqueeze(-1), upper=False).squeeze(-1)
        log_determinant = torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
        standardized_nll = (.5 * whitened.square().sum(dim=-1) + log_determinant
                            + .5 * model.config.reduced_dim * math.log(2 * math.pi))
        return standardized_nll + torch.log(model._sde_scale()).sum()


class MonteCarloMultistepNLL(nn.Module):
    """Velocity or displacement mixture likelihood over latent velocity paths.

    Position and native-frame velocity are both simulated. Score either the
    endpoint velocity or displacement from the observed starting position.
    Paths are shared across requested horizons;
    a distribution scored at a shorter horizon is sampled only when that state
    is needed as an intermediate state for a longer horizon.
    """

    objective_name = "gmm_nll"
    metric_name = "nll"

    def __init__(self, horizons: tuple[int, ...] = (2, 4), particles: int = 100,
                 *, target: str = "velocity",
                 mode_counts: tuple[int, ...] | None = None,
                 pod_coordinate_offset: int = 0,
                 particle_counts: tuple[int, ...] | None = None,
                 horizon_weights: tuple[float, ...] | None = None):
        super().__init__()
        horizons = tuple(horizons)
        if (not horizons or any(not isinstance(horizon, int) or isinstance(horizon, bool)
                                or horizon < 1 for horizon in horizons)
                or tuple(sorted(set(horizons))) != horizons):
            raise ValueError("horizons must be strictly increasing positive integers")
        if not isinstance(particles, int) or isinstance(particles, bool) or particles < 1:
            raise ValueError("particles must be a positive integer")
        if particle_counts is None:
            particle_counts = (particles,) * len(horizons)
        else:
            particle_counts = tuple(particle_counts)
            if (len(particle_counts) != len(horizons)
                    or any(not isinstance(count, int) or isinstance(count, bool) or count < 1
                           for count in particle_counts)
                    or any(following > current for current, following in
                           zip(particle_counts, particle_counts[1:]))):
                raise ValueError(
                    "particle_counts must be one nonincreasing positive count per horizon")
        if target not in {"velocity", "displacement"}:
            raise ValueError("multistep target must be velocity or displacement")
        if mode_counts is not None:
            mode_counts = tuple(mode_counts)
            if (target != "velocity" or len(mode_counts) != len(horizons)
                    or any(not isinstance(count, int) or isinstance(count, bool) or count < 1
                           for count in mode_counts)):
                raise ValueError(
                    "mode_counts require one positive POD-mode count per velocity horizon")
        if (not isinstance(pod_coordinate_offset, int)
                or isinstance(pod_coordinate_offset, bool)
                or pod_coordinate_offset < 0):
            raise ValueError("pod_coordinate_offset must be a nonnegative integer")
        self.horizons = horizons
        self.particles = particles
        self.particle_counts = particle_counts
        self.target = target
        self.mode_counts = mode_counts
        self.pod_coordinate_offset = pod_coordinate_offset
        if horizon_weights is None:
            horizon_weights = (1 / len(horizons),) * len(horizons)
        else:
            horizon_weights = tuple(horizon_weights)
            if (len(horizon_weights) != len(horizons)
                    or any(not math.isfinite(weight) or weight < 0
                           for weight in horizon_weights)
                    or sum(horizon_weights) <= 0):
                raise ValueError(
                    "horizon_weights must contain one nonnegative finite value per horizon "
                    "and have a positive sum")
            total = sum(horizon_weights)
            horizon_weights = tuple(weight / total for weight in horizon_weights)
        self.horizon_weights = horizon_weights

    def aggregate(self, losses: dict[int, Tensor]) -> Tensor:
        """Combine per-horizon losses using normalized configured weights."""
        return sum(weight * losses[horizon]
                   for horizon, weight in zip(
                       self.horizons, self.horizon_weights, strict=True))

    def _selected_values(self, horizon: int, values: Tensor) -> Tensor:
        """Select the configured POD coordinates from a final-coordinate tensor."""
        if self.mode_counts is None:
            return values
        count = self.mode_counts[self.horizons.index(horizon)]
        stop = self.pod_coordinate_offset + count
        if stop > values.shape[-1]:
            raise ValueError(
                f"horizon {horizon} requests {count} POD modes, but only "
                f"{values.shape[-1] - self.pod_coordinate_offset} are available")
        return values[..., self.pod_coordinate_offset:stop]

    def _scored_distribution(self, horizon: int, target: Tensor, mean: Tensor,
                             factor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the requested POD marginal, or the legacy full distribution."""
        if self.mode_counts is None:
            return target, mean, factor
        count = self.mode_counts[self.horizons.index(horizon)]
        stop = self.pod_coordinate_offset + count
        if stop > target.shape[-1]:
            raise ValueError(
                f"horizon {horizon} requests {count} POD modes, but only "
                f"{target.shape[-1] - self.pod_coordinate_offset} are available")
        indices = torch.arange(
            self.pod_coordinate_offset, stop, device=target.device)
        selected_mean, selected_factor = _gaussian_marginal(mean, factor, indices)
        return target.index_select(-1, indices), selected_mean, selected_factor

    def forward(self, model: NeuralSDE, current: Tensor, targets: Tensor,
                *, history_states: Tensor, generator: torch.Generator | None = None,
                zero_drift: bool = False) -> Tensor:
        losses = self.per_horizon(
            model, current, targets, history_states=history_states,
            generator=generator, zero_drift=zero_drift)
        return self.aggregate(losses)

    def per_horizon(self, model: NeuralSDE, current: Tensor, targets: Tensor,
                    *, history_states: Tensor,
                    generator: torch.Generator | None = None,
                    zero_drift: bool = False,
                    particle_diagnostics: dict[int, dict[str, Tensor]] | None = None
                    ) -> dict[int, Tensor]:
        if model.config.state_variable != "velocity":
            raise ValueError("multistep likelihood requires state_variable velocity")
        dimension = model.config.reduced_dim
        if current.ndim != 2 or current.shape[-1] != dimension:
            raise ValueError("current must have shape [batch, reduced_dim]")
        expected_history = (len(current), model.config.history_steps, dimension)
        if history_states.shape != expected_history:
            raise ValueError(f"history_states must have shape {expected_history}")
        expected_targets = (len(current), len(self.horizons), dimension)
        if targets.shape != expected_targets:
            raise ValueError(f"targets must have shape {expected_targets}")

        target_by_horizon = dict(zip(self.horizons, targets.unbind(dim=1)))
        losses: dict[int, Tensor] = {}
        initial_velocity = model._frame_velocity(current, history_states)
        initial_oscillator = (
            model.velocity_oscillator_state(current, history_states)
            if model.config.velocity_oscillator_periods is not None else None)
        initial_innovation = (
            model.velocity_innovation_state(current, history_states)
            if model.config.velocity_innovation_offsets is not None else None)
        runtime_history = history_states
        if initial_oscillator is not None:
            runtime_history = history_states[..., :max(
                model.config.effective_history_offsets), :]
        step_size = model.config.lag_steps

        # H=1 has no latent path. Evaluate it without a redundant identical
        # particle mixture. Displacement is h*v_next, including its Jacobian.
        if 1 in target_by_horizon:
            mean, factor = _velocity_distribution(
                model, current, runtime_history, initial_velocity,
                zero_drift=zero_drift, oscillator_state=initial_oscillator,
                innovation_state=initial_innovation)
            if self.target == "displacement":
                mean, factor = step_size * mean, step_size * factor
            target, mean, factor = self._scored_distribution(
                1, target_by_horizon[1], mean, factor)
            log_likelihood = _gaussian_log_prob(target, mean, factor)
            losses[1] = -log_likelihood.mean()
            if particle_diagnostics is not None:
                ones = torch.ones_like(log_likelihood)
                zeros = torch.zeros_like(log_likelihood)
                particle_diagnostics[1] = dict(
                    log_likelihood=log_likelihood,
                    effective_sample_size=ones,
                    effective_sample_fraction=ones,
                    coefficient_of_variation_squared=zeros,
                    relative_standard_error=zeros,
                    largest_component_fraction=ones,
                )

        maximum = self.horizons[-1]
        if maximum == 1:
            return losses
        first_latent_index = next(
            index for index, horizon in enumerate(self.horizons) if horizon > 1)
        active_particles = self.particle_counts[first_latent_index]
        state = current[:, None, :].expand(-1, active_particles, -1)
        velocity = initial_velocity[:, None, :].expand(-1, active_particles, -1)
        history = runtime_history[:, None, :, :].expand(
            -1, active_particles, -1, -1)
        oscillator_state = (None if initial_oscillator is None else
                            initial_oscillator[:, None, :].expand(
                                -1, active_particles, -1))
        innovation_state = (None if initial_innovation is None else
                            initial_innovation[:, None, :, :].expand(
                                -1, active_particles, -1, -1))
        # Accumulate relative displacement directly, avoiding subtraction of
        # potentially large absolute positions in the mixture means.
        displacement = torch.zeros_like(state) if self.target == "displacement" else None
        tcn_cache = model.new_tcn_cache() if getattr(model, "tcn_rollout_cache", False) else None
        initial_pair = None
        if tcn_cache is not None and not zero_drift:
            # All particles start from identical measured histories. Warm the
            # TCN once per condition; expand differentiable buffers only after
            # that work, preserving gradient summation across particles.
            first_mean, first_factor = model.velocity_base_step_distribution(
                current, runtime_history, initial_velocity, tcn_cache=tcn_cache)
            initial_pair = (first_mean[:, None].expand(-1, active_particles, -1),
                            first_factor if first_factor.ndim == 2 else
                            first_factor[:, None].expand(-1, active_particles, -1, -1))
            tcn_cache["buffers"] = tuple(
                buffer[:, None].expand(-1, active_particles, -1, -1)
                for buffer in tcn_cache["buffers"])

        for step in range(1, maximum + 1):
            if step == 1 and initial_pair is not None:
                base_mean, factor = initial_pair
            else:
                base_mean, factor = model.velocity_base_step_distribution(
                    state, history, velocity, zero_drift=zero_drift,
                    oscillator_state=oscillator_state,
                    **({"tcn_cache": tcn_cache} if tcn_cache is not None else {}))
            mean = base_mean
            if innovation_state is not None and not zero_drift:
                mean = mean + model.target_std * model.velocity_innovation_correction(
                    innovation_state)
            if step in target_by_horizon and step != 1:
                target = target_by_horizon[step][:, None, :].expand(
                    -1, active_particles, -1)
                score_mean, score_factor = mean, factor
                if self.target == "displacement":
                    score_mean = displacement + step_size * mean
                    score_factor = step_size * factor
                target, score_mean, score_factor = self._scored_distribution(
                    step, target, score_mean, score_factor)
                log_prob = _gaussian_log_prob(target, score_mean, score_factor)
                log_weight_sum = torch.logsumexp(log_prob, dim=1)
                log_likelihood = log_weight_sum - math.log(active_particles)
                mixture_nll = -log_likelihood
                losses[step] = mixture_nll.mean()
                if particle_diagnostics is not None:
                    log_alpha = log_prob - log_weight_sum[:, None]
                    effective_sample_size = torch.clamp(
                        torch.exp(-torch.logsumexp(2 * log_alpha, dim=1)),
                        min=1, max=active_particles)
                    cv_squared = active_particles / effective_sample_size - 1
                    particle_diagnostics[step] = dict(
                        log_likelihood=log_likelihood,
                        effective_sample_size=effective_sample_size,
                        effective_sample_fraction=(
                            effective_sample_size / active_particles),
                        coefficient_of_variation_squared=cv_squared,
                        relative_standard_error=torch.sqrt(
                            torch.clamp(cv_squared / active_particles, min=0)),
                        largest_component_fraction=torch.exp(
                            log_alpha.max(dim=1).values),
                    )
            if step == maximum:
                break
            if step in target_by_horizon:
                horizon_index = self.horizons.index(step)
                next_particles = self.particle_counts[horizon_index + 1]
                if next_particles < active_particles:
                    particle_slice = (slice(None), slice(0, next_particles))
                    state = state[particle_slice]
                    velocity = velocity[particle_slice]
                    history = history[particle_slice]
                    if tcn_cache is not None and tcn_cache["buffers"] is not None:
                        tcn_cache["buffers"] = tuple(
                            buffer[particle_slice] for buffer in tcn_cache["buffers"])
                    mean = mean[particle_slice]
                    if factor.ndim > 2:
                        factor = factor[particle_slice]
                    if displacement is not None:
                        displacement = displacement[particle_slice]
                    if oscillator_state is not None:
                        oscillator_state = oscillator_state[particle_slice]
                    if innovation_state is not None:
                        innovation_state = innovation_state[particle_slice]
                    base_mean = base_mean[particle_slice]
                    active_particles = next_particles
            noise = torch.randn(
                velocity.shape, device=velocity.device, dtype=velocity.dtype,
                generator=generator)
            stochastic = (torch.matmul(noise, factor.T) if factor.ndim == 2 else
                          torch.matmul(factor, noise.unsqueeze(-1)).squeeze(-1))
            following_velocity = mean + stochastic
            following_state = state + step_size * following_velocity
            if oscillator_state is not None:
                oscillator_state = model.advance_velocity_oscillator(
                    oscillator_state, following_velocity)
            if innovation_state is not None:
                innovation_state = model.advance_velocity_innovation(
                    innovation_state, following_velocity, base_mean)
            if displacement is not None:
                displacement = displacement + step_size * following_velocity
            history = (state.unsqueeze(-2) if model.config.history_steps == 1 else
                       torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2))
            state, velocity = following_state, following_velocity
        return {horizon: losses[horizon] for horizon in self.horizons}


class MonteCarloMultistepVelocityNLL(MonteCarloMultistepNLL):
    """Backward-compatible velocity-only API, including legacy keyword names."""

    def __init__(self, horizons: tuple[int, ...] = (2, 4), particles: int = 100,
                 *, mode_counts: tuple[int, ...] | None = None,
                 pod_coordinate_offset: int = 0,
                 particle_counts: tuple[int, ...] | None = None):
        super().__init__(horizons, particles, target="velocity",
                         mode_counts=mode_counts,
                         pod_coordinate_offset=pod_coordinate_offset,
                         particle_counts=particle_counts)

    def forward(self, model, current, target_velocities, **kwargs):
        return super().forward(model, current, target_velocities, **kwargs)

    def per_horizon(self, model, current, target_velocities, **kwargs):
        return super().per_horizon(model, current, target_velocities, **kwargs)


class MultistepVelocityEnergyScore(MonteCarloMultistepNLL):
    """Conditional endpoint energy score from complete stochastic rollouts.

    Each measured starting window produces M independent autoregressive paths.
    At every requested horizon the score uses either endpoint velocity or the
    joint vector of displacement from the measured start and endpoint velocity.
    Distances use the checkpoint's position and native-velocity scales. The pair
    term is the unbiased U-statistic estimate: diagonal/self pairs are excluded.
    """

    objective_name = "energy_score"
    metric_name = "energy_score"

    def __init__(self, horizons: tuple[int, ...] = (2, 4), particles: int = 8,
                 *, mode_counts: tuple[int, ...] | None = None,
                 pod_coordinate_offset: int = 0,
                 particle_counts: tuple[int, ...] | None = None,
                 target: str = "velocity", position_weight: float = 1.,
                 velocity_weight: float = 1.,
                 horizon_weights: tuple[float, ...] | None = None):
        if target not in {"velocity", "joint"}:
            raise ValueError("energy-score target must be velocity or joint")
        if target == "joint" and mode_counts is not None:
            raise ValueError("joint energy score does not support mode_counts")
        if (not math.isfinite(position_weight) or position_weight <= 0
                or not math.isfinite(velocity_weight) or velocity_weight <= 0):
            raise ValueError("joint energy-score block weights must be positive and finite")
        super().__init__(
            horizons, particles, target="velocity", mode_counts=mode_counts,
            pod_coordinate_offset=pod_coordinate_offset,
            particle_counts=particle_counts, horizon_weights=horizon_weights)
        self.target = target
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        if any(count < 2 for count in self.particle_counts):
            raise ValueError(
                "energy-score particle counts must be at least two at every horizon")

    def _score_samples(self, model: NeuralSDE, horizon: int, samples: Tensor,
                       target: Tensor) -> Tensor:
        """Mean unbiased energy score over independent starting conditions."""
        if self.target == "velocity":
            samples = self._selected_values(horizon, samples)
            target = self._selected_values(horizon, target)
            scale = self._selected_values(horizon, model.target_std)
            standardized_errors = (samples - target[:, None, :]) / scale
            standardized_samples = samples / scale
        else:
            dimension = model.config.reduced_dim
            displacement, velocity = samples.split(dimension, dim=-1)
            target_displacement, target_velocity = target.split(dimension, dim=-1)
            standardized_errors = torch.cat((
                math.sqrt(self.position_weight)
                * (displacement - target_displacement[:, None, :]) / model.state_std,
                math.sqrt(self.velocity_weight)
                * (velocity - target_velocity[:, None, :]) / model.target_std,
            ), dim=-1)
            standardized_samples = torch.cat((
                math.sqrt(self.position_weight) * displacement / model.state_std,
                math.sqrt(self.velocity_weight) * velocity / model.target_std,
            ), dim=-1)
        observation_distance = torch.linalg.vector_norm(
            standardized_errors, dim=-1).mean(dim=1)

        particle_count = samples.shape[1]
        first, second = torch.triu_indices(
            particle_count, particle_count, offset=1, device=samples.device)
        pair_differences = (
            standardized_samples[:, first, :] - standardized_samples[:, second, :])
        # sum_{m<n} d(X_m, X_n) / (M(M-1)) equals one half of the
        # distinct-pair expectation in the population energy score.
        pair_distance = torch.linalg.vector_norm(
            pair_differences, dim=-1).sum(dim=1) / (
                particle_count * (particle_count - 1))
        return (observation_distance - pair_distance).mean()

    def per_horizon(self, model: NeuralSDE, current: Tensor, targets: Tensor,
                    *, history_states: Tensor,
                    generator: torch.Generator | None = None,
                    zero_drift: bool = False, **_ignored) -> dict[int, Tensor]:
        if model.config.state_variable != "velocity":
            raise ValueError("multistep energy score requires state_variable velocity")
        dimension = model.config.reduced_dim
        if current.ndim != 2 or current.shape[-1] != dimension:
            raise ValueError("current must have shape [batch, reduced_dim]")
        expected_history = (len(current), model.config.history_steps, dimension)
        if history_states.shape != expected_history:
            raise ValueError(f"history_states must have shape {expected_history}")
        target_dimension = dimension if self.target == "velocity" else 2 * dimension
        expected_targets = (len(current), len(self.horizons), target_dimension)
        if targets.shape != expected_targets:
            raise ValueError(f"targets must have shape {expected_targets}")

        target_by_horizon = dict(zip(self.horizons, targets.unbind(dim=1)))
        initial_velocity = model._frame_velocity(current, history_states)
        initial_oscillator = (
            model.velocity_oscillator_state(current, history_states)
            if model.config.velocity_oscillator_periods is not None else None)
        initial_innovation = (
            model.velocity_innovation_state(current, history_states)
            if model.config.velocity_innovation_offsets is not None else None)
        runtime_history = history_states
        if initial_oscillator is not None:
            runtime_history = history_states[..., :max(
                model.config.effective_history_offsets), :]

        active_particles = self.particle_counts[0]
        initial_state = current[:, None, :]
        state = current[:, None, :].expand(-1, active_particles, -1)
        velocity = initial_velocity[:, None, :].expand(-1, active_particles, -1)
        history = runtime_history[:, None, :, :].expand(
            -1, active_particles, -1, -1)
        oscillator_state = (None if initial_oscillator is None else
                            initial_oscillator[:, None, :].expand(
                                -1, active_particles, -1))
        innovation_state = (None if initial_innovation is None else
                            initial_innovation[:, None, :, :].expand(
                                -1, active_particles, -1, -1))
        step_size = model.config.lag_steps
        losses: dict[int, Tensor] = {}

        tcn_cache = model.new_tcn_cache() if getattr(
            model, "tcn_rollout_cache", False) else None
        initial_pair = None
        if tcn_cache is not None and not zero_drift:
            first_mean, first_factor = model.velocity_base_step_distribution(
                current, runtime_history, initial_velocity, tcn_cache=tcn_cache)
            initial_pair = (
                first_mean[:, None].expand(-1, active_particles, -1),
                first_factor if first_factor.ndim == 2 else
                first_factor[:, None].expand(-1, active_particles, -1, -1))
            tcn_cache["buffers"] = tuple(
                buffer[:, None].expand(-1, active_particles, -1, -1)
                for buffer in tcn_cache["buffers"])

        maximum = self.horizons[-1]
        for step in range(1, maximum + 1):
            if step == 1 and initial_pair is not None:
                base_mean, factor = initial_pair
            else:
                base_mean, factor = model.velocity_base_step_distribution(
                    state, history, velocity, zero_drift=zero_drift,
                    oscillator_state=oscillator_state,
                    **({"tcn_cache": tcn_cache} if tcn_cache is not None else {}))
            mean = base_mean
            if innovation_state is not None and not zero_drift:
                mean = mean + model.target_std * model.velocity_innovation_correction(
                    innovation_state)
            noise = torch.randn(
                velocity.shape, device=velocity.device, dtype=velocity.dtype,
                generator=generator)
            stochastic = (torch.matmul(noise, factor.T) if factor.ndim == 2 else
                          torch.matmul(factor, noise.unsqueeze(-1)).squeeze(-1))
            following_velocity = mean + stochastic
            following_state = state + step_size * following_velocity

            if step in target_by_horizon:
                samples = following_velocity
                if self.target == "joint":
                    samples = torch.cat((following_state - initial_state,
                                         following_velocity), dim=-1)
                losses[step] = self._score_samples(
                    model, step, samples, target_by_horizon[step])
            if step == maximum:
                break

            if oscillator_state is not None:
                oscillator_state = model.advance_velocity_oscillator(
                    oscillator_state, following_velocity)
            if innovation_state is not None:
                innovation_state = model.advance_velocity_innovation(
                    innovation_state, following_velocity, base_mean)
            history = (state.unsqueeze(-2) if model.config.history_steps == 1 else
                       torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2))
            state, velocity = following_state, following_velocity

            if step in target_by_horizon:
                horizon_index = self.horizons.index(step)
                next_particles = self.particle_counts[horizon_index + 1]
                if next_particles < active_particles:
                    particle_slice = (slice(None), slice(0, next_particles))
                    state = state[particle_slice]
                    velocity = velocity[particle_slice]
                    history = history[particle_slice]
                    if oscillator_state is not None:
                        oscillator_state = oscillator_state[particle_slice]
                    if innovation_state is not None:
                        innovation_state = innovation_state[particle_slice]
                    if tcn_cache is not None and tcn_cache["buffers"] is not None:
                        tcn_cache["buffers"] = tuple(
                            buffer[particle_slice] for buffer in tcn_cache["buffers"])
                    active_particles = next_particles

        return {horizon: losses[horizon] for horizon in self.horizons}


class StationaryMeanLoss(nn.Module):
    """Match generated and measured endpoint moments with independent branches."""

    def __init__(self, horizon: int | tuple[int, ...] = 100, *,
                 pod_coordinate_offset: int = 0,
                 variance_pod_mode_count: int | None = None,
                 lagged_covariance_mode_count: int | None = None,
                 lagged_covariance_epsilon: float = 1e-6,
                 lagged_covariance_coordinates: tuple[int, ...] | None = None,
                 lagged_covariance_reference_variance: Tensor | None = None,
                 lagged_covariance_horizons: tuple[int, ...] | None = None):
        super().__init__()
        horizons = (horizon,) if isinstance(horizon, int) else tuple(horizon)
        if (not horizons
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                       for value in horizons)
                or tuple(sorted(set(horizons))) != horizons):
            raise ValueError(
                "stationary mean horizons must be strictly increasing positive integers")
        if (not isinstance(pod_coordinate_offset, int)
                or isinstance(pod_coordinate_offset, bool)
                or pod_coordinate_offset < 0):
            raise ValueError("pod coordinate offset must be a nonnegative integer")
        if (variance_pod_mode_count is not None
                and (not isinstance(variance_pod_mode_count, int)
                     or isinstance(variance_pod_mode_count, bool)
                     or variance_pod_mode_count < 0)):
            raise ValueError("variance POD mode count must be nonnegative")
        if (lagged_covariance_mode_count is not None
                and (not isinstance(lagged_covariance_mode_count, int)
                     or isinstance(lagged_covariance_mode_count, bool)
                     or lagged_covariance_mode_count < 1)):
            raise ValueError("lagged covariance mode count must be positive")
        if (not math.isfinite(lagged_covariance_epsilon)
                or lagged_covariance_epsilon <= 0):
            raise ValueError("lagged covariance epsilon must be positive")
        self.endpoint_horizons = horizons
        correlation_horizons = horizons if lagged_covariance_horizons is None else tuple(lagged_covariance_horizons)
        if (not correlation_horizons or tuple(sorted(set(correlation_horizons))) != correlation_horizons
                or any(not isinstance(h,int) or isinstance(h,bool) or h<1 for h in correlation_horizons)):
            raise ValueError("lagged covariance horizons must be increasing positive integers")
        self.lagged_covariance_horizons = correlation_horizons
        self.horizons = tuple(sorted(set(horizons+correlation_horizons)))
        self.pod_coordinate_offset = pod_coordinate_offset
        self.variance_pod_mode_count = variance_pod_mode_count
        self.lagged_covariance_mode_count = lagged_covariance_mode_count
        self.lagged_covariance_epsilon = lagged_covariance_epsilon
        if lagged_covariance_coordinates is not None:
            coordinates = tuple(lagged_covariance_coordinates)
            if (not coordinates or tuple(sorted(set(coordinates))) != coordinates
                    or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in coordinates)):
                raise ValueError("lagged covariance coordinates must be increasing nonnegative integers")
            if lagged_covariance_mode_count is not None:
                raise ValueError("choose explicit covariance coordinates OR leading POD mode count")
            lagged_covariance_coordinates = coordinates
        self.lagged_covariance_coordinates = lagged_covariance_coordinates
        variance = (None if lagged_covariance_reference_variance is None else
                    torch.as_tensor(lagged_covariance_reference_variance).detach().clone())
        if variance is not None and (variance.ndim != 1 or not torch.isfinite(variance).all()
                                     or (variance <= 0).any()):
            raise ValueError("reference variance must be a finite positive vector in normalized state units")
        self.register_buffer("lagged_covariance_reference_variance", variance)
        # Retain the scalar attribute used by existing training/sampling code.
        self.horizon = self.horizons[-1]

    def _per_horizon_statistics(
            self, model: NeuralSDE, current: Tensor, data_endpoint: Tensor, *,
            history_states: Tensor, generator: torch.Generator | None = None,
            include_lagged_covariance: bool = False) -> dict[int, tuple[Tensor, ...]]:
        """Return requested endpoint statistics along the same generated paths."""
        if model.config.state_variable != "velocity":
            raise ValueError("stationary mean loss requires state_variable velocity")
        dimension = model.config.reduced_dim
        if current.ndim != 2 or current.shape[-1] != dimension:
            raise ValueError("current must have shape [trajectory, reduced_dim]")
        if data_endpoint.ndim == 2:
            data_endpoint = data_endpoint.unsqueeze(1)
        expected_endpoints = (len(current), len(self.horizons), dimension)
        if data_endpoint.shape != expected_endpoints:
            raise ValueError(f"data_endpoint must have shape {expected_endpoints}")
        expected_history = (len(current), model.config.history_steps, dimension)
        if history_states.shape != expected_history:
            raise ValueError(f"history_states must have shape {expected_history}")
        variance_stop = (dimension if self.variance_pod_mode_count is None else
                         self.pod_coordinate_offset + self.variance_pod_mode_count)
        if variance_stop < 1 or variance_stop > dimension:
            raise ValueError(
                "stationary variance coordinate selection must contain at least one "
                "coordinate and cannot exceed the reduced dimension")
        variance_slice = slice(0, variance_stop)
        mode_stop = (dimension if self.lagged_covariance_mode_count is None else
                     self.pod_coordinate_offset + self.lagged_covariance_mode_count)
        if include_lagged_covariance and mode_stop > dimension:
            raise ValueError(
                "lagged covariance POD coordinate selection exceeds reduced dimension")
        mode_slice = (slice(self.pod_coordinate_offset, mode_stop)
                      if self.lagged_covariance_coordinates is None else
                      list(self.lagged_covariance_coordinates))
        if self.lagged_covariance_coordinates is not None and max(mode_slice) >= dimension:
            raise ValueError("lagged covariance coordinate exceeds reduced dimension")
        normalized_current = (current - model.state_mean) / model.state_std
        selected_current = normalized_current[:, mode_slice]
        centered_current = selected_current - selected_current.mean(dim=0, keepdim=True)
        start_variance = centered_current.square().sum(dim=0) / (len(current) - 1)
        if self.lagged_covariance_reference_variance is not None:
            if self.lagged_covariance_reference_variance.shape != (dimension,):
                raise ValueError("fixed reference variance must contain every model coordinate")
            start_variance = self.lagged_covariance_reference_variance[mode_slice].to(current)

        # Two branches share observed starts but use independent noise. Their
        # cross-product removes rollout Monte Carlo variance from the expected
        # squared batch-mean error. The same paths are reused at every horizon.
        state = current[:, None, :].expand(-1, 2, -1)
        history = history_states[:, None, :, :].expand(-1, 2, -1, -1)
        velocity = model._frame_velocity(state, history)
        oscillator_state = (
            model.velocity_oscillator_state(state, history)
            if model.config.velocity_oscillator_periods is not None else None)
        innovation_state = (
            model.velocity_innovation_state(state, history)
            if model.config.velocity_innovation_offsets is not None else None)
        if oscillator_state is not None:
            history = history[..., :max(model.config.effective_history_offsets), :]
        step_size = model.config.lag_steps
        results = {}
        endpoint_index = 0
        tcn_cache = model.new_tcn_cache() if getattr(model, "tcn_rollout_cache", False) else None
        for step in range(1, self.horizon + 1):
            base_mean, factor = model.velocity_base_step_distribution(
                state, history, velocity, oscillator_state=oscillator_state,
                **({"tcn_cache": tcn_cache} if tcn_cache is not None else {}))
            mean = base_mean
            if innovation_state is not None:
                mean = mean + model.target_std * model.velocity_innovation_correction(
                    innovation_state)
            noise = torch.randn(
                velocity.shape, device=velocity.device, dtype=velocity.dtype,
                generator=generator)
            stochastic = (torch.matmul(noise, factor.T) if factor.ndim == 2 else
                          torch.matmul(factor, noise.unsqueeze(-1)).squeeze(-1))
            following_velocity = mean + stochastic
            following_state = state + step_size * following_velocity
            if oscillator_state is not None:
                oscillator_state = model.advance_velocity_oscillator(
                    oscillator_state, following_velocity)
            if innovation_state is not None:
                innovation_state = model.advance_velocity_innovation(
                    innovation_state, following_velocity, base_mean)
            history = (state.unsqueeze(-2) if model.config.history_steps == 1 else
                       torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2))
            state, velocity = following_state, following_velocity
            if step != self.horizons[endpoint_index]:
                continue

            normalized_state = (state - model.state_mean) / model.state_std
            normalized_data = (
                data_endpoint[:, endpoint_index] - model.state_mean) / model.state_std
            generated_means = normalized_state.mean(dim=0)
            data_mean = normalized_data.mean(dim=0)
            mean_errors = generated_means - data_mean
            mean_loss = mean_errors[0].mul(mean_errors[1]).mean()

            generated_variances = normalized_state.var(dim=0, correction=1)
            data_variance = normalized_data.var(dim=0, correction=1)
            variance_errors = generated_variances - data_variance
            variance_loss = variance_errors[0, variance_slice].mul(
                variance_errors[1, variance_slice]).mean()
            statistics = (mean_loss, mean_errors.mean(dim=0),
                          variance_loss, variance_errors.mean(dim=0))
            if include_lagged_covariance:
                selected_generated = normalized_state[..., mode_slice]
                selected_data = normalized_data[..., mode_slice]
                centered_generated = (
                    selected_generated
                    - selected_generated.mean(dim=0, keepdim=True))
                centered_data = selected_data - selected_data.mean(dim=0, keepdim=True)
                generated_covariance = (
                    centered_current[:, None, :].mul(centered_generated).sum(dim=0)
                    / (len(current) - 1))
                data_covariance = (
                    centered_current.mul(centered_data).sum(dim=0) / (len(current) - 1))
                lagged_covariance_errors = (
                    (generated_covariance - data_covariance.unsqueeze(0))
                    / (start_variance.unsqueeze(0) + self.lagged_covariance_epsilon))
                lagged_covariance_loss = lagged_covariance_errors[0].mul(
                    lagged_covariance_errors[1]).mean()
                statistics += (lagged_covariance_loss,
                               lagged_covariance_errors.mean(dim=0))
            results[step] = statistics
            endpoint_index += 1
            if endpoint_index == len(self.horizons):
                break
        return results

    def selected_variance_error(self, error: Tensor) -> Tensor:
        """Select the coordinates used by the endpoint-variance reduction."""
        if self.variance_pod_mode_count is None:
            return error
        stop = self.pod_coordinate_offset + self.variance_pod_mode_count
        if stop < 1 or stop > error.shape[-1]:
            raise ValueError(
                "stationary variance coordinate selection must contain at least one "
                "coordinate and cannot exceed the reduced dimension")
        return error[..., :stop]

    def per_horizon_statistics(
            self, model: NeuralSDE, current: Tensor, data_endpoint: Tensor, *,
            history_states: Tensor, generator: torch.Generator | None = None
            ) -> dict[int, tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Return mean/variance losses and errors at every requested horizon."""
        return self._per_horizon_statistics(
            model, current, data_endpoint, history_states=history_states,
            generator=generator, include_lagged_covariance=False)

    def per_horizon_statistics_with_lagged_covariance(
            self, model: NeuralSDE, current: Tensor, data_endpoint: Tensor, *,
            history_states: Tensor, generator: torch.Generator | None = None
            ) -> dict[int, tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Also match centered start-to-endpoint covariance for selected POD modes."""
        return self._per_horizon_statistics(
            model, current, data_endpoint, history_states=history_states,
            generator=generator, include_lagged_covariance=True)

    def statistics(self, model: NeuralSDE, current: Tensor, data_endpoint: Tensor, *,
                   history_states: Tensor,
                   generator: torch.Generator | None = None
                   ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        per_horizon = self.per_horizon_statistics(
            model, current, data_endpoint, history_states=history_states,
            generator=generator)
        columns = tuple(zip(*(per_horizon[h] for h in self.endpoint_horizons)))
        return tuple(torch.stack(values).mean(dim=0) for values in columns)

    def forward(self, model: NeuralSDE, current: Tensor, data_endpoint: Tensor, *,
                history_states: Tensor,
                generator: torch.Generator | None = None) -> Tensor:
        mean_loss, _, _, _ = self.statistics(
            model, current, data_endpoint, history_states=history_states,
            generator=generator)
        return mean_loss
