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
                           velocity: Tensor, *, zero_drift: bool = False
                           ) -> tuple[Tensor, Tensor]:
    """Euler--Maruyama conditional mean/factor for native-frame velocity."""
    return model.velocity_step_distribution(
        state, history_states, velocity, zero_drift=zero_drift)


class EulerMaruyamaNLL(nn.Module):
    """Mean physical-coordinate Gaussian NLL for the configured SDE variable."""

    def forward(self, model: NeuralSDE, current: Tensor, following: Tensor,
                *, history_states: Tensor | None = None,
                following_history_states: Tensor | None = None,
                zero_drift: bool = False) -> Tensor:
        return self.per_example(
            model, current, following, history_states=history_states,
            following_history_states=following_history_states,
            zero_drift=zero_drift).mean()

    def per_example(self, model: NeuralSDE, current: Tensor, following: Tensor,
                    *, history_states: Tensor | None = None,
                    following_history_states: Tensor | None = None,
                    zero_drift: bool = False) -> Tensor:
        """Return the existing joint-coordinate NLL before batch averaging."""
        if current.ndim != 2 or current.shape != following.shape:
            raise ValueError("current and following must be matching [batch, reduced_dim] tensors")
        if model.config.state_variable == "velocity":
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

    def __init__(self, horizons: tuple[int, ...] = (2, 4), particles: int = 100,
                 *, target: str = "velocity",
                 mode_counts: tuple[int, ...] | None = None,
                 pod_coordinate_offset: int = 0,
                 particle_counts: tuple[int, ...] | None = None):
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
        return torch.stack(tuple(losses.values())).mean()

    def per_horizon(self, model: NeuralSDE, current: Tensor, targets: Tensor,
                    *, history_states: Tensor,
                    generator: torch.Generator | None = None,
                    zero_drift: bool = False) -> dict[int, Tensor]:
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
        step_size = model.config.lag_steps

        # H=1 has no latent path. Evaluate it without a redundant identical
        # particle mixture. Displacement is h*v_next, including its Jacobian.
        if 1 in target_by_horizon:
            mean, factor = _velocity_distribution(
                model, current, history_states, initial_velocity,
                zero_drift=zero_drift)
            if self.target == "displacement":
                mean, factor = step_size * mean, step_size * factor
            target, mean, factor = self._scored_distribution(
                1, target_by_horizon[1], mean, factor)
            losses[1] = -_gaussian_log_prob(
                target, mean, factor).mean()

        maximum = self.horizons[-1]
        if maximum == 1:
            return losses
        first_latent_index = next(
            index for index, horizon in enumerate(self.horizons) if horizon > 1)
        active_particles = self.particle_counts[first_latent_index]
        state = current[:, None, :].expand(-1, active_particles, -1)
        velocity = initial_velocity[:, None, :].expand(-1, active_particles, -1)
        history = history_states[:, None, :, :].expand(
            -1, active_particles, -1, -1)
        # Accumulate relative displacement directly, avoiding subtraction of
        # potentially large absolute positions in the mixture means.
        displacement = torch.zeros_like(state) if self.target == "displacement" else None

        for step in range(1, maximum + 1):
            mean, factor = _velocity_distribution(
                model, state, history, velocity, zero_drift=zero_drift)
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
                mixture_nll = (-torch.logsumexp(log_prob, dim=1)
                               + math.log(active_particles))
                losses[step] = mixture_nll.mean()
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
                    mean = mean[particle_slice]
                    if factor.ndim > 2:
                        factor = factor[particle_slice]
                    if displacement is not None:
                        displacement = displacement[particle_slice]
                    active_particles = next_particles
            noise = torch.randn(
                velocity.shape, device=velocity.device, dtype=velocity.dtype,
                generator=generator)
            stochastic = (torch.matmul(noise, factor.T) if factor.ndim == 2 else
                          torch.matmul(factor, noise.unsqueeze(-1)).squeeze(-1))
            following_velocity = mean + stochastic
            following_state = state + step_size * following_velocity
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


class StationaryMeanLoss(nn.Module):
    """Match generated and measured endpoint means with independent branches."""

    def __init__(self, horizon: int = 100):
        super().__init__()
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("stationary mean horizon must be a positive integer")
        self.horizon = horizon

    def statistics(self, model: NeuralSDE, current: Tensor, data_endpoint: Tensor, *,
                   history_states: Tensor,
                   generator: torch.Generator | None = None
                   ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if model.config.state_variable != "velocity":
            raise ValueError("stationary mean loss requires state_variable velocity")
        dimension = model.config.reduced_dim
        if current.ndim != 2 or current.shape[-1] != dimension:
            raise ValueError("current must have shape [trajectory, reduced_dim]")
        if data_endpoint.shape != current.shape:
            raise ValueError("data_endpoint must match current")
        expected_history = (len(current), model.config.history_steps, dimension)
        if history_states.shape != expected_history:
            raise ValueError(f"history_states must have shape {expected_history}")

        # Two branches share observed starts but use independent noise. Their
        # cross-product removes rollout Monte Carlo variance from the expected
        # squared batch-mean error.
        state = current[:, None, :].expand(-1, 2, -1)
        history = history_states[:, None, :, :].expand(-1, 2, -1, -1)
        velocity = model._frame_velocity(state, history)
        step_size = model.config.lag_steps
        for _ in range(self.horizon):
            mean, factor = _velocity_distribution(model, state, history, velocity)
            noise = torch.randn(
                velocity.shape, device=velocity.device, dtype=velocity.dtype,
                generator=generator)
            stochastic = (torch.matmul(noise, factor.T) if factor.ndim == 2 else
                          torch.matmul(factor, noise.unsqueeze(-1)).squeeze(-1))
            following_velocity = mean + stochastic
            following_state = state + step_size * following_velocity
            history = (state.unsqueeze(-2) if model.config.history_steps == 1 else
                       torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2))
            state, velocity = following_state, following_velocity

        normalized_state = (state - model.state_mean) / model.state_std
        normalized_data = (data_endpoint - model.state_mean) / model.state_std
        generated_means = normalized_state.mean(dim=0)
        data_mean = normalized_data.mean(dim=0)
        mean_errors = generated_means - data_mean
        mean_loss = mean_errors[0].mul(mean_errors[1]).mean()

        generated_variances = normalized_state.var(dim=0, correction=1)
        data_variance = normalized_data.var(dim=0, correction=1)
        variance_errors = generated_variances - data_variance
        variance_loss = variance_errors[0].mul(variance_errors[1]).mean()
        return (mean_loss, mean_errors.mean(dim=0),
                variance_loss, variance_errors.mean(dim=0))

    def forward(self, model: NeuralSDE, current: Tensor, data_endpoint: Tensor, *,
                history_states: Tensor,
                generator: torch.Generator | None = None) -> Tensor:
        mean_loss, _, _, _ = self.statistics(
            model, current, data_endpoint, history_states=history_states,
            generator=generator)
        return mean_loss
