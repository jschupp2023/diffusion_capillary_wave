"""Numerical core copied from the original stochastic-OpInf workflow."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter
import torch
from torch import Tensor, nn

from .config import OpInfConfig


def page_cov(states: np.ndarray) -> np.ndarray:
    """Sample covariance at each time for [coordinate, realization, time]."""
    if states.ndim != 3 or states.shape[1] < 2:
        raise ValueError("states must contain at least two realizations")
    centered = states - states.mean(axis=1, keepdims=True)
    return np.einsum("ilt,jlt->ijt", centered, centered) / (states.shape[1] - 1)


def central_finite_differences(
    values: np.ndarray, dt: float, order: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original centered first-derivative stencil along the time axis."""
    stencils = {
        2: ([-1, 0, 1], [-1 / 2, 0, 1 / 2]),
        4: ([-2, -1, 0, 1, 2], [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
        6: ([-3, -2, -1, 0, 1, 2, 3],
            [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
        8: ([-4, -3, -2, -1, 0, 1, 2, 3, 4],
            [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]),
    }
    if order not in stencils:
        raise ValueError("order must be 2, 4, 6, or 8")
    offsets, coefficients = stencils[order]
    radius = len(offsets) // 2
    if values.ndim != 2 or values.shape[1] <= 2 * radius:
        raise ValueError("not enough time samples for the derivative stencil")
    indices = np.arange(radius, values.shape[1] - radius)
    derivative = sum(
        coefficient * values[:, indices + offset]
        for offset, coefficient in zip(offsets, coefficients)
    ) / dt
    return values[:, indices], derivative, indices


def regularizer(
    dimension: int,
    regularization: np.ndarray,
    is_bilinear: bool,
    has_input_term: bool,
) -> np.ndarray:
    """Construct the original block-diagonal Tikhonov factor."""
    width = dimension + int(has_input_term) + dimension * int(is_bilinear)
    diagonal = np.zeros(width)
    diagonal[:dimension] = regularization[0]
    if has_input_term:
        diagonal[dimension] = regularization[1]
        if is_bilinear:
            diagonal[dimension + 1:] = regularization[2]
    elif is_bilinear:
        diagonal[dimension:] = regularization[1]
    return np.diag(diagonal)


def infer_drift(
    mean: np.ndarray,
    input_signal: np.ndarray,
    dt: float,
    *,
    is_bilinear: bool,
    has_input_term: bool,
    regularization: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer E, A, B, and N with the original regularized least squares."""
    state, derivative, indices = central_finite_differences(mean, dt, order=6)
    count = len(indices)
    # Preserve the original implementation's alignment of u[:count] with the
    # derivative-interior state columns.
    input_interior = input_signal[:count]
    blocks = [state]
    if has_input_term:
        blocks.append(input_interior[None])
    if is_bilinear:
        blocks.append(state * input_interior[None])
    data = np.vstack(blocks)
    gamma = regularizer(mean.shape[0], regularization, is_bilinear, has_input_term)
    normal = data @ data.T + gamma.T @ gamma
    operators = np.linalg.solve(normal, data @ derivative.T).T

    dimension = mean.shape[0]
    drift = operators[:, :dimension]
    cursor = dimension
    if has_input_term:
        input_operator = operators[:, cursor:cursor + 1]
        cursor += 1
    else:
        input_operator = np.zeros((dimension, 1))
    bilinear = operators[:, cursor:] if is_bilinear else np.zeros((dimension, dimension))
    return np.eye(dimension), drift, input_operator, bilinear


def infer_diffusion(
    covariance: np.ndarray,
    input_signal: np.ndarray,
    dt: float,
    drift: np.ndarray,
    bilinear: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer constant diffusion from covariance dynamics, then project to PSD."""
    derivative = savgol_filter(
        covariance, window_length=7, polyorder=3, deriv=1, delta=dt, axis=2
    )
    covariance_rate = np.zeros_like(drift)
    for time_index in range(covariance.shape[2]):
        effective_drift = drift + bilinear * input_signal[time_index]
        lyapunov = effective_drift @ covariance[:, :, time_index]
        residual = derivative[:, :, time_index] - (lyapunov + lyapunov.T)
        covariance_rate += 0.5 * (residual + residual.T)
    covariance_rate /= covariance.shape[2] + regularization

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_rate)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    tolerance = eigenvalues.max(initial=0.0) / 1.0e3
    noise_dimension = int(np.count_nonzero(eigenvalues >= tolerance))
    if noise_dimension:
        diffusion = eigenvectors[:, :noise_dimension] * np.sqrt(
            eigenvalues[:noise_dimension]
        )[None]
    else:
        diffusion = np.zeros((drift.shape[0], 0))
    return diffusion, np.eye(noise_dimension)


def input_signal(config: OpInfConfig, count: int) -> np.ndarray:
    """Relative-time cosine input used by every training/evaluation window."""
    time = np.arange(count, dtype=np.float64) * config.dt
    return config.input_amplitude * np.cos(2 * np.pi * config.input_frequency * time)


def original_moment_errors(
    *,
    config: OpInfConfig,
    mass: np.ndarray,
    drift: np.ndarray,
    input_operator: np.ndarray,
    bilinear: np.ndarray,
    diffusion: np.ndarray,
    initial_conditions: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    """Reproduce the original grid-search simulation and aggregate errors.

    In particular, output index zero is the result of one implicit step and
    the first two updates both use ``u[0]``, matching ``stepSDE`` exactly.
    """
    rng = np.random.RandomState(seed)
    signal = input_signal(config, reference_mean.shape[1])
    state = np.asarray(initial_conditions, dtype=np.float64).copy()
    noise_all = rng.standard_normal(
        (diffusion.shape[1], state.shape[1], reference_mean.shape[1])
    )
    mean_error_squared = covariance_error_squared = 0.0
    for index in range(reference_mean.shape[1]):
        input_index = 0 if index == 0 else index - 1
        value = signal[input_index]
        lhs = mass - config.dt * drift - config.dt * bilinear * value
        noise = noise_all[:, :, index]
        rhs = state + config.dt * input_operator * value
        if diffusion.shape[1]:
            rhs = rhs + math.sqrt(config.dt) * config.sigma * diffusion @ noise
        state = np.linalg.solve(lhs, rhs)
        predicted_mean = state.mean(axis=1)
        centered = state - predicted_mean[:, None]
        predicted_covariance = centered @ centered.T / (state.shape[1] - 1)
        mean_error_squared += np.square(
            reference_mean[:, index] - predicted_mean
        ).sum()
        covariance_error_squared += np.square(
            reference_covariance[:, :, index] - predicted_covariance
        ).sum()
    mean_denominator = np.square(reference_mean).sum()
    covariance_denominator = np.square(reference_covariance).sum()
    return (
        math.sqrt(mean_error_squared / mean_denominator),
        math.sqrt(covariance_error_squared / covariance_denominator),
    )


class StochasticOpInfModel(nn.Module):
    """Frozen inferred operators with a shared-evaluator-compatible rollout."""

    def __init__(
        self,
        config: OpInfConfig,
        *,
        mass: np.ndarray | Tensor | None = None,
        drift: np.ndarray | Tensor | None = None,
        input_operator: np.ndarray | Tensor | None = None,
        bilinear: np.ndarray | Tensor | None = None,
        diffusion: np.ndarray | Tensor | None = None,
        state_mean: np.ndarray | Tensor | None = None,
        state_std: np.ndarray | Tensor | None = None,
        target_mean: np.ndarray | Tensor | None = None,
        target_std: np.ndarray | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        dimension = config.reduced_dim

        def tensor(value, default):
            return torch.as_tensor(default if value is None else value, dtype=torch.float32)

        # A frozen parameter lets the existing evaluator discover the device.
        self.mass = nn.Parameter(tensor(mass, np.eye(dimension)), requires_grad=False)
        self.register_buffer("drift", tensor(drift, np.zeros((dimension, dimension))))
        self.register_buffer("input_operator", tensor(input_operator, np.zeros((dimension, 1))))
        self.register_buffer("bilinear", tensor(bilinear, np.zeros((dimension, dimension))))
        if diffusion is None:
            diffusion = np.zeros((dimension, 0))
        self.register_buffer("diffusion", tensor(diffusion, diffusion))
        for name, value, default in (
            ("state_mean", state_mean, np.zeros(dimension)),
            ("state_std", state_std, np.ones(dimension)),
            ("target_mean", target_mean, np.zeros(dimension)),
            ("target_std", target_std, np.ones(dimension)),
        ):
            self.register_buffer(name, tensor(value, default))
        if self.diffusion.ndim != 2 or self.diffusion.shape[0] != dimension:
            raise ValueError("diffusion has incompatible dimensions")

    @torch.no_grad()
    def rollout(
        self,
        initial_state: Tensor,
        *,
        horizon: int,
        num_trajectories: int,
        history_states=None,
        num_steps=None,
        seed: int | None = None,
    ) -> Tensor:
        """Return physical states [condition, ensemble, horizon+1, coordinate]."""
        if initial_state.ndim != 2 or initial_state.shape[1] != self.config.reduced_dim:
            raise ValueError("initial_state must be [condition, reduced_dim]")
        if min(horizon, num_trajectories) < 1:
            raise ValueError("horizon and num_trajectories must be positive")
        if history_states is not None or num_steps is not None:
            raise ValueError("history and sampling steps are unsupported")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=initial_state.device).manual_seed(seed)
        normalized = (initial_state - self.state_mean) / self.state_std
        normalized = normalized[:, None].expand(-1, num_trajectories, -1).clone()
        trajectories = [initial_state[:, None].expand(-1, num_trajectories, -1)]
        for step in range(horizon):
            value = self.config.input_amplitude * math.cos(
                2 * math.pi * self.config.input_frequency * step * self.config.dt
            )
            lhs = self.mass - self.config.dt * self.drift - self.config.dt * self.bilinear * value
            rhs = normalized + self.config.dt * self.input_operator[:, 0] * value
            if self.diffusion.shape[1]:
                noise = torch.randn(
                    (*normalized.shape[:-1], self.diffusion.shape[1]),
                    dtype=normalized.dtype,
                    device=normalized.device,
                    generator=generator,
                )
                rhs = rhs + math.sqrt(self.config.dt) * self.config.sigma * noise @ self.diffusion.T
            normalized = torch.linalg.solve(lhs, rhs.reshape(-1, rhs.shape[-1]).T).T.reshape_as(rhs)
            trajectories.append(normalized * self.state_std + self.state_mean)
        return torch.stack(trajectories, dim=2)
