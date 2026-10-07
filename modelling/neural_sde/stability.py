"""Fixed quadratic, discrete conditional-drift regularization for velocity SDEs.

V is quadratic in physical, uncentered (z,v). Computation uses a fixed diagonal
change of units, never a learned metric or an estimated Monte Carlo expectation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
import torch
from torch import Tensor, nn

from .model import NeuralSDE


@dataclass(frozen=True)
class StabilityConfig:
    weight: float = 0.
    a: float = .01
    b: float = .01
    metric: str = "auto"
    perturb_scale: float = 0.
    generated_steps: int = 0
    validation_seed: int = 0

    def __post_init__(self):
        if (not math.isfinite(self.weight) or self.weight < 0
                or not math.isfinite(self.a) or self.a < 0
                or not math.isfinite(self.b) or not 0 < self.b < 1
                or not math.isfinite(self.perturb_scale) or self.perturb_scale < 0):
            raise ValueError("stability requires finite weight,a,perturb_scale >= 0 and 0 < b < 1")
        if self.metric not in {"auto", "lyapunov", "data_scaled"}:
            raise ValueError("stability metric must be auto, lyapunov, or data_scaled")
        if (not isinstance(self.generated_steps, int) or isinstance(self.generated_steps, bool)
                or self.generated_steps < 0 or not isinstance(self.validation_seed, int)
                or isinstance(self.validation_seed, bool)):
            raise ValueError("stability generated_steps must be a nonnegative integer; seed an integer")

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def discrete_linear_core(model: NeuralSDE) -> Tensor:
    """Homogeneous damped core in s_scaled = (z/state_std, v/target_std).

    The normalization means produce an affine offset in the actual update,
    retained by augmented_step_distribution. The residual is excluded only
    from this reference construction, never from the penalty.
    """
    if model.config.drift_type != "damped_residual":
        raise ValueError("discrete linear core requires damped_residual drift")
    k = model.drift_model.stiffness_matrix().detach().double()
    c = model.drift_model.damping_matrix().detach().double()
    identity = torch.eye(model.config.reduced_dim, dtype=k.dtype, device=k.device)
    ratio = torch.diag(model.target_std.double() / model.state_std.double())
    h = model.config.lag_steps
    vv = identity - h * c
    return torch.cat((torch.cat((identity - h*h * ratio @ k, h * ratio @ vv), dim=1),
                      torch.cat((-h * k, vv), dim=1)), dim=0)


class DiscreteLyapunovRegularizer(nn.Module):
    def __init__(self, config: StabilityConfig, metric: Tensor, scale: Tensor,
                 reference: dict):
        super().__init__()
        self.config = config
        # Double precision limits cancellation in E[V(next)] - V(current).
        self.register_buffer("metric", metric.detach().clone().double())
        self.register_buffer("scale", scale.detach().clone().double())
        self.reference = reference

    @property
    def P(self) -> Tensor:
        """Frozen matrix in physical coordinates, V(s)=s^T P s."""
        return self.metric / self.scale[:, None] / self.scale[None, :]

    def value(self, augmented: Tensor) -> Tensor:
        scaled = augmented.double() / self.scale
        return ((scaled @ self.metric) * scaled).sum(-1)

    def diagnostics(self, model: NeuralSDE, state: Tensor, history: Tensor,
                    velocity: Tensor | None = None, *, zero_drift: bool = False):
        if velocity is None:
            velocity = model._frame_velocity(state, history)
        mean, factor = model.augmented_step_distribution(
            state, history, velocity, zero_drift=zero_drift)
        scaled_factor = factor.double() / self.scale[:, None]
        # tr(P H H^T) = tr(H^T P H), including both covariance cross blocks.
        trace = (scaled_factor * (self.metric @ scaled_factor)).sum(dim=(-2, -1))
        value = self.value(torch.cat((state, velocity), dim=-1))
        delta = self.value(mean) + trace - value
        excess = delta - self.config.a + self.config.b * value
        return dict(penalty=excess.relu().square().mean(),
                    violation_fraction=(excess > 0).double().mean(),
                    mean_v=value.mean(), mean_delta_v=delta.mean(),
                    mean_excess=excess.mean(), max_excess=excess.max())

    def forward(self, model: NeuralSDE, state: Tensor, *, history_states: Tensor,
                generator: torch.Generator | None = None, zero_drift: bool = False):
        """Pool observed states, one optional perturbed copy, and generated steps.

        Generated states are detached sampling locations, not BPTT paths. The
        conditional drift and diffusion at every location remain differentiable.
        """
        velocity = model._frame_velocity(state, history_states)
        groups = {"observed": [self.diagnostics(
            model, state, history_states, velocity, zero_drift=zero_drift)]}
        if self.config.perturb_scale > 0:
            with torch.no_grad():
                noise = torch.randn((*state.shape[:-1], 2 * state.shape[-1]),
                                    device=state.device, dtype=state.dtype, generator=generator)
                dz, dv = (self.config.perturb_scale * self.scale.to(state.dtype) * noise).chunk(2, -1)
                offsets = torch.arange(1, model.config.history_steps + 1,
                                       device=state.device, dtype=state.dtype)[:, None]
                perturbed_history = history_states + dz.unsqueeze(-2) - offsets * dv.unsqueeze(-2)
            groups["perturbed"] = [self.diagnostics(
                model, state + dz, perturbed_history, velocity + dv, zero_drift=zero_drift)]
        if self.config.generated_steps:
            groups["generated"] = []
            history = history_states
            for _ in range(self.config.generated_steps):
                with torch.no_grad():
                    mean, factor = model.velocity_step_distribution(
                        state, history, velocity, zero_drift=zero_drift)
                    noise = torch.randn(velocity.shape, device=state.device,
                                        dtype=state.dtype, generator=generator)
                    velocity = mean + (factor @ noise.unsqueeze(-1)).squeeze(-1)
                    following = state + model.config.lag_steps * velocity
                    history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
                    state = following
                groups["generated"].append(self.diagnostics(
                    model, state, history, velocity, zero_drift=zero_drift))
        all_diagnostics = [entry for entries in groups.values() for entry in entries]

        def aggregate(entries):
            return {key: (torch.stack([entry[key] for entry in entries]).max()
                          if key == "max_excess" else
                          torch.stack([entry[key] for entry in entries]).mean())
                    for key in entries[0]}

        diagnostics = aggregate(all_diagnostics)
        for name, entries in groups.items():
            diagnostics.update({f"{name}_{key}": value for key, value in aggregate(entries).items()})
        diagnostics["weighted_penalty"] = self.config.weight * diagnostics["penalty"]
        return diagnostics

    def to_metadata(self):
        return dict(config=self.config.to_dict(), reference=self.reference,
                    metric=self.metric.cpu().tolist(), scale=self.scale.cpu().tolist(),
                    P=self.P.cpu().tolist(), coordinate_space="physical_uncentered_position_velocity")


@torch.no_grad()
def fit_stability_regularizer(model: NeuralSDE, config: StabilityConfig, batches
                             ) -> DiscreteLyapunovRegularizer:
    """Freeze an initial-core Lyapunov or empirical metric using training only.

    ``batches`` yields retained physical (state, history) tensors. In scaled
    coordinates solve F^T P F - P = -I, then divide P by training mean V.
    In physical coordinates Q is diag(scale^-2), divided by the same mean.
    """
    if model.config.state_variable != "velocity":
        raise ValueError("stability regularization requires state_variable velocity")
    scale = torch.cat((model.state_std, model.target_std)).detach().double()
    dimension = len(scale)
    metric = torch.eye(dimension, dtype=torch.float64, device=scale.device)
    reference = dict(kind="data_scaled", empirical=True, lag_steps=model.config.lag_steps,
                     history_in_v=False, history_offsets=list(model.config.effective_history_offsets))
    reason = None
    if config.metric != "data_scaled":
        if model.config.drift_type != "damped_residual":
            reason = "no damped-residual linear core"
        else:
            core = discrete_linear_core(model).cpu().numpy()
            radius = float(np.max(np.abs(np.linalg.eigvals(core))))
            reference.update(spectral_radius=radius, F_scaled=core.tolist())
            if not math.isfinite(radius) or radius >= 1:
                reason = f"initial discrete core is not Schur-stable (spectral radius={radius:g})"
            else:
                from scipy.linalg import solve_discrete_lyapunov

                try:
                    # scipy solves A X A^T-X=-Q; transpose F to obtain F^T P F.
                    solution = solve_discrete_lyapunov(core.T, np.eye(dimension))
                    solution = (solution + solution.T) / 2
                    residual = core.T @ solution @ core - solution + np.eye(dimension)
                    error = float(np.linalg.norm(residual) / np.sqrt(dimension))
                    if (not np.isfinite(solution).all() or not math.isfinite(error)
                            or error > 1e-6 or np.linalg.eigvalsh(solution).min() <= 0):
                        raise ValueError("non-positive or inaccurate discrete Lyapunov solution")
                    metric = torch.as_tensor(solution, device=scale.device)
                    reference.update(kind="lyapunov", empirical=False,
                                     equation_relative_residual=error)
                except (ValueError, np.linalg.LinAlgError) as error:
                    reason = str(error)
        if reason:
            if config.metric == "lyapunov":
                raise ValueError(f"Lyapunov metric unavailable: {reason}")
            warnings.warn(f"Using empirical data-scaled stability metric: {reason}", stacklevel=2)
            reference["fallback_reason"] = reason
    total, count = 0., 0
    for state, history in batches:
        state, history = state.to(scale.device), history.to(scale.device)
        velocity = model._frame_velocity(state, history)
        scaled = torch.cat((state, velocity), dim=-1).double() / scale
        total += float(((scaled @ metric) * scaled).sum())
        count += len(state)
    mean = total / count if count else 0.
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("stability metric requires a positive finite training mean V")
    metric = metric / mean
    reference.update(training_state_count=count, unscaled_training_mean_v=mean,
                     training_mean_v=1., Q_scaled_diagonal=(
                         [1. / mean] * dimension if reference["kind"] == "lyapunov" else None))
    return DiscreteLyapunovRegularizer(config, metric, scale, reference)
