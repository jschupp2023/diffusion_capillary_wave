"""Separated drift, diffusion, and Euler-Maruyama sampling."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import SDEConfig


class MLPDrift(nn.Module):
    """Per-step drift from standardized state and history features."""

    def __init__(self, dimension: int, hidden_layers: tuple[int, ...], activation: str,
                 history_steps: int = 0, input_blocks: int | None = None):
        super().__init__()
        nonlinearity = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        widths = (dimension * (input_blocks or history_steps + 1), *hidden_layers, dimension)
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(nonlinearity())
        self.network = nn.Sequential(*layers)

    def forward(self, normalized_input: Tensor) -> Tensor:
        return self.network(normalized_input)

    def zero_output(self) -> None:
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


class SPSDOperator(nn.Module):
    """Symmetric positive-semidefinite matrix represented as ``L L^T``."""

    def __init__(self, dimension: int, initial_value: float):
        super().__init__()
        indices = torch.tril_indices(dimension, dimension)
        initial = torch.zeros(indices.shape[1])
        initial[indices[0] == indices[1]] = math.sqrt(initial_value)
        self.entries = nn.Parameter(initial)
        self.dimension = dimension
        self.register_buffer("rows", indices[0], persistent=False)
        self.register_buffer("columns", indices[1], persistent=False)

    def factor(self) -> Tensor:
        factor = self.entries.new_zeros((self.dimension, self.dimension))
        factor[self.rows, self.columns] = self.entries
        return factor

    def forward(self) -> Tensor:
        factor = self.factor()
        return factor @ factor.T


class DampedResidualDrift(nn.Module):
    """Constrained restoring/damping operators plus a nonlinear residual.

    The operators act on separately standardized position and native-frame
    velocity, so their learned values are normalized-coordinate coefficients.
    """

    def __init__(self, dimension: int, hidden_layers: tuple[int, ...], activation: str,
                 stiffness_init: float, damping_init: float, operator: str = "diagonal",
                 input_blocks: int = 2):
        super().__init__()
        self.dimension = dimension
        self.operator = operator
        if operator == "spsd":
            self.stiffness_factor = SPSDOperator(dimension, stiffness_init)
            self.damping_factor = SPSDOperator(dimension, damping_init)
        else:
            self.raw_stiffness = nn.Parameter(torch.full(
                (dimension,), stiffness_init + math.log(-math.expm1(-stiffness_init))))
            self.raw_damping = nn.Parameter(torch.full(
                (dimension,), damping_init + math.log(-math.expm1(-damping_init))))
        self.residual = MLPDrift(
            dimension, hidden_layers, activation, input_blocks=input_blocks)
        self.residual.zero_output()

    @property
    def network(self):
        """Expose the residual MLP for existing architecture diagnostics."""
        return self.residual.network

    def stiffness_diagonal(self) -> Tensor:
        return torch.diagonal(self.stiffness_matrix())

    def damping_diagonal(self) -> Tensor:
        return torch.diagonal(self.damping_matrix())

    def stiffness_matrix(self) -> Tensor:
        if self.operator == "spsd":
            return self.stiffness_factor()
        return torch.diag(F.softplus(self.raw_stiffness))

    def damping_matrix(self) -> Tensor:
        if self.operator == "spsd":
            return self.damping_factor()
        return torch.diag(F.softplus(self.raw_damping))

    def forward(self, normalized_input: Tensor) -> Tensor:
        if normalized_input.shape[-1] < 2 * self.dimension:
            raise ValueError("damped_residual drift requires position and velocity blocks")
        position = normalized_input[..., :self.dimension]
        velocity = normalized_input[..., self.dimension:2 * self.dimension]
        if self.operator == "spsd":
            linear = (-position @ self.stiffness_matrix()
                      - velocity @ self.damping_matrix())
        else:
            linear = (-F.softplus(self.raw_stiffness) * position
                      - F.softplus(self.raw_damping) * velocity)
        return linear + self.residual(normalized_input)

    def zero_output(self) -> None:
        # Diffusion-only Stage 1 checkpoints must retain their legacy zero-drift
        # behavior; diagonal softplus coefficients are made numerically negligible.
        with torch.no_grad():
            if self.operator == "spsd":
                self.stiffness_factor.entries.zero_()
                self.damping_factor.entries.zero_()
            else:
                self.raw_stiffness.fill_(-30.)
                self.raw_damping.fill_(-30.)
        self.residual.zero_output()


class CapillaryEnergyDrift(nn.Module):
    """Fixed capillary restoring energy with learned mass, damping, and residual.

    Position and velocity are the model's standardized variables. Velocity is
    the native-frame displacement used by velocity mode, so this module returns
    its change per frame; the existing loss and rollout apply ``lag_steps``.
    """

    def __init__(self, dimension: int, hidden_layers: tuple[int, ...], activation: str,
                 stiffness_init: float, damping_init: float, mean_mode: bool,
                 input_blocks: int = 2):
        super().__init__()
        self.dimension = dimension
        self.mean_mode = mean_mode
        mass_multiplier = 1. / stiffness_init
        raw_mass = mass_multiplier + math.log(-math.expm1(-mass_multiplier))
        self.raw_mass_multiplier = nn.Parameter(torch.full((dimension,), raw_mass))
        self.damping_factor = SPSDOperator(
            dimension, damping_init / stiffness_init)
        if mean_mode:
            raw_stiffness = stiffness_init + math.log(-math.expm1(-stiffness_init))
            self.raw_mean_stiffness = nn.Parameter(torch.tensor(raw_stiffness))
        else:
            self.register_parameter("raw_mean_stiffness", None)
        self.residual = MLPDrift(
            dimension, hidden_layers, activation, input_blocks=input_blocks)
        self.residual.zero_output()
        self.register_buffer("energy_matrix", torch.zeros(dimension, dimension))
        self.register_buffer("mass_scale", torch.ones(dimension))

    @property
    def network(self):
        """Expose the residual MLP for existing architecture diagnostics."""
        return self.residual.network

    def set_energy_matrix(self, matrix: Tensor) -> None:
        if (matrix.shape != self.energy_matrix.shape or not torch.isfinite(matrix).all()
                or not torch.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8)):
            raise ValueError("capillary energy matrix must be finite and symmetric")
        eigenvalues = torch.linalg.eigvalsh(matrix.double())
        tolerance = 1e-7 * max(float(eigenvalues.abs().max()), torch.finfo(torch.float64).tiny)
        if float(eigenvalues.min()) < -tolerance:
            raise ValueError("capillary energy matrix must be positive semidefinite")
        diagonal = torch.diagonal(matrix).clone()
        pod_diagonal = diagonal[int(self.mean_mode):]
        positive = pod_diagonal[pod_diagonal > tolerance]
        if not len(positive):
            raise ValueError("capillary energy matrix has no positive POD diagonal")
        representative = positive.median()
        floor = representative * 1e-8
        scale = torch.clamp(diagonal, min=floor)
        if self.mean_mode:
            scale[0] = representative
        self.energy_matrix.copy_(matrix)
        self.mass_scale.copy_(scale)

    def mass_diagonal(self) -> Tensor:
        return self.mass_scale * F.softplus(self.raw_mass_multiplier)

    def damping_matrix(self) -> Tensor:
        root_scale = torch.sqrt(self.mass_scale)
        return root_scale[:, None] * self.damping_factor() * root_scale[None, :]

    def mean_stiffness(self) -> Tensor | None:
        return (F.softplus(self.raw_mean_stiffness)
                if self.raw_mean_stiffness is not None else None)

    def forward(self, normalized_input: Tensor) -> Tensor:
        if normalized_input.shape[-1] < 2 * self.dimension:
            raise ValueError("capillary_energy drift requires position and velocity blocks")
        position = normalized_input[..., :self.dimension]
        velocity = normalized_input[..., self.dimension:2 * self.dimension]
        force = position @ self.energy_matrix + velocity @ self.damping_matrix()
        mass = torch.diag(self.mass_diagonal())
        linear = -torch.linalg.solve(mass, force.unsqueeze(-1)).squeeze(-1)
        if self.mean_mode:
            mean_restoring = self.mean_stiffness() * position[..., 0]
            linear = torch.cat(((linear[..., 0] - mean_restoring).unsqueeze(-1),
                                linear[..., 1:]), dim=-1)
        return linear + self.residual(normalized_input)

    def zero_output(self) -> None:
        with torch.no_grad():
            self.energy_matrix.zero_()
            self.damping_factor.entries.zero_()
            if self.raw_mean_stiffness is not None:
                self.raw_mean_stiffness.fill_(-30.)
        self.residual.zero_output()

    def reset_structured_terms(self, stiffness_init: float, damping_init: float) -> None:
        """Restore trainable structured terms after a zero-drift diffusion stage."""
        with torch.no_grad():
            self.damping_factor.entries.zero_()
            diagonal = self.damping_factor.rows == self.damping_factor.columns
            self.damping_factor.entries[diagonal] = math.sqrt(damping_init / stiffness_init)
            if self.raw_mean_stiffness is not None:
                raw = stiffness_init + math.log(-math.expm1(-stiffness_init))
                self.raw_mean_stiffness.fill_(raw)


class ConstantDiagonalDiffusion(nn.Module):
    """Positive per-step standard deviations in standardized coordinates."""

    def __init__(self, dimension: int, initial_value: float):
        super().__init__()
        raw = initial_value + math.log(-math.expm1(-initial_value))
        self.raw_diagonal = nn.Parameter(torch.full((dimension,), raw))

    def forward(self) -> Tensor:
        return F.softplus(self.raw_diagonal) + 1e-6


class ConstantFullDiffusion(nn.Module):
    """Lower-triangular factor of a full covariance in standardized coordinates."""

    def __init__(self, dimension: int, initial_value: float):
        super().__init__()
        raw = initial_value + math.log(-math.expm1(-initial_value))
        self.raw_diagonal = nn.Parameter(torch.full((dimension,), raw))
        self.lower_entries = nn.Parameter(torch.zeros(dimension * (dimension - 1) // 2))
        indices = torch.tril_indices(dimension, dimension, offset=-1)
        self.register_buffer("lower_rows", indices[0], persistent=False)
        self.register_buffer("lower_columns", indices[1], persistent=False)

    def forward(self) -> Tensor:
        factor = torch.diag(F.softplus(self.raw_diagonal) + 1e-6)
        factor[self.lower_rows, self.lower_columns] = self.lower_entries
        return factor


class StateDiagonalDiffusion(MLPDrift):
    """Positive diagonal factor conditioned on the drift's normalized inputs."""

    def __init__(self, dimension: int, hidden_layers: tuple[int, ...], activation: str,
                 initial_value: float, history_steps: int = 0,
                 input_blocks: int | None = None):
        super().__init__(dimension, hidden_layers, activation, history_steps, input_blocks)
        raw = initial_value + math.log(-math.expm1(-initial_value))
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, raw)

    def forward(self, normalized_input: Tensor) -> Tensor:
        return F.softplus(super().forward(normalized_input)) + 1e-6


class BoundedStateDiagonalDiffusion(MLPDrift):
    """Diagonal factor with smoothly bounded state-dependent modulation."""

    def __init__(self, dimension: int, hidden_layers: tuple[int, ...], activation: str,
                 initial_value: float, log_range: float, history_steps: int = 0,
                 input_blocks: int | None = None):
        super().__init__(dimension, hidden_layers, activation, history_steps, input_blocks)
        raw = initial_value + math.log(-math.expm1(-initial_value))
        self.raw_baseline = nn.Parameter(torch.full((dimension,), raw))
        self.log_range = log_range
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def baseline(self) -> Tensor:
        return F.softplus(self.raw_baseline) + 1e-6

    def forward(self, normalized_input: Tensor) -> Tensor:
        log_multiplier = self.log_range * torch.tanh(super().forward(normalized_input))
        return self.baseline() * torch.exp(log_multiplier)


class NeuralSDE(nn.Module):
    """Evolve the state, a legacy lagged increment, or per-frame velocity.

    The MLP and learned diffusion use standardized coordinates and one-step
    units for conditioning. drift() and diffusion_matrix() convert these to
    physical units. State and legacy-increment modes use seconds; velocity mode
    uses one native frame interval as its unit of time.
    """

    def __init__(self, config: SDEConfig):
        super().__init__()
        self.config = config
        input_blocks = len(config.effective_history_offsets) + 1
        if config.drift_type == "damped_residual":
            self.drift_model = DampedResidualDrift(
                config.reduced_dim, config.hidden_layers, config.activation,
                config.stiffness_init, config.damping_init, config.damped_operator,
                input_blocks)
        elif config.drift_type == "capillary_energy":
            self.drift_model = CapillaryEnergyDrift(
                config.reduced_dim, config.hidden_layers, config.activation,
                config.stiffness_init, config.damping_init, config.capillary_mean_mode,
                input_blocks)
        else:
            self.drift_model = MLPDrift(config.reduced_dim, config.hidden_layers,
                                        config.activation, config.history_steps, input_blocks)
        if config.diffusion_type == "state_diagonal":
            self.diffusion_model = StateDiagonalDiffusion(
                config.reduced_dim, config.hidden_layers, config.activation,
                config.diffusion_init, config.history_steps, input_blocks)
        elif config.diffusion_type == "bounded_state_diagonal":
            self.diffusion_model = BoundedStateDiagonalDiffusion(
                config.reduced_dim, config.hidden_layers, config.activation,
                config.diffusion_init, config.diffusion_log_range,
                config.history_steps, input_blocks)
        else:
            diffusion = {"diagonal": ConstantDiagonalDiffusion,
                         "full": ConstantFullDiffusion}[config.diffusion_type]
            self.diffusion_model = diffusion(config.reduced_dim, config.diffusion_init)
        for name, value in (("state_mean", 0.), ("state_std", 1.),
                            ("target_mean", 0.), ("target_std", 1.)):
            self.register_buffer(name, torch.full((config.reduced_dim,), value))

    def set_normalization(self, stats) -> None:
        for name in ("state_mean", "state_std", "target_mean", "target_std"):
            value = getattr(stats, name)
            if value.shape != (self.config.reduced_dim,) or not torch.isfinite(value).all():
                raise ValueError(f"invalid {name} shape or values")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"{name} must be positive")
            getattr(self, name).copy_(value)

    def _normalized_state(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.config.reduced_dim:
            raise ValueError("last state dimension must equal reduced_dim")
        return (state - self.state_mean) / self.state_std

    def _frame_velocity(self, state: Tensor, history_states: Tensor | None) -> Tensor:
        """Causal coordinate change per native frame."""
        expected = (*state.shape[:-1], self.config.history_steps, self.config.reduced_dim)
        if history_states is None or history_states.shape != expected:
            raise ValueError(f"history_states must have shape {expected}")
        return state - history_states[..., 0, :]

    def _drift_input(self, state: Tensor, history_states: Tensor | None = None,
                     velocity: Tensor | None = None) -> Tensor:
        current = self._normalized_state(state)
        if self.config.state_variable == "velocity":
            if velocity is None:
                velocity = self._frame_velocity(state, history_states)
            elif velocity.shape != state.shape:
                raise ValueError("velocity must match the state shape")
            normalized_velocity = (velocity - self.target_mean) / self.target_std
            older_offsets = self.config.effective_history_offsets[1:]
            if not older_offsets:
                return torch.cat((current, normalized_velocity), dim=-1)
            expected = (*state.shape[:-1], self.config.history_steps,
                        self.config.reduced_dim)
            if history_states is None or history_states.shape != expected:
                raise ValueError(f"history_states must have shape {expected}")
            indices = [offset - 1 for offset in older_offsets]
            older = (history_states[..., indices, :] - self.state_mean) / self.state_std
            return torch.cat((current, normalized_velocity,
                              older.flatten(start_dim=-2)), dim=-1)
        if not self.config.history_steps:
            if history_states is not None:
                raise ValueError("History was supplied to a history-free model")
            return current
        expected = (*state.shape[:-1], self.config.history_steps, self.config.reduced_dim)
        if history_states is None or history_states.shape != expected:
            raise ValueError(f"history_states must have shape {expected}")
        if self.config.state_variable == "increment":
            increment = (state - history_states[..., 0, :] - self.target_mean) / self.target_std
            older_offsets = self.config.effective_history_offsets[1:]
            if not older_offsets:
                return torch.cat((current, increment), dim=-1)
            indices = [offset - 1 for offset in older_offsets]
            older = (history_states[..., indices, :] - self.state_mean) / self.state_std
            return torch.cat((current, increment, older.flatten(start_dim=-2)), dim=-1)
        indices = [offset - 1 for offset in self.config.effective_history_offsets]
        past = (history_states[..., indices, :] - self.state_mean) / self.state_std
        return torch.cat((current, past.flatten(start_dim=-2)), dim=-1)

    def _sde_scale(self) -> Tensor:
        return (self.target_std
                if self.config.state_variable in {"increment", "velocity"}
                else self.state_std)

    def drift(self, state: Tensor, history_states: Tensor | None = None,
              velocity: Tensor | None = None) -> Tensor:
        """Physical drift of the configured SDE variable, shape [..., reduced_dim]."""
        time_unit = 1. if self.config.state_variable == "velocity" else self.config.dt
        return self._sde_scale() * self.drift_model(
            self._drift_input(state, history_states, velocity)) / time_unit

    def normalized_diffusion_matrix(self, state: Tensor | None = None,
                                    history_states: Tensor | None = None,
                                    velocity: Tensor | None = None,
                                    normalized_input: Tensor | None = None) -> Tensor:
        """Per-native-step factor G, shaped [d,d] or [...,d,d] when conditioned."""
        if self.config.diffusion_type in {"state_diagonal", "bounded_state_diagonal"}:
            if normalized_input is None:
                if state is None:
                    raise ValueError("State-dependent diffusion requires conditioning state.")
                normalized_input = self._drift_input(state, history_states, velocity)
            return torch.diag_embed(self.diffusion_model(normalized_input))
        factor = self.diffusion_model()
        return torch.diag(factor) if factor.ndim == 1 else factor

    def diffusion_matrix(self, state: Tensor | None = None,
                         history_states: Tensor | None = None,
                         velocity: Tensor | None = None,
                         normalized_input: Tensor | None = None) -> Tensor:
        """Physical factor G, shaped [d,d] or [...,d,d] when conditioned."""
        time_unit = 1. if self.config.state_variable == "velocity" else self.config.dt
        factor = self.normalized_diffusion_matrix(
            state, history_states, velocity, normalized_input)
        return self._sde_scale().unsqueeze(-1) * factor / math.sqrt(time_unit)

    def diffusion_diagonal(self, state: Tensor | None = None,
                           history_states: Tensor | None = None,
                           velocity: Tensor | None = None,
                           normalized_input: Tensor | None = None) -> Tensor:
        """Diagonal of physical G, shaped [d] or [...,d] when conditioned."""
        return torch.diagonal(self.diffusion_matrix(
            state, history_states, velocity, normalized_input), dim1=-2, dim2=-1)

    def velocity_step_distribution(self, state: Tensor, history_states: Tensor,
                                   velocity: Tensor, *, zero_drift: bool = False
                                   ) -> tuple[Tensor, Tensor]:
        """Conditional velocity mean and Cholesky factor for one model step."""
        if self.config.state_variable != "velocity":
            raise ValueError("velocity step requires state_variable velocity")
        drift = (torch.zeros_like(velocity) if zero_drift else
                 self.drift(state, history_states, velocity))
        step_size = self.config.lag_steps
        mean = velocity + step_size * drift
        factor = math.sqrt(step_size) * self.diffusion_matrix(
            state, history_states, velocity)
        return mean, factor

    def augmented_step_distribution(self, state: Tensor, history_states: Tensor,
                                    velocity: Tensor, *, zero_drift: bool = False
                                    ) -> tuple[Tensor, Tensor]:
        """Physical (position, velocity) mean and rectangular noise factor.

        With velocity factor L and h=lag_steps, the augmented factor is
        [h L; L]. Its product with its transpose includes the position--velocity
        covariance; the augmented Gaussian is intentionally rank deficient.
        """
        mean_velocity, factor = self.velocity_step_distribution(
            state, history_states, velocity, zero_drift=zero_drift)
        h = self.config.lag_steps
        return (torch.cat((state + h * mean_velocity, mean_velocity), dim=-1),
                torch.cat((h * factor, factor), dim=-2))

    @torch.no_grad()
    def rollout(self, initial_state: Tensor, *, horizon: int, num_trajectories: int,
                history_states=None, num_steps=None, seed: int | None = None,
                diffusion_scale: float = 1.) -> Tensor:
        """Return physical states [condition, ensemble, horizon+1, reduced_dim]."""
        if initial_state.ndim != 2 or initial_state.shape[-1] != self.config.reduced_dim:
            raise ValueError("initial_state must have shape [condition, reduced_dim]")
        if min(horizon, num_trajectories) < 1 or num_steps is not None:
            raise ValueError("positive horizon/ensemble required; sampling steps are unsupported")
        if not math.isfinite(diffusion_scale) or diffusion_scale < 0:
            raise ValueError("diffusion_scale must be finite and nonnegative")
        if self.config.history_steps:
            expected = (len(initial_state), self.config.history_steps, self.config.reduced_dim)
            if history_states is None or history_states.shape != expected:
                raise ValueError(f"history_states must have shape {expected}")
            history = history_states[:, None].expand(-1, num_trajectories, -1, -1).clone()
        else:
            if history_states is not None:
                raise ValueError("History was supplied to a history-free model")
            history = None
        generator = None
        if seed is not None:
            generator = torch.Generator(device=initial_state.device).manual_seed(seed)
        state = initial_state[:, None, :].expand(-1, num_trajectories, -1).clone()
        velocity = self._frame_velocity(state, history) if self.config.state_variable == "velocity" else None
        states = [state]
        step_size = self.config.lag_steps * (
            1. if self.config.state_variable == "velocity" else self.config.dt)
        for _ in range(horizon):
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            if self.config.state_variable == "velocity":
                mean_velocity, noise_factor = self.velocity_step_distribution(
                    state, history, velocity)
                noise_factor = diffusion_scale * noise_factor
            else:
                drift = self.drift(state, history)
                diffusion = self.diffusion_matrix(state, history)
                noise_factor = diffusion_scale * math.sqrt(step_size) * diffusion
            stochastic = (torch.matmul(noise, noise_factor.T) if noise_factor.ndim == 2
                          else torch.matmul(noise_factor, noise.unsqueeze(-1)).squeeze(-1))
            if self.config.state_variable == "increment":
                change = step_size * drift + stochastic
                increment = state - history[..., 0, :]
                following = state + increment + change
            elif self.config.state_variable == "velocity":
                velocity = mean_velocity + stochastic
                following = state + step_size * velocity
            else:
                change = step_size * drift + stochastic
                following = state + change
            if history is not None:
                history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
            state = following
            states.append(state)
        return torch.stack(states, dim=2)
