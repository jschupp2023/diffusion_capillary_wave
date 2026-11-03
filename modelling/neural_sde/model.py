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


class CausalTemporalBlock(nn.Module):
    """One left-padded dilated convolution with a channel-wise skip connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, activation: str):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.activation = {"silu": nn.SiLU, "gelu": nn.GELU,
                           "relu": nn.ReLU, "tanh": nn.Tanh}[activation]()

    def forward(self, sequence: Tensor) -> Tensor:
        return sequence + self.activation(
            self.convolution(F.pad(sequence, (self.left_padding, 0))))


class TCNDrift(nn.Module):
    """Causal temporal encoder; last token predicts normalized drift.

    Tokens are position/velocity (2*d) by default, or core innovations (d).
    Input is oldest first. No temporal pooling or normalization mixes future
    tokens into earlier outputs.
    """

    def __init__(self, dimension: int, channels: int, kernel_size: int,
                 dilations: tuple[int, ...], activation: str,
                 input_dimension: int | None = None):
        super().__init__()
        self.dimension = dimension
        self.input_dimension = input_dimension or 2 * dimension
        self.network = nn.Sequential(
            nn.Conv1d(self.input_dimension, channels, 1),
            *(CausalTemporalBlock(channels, kernel_size, dilation, activation)
              for dilation in dilations),
            nn.Conv1d(channels, dimension, 1))

    def sequence_output(self, sequence: Tensor) -> Tensor:
        if sequence.ndim < 2 or sequence.shape[-1] != self.input_dimension:
            raise ValueError(f"TCN input must have shape [..., time, {self.input_dimension}]")
        length = sequence.shape[-2]
        features = sequence.reshape(-1, length, self.input_dimension).transpose(1, 2)
        output = self.network(features).transpose(1, 2)
        return output.reshape(*sequence.shape[:-2], length, self.dimension)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.sequence_output(sequence)[..., -1, :]

    def cached_forward(self, sequence: Tensor, buffers=None):
        """Return the last output and differentiable per-layer input buffers.

        Warm up with a full receptive field. Subsequent calls supply TWO
        tokens: the previous position with its reconstructed velocity, then
        the current position with its explicit sampled velocity. Retain the
        prefix excluding the last token. This preserves the ordinary sliding
        window's velocity rounding and autograd semantics, not just its
        real-arithmetic recurrence. No buffers are detached or mutated.
        """
        leading = sequence.shape[:-2]
        x = F.linear(sequence, self.network[0].weight[..., 0], self.network[0].bias)
        updated = []
        for index, block in enumerate(self.network[1:-1]):
            padding = block.left_padding
            if buffers is None:
                prefix = x.new_zeros((*leading, padding, x.shape[-1]))
            else:
                prefix = buffers[index]
            combined = torch.cat((prefix, x), dim=-2)
            # Each window is [channels, kernel taps], matching Conv1d's
            # weight layout. Linear kernels avoid tiny sequential Conv1d
            # launches and support arbitrary condition/particle dimensions.
            taps = combined.unfold(-2, padding + 1, 1)[..., ::block.convolution.dilation[0]]
            y = F.linear(taps.flatten(-2), block.convolution.weight.flatten(1),
                         block.convolution.bias)
            updated.append(combined[..., -(padding + 1):-1, :] if padding else
                           combined[..., :0, :])
            x = x + block.activation(y)
        output = F.linear(x[..., -1, :], self.network[-1].weight[..., 0],
                          self.network[-1].bias)
        return output, tuple(updated)

    def zero_output(self) -> None:
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)


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
                 input_blocks: int = 2, include_damping: bool = True,
                 residual: nn.Module | None = None):
        super().__init__()
        self.dimension = dimension
        self.operator = operator
        self.include_damping = include_damping
        if operator == "spsd":
            self.stiffness_factor = SPSDOperator(dimension, stiffness_init)
            self.damping_factor = (SPSDOperator(dimension, damping_init)
                                   if include_damping else None)
        else:
            self.raw_stiffness = nn.Parameter(torch.full(
                (dimension,), stiffness_init + math.log(-math.expm1(-stiffness_init))))
            if include_damping:
                self.raw_damping = nn.Parameter(torch.full(
                    (dimension,), damping_init + math.log(-math.expm1(-damping_init))))
            else:
                self.register_parameter("raw_damping", None)
        self.residual = (MLPDrift(
            dimension, hidden_layers, activation, input_blocks=input_blocks)
            if residual is None else residual)
        self.residual.zero_output()

    @property
    def network(self):
        """Expose the selected residual network for architecture diagnostics."""
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
        if not self.include_damping:
            return torch.zeros_like(self.stiffness_matrix())
        if self.operator == "spsd":
            return self.damping_factor()
        return torch.diag(F.softplus(self.raw_damping))

    def forward(self, normalized_input: Tensor, sequence_input: Tensor | None = None,
                residual_output: Tensor | None = None) -> Tensor:
        if normalized_input.shape[-1] < 2 * self.dimension:
            raise ValueError("damped_residual drift requires position and velocity blocks")
        position = normalized_input[..., :self.dimension]
        velocity = normalized_input[..., self.dimension:2 * self.dimension]
        if self.operator == "spsd":
            linear = -position @ self.stiffness_matrix()
            if self.include_damping:
                linear = linear - velocity @ self.damping_matrix()
        else:
            linear = -F.softplus(self.raw_stiffness) * position
            if self.include_damping:
                linear = linear - F.softplus(self.raw_damping) * velocity
        return linear + (self.residual(
            normalized_input if sequence_input is None else sequence_input)
            if residual_output is None else residual_output)

    def zero_output(self) -> None:
        # Diffusion-only Stage 1 checkpoints must retain their legacy zero-drift
        # behavior; diagonal softplus coefficients are made numerically negligible.
        with torch.no_grad():
            if self.operator == "spsd":
                self.stiffness_factor.entries.zero_()
                if self.include_damping:
                    self.damping_factor.entries.zero_()
            else:
                self.raw_stiffness.fill_(-30.)
                if self.include_damping:
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
        temporal = (TCNDrift(config.reduced_dim, config.tcn_channels,
                             config.tcn_kernel_size, config.tcn_dilations, config.activation,
                             input_dimension=(config.reduced_dim
                                 if config.residual_input == "core_innovation" else None))
                    if config.residual_architecture == "tcn" else None)
        if config.drift_type == "damped_residual":
            self.drift_model = DampedResidualDrift(
                config.reduced_dim, config.hidden_layers, config.activation,
                config.stiffness_init, config.damping_init, config.damped_operator,
                input_blocks, include_damping=config.structured_damping, residual=temporal)
        elif config.drift_type == "imported_oscillator_residual":
            from .imported_oscillator import ImportedOscillatorResidual
            self.drift_model = ImportedOscillatorResidual(
                config.reduced_dim, config.hidden_layers, config.activation,
                input_blocks, config.dt, residual=temporal,
                forcing_filter=config.forcing_filter)
        elif config.drift_type == "capillary_energy":
            self.drift_model = CapillaryEnergyDrift(
                config.reduced_dim, config.hidden_layers, config.activation,
                config.stiffness_init, config.damping_init, config.capillary_mean_mode,
                input_blocks)
        else:
            self.drift_model = (MLPDrift(config.reduced_dim, config.hidden_layers,
                                        config.activation, config.history_steps, input_blocks)
                                if temporal is None else temporal)
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
        if config.velocity_memory_offsets is None:
            self.register_parameter("velocity_memory", None)
            self.register_parameter("velocity_memory_left", None)
            self.register_parameter("velocity_memory_right", None)
            self.register_parameter("velocity_memory_coefficients", None)
        elif config.velocity_memory_rank is None:
            # Each row is a diagonal operator applied to one standardized past
            # velocity.  Zero initialization makes an expanded warm start
            # exactly reproduce its memory-free source model.
            self.velocity_memory = nn.Parameter(torch.zeros(
                len(config.velocity_memory_offsets), config.reduced_dim))
            self.register_parameter("velocity_memory_left", None)
            self.register_parameter("velocity_memory_right", None)
            self.register_parameter("velocity_memory_coefficients", None)
        else:
            self.register_parameter("velocity_memory", None)
            rank = config.velocity_memory_rank
            # Deterministic, nonzero factors let the zero coefficients learn on
            # the first update while preserving the parent exactly at startup.
            grid = torch.arange(config.reduced_dim, dtype=torch.get_default_dtype())[:, None]
            modes = torch.arange(rank, dtype=torch.get_default_dtype())[None, :]
            basis = torch.cos(math.pi * (grid + .5) * modes / config.reduced_dim)
            basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True)
            self.velocity_memory_left = nn.Parameter(basis.clone())
            self.velocity_memory_right = nn.Parameter(basis.clone())
            self.velocity_memory_coefficients = nn.Parameter(torch.zeros(
                len(config.velocity_memory_offsets), rank))
        if config.velocity_oscillator_periods is None:
            self.register_parameter("velocity_oscillator_input", None)
            self.register_parameter("velocity_oscillator_output", None)
            self.register_parameter("velocity_oscillator_coefficients", None)
            self.register_parameter("velocity_oscillator_raw_radius", None)
            self.register_parameter("velocity_oscillator_raw_phase", None)
        else:
            latent = 2 * len(config.velocity_oscillator_periods)
            grid = torch.arange(config.reduced_dim, dtype=torch.get_default_dtype())[:, None]
            modes = torch.arange(latent, dtype=torch.get_default_dtype())[None, :]
            basis = torch.cos(math.pi * (grid + .5) * modes / config.reduced_dim)
            basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True)
            self.velocity_oscillator_input = nn.Parameter(basis.clone())
            self.velocity_oscillator_output = nn.Parameter(basis.clone())
            # Zero output coefficients preserve the warm-start parent exactly,
            # while the nonzero deterministic factors receive gradients after
            # the first coefficient update.
            self.velocity_oscillator_coefficients = nn.Parameter(torch.zeros(latent))
            maximum_radius = .9999
            radius = config.velocity_oscillator_radius_init
            raw_radius = math.log(radius / (maximum_radius - radius))
            self.velocity_oscillator_raw_radius = nn.Parameter(torch.full(
                (len(config.velocity_oscillator_periods),), raw_radius))
            phase_fraction = torch.tensor(
                [2. / period for period in config.velocity_oscillator_periods],
                dtype=torch.get_default_dtype())
            self.velocity_oscillator_raw_phase = nn.Parameter(torch.logit(phase_fraction))
        if config.velocity_innovation_offsets is None:
            self.register_parameter("velocity_innovation_left", None)
            self.register_parameter("velocity_innovation_right", None)
            self.register_parameter("velocity_innovation_coefficients", None)
        else:
            rank = config.velocity_innovation_rank
            grid = torch.arange(config.reduced_dim, dtype=torch.get_default_dtype())[:, None]
            modes = torch.arange(rank, dtype=torch.get_default_dtype())[None, :]
            basis = torch.cos(math.pi * (grid + .5) * modes / config.reduced_dim)
            basis = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True)
            self.velocity_innovation_left = nn.Parameter(basis.clone())
            self.velocity_innovation_right = nn.Parameter(basis.clone())
            # Exact parent equivalence at initialization.
            self.velocity_innovation_coefficients = nn.Parameter(torch.zeros(
                len(config.velocity_innovation_offsets), rank))
        for name, value in (("state_mean", 0.), ("state_std", 1.),
                            ("target_mean", 0.), ("target_std", 1.)):
            self.register_buffer(name, torch.full((config.reduced_dim,), value))
        if config.forcing_filter != "none":
            self.register_buffer("forcing_mean", torch.zeros(config.reduced_dim))
            self.register_buffer("forcing_std", torch.ones(config.reduced_dim))
        self.fixed_memory = None
        if config.fixed_memory_frequencies is not None:
            from .fixed_memory import FixedOscillatorMemory
            self.fixed_memory = FixedOscillatorMemory(config)

    def set_normalization(self, stats) -> None:
        for name in ("state_mean", "state_std", "target_mean", "target_std"):
            value = getattr(stats, name)
            if value.shape != (self.config.reduced_dim,) or not torch.isfinite(value).all():
                raise ValueError(f"invalid {name} shape or values")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"{name} must be positive")
            getattr(self, name).copy_(value)
        if self.config.drift_type == "imported_oscillator_residual":
            self.drift_model.refresh_normalization(self)

    def _normalized_state(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.config.reduced_dim:
            raise ValueError("last state dimension must equal reduced_dim")
        return (state - self.state_mean) / self.state_std

    def _frame_velocity(self, state: Tensor, history_states: Tensor | None) -> Tensor:
        """Causal coordinate change per native frame."""
        if (history_states is None or history_states.shape[:-2] != state.shape[:-1]
                or history_states.shape[-1] != self.config.reduced_dim
                or history_states.shape[-2] < 1):
            raise ValueError(
                "history_states must match the state batch shape and contain a previous state")
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
            required = max(older_offsets)
            if (history_states is None or history_states.shape[:-2] != state.shape[:-1]
                    or history_states.shape[-1] != self.config.reduced_dim
                    or history_states.shape[-2] < required):
                raise ValueError(
                    f"history_states must match the state batch shape and contain "
                    f"at least {required} preceding states")
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

    def _tcn_input(self, state: Tensor, history_states: Tensor | None,
                   velocity: Tensor | None = None) -> Tensor:
        """Build chronological state or core-innovation tokens."""
        if self.config.residual_input == "core_innovation":
            return self._core_innovation_input(
                state, history_states, velocity, self.config.tcn_context_frames)
        frames = self.config.tcn_context_frames
        if (history_states is None or history_states.ndim != state.ndim + 1
                or history_states.shape[:-2] != state.shape[:-1]
                or history_states.shape[-1] != self.config.reduced_dim
                or history_states.shape[-2] < frames):
            raise ValueError(f"TCN requires at least {frames} preceding states per example")
        positions = torch.cat((history_states[..., :frames, :].flip(-2),
                               state.unsqueeze(-2)), dim=-2)
        velocities = positions[..., 1:, :] - positions[..., :-1, :]
        if velocity is not None:
            if velocity.shape != state.shape:
                raise ValueError("velocity must match the state shape")
            velocities = torch.cat((velocities[..., :-1, :], velocity.unsqueeze(-2)), dim=-2)
        return torch.cat(((positions[..., 1:, :] - self.state_mean) / self.state_std,
                          (velocities - self.target_mean) / self.target_std), dim=-1)

    def _core_innovation_input(self, state, history, velocity, frames):
        """Causal e[n] = u[n]-u[n-1]-F_core(x[n-1],u[n-1]), oldest first.

        Scale by std(u), just like the residual output; do not subtract mean(u)
        from an acceleration increment. The newest explicit velocity preserves
        the usual sampled-velocity semantics during differentiable rollout.
        """
        if (history is None or history.shape[:-2] != state.shape[:-1]
                or history.shape[-1] != self.config.reduced_dim
                or history.shape[-2] < frames + 1):
            raise ValueError(f"Core innovation TCN requires at least {frames + 1} preceding states")
        positions = torch.cat((history[..., :frames + 1, :].flip(-2),
                               state.unsqueeze(-2)), dim=-2)
        velocities = positions[..., 1:, :] - positions[..., :-1, :]
        if velocity is not None:
            if velocity.shape != state.shape:
                raise ValueError("velocity must match the state shape")
            velocities = torch.cat((velocities[..., :-1, :], velocity.unsqueeze(-2)), dim=-2)
        predictors = torch.cat((
            (positions[..., 1:-1, :] - self.state_mean) / self.state_std,
            (velocities[..., :-1, :] - self.target_mean) / self.target_std), dim=-1)
        fixed = self.drift_model.linear(predictors)
        return (velocities[..., 1:, :] - velocities[..., :-1, :]) / self.target_std - fixed

    @property
    def has_forcing_filter(self) -> bool:
        return self.config.forcing_filter == "modal_biquad"

    def forcing_filter_state(self, state: Tensor, history_states: Tensor,
                             velocity: Tensor | None = None) -> Tensor:
        """Reconstruct newest-first raw forcing from an observed position history.

        The finite observed context starts the stable inverse filter from zero.
        Training supplies full-repetition precomputed states; this initializer is
        used for arbitrary measured rollout windows.
        """
        if not self.has_forcing_filter:
            raise RuntimeError("forcing filter is disabled")
        frames = self.config.tcn_context_frames
        actual = self._core_innovation_input(
            state, history_states, velocity, frames) * self.target_std
        raw_values = []
        previous = actual.new_zeros(actual.shape[:-2] + (2, self.config.reduced_dim))
        filtered_previous = actual.new_zeros(
            actual.shape[:-2] + (2, self.config.reduced_dim))
        for index in range(frames):
            raw = self.drift_model.inverse_filter_forcing(
                actual[..., index, :], previous, filtered_previous)
            raw_values.append(raw)
            previous = torch.stack((raw, previous[..., 0, :]), dim=-2)
            filtered_previous = torch.stack(
                (actual[..., index, :], filtered_previous[..., 0, :]), dim=-2)
        return (torch.stack(raw_values[::-1], dim=-2),
                actual[..., -2:, :].flip(-2))

    def raw_forcing_mean(self, forcing_state: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        """Physical conditional mean of inverse-filtered broadband forcing."""
        if isinstance(forcing_state, tuple):
            forcing_state = forcing_state[0]
        frames = self.config.tcn_context_frames
        expected = (*forcing_state.shape[:-2], frames, self.config.reduced_dim)
        if forcing_state.shape != expected:
            raise ValueError(f"forcing_state must have shape {expected}")
        sequence = ((forcing_state.flip(-2) - self.forcing_mean) / self.forcing_std)
        normalized = self.drift_model.residual(sequence)
        return self.forcing_std * normalized

    def raw_forcing_distribution(self, forcing_state) -> tuple[Tensor, Tensor]:
        """Gaussian distribution learned before the fixed modal biquad filter."""
        factor = self.forcing_std.unsqueeze(-1) * self.normalized_diffusion_matrix()
        return self.raw_forcing_mean(forcing_state), factor

    @torch.no_grad()
    def set_forcing_normalization(self, mean: Tensor, std: Tensor) -> None:
        if not self.has_forcing_filter:
            raise RuntimeError("forcing normalization requires the forcing filter")
        for name, value in (("forcing_mean", mean), ("forcing_std", std)):
            value = torch.as_tensor(value, dtype=getattr(self, name).dtype,
                                    device=getattr(self, name).device)
            if (value.shape != (self.config.reduced_dim,)
                    or not torch.isfinite(value).all()
                    or (name == "forcing_std" and torch.any(value <= 0))):
                raise ValueError(f"invalid {name}")
            getattr(self, name).copy_(value)

    def filtered_forcing_distribution(self, forcing_state) -> tuple[Tensor, Tensor]:
        """Mean and Cholesky factor after the fixed modal biquad filter."""
        if not isinstance(forcing_state, tuple):
            raise ValueError("biquad forcing state requires raw and filtered histories")
        raw_state, filtered_history = forcing_state
        raw_mean, raw_factor = self.raw_forcing_distribution(raw_state)
        mean = self.drift_model.filter_forcing(
            raw_mean, raw_state[..., :2, :], filtered_history)
        direct = self.drift_model.filter_direct_matrix(raw_mean)
        factor = direct @ raw_factor
        covariance = factor @ factor.T
        cholesky = torch.linalg.cholesky(covariance)
        return mean, cholesky

    def advance_forcing_filter(self, forcing_state,
                               filtered_forcing: Tensor) -> Tensor:
        raw_state, filtered_history = forcing_state
        raw = self.drift_model.inverse_filter_forcing(
            filtered_forcing, raw_state[..., :2, :], filtered_history)
        return (torch.cat((raw.unsqueeze(-2), raw_state[..., :-1, :]), dim=-2),
                torch.stack((filtered_forcing, filtered_history[..., 0, :]), dim=-2))

    def forcing_filter_logabsdet(self) -> Tensor:
        """Log determinant of the instantaneous raw-to-filtered forcing map."""
        if not self.has_forcing_filter or not self.drift_model.filter_ready:
            raise RuntimeError("forcing filter is not configured")
        return torch.log(self.drift_model.forcing_h0.to(
            dtype=self.state_mean.dtype, device=self.state_mean.device)).sum()

    def _sde_scale(self) -> Tensor:
        return (self.target_std
                if self.config.state_variable in {"increment", "velocity"}
                else self.state_std)

    def nonlinear_drift(self, state: Tensor, history_states: Tensor | None = None,
                        velocity: Tensor | None = None,
                        normalized_input: Tensor | None = None) -> Tensor:
        """Normalized neural term alone, for drift-component regularizers."""
        network = (self.drift_model if self.config.drift_type == "mlp"
                   else self.drift_model.residual)
        if self.config.residual_architecture == "tcn":
            features = self._tcn_input(state, history_states, velocity)
        else:
            features = (self._drift_input(state, history_states, velocity)
                        if normalized_input is None else normalized_input)
        return network(features)

    def velocity_memory_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in (
            self.velocity_memory, self.velocity_memory_left,
            self.velocity_memory_right, self.velocity_memory_coefficients)
            if parameter is not None)

    def velocity_oscillator_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in (
            self.velocity_oscillator_input, self.velocity_oscillator_output,
            self.velocity_oscillator_coefficients,
            self.velocity_oscillator_raw_radius,
            self.velocity_oscillator_raw_phase)
            if parameter is not None)

    def memory_parameters(self) -> tuple[nn.Parameter, ...]:
        innovation = tuple(parameter for parameter in (
            self.velocity_innovation_left, self.velocity_innovation_right,
            self.velocity_innovation_coefficients) if parameter is not None)
        return (self.velocity_memory_parameters() + self.velocity_oscillator_parameters()
                + innovation)

    @staticmethod
    def _bounded_memory_factor(parameter: Tensor) -> Tensor:
        # ||A||_2 <= ||A||_F, so both factors have induced norm at most one.
        return parameter / torch.linalg.vector_norm(parameter).clamp_min(1.)

    def velocity_memory_kernel_norms(self) -> list[float]:
        if self.config.velocity_memory_offsets is None:
            return []
        if self.config.velocity_memory_rank is None:
            return [float(row.detach().abs().max().cpu()) for row in self.velocity_memory]
        left = self._bounded_memory_factor(self.velocity_memory_left)
        right = self._bounded_memory_factor(self.velocity_memory_right)
        coefficients = (self.config.velocity_memory_norm_cap
                        * torch.tanh(self.velocity_memory_coefficients))
        kernels = torch.einsum("ir,kr,jr->kij", left, coefficients, right)
        return [float(torch.linalg.matrix_norm(kernel.detach(), ord=2).cpu())
                for kernel in kernels]

    def validate_velocity_memory(self) -> None:
        if any(not bool(torch.isfinite(parameter).all())
               for parameter in self.memory_parameters()):
            raise FloatingPointError("nonfinite velocity-memory parameter")

    def velocity_innovation_coefficients_bounded(self) -> Tensor:
        if self.velocity_innovation_coefficients is None:
            return self.state_mean.new_empty((0, 0))
        raw = torch.tanh(self.velocity_innovation_coefficients)
        # Together with unit-bounded factors, this caps the sum of kernel
        # induced norms by norm_cap, a conservative contraction condition.
        scale = raw.abs().sum().clamp_min(1.)
        return self.config.velocity_innovation_norm_cap * raw / scale

    def velocity_innovation_kernel_norms(self) -> list[float]:
        if self.config.velocity_innovation_offsets is None:
            return []
        left = self._bounded_memory_factor(self.velocity_innovation_left)
        right = self._bounded_memory_factor(self.velocity_innovation_right)
        kernels = torch.einsum(
            "ir,kr,jr->kij", left,
            self.velocity_innovation_coefficients_bounded(), right)
        return [float(torch.linalg.matrix_norm(kernel.detach(), ord=2).cpu())
                for kernel in kernels]

    def velocity_innovation_correction(self, innovation_state: Tensor) -> Tensor:
        """Predict the next standardized residual from recent residuals."""
        if self.config.velocity_innovation_offsets is None:
            return innovation_state.new_zeros(innovation_state.shape[:-2]
                                              + (self.config.reduced_dim,))
        maximum = self.config.velocity_innovation_offsets[-1]
        expected = (*innovation_state.shape[:-2], maximum, self.config.reduced_dim)
        if innovation_state.shape != expected:
            raise ValueError(f"innovation_state must have shape {expected}")
        indices = [offset - 1 for offset in self.config.velocity_innovation_offsets]
        selected = innovation_state[..., indices, :]
        left = self._bounded_memory_factor(self.velocity_innovation_left)
        right = self._bounded_memory_factor(self.velocity_innovation_right)
        projected = torch.einsum("...kd,dr->...kr", selected, right)
        return torch.einsum(
            "...kr,kr,dr->...d", projected,
            self.velocity_innovation_coefficients_bounded(), left)

    def velocity_base_step_distribution(
            self, state: Tensor, history_states: Tensor, velocity: Tensor,
            *, zero_drift: bool = False,
            oscillator_state: Tensor | None = None,
            tcn_cache: dict | None = None,
            fixed_memory_state: Tensor | None = None,
            forcing_filter_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Parent conditional distribution, excluding innovation AR memory."""
        if self.has_forcing_filter:
            if tcn_cache is not None:
                raise ValueError("modal forcing filter does not use the TCN rollout cache")
            if forcing_filter_state is None:
                forcing_filter_state = self.forcing_filter_state(
                    state, history_states, velocity)
            forcing_mean, factor = self.filtered_forcing_distribution(
                forcing_filter_state)
            core = self.imported_oscillator_drift(state, history_states, velocity)
            return velocity + core + (0. if zero_drift else forcing_mean), factor
        if zero_drift:
            drift = torch.zeros_like(velocity)
        elif tcn_cache is not None:
            drift = self.cached_tcn_drift(state, history_states, velocity, tcn_cache)
            if self.fixed_memory is not None:
                if fixed_memory_state is None:
                    fixed_memory_state, _ = self.fixed_memory.initialize(
                        state, history_states, self.target_mean, self.target_std)
                drift = drift + self.target_std * self.fixed_memory(fixed_memory_state)
        elif self.fixed_memory is not None:
            drift = self.drift(state, history_states, velocity,
                               fixed_memory_state=fixed_memory_state)
        elif oscillator_state is None:
            drift = self.drift(state, history_states, velocity)
        else:
            drift = self.drift(state, history_states, velocity, oscillator_state)
        step_size = self.config.lag_steps
        mean = velocity + step_size * drift
        if tcn_cache is not None and self.config.diffusion_type in {"diagonal", "full"}:
            # Constant diffusion is the same differentiable tensor at every
            # step; do not repeatedly rebuild its factor inside the rollout.
            if "factor" not in tcn_cache:
                tcn_cache["factor"] = math.sqrt(step_size) * self.diffusion_matrix()
            factor = tcn_cache["factor"]
        else:
            factor = math.sqrt(step_size) * self.diffusion_matrix(
                state, history_states, velocity)
        return mean, factor

    def new_tcn_cache(self):
        """A call-local execution cache; never serialized in a checkpoint.

        Short contexts reset internal left padding on every sliding window,
        so streaming would change their function. Keep the reference path for
        those models (and for non-TCN models).
        """
        if (not self.has_forcing_filter
                and getattr(self, "tcn_rollout_cache", False)
                and self.config.residual_architecture == "tcn"
                and self.config.tcn_context_frames == self.config.tcn_receptive_field):
            return {"buffers": None}
        return None

    def cached_tcn_drift(self, state, history, velocity, cache):
        network = (self.drift_model if self.config.drift_type == "mlp"
                   else self.drift_model.residual)
        if cache["buffers"] is None:
            sequence = self._tcn_input(state, history, velocity)
        elif self.config.residual_input == "core_innovation":
            # Rebuild the previous token with its reconstructed velocity, then
            # append the new one, matching the ordinary sliding-window path.
            sequence = self._core_innovation_input(
                state, history, velocity, min(2, self.config.tcn_receptive_field))
        else:
            # The old current token becomes a history token, where velocity
            # is a difference of positions, NOT the saved sampled velocity.
            if self.config.tcn_receptive_field == 1:
                positions, velocities = state.unsqueeze(-2), velocity.unsqueeze(-2)
            else:
                positions = torch.stack((history[..., 0, :], state), dim=-2)
                velocities = torch.stack((history[..., 0, :] - history[..., 1, :],
                                          velocity), dim=-2)
            sequence = torch.cat(((positions - self.state_mean) / self.state_std,
                                  (velocities - self.target_mean) / self.target_std), dim=-1)
        evaluate = network.cached_forward
        # The first call has a long context and ``buffers=None``; all later
        # calls have a two-token input and a tuple of fixed-width buffers.
        # Compiling both signatures through one Dynamo wrapper causes a fresh
        # warm-up graph for changing sampled batch sizes and can exhaust the
        # recompile limit when multistep and moment objectives are combined.
        # Keep the single warm-up call eager and compile only the recurrent
        # kernel that dominates long-rollout runtime.
        if (getattr(self, "tcn_rollout_compile", False)
                and cache["buffers"] is not None):
            if not hasattr(network, "_compiled_cached_recurrent_forward"):
                # Endpoint-moment training deliberately differentiates the
                # same rollout graph once for drift parameters and once for
                # diffusion parameters.  AOTAutograd's donated-buffer
                # optimization forbids that retain_graph=True pattern, so it
                # must be disabled before this lazily compiled callable is
                # created.  PyTorch explicitly recommends this setting in the
                # resulting runtime error.
                try:
                    import torch._functorch.config as functorch_config
                    functorch_config.donated_buffer = False
                except (ImportError, AttributeError):
                    # Older PyTorch versions do not expose this private
                    # optimization; they also do not need the workaround.
                    pass
                # Compiled callable is execution-only: parameters/state_dict
                # retain their original names and no cache enters checkpoints.
                network._compiled_cached_recurrent_forward = torch.compile(
                    network.cached_forward, fullgraph=True, dynamic=True)
            evaluate = network._compiled_cached_recurrent_forward
        normalized, cache["buffers"] = evaluate(sequence, cache["buffers"])
        if self.config.drift_type in {"damped_residual", "imported_oscillator_residual"}:
            normalized = self.drift_model(self._drift_input(state, history, velocity),
                                          residual_output=normalized)
        return self._sde_scale() * normalized

    def velocity_innovation_state(self, state: Tensor, history_states: Tensor) -> Tensor:
        """Reconstruct recent standardized parent residuals from observed states."""
        if self.config.velocity_innovation_offsets is None:
            return state.new_empty((*state.shape[:-1], 0, self.config.reduced_dim))
        maximum = self.config.velocity_innovation_offsets[-1]
        required = maximum + max(self.config.effective_history_offsets)
        if (history_states.shape[:-2] != state.shape[:-1]
                or history_states.shape[-1] != self.config.reduced_dim
                or history_states.shape[-2] < required):
            raise ValueError(
                f"innovation memory requires at least {required} preceding states")
        innovations = []
        for age in range(1, maximum + 1):
            endpoint = state if age == 1 else history_states[..., age - 2, :]
            predictor = history_states[..., age - 1, :]
            predictor_history = history_states[..., age:, :]
            predictor_velocity = predictor - predictor_history[..., 0, :]
            base_mean, _ = self.velocity_base_step_distribution(
                predictor, predictor_history, predictor_velocity)
            observed_velocity = endpoint - predictor
            innovations.append((observed_velocity - base_mean) / self.target_std)
        return torch.stack(innovations, dim=-2)

    def advance_velocity_innovation(self, innovation_state: Tensor,
                                    velocity: Tensor, base_mean: Tensor) -> Tensor:
        innovation = (velocity - base_mean) / self.target_std
        return torch.cat((innovation.unsqueeze(-2), innovation_state[..., :-1, :]), dim=-2)

    def velocity_oscillator_radius(self) -> Tensor:
        if self.velocity_oscillator_raw_radius is None:
            return self.state_mean.new_empty((0,))
        return .9999 * torch.sigmoid(self.velocity_oscillator_raw_radius)

    def velocity_oscillator_phase(self) -> Tensor:
        if self.velocity_oscillator_raw_phase is None:
            return self.state_mean.new_empty((0,))
        return math.pi * torch.sigmoid(self.velocity_oscillator_raw_phase)

    def velocity_oscillator_state(self, state: Tensor, history_states: Tensor) -> Tensor:
        """Reconstruct the stable recurrent state from observed recent velocities."""
        if self.config.velocity_oscillator_periods is None:
            return state.new_empty((*state.shape[:-1], 0))
        burnin = self.config.velocity_oscillator_burnin
        if (history_states.shape[:-2] != state.shape[:-1]
                or history_states.shape[-1] != self.config.reduced_dim
                or history_states.shape[-2] < burnin):
            raise ValueError(
                f"oscillator memory requires at least {burnin} preceding states")
        recent = history_states[..., :burnin, :]
        current_velocity = state - recent[..., 0, :]
        older_velocity = recent[..., :-1, :] - recent[..., 1:, :]
        # Sequence is ordered by age: current velocity first, then v[t-1], ...
        velocities = torch.cat((current_velocity.unsqueeze(-2), older_velocity), dim=-2)
        normalized = (velocities - self.target_mean) / self.target_std
        right = self._bounded_memory_factor(self.velocity_oscillator_input)
        forcing = normalized @ right
        pairs = forcing.unflatten(-1, (-1, 2))

        ages = torch.arange(burnin, device=state.device, dtype=state.dtype)
        radius = self.velocity_oscillator_radius()
        phase = self.velocity_oscillator_phase()
        magnitude = radius.unsqueeze(-1).pow(ages)
        angle = phase.unsqueeze(-1) * ages
        cosine, sine = torch.cos(angle), torch.sin(angle)
        x = pairs[..., 0].movedim(-2, -1)
        y = pairs[..., 1].movedim(-2, -1)
        rotated_x = x * cosine - y * sine
        rotated_y = x * sine + y * cosine
        gain = torch.sqrt(torch.clamp(1. - radius.square(), min=1e-8))
        result = gain[..., None] * torch.stack((
            (magnitude * rotated_x).sum(dim=-1),
            (magnitude * rotated_y).sum(dim=-1)), dim=-1)
        return result.flatten(start_dim=-2)

    def advance_velocity_oscillator(self, oscillator_state: Tensor,
                                    velocity: Tensor) -> Tensor:
        """Advance recurrent memory using a newly observed or generated velocity."""
        if self.config.velocity_oscillator_periods is None:
            return oscillator_state
        latent = 2 * len(self.config.velocity_oscillator_periods)
        expected = (*velocity.shape[:-1], latent)
        if oscillator_state.shape != expected:
            raise ValueError(f"oscillator_state must have shape {expected}")
        normalized = (velocity - self.target_mean) / self.target_std
        right = self._bounded_memory_factor(self.velocity_oscillator_input)
        forcing = (normalized @ right).unflatten(-1, (-1, 2))
        pairs = oscillator_state.unflatten(-1, (-1, 2))
        radius = self.velocity_oscillator_radius()
        phase = self.velocity_oscillator_phase()
        cosine, sine = torch.cos(phase), torch.sin(phase)
        x, y = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((cosine * x - sine * y,
                               sine * x + cosine * y), dim=-1)
        gain = torch.sqrt(torch.clamp(1. - radius.square(), min=1e-8))
        return (radius[..., None] * rotated + gain[..., None] * forcing).flatten(
            start_dim=-2)

    def velocity_oscillator_drift(self, oscillator_state: Tensor) -> Tensor:
        left = self._bounded_memory_factor(self.velocity_oscillator_output)
        coefficients = (self.config.velocity_oscillator_norm_cap
                        * torch.tanh(self.velocity_oscillator_coefficients))
        return (oscillator_state * coefficients) @ left.T

    def imported_oscillator_drift(self, state: Tensor, history_states: Tensor | None = None,
                                  velocity: Tensor | None = None) -> Tensor:
        """Fixed contribution in physical per-frame velocity-change units."""
        if self.config.drift_type != "imported_oscillator_residual":
            raise ValueError("requires imported_oscillator_residual drift")
        return self._sde_scale() * self.drift_model.linear(
            self._drift_input(state, history_states, velocity))

    def drift(self, state: Tensor, history_states: Tensor | None = None,
              velocity: Tensor | None = None,
              oscillator_state: Tensor | None = None,
              fixed_memory_state: Tensor | None = None) -> Tensor:
        """Physical drift of the configured SDE variable, shape [..., reduced_dim]."""
        time_unit = 1. if self.config.state_variable == "velocity" else self.config.dt
        if self.config.residual_architecture == "tcn":
            sequence = self._tcn_input(state, history_states, velocity)
            normalized_drift = (self.drift_model(
                self._drift_input(state, history_states, velocity), sequence)
                if self.config.drift_type in {"damped_residual", "imported_oscillator_residual"} else self.drift_model(sequence))
        else:
            normalized_drift = self.drift_model(
                self._drift_input(state, history_states, velocity))
        if self.config.velocity_memory_offsets is not None:
            expected = (*state.shape[:-1], self.config.history_steps,
                        self.config.reduced_dim)
            if history_states is None or history_states.shape != expected:
                raise ValueError(f"history_states must have shape {expected}")
            offsets = self.config.velocity_memory_offsets
            newer = torch.stack(
                [history_states[..., offset - 1, :] for offset in offsets], dim=-2)
            older = torch.stack(
                [history_states[..., offset, :] for offset in offsets], dim=-2)
            past_velocity = (newer - older - self.target_mean) / self.target_std
            if self.config.velocity_memory_rank is None:
                memory_drift = (past_velocity * self.velocity_memory).sum(dim=-2)
            else:
                left = self._bounded_memory_factor(self.velocity_memory_left)
                right = self._bounded_memory_factor(self.velocity_memory_right)
                coefficients = (self.config.velocity_memory_norm_cap
                                * torch.tanh(self.velocity_memory_coefficients))
                projected = torch.einsum("...kd,dr->...kr", past_velocity, right)
                memory_drift = torch.einsum(
                    "...kr,kr,dr->...d", projected, coefficients, left)
            normalized_drift = normalized_drift + memory_drift
        if self.config.velocity_oscillator_periods is not None:
            if oscillator_state is None:
                oscillator_state = self.velocity_oscillator_state(state, history_states)
            normalized_drift = (normalized_drift
                                + self.velocity_oscillator_drift(oscillator_state))
        if self.fixed_memory is not None:
            if fixed_memory_state is None:
                fixed_memory_state, _ = self.fixed_memory.initialize(
                    state, history_states, self.target_mean, self.target_std)
            normalized_drift = normalized_drift + self.fixed_memory(fixed_memory_state)
        return self._sde_scale() * normalized_drift / time_unit

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
                                   velocity: Tensor, *, zero_drift: bool = False,
                                   oscillator_state: Tensor | None = None,
                                   innovation_state: Tensor | None = None,
                                   forcing_filter_state: Tensor | None = None
                                   ) -> tuple[Tensor, Tensor]:
        """Conditional velocity mean and Cholesky factor for one model step."""
        if self.config.state_variable != "velocity":
            raise ValueError("velocity step requires state_variable velocity")
        mean, factor = self.velocity_base_step_distribution(
            state, history_states, velocity, zero_drift=zero_drift,
            oscillator_state=oscillator_state,
            forcing_filter_state=forcing_filter_state)
        if self.config.velocity_innovation_offsets is not None and not zero_drift:
            if innovation_state is None:
                innovation_state = self.velocity_innovation_state(state, history_states)
            mean = mean + self.target_std * self.velocity_innovation_correction(
                innovation_state)
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
            parent_history = self.config.history_steps
            if self.fixed_memory is not None:
                parent_history = max(max(self.config.effective_history_offsets),
                                     self.config.tcn_context_frames
                                     if self.config.residual_architecture == "tcn" else 1)
            history = history_states[:, None, :parent_history].expand(
                -1, num_trajectories, -1, -1).clone()
        else:
            if history_states is not None:
                raise ValueError("History was supplied to a history-free model")
            history = None
        generator = None
        if seed is not None:
            generator = torch.Generator(device=initial_state.device).manual_seed(seed)
        state = initial_state[:, None, :].expand(-1, num_trajectories, -1).clone()
        memory_state = memory_ring = None
        if self.fixed_memory is not None:
            memory_state, memory_ring = self.fixed_memory.initialize(
                initial_state, history_states, self.target_mean, self.target_std)
            memory_state = memory_state[:, None].expand(-1, num_trajectories, -1, -1).clone()
            memory_ring = memory_ring[:, None].expand(-1, num_trajectories, -1, -1).clone()
            memory_cursor = self.fixed_memory.window - 1
        velocity = self._frame_velocity(state, history) if self.config.state_variable == "velocity" else None
        oscillator_state = (self.velocity_oscillator_state(state, history)
                            if self.config.velocity_oscillator_periods is not None else None)
        innovation_state = (self.velocity_innovation_state(state, history)
                            if self.config.velocity_innovation_offsets is not None else None)
        forcing_filter_state = (self.forcing_filter_state(state, history, velocity)
                                if self.has_forcing_filter else None)
        if oscillator_state is not None:
            history = history[..., :max(self.config.effective_history_offsets), :]
        states = [state]
        tcn_cache = self.new_tcn_cache()
        step_size = self.config.lag_steps * (
            1. if self.config.state_variable == "velocity" else self.config.dt)
        for _ in range(horizon):
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            if self.config.state_variable == "velocity":
                base_mean, noise_factor = self.velocity_base_step_distribution(
                    state, history, velocity, oscillator_state=oscillator_state,
                    forcing_filter_state=forcing_filter_state,
                    **({"fixed_memory_state": memory_state} if memory_state is not None else {}),
                    **({"tcn_cache": tcn_cache} if tcn_cache is not None else {}))
                mean_velocity = base_mean
                if innovation_state is not None:
                    mean_velocity = mean_velocity + self.target_std * (
                        self.velocity_innovation_correction(innovation_state))
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
                previous_velocity = velocity
                velocity = mean_velocity + stochastic
                following = state + step_size * velocity
                if forcing_filter_state is not None:
                    core = self.imported_oscillator_drift(
                        state, history, previous_velocity)
                    forcing_filter_state = self.advance_forcing_filter(
                        forcing_filter_state,
                        velocity - previous_velocity - core)
                if memory_state is not None:
                    forcing = self.fixed_memory.forcing(velocity, self.target_mean, self.target_std)
                    memory_state = self.fixed_memory.advance(
                        memory_state, forcing, memory_ring[..., memory_cursor, :])
                    memory_ring[..., memory_cursor, :] = forcing
                    memory_cursor = (memory_cursor - 1) % self.fixed_memory.window
                if oscillator_state is not None:
                    oscillator_state = self.advance_velocity_oscillator(
                        oscillator_state, velocity)
                if innovation_state is not None:
                    innovation_state = self.advance_velocity_innovation(
                        innovation_state, velocity, base_mean)
            else:
                change = step_size * drift + stochastic
                following = state + change
            if history is not None:
                history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
            state = following
            states.append(state)
        return torch.stack(states, dim=2)
