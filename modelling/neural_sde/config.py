"""Small, serializable configuration for the Neural SDE baseline."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SDEConfig:
    reduced_dim: int
    dt: float
    hidden_layers: tuple[int, ...] = (64, 64)
    activation: str = "silu"
    diffusion_init: float = 0.1
    diffusion_type: str = "diagonal"
    diffusion_log_range: float = 2.0

    # The shared evaluator uses these to select matching reference windows.
    lag_steps: int = 1
    history_steps: int = 0
    history_offsets: tuple[int, ...] | None = None
    state_variable: str = "state"

    # Appended after legacy fields to preserve positional construction.
    drift_type: str = "mlp"
    stiffness_init: float = 0.01
    damping_init: float = 0.01
    damped_operator: str = "diagonal"
    surface_tension: float = 0.0728
    capillary_mean_mode: bool = True

    # Optional diagonal finite-memory correction to the velocity drift.  An
    # offset k uses the previous native-frame velocity v[t-k].  ``history_steps``
    # is extended automatically to retain the two positions needed for the
    # largest requested velocity offset, while ``history_offsets`` continues to
    # describe only the position taps supplied to the residual MLP.
    velocity_memory_offsets: tuple[int, ...] | None = None
    velocity_memory_rank: int | None = None
    velocity_memory_norm_cap: float = 0.25

    # Stable recurrent velocity memory.  Each requested period creates one
    # two-dimensional damped-rotation state.  Training examples reconstruct
    # that state from ``velocity_oscillator_burnin`` observed native velocities;
    # generated rollouts then carry it forward recursively.
    velocity_oscillator_periods: tuple[float, ...] | None = None
    velocity_oscillator_burnin: int = 256
    velocity_oscillator_radius_init: float = 0.98
    velocity_oscillator_norm_cap: float = 0.25

    # Autoregressive memory of the parent model's standardized one-step
    # velocity residuals.  This turns sampled innovations into a causal latent
    # state during rollout while leaving the parent drift and diffusion fixed.
    velocity_innovation_offsets: tuple[int, ...] | None = None
    velocity_innovation_rank: int | None = None
    velocity_innovation_norm_cap: float = 0.95

    # Opt-in restoring-only ablation of damped_residual. Appended to preserve
    # positional construction of all earlier configuration fields.
    structured_damping: bool = True

    # Replace the nonlinear drift network, not an additive memory correction.
    residual_architecture: str = "mlp"
    tcn_context_frames: int = 63
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)

    # Frozen causal resonant filters; only their linear drift readout is fitted.
    fixed_memory_frequencies: tuple[float, ...] | None = None
    fixed_memory_damping: tuple[float, ...] | None = None
    fixed_memory_window: int = 0

    # Input to the imported oscillator's TCN: observed state, or past forcing
    # increments after subtracting the fixed native-step oscillator update.
    residual_input: str = "state"

    # Optional fixed minimum-phase biquad filter around core-innovation forcing.
    # The TCN and diffusion model the inverse-filtered broadband forcing; the
    # filter restores the measured modal notches before forcing the oscillator.
    forcing_filter: str = "none"
    forcing_filter_burnin: int = 32768

    def __post_init__(self):
        position_offsets = None
        if self.history_offsets is not None:
            offsets = tuple(self.history_offsets)
            if (not offsets or any(not isinstance(offset, int) or isinstance(offset, bool)
                                   or offset < 1 for offset in offsets)
                    or tuple(sorted(set(offsets))) != offsets):
                raise ValueError("history_offsets must be strictly increasing positive integers")
            if (self.history_steps not in {0, offsets[-1]}
                    and self.velocity_memory_offsets is None
                    and self.velocity_oscillator_periods is None
                    and self.velocity_innovation_offsets is None
                    and self.fixed_memory_frequencies is None
                    and self.residual_architecture != "tcn"):
                raise ValueError("history_steps must equal the largest explicit history offset")
            object.__setattr__(self, "history_offsets", offsets)
            position_offsets = offsets
        if self.velocity_memory_offsets is not None:
            memory_offsets = tuple(self.velocity_memory_offsets)
            if (not memory_offsets
                    or any(not isinstance(offset, int) or isinstance(offset, bool)
                           or offset < 1 for offset in memory_offsets)
                    or tuple(sorted(set(memory_offsets))) != memory_offsets):
                raise ValueError(
                    "velocity_memory_offsets must be strictly increasing positive integers")
            if self.state_variable != "velocity" or self.lag_steps != 1:
                raise ValueError(
                    "velocity memory requires native-lag state_variable velocity")
            if position_offsets is None:
                if self.history_steps < 1:
                    raise ValueError("velocity memory requires a current-velocity history frame")
                position_offsets = tuple(range(1, self.history_steps + 1))
                object.__setattr__(self, "history_offsets", position_offsets)
            object.__setattr__(self, "velocity_memory_offsets", memory_offsets)
            object.__setattr__(self, "history_steps", max(
                position_offsets[-1], memory_offsets[-1] + 1))
        elif position_offsets is not None:
            object.__setattr__(self, "history_steps", position_offsets[-1])
        if self.velocity_oscillator_periods is not None:
            periods = tuple(float(period) for period in self.velocity_oscillator_periods)
            if (not periods or any(not math.isfinite(period) or period <= 2
                                   for period in periods)
                    or tuple(sorted(set(periods))) != periods):
                raise ValueError(
                    "velocity_oscillator_periods must be strictly increasing finite values > 2")
            if self.velocity_memory_offsets is not None:
                raise ValueError(
                    "velocity oscillator memory and finite velocity memory are mutually exclusive")
            if self.state_variable != "velocity" or self.lag_steps != 1:
                raise ValueError(
                    "velocity oscillator memory requires native-lag state_variable velocity")
            if 2 * len(periods) > self.reduced_dim:
                raise ValueError(
                    "twice the number of oscillator periods cannot exceed reduced_dim")
            if (not isinstance(self.velocity_oscillator_burnin, int)
                    or isinstance(self.velocity_oscillator_burnin, bool)
                    or self.velocity_oscillator_burnin < 1):
                raise ValueError("velocity_oscillator_burnin must be a positive integer")
            if position_offsets is None:
                if self.history_steps < 1:
                    raise ValueError(
                        "velocity oscillator memory requires a current-velocity history frame")
                position_offsets = tuple(range(1, self.history_steps + 1))
                object.__setattr__(self, "history_offsets", position_offsets)
            object.__setattr__(self, "velocity_oscillator_periods", periods)
            object.__setattr__(self, "history_steps", max(
                self.history_steps, self.velocity_oscillator_burnin))
        if self.velocity_innovation_offsets is not None:
            offsets = tuple(self.velocity_innovation_offsets)
            if (not offsets
                    or any(not isinstance(offset, int) or isinstance(offset, bool)
                           or offset < 1 for offset in offsets)
                    or tuple(sorted(set(offsets))) != offsets):
                raise ValueError(
                    "velocity_innovation_offsets must be strictly increasing positive integers")
            if (self.velocity_memory_offsets is not None
                    or self.velocity_oscillator_periods is not None):
                raise ValueError(
                    "innovation, oscillator, and finite velocity memory are mutually exclusive")
            if self.state_variable != "velocity" or self.lag_steps != 1:
                raise ValueError(
                    "velocity innovation memory requires native-lag state_variable velocity")
            if self.velocity_innovation_rank is None:
                raise ValueError(
                    "velocity_innovation_rank requires velocity_innovation_offsets")
            if position_offsets is None:
                if self.history_steps < 1:
                    raise ValueError(
                        "velocity innovation memory requires a current-velocity history frame")
                position_offsets = tuple(range(1, self.history_steps + 1))
                object.__setattr__(self, "history_offsets", position_offsets)
            object.__setattr__(self, "velocity_innovation_offsets", offsets)
            # Reconstruct e[t+1-k] using the parent transition ending there.
            object.__setattr__(self, "history_steps", max(
                self.history_steps, offsets[-1] + position_offsets[-1]))
        if self.residual_architecture not in {"mlp", "tcn"}:
            raise ValueError("residual_architecture must be mlp or tcn")
        if self.residual_input not in {"state", "core_innovation"}:
            raise ValueError("residual_input must be state or core_innovation")
        if self.residual_input == "core_innovation":
            if (self.residual_architecture != "tcn"
                    or self.drift_type != "imported_oscillator_residual"):
                raise ValueError("core_innovation requires an imported oscillator TCN")
            if self.diffusion_type not in {"diagonal", "full"}:
                raise ValueError("core_innovation requires constant diffusion")
        if self.forcing_filter not in {"none", "modal_biquad"}:
            raise ValueError("forcing_filter must be none or modal_biquad")
        if self.forcing_filter != "none":
            if (self.residual_input != "core_innovation"
                    or self.residual_architecture != "tcn"
                    or self.drift_type != "imported_oscillator_residual"
                    or self.state_variable != "velocity" or self.lag_steps != 1):
                raise ValueError(
                    "forcing_filter requires a native-lag core-innovation imported oscillator TCN")
            if (not isinstance(self.forcing_filter_burnin, int)
                    or isinstance(self.forcing_filter_burnin, bool)
                    or self.forcing_filter_burnin < self.tcn_context_frames):
                raise ValueError(
                    "forcing_filter_burnin must be an integer at least tcn_context_frames")
        dilations = tuple(self.tcn_dilations)
        for name, values in (
                ("tcn_context_frames", (self.tcn_context_frames,)),
                ("tcn_channels", (self.tcn_channels,)),
                ("tcn_kernel_size", (self.tcn_kernel_size,)),
                ("tcn_dilations", dilations)):
            if not values or any(not isinstance(v, int) or isinstance(v, bool) or v < 1
                                 for v in values):
                raise ValueError(f"{name} must contain positive integers")
        object.__setattr__(self, "tcn_dilations", dilations)
        if self.residual_architecture == "tcn":
            if self.state_variable != "velocity" or self.lag_steps != 1:
                raise ValueError("TCN drift requires native-lag state_variable velocity")
            if self.drift_type not in {"mlp", "damped_residual", "imported_oscillator_residual"}:
                raise ValueError("TCN drift supports mlp, damped_residual, or imported_oscillator_residual drift_type")
            if any(value is not None for value in (
                    self.velocity_memory_offsets, self.velocity_oscillator_periods,
                    self.velocity_innovation_offsets)):
                raise ValueError("TCN drift cannot be combined with additive memory extensions")
            if self.tcn_context_frames > self.tcn_receptive_field:
                raise ValueError("tcn_context_frames exceeds the TCN receptive field")
            # Preserve the original flat conditioning taps (also used by
            # state-dependent diffusion), independently of dense TCN history.
            if position_offsets is None:
                position_offsets = tuple(range(1, max(1, self.history_steps) + 1))
                object.__setattr__(self, "history_offsets", position_offsets)
            # H tokens ending at t need a[t-H] for the oldest token's velocity.
            # Core innovations need one more position for its preceding velocity.
            object.__setattr__(self, "history_steps", max(
                position_offsets[-1], self.tcn_context_frames
                + int(self.residual_input == "core_innovation")))
        if self.fixed_memory_frequencies is not None:
            frequencies = tuple(float(x) for x in self.fixed_memory_frequencies)
            damping = tuple(float(x) for x in (self.fixed_memory_damping or ()))
            if (not frequencies or len(damping) != len(frequencies)
                    or any(not math.isfinite(f) or not 0 < f < .5 / self.dt
                           for f in frequencies)
                    or any(not math.isfinite(g) or g <= 0 for g in damping)):
                raise ValueError("fixed memory requires paired positive damping and sub-Nyquist frequencies")
            if (self.state_variable != "velocity" or self.lag_steps != 1
                    or any(x is not None for x in (self.velocity_memory_offsets,
                           self.velocity_oscillator_periods, self.velocity_innovation_offsets))):
                raise ValueError("fixed memory requires native-lag velocity and no other additive memory")
            if (not isinstance(self.fixed_memory_window, int)
                    or isinstance(self.fixed_memory_window, bool) or self.fixed_memory_window < 1):
                raise ValueError("fixed_memory_window must be a positive integer")
            if position_offsets is None:
                position_offsets = tuple(range(1, max(1, self.history_steps) + 1))
                object.__setattr__(self, "history_offsets", position_offsets)
            object.__setattr__(self, "fixed_memory_frequencies", frequencies)
            object.__setattr__(self, "fixed_memory_damping", damping)
            object.__setattr__(self, "history_steps", max(self.history_steps, self.fixed_memory_window))
        elif self.fixed_memory_damping is not None or self.fixed_memory_window:
            raise ValueError("fixed memory damping/window require frequencies")
        if (not math.isfinite(self.velocity_oscillator_radius_init)
                or not 0 < self.velocity_oscillator_radius_init < .9999):
            raise ValueError("velocity_oscillator_radius_init must be in (0, 0.9999)")
        if (not math.isfinite(self.velocity_oscillator_norm_cap)
                or self.velocity_oscillator_norm_cap <= 0):
            raise ValueError("velocity_oscillator_norm_cap must be positive")
        if self.velocity_innovation_rank is not None:
            if self.velocity_innovation_offsets is None:
                raise ValueError(
                    "velocity_innovation_rank requires velocity_innovation_offsets")
            if (not isinstance(self.velocity_innovation_rank, int)
                    or isinstance(self.velocity_innovation_rank, bool)
                    or not 1 <= self.velocity_innovation_rank <= self.reduced_dim):
                raise ValueError(
                    "velocity_innovation_rank must be between 1 and reduced_dim")
        if (not math.isfinite(self.velocity_innovation_norm_cap)
                or not 0 < self.velocity_innovation_norm_cap < 1):
            raise ValueError("velocity_innovation_norm_cap must be in (0, 1)")
        if self.velocity_memory_rank is not None:
            if self.velocity_memory_offsets is None:
                raise ValueError("velocity_memory_rank requires velocity_memory_offsets")
            if (not isinstance(self.velocity_memory_rank, int)
                    or isinstance(self.velocity_memory_rank, bool)
                    or not 1 <= self.velocity_memory_rank <= self.reduced_dim):
                raise ValueError("velocity_memory_rank must be between 1 and reduced_dim")
        if (not math.isfinite(self.velocity_memory_norm_cap)
                or self.velocity_memory_norm_cap <= 0):
            raise ValueError("velocity_memory_norm_cap must be positive")
        if self.reduced_dim < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("reduced_dim and dt must be positive")
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if self.activation not in {"silu", "gelu", "relu", "tanh"}:
            raise ValueError("unsupported activation")
        if self.drift_type not in {"mlp", "damped_residual", "capillary_energy", "imported_oscillator_residual"}:
            raise ValueError("drift_type must be mlp, damped_residual, capillary_energy, or imported_oscillator_residual")
        if not isinstance(self.structured_damping, bool):
            raise ValueError("structured_damping must be boolean")
        if not self.structured_damping and self.drift_type != "damped_residual":
            raise ValueError("disabling structured damping requires damped_residual drift")
        if (not math.isfinite(self.stiffness_init) or self.stiffness_init <= 0
                or not math.isfinite(self.damping_init) or self.damping_init <= 0):
            raise ValueError("stiffness_init and damping_init must be positive")
        if self.damped_operator not in {"diagonal", "spsd"}:
            raise ValueError("damped_operator must be diagonal or spsd")
        if not math.isfinite(self.surface_tension) or self.surface_tension <= 0:
            raise ValueError("surface_tension must be positive")
        if not math.isfinite(self.diffusion_init) or self.diffusion_init <= 0:
            raise ValueError("diffusion_init must be positive")
        diffusion_types = {"diagonal", "full", "state_diagonal", "bounded_state_diagonal"}
        if self.diffusion_type not in diffusion_types:
            raise ValueError(f"diffusion_type must be one of {sorted(diffusion_types)}")
        if not math.isfinite(self.diffusion_log_range) or self.diffusion_log_range <= 0:
            raise ValueError("diffusion_log_range must be positive")
        if self.lag_steps < 1 or self.history_steps < 0:
            raise ValueError("lag_steps must be positive and history_steps nonnegative")
        if self.state_variable not in {"state", "increment", "velocity"}:
            raise ValueError("state_variable must be state, increment, or velocity")
        if self.state_variable == "increment" and self.history_steps < 1:
            raise ValueError("increment state requires at least one preceding state")
        if self.state_variable == "increment" and self.effective_history_offsets[0] != 1:
            raise ValueError("increment state history_offsets must begin with 1")
        if self.state_variable == "velocity" and self.history_steps < 1:
            raise ValueError("velocity state requires at least one preceding native state")
        if (self.state_variable == "velocity" and self.history_offsets is not None
                and self.effective_history_offsets[0] != 1):
            raise ValueError("velocity state history_offsets must begin with 1")
        if (self.state_variable == "velocity" and self.history_steps > 1
                and self.lag_steps != 1):
            raise ValueError("velocity state supports additional history only at lag_steps 1")
        if self.drift_type in {"damped_residual", "capillary_energy", "imported_oscillator_residual"} and self.state_variable != "velocity":
            raise ValueError(f"{self.drift_type} drift requires state_variable velocity")
        if self.drift_type == "imported_oscillator_residual":
            if self.lag_steps != 1:
                raise ValueError("imported oscillator requires native-lag velocity")
            if any(value is not None for value in (
                    self.velocity_memory_offsets, self.velocity_oscillator_periods,
                    self.velocity_innovation_offsets, self.fixed_memory_frequencies)):
                raise ValueError("imported oscillator does not support additive memory extensions")
        if self.drift_type != "damped_residual" and self.damped_operator != "diagonal":
            raise ValueError("non-diagonal damped_operator requires damped_residual drift")

    @property
    def tcn_receptive_field(self) -> int:
        return 1 + (self.tcn_kernel_size - 1) * sum(self.tcn_dilations)

    @property
    def history_conditioning(self) -> bool:
        return self.history_steps > 0

    @property
    def effective_history_offsets(self) -> tuple[int, ...]:
        if self.history_offsets is not None:
            return self.history_offsets
        return tuple(range(1, self.history_steps + 1))

    def to_dict(self):
        return {**asdict(self), "history_conditioning": self.history_conditioning}

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        legacy_flag = values.pop("history_conditioning", None)
        values.setdefault("history_steps", int(bool(legacy_flag)))
        values.setdefault("history_offsets", None)
        values.setdefault("drift_type", "mlp")
        values.setdefault("stiffness_init", .01)
        values.setdefault("damping_init", .01)
        values.setdefault("damped_operator", "diagonal")
        values.setdefault("surface_tension", .0728)
        values.setdefault("capillary_mean_mode", True)
        values.setdefault("velocity_memory_offsets", None)
        values.setdefault("velocity_memory_rank", None)
        values.setdefault("velocity_memory_norm_cap", .25)
        values.setdefault("velocity_oscillator_periods", None)
        values.setdefault("velocity_oscillator_burnin", 256)
        values.setdefault("velocity_oscillator_radius_init", .98)
        values.setdefault("velocity_oscillator_norm_cap", .25)
        values.setdefault("velocity_innovation_offsets", None)
        values.setdefault("velocity_innovation_rank", None)
        values.setdefault("velocity_innovation_norm_cap", .95)
        values.setdefault("forcing_filter", "none")
        values.setdefault("forcing_filter_burnin", 32768)
        values.setdefault("structured_damping", True)
        values.setdefault("diffusion_type", "diagonal")
        values.setdefault("diffusion_log_range", 2.)
        if legacy_flag is not None and bool(legacy_flag) != (values["history_steps"] > 0):
            raise ValueError("history_conditioning and history_steps disagree")
        if values["history_offsets"] is not None:
            values["history_offsets"] = tuple(values["history_offsets"])
        if values["velocity_memory_offsets"] is not None:
            values["velocity_memory_offsets"] = tuple(values["velocity_memory_offsets"])
        if values["velocity_oscillator_periods"] is not None:
            values["velocity_oscillator_periods"] = tuple(
                values["velocity_oscillator_periods"])
        if values["velocity_innovation_offsets"] is not None:
            values["velocity_innovation_offsets"] = tuple(
                values["velocity_innovation_offsets"])
        return cls(**{**values, "hidden_layers": tuple(values["hidden_layers"])})
