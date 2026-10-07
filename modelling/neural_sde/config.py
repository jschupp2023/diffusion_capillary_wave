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

    def __post_init__(self):
        if self.history_offsets is not None:
            offsets = tuple(self.history_offsets)
            if (not offsets or any(not isinstance(offset, int) or isinstance(offset, bool)
                                   or offset < 1 for offset in offsets)
                    or tuple(sorted(set(offsets))) != offsets):
                raise ValueError("history_offsets must be strictly increasing positive integers")
            if self.history_steps not in {0, offsets[-1]}:
                raise ValueError("history_steps must equal the largest explicit history offset")
            object.__setattr__(self, "history_offsets", offsets)
            object.__setattr__(self, "history_steps", offsets[-1])
        if self.reduced_dim < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("reduced_dim and dt must be positive")
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if self.activation not in {"silu", "gelu", "relu", "tanh"}:
            raise ValueError("unsupported activation")
        if self.drift_type not in {"mlp", "damped_residual", "capillary_energy"}:
            raise ValueError("drift_type must be mlp, damped_residual, or capillary_energy")
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
        if self.drift_type in {"damped_residual", "capillary_energy"} and self.state_variable != "velocity":
            raise ValueError(f"{self.drift_type} drift requires state_variable velocity")
        if self.drift_type != "damped_residual" and self.damped_operator != "diagonal":
            raise ValueError("non-diagonal damped_operator requires damped_residual drift")

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
        values.setdefault("diffusion_type", "diagonal")
        values.setdefault("diffusion_log_range", 2.)
        if legacy_flag is not None and bool(legacy_flag) != (values["history_steps"] > 0):
            raise ValueError("history_conditioning and history_steps disagree")
        if values["history_offsets"] is not None:
            values["history_offsets"] = tuple(values["history_offsets"])
        return cls(**{**values, "hidden_layers": tuple(values["hidden_layers"])})
