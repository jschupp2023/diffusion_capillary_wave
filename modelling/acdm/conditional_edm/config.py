from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class EDMConfig:
    """Model, training-noise, and sampling settings in normalized coordinates."""

    reduced_dim: int
    param_dim: int = 0
    native_dt: float | None = None
    lag_steps: int = 1
    stride_steps: int = 1
    history_conditioning: bool = False
    history_steps: int = 1
    diffusion_formulation: Literal["edm", "ddpm"] = "edm"
    conditioning_mode: Literal["clean", "joint_noised"] = "joint_noised"
    target_mode: Literal["increment", "future_state"] = "increment"
    hidden_dim: int = 256
    num_blocks: int = 4
    noise_embedding_dim: int = 64
    activation: Literal["silu", "gelu", "relu"] = "silu"
    dropout: float = 0.0

    sigma_data: float = 1.0
    p_mean: float = -1.2
    p_std: float = 1.2
    train_sigma_min: float | None = 0.002
    train_sigma_max: float | None = 80.0

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    num_sampling_steps: int = 32

    def __post_init__(self) -> None:
        finite_values = (
            self.dropout,
            self.sigma_data,
            self.p_mean,
            self.p_std,
            self.sigma_min,
            self.sigma_max,
            self.rho,
        )
        finite_values += tuple(
            value
            for value in (self.native_dt, self.train_sigma_min, self.train_sigma_max)
            if value is not None
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("all floating-point model settings must be finite")
        if self.reduced_dim < 1 or self.param_dim < 0:
            raise ValueError("reduced_dim must be positive and param_dim nonnegative")
        if self.native_dt is not None and self.native_dt <= 0:
            raise ValueError("native_dt must be positive when supplied")
        if self.lag_steps < 1 or self.stride_steps < 1:
            raise ValueError("lag_steps and stride_steps must be positive")
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")
        if self.diffusion_formulation not in {"edm", "ddpm"}:
            raise ValueError("diffusion_formulation must be 'edm' or 'ddpm'")
        if self.conditioning_mode not in {"clean", "joint_noised"}:
            raise ValueError("conditioning_mode must be 'clean' or 'joint_noised'")
        if self.diffusion_formulation == "ddpm" and self.conditioning_mode != "joint_noised":
            raise ValueError("Kohl DDPM requires conditioning_mode='joint_noised'")
        if self.target_mode not in {"increment", "future_state"}:
            raise ValueError("target_mode must be 'increment' or 'future_state'")
        if self.hidden_dim < 1 or self.num_blocks < 0:
            raise ValueError("hidden_dim must be positive and num_blocks nonnegative")
        if self.noise_embedding_dim < 2:
            raise ValueError("noise_embedding_dim must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.sigma_data <= 0 or self.p_std < 0:
            raise ValueError("sigma_data must be positive and p_std nonnegative")
        if self.sigma_min <= 0 or self.sigma_max <= self.sigma_min:
            raise ValueError("require 0 < sigma_min < sigma_max")
        if self.rho <= 0 or self.num_sampling_steps < 2:
            raise ValueError("rho must be positive and num_sampling_steps at least 2")
        if self.train_sigma_min is not None and self.train_sigma_min <= 0:
            raise ValueError("train_sigma_min must be positive")
        if self.train_sigma_max is not None and self.train_sigma_max <= 0:
            raise ValueError("train_sigma_max must be positive")
        if (
            self.train_sigma_min is not None
            and self.train_sigma_max is not None
            and self.train_sigma_max < self.train_sigma_min
        ):
            raise ValueError("train_sigma_max must not be below train_sigma_min")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def conditioning_dim(self) -> int:
        return self.reduced_dim * (1 + (self.history_steps if self.history_conditioning else 0)) + self.param_dim

    @property
    def joint_dim(self) -> int:
        return self.conditioning_dim + self.reduced_dim

    @property
    def physical_lag(self) -> float | None:
        return None if self.native_dt is None else self.lag_steps * self.native_dt

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EDMConfig":
        values = dict(values)
        values.setdefault("diffusion_formulation", "edm")
        # Checkpoints predating the explicit mode used either clean conditioning
        # or the equivalent equal-sigma/equal-weight joint formulation.
        if "conditioning_mode" not in values:
            scale = values.get("conditioning_noise_scale", 1.0)
            weight = values.get("conditioning_loss_weight", 1.0)
            if values.get("conditioning_noise", False) and (scale != 1.0 or weight != 1.0):
                raise ValueError("legacy unequal conditioning noise/loss settings cannot map to joint_noised")
            values["conditioning_mode"] = (
                "joint_noised" if values.pop("conditioning_noise", False) else "clean"
            )
        values.pop("conditioning_noise", None)
        values.pop("conditioning_noise_scale", None)
        values.pop("conditioning_loss_weight", None)
        return cls(**values)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    lr_scheduler: Literal["constant", "cosine"] = "constant"
    min_learning_rate: float = 0.0
    weight_decay: float = 0.0
    epochs: int = 20
    seed: int = 0
    max_train_batches: int | None = None
    max_validation_batches: int | None = None
    cache_gib: float = 2.0
    fixed_mse_batch_size: int = 256
    fixed_sigmas: tuple[float, float, float] = (.03, .3, 3.)
    rollout_every: int = 10
    one_step_conditions: int = 32
    one_step_ensemble_size: int = 32
    rollout_horizon: int = 200
    rollout_ensemble_size: int = 2
    rollout_energy_factor: float = 10.
    surface_tension: float = .0728

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.learning_rate,
                self.min_learning_rate,
                self.weight_decay,
            )
        ):
            raise ValueError("all floating-point training settings must be finite")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler must be 'constant' or 'cosine'")
        if self.min_learning_rate < 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must lie in [0, learning_rate]")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive when set")
        if self.max_validation_batches is not None and self.max_validation_batches < 1:
            raise ValueError("max_validation_batches must be positive when set")
        if not math.isfinite(self.cache_gib) or self.cache_gib < 0:
            raise ValueError("cache_gib must be finite and nonnegative")
        if self.fixed_mse_batch_size < 0 or self.rollout_every < 0 or self.one_step_conditions < 0:
            raise ValueError("Diagnostic batch size and frequency must be nonnegative; zero disables")
        if self.one_step_ensemble_size < 1:
            raise ValueError("one_step_ensemble_size must be positive")
        if self.one_step_conditions and self.one_step_ensemble_size < 2:
            raise ValueError("one_step_ensemble_size must be at least two when one-step diagnostics are enabled")
        if min(self.rollout_horizon, self.rollout_ensemble_size) < 1:
            raise ValueError("Rollout horizon and ensemble size must be positive")
        if len(self.fixed_sigmas) != 3 or any(not math.isfinite(s) or s <= 0 for s in self.fixed_sigmas):
            raise ValueError("fixed_sigmas must contain three positive finite values")
        if not math.isfinite(self.rollout_energy_factor) or self.rollout_energy_factor <= 1:
            raise ValueError("rollout_energy_factor must be finite and greater than one")
        if not math.isfinite(self.surface_tension) or self.surface_tension <= 0:
            raise ValueError("surface_tension must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        values = dict(values)
        if "fixed_sigmas" in values:
            values["fixed_sigmas"] = tuple(values["fixed_sigmas"])
        return cls(**values)


# These change monitoring/IO only and may be overridden when resuming.
RUNTIME_OPTIONS = ("cache_gib", "fixed_mse_batch_size", "fixed_sigmas", "rollout_every",
                   "one_step_conditions", "one_step_ensemble_size",
                   "rollout_horizon", "rollout_ensemble_size", "rollout_energy_factor", "surface_tension")
