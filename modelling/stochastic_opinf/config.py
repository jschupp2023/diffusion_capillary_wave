"""Serializable stochastic-OpInf configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class OpInfConfig:
    reduced_dim: int
    dt: float
    model_form: str = "A"
    input_amplitude: float = 1.0
    input_frequency: float = 7.0e6
    sigma: float = 1.0

    # Attributes consumed by the shared rollout evaluator.
    lag_steps: int = 1
    history_steps: int = 0

    def __post_init__(self) -> None:
        if self.reduced_dim < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("reduced_dim and dt must be positive")
        if self.model_form not in {"A", "AB", "AN", "ABN"}:
            raise ValueError("model_form must be A, AB, AN, or ABN")
        if not all(math.isfinite(value) for value in
                   (self.input_amplitude, self.input_frequency, self.sigma)):
            raise ValueError("input and noise settings must be finite")
        if self.input_frequency < 0 or self.sigma < 0:
            raise ValueError("input_frequency and sigma must be nonnegative")
        if self.lag_steps != 1 or self.history_steps != 0:
            raise ValueError("stochastic OpInf uses native state steps without history")

    @property
    def history_conditioning(self) -> bool:
        return False

    @property
    def is_bilinear(self) -> bool:
        return "N" in self.model_form

    @property
    def has_input_term(self) -> bool:
        return "B" in self.model_form

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "OpInfConfig":
        return cls(**values)
