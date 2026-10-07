"""Shared labels for experimental forcing conditions."""

from __future__ import annotations


CA_AC_TIMES_1E3 = {
    "0p0": 0.00,
    "0p04": 0.25,
    "0p07": 0.66,
    "0p08": 0.67,
    "0p10": 0.83,
    "0p15": 1.15,
    "0p18": 1.56,
    "0p20": 1.76,
    "0p25": 1.99,
    "0p30": 2.84,
    "0p35": 3.32,
}


def ca_ac_label(condition: str) -> str:
    """Return a presentation label for a condition's Ca_ac value."""
    try:
        value = CA_AC_TIMES_1E3[condition]
    except KeyError as exc:
        raise ValueError(f"Missing Ca_ac conversion for {condition!r}") from exc
    return f"{value:.2f} × 10⁻³"


def ca_ac_labels(conditions: list[str] | tuple[str, ...]) -> list[str]:
    """Return Ca_ac presentation labels in the supplied condition order."""
    return [ca_ac_label(condition) for condition in conditions]
