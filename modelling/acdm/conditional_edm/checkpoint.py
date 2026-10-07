from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import EDMConfig
from .model import ConditionalEDM


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace a training checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=True)


def load_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[ConditionalEDM, dict[str, Any]]:
    checkpoint = load_checkpoint(path, device)
    model = ConditionalEDM(EDMConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
