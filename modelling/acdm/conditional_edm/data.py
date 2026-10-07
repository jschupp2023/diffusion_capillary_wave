"""Torch adapter for the shared-POD data source; preparation owns all splits."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from modelling.data_preparation.prepare_shared_pod_training import RunningMoments, SharedPODTrainingData
from .config import EDMConfig


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: Tensor
    state_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    def to_dict(self):
        return {name: getattr(self, name).cpu() for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values):
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})


def transition_options(config: EDMConfig):
    return dict(lag_steps=config.lag_steps, stride_steps=config.stride_steps,
                history_steps=config.history_steps if config.history_conditioning else 0)


def transition_batches(data: SharedPODTrainingData, config: EDMConfig, split: str,
                       batch_size: int, seed: int | None = None):
    """Already-batched physical states; normalization happens inside the model."""
    for _, batch in data.iter_transitions(split, batch_size=batch_size, seed=seed,
                                          **transition_options(config)):
        yield {key: torch.from_numpy(value).float() for key, value in batch.items()
               if key in ("current_state", "next_state", "history_states")}


def fit_normalization(data: SharedPODTrainingData, config: EDMConfig) -> NormalizationStats:
    """Streaming float64 moments, fitted only on preparation's training split."""
    states = RunningMoments(data.n_coordinates)
    for _, _, values in data._iter_physical_batches("train", data.batch_size):
        states.update(values)
    mean, _, scale, _ = states.finish()
    if config.target_mode == "future_state":
        target_mean, target_scale = mean.copy(), scale.copy()
    else:
        targets = RunningMoments(data.n_coordinates)
        for _, batch in data.iter_transitions("train", batch_size=data.batch_size,
                                              **transition_options(config)):
            targets.update(batch["next_state"] - batch["current_state"])
        target_mean, _, target_scale, _ = targets.finish()
    return NormalizationStats(*(torch.as_tensor(value, dtype=torch.float32)
                                for value in (mean, scale, target_mean, target_scale)))


def checkpoint_data(checkpoint, *, data_root=None, shared_basis=None):
    """Reopen the exact source/splits recorded by training, with optional relocation."""
    options = dict(checkpoint["data_config"])
    if data_root is not None:
        options["data_root"] = data_root
    if shared_basis is not None:
        options["shared_basis"] = shared_basis
    data = SharedPODTrainingData(**options)
    if data.signature() != checkpoint["data_signature"]:
        raise ValueError("Checkpoint data do not match the shared basis, source files, or split.")
    return data
