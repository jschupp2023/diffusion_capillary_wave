"""Torch adapter for the shared-POD data source; preparation owns all splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def _trim_keep(batch, cutoff: float, state_std: Tensor | np.ndarray) -> np.ndarray:
    """Keep complete history-to-target stencils with only ordinary native jumps."""
    path = torch.from_numpy(np.asarray(batch["state_path"])).float()
    scale = torch.as_tensor(state_std, dtype=path.dtype)
    normalized = (path[:, 1:] - path[:, :-1]) / scale
    return (torch.linalg.vector_norm(normalized, dim=2) <= cutoff).all(dim=1).numpy()


def transition_batches(data: SharedPODTrainingData, config: EDMConfig, split: str,
                       batch_size: int, seed: int | None = None, *,
                       trim_cutoff: float | None = None,
                       trim_state_std: Tensor | np.ndarray | None = None):
    """Already-batched physical states; normalization happens inside the model."""
    if (trim_cutoff is None) != (trim_state_std is None):
        raise ValueError("trim_cutoff and trim_state_std must be supplied together")
    options = transition_options(config)
    if trim_cutoff is not None:
        options["include_state_path"] = True
    for _, batch in data.iter_transitions(split, batch_size=batch_size, seed=seed,
                                          **options):
        if trim_cutoff is not None:
            keep = _trim_keep(batch, trim_cutoff, trim_state_std)
            if not keep.any():
                continue
            batch = {key: value[keep] for key, value in batch.items()}
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


def _native_increment_norms(data: SharedPODTrainingData, split: str,
                            state_std: Tensor | np.ndarray) -> np.ndarray:
    """State-normalized one-frame increment norms, matching Neural-SDE trimming."""
    scale = np.asarray(state_std, dtype=np.float32)
    norms = []
    for _, batch in data.iter_transitions(
            split, lag_steps=1, history_steps=0, batch_size=data.batch_size):
        current = np.asarray(batch["current_state"], dtype=np.float32)
        following = np.asarray(batch["next_state"], dtype=np.float32)
        norms.append(np.linalg.norm((following - current) / scale, axis=1))
    if not norms:
        raise ValueError(f"No {split} increments are available for training trim.")
    return np.concatenate(norms)


def _trim_transition_counts(data: SharedPODTrainingData, config: EDMConfig, split: str,
                            cutoff: float, state_std: Tensor | np.ndarray) -> tuple[int, int]:
    total = removed = 0
    for _, batch in data.iter_transitions(
            split, batch_size=data.batch_size, include_state_path=True,
            **transition_options(config)):
        keep = _trim_keep(batch, cutoff, state_std)
        total += len(keep)
        removed += int((~keep).sum())
    return total, removed


def fit_trimmed_normalization(data: SharedPODTrainingData, config: EDMConfig,
                              trim_percent: float,
                              preliminary: NormalizationStats) -> tuple[NormalizationStats, dict]:
    """Freeze a training-only native-increment cutoff, then refit retained moments."""
    if not 0 < trim_percent < 100:
        raise ValueError("Training trim percentage must lie in (0, 100).")
    training_norms = _native_increment_norms(data, "train", preliminary.state_std)
    cutoff = float(np.percentile(training_norms, 100 - trim_percent))
    training_removed = training_norms > cutoff
    report = {
        "train_trim_percent": float(trim_percent),
        "train_trim_increment_count": len(training_norms),
        "train_trim_removed_count": int(training_removed.sum()),
        "train_trim_removed_percent": 100 * float(training_removed.mean()),
        "train_trim_norm_cutoff": cutoff,
        "train_trim_state_std": preliminary.state_std.tolist(),
        "train_trim_rule": (
            "cutoff is the training-only percentile of native one-frame increment norms; "
            "retain a transition only when every native increment from required history "
            "through its target is at or below the frozen cutoff"
        ),
    }
    for split in ("train", "validation"):
        norms = training_norms if split == "train" else _native_increment_norms(
            data, split, preliminary.state_std)
        removed = norms > cutoff
        total_transitions, removed_transitions = _trim_transition_counts(
            data, config, split, cutoff, preliminary.state_std)
        report.update({
            f"{split}_trim_increment_count": len(norms),
            f"{split}_trim_removed_count": int(removed.sum()),
            f"{split}_trim_removed_percent": 100 * float(removed.mean()),
            f"{split}_trim_transition_count": total_transitions,
            f"{split}_trim_removed_transition_count": removed_transitions,
            f"{split}_trim_removed_transition_percent": (
                100 * removed_transitions / total_transitions),
        })
        if removed_transitions == total_transitions:
            raise ValueError(f"Training trim retained no {split} transitions.")

    states = RunningMoments(data.n_coordinates)
    targets = RunningMoments(data.n_coordinates)
    retained = 0
    for _, batch in data.iter_transitions(
            "train", batch_size=data.batch_size, include_state_path=True,
            **transition_options(config)):
        keep = _trim_keep(batch, cutoff, preliminary.state_std)
        if not keep.any():
            continue
        current = np.asarray(batch["current_state"][keep], dtype=np.float64)
        following = np.asarray(batch["next_state"][keep], dtype=np.float64)
        states.update(current)
        targets.update(following - current if config.target_mode == "increment" else following)
        retained += len(current)
    state_mean, _, state_scale, _ = states.finish()
    target_mean, _, target_scale, _ = targets.finish()
    report.update(
        normalization_fit="retained_training_transitions",
        normalization_retained_transition_count=retained,
    )
    stats = NormalizationStats(*(torch.as_tensor(value, dtype=torch.float32)
                                 for value in (state_mean, state_scale,
                                               target_mean, target_scale)))
    return stats, report


def checkpoint_data(checkpoint, *, data_root=None, shared_basis=None):
    """Reopen the exact source/splits recorded by training, with optional relocation."""
    options = dict(checkpoint["data_config"])
    if data_root is not None:
        options["data_root"] = data_root
    if shared_basis is not None:
        options["shared_basis"] = shared_basis
    if 'temporal_component_manifest' in options:
        from modelling.data_preparation.frequency_components import FrequencyComponentData
        data = FrequencyComponentData(**options)
    else:
        data = SharedPODTrainingData(**options)
    if data.signature() != checkpoint["data_signature"]:
        raise ValueError("Checkpoint data do not match the shared basis, source files, or split.")
    return data
