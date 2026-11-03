"""Indexed velocity windows with the legacy job order and exact trim stencil.

Only compact per-frame coordinates and invalid-increment prefix counts are
cached, never the much larger collection of overlapping history windows.
"""
import math

import numpy as np
import torch


def indexed_velocity_batches(data, config, split, batch_size, seed=None,
                             horizons=None, cutoff=None, state_std=None,
                             selection_fraction=1., selection_generator=None,
                             eligibility_history_steps=None,
                             include_window_ids=False):
    if config.state_variable != "velocity":
        raise ValueError("indexed windows require velocity coordinates")
    if not 0 < selection_fraction <= 1:
        raise ValueError("selection fraction must be in (0, 1]")
    model_history = config.history_steps
    history = (model_history if eligibility_history_steps is None
               else eligibility_history_steps)
    if (not isinstance(history, int) or isinstance(history, bool)
            or history < model_history):
        raise ValueError(
            "eligibility_history_steps must be an integer at least as large as model history")
    lag = config.lag_steps * max(horizons or (1,))
    width = min(batch_size, data.batch_size)
    jobs = []
    for name in data.splits[split]:
        last = data.frame_counts[name] - lag
        if history >= last:
            raise ValueError(f"Repetition {name} is too short for this history and lag.")
        jobs.extend((name, start, min(start + width, last))
                    for start in range(history, last, width))
    rng = np.random.default_rng(seed)
    if seed is not None:
        rng.shuffle(jobs)
    scale = None if state_std is None else torch.as_tensor(state_std, dtype=torch.float32).cpu()
    if cutoff is not None and scale is None:
        raise ValueError("trim cutoff requires the saved state standard deviation")
    cache_key = None if cutoff is None else (float(cutoff), tuple(scale.tolist()))
    # This cache is tied to this prepared-data object, not to paths on disk.
    # If coordinate caching is unavailable, read only the current job instead.
    coordinates = getattr(data, "_coordinate_cache", {})
    if not hasattr(data, "_sde_trim_prefix_cache"):
        data._sde_trim_prefix_cache = {}
    prefixes = data._sde_trim_prefix_cache.setdefault(cache_key, {})
    source_indices = {name: index for index, name in enumerate(data.splits[split])}
    for name, start, stop in jobs:
        indices = np.arange(start, stop)
        if seed is not None:
            rng.shuffle(indices)
        if name in coordinates:
            values = coordinates[name][1]
            offset = 0
            if cutoff is not None and name not in prefixes:
                # Match the legacy float32 path subtraction and torch norm.
                # Chunking bounds temporary memory without crossing rep gaps.
                prefix = np.zeros(len(values), dtype=np.int64)
                for left in range(0, len(values) - 1, 65536):
                    right = min(left + 65536, len(values) - 1)
                    v = torch.from_numpy(values[left:right + 1].copy()).float()
                    bad = ~(torch.linalg.vector_norm((v[1:] - v[:-1]) / scale, dim=-1) <= cutoff)
                    prefix[left + 1:right + 1] = prefix[left] + np.cumsum(bad.numpy())
                prefixes[name] = prefix
            prefix = prefixes.get(name)
        else:
            offset = start - history
            _, values = data.read_coordinates(name, offset, stop + lag)
            prefix = None
            if cutoff is not None:
                v = torch.from_numpy(values.copy()).float()
                bad = ~(torch.linalg.vector_norm((v[1:] - v[:-1]) / scale, dim=-1) <= cutoff)
                prefix = np.r_[0, np.cumsum(bad.numpy())]
        local = indices - offset
        if cutoff is not None:
            keep = prefix[local + lag] == prefix[local - history]
            local = local[keep]
        if not len(local):
            continue
        count = max(1, math.ceil(selection_fraction * len(local)))
        if count < len(local):
            selected = torch.randperm(len(local), generator=selection_generator)[:count].sort().values
            local = local[selected.numpy()]
        following = local + config.lag_steps
        offsets = np.arange(1, model_history + 1)
        batch = dict(current_state=values[local], next_state=values[following],
                     history_states=values[local[:, None] - offsets],
                     next_history_states=values[following[:, None] - offsets])
        if horizons is not None:
            ends = local[:, None] + config.lag_steps * np.asarray(horizons)
            # Subtract in source precision before casting, like the reference
            # multistep sampler (unlike the float32 trim test above).
            batch["target_velocities"] = values[ends] - values[ends - 1]
            batch["target_displacements"] = values[ends] - values[local, None]
        if include_window_ids:
            batch["window_start_index"] = local + offset
            batch["window_source_index"] = np.full(
                len(local), source_indices[name], dtype=np.int64)
        yield {key: torch.from_numpy(value).float() for key, value in batch.items()}
