"""Training-only fixed variance for the correlation-matching auxiliary.

All eligible observed starts are counted once. Complete history-to-maximum-lag
stencils must satisfy the saved trim; neither validation nor test is read.
"""
import numpy as np
import torch

from modelling.data_preparation.prepare_shared_pod_training import RunningMoments


def fit_correlation_reference(data, model, horizons, cutoff=None, trim_scale=None):
    if model.config.state_variable != "velocity":
        raise ValueError("correlation reference requires a velocity model")
    horizons = tuple(horizons)
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("correlation horizons must be positive integers")
    history = model.config.history_steps
    lag = model.config.lag_steps * max(horizons)
    stats = RunningMoments(data.n_coordinates)
    mean = model.state_mean.detach().cpu().numpy().astype(np.float64)
    scale = model.state_std.detach().cpu().numpy().astype(np.float64)
    counts = {}
    for name in data.splits["train"]:
        cached = getattr(data, "_coordinate_cache", {})
        values = (cached[name][1] if name in cached else
                  data.read_coordinates(name, 0, data.frame_counts[name])[1])
        starts = np.arange(history, len(values)-lag)
        eligible = len(starts)
        if cutoff is not None:
            if trim_scale is None:
                raise ValueError("trim requires saved normalization scale")
            positions = torch.tensor(np.asarray(values), dtype=torch.float32)
            bad = ~(torch.linalg.vector_norm((positions[1:]-positions[:-1]) /
                torch.as_tensor(trim_scale, dtype=torch.float32), dim=-1) <= cutoff)
            prefix = np.r_[0, np.cumsum(bad.numpy())]
            starts = starts[prefix[starts+lag] == prefix[starts-history]]
        counts[name] = dict(eligible=eligible, retained=len(starts))
        for begin in range(0, len(starts), 65536):
            # Match float32 model inputs, then accumulate in float64.
            current = values[starts[begin:begin+65536]].astype(np.float32).astype(np.float64)
            stats.update((current-mean)/scale)
    if stats.count < 2:
        raise ValueError("fewer than two training starts survive the correlation stencil")
    variance = stats.m2/(stats.count-1)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("training reference contains a constant or invalid coordinate")
    report = dict(split="train", counts=counts, total_starts=stats.count,
                  history_steps=history, maximum_lag_steps=lag,
                  normalization="sample variance of normalized observed starts; frozen across batches and splits",
                  normalized_variance=variance.tolist(), normalized_mean=stats.mean.tolist(),
                  trim_cutoff=cutoff)
    return torch.tensor(variance, dtype=model.state_std.dtype), report
