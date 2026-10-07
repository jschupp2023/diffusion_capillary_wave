"""Evaluate the existing wavelet metrics on a subset of the original grid.

This changes frequency-map sampling, not the wavelet, sampling rate, frequency
endpoints, or averaging interval. All sum frequencies are still represented
exactly. Dense cached maps can be sliced to this grid without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pywt

from bis_bic import BispectralResult, frequency_grid


@dataclass(frozen=True)
class SparseGrid:
    output_positions: np.ndarray
    frequency_hz: np.ndarray
    transform_frequency_hz: np.ndarray
    scales: np.ndarray
    input_indices: np.ndarray
    sum_indices: np.ndarray


def sparse_grid(fs, fb_low=150, fb_high=57000, Ns=512, wavelet="cgau1", axis_samples=64):
    if not isinstance(axis_samples, (int, np.integer)) or axis_samples < 2:
        raise ValueError("axis_samples must be an integer >= 2.")
    scales, frequencies, inputs, sums = frequency_grid(fs, fb_low, fb_high, Ns, wavelet)
    f = frequencies[inputs]
    # Pick nearest original frequencies to log-spaced targets; collisions at the
    # densely requested low end are removed, so actual count can be smaller.
    target = np.geomspace(f[0], f[-1], min(axis_samples, len(f)))
    right = np.clip(np.searchsorted(f, target), 0, len(f) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.unique(np.where(np.abs(f[left] - target) < np.abs(f[right] - target), left, right))
    sum_source = sums[np.ix_(selected, selected)]
    needed = np.unique(np.r_[inputs[selected], sum_source.ravel()])
    return SparseGrid(
        selected, f[selected], frequencies[needed], scales[needed],
        np.searchsorted(needed, inputs[selected]), np.searchsorted(needed, sum_source),
    )


def sparse_wavelet_metrics(signal, t, fs, fb_low=150, fb_high=57000, Ns=512,
                           wavelet="cgau1", axis_samples=64):
    """Same triple-product formula and edge exclusion as bis_bic.py.

    Returns result and timing metadata. Wavelets are computed only at the input
    and sum frequencies required by the sparse grid. Both metrics share them.
    """
    started = time.perf_counter()
    signal, t = np.asarray(signal, dtype=np.float64), np.asarray(t, dtype=np.float64)
    if signal.ndim != 1 or t.shape != signal.shape or len(signal) < 4:
        raise ValueError("Signal and t must have the same one-dimensional shape, length >= 4.")
    if not np.isfinite(signal).all() or not np.isfinite(t).all():
        raise ValueError("Signal and timestamps must be finite.")
    grid = sparse_grid(fs, fb_low, fb_high, Ns, wavelet, axis_samples)
    if not np.allclose(np.diff(t), 1/fs, rtol=.01, atol=1e-12/fs):
        raise ValueError("Timestamps must be uniformly sampled at fs (1% tolerance).")
    wav = pywt.ContinuousWavelet(wavelet)
    margin = int(np.ceil(max(abs(wav.lower_bound), abs(wav.upper_bound)) * grid.scales.max()))
    start, stop = margin, len(t) - margin
    if stop - start < 4:
        raise ValueError("Record is too short after wavelet edge exclusion.")
    temporal_mean = float(signal.mean())
    centered = signal - temporal_mean
    spec = np.empty((len(grid.scales), stop-start), dtype=np.complex128)
    for first in range(0, len(grid.scales), 32):
        block, _ = pywt.cwt(centered, grid.scales[first:first+32], wav, 1/fs, method="fft")
        spec[first:first+len(block)] = block[:, start:stop]
    del block
    cwt_finished = time.perf_counter()
    magnitude = np.abs(spec)
    conjugate = np.conjugate(spec)
    size = len(grid.frequency_hz)
    numerator = np.empty((size, size), dtype=np.complex128)
    denominator = np.empty((size, size), dtype=np.float64)
    for i, index in enumerate(grid.input_indices):
        # Small row batches bound the temporary arrays from noncontiguous indexing.
        for first in range(i, size, 16):
            last = min(first + 16, size)
            rows, summed = grid.input_indices[first:last], grid.sum_indices[i, first:last]
            numerator[i, first:last] = np.einsum(
                "t,jt,jt->j", spec[index], spec[rows], conjugate[summed], optimize=False,
            ) / (stop-start)
            denominator[i, first:last] = np.einsum(
                "t,jt,jt->j", magnitude[index], magnitude[rows], magnitude[summed], optimize=False,
            ) / (stop-start)
        numerator[i:, i] = numerator[i, i:]
        denominator[i:, i] = denominator[i, i:]
    b = np.full_like(denominator, np.nan)
    np.divide(np.abs(numerator), denominator, out=b, where=denominator > 0)
    np.clip(b, 0, 1, out=b)
    result = BispectralResult(grid.frequency_hz, numerator, denominator, b,
                             grid.transform_frequency_hz, grid.scales, start, stop, temporal_mean)
    timing = {"total_seconds": time.perf_counter()-started, "cwt_seconds": cwt_finished-started,
              "axis_samples_requested": axis_samples, "axis_samples_actual": size,
              "transform_scales_actual": len(grid.scales)}
    return result, timing
