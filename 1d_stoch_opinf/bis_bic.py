"""Wavelet bispectrum and Orosco et al. (2023), Eq. 11 bicoherence.

The normalization is |mean(Q)| / mean(|Q|), where
Q = W(f1) W(f2) conj(W(f1 + f2)). It is not squared bicoherence.

Frequency sums are matched in Hz. Default frequencies are positive integer
multiples of fb_high / Ns, so every sum is represented exactly. fb_high bounds
the transform, including sum frequencies; the square output ends at fb_high/2.
The former hard-coded 34500 scale factor and offset index addition are corrected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pywt

# Retained for existing imports; these constants do not enter the metrics.
surf_ten = 0.0728
density = 998
h = 725 / 10**6
L = 9.525 / 10**3
km = np.pi / L


@dataclass(frozen=True)
class BispectralResult:
    frequency_hz: np.ndarray
    complex_bispectrum: np.ndarray
    normalization_magnitude: np.ndarray
    bicoherence: np.ndarray
    transform_frequency_hz: np.ndarray
    scales: np.ndarray
    sample_start: int
    sample_stop: int
    removed_temporal_mean: float

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.complex_bispectrum)


def frequency_grid(fs, fb_low, fb_high, Ns=1024, wavelet="cgau1", scales=None):
    """Return scales, CWT frequencies, input indices, and exact sum indices.

    Custom scales must contain every required sum frequency. Missing frequencies
    raise an error rather than silently substituting a nearby coefficient.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive.")
    if not 0 <= fb_low < fb_high / 2 or not fb_high < fs / 2:
        raise ValueError("Require 0 <= fb_low < fb_high/2 and fb_high < Nyquist.")
    if not isinstance(Ns, (int, np.integer)) or Ns < 4:
        raise ValueError("Ns must be an integer of at least 4.")
    if scales is None:
        step = fb_high / Ns
        first = max(1, int(np.ceil(fb_low / step)))
        frequencies = step * np.arange(first, Ns + 1, dtype=np.float64)
        scales = pywt.frequency2scale(wavelet, frequencies / fs)
    else:
        scales = np.asarray(scales, dtype=np.float64)
        if scales.ndim != 1 or not len(scales) or not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError("scales must be a nonempty vector of finite positive values.")
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    scales = np.asarray(scales)[order]
    if np.any(np.diff(frequencies) <= 0) or np.any(frequencies >= fs / 2):
        raise ValueError("CWT frequencies must be distinct and below Nyquist.")
    tolerance = 1e-10 * fb_high
    indices = np.flatnonzero(
        (frequencies >= fb_low - tolerance)
        & (frequencies <= fb_high / 2 + tolerance)
    )
    if len(indices) < 2:
        raise ValueError("The requested grid has fewer than two output frequencies.")
    inputs = frequencies[indices]
    sums = inputs[:, None] + inputs[None, :]
    right = np.clip(np.searchsorted(frequencies, sums), 0, len(frequencies) - 1)
    left = np.maximum(right - 1, 0)
    sum_indices = np.where(
        np.abs(frequencies[left] - sums) < np.abs(frequencies[right] - sums), left, right
    )
    if not np.allclose(frequencies[sum_indices], sums, rtol=1e-10, atol=tolerance):
        raise ValueError(
            "Custom scales do not contain all exact f1 + f2 frequencies. "
            "Use scales=None or a sum-closed frequency grid."
        )
    return scales, frequencies, indices, sum_indices


def _rows(array, indices):
    """Use a view for contiguous frequency rows, avoiding large fancy-index copies."""
    if len(indices) == 1 or np.all(np.diff(indices) == 1):
        return array[indices[0] : indices[-1] + 1]
    return array[indices]


def wavelet_bispectral_metrics(
    zf, t, fs, fb_low=150.0, fb_high=57000.0, scales=None, Ns=1024,
    wavelet="cgau1", *, demean=True, trim_edges=True,
    progress: Callable[[str], None] | None = None,
) -> BispectralResult:
    """Calculate both metrics with float64/complex128 arithmetic.

    By default, subtract a single temporal mean and exclude both record ends by
    the largest wavelet support radius. All frequency pairs use the same retained
    time interval. No linear detrending, resampling, or ensemble averaging occurs.
    Timestamp step variations up to 1% are tolerated for quantized stored times;
    the CWT itself assumes uniform sampling at fs.

    Accumulation uses only the upper triangle and mirrors it. It never constructs
    a time x frequency x frequency tensor. Zero denominators produce NaN.
    """
    signal = np.asarray(zf, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    if signal.ndim != 1 or times.shape != signal.shape or len(signal) < 4:
        raise ValueError("zf and t must be matching one-dimensional arrays, length >= 4.")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(times)):
        raise ValueError("Signal and timestamps must be finite.")
    scales, frequencies, indices, sum_indices = frequency_grid(
        fs, fb_low, fb_high, Ns, wavelet, scales
    )
    if not np.allclose(np.diff(times), 1 / fs, rtol=0.01, atol=1e-12 / fs):
        raise ValueError("Timestamps must be uniformly sampled at fs (1% tolerance).")
    wav = pywt.ContinuousWavelet(wavelet)
    radius = max(abs(wav.lower_bound), abs(wav.upper_bound))
    margin = int(np.ceil(radius * max(scales))) if trim_edges else 0
    start, stop = margin, len(signal) - margin
    if stop - start < 4:
        raise ValueError("Record is too short after excluding the wavelet edge support.")
    removed_mean = float(np.mean(signal)) if demean else 0.0
    if progress:
        progress(f"CWT: {len(scales)} scales; retain samples [{start}:{stop}].")
    # Scale batches keep the transient FFT arrays small while storing one CWT.
    spec = np.empty((len(scales), stop - start), dtype=np.complex128)
    centered = signal - removed_mean
    for first in range(0, len(scales), 32):
        block, _ = pywt.cwt(
            centered, scales[first : first + 32], wav,
            sampling_period=1 / fs, method="fft",
        )
        spec[first : first + len(block)] = block[:, start:stop]
    del block
    magnitude = np.abs(spec)
    conjugate = np.conjugate(spec)
    size = len(indices)
    numerator = np.empty((size, size), dtype=np.complex128)
    denominator = np.empty((size, size), dtype=np.float64)
    count = stop - start
    for i, source in enumerate(indices):
        # einsum sums the products in time without storing Q(t,f1,f2).
        numerator[i, i:] = np.einsum(
            "t,jt,jt->j", spec[source], _rows(spec, indices[i:]),
            _rows(conjugate, sum_indices[i, i:]), optimize=False,
        ) / count
        denominator[i, i:] = np.einsum(
            "t,jt,jt->j", magnitude[source], _rows(magnitude, indices[i:]),
            _rows(magnitude, sum_indices[i, i:]), optimize=False,
        ) / count
        numerator[i:, i] = numerator[i, i:]
        denominator[i:, i] = denominator[i, i:]
        if progress and ((i + 1) % max(1, size // 10) == 0 or i + 1 == size):
            progress(f"Frequency rows: {i + 1}/{size}.")
    bicoherence = np.full_like(denominator, np.nan)
    np.divide(np.abs(numerator), denominator, out=bicoherence, where=denominator > 0)
    np.clip(bicoherence, 0, 1, out=bicoherence)
    return BispectralResult(
        frequencies[indices], numerator, denominator, bicoherence,
        frequencies, scales, start, stop, removed_mean,
    )


def wav_bispectrum(zf, t, fs, fb_low, fb_high, scales=None, Ns=2**10,
                   wavelet="cgau1", bico=False, **kwargs):
    """Compatibility API returning frequencies and the complex time average.

    bico=True includes available positive input frequencies below fb_low, as in
    the original deferred-cropping API. Defaults now demean and trim edges.
    """
    if bico and scales is None:
        scales = pywt.frequency2scale(wavelet, (fb_high / Ns) * np.arange(1, Ns + 1) / fs)
    result = wavelet_bispectral_metrics(
        zf, t, fs, 0 if bico else fb_low, fb_high, scales, Ns, wavelet, **kwargs
    )
    return result.frequency_hz, result.complex_bispectrum


def wav_bicoherence(zf, t, fs, fb_low, fb_high, scales=None, Ns=2**10,
                    wavelet="cgau1", **kwargs):
    """Compatibility API returning frequencies, bispectrum magnitude, and b."""
    result = wavelet_bispectral_metrics(
        zf, t, fs, fb_low, fb_high, scales, Ns, wavelet, **kwargs
    )
    return result.frequency_hz, result.magnitude, result.bicoherence

