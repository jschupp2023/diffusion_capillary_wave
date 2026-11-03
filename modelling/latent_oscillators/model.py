"""Stable oscillators with unit stationary displacement variance and real readout.

For each independent oscillator, dq = omega*u dt and
du = (-omega*q - 2*gamma*u)dt + sqrt(4*gamma)dW. Thus u=p/omega,
Cov(q,u)=I, and the readout carries the otherwise unidentified forcing scale.
Several independent copies may share a pole (spectral rank > 1).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import expm
from scipy.signal import get_window


def welch_matrix(records: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-record Welch matrices, S_ij=E[X_i conj(X_j)], no joined records."""
    records = np.asarray(records, dtype=float)
    if records.ndim != 3 or records.shape[1] < nperseg or nperseg < 4:
        raise ValueError('Need (record,time,coordinate) data, time >= nperseg >= 4')
    if not np.isfinite(records).all() or fs <= 0:
        raise ValueError('Data must be finite and sampling rate positive')
    window = get_window('hann', nperseg)
    starts = range(0, records.shape[1]-nperseg+1, nperseg-nperseg//2)
    total = np.zeros((len(records), nperseg//2+1, records.shape[-1], records.shape[-1]), complex)
    for start in starts:
        x = records[:, start:start+nperseg]
        x = x-x.mean(axis=1, keepdims=True)
        z = np.fft.rfft(x*window[None, :, None], axis=1)
        total += np.einsum('rfi,rfj->rfij', z, z.conj())
    total /= len(starts)*fs*np.sum(window**2)
    total[:, 1:-1 if nperseg % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(nperseg, 1/fs), total


def taper_correlation(n: int) -> np.ndarray:
    w = get_window('hann', n)
    return np.correlate(w, w, mode='full')[n-1:]/np.dot(w, w)


def expected_welch(correlation: torch.Tensor, fs: float, taper: torch.Tensor) -> torch.Tensor:
    """Expected one-sided Hann periodogram from nonnegative-lag covariance.

    Exact finite-window smoothing including sampling aliasing. Constant detrend
    is immaterial at bins >= 2 for a periodic Hann taper; bins 0/1 are not fitted.
    Input (...,lag); output (...,frequency). All n-1 nonzero lags contribute.
    """
    if correlation.numel()==0:
        return correlation.new_empty((*correlation.shape[:-1],correlation.shape[-1]//2+1))
    weighted = correlation*taper
    folded = torch.cat((weighted[..., :1], weighted[..., 1:]+weighted[..., 1:].flip(-1)), -1)
    power = torch.fft.rfft(folded, dim=-1).real/fs
    scale = torch.full_like(power, 2.)
    scale[..., 0] = 1.
    if correlation.shape[-1] % 2 == 0:
        scale[..., -1] = 1.
    return power*scale


class ResonanceModel(torch.nn.Module):
    """Two-parameter dispersion families, damping, shared real mixing, OU floor."""
    def __init__(self, family_parameters, counts, dimension, rank=1,
                 background_hz=(30., 300., 3000.), fs=14400., nperseg=2048):
        super().__init__()
        if dimension < 1 or rank < 1 or fs <= 0 or nperseg < 4:
            raise ValueError('Invalid dimension, rank, sampling rate or window')
        parameters = np.asarray(family_parameters, float).reshape(-1, 2)
        if len(parameters) != len(counts) or np.any(parameters <= 0) or any(c < 1 for c in counts):
            raise ValueError('Positive (a,b) and count required for each family')
        if np.any(np.asarray(background_hz) <= 0):
            raise ValueError('Background corner frequencies must be positive')
        self.fs, self.nperseg = float(fs), int(nperseg)
        self.counts, self.rank, self.dimension = list(counts), int(rank), int(dimension)
        self.log_family = torch.nn.Parameter(torch.tensor(np.log(parameters), dtype=torch.float64))
        modes = sum(counts)
        self.log_width = torch.nn.Parameter(torch.full((modes,), np.log(25.), dtype=torch.float64))
        self.mixing = torch.nn.Parameter(torch.zeros(modes, dimension, rank, dtype=torch.float64))
        self.background = torch.nn.Parameter(torch.stack([
            torch.eye(dimension, dtype=torch.float64)*.15 for _ in background_hz]))
        self.register_buffer('background_hz', torch.tensor(background_hz, dtype=torch.float64))
        self.register_buffer('lags', torch.arange(nperseg, dtype=torch.float64)/fs)
        self.register_buffer('taper', torch.tensor(taper_correlation(nperseg), dtype=torch.float64))

    def frequencies(self):
        ab = self.log_family.exp()
        if not self.counts:
            return self.log_family.new_empty(0)
        return torch.cat([(a*torch.arange(n, dtype=ab.dtype, device=ab.device)+b)**1.5
                          for (a,b), n in zip(ab, self.counts)])

    def damping(self):
        # Keep every pole underdamped; broad widths may approach critical damping.
        return torch.minimum(np.pi*self.log_width.exp(), .95*2*np.pi*self.frequencies())

    def correlations(self, times=None):
        t = self.lags if times is None else times
        omega = 2*np.pi*self.frequencies()
        gamma = self.damping()
        wd = (omega**2-gamma**2).sqrt()
        osc = torch.exp(-gamma[:,None]*t)*(torch.cos(wd[:,None]*t)+
                    gamma[:,None]/wd[:,None]*torch.sin(wd[:,None]*t))
        bg = torch.exp(-2*np.pi*self.background_hz[:,None]*t)
        return osc, bg

    def spectral_matrix(self):
        osc, bg = self.correlations()
        osc_psd = expected_welch(osc, self.fs, self.taper)
        bg_psd = expected_welch(bg, self.fs, self.taper)
        return (torch.einsum('mf,mir,mjr->fij', osc_psd, self.mixing, self.mixing)
                +torch.einsum('bf,bir,bjr->fij', bg_psd, self.background, self.background))

    def covariance(self, times):
        osc, bg = self.correlations(torch.as_tensor(times, dtype=torch.float64))
        return (torch.einsum('mt,mir,mjr->tij', osc, self.mixing, self.mixing)
                +torch.einsum('bt,bir,bjr->tij', bg, self.background, self.background))

    def save(self, path, mean, scale, **metadata):
        np.savez_compressed(path, frequencies_hz=self.frequencies().detach().numpy(),
            damping_per_second=self.damping().detach().numpy(), mixing=self.mixing.detach().numpy(),
            background=self.background.detach().numpy(), background_hz=self.background_hz.numpy(),
            mean=mean, scale=scale, fs=self.fs, family_ab=self.log_family.exp().detach().numpy(),
            counts=self.counts, **metadata)


def oscillator_transition(frequency_hz, gamma, dt):
    """Exact transition and noise factor in unit-covariance (q,p/omega) units."""
    if frequency_hz <= 0 or gamma <= 0 or dt <= 0:
        raise ValueError('Frequency, damping and dt must be positive')
    omega = 2*np.pi*frequency_hz
    transition = expm(np.array([[0., omega], [-omega, -2*gamma]])*dt)
    noise = np.eye(2)-transition@transition.T
    values, vectors = np.linalg.eigh((noise+noise.T)/2)
    if values.min() < -1e-10:
        raise ValueError('Invalid exact transition covariance')
    return transition, vectors@np.diag(np.sqrt(np.maximum(values, 0.)))


def simulate(saved, steps, paths=16, seed=20261023):
    """Stationary unconditional rollout; no validation-conditioned state fitting."""
    if steps < 1 or paths < 1:
        raise ValueError('Positive steps and path count required')
    rng = np.random.default_rng(seed)
    mix = saved['mixing']
    m, d, rank = mix.shape
    transition, factor = [], []
    for f,g in zip(saved['frequencies_hz'], saved['damping_per_second']):
        a,l = oscillator_transition(f, g, 1/float(saved['fs']))
        transition.append(a); factor.append(l)
    transition = np.asarray(transition).reshape(m,2,2)
    factor = np.asarray(factor).reshape(m,2,2)
    state = rng.normal(size=(paths,m,rank,2))
    bg = saved['background']
    background_state = rng.normal(size=(paths, len(bg), d))
    decay = np.exp(-2*np.pi*saved['background_hz']/float(saved['fs']))
    output = np.empty((paths,steps,d))
    for t in range(steps):
        output[:,t] = (np.einsum('pmr,mdr->pd', state[...,0], mix)
                       +np.einsum('pbr,bdr->pd', background_state, bg))
        state = (np.einsum('mij,pmrj->pmri', transition, state)
                 +np.einsum('mij,pmrj->pmri', factor, rng.normal(size=state.shape)))
        background_state = (decay[None,:,None]*background_state
            +np.sqrt(1-decay**2)[None,:,None]*rng.normal(size=background_state.shape))
    return saved['mean']+output*saved['scale']
