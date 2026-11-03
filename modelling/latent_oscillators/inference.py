"""Exact, frozen linear-Gaussian inference for the saved oscillator model.

State order is (q,u) for each pole/copy, followed by each vector OU state.
All states have unit stationary covariance. Observations are in the saved
training normalization. There is no added measurement noise: the colored
background remains part of the dynamical state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from .model import oscillator_transition


def symmetric(a):
    return (a + a.T) / 2


@dataclass
class StateSpace:
    transition: np.ndarray
    noise: np.ndarray
    readout: np.ndarray
    oscillator_readout: np.ndarray
    q_indices: np.ndarray
    u_indices: np.ndarray
    fs: float

    @classmethod
    def from_saved(cls, saved):
        mix = np.asarray(saved['mixing'])
        poles, dimension, rank = mix.shape
        background = np.asarray(saved['background'])
        oscillator_size = 2 * poles * rank
        size = oscillator_size + len(background) * dimension
        f = np.zeros((size, size))
        h = np.zeros((dimension, size))
        fs = float(saved['fs'])
        for pole, (frequency, damping) in enumerate(zip(
                saved['frequencies_hz'], saved['damping_per_second'])):
            block, _ = oscillator_transition(frequency, damping, 1 / fs)
            for copy in range(rank):
                start = 2 * (pole * rank + copy)
                f[start:start+2, start:start+2] = block
                h[:, start] = mix[pole, :, copy]
        h_osc = h.copy()
        for k, (factor, frequency) in enumerate(zip(background, saved['background_hz'])):
            start = oscillator_size + k * dimension
            f[start:start+dimension, start:start+dimension] = (
                np.eye(dimension) * np.exp(-2*np.pi*frequency/fs))
            h[:, start:start+dimension] = factor
        return cls(f, symmetric(np.eye(size)-f@f.T), h, h_osc,
                   np.arange(0, oscillator_size, 2), np.arange(1, oscillator_size, 2), fs)


@dataclass
class FilterSchedule:
    gains: np.ndarray
    innovation_factors: np.ndarray
    posterior: dict[int, np.ndarray]


def filter_schedule(system, steps, checkpoints=()):
    """Data-independent Riccati recursion, initialized with stationary P=I.

    Index zero assimilates y[0] without a preceding transition. Checkpoints
    count observations (not elapsed steps). No steady-gain approximation.
    """
    if steps < 1:
        raise ValueError('At least one observation is required')
    f, q, h = system.transition, system.noise, system.readout
    size, dimension = h.shape[1], h.shape[0]
    gains = np.empty((steps, size, dimension))
    factors = np.empty((steps, dimension, dimension))
    posterior = {}
    wanted = set(checkpoints) | {steps}
    p = np.eye(size)
    for t in range(steps):
        if t:
            p = symmetric(f@p@f.T+q)
        ph = p@h.T
        factors[t] = cholesky(symmetric(h@ph), lower=True)
        gains[t] = cho_solve((factors[t], True), ph.T).T
        p = symmetric(p-gains[t]@ph.T)
        if t+1 in wanted:
            posterior[t+1] = p.copy()
    return FilterSchedule(gains, factors, posterior)


def filter_observations(system, observations, schedule, keep_path=False):
    """Filter independent batches; each starts at stationary mean zero.

    Returns terminal means and, optionally, filtered means and *pre-update*
    whitened innovations. Only the supplied observation prefix is consumed.
    """
    y = np.asarray(observations)
    if y.ndim != 3 or y.shape[1] > len(schedule.gains):
        raise ValueError('Expected (batch,time,coordinate) within schedule length')
    m = np.zeros((len(y), system.readout.shape[1]))
    path = np.empty((len(y), y.shape[1], m.shape[1])) if keep_path else None
    white = np.empty_like(y, dtype=float) if keep_path else None
    for t in range(y.shape[1]):
        if t:
            m = m@system.transition.T
        innovation = y[:, t]-m@system.readout.T
        if keep_path:
            white[:, t] = solve_triangular(schedule.innovation_factors[t],
                                           innovation.T, lower=True).T
        m += innovation@schedule.gains[t].T
        if keep_path:
            path[:, t] = m
    return m, path, white


def forecast(system, mean, covariance, steps):
    """Conditional observed moments after steps, without further assimilation."""
    if steps < 1:
        raise ValueError('Forecast horizon must be positive')
    f = np.linalg.matrix_power(system.transition, steps)
    p = symmetric(np.eye(len(f)) + f@(covariance-np.eye(len(f)))@f.T)
    return mean@f.T@system.readout.T, symmetric(system.readout@p@system.readout.T)


def smooth_endpoint(system, mean, covariance, future):
    """Noncausal fixed-point smoother of the endpoint using later observations.

    This is a diagnostic upper bound on information, never a forecast input.
    Tracks the cross covariance of the endpoint with the current filter state.
    """
    f, q, h = system.transition, system.noise, system.readout
    endpoint_mean, endpoint_cov = mean.copy(), covariance.copy()
    current_mean, current_cov = mean.copy(), covariance.copy()
    cross = covariance.copy()
    for t in range(future.shape[1]):
        current_mean = current_mean@f.T
        current_cov = symmetric(f@current_cov@f.T+q)
        cross = cross@f.T
        ph = current_cov@h.T
        factor = cholesky(symmetric(h@ph), lower=True)
        gain = cho_solve((factor, True), ph.T).T
        cross_h = cross@h.T
        endpoint_gain = cho_solve((factor, True), cross_h.T).T
        innovation = future[:, t]-current_mean@h.T
        endpoint_mean += innovation@endpoint_gain.T
        endpoint_cov = symmetric(endpoint_cov-endpoint_gain@cross_h.T)
        cross -= endpoint_gain@ph.T
        current_mean += innovation@gain.T
        current_cov = symmetric(current_cov-gain@ph.T)
    return endpoint_mean, endpoint_cov


def simulate_observations(system, steps, paths, endpoint, seed):
    """Stationary independent paths with the true state at one selected time."""
    rng = np.random.default_rng(seed)
    values, vectors = np.linalg.eigh(system.noise)
    if values.min() < -1e-10 or not 0 <= endpoint < steps:
        raise ValueError('Invalid process covariance or endpoint')
    factor = vectors*np.sqrt(np.maximum(values, 0))
    state = rng.normal(size=(paths, len(system.transition)))
    y = np.empty((paths, steps, len(system.readout)))
    truth = None
    for t in range(steps):
        y[:, t] = state@system.readout.T
        if t == endpoint:
            truth = state.copy()
        if t+1 < steps:
            state = state@system.transition.T + rng.normal(size=state.shape)@factor.T
    return y, truth
