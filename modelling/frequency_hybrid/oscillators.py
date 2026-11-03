"""Observed-coordinate vector oscillators with exact stationary transitions."""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import correlate, get_window

from modelling.latent_oscillators.model import oscillator_transition


def oscillator_psd(frequency_hz, gamma, fs, nperseg):
    # Unit displacement covariance; exactly sampled covariance and Hann bias.
    omega = 2*np.pi*frequency_hz
    if not 0 < gamma < omega:
        raise ValueError('Positive underdamped oscillator required')
    t = np.arange(nperseg)/fs
    wd = np.sqrt(omega**2-gamma**2)
    rho = np.exp(-gamma*t)*(np.cos(wd*t)+gamma/wd*np.sin(wd*t))
    w = get_window('hann', nperseg)
    taper = correlate(w, w, mode='full', method='fft')[nperseg-1:]/(w@w)
    a = rho*taper
    folded = np.r_[a[:1], a[1:]+a[1:][::-1]]
    power = np.fft.rfft(folded).real/fs
    power[1:-1 if nperseg % 2 == 0 else None] *= 2
    return np.maximum(power, 0.)


def fit_band(records, frequency, spectral_matrix, fs, edges, shrinkage=.02):
    # Centre each repetition separately; no held-out statistics enter the fit.
    covariance = np.mean([np.cov(x, rowvar=False, bias=True) for x in records], axis=0)
    covariance = (1-shrinkage)*covariance + shrinkage*np.diag(np.diag(covariance))
    diagonal = np.diagonal(spectral_matrix, axis1=-2, axis2=-1).real
    target = np.mean(diagonal/np.maximum(np.diag(covariance),1e-30),axis=-1)
    low, high = edges
    width = high-low
    mask = (frequency >= max(1., low-width)) & (frequency <= high+width)
    nperseg = 2*(len(frequency)-1)
    scale = max(float(target[mask].max()), 1e-30)
    peak = float(frequency[np.flatnonzero(mask)[np.argmax(target[mask])]])
    def residual(params):
        natural, gamma = params[0], np.exp(params[1])
        return (oscillator_psd(natural, gamma, fs, nperseg)[mask]-target[mask])/scale
    result = least_squares(residual, [np.clip(peak,low+.1,high-.1), np.log(np.pi*width/3)],
        bounds=([low, np.log(.5)], [high, np.log(min(np.pi*width,2*np.pi*low*.9))]))
    if not result.success:
        raise RuntimeError(result.message)
    f, gamma = float(result.x[0]), float(np.exp(result.x[1]))
    ev, vectors = np.linalg.eigh(covariance)
    factor = vectors*np.sqrt(np.maximum(ev,0.))
    return dict(frequency_hz=f, gamma_per_second=gamma, covariance=covariance,
        factor=factor, edges_hz=np.asarray(edges), fs=float(fs), shrinkage=shrinkage,
        spectral_fit_relative_rmse=float(np.sqrt(np.mean(result.fun**2))),
        approximate_fwhm_hz=gamma/np.pi)


def simulate_band(model, steps, paths=8, seed=0):
    """q and u=p/omega share covariance C; no inferred hidden mixing state."""
    rng=np.random.default_rng(seed)
    c=model['factor']; d=len(c)
    a,l=oscillator_transition(float(model['frequency_hz']),float(model['gamma_per_second']),
                              1/float(model['fs']))
    state=rng.normal(size=(paths,d,2))
    output=np.empty((paths,steps,d))
    for i in range(steps):
        output[:,i]=state[:,:,0]@c.T
        state=state@a.T+rng.normal(size=state.shape)@l.T
    return output
