"""Observed-coordinate second-order spectral model, without latent readout noise.

z=Tq, dz=v dt, dv=(-Kz-Dv)dt+GdW. Internal time is milliseconds.
The exact sampled stationary covariance supplies expected Hann Welch spectra;
no trajectory simulation or long unrolled gradient is needed.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import taper_correlation


class ObservedSpectralModel(torch.nn.Module):
    def __init__(self, peaks, basis, condition_cap=1., fs=14400., nperseg=4096):
        super().__init__()
        basis = np.asarray(basis, dtype=float)
        d = len(basis)
        if (basis.shape != (d, d) or not np.allclose(basis.T @ basis, np.eye(d), atol=1e-8)
                or not 0 < len(peaks) <= d or condition_cap < 1 or fs <= 0 or nperseg < 4):
            raise ValueError('Require orthogonal square basis, 0 < peaks <= dimension and cap >= 1')
        self.d, self.m = d, len(peaks)
        self.cap, self.fs, self.nperseg = float(condition_cap), float(fs), int(nperseg)
        self.register_buffer('basis', torch.tensor(basis, dtype=torch.float64))
        self.register_buffer('peaks', torch.tensor(peaks, dtype=torch.float64))
        self.register_buffer('times', torch.arange(nperseg, dtype=torch.float64)*1000/fs)
        self.register_buffer('taper', torch.tensor(taper_correlation(nperseg)))
        self.rotation = torch.nn.Parameter(torch.zeros(d, d, dtype=torch.float64))
        self.stretch = torch.nn.Parameter(torch.zeros(d, d, dtype=torch.float64), requires_grad=self.cap > 1)
        self.peak_shift = torch.nn.Parameter(torch.zeros(self.m, dtype=torch.float64))
        self.width = torch.nn.Parameter(torch.full((self.m,), -2., dtype=torch.float64))
        self.broad_frequency = torch.nn.Parameter(torch.tensor(np.log(np.geomspace(40, 4000, d-self.m))))
        self.broad_damping = torch.nn.Parameter(torch.zeros(d-self.m, dtype=torch.float64))
        # Full, real white acceleration covariance, identical family for every T cap.
        self.force = torch.nn.Parameter(torch.eye(d, dtype=torch.float64)*.6)

    def transform(self):
        o = self.basis @ torch.matrix_exp(self.rotation-self.rotation.T)
        t = o @ torch.matrix_exp((self.stretch+self.stretch.T)/2)
        return t / torch.linalg.vector_norm(t, dim=0)

    @torch.no_grad()
    def project(self):
        # Fix the unit-column gauge, enforce the actual normalized T condition cap.
        self.broad_frequency.clamp_(np.log(30), np.log(6000))
        s = (self.stretch+self.stretch.T)/2
        ev, u = torch.linalg.eigh(s)
        bound = .5*np.log(self.cap)
        self.stretch.copy_((u*ev.clamp(-bound, bound)) @ u.T)
        for _ in range(40):
            if torch.linalg.cond(self.transform()) <= self.cap*(1+1e-10):
                return
            self.stretch.mul_(.8)
        raise RuntimeError('Condition projection failed')

    def poles(self):
        peak = self.peaks + 8*torch.tanh(self.peak_shift)
        gamma = 2*np.pi*(1+59*torch.sigmoid(self.width))/1000
        # Peak of an isolated, white-acceleration-driven displacement oscillator.
        omega = torch.sqrt((2*np.pi*peak/1000)**2+2*gamma**2)
        broad = 2*np.pi*self.broad_frequency.exp()/1000
        return (torch.cat((omega, broad)),
                torch.cat((gamma, broad*(.8+.19*torch.sigmoid(self.broad_damping)))))

    def force_factor(self):
        return self.force

    def modal_statistics(self):
        w, g = self.poles()
        zero, one = torch.zeros_like(w), torch.ones_like(w)
        a = torch.stack((zero, one, -w*w, -2*g), -1).reshape(self.d, 2, 2)
        # Per-mode stationary q variance = 1 for a diagonal unit force factor.
        b = torch.sqrt(4*g*w*w)[:, None]*self.force_factor()
        q = b @ b.T
        eye = torch.eye(2, dtype=w.dtype, device=w.device)
        # Row-vectorized pair Lyapunov: Ai Pij + Pij Aj^T + Qij = 0.
        operator = (torch.einsum('iac,be->iabce', a, eye)[:, None]
                    + torch.einsum('ac,jbe->jabce', eye, a)[None]).reshape(self.d, self.d, 4, 4)
        rhs = torch.stack((q*0, q*0, q*0, -q), -1)
        p = torch.linalg.solve(operator, rhs.unsqueeze(-1)).squeeze(-1).reshape(self.d, self.d, 2, 2)
        return w, g, q, p

    def spectrum(self, indices=None):
        w, g, _, p = self.modal_statistics()
        wd = torch.sqrt(w*w-g*g)
        t = self.times[None]
        decay = torch.exp(-g[:, None]*t)
        sn = torch.sin(wd[:, None]*t)/wd[:, None]
        aa = decay*(torch.cos(wd[:, None]*t)+g[:, None]*sn)
        bb = decay*sn
        af = torch.fft.rfft(aa*self.taper, dim=-1).T
        bf = torch.fft.rfft(bb*self.taper, dim=-1).T
        factors = torch.full((len(af),), 2., dtype=w.dtype, device=w.device)
        factors[0] = 1
        if self.nperseg % 2 == 0:
            factors[-1] = 1
        if indices is not None:
            af, bf, factors = af[indices], bf[indices], factors[indices]
        positive = af[:, :, None]*p[None, :, :, 0, 0]+bf[:, :, None]*p[None, :, :, 1, 0]
        s = (positive+positive.mH-p[None, :, :, 0, 0])/self.fs
        tt = self.transform().to(s.dtype)
        return (tt @ s @ tt.T)*factors[:, None, None]

    def operators(self):
        t = self.transform()
        w, g, q, _ = self.modal_statistics()
        inv = torch.linalg.solve(t, torch.eye(self.d, dtype=t.dtype, device=t.device))
        return t, (t*(w*w)) @ inv*1e6, (t*(2*g)) @ inv*1000, t @ q @ t.T*1e9


def spectral_loss(prediction, reference, cross_weight=2.):
    diag = reference.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-15)
    pd = prediction.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-15)
    log = (pd.log()-diag.log()).square().mean()
    error = (prediction-reference)/torch.sqrt(diag[:, :, None]*diag[:, None, :])
    off = ~torch.eye(len(diag[0]), dtype=torch.bool, device=diag.device)
    cross = error[:, off].abs().square().mean()
    return log+cross_weight*cross
