"""Check a fully resonant imported core before training any neural residual.

Uses the native Neural-SDE semi-implicit position/velocity update. The noisy
control has constant diagonal noise in standardized observed coordinates, with
one common amplitude chosen to give unit mean stationary position variance.
It does not import the spectral fit's forcing covariance or background.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import welch


def core_matrix(k, damping, dt):
    d = len(k)
    eye = np.eye(d)
    return np.block([[eye-dt**2*k, eye-dt*damping],
                     [-dt**2*k, eye-dt*damping]])


def stationary_noise(core):
    """Normalize a constant diagonal velocity increment factor, not a fitted PSD."""
    d = len(core)//2
    if np.max(abs(np.linalg.eigvals(core))) >= 1:
        raise ValueError('Fixed native-rate core is not strictly stable')
    # New velocity is used to advance position, so both receive the same noise.
    b = np.vstack((np.eye(d), np.eye(d)))
    p = solve_discrete_lyapunov(core, b@b.T)
    p = (p+p.T)/2
    if not np.isfinite(p).all() or np.linalg.eigvalsh(p).min() <= 0:
        raise ValueError('Invalid stationary covariance')
    sigma = 1/np.sqrt(np.trace(p[:d,:d])/d)
    p *= sigma**2
    error = np.linalg.norm(p-core@p@core.T-sigma**2*b@b.T)/np.linalg.norm(p)
    if error > 1e-8:
        raise ValueError(f'Discrete Lyapunov residual too large: {error:g}')
    return sigma, p, error


def simulate(k, damping, dt, initial, sigma, steps, rng):
    """Per-frame velocity update followed by position update, in float64."""
    d = len(k)
    x, u = np.array(initial[:d],copy=True), np.array(initial[d:],copy=True)
    positions = np.empty((steps+1,d)); positions[0] = x
    for j in range(steps):
        u = u-dt**2*(k@x)-dt*(damping@u)
        if sigma:
            u += sigma*rng.standard_normal(d)
        x = x+u
        positions[j+1] = x
    if not np.isfinite(positions).all():
        raise RuntimeError('Core rollout became nonfinite; no clipping applied')
    return positions, u


def discrete_psd(t, omega, gamma, dt, sigma, frequency):
    """Exact discrete stationary density; Welch smoothing is not included."""
    angle = 2*np.pi*np.asarray(frequency)*dt
    denominator = (np.exp(1j*angle[:,None])-(2-2*dt*gamma-dt**2*omega**2)
                   +(1-2*dt*gamma)*np.exp(-1j*angle[:,None]))
    h = np.einsum('ij,fj,jk->fik',t,1/denominator,np.linalg.inv(t))
    psd = sigma**2*dt*np.sum(abs(h)**2,axis=-1)
    factors = np.full(len(angle),2.)
    factors[np.isclose(angle,0)|np.isclose(angle,np.pi)] = 1.
    return factors[:,None]*psd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=90000)
    parser.add_argument('--nperseg',type=int,default=8192)
    parser.add_argument('--seed',type=int,default=17)
    args = parser.parse_args(argv)
    if args.steps < 8 or not 4 <= args.nperseg <= args.steps:
        parser.error('Require steps >= nperseg >= 4 and steps >= 8')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty')
    protocol_path = args.artifact.parent.parent/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    source = protocol['background_protocol']['source_protocol']['source_protocol']['source_provenance']
    dt = 1/source['single_rollout']['sampling_frequency_hz']
    with np.load(args.artifact,allow_pickle=False) as saved:
        t,k,damping,omega,gamma,mean,scale,peaks = [np.array(saved[name]) for name in
            ('T','K','D','omega_per_second','gamma_per_second','mean','scale','prescribed_peaks_hz')]
    d = len(k)
    if len(peaks) != d:
        raise ValueError('This check requires one resonance per observed position coordinate; refit with --rank 19')
    if (k.shape != (d,d) or damping.shape != (d,d) or t.shape != (d,d)
            or omega.shape != (d,) or gamma.shape != (d,)
            or not all(np.isfinite(a).all() for a in (t,k,damping,omega,gamma,mean,scale))
            or np.any(gamma <= 0) or np.any(scale <= 0)):
        raise ValueError('Invalid oscillator artifact')
    inverse = np.linalg.inv(t)
    np.testing.assert_allclose(k,(t*omega**2)@inverse,rtol=1e-6,atol=1e-5)
    np.testing.assert_allclose(damping,(t*(2*gamma))@inverse,rtol=1e-6,atol=1e-5)
    core = core_matrix(k,damping,dt)
    sigma,p,error = stationary_noise(core)
    rng = np.random.default_rng(args.seed)
    # Same nonzero stationary initial state for noisy and noise-free controls.
    initial = np.linalg.cholesky(p)@rng.standard_normal(2*d)
    deterministic,du = simulate(k,damping,dt,initial,0.,args.steps,rng)
    noisy,nu = simulate(k,damping,dt,initial,sigma,args.steps,rng)
    frequency,psd = welch(noisy,fs=1/dt,nperseg=args.nperseg,axis=0)
    expected = discrete_psd(t,omega,gamma,dt,sigma,frequency)
    _,transient_psd = welch(deterministic,fs=1/dt,nperseg=args.nperseg,axis=0)
    empirical = np.var(noisy,axis=0)
    expected_var = np.diag(p)[:d]
    report = dict(artifact=str(args.artifact.resolve()),artifact_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        coordinate_space='standardized fitted observed positions; u is displacement per native frame',
        update='u_next = u - dt^2 K z - dt D u + sigma xi; z_next = z + u_next',
        neural_residual=False,background_imported=False,fitted_forcing_imported=False,
        dimension=d,resonances=len(peaks),completion_directions=0,dt=dt,steps=args.steps,
        nperseg=args.nperseg,seed=args.seed,duration_seconds=args.steps*dt,
        initial_state='same stationary Gaussian sample for both controls',
        condition_standardized=float(np.linalg.cond(t)),
        discrete_spectral_radius=float(max(abs(np.linalg.eigvals(core)))),
        continuous_max_eigenvalue_real=float(np.linalg.eigvals(np.block([[np.zeros_like(k),np.eye(d)],[-k,-damping]])).real.max()),
        diagonal_noise_std_per_native_velocity_step=float(sigma),
        noise_normalization='one scalar makes mean theoretical standardized position variance 1; not fitted to observed spectra',
        expected_position_variance=expected_var.tolist(),empirical_position_variance=empirical.tolist(),
        mean_empirical_to_expected_variance=float(empirical.sum()/expected_var.sum()),
        lyapunov_relative_residual=float(error),
        deterministic_initial_state_norm=float(np.linalg.norm(initial)),
        deterministic_final_state_norm=float(np.linalg.norm(np.r_[deterministic[-1],du])),
        deterministic_max_position_norm=float(np.linalg.norm(deterministic,axis=1).max()),
        stochastic_max_position_norm=float(np.linalg.norm(noisy,axis=1).max()),
        finite=True,precision='float64',
        limitation='Linear core check only. No observational initial-state comparison, neural stability claim, or fitted PSD-amplitude reproduction.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'core_rollouts.npz',deterministic=deterministic,stochastic=noisy,
        initial_state=initial,dt=dt,fit_mean=mean,fit_scale=scale,frequency_hz=frequency,
        stochastic_psd=psd,deterministic_transient_psd=transient_psd,expected_discrete_psd=expected)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for j,ax in enumerate(axes.flat):
        if j >= d:
            ax.set_visible(False);continue
        ax.plot(frequency,psd[:,j],label='Noisy rollout Welch',lw=1.)
        ax.plot(frequency,expected[:,j],label='Discrete stationary theory',lw=1.)
        for peak in peaks:
            ax.axvline(peak,color='.7',lw=.4)
        ax.set(xlim=(50,1500),yscale='log',xlabel='Frequency [Hz]',ylabel='PSD [standardized units²/Hz]',
               title='Spatial mean' if j==0 else f'POD {j}')
    axes[0,0].legend();fig.savefig(args.output/'psd.png',dpi=150);plt.close(fig)
    fig,axes = plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
    times = np.arange(args.steps+1)*dt
    for ax,values,label in zip(axes,(deterministic,noisy),('Zero diffusion','Constant diagonal diffusion')):
        ax.plot(times,np.linalg.norm(values,axis=1),lw=.7)
        ax.set(ylabel='Position norm',title=label)
    axes[-1].set_xlabel('Time [s]');fig.savefig(args.output/'trajectories.png',dpi=150);plt.close(fig)
    (args.output/'README.md').write_text(
        '# Fixed oscillator only: native-rate check\n\n'
        '[Trajectories](trajectories.png) · [Spectra](psd.png) · [Numbers](report.json)\n\n'
        f'{d} resonances in {d} observed coordinates; no completion direction and no neural residual.\n\n'
        f'Discrete spectral radius: **{report["discrete_spectral_radius"]:.9f}**. '
        f'Both {args.steps:,}-step trajectories are finite. '
        f'Deterministic augmented-state norm: {report["deterministic_initial_state_norm"]:.4g} → '
        f'{report["deterministic_final_state_norm"]:.4g}. '
        f'Noisy empirical/theoretical total position variance: {report["mean_empirical_to_expected_variance"]:.3f}.\n\n'
        'Noise is constant diagonal in standardized observed coordinates; it is also diagonal after physical coordinate rescaling. '
        'Its single amplitude is normalized analytically to unit mean stationary position variance. '
        'No fitted forcing covariance or broadband background is imported. '
        'Both controls start from the same stationary Gaussian sample. The noise-free spectrum is a transient, not a stationary PSD.\n\n'
        'PSD plots show native-step stationary theory and one finite noisy realization, with prescribed frequencies marked. '
        'Not every oscillator must appear as a distinct maximum in every coordinate. Welch smoothing and overlapping resonances matter. '
        'This checks the fixed linear mechanism, not agreement with experimental amplitudes or stability of the future neural residual. '
        'The recurrence is evaluated in float64; training/standard rollout commonly uses float32.\n')
    print(json.dumps({key:report[key] for key in ('discrete_spectral_radius','deterministic_final_state_norm',
        'mean_empirical_to_expected_variance','finite')},indent=2))
    print(args.output/'README.md')


if __name__ == '__main__':
    main()
