"""Tools for analyzing and classifying turbulent capillary waves.

Authors: Jeremy Orosco, William Connacher, and James Friend
Date: 04/08/2022
Notes:
    This tool set corresponds to the forthcoming publication
    
        Identification of weakly- to strongly-turbulent three-wave processes in a micro-scale system
        
    See that article and the associated supplemental materials for more information.

Copyright 2022 Jeremy Orosco
This work is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
import numpy as np
from scipy.optimize import root
from scipy.interpolate import interp1d
import pywt
from scipy.ndimage import uniform_filter1d

# system properties
surf_ten = 0.0728   # surface tension [N/m]
density = 998       # fluid density [kg/m**3]
h = 725/10**6       # fluid volume (or basin) depth [m]
L = 9.525/10**3     # fluid volume (or basin) width [m]
km = np.pi/L        # smallest possible wavemode wavenumber given homogeneous boundary [rad/m]

def wav_bispectrum(zf,t,fs,fb_low,fb_high,scales=None,Ns=2**10,wavelet='cgau1',bico=False):
    """
    Computes the wavelet-based bispectrum of a given time series.

    Parameters
    ----------
    zf : time series of surface height, 1d float array
    t : time, 1d float array
    fs : sampling rate, float scalar
    fb_low: lower bound for output frequency [Hz] array, float scalar
    fb_high: upper bound for output frequency [Hz] array, float scalar
    scales : scales for wavelet transform (if none, autocomputed from f bounds), 1d float array
    Ns : number of scales (default is 1024), integer scalar
    wavelet : wavelet type (default is complex gaussian 'cgau1'), string
    bico : flag to delay freq bound cropping if computing bico (default is False), boolean

    Returns
    -------
    f_out : x-axis and y-axis frequencies, 1d float array
    Bi_spec : bispectrum, 2d complex array
    """
    
    # scales grid
    if scales is None:
        scales = 34500/np.linspace(fb_high,fb_low,Ns)

    # wavelet transform
    spec, f = pywt.cwt(zf,scales,wavelet,1/fs)

    # arguments to spectra, flip f to ascending
    idx_in = np.arange(f.size//2)
    idx_sum = idx_in[:, None] + idx_in[None,:]
    f_out = f[::-1][idx_in]

    # transpose (f,t) -> (t,f), flip for ascending in f
    spec = np.transpose(spec,[1,0])[:,::-1].astype('complex64')

    # bispectrum
    Bi_spec = (spec[:,idx_in,None]*spec[:,None,idx_in]*np.conjugate(spec[:,idx_sum])).astype('complex64')

    # crop to relevant spectrum
    if not bico:
        f_idx = (f_out>=fb_low) & (f_out<=fb_high)
        f_out = f_out[f_idx]
        Bi_spec = Bi_spec[:,f_idx,:]
        Bi_spec = Bi_spec[:,:,f_idx]

    return f_out, Bi_spec


def wav_bicoherence(zf,t,fs,fb_low,fb_high,scales=None,Ns=2**10,wavelet='cgau1'):
    """
    Computes the wavelet-based bicoherence of a given time series.

    Parameters
    ----------
    zf : time series of surface height, 1d float array
    t : time, 1d float array
    fs : sampling rate, float scalar
    fb_low: lower bound for output frequency [Hz] array, float scalar
    fb_high: upper bound for output frequency [Hz] array, float scalar
    scales : scales for wavelet transform (if none, autocomputed from f bounds), 1d float array
    Ns : number of scales (default is 1024), integer scalar
    wavelet : wavelet type (default is complex gaussian 'cgau1'), string

    Returns
    -------
    f_out : x-axis and y-axis frequencies, 1d float array
    Bi_co : bicoherence, 2d float array
    """

    # bispectrum
    f, Bi_spec = wav_bispectrum(zf,t,fs,fb_low,fb_high,scales=scales,Ns=Ns,wavelet=wavelet,bico=True)

    # bicoherence
    bico_den_mag = np.full((t.size,f.size,f.size),np.nan,dtype='float32')
    bico_num = np.abs(np.mean(Bi_spec,axis=0))
    for k in range(t.size):
        bico_den_mag[k,:,:] = np.abs(Bi_spec[k,:,:])
    bico_den = np.mean(bico_den_mag,axis=0)
    Bi_co = bico_num/bico_den

    # crop to relevant spectrum
    f_idx = (f>=fb_low) & (f<=fb_high)
    f = f[f_idx]
    Bi_co = Bi_co[f_idx,:]
    Bi_co = Bi_co[:,f_idx]

    return f, Bi_spec, Bi_co


def check_wav_scales(scales,fs,wavelet='cgau1'):
    """
    Displays the frequencies associated with a given set of scales for a given wavelet type.

    Parameters
    ----------
    scales : scales for wavelet transform, 1d float array
    fs : sampling frequency of the time series, float scalar
    wavelet : wavelet type (default is complex gaussian 'cgau1'), string

    Returns
    -------
    None.
    """

    print(pywt.scale2frequency(wavelet, scales, precision=10)*fs)
    
def get_nrb(fv,zetav,no_strong=True):
    """
    Given a spectral series, returns nonlinear resonance broadening associated with spectra.

    Parameters
    ----------
    fv : frequencies, 1d float array
    zetav : magnitude of properly normalized fourier transform of surface height, 1d float array
    no_strong : flag for including strong turbulence (default is False), boolean

    Returns
    -------
    kv : wavenumbers, 1d float array
    g : nonlinear resonance broadening, 1d float array

    """

    # depth regime bounds
    fs_bnd = 50    # shallow regime frequency bound
    fd_bnd = 5000  # deep regime frequency bound
    
    ks_bnd = 1250  # shallow regime wavenumber bound
    kd_bnd = 23800 # deep regime wavenumber bound

    # get wt regime indices
    kv, disc_idx, meso_idx, kin_idx, str_idx = get_regime_idxs(fv,zetav,no_strong=no_strong)

    # get wt regime bounds
    _, _, dm_bnd, mk_bnd, kst_bnd = get_regime_bounds(fv[0],fv[-1],np.amin(zetav),np.amax(zetav),fvo=fv)

    # amplitude regime bounds
    zds_bnd = zetav[fv>=fs_bnd][0]
    zdd_bnd = zetav[fv<=fd_bnd][-1]
    zks_bnd = zetav[fv>=fs_bnd][0]
    zkd_bnd = zetav[fv<=fd_bnd][-1]

    # resonance broadening functions
    gamma_ds = lambda k, zeta : (k**2*(zeta/10**6)*np.sqrt(surf_ten/(density*h)))/(8*np.pi)
    gamma_dd = lambda k, zeta : gamma_ds(k,zeta)*np.sqrt(k*h)
    gamma_ks = lambda k, zeta : ((k**4*(zeta/10**6)**2)/(64*km**2))*np.sqrt(surf_ten/(density*h**3))
    gamma_kd = lambda k, zeta : gamma_ks(k,zeta)*(k*h)**(3/2)

    # interpolation points
    xks = np.log10(ks_bnd)
    xkd = np.log10(kd_bnd)
    xzds = np.log10(zds_bnd/10**6)
    xzdd = np.log10(zdd_bnd/10**6)
    xzks = np.log10(zks_bnd/10**6)
    xzkd = np.log10(zkd_bnd/10**6)
    yds = np.log10(gamma_ds(ks_bnd,zds_bnd))
    ydd = np.log10(gamma_dd(kd_bnd,zdd_bnd))
    yks = np.log10(gamma_ks(ks_bnd,zks_bnd))
    ykd = np.log10(gamma_kd(kd_bnd,zkd_bnd))

    # discrete problem
    Xd = [[xks**2, xzds**2, xks*xzds, xks, xzds, 1],
         [xkd**2, xzdd**2, xkd*xzdd, xkd, xzdd, 1],
         [2*xks, 0, xzds, 1, 0, 1],
         [2*xkd, 0, xzdd, 1, 0, 1],
         [0,2*xzds,xks,0,1,0],
         [0,2*xzdd,xkd,0,1,0]]
    beta_kds = 2
    beta_kdd = 5/2
    beta_zds = 1
    beta_zdd = 1
    bd = [[yds],
          [ydd],
          [beta_kds],
          [beta_kdd],
          [beta_zds],
          [beta_zdd]]

    # kinetic problem
    Xk = [[xks**2, xzks**2, xks*xzks, xks, xzks, 1],
         [xkd**2, xzkd**2, xkd*xzkd, xkd, xzkd, 1],
         [2*xks, 0, xzks, 1, 0, 1],
         [2*xkd, 0, xzkd, 1, 0, 1],
         [0, 2*xzks, xks, 0, 1, 0],
         [0, 2*xzkd, xkd, 0, 1, 0]]
    beta_kks = 4
    beta_kkd = 11/2
    beta_zks = 2
    beta_zkd = 2
    bk = [[yks],
          [ykd],
          [beta_kks],
          [beta_kkd],
          [beta_zks],
          [beta_zkd]]

    # solve system
    cd = np.linalg.solve(Xd, bd)
    ck = np.linalg.solve(Xk, bk)

    # interpolation functions
    gamma_di = lambda k, zeta: 10**(cd[0]*np.log10(k)**2+cd[1]*np.log10(zeta/10**6)**2\
                                   +cd[2]*np.log10(k)*np.log10(zeta/10**6)+cd[3]*np.log10(k)\
                                   +cd[4]*np.log10(zeta/10**6)+cd[5])
    gamma_ki = lambda k, zeta: 10**(ck[0]*np.log10(k)**2+ck[1]*np.log10(zeta/10**6)**2\
                                   +ck[2]*np.log10(k)*np.log10(zeta/10**6)+ck[3]*np.log10(k)\
                                   +ck[4]*np.log10(zeta/10**6)+ck[5])

    # get discrete and kinetic curves
    s_idx = kv <= ks_bnd
    i_idx = (kv>ks_bnd) & (kv<kd_bnd)
    d_idx = kv >= kd_bnd
    gd = np.full((kv.size,),np.nan)
    gk = np.full((kv.size,),np.nan)
    gd[s_idx], gd[i_idx], gd[d_idx] = gamma_ds(kv[s_idx],zetav[s_idx]),gamma_di(kv[i_idx],zetav[i_idx]),gamma_dd(kv[d_idx],zetav[d_idx])
    gk[s_idx], gk[i_idx], gk[d_idx] = gamma_ks(kv[s_idx],zetav[s_idx]),gamma_ki(kv[i_idx],zetav[i_idx]),gamma_kd(kv[d_idx],zetav[d_idx])

    # interpolate across wt regimes
    mu = 1-(zetav-dm_bnd)/(mk_bnd-dm_bnd)
    gm = mu*gd+(1-mu)*gk

    # build nonlinear resonance broadening
    g = np.full((kv.size,),np.nan)
    g[disc_idx], g[meso_idx], g[kin_idx] = gd[disc_idx], gm[meso_idx], gk[kin_idx]

    return kv, g

def get_regime_idxs(fv, zetav, no_strong=False):
    """
    Given a spectral series, returns indices associated with wt regimes.

    Parameters
    ----------
    fv : frequencies, 1d float array
    zetav : magnitude of properly normalized fourier transform of surface height, 1d float array 
    no_strong : flag for including strong turbulence indices (default is False), boolean

    Returns
    -------
    kv : wavenumbers, 1d float array
    disc_idx : discrete regime indices, 1d integer array
    meso_idx : mesoscopic regime indices, 1d integer array
    kin_idx : kinetic regime indices, 1d integer array
    str_idx : strong regime indices, 1d integer array
    """

    # get regime bounds
    fv, kv, dm_bnd, mk_bnd, ks_bnd = get_regime_bounds(fv[0], fv[-1], zetav[0], zetav[-1], fvo=fv)

    # get indices
    disc_idx = zetav <= dm_bnd
    meso_idx = (zetav > dm_bnd) & (zetav < mk_bnd)
    
    if no_strong:
        kin_idx = zetav >= mk_bnd
        str_idx = zetav < -1
    else:
        kin_idx = (zetav >= mk_bnd) & (zetav <= ks_bnd)
        str_idx = zetav > ks_bnd

    return kv, disc_idx, meso_idx, kin_idx, str_idx


def get_regime_bounds(fi, ff, zetai, zetaf, fvo=None, theta=5*np.pi, Nf=120):
    """
    Given a system, determine the amplitude bounds associated with wt regimes.

    Parameters
    ----------
    fi : initial frequency, float scalar
    ff : terminal frequency, float scalar
    zetai : lowest amplitude, float scalar
    zetaf : highest amplitude, float scalar
    fvo : frequencies (default is None), 1d float array
    theta : >> definition (see supplemental information, default is 5*np.pi), float scalar
    Nf : number of frequencies to sample (default is 120), integer scalar

    Returns
    -------
    fv : frequencies, 1d float array
    kv : wavenumbers, 1d float array
    dm_bnd: discrete-mesoscopic bound, 1d float array
    mk_bnd: mesoscopic-kinetic bound, 1d float array
    ks_bnd: kinetic-strog bound, 1d float array
    """

    # regime bounds
    fs_bnd = 50
    fd_bnd = 5000

    # minimum interpolation grid
    fgi = 10
    fgf = 40000
    zetagi = 0.01
    zetagf = 4000
    if fi < fgi:
        fgi = fi
    if ff > fgf:
        fgf = ff
    if zetai < zetagi:
        zetagi = zetai
    if zetaf > zetagf:
        zetagf = zetaf

    # frequency grid (to ensure smooth interpolation)
    frev = np.logspace(np.log10(fgf), np.log10(fgi), Nf, dtype='float64')
    fv = frev[::-1] # ascending order
    
    kv = get_wavenumbers(fv) # frequency -> wavenumber

    # system parameters
    L = 9.525/10**3   # characteristic length (annulus diameter)
    h = 725/10**6     # fluid height (thickness)
    # h = 690/10**6     # thickness
    km = np.pi/L

    # dispersion relation and metrics
    # Shallow regime amplitude bounds for discrete, mesoscopic, strong
    zDs = lambda k : (16*np.pi*h*km)/(k*theta)
    zLs = lambda k : np.sqrt((128*h**2*km**3*theta)/(k**3))
    zSs = lambda k : np.sqrt((128*h**2*km**3*theta**2.45)/(k**3))
    
    # Deep regime amplitude bounds
    zDd = lambda k : (12*np.pi*km)/(k**2*theta)
    zLd = lambda k : np.sqrt((96*km**3*theta)/(k**5))
    zSd = lambda k : np.sqrt((96*km**3*theta**2.45)/(k**5))

    # define domain
    ks_bnd = get_wavenumbers(np.asarray([fs_bnd, fs_bnd]))[0]
    kd_bnd = get_wavenumbers(np.asarray([fd_bnd, fd_bnd]))[0]
    xs = np.log10(ks_bnd)
    xd = np.log10(kd_bnd)
    X = [[xs**3, xs**2, xs, 1],
         [xd**3, xd**2, xd, 1],
         [3*xs**2, 2*xs, 1, 0],
         [3*xd**2, 2*xd, 1, 0]]

    # get shallow y-axis value
    yds = np.log10(zDs(ks_bnd))
    yks = np.log10(zLs(ks_bnd))
    yss = np.log10(zSs(ks_bnd))

    # get deep y-axis value
    ydd = np.log10(zDd(kd_bnd))
    ykd = np.log10(zLd(kd_bnd))
    ysd = np.log10(zSd(kd_bnd))

    # discrete bound conditions
    gammads = 1
    gammadd = 2
    bd = [[yds],
         [ydd],
         [-gammads],
         [-gammadd]]

    # kinetic bound conditions
    gammaks = 1.5
    gammakd = 2.5
    bk = [[yks],
         [ykd],
         [-gammaks],
         [-gammakd]]

    # strong bound conditions
    bs = [[yss],
         [ysd],
         [-gammaks],
         [-gammakd]]

    # solve system
    cd = np.linalg.solve(X, bd)     # coeffs for discrete-mesoscopic bound
    ck = np.linalg.solve(X, bk)     # coeffs for mesoscopic-kinetic bound
    cs = np.linalg.solve(X, bs)     # coeffs for kinetic-strong bound

    # preallocate and find points
    dm_bnd = np.full((len(fv),),np.nan)
    mk_bnd = np.full((len(fv),),np.nan)
    ks_bnd = np.full((len(fv),),np.nan)

    # shallow regime
    shal_idx = fv <= fs_bnd
    dm_bnd[shal_idx] = zDs(kv[shal_idx])
    mk_bnd[shal_idx] = zLs(kv[shal_idx])
    ks_bnd[shal_idx] = zSs(kv[shal_idx])

    # intermediate regime
    int_idx = (fv > fs_bnd) & (fv < fd_bnd)
    lk = np.log10(kv[int_idx])
    dm_bnd[int_idx] = 10**(cd[0]*lk**3+cd[1]*lk**2+cd[2]*lk+cd[3])
    mk_bnd[int_idx] = 10**(ck[0]*lk**3+ck[1]*lk**2+ck[2]*lk+ck[3])
    ks_bnd[int_idx] = 10**(cs[0]*lk**3+cs[1]*lk**2+cs[2]*lk+cs[3])

    # deep regime
    deep_idx = fv >= fd_bnd
    dm_bnd[deep_idx] = zDd(kv[deep_idx])
    mk_bnd[deep_idx] = zLd(kv[deep_idx])
    ks_bnd[deep_idx] = zSd(kv[deep_idx])

    # trim extra values
    out_idx = (fv >= 0.9*fi) & (fv <= 1.1*ff)
    fv = fv[out_idx]
    kv = kv[out_idx]
    dm_bnd = dm_bnd[out_idx]
    mk_bnd = mk_bnd[out_idx]
    ks_bnd = ks_bnd[out_idx]

    # interpolate back to the original grid (if provided)
    if fvo is not None:
        xvi = np.log10(kv)
        xvo = np.log10(get_wavenumbers(fvo))
        ydm = np.log10(dm_bnd)
        ymk = np.log10(mk_bnd)
        yks = np.log10(ks_bnd)
        ydmo = interp1d(xvi,ydm)(xvo)
        ymko = interp1d(xvi,ymk)(xvo)
        ykso = interp1d(xvi,yks)(xvo)
        dm_bnd = 10**ydmo
        mk_bnd = 10**ymko
        ks_bnd = 10**ykso
        fv = fvo
        kv = 10**xvo

    return fv, kv, dm_bnd*10**6, mk_bnd*10**6, ks_bnd*10**6


def get_wavenumbers(fv):
    """
    Given a set of frequencies associated with pure capillary waves, returns associated set of wavenumbers.
    
    Notes
    -----
    (1) uses general capillary wave dispersion relation
    (2) assumes gravity plays a negligible role in wave dynamics

    Parameters
    ----------
    fv : frequencies, 1d float array

    Returns
    -------
    kv : wavenumbers, 1d float array

    """

    # zero tolerance for root finding
    tol = 10**-10

    # vectorize
    if np.isscalar(fv):
        sidx = 1
        fv = np.asarray([0.5*fv, fv], dtype='float64')
    elif fv.size > 1:
        sidx = 0
        fv = np.asarray(fv, dtype='float64')

    # frequency -> angular frequency
    wv = 2*np.pi*fv

    # General dispersion relation for surface tension dominated (capillary) waves
    # w_k^2 = surf_ten/density*k^3*tanh(kh)
    def cw_dispersion(k, w):
        ret_val = w - np.sqrt(k**3*(surf_ten/density)*np.tanh(k*h))
        return ret_val

    # preallocate, iterate, and solve
    kv = np.empty((wv.size,))
    for n,w in enumerate(wv):

        # initialize guess
        if n == 0:
            if w > tol:
                x0 = w
            else:
                x0 = wv[1]

        # find the root
        opt_result = root(cw_dispersion, x0, args=(w, ), method='hybr')
        k = opt_result.x[0]
        
        # opt_result = root(cw_dispersion, x0, args=(w), method='hybr') # this is risky b/c args should be a tuple
        # k = opt_result.x
        

        # validate solution
        check_val = opt_result.success
        if check_val & (k>=0):
            kv[n] = k
        else:
            print(x0)
            sys.exit('failed at iteration {0}, k = {1:.1f}'.format(n,kv[n-1]))
        if k > tol:
            x0 = k
        else:
            x0 = wv[n+1]

    return kv[sidx:]
