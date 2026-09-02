import numpy as np
from scipy.signal import savgol_filter, welch
# import capillary_wave_analysis as cwa

# # System parameters
# sigma = cwa.surf_ten
# rho   = cwa.density
# h     = cwa.h
# L     = cwa.L
# km    = np.pi / L

def psd_slope_in_range(k, Sk, kmin, kmax):

    k = np.asarray(k)
    Sk = np.asarray(Sk)

    # Find index range corresponding to kmin, kmax
    mask = (k >= kmin) & (k <= kmax)
    idx = np.where(mask)[0]
    
    if len(idx) < 3:
        return np.nan, np.nan

    i0 = idx[0]
    i1 = idx[-1]

    logk = np.log10(k + 1e-30)
    logS = np.log10(np.maximum(Sk, np.finfo(float).tiny))

    # x = logk
    # y = logS

    x = logk
    y = Sk

    X  = np.r_[0.0, np.cumsum(x)]
    Y  = np.r_[0.0, np.cumsum(y)]
    X2 = np.r_[0.0, np.cumsum(x*x)]
    Y2 = np.r_[0.0, np.cumsum(y*y)]
    XY = np.r_[0.0, np.cumsum(x*y)]

    m  = i1 - i0 + 1
    Sx = X[i1+1]-X[i0]
    Sy = Y[i1+1]-Y[i0]
    Sxx= X2[i1+1]-X2[i0]
    Syy= Y2[i1+1]-Y2[i0]
    Sxy= XY[i1+1]-XY[i0]

    denx = m*Sxx - Sx*Sx
    deny = m*Syy - Sy*Sy
    den  = np.sqrt(max(denx, 0.0) * max(deny, 0.0))

    if den <= 0:
        return np.nan, -np.inf

    r   = (m*Sxy - Sx*Sy) / den
    r2  = r*r
    slope = (m*Sxy - Sx*Sy) / max(denx, 1e-300)
    
    return slope, r2

def inertial_range_mask(k, Sk,
                        smooth=False,
                        smooth_win=21, smooth_poly=2,
                        min_pts=5,
                        slope_bounds=(-8.0, -1.0),
                        prefer_longer=False,
                        length_metric="logx",
                        length_weight=1e-3):

    k = np.asarray(k)
    Sk = np.asarray(Sk)
    n = k.size

    # Convert k (wavenumber) and Sk (spectral density values) into log-log space
    logk = np.log10(k + 1e-30)
    logS = np.log10(np.maximum(Sk, np.finfo(float).tiny))

    if smooth:
        win = max(5, int(smooth_win))
        if win % 2 == 0:
            win += 1
        if win >= n:
            win = n - 1 if (n % 2 == 0) else n
        if win < 3:
            win = 3
        try:
            logS_filtered = savgol_filter(logS, window_length=win, polyorder=min(smooth_poly, 3))
        except Exception:
            logS_filtered = logS.copy()
    else:
        logS_filtered = logS

    x = logk
    y = logS_filtered

    X  = np.r_[0.0, np.cumsum(x)]   
    Y  = np.r_[0.0, np.cumsum(y)]
    X2 = np.r_[0.0, np.cumsum(x*x)]
    Y2 = np.r_[0.0, np.cumsum(y*y)]
    XY = np.r_[0.0, np.cumsum(x*y)]

    def seg_stats(i, j):
        m  = j - i + 1
        Sx = X[j+1]-X[i]
        Sy = Y[j+1]-Y[i]
        Sxx= X2[j+1]-X2[i]
        Syy= Y2[j+1]-Y2[i]
        Sxy= XY[j+1]-XY[i]

        denx = m*Sxx - Sx*Sx
        deny = m*Syy - Sy*Sy
        den  = np.sqrt(max(denx, 0.0) * max(deny, 0.0))

        if den <= 0:
            r2 = -np.inf
            slope = np.nan
        else:
            r   = (m*Sxy - Sx*Sy) / den
            r2  = r*r
            slope = (m*Sxy - Sx*Sy) / max(denx, 1e-300)
        return m, slope, r2

    best = (-np.inf, None, None, None)
    for i in range(0, n - (min_pts - 1)):    
        for j in range(i + (min_pts - 1), n - 1):
            m, slope, r2 = seg_stats(i, j)
            if np.isfinite(slope) and slope_bounds is not None:
                lo, hi = slope_bounds
                if not (lo <= slope <= hi):
                    continue
            if prefer_longer:
                length = max(x[j] - x[i], 0.0) if length_metric=="logx" else max(10**x[j] - 10**x[i], 0.0)
                score = r2 + length_weight * length
            else:
                score = r2
            
            # Update the best score    
            if score > best[0]:
                best = (score, i, j, slope)

    score_best, i_best, j_best, slope_best = best
    if i_best is None:
        return np.zeros(n, dtype=bool), np.nan, (None, None)
    if not (score_best > 0.95):
        return np.zeros(n, dtype=bool), np.nan, -np.inf

    mask = np.zeros(n, dtype=bool)
    mask[i_best:j_best+1] = True
    
    return mask, float(slope_best), score_best


def welch_psd_k(z_um, fs, target_segments):
    n = len(z_um)
    # target_segments = 100
    nperseg = max(128, int(2*n/(target_segments+1)))
    
    if nperseg % 2:
        nperseg -= 1
    
    f, S_f = welch(z_um, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg//2,
                   return_onesided=True, scaling='density')

    k = cwa.get_wavenumbers(f)        
    df_dk = 1.0 / np.gradient(k, f)    
    
    # dk_df = np.gradient(k, f)
    # df_dk = np.divide(1.0, dk_df, where=dk_df!=0)

    S_k = S_f * np.abs(df_dk)
    
    # # Test
    # f_area, k_area = np.trapz(S_f, f), np.trapz(S_k, k)
    # print(f"f_area: {f_area}")
    # print(f"k_area: {k_area}")
    
    return k, S_k, f


def compute_metrics(z_um, fs, min_pts, target_segments):
    k_psd, S_k, f_psd = welch_psd_k(z_um, fs, target_segments)
    # mask = (k_psd >= 8000/1.4) & (k_psd <= 64_000*1.4)   
    mask = (k_psd >= 10) & (k_psd <= 120_000)   
    # mask = (k_psd >= 4000) & (k_psd <= 128_000)   
    
    k_psd = k_psd[mask]
    S_k = S_k[mask]
    f_psd = f_psd[mask]
    mask_inertial, slope_full, r2_full = inertial_range_mask(k=k_psd, Sk=S_k, min_pts=min_pts)

    indices = np.where(mask_inertial)[0]
    if len(indices) > 0:
        i_best = indices[0]
        j_best = indices[-1]
    else:
        i_best = j_best = None

    return {
        "k_psd": k_psd,
        "S_k": S_k,
        "f_psd": f_psd,
        "mask_inertial": mask_inertial,
        "slope_full": slope_full,
        "r2_full": r2_full,
        "i_best": i_best,
        "j_best": j_best,
    }


def compute_PSD(x, FS_DHM, min_pts, target_segments):
    metrics = compute_metrics(x, FS_DHM, min_pts, target_segments)
    k = metrics["k_psd"]
    k_krad = k * 1e-3
    f_psd = metrics["f_psd"]
    S_k = metrics["S_k"]
    log10_psd = np.log10(np.maximum(S_k, np.finfo(float).tiny))
    slope = metrics["slope_full"]
    r2 = metrics["r2_full"]
    i_best = metrics["i_best"]
    j_best = metrics["j_best"]
    
    return k_krad, log10_psd, f_psd, slope, r2, i_best, j_best



XMIN_RAW = 8.0 / 2      # gravity wave regime
XMAX_RAW = 64.0 * 2     # dissipation range
XMIN_LOG2 = np.log2(XMIN_RAW)
XMAX_LOG2 = np.log2(XMAX_RAW)

def amplitude_welch(z, fs):
    """
    Welch-matched amplitude spectrum (RMS per bin).
    This is for regime classification (not PSD, but RMS amplitude)
    """
    n = len(z)
    target_segments = 100
    nperseg = max(256, int(2*n/(target_segments+1)))
    if nperseg % 2:
        nperseg -= 1

    w = np.hanning(nperseg)
    noverlap = nperseg // 2
    step = nperseg - noverlap
    f_seg = np.fft.rfftfreq(nperseg, d=1.0/fs)
    acc = np.zeros_like(f_seg, dtype=float)
    count = 0

    for start in range(0, n - nperseg + 1, step):
        seg = z[start:start+nperseg]
        Zs  = np.fft.rfft(seg * w)
        amp = (2.0 / np.sum(w)) * np.abs(Zs)
        acc = acc + (amp**2)
        count += 1

    if count == 0:
        raise RuntimeError("No segments found for Welch-matched amplitude.")
    amp_out = np.sqrt(acc / count)
    
    return f_seg, amp_out

def df_dk_from_k(k):
    """
    Analytical df/dk for capillary-dominant dispersion:
        ω^2 = (σ/ρ) k^3 tanh(kh)
        f = ω / (2π)
        df/dk = (1/(4π ω)) d(ω^2)/dk
    """
    kh = k * h
    tanh_kh  = np.tanh(kh)
    sech2_kh = 1.0 / np.cosh(kh)**2
    omega    = np.sqrt((sigma/rho) * k**3 * tanh_kh)
    domega2_dk = (sigma/rho) * (3*k**2*tanh_kh + k**3*sech2_kh*h)
    with np.errstate(divide="ignore", invalid="ignore"):
        df_dk = domega2_dk / (4*np.pi*omega)  # df/dk = (1/2π) * dω/dk
        df_dk[~np.isfinite(df_dk)] = 0.0
    return df_dk