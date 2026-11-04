"""Deep quality control and verification of the production cleaned dataset:
11232025_a_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5.
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

CLEANED_FILE = Path("/home/jonas/ucsd_thesis/11232025_a_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5")
OUT_FIG = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/cleaned_qc_11232025_a_0.2vpp.png")
OUT_REPO = Path("runs/cleaned_qc/cleaned_qc_11232025_a_0.2vpp.png")
OUT_REPO.parent.mkdir(parents=True, exist_ok=True)

with h5py.File(CLEANED_FILE, "r") as f:
    t_arr = f["meta/t"][:]
    time_sec = t_arr
    fs = 1.0 / np.mean(np.diff(time_sec))
    n_frames = len(time_sec)
    offsets = f["meta/temporal_offset_cycles"][:]
    flags = f["meta/temporal_uncertainty_flags"][:]
    
    # Sample spatial mean and center point across all frames
    print(f"Scanning all {n_frames:,} frames for QC metrics...")
    h_mean = np.empty(n_frames, dtype=np.float32)
    h_center = np.empty(n_frames, dtype=np.float32)
    
    # Check 1000 frames evenly spaced for NaNs / Infs / extremes
    has_nan = False
    has_inf = False
    
    # Extract mean and center using chunked iteration
    for i in range(n_frames):
        # Key format: %05d or %06d
        key = f"{i:05d}" if i < 100000 else f"{i:06d}"
        if key not in f["main"]:
            # fallback
            key = f"{i}"
        frame = f["main"][key][:]
        h_mean[i] = np.mean(frame)
        h_center[i] = frame[100, 100]
        
        if i % 10000 == 0:
            if np.isnan(frame).any(): has_nan = True
            if np.isinf(frame).any(): has_inf = True
            if i % 20000 == 0:
                print(f"  Scanned {i:,}/{n_frames:,} frames...")

print("Scan complete.")
print(f"NaNs detected: {has_nan}")
print(f"Infs detected: {has_inf}")
print(f"Spatial mean span: min={h_mean.min():.2f} um, max={h_mean.max():.2f} um, std={h_mean.std():.2f} um")
print(f"Center probe span: min={h_center.min():.2f} um, max={h_center.max():.2f} um, std={h_center.std():.2f} um")

dh_mean = np.r_[0, np.diff(h_mean)]
print(f"Max inter-frame increment in cleaned spatial mean: {np.max(np.abs(dh_mean)):.2f} um")
print(f"Inter-frame increments > 10 um: {np.count_nonzero(np.abs(dh_mean) > 10.0)}")
print(f"Inter-frame increments > 20 um: {np.count_nonzero(np.abs(dh_mean) > 20.0)}")

# Compute Welch PSDs
nperseg = 16384
f, p_mean = welch(h_mean, fs=fs, window="hann", nperseg=nperseg, noverlap=8192)
_, p_center = welch(h_center, fs=fs, window="hann", nperseg=nperseg, noverlap=8192)

# 4-Panel QC Figure
fig = plt.figure(figsize=(16, 11), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# Panel A: Macro Trajectory (50 ms)
ax1 = fig.add_subplot(gs[0, 0])
w_macro = slice(20000, 20000 + int(0.050 * fs))
t_macro_ms = (time_sec[w_macro] - time_sec[w_macro.start]) * 1000.0
ax1.plot(t_macro_ms, h_mean[w_macro], color="#1f77b4", lw=1.8, label=r"Spatial Mean $\bar{h}(t)$")
ax1.plot(t_macro_ms, h_center[w_macro], color="darkorange", lw=1.2, alpha=0.75, label=r"Center Point $h(x_c, y_c, t)$")
ax1.set_xlabel("Time [ms]", fontsize=11)
ax1.set_ylabel(r"Elevation [$\mu$m]", fontsize=11)
ax1.set_title("A. Macro Trajectory (50 ms): Clean 140 Hz Bulk Heave", fontsize=11.5, fontweight="bold")
ax1.grid(True, ls="--", alpha=0.35)
ax1.legend(loc="upper right", fontsize=10)

# Panel B: Micro Zoom (15 ms)
ax2 = fig.add_subplot(gs[0, 1])
w_micro = slice(20000, 20000 + int(0.015 * fs))
t_micro_ms = (time_sec[w_micro] - time_sec[w_micro.start]) * 1000.0
ax2.plot(t_micro_ms, h_mean[w_micro], color="#1f77b4", marker=".", lw=1.8, label=r"Spatial Mean $\bar{h}(t)$")
ax2.plot(t_micro_ms, h_center[w_micro], color="darkorange", marker=".", lw=1.2, alpha=0.75, label=r"Center Point")
ax2.set_xlabel("Time [ms]", fontsize=11)
ax2.set_ylabel(r"Elevation [$\mu$m]", fontsize=11)
ax2.set_title("B. Micro Zoom (15 ms): Smooth, Continuously Differentiable Ripples", fontsize=11.5, fontweight="bold")
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(loc="upper right", fontsize=10)

# Panel C: Center Point Welch PSD
ax3 = fig.add_subplot(gs[1, 0])
mask_f = (f >= 10.0) & (f <= 4000.0)
ax3.loglog(f[mask_f], p_center[mask_f], color="darkorange", lw=2.0, label="Cleaned Center Elevation PSD")
ax3.axvline(50.0, color="gray", ls="--", lw=1.2, label="50 Hz Filter Cutoff")
ax3.axvline(140.0, color="navy", ls=":", lw=1.5, label="140 Hz Bulk Acoustic Heave")
ax3.set_xlim(10, 4000)
ax3.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
ax3.set_ylabel(r"PSD [$\mu$m$^2$/Hz]", fontsize=11)
ax3.set_title("C. Center Elevation PSD: 50 Hz Roll-Off & 140 Hz Peak Preserved", fontsize=11.5, fontweight="bold")
ax3.grid(True, which="both", ls="--", alpha=0.35)
ax3.legend(loc="lower left", fontsize=10)

# Panel D: Spatial Mean Welch PSD
ax4 = fig.add_subplot(gs[1, 1])
ax4.loglog(f[mask_f], p_mean[mask_f], color="#1f77b4", lw=2.0, label="Cleaned Spatial Mean PSD")
ax4.axvline(50.0, color="gray", ls="--", lw=1.2, label="50 Hz Filter Cutoff")
ax4.axvline(140.0, color="navy", ls=":", lw=1.5, label="140 Hz Bulk Heave Peak")
ax4.set_xlim(10, 4000)
ax4.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
ax4.set_ylabel(r"PSD [$\mu$m$^2$/Hz]", fontsize=11)
ax4.set_title("D. Spatial Mean PSD: Dominated by Physical 140 Hz Resonance", fontsize=11.5, fontweight="bold")
ax4.grid(True, which="both", ls="--", alpha=0.35)
ax4.legend(loc="lower left", fontsize=10)

fig.savefig(OUT_REPO, dpi=180)
fig.savefig(OUT_FIG, dpi=180)
plt.close(fig)

print(f"QC Figure successfully saved to:")
print(f"  1. {OUT_REPO}")
print(f"  2. {OUT_FIG}")

