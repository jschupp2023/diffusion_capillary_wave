"""Plot low-frequency zoom (40 - 600 Hz) highlighting discrete harmonic peaks across regimes.

Compares:
- 0.08 Vpp (Weak/Intermediate: sharp coherent discrete harmonics)
- 0.20 Vpp (Turbulent: distinct subharmonic and 280 Hz peak with emerging cascade)
- 0.35 Vpp (Strong Turbulent: broadened, attenuated peaks melting into the continuum)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

d4 = np.load("runs/psd_4regimes_comparison/center_signals_4regimes.npz")
fs = float(d4["fs"])
c08 = d4["c08_cln"]
sos = butter(4, 50.0, btype="highpass", fs=fs, output="sos")
c08_hp = sosfiltfilt(sos, c08.astype(np.float64)).astype(np.float32)

d20 = np.load("runs/data_cleaning_production/hybrid_center_signals_11272025_c_0.2vpp.npz")
c20 = d20["hybrid_center"]

d35 = np.load("runs/data_cleaning_production/center_signals_11252025_e_0.35vpp.npz")
c35 = d35["cln_center"]

nperseg = 16384  # high frequency resolution ~7 Hz
noverlap = nperseg // 2

f08, p08 = welch(c08_hp, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
f20, p20 = welch(c20, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
f35, p35 = welch(c35, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)

mask = (f35 >= 45.0) & (f35 <= 600.0)

# Panel 1: Log-scale PSD
ax1.semilogy(f08[mask], p08[mask], color="#2ca02c", lw=1.5, label=r"$\Gamma = 0.08\,\mathrm{g}$ (Sharp coherent peaks)")
ax1.semilogy(f20[mask], p20[mask], color="#ff7f0e", lw=1.8, label=r"$\Gamma = 0.20\,\mathrm{g}$ (Turbulent: peaks + cascade)")
ax1.semilogy(f35[mask], p35[mask], color="#1f77b4", lw=2.0, label=r"$\Gamma = 0.35\,\mathrm{g}$ (Strongly broadened peaks)")

# Mark harmonics
harmonics = [70, 140, 210, 280, 350, 420, 490, 560]
for h in harmonics:
    ax1.axvline(h, color="gray", linestyle=":", alpha=0.55)
    ax2.axvline(h, color="gray", linestyle=":", alpha=0.55)
    lbl = f"{h} Hz"
    if h == 70:
        lbl += "\n(f0/2)"
    elif h == 140:
        lbl += "\n(f0 pump)"
    ax1.text(h, 2.5e1, lbl, rotation=0, fontsize=8, ha="center", va="bottom", color="#495057")

ax1.set_ylabel(r"$S_{hh}(f)$ [$\mu\mathrm{m}^2/\mathrm{Hz}$]", fontsize=11, fontweight="bold")
ax1.set_title(r"Discrete Harmonic Peak Evolution Across Forcing Regimes ($45 - 600\,\mathrm{Hz}$)", fontsize=12, fontweight="bold")
ax1.set_ylim(1e-3, 1e2)
ax1.grid(True, which="both", ls=":", alpha=0.4)
ax1.legend(loc="upper right", fontsize=10, framealpha=0.95)

# Panel 2: Normalized PSD (divided by local smoothed continuum)
from scipy.ndimage import uniform_filter1d
# Smooth with a wide window (~50 Hz)
p08_cont = uniform_filter1d(p08, size=15)
p20_cont = uniform_filter1d(p20, size=15)
p35_cont = uniform_filter1d(p35, size=15)

ax2.plot(f08[mask], p08[mask] / p08_cont[mask], color="#2ca02c", lw=1.4, label=r"$\Gamma = 0.08\,\mathrm{g}$ Contrast")
ax2.plot(f20[mask], p20[mask] / p20_cont[mask], color="#ff7f0e", lw=1.6, label=r"$\Gamma = 0.20\,\mathrm{g}$ Contrast")
ax2.plot(f35[mask], p35[mask] / p35_cont[mask], color="#1f77b4", lw=1.8, label=r"$\Gamma = 0.35\,\mathrm{g}$ Contrast")
ax2.axhline(1.0, color="black", linestyle="--", alpha=0.5)

ax2.set_xlabel("Frequency [Hz]", fontsize=11, fontweight="bold")
ax2.set_ylabel(r"Peak-to-Continuum Ratio", fontsize=11, fontweight="bold")
ax2.set_title(r"Peak Prominence Contrast Relative to Turbulent Continuum", fontsize=12, fontweight="bold")
ax2.set_xlim(45, 600)
ax2.set_ylim(0.2, 3.2)
ax2.grid(True, ls=":", alpha=0.4)
ax2.legend(loc="upper right", fontsize=10, framealpha=0.95)

out_fig = Path("thesis_notes/figures/peaks_comparison_low_freq.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_fig, dpi=200)
fig.savefig(Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/peaks_comparison_low_freq.png"), dpi=200)
plt.close(fig)
print("Saved figure to", out_fig)
