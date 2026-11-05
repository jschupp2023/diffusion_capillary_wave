"""Generate publication-grade diagnostic figure demonstrating how temporal cycle alignment
eliminates massive unphysical jumps in the spatial mean elevation.
"""

from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_NPZ = Path("runs/data_cleaning_production/center_signals_11272025_c_0.2vpp.npz")
OUT_THESIS = Path("thesis_notes/figures/spatial_mean_alignment_comparison.png")
OUT_ARTIFACT = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/spatial_mean_alignment_comparison.png")

d = np.load(DATA_NPZ)
raw_mean = d["raw_mean"]
cln_mean = d["cln_mean"]
fs = float(d["fs"])
dt = 1.0 / fs
n_frames = len(raw_mean)
time_ms = np.arange(n_frames) * dt * 1000.0

fig = plt.figure(figsize=(13, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

# -------------------------------------------------------------
# Panel A: Macro window (100 ms) showing massive multi-cycle cliffs vs smooth physical heave
# -------------------------------------------------------------
ax_macro = fig.add_subplot(gs[0, :])
w_macro = slice(16500, 20000)  # ~30 ms window containing multiple massive bursts
t_macro = (time_ms[w_macro] - time_ms[w_macro.start])

# Center raw around 0 in this window to allow direct visual comparison
raw_window = raw_mean[w_macro] - raw_mean[w_macro.start]
cln_window = cln_mean[w_macro] - cln_mean[w_macro.start]

ax_macro.plot(t_macro, raw_window, color="#d62728", lw=1.3, alpha=0.8, label="Raw DHM Spatial Mean (optical seed slips)")
ax_macro.plot(t_macro, cln_window, color="#1f77b4", lw=2.0, alpha=0.95, label="Cleaned Spatial Mean (temporal cycle-aligned)")
ax_macro.set_title("A. Macro Trajectory of Spatial Mean Elevation (30 ms Window, ~4 Bulk Heave Cycles)", fontsize=12, fontweight="bold")
ax_macro.set_xlabel("Time [ms]", fontsize=11)
ax_macro.set_ylabel(r"Relative $\bar{h}(t)$ [$\mu$m]", fontsize=11)
ax_macro.grid(True, ls=":", alpha=0.6)
ax_macro.legend(loc="upper left", fontsize=10.5, framealpha=0.95)

# Annotate massive cliff
ax_macro.annotate(
    "Optical Seed Slips\n(Jumps > 90 um)",
    xy=(4.65, raw_window[17035 - 16500]),
    xytext=(7.0, raw_window[17035 - 16500] + 50),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
    fontsize=10,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", alpha=0.9),
)

# -------------------------------------------------------------
# Panel B: Zoom-in on Frame 17034 (-97.1 um cliff eliminated in 8.68 us)
# -------------------------------------------------------------
ax_zoom = fig.add_subplot(gs[1, 0])
w_zoom = slice(17025, 17055)
n_zoom = w_zoom.stop - w_zoom.start
t_zoom_us = (np.arange(n_zoom) - 9) * dt * 1e6  # relative time in microseconds

raw_zoom = raw_mean[w_zoom] - raw_mean[17034]
cln_zoom = cln_mean[w_zoom] - cln_mean[17034]

ax_zoom.plot(t_zoom_us, raw_zoom, "o-", color="#d62728", lw=1.8, ms=4.5, label=r"Raw (jump = $-97.1\,\mu$m)", alpha=0.85)
ax_zoom.plot(t_zoom_us, cln_zoom, "s--", color="#1f77b4", lw=2.2, ms=4.5, label=r"Cleaned ($\Delta\bar{h} = -0.65\,\mu$m)", alpha=0.95)
ax_zoom.axvline(0, color="k", ls=":", alpha=0.5)
ax_zoom.set_title(r"B. Zoom on Frame 17035: $-97.1\,\mu$m ($-48.6\,q$) Step Cliff Repaired", fontsize=11, fontweight="bold")
ax_zoom.set_xlabel(r"Time Relative to Slip Event [$\mu$s]", fontsize=10)
ax_zoom.set_ylabel(r"Elevation Change $\Delta \bar{h}$ [$\mu$m]", fontsize=10)
ax_zoom.grid(True, ls=":", alpha=0.6)
ax_zoom.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

# -------------------------------------------------------------
# Panel C: Velocity Increment Distribution |Delta h_mean| (log scale)
# -------------------------------------------------------------
ax_hist = fig.add_subplot(gs[1, 1])
dh_raw = np.abs(np.diff(raw_mean))
dh_cln = np.abs(np.diff(cln_mean))

bins = np.linspace(0, 50, 100)
h_raw, _ = np.histogram(dh_raw, bins=bins)
h_cln, _ = np.histogram(dh_cln, bins=bins)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

ax_hist.semilogy(bin_centers, h_raw, color="#d62728", lw=1.8, label=f"Raw increments (max = {np.max(dh_raw):.1f} um)")
ax_hist.semilogy(bin_centers, h_cln, color="#1f77b4", lw=2.2, label=f"Cleaned increments (max = {np.max(dh_cln):.1f} um)")
ax_hist.axvline(1.9969696 / 2, color="gray", ls="--", alpha=0.7, label=r"Optical Nyquist $q/2 \approx 1.0\,\mu$m")
ax_hist.axvline(10.0, color="k", ls=":", alpha=0.7, label="Hydrodynamic threshold (10 um)")

ax_hist.set_title(r"C. Frame Increment Distribution $|\Delta \bar{h}_t|$ (100k Timesteps)", fontsize=11, fontweight="bold")
ax_hist.set_xlabel(r"Frame-to-Frame Increment $|\Delta \bar{h}|$ [$\mu$m]", fontsize=10)
ax_hist.set_ylabel("Occurrences (log scale)", fontsize=10)
ax_hist.set_ylim(0.5, 1e5)
ax_hist.grid(True, which="both", ls=":", alpha=0.5)
ax_hist.legend(loc="upper right", fontsize=9.2, framealpha=0.95)

for out_p in [OUT_THESIS, OUT_ARTIFACT]:
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200)
    print(f"Saved: {out_p}")

plt.close(fig)
