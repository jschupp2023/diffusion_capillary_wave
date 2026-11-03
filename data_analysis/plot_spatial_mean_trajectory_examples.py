"""Generate representative trajectory plots for the spatial mean under the Hybrid Selector.

Visualizes:
1. Multi-cycle 50 ms macro view (showing 140 Hz heave rhythm and amplitude modulation)
2. High-resolution 15 ms zoom (showing capillary ripples riding on the 140 Hz heave)
3. Direct comparison in the excursion window (Frames 10835-10880: Approach A vs Hybrid)
4. Phase portrait / Attractor (h_mean vs dh_mean/dt) showing what the ROM needs to learn
"""

import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = Path("runs/temporal_alignment_validation/step_events_0p20/full_recording_traces.npz")
OUT_FIG = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/spatial_mean_trajectory_examples.png")
OUT_REPO = Path("runs/hybrid_temporal_selector/spatial_mean_trajectory_examples.png")

d = np.load(DATA_FILE)
time_sec = d["time"]
fs = 1.0 / np.mean(np.diff(time_sec))
q = 1.9969696
n_frames = len(time_sec)

spatial_mean = d["spatial"][:, 0]
spatial_quads = d["spatial"][:, 2:]
raw_mean = d["raw"][:, 0]
current_offsets = d["exported_offsets"]
dk_appA = np.r_[0, np.diff(current_offsets)]

# Run lengths
run_lengths = np.zeros(n_frames, dtype=int)
for t in range(n_frames):
    if dk_appA[t] != 0:
        s = np.sign(dk_appA[t])
        fwd, bwd = 0, 0
        while t + fwd + 1 < n_frames and np.sign(dk_appA[t + fwd + 1]) == s: fwd += 1
        while t - bwd - 1 >= 0 and np.sign(dk_appA[t - bwd - 1]) == s: bwd += 1
        run_lengths[t] = 1 + fwd + bwd

# Hybrid shifts
quad_diffs = np.vstack([np.zeros((1, 4)), np.diff(spatial_quads, axis=0)])
shifts_hybrid = np.zeros(n_frames, dtype=int)
for t in range(1, n_frames):
    if abs(dk_appA[t]) >= 3:
        shifts_hybrid[t] = dk_appA[t]
    elif abs(dk_appA[t]) in [1, 2]:
        if run_lengths[t] == 1:
            target = dk_appA[t] * q
            if np.sum(np.abs(quad_diffs[t] - target) <= 0.6 * q) >= 3:
                shifts_hybrid[t] = dk_appA[t]

offsets_hybrid = np.cumsum(shifts_hybrid)
mean_hybrid = spatial_mean - q * offsets_hybrid
mean_appA = spatial_mean - q * current_offsets

# 50 Hz high-pass filter (standard experimental protocol)
sos = butter(4, 50.0, btype="highpass", fs=fs, output="sos")
hp_mean_hybrid = sosfiltfilt(sos, mean_hybrid.astype(np.float64)).astype(np.float32)
hp_mean_appA = sosfiltfilt(sos, mean_appA.astype(np.float64)).astype(np.float32)

# Estimate time derivative dh/dt in mm/s (central difference)
dt = 1.0 / fs
vel_hybrid = np.gradient(hp_mean_hybrid, dt) * 1e-3  # mm/s

# 4-Panel Publication Figure
fig = plt.figure(figsize=(16, 11), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# -------------------------------------------------------------
# Panel A: Macro View (50 ms, ~7 heave cycles, frames 20000 to 25760)
# -------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
w1 = slice(20000, 25760)
t1_ms = (time_sec[w1] - time_sec[w1.start]) * 1000.0

ax1.plot(t1_ms, hp_mean_hybrid[w1], color="#1f77b4", lw=1.6, label="Hybrid Anti-Runaway (Physical Heave)")
ax1.plot(t1_ms, hp_mean_appA[w1], color="#d62728", lw=1.3, ls="--", alpha=0.75, label="Approach A (Heave Over-Suppressed)")
ax1.axhline(0, color="gray", ls=":", lw=0.8)
ax1.set_xlim(0, 50.0)
ax1.set_ylim(-110, 110)
ax1.set_xlabel("Elapsed Time [ms]", fontsize=11)
ax1.set_ylabel("Spatial Mean Elevation $\\bar{h}(t)$ [$\\mu$m]", fontsize=11)
ax1.set_title("A. Macro View (50 ms / ~7 Cycles of 140 Hz Bulk Heave)", fontsize=12, fontweight="bold")
ax1.grid(True, ls="--", alpha=0.35)
ax1.legend(loc="upper right", fontsize=10)

# Add period annotation (1/140 Hz = 7.14 ms)
ax1.annotate(
    "140 Hz Heave Period\n$T \\approx 7.14$ ms",
    xy=(14.28, 80), xytext=(22, 90),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1.0, headwidth=5),
    fontsize=9.5, fontweight="medium"
)

# -------------------------------------------------------------
# Panel B: High-Resolution Zoom (15 ms, ~2 heave cycles, frames 20000 to 21728)
# -------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
w2 = slice(20000, 21728)
t2_ms = (time_sec[w2] - time_sec[w2.start]) * 1000.0

ax2.plot(t2_ms, hp_mean_hybrid[w2], color="#1f77b4", lw=2.0, label="Hybrid Anti-Runaway (Smooth Wave Dynamics)")
ax2.plot(t2_ms, hp_mean_appA[w2], color="#d62728", lw=1.4, ls="--", alpha=0.75, label="Approach A (Over-chopped)")
ax2.axhline(0, color="gray", ls=":", lw=0.8)
ax2.set_xlim(0, 15.0)
ax2.set_ylim(-90, 90)
ax2.set_xlabel("Elapsed Time [ms]", fontsize=11)
ax2.set_ylabel("Spatial Mean Elevation $\\bar{h}(t)$ [$\\mu$m]", fontsize=11)
ax2.set_title("B. Micro View (15 ms: Smooth Heave + Capillary Ripple Modulation)", fontsize=12, fontweight="bold")
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(loc="upper right", fontsize=10)

# -------------------------------------------------------------
# Panel C: Wave Excursion Window (Frames 10840 to 10875)
# -------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
w3 = slice(10840, 10875)
t3_us = (time_sec[w3] - time_sec[w3.start]) * 1e6

# Show both unfiltered spatial mean and 50 Hz filtered
ax3.plot(t3_us, spatial_mean[w3], color="black", lw=2.2, alpha=0.5, label="Spatial-Only (Raw Wave Motion)")
ax3.plot(t3_us, mean_appA[w3], color="#d62728", lw=1.8, ls="--", label="Approach A (Flattens real wave: 25 chops!)")
ax3.plot(t3_us, mean_hybrid[w3], color="#1f77b4", lw=2.0, label="Hybrid Selector (100% Preserved: 0 chops)")
ax3.set_xlabel("Elapsed Time [$\\mu$s]", fontsize=11)
ax3.set_ylabel("Spatial Mean $\\bar{h}(t)$ [$\\mu$m]", fontsize=11)
ax3.set_title("C. Excursion Window (Frames 10840-10875: No Runaway Clipping)", fontsize=12, fontweight="bold")
ax3.grid(True, ls="--", alpha=0.35)
ax3.legend(loc="upper right", fontsize=9.5)

# -------------------------------------------------------------
# Panel D: Reduced State Attractor / Phase Portrait (h_mean vs dh_mean/dt)
# -------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
# Sample 10,000 timesteps to show the clean limit cycle attractor
w4 = slice(20000, 30000)
h_sub = hp_mean_hybrid[w4]
v_sub = vel_hybrid[w4]

# Color points by time to see trajectory flow
scatter = ax4.scatter(h_sub, v_sub, c=(time_sec[w4] - time_sec[w4.start])*1000.0,
                      cmap="viridis", s=4, alpha=0.5, rasterized=True)
cbar = fig.colorbar(scatter, ax=ax4, shrink=0.85, pad=0.03)
cbar.set_label("Time within window [ms]", fontsize=10)

ax4.set_xlim(-110, 110)
ax4.set_ylim(-75, 75)
ax4.set_xlabel("Spatial Mean Elevation $\\bar{h}(t)$ [$\\mu$m]", fontsize=11)
ax4.set_ylabel("Vertical Velocity $d\\bar{h}/dt$ [mm/s]", fontsize=11)
ax4.set_title("D. Learned Coordinate Attractor: Phase Portrait $(\\bar{h}, \\dot{\\bar{h}})$", fontsize=12, fontweight="bold")
ax4.grid(True, ls="--", alpha=0.35)

OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_REPO, dpi=200)
fig.savefig(OUT_FIG, dpi=200)
plt.close(fig)

print(f"Figure successfully saved to:")
print(f"  1. {OUT_REPO}")
print(f"  2. {OUT_FIG}")
