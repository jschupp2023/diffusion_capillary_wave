"""Compare Advanced Kinematic Temporal Selector with Local-Motion Veto against Greedy and Prototype Hybrid.

Outputs diagnostic plot and quantitative comparison table on 0.20 Vpp.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

from data_analysis.temporal_alignment import (
    DEFAULT_Q,
    DEFAULT_TAU,
    select_temporal_kinematic,
)
from data_analysis.review_hybrid_temporal_selector import select as select_proto_hybrid

DATA_FILE = Path("runs/temporal_alignment_validation/step_events_0p20/full_recording_traces.npz")
OUT_REPO = Path("runs/advanced_temporal_selector/advanced_vs_hybrid_comparison.png")
OUT_ARTIFACT = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/advanced_vs_hybrid_comparison.png")
OUT_REPO.parent.mkdir(parents=True, exist_ok=True)

d = np.load(DATA_FILE)
time_sec = d["time"]
fs = 1.0 / np.mean(np.diff(time_sec))
q = DEFAULT_Q
n_frames = len(time_sec)

spatial = d["spatial"]
spatial_mean = spatial[:, 0]
spatial_center = spatial[:, 1]
spatial_quads = spatial[:, 2:]
dh_mean = np.r_[0, np.diff(spatial_mean)]
dq = np.vstack([np.zeros((1, 4)), np.diff(spatial_quads, axis=0)])

# 1. Greedy (Approach A)
greedy_offsets = d["exported_offsets"]
dk_appA = np.r_[0, np.diff(greedy_offsets)]

# 2. Prototype Hybrid
shifts_proto = select_proto_hybrid(dk_appA, dq, q)
offsets_proto = np.cumsum(shifts_proto)

# 3. Advanced Kinematic Selector with Local-Motion Veto
res_adv = select_temporal_kinematic(dh_mean, dq, q=q)
shifts_adv = res_adv.shifts
offsets_adv = res_adv.offsets

# Reconstructed signals
signals = {
    "spatial_only": spatial[:, :2],
    "greedy": spatial[:, :2] - q * greedy_offsets[:, None],
    "hybrid": spatial[:, :2] - q * offsets_proto[:, None],
    "advanced_veto": spatial[:, :2] - q * offsets_adv[:, None],
}

# 50 Hz zero-phase Butterworth filter for spectra
sos = butter(4, 50.0, btype="highpass", fs=fs, output="sos")
psds = {}
for name, sig in signals.items():
    sig_hp = sosfiltfilt(sos, sig.astype(np.float64), axis=0)
    freq, p = welch(sig_hp, fs=fs, window="hann", nperseg=16384, noverlap=8192, axis=0)
    psds[name] = p

# 4-Panel Master Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

# -------------------------------------------------------------
# Panel A: Excursion Window (Frames 10840-10875)
# -------------------------------------------------------------
ax1 = axes[0, 0]
w_zoom = slice(10840, 10875)
t_ms = (time_sec[w_zoom] - time_sec[w_zoom.start]) * 1000.0

ax1.plot(t_ms, signals["spatial_only"][w_zoom, 0] - signals["spatial_only"][w_zoom.start, 0], color="gray", lw=2.5, ls="--", label="Spatial-Only (True Excursion, -63.7 um)")
ax1.plot(t_ms, signals["greedy"][w_zoom, 0] - signals["greedy"][w_zoom.start, 0], color="crimson", lw=2.0, label="Legacy Greedy (Chopped by 79%, -13.8 um)")
ax1.plot(t_ms, signals["hybrid"][w_zoom, 0] - signals["hybrid"][w_zoom.start, 0], color="orange", lw=2.0, ls="-.", label="Prototype Hybrid (Preserved, -63.7 um)")
ax1.plot(t_ms, signals["advanced_veto"][w_zoom, 0] - signals["advanced_veto"][w_zoom.start, 0], color="blue", lw=2.2, label="Advanced Veto (Preserved, -63.7 um)")

ax1.set_xlabel("Time [ms]", fontsize=11)
ax1.set_ylabel(r"Relative $\bar{h}(t)$ [$\mu$m]", fontsize=11)
ax1.set_title("A. Wave Excursion Preservation (Frames 10845–10869)", fontsize=11.5, fontweight="bold")
ax1.grid(True, ls="--", alpha=0.35)
ax1.legend(loc="lower left", fontsize=9.5)

# -------------------------------------------------------------
# Panel B: Synthetic Stress Test (Original Phased 900 Hz Regression)
# -------------------------------------------------------------
ax2 = axes[0, 1]
st = np.arange(8192) / fs
boundary_900 = 1.0001 * q / (4 * np.sin(np.pi / 128)) * np.sin(2 * np.pi * (fs / 128) * st - np.pi / 128)
diff_900 = np.r_[0, np.diff(boundary_900)]
quads_900 = np.repeat(diff_900[:, None], 4, axis=1)

# Prototype hybrid on phased regression
k_900 = np.rint(diff_900 / q).astype(int)
shifts_proto_900 = select_proto_hybrid(k_900, quads_900, q)
out_proto_900 = boundary_900 - q * np.cumsum(shifts_proto_900)

res_adv_900 = select_temporal_kinematic(diff_900, quads_900, q=q)
out_adv_900 = boundary_900 - q * res_adv_900.offsets

t_syn_ms = st[:250] * 1000.0
ax2.plot(t_syn_ms, boundary_900[:250], color="black", lw=2.5, ls="--", label="Ground Truth (Phased 900 Hz Sine)")
ax2.plot(t_syn_ms, out_proto_900[:250], color="orange", lw=1.8, ls="-.", label=f"Prototype Hybrid ({np.count_nonzero(shifts_proto_900)} Notches, Error: {np.max(np.abs(out_proto_900-boundary_900)):.2f} um)")
ax2.plot(t_syn_ms, out_adv_900[:250], color="blue", lw=2.2, label=f"Advanced Veto ({res_adv_900.n_corrections} Notches, Error: {np.max(np.abs(out_adv_900-boundary_900)):.2f} um)")

ax2.set_xlim(0, 2.2)
ax2.set_xlabel("Time [ms]", fontsize=11)
ax2.set_ylabel(r"Amplitude [$\mu$m]", fontsize=11)
ax2.set_title("B. Synthetic Stress Test: Phased 900 Hz Threshold Crossing", fontsize=11.5, fontweight="bold")
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(loc="upper right", fontsize=9.5)

# -------------------------------------------------------------
# Panel C: Center-Point Welch PSD Comparison (f >= 50 Hz)
# -------------------------------------------------------------
ax3 = axes[1, 0]
mask_f = (freq >= 50.0) & (freq <= 3000.0)
ax3.loglog(freq[mask_f], psds["spatial_only"][mask_f, 1], color="gray", lw=1.8, ls="--", label="Spatial-Only (Zero Temporal Mod)")
ax3.loglog(freq[mask_f], psds["greedy"][mask_f, 1], color="crimson", lw=1.8, label="Legacy Greedy (Severe 80% Suppression)")
ax3.loglog(freq[mask_f], psds["hybrid"][mask_f, 1], color="orange", lw=1.8, ls="-.", label="Prototype Hybrid (72.8% 100-500 Hz)")
ax3.loglog(freq[mask_f], psds["advanced_veto"][mask_f, 1], color="blue", lw=2.2, label="Advanced Veto (75.3% 100-500 Hz)")

ax3.axvline(140.0, color="navy", ls=":", lw=1.2, label="140 Hz Acoustic Heave")
ax3.set_xlim(50, 3000)
ax3.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
ax3.set_ylabel(r"Center PSD [$\mu$m$^2$/Hz]", fontsize=11)
ax3.set_title("C. Center-Point Elevation PSD (50 Hz High-Passed)", fontsize=11.5, fontweight="bold")
ax3.grid(True, which="both", ls="--", alpha=0.35)
ax3.legend(loc="lower left", fontsize=9.5)

# -------------------------------------------------------------
# Panel D: Spatial Mean Welch PSD Comparison (f >= 50 Hz)
# -------------------------------------------------------------
ax4 = axes[1, 1]
ax4.loglog(freq[mask_f], psds["spatial_only"][mask_f, 0], color="gray", lw=1.8, ls="--", label="Spatial-Only")
ax4.loglog(freq[mask_f], psds["greedy"][mask_f, 0], color="crimson", lw=1.8, label="Legacy Greedy (3.1x Heave Suppression)")
ax4.loglog(freq[mask_f], psds["hybrid"][mask_f, 0], color="orange", lw=1.8, ls="-.", label="Prototype Hybrid (70.2% 100-500 Hz)")
ax4.loglog(freq[mask_f], psds["advanced_veto"][mask_f, 0], color="blue", lw=2.2, label="Advanced Veto (72.8% 100-500 Hz)")

ax4.axvline(140.0, color="navy", ls=":", lw=1.2, label="140 Hz Bulk Heave")
ax4.set_xlim(50, 3000)
ax4.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
ax4.set_ylabel(r"Spatial Mean PSD [$\mu$m$^2$/Hz]", fontsize=11)
ax4.set_title("D. Spatial Mean Elevation PSD (50 Hz High-Passed)", fontsize=11.5, fontweight="bold")
ax4.grid(True, which="both", ls="--", alpha=0.35)
ax4.legend(loc="lower left", fontsize=9.5)

fig.savefig(OUT_REPO, dpi=200)
fig.savefig(OUT_ARTIFACT, dpi=200)
plt.close(fig)

print(f"Comparison figure saved to:")
print(f"  1. {OUT_REPO}")
print(f"  2. {OUT_ARTIFACT}")
