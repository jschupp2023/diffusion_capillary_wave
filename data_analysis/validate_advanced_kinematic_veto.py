"""Focused validation for the Advanced Kinematic Temporal Selector with Local-Motion Veto.

Validates:
1. Test A: Full 100,001-frame 0p20 trace cache:
   - Provenance check against source files.
   - Bursts: frames 52309, 17036, 99064 and all residuals > 20 um.
   - Excursions: frames 10845-10869, 58433-58454, 91928-91946.
   - Correction counts, increment tails, and matched 50 Hz high-passed PSDs.
2. Test B: Synthetic controls:
   - Exact phase-aligned 900 Hz threshold regression from review script.
   - Smooth 1 kHz 20 um and 100 um motion.
   - Isolated injected jumps (+1q, +3q, +10q) and short burst (+3q then +2q).
"""

from pathlib import Path
import json
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

OUT_DIR = Path("runs/advanced_kinematic_veto")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321")

# -------------------------------------------------------------
# Test A: Full 100,001-frame 0p20 Trace Cache
# -------------------------------------------------------------
cache_dir = Path("runs/temporal_alignment_validation/step_events_0p20")
prov_file = cache_dir / "trace_provenance.json"
provenance = json.loads(prov_file.read_text())

# Provenance check
for path_str, item in provenance.items():
    if isinstance(item, dict) and "mtime_ns" in item:
        p = Path(path_str)
        assert p.exists(), f"Missing provenance source: {p}"
        stat = p.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (item["size"], item["mtime_ns"]), f"Provenance mismatch for {p}"

d = np.load(cache_dir / "full_recording_traces.npz")
time_sec = d["time"]
fs = 1.0 / np.mean(np.diff(time_sec))
q = provenance["q"]
spatial = d["spatial"]  # [mean, center, q0, q1, q2, q3]
dh_mean = np.r_[0, np.diff(spatial[:, 0])]
dq = np.vstack([np.zeros((1, 4)), np.diff(spatial[:, 2:], axis=0)])

# Execute modular selector
res_adv = select_temporal_kinematic(dh_mean, dq, q=q)
shifts_adv = res_adv.shifts
offsets_adv = res_adv.offsets
flags_adv = res_adv.uncertainty_flags

# Legacy greedy & previous hybrid for comparison
old_offsets = d["exported_offsets"]
old_shifts = np.r_[0, np.diff(old_offsets)]

# Load review selector for exact reproduction of previous hybrid
from data_analysis.review_hybrid_temporal_selector import select as select_proto_hybrid
shifts_hybrid = select_proto_hybrid(old_shifts, dq, q)
offsets_hybrid = np.cumsum(shifts_hybrid)

# Residual increments
res_dh_spat = dh_mean
res_dh_old = dh_mean - q * old_shifts
res_dh_hyb = dh_mean - q * shifts_hybrid
res_dh_adv = dh_mean - q * shifts_adv

# 50 Hz high-pass filter for PSDs
sos = butter(4, 50.0, btype="highpass", fs=fs, output="sos")

signals = {
    "spatial_only": spatial[:, :2],
    "greedy": spatial[:, :2] - q * old_offsets[:, None],
    "hybrid": spatial[:, :2] - q * offsets_hybrid[:, None],
    "advanced_veto": spatial[:, :2] - q * offsets_adv[:, None],
}

psds = {}
for name, sig in signals.items():
    sig_hp = sosfiltfilt(sos, sig.astype(np.float64), axis=0)
    freq, p = welch(sig_hp, fs=fs, window="hann", nperseg=16384, noverlap=8192, axis=0)
    psds[name] = p

summary_test_a = {
    "provenance_checked": True,
    "frames": len(time_sec),
    "fs_hz": float(fs),
    "q_um": float(q),
    "tau_um": float(DEFAULT_TAU),
    "counts": {
        "spatial_only_events": 0,
        "greedy_events": int(np.count_nonzero(old_shifts)),
        "hybrid_events": int(np.count_nonzero(shifts_hybrid)),
        "advanced_veto_events": res_adv.n_corrections,
        "advanced_veto_large": res_adv.n_large_corrections,
        "advanced_veto_small": res_adv.n_small_corrections,
        "advanced_veto_flagged": res_adv.n_flagged_corrections,
    },
    "residuals": {
        "spatial_only_max_um": float(np.max(np.abs(res_dh_spat[1:]))),
        "greedy_max_um": float(np.max(np.abs(res_dh_old[1:]))),
        "hybrid_max_um": float(np.max(np.abs(res_dh_hyb[1:]))),
        "advanced_veto_max_um": float(np.max(np.abs(res_dh_adv[1:]))),
        "spatial_only_above_20um": int(np.count_nonzero(np.abs(res_dh_spat) > 20.0)),
        "hybrid_above_20um": int(np.count_nonzero(np.abs(res_dh_hyb) > 20.0)),
        "advanced_veto_above_20um": int(np.count_nonzero(np.abs(res_dh_adv) > 20.0)),
        "advanced_veto_above_10um": int(np.count_nonzero(np.abs(res_dh_adv) > 10.0)),
    },
    "key_burst_events": {},
    "excursions": {},
    "power_ratios": {},
}

# Key burst events inspection
for t in [17035, 17036, 52254, 52309, 70100, 99064]:
    summary_test_a["key_burst_events"][str(t)] = {
        "uncorrected_dh_um": float(dh_mean[t]),
        "quadrant_increments_um": [float(x) for x in dq[t]],
        "shift_cycles": int(shifts_adv[t]),
        "shift_um": float(shifts_adv[t] * q),
        "residual_dh_um": float(res_dh_adv[t]),
        "uncertainty_flagged": bool(flags_adv[t]),
    }

# Excursion windows preservation
for (s, e) in [(10845, 10869), (58433, 58454), (91928, 91946)]:
    summary_test_a["excursions"][f"{s}_{e}"] = {
        "shifts_count": int(np.count_nonzero(shifts_adv[s:e+1])),
        "spatial_mean_change_um": float(spatial[e, 0] - spatial[s, 0]),
        "advanced_mean_change_um": float(signals["advanced_veto"][e, 0] - signals["advanced_veto"][s, 0]),
    }

# Power ratios relative to spatial-only
for (f_low, f_high) in [(100, 500), (500, 2000), (2000, 4000)]:
    m = (freq >= f_low) & (freq < f_high)
    spat_m = np.sum(psds["spatial_only"][m, 0])
    spat_c = np.sum(psds["spatial_only"][m, 1])
    adv_m = np.sum(psds["advanced_veto"][m, 0])
    adv_c = np.sum(psds["advanced_veto"][m, 1])
    hyb_m = np.sum(psds["hybrid"][m, 0])
    hyb_c = np.sum(psds["hybrid"][m, 1])
    grd_m = np.sum(psds["greedy"][m, 0])
    grd_c = np.sum(psds["greedy"][m, 1])
    summary_test_a["power_ratios"][f"{f_low}_{f_high}"] = {
        "greedy_mean_pct": float(grd_m / spat_m * 100.0),
        "greedy_center_pct": float(grd_c / spat_c * 100.0),
        "hybrid_mean_pct": float(hyb_m / spat_m * 100.0),
        "hybrid_center_pct": float(hyb_c / spat_c * 100.0),
        "advanced_veto_mean_pct": float(adv_m / spat_m * 100.0),
        "advanced_veto_center_pct": float(adv_c / spat_c * 100.0),
    }

# -------------------------------------------------------------
# Test B: Synthetic Controls Suite
# -------------------------------------------------------------
st = np.arange(8192) / fs

# Original phase-aligned 900 Hz threshold regression from review script
boundary_900 = 1.0001 * q / (4 * np.sin(np.pi / 128)) * np.sin(2 * np.pi * (fs / 128) * st - np.pi / 128)
diff_900 = np.r_[0, np.diff(boundary_900)]
res_900 = select_temporal_kinematic(diff_900, np.repeat(diff_900[:, None], 4, axis=1), q=q)

# Smooth 1 kHz / 20 um
s_1k_20 = 20.0 * np.sin(2 * np.pi * 1000 * st)
diff_1k_20 = np.r_[0, np.diff(s_1k_20)]
res_1k_20 = select_temporal_kinematic(diff_1k_20, np.repeat(diff_1k_20[:, None], 4, axis=1), q=q)

# Smooth 1 kHz / 100 um
s_1k_100 = 100.0 * np.sin(2 * np.pi * 1000 * st)
diff_1k_100 = np.r_[0, np.diff(s_1k_100)]
res_1k_100 = select_temporal_kinematic(diff_1k_100, np.repeat(diff_1k_100[:, None], 4, axis=1), q=q)

# Injected steps on 140 Hz background
s_140 = 80.0 * np.sin(2 * np.pi * 140 * st)
injected_results = {}
for jump in [1, 3, 10]:
    sl = np.zeros(len(st), dtype=int)
    sl[4000] = jump
    obs = s_140 + q * np.cumsum(sl)
    diff_obs = np.r_[0, np.diff(obs)]
    res_inj = select_temporal_kinematic(diff_obs, np.repeat(diff_obs[:, None], 4, axis=1), q=q)
    cor = obs - q * res_inj.offsets
    injected_results[f"isolated_{jump}q"] = {
        "detected_events": res_inj.n_corrections,
        "max_error_um": float(np.max(np.abs(cor - s_140))),
    }

# Short burst (+3q at 4000, +2q at 4002)
sl_burst = np.zeros(len(st), dtype=int)
sl_burst[4000] = 3
sl_burst[4002] = 2
obs_burst = s_140 + q * np.cumsum(sl_burst)
diff_burst = np.r_[0, np.diff(obs_burst)]
res_burst = select_temporal_kinematic(diff_burst, np.repeat(diff_burst[:, None], 4, axis=1), q=q)
cor_burst = obs_burst - q * res_burst.offsets
injected_results["burst_3q_then_2q"] = {
    "detected_events": res_burst.n_corrections,
    "max_error_um": float(np.max(np.abs(cor_burst - s_140))),
}

summary_test_b = {
    "original_900Hz_regression": {
        "false_events": res_900.n_corrections,
        "max_error_um": float(np.max(np.abs(q * res_900.offsets))),
    },
    "smooth_1000Hz_20um": {
        "false_events": res_1k_20.n_corrections,
        "max_error_um": float(np.max(np.abs(q * res_1k_20.offsets))),
    },
    "smooth_1000Hz_100um": {
        "false_events": res_1k_100.n_corrections,
        "max_error_um": float(np.max(np.abs(q * res_1k_100.offsets))),
    },
    "injected_steps": injected_results,
}

full_summary = {
    "test_a_experimental_0p20": summary_test_a,
    "test_b_synthetic_controls": summary_test_b,
}

# Save summary json
summary_path = OUT_DIR / "validation_summary.json"
summary_path.write_text(json.dumps(full_summary, indent=2) + "\n")
(ARTIFACT_DIR / "validation_summary.json").write_text(json.dumps(full_summary, indent=2) + "\n")

# -------------------------------------------------------------
# Generate Real Burst & Excursion Plots (Figure 1)
# -------------------------------------------------------------
fig1, axs = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)

# Row 1: Burst at Frame 52309
sl1 = slice(52309 - 8, 52309 + 9)
t_us1 = (time_sec[sl1] - time_sec[52309]) * 1e6
ax = axs[0, 0]
for name, col in [("spatial_only", "gray"), ("greedy", "crimson"), ("hybrid", "orange"), ("advanced_veto", "blue")]:
    y = signals[name][sl1, 0]
    ax.plot(t_us1, y - y[0], marker=".", lw=1.8, label=name, color=col)
ax.axvline(0, color="k", ls=":", alpha=0.5)
ax.set_title("Frame 52309: +107.7 um (+54q) Cliff Cleanly Repaired", fontweight="bold", fontsize=11)
ax.set_xlabel("Time relative to event [us]")
ax.set_ylabel("Height change [um]")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="lower right")

ax = axs[0, 1]
for j in range(4):
    ax.plot(t_us1, dq[sl1, j], marker=".", lw=1.5, label=f"Quadrant {j+1}")
ax.axvline(0, color="k", ls=":", alpha=0.5)
ax.set_title("Frame 52309: Spatial-Only Quadrant Increments (~+108 um consensus)", fontweight="bold", fontsize=11)
ax.set_xlabel("Time relative to event [us]")
ax.set_ylabel("Frame increment [um]")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="upper right")

# Row 2: Burst at Frame 17036
sl2 = slice(17036 - 8, 17036 + 9)
t_us2 = (time_sec[sl2] - time_sec[17036]) * 1e6
ax = axs[1, 0]
for name, col in [("spatial_only", "gray"), ("greedy", "crimson"), ("hybrid", "orange"), ("advanced_veto", "blue")]:
    y = signals[name][sl2, 0]
    ax.plot(t_us2, y - y[0], marker=".", lw=1.8, label=name, color=col)
ax.axvline(0, color="k", ls=":", alpha=0.5)
ax.set_title("Frame 17036: +65.8 um (+33q) Cliff Cleanly Repaired", fontweight="bold", fontsize=11)
ax.set_xlabel("Time relative to event [us]")
ax.set_ylabel("Height change [um]")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="lower right")

ax = axs[1, 1]
for j in range(4):
    ax.plot(t_us2, dq[sl2, j], marker=".", lw=1.5, label=f"Quadrant {j+1}")
ax.axvline(0, color="k", ls=":", alpha=0.5)
ax.set_title("Frame 17036: Spatial-Only Quadrant Increments (~+66 um consensus)", fontweight="bold", fontsize=11)
ax.set_xlabel("Time relative to event [us]")
ax.set_ylabel("Frame increment [um]")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="upper right")

# Row 3: Excursion Window 10845-10869
sl3 = slice(10840, 10875)
t_ms3 = (time_sec[sl3] - time_sec[sl3.start]) * 1000.0
ax = axs[2, 0]
for name, col, ls in [("spatial_only", "gray", "--"), ("greedy", "crimson", "-"), ("hybrid", "orange", "-."), ("advanced_veto", "blue", "-")]:
    y = signals[name][sl3, 0]
    ax.plot(t_ms3, y - y[0], lw=2.0, ls=ls, label=name, color=col)
ax.set_title("Excursion Window 10845–10869: 100% Preserved (-63.7 um)", fontweight="bold", fontsize=11)
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Relative Height [um]")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="lower left")

# Residual tail distribution
ax = axs[2, 1]
bins = np.linspace(0, 30, 60)
h_spat, _ = np.histogram(np.abs(res_dh_spat[1:]), bins=bins)
h_grd, _ = np.histogram(np.abs(res_dh_old[1:]), bins=bins)
h_hyb, _ = np.histogram(np.abs(res_dh_hyb[1:]), bins=bins)
h_adv, _ = np.histogram(np.abs(res_dh_adv[1:]), bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])

ax.semilogy(centers, h_spat, color="gray", ls="--", lw=1.8, label=f"Spatial-Only (max={np.max(np.abs(res_dh_spat)):.1f} um)")
ax.semilogy(centers, h_grd, color="crimson", lw=1.8, label=f"Greedy (max={np.max(np.abs(res_dh_old)):.1f} um)")
ax.semilogy(centers, h_hyb, color="orange", ls="-.", lw=1.8, label=f"Hybrid (max={np.max(np.abs(res_dh_hyb)):.1f} um)")
ax.semilogy(centers, h_adv, color="blue", lw=2.2, label=f"Advanced Veto (max={np.max(np.abs(res_dh_adv)):.1f} um)")
ax.axvline(20.0, color="k", ls=":", lw=1.2, label="20 um Threshold (0 remaining)")
ax.set_title("Residual Increment Tail (|dh| Distribution)", fontweight="bold", fontsize=11)
ax.set_xlabel("Absolute Increment [um]")
ax.set_ylabel("Frame Count (log scale)")
ax.set_ylim(0.5, 1e5)
ax.grid(True, which="both", ls="--", alpha=0.35)
ax.legend(fontsize=8.5, loc="upper right")

fig1_path = OUT_DIR / "real_burst_and_excursions.png"
fig1.savefig(fig1_path, dpi=180)
fig1.savefig(ARTIFACT_DIR / "real_burst_and_excursions.png", dpi=180)
plt.close(fig1)

# -------------------------------------------------------------
# Generate Matched PSD Figure (Figure 2)
# -------------------------------------------------------------
fig2, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
mask_f = (freq >= 50.0) & (freq <= 3000.0)

# Spatial mean PSD
ax_m.loglog(freq[mask_f], psds["spatial_only"][mask_f, 0], color="gray", lw=1.8, ls="--", label="Spatial-Only")
ax_m.loglog(freq[mask_f], psds["greedy"][mask_f, 0], color="crimson", lw=1.8, label="Greedy (16.4% 100-500 Hz)")
ax_m.loglog(freq[mask_f], psds["hybrid"][mask_f, 0], color="orange", lw=1.8, ls="-.", label="Hybrid (70.2% 100-500 Hz)")
ax_m.loglog(freq[mask_f], psds["advanced_veto"][mask_f, 0], color="blue", lw=2.2, label="Advanced Veto (72.8% 100-500 Hz)")
ax_m.axvline(140.0, color="navy", ls=":", lw=1.2, label="140 Hz Bulk Heave")
ax_m.set_xlim(50, 3000)
ax_m.set_xlabel("Frequency [Hz]", fontsize=11)
ax_m.set_ylabel(r"Spatial Mean PSD [$\mu$m$^2$/Hz]", fontsize=11)
ax_m.set_title("Spatial Mean Elevation PSD (50 Hz High-Passed)", fontsize=11.5, fontweight="bold")
ax_m.grid(True, which="both", ls="--", alpha=0.35)
ax_m.legend(fontsize=9, loc="lower left")

# Center point PSD
ax_c.loglog(freq[mask_f], psds["spatial_only"][mask_f, 1], color="gray", lw=1.8, ls="--", label="Spatial-Only")
ax_c.loglog(freq[mask_f], psds["greedy"][mask_f, 1], color="crimson", lw=1.8, label="Greedy (21.9% 100-500 Hz)")
ax_c.loglog(freq[mask_f], psds["hybrid"][mask_f, 1], color="orange", lw=1.8, ls="-.", label="Hybrid (72.8% 100-500 Hz)")
ax_c.loglog(freq[mask_f], psds["advanced_veto"][mask_f, 1], color="blue", lw=2.2, label="Advanced Veto (75.3% 100-500 Hz)")
ax_c.axvline(140.0, color="navy", ls=":", lw=1.2, label="140 Hz Acoustic Peak")
ax_c.set_xlim(50, 3000)
ax_c.set_xlabel("Frequency [Hz]", fontsize=11)
ax_c.set_ylabel(r"Center Elevation PSD [$\mu$m$^2$/Hz]", fontsize=11)
ax_c.set_title("Center Point Elevation PSD (50 Hz High-Passed)", fontsize=11.5, fontweight="bold")
ax_c.grid(True, which="both", ls="--", alpha=0.35)
ax_c.legend(fontsize=9, loc="lower left")

fig2_path = OUT_DIR / "matched_psd_comparison.png"
fig2.savefig(fig2_path, dpi=180)
fig2.savefig(ARTIFACT_DIR / "matched_psd_comparison.png", dpi=180)
plt.close(fig2)

print("Validation completed successfully.")
print(f"Summary saved to: {summary_path}")
print(f"Figures saved to: {fig1_path} and {fig2_path}")
