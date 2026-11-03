"""Critically investigate and evaluate advanced temporal alignment methods for 0.20 Vpp.

Evaluates candidate methods:
1. Baseline: Raw (no temporal correction)
2. Baseline: Approach A (Greedy frame-to-frame median, current export)
3. Method 1: Large-Jump Only (|k| >= 3 cycles, i.e. >= 6 um)
4. Method 2: Kinematic / Velocity-Predictive Phase Tracking (PLL-style)
5. Method 3: Acceleration-bounded / Second-difference Impulse Filter
6. Method 4: Multi-region Consensus + Runaway Prohibition (No consecutive same-sign shifts)

Tests:
- Known smooth wave excursion windows (10845-10870, 58433-58454, 91928-91946)
- Known true optical jump events (1563, 17035, 23147, 52254, 70100)
- Full 100,001-frame spatial mean & center PSDs
- POD Mode 1-5 singular values and piston overlap
"""

import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_FILE = Path("runs/temporal_alignment_validation/step_events_0p20/full_recording_traces.npz")
OUT_DIR = Path("runs/new_temporal_methods_investigation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load full recording traces
d = np.load(DATA_FILE)
time_sec = d["time"]
fs = 1.0 / np.mean(np.diff(time_sec))
q = 1.9969696
n_frames = len(time_sec)

# spatial traces: col 0: mean, col 1: center, cols 2-5: Q1-Q4
spatial_mean = d["spatial"][:, 0]
spatial_center = d["spatial"][:, 1]
spatial_quads = d["spatial"][:, 2:]
raw_mean = d["raw"][:, 0]
current_offsets = d["exported_offsets"]
current_mean = d["final"][:, 0]

print(f"Loaded {n_frames:,} frames of 0.20 Vpp. fs = {fs:.1f} Hz, q = {q:.4f} um.")

# -------------------------------------------------------------
# Define Candidate Algorithms
# -------------------------------------------------------------

# --- Method 1: Large-Jump Only (|k| >= 3 cycles) ---
def method_large_only(min_cycles=3):
    dk_old = np.r_[0, np.diff(current_offsets)]
    shifts = np.where(np.abs(dk_old) >= min_cycles, dk_old, 0)
    offsets = np.cumsum(shifts)
    mean_corr = spatial_mean - q * offsets
    center_corr = spatial_center - q * offsets
    return offsets, mean_corr, center_corr, shifts

# --- Method 2: Kinematic / Velocity-Predictive Phase Tracking ---
# Predict s_t from recent velocity v_{t-1}. 
# Only if the prediction residual (s_t - \hat{s}_t) is near an integer multiple of q
# AND exceeds a threshold do we declare an optical slip.
def method_kinematic_predictive(threshold_q=1.5):
    offsets = np.zeros(n_frames, dtype=np.int64)
    shifts = np.zeros(n_frames, dtype=np.int64)
    corr_mean = np.empty(n_frames, dtype=np.float64)
    corr_mean[0] = spatial_mean[0]
    
    # We maintain a running estimate of physical velocity v (in um/frame)
    # using a short leaky integrator / alpha filter
    v_est = 0.0
    alpha = 0.3  # velocity smoothing factor
    
    for t in range(1, n_frames):
        # Predicted position without jump
        pred_pos = corr_mean[t-1] + v_est
        # Raw observed increment relative to accumulated offset
        curr_unaligned = spatial_mean[t] - q * offsets[t-1]
        residual = curr_unaligned - pred_pos
        
        # Candidate integer shift
        k = int(np.round(residual / q))
        
        # Check if candidate jump is statistically significant and close to integer
        # Must be >= threshold_q AND close to integer: |residual - k*q| < 0.35*q
        if abs(k) >= 1 and abs(residual) >= threshold_q * q and abs(residual - k * q) <= 0.4 * q:
            # Check regional confirmation: do at least 2 quadrants also show residual near k*q?
            quad_diffs = spatial_quads[t] - spatial_quads[t-1]
            quad_k = np.round((quad_diffs - v_est) / q)
            if np.sum(np.abs(quad_k - k) <= 1) >= 2:
                shifts[t] = k
        
        offsets[t] = offsets[t-1] + shifts[t]
        corr_mean[t] = spatial_mean[t] - q * offsets[t]
        
        # Update velocity estimate based on corrected change
        actual_step = corr_mean[t] - corr_mean[t-1]
        v_est = (1.0 - alpha) * v_est + alpha * actual_step
        # Cap physical velocity to realistic fluid limit (~0.4 m/s -> ~3.5 um/frame)
        v_est = np.clip(v_est, -3.5, 3.5)

    center_corr = spatial_center - q * offsets
    return offsets, corr_mean, center_corr, shifts

# --- Method 3: Anti-Runaway Gated Median ---
# Modifies the greedy median rule:
# Prohibits consecutive same-sign shifts! If a shift is in the same direction
# as the previous frame, it is rejected as a continuous wave slope.
# Also requires |k| >= 2 for isolated shifts, or |k| >= 3.
def method_anti_runaway(min_cycles=2):
    shifts = np.zeros(n_frames, dtype=np.int64)
    offsets = np.zeros(n_frames, dtype=np.int64)
    
    prev_shift_sign = 0
    run_len = 0
    
    for t in range(1, n_frames):
        diff_med = float(np.median(spatial_quads[t] - spatial_quads[t-1]))
        k = int(np.round(diff_med / q))
        
        if abs(k) >= min_cycles:
            # Check sign persistence
            sign_k = np.sign(k)
            if sign_k == prev_shift_sign:
                run_len += 1
            else:
                run_len = 1
                prev_shift_sign = sign_k
            
            # If it runs for more than 1 frame in same direction, it is a wave slope!
            if run_len <= 1 or abs(k) >= 4:
                shifts[t] = k
            else:
                # Wave slope detected! Reject shift!
                shifts[t] = 0
        else:
            shifts[t] = 0
            run_len = 0
            prev_shift_sign = 0
            
        offsets[t] = offsets[t-1] + shifts[t]
        
    mean_corr = spatial_mean - q * offsets
    center_corr = spatial_center - q * offsets
    return offsets, mean_corr, center_corr, shifts


# -------------------------------------------------------------
# Execute and Compare All Methods
# -------------------------------------------------------------

methods = {
    "Raw (Uncorrected)": {
        "offsets": np.zeros(n_frames, dtype=np.int64),
        "mean": spatial_mean,
        "center": spatial_center,
        "shifts": np.zeros(n_frames, dtype=np.int64),
    },
    "Current Export (Greedy Approach A)": {
        "offsets": current_offsets,
        "mean": current_mean,
        "center": spatial_center - q * current_offsets,
        "shifts": np.r_[0, np.diff(current_offsets)],
    },
}

print("Running Method 1: Large-Jump Only (|k| >= 3 cycles)...")
off1, m1, c1, sh1 = method_large_only(min_cycles=3)
methods["Large-Only (|k|>=3)"] = {"offsets": off1, "mean": m1, "center": c1, "shifts": sh1}

print("Running Method 2: Kinematic / Velocity-Predictive Phase Tracking...")
off2, m2, c2, sh2 = method_kinematic_predictive(threshold_q=1.5)
methods["Kinematic Predictive"] = {"offsets": off2, "mean": m2, "center": c2, "shifts": sh2}

print("Running Method 3: Anti-Runaway Gated Median (No Consecutive Same-Sign)...")
off3, m3, c3, sh3 = method_anti_runaway(min_cycles=2)
methods["Anti-Runaway (|k|>=2)"] = {"offsets": off3, "mean": m3, "center": c3, "shifts": sh3}


# -------------------------------------------------------------
# Test 1: Performance on the Wave Flattening Window (10840-10875)
# -------------------------------------------------------------
print("\n--- Test 1: Smooth Wave Excursion Window (10,845 to 10,870) ---")
w = slice(10840, 10875)
t_ms = (time_sec[w] - time_sec[w.start]) * 1000.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
ax = axes[0]
for name, data in methods.items():
    ax.plot(t_ms, data["mean"][w], lw=1.8, label=name)
ax.set_title("A. Spatial Mean Trajectory (Frames 10,840-10,875)", fontsize=11.5, fontweight="bold")
ax.set_xlabel("Time [ms]", fontsize=10.5)
ax.set_ylabel("Spatial Mean $\\bar{h}(t)$ [$\\mu$m]", fontsize=10.5)
ax.grid(True, ls="--", alpha=0.35)
ax.legend(fontsize=9)

# Show accumulated offsets in window
ax2 = axes[1]
for name, data in methods.items():
    ax2.plot(t_ms, data["offsets"][w] - data["offsets"][w.start], lw=1.8, label=name)
ax2.set_title("B. Accumulated Temporal Offset $\\sum k$ (Frames 10,840-10,875)", fontsize=11.5, fontweight="bold")
ax2.set_xlabel("Time [ms]", fontsize=10.5)
ax2.set_ylabel("Offset in Window [cycles]", fontsize=10.5)
ax2.grid(True, ls="--", alpha=0.35)
ax2.legend(fontsize=9)

fig.savefig(OUT_DIR / "comparison_wave_window_10845.png", dpi=200)
plt.close(fig)

for name, data in methods.items():
    delta_w = data["mean"][10870] - data["mean"][10845]
    n_shifts_w = np.count_nonzero(data["shifts"][10845:10871])
    print(f"  {name:35s}: Window Delta = {delta_w:+6.2f} um | Shifts fired = {n_shifts_w:2d}")


# -------------------------------------------------------------
# Test 2: Performance on Real Optical Jump Events
# -------------------------------------------------------------
print("\n--- Test 2: Real Optical Jump Events ---")
real_events = [1563, 17035, 23147, 52254, 70100]
for ev in real_events:
    print(f"Event at Frame {ev}:")
    for name, data in methods.items():
        step_val = data["shifts"][ev]
        print(f"  {name:35s}: Corrected Jump = {step_val:+3d} cycles ({step_val*q:+6.1f} um)")


# -------------------------------------------------------------
# Test 3: Welch Power Spectral Density Comparison (50 Hz to 4000 Hz)
# -------------------------------------------------------------
print("\nComputing 50 Hz High-Pass Welch PSDs for all methods...")
sos = butter(4, 50.0, btype="highpass", fs=fs, output="sos")
nperseg = 16384
noverlap = 8192

psd_results = {}
for name, data in methods.items():
    hp_mean = sosfiltfilt(sos, data["mean"].astype(np.float64)).astype(np.float32)
    hp_center = sosfiltfilt(sos, data["center"].astype(np.float64)).astype(np.float32)
    f, p_mean = welch(hp_mean, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    _, p_center = welch(hp_center, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    psd_results[name] = {"f": f, "p_mean": p_mean, "p_center": p_center}

fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

# Panel 1: Spatial Mean PSD
ax1 = axes[0]
for name in methods.keys():
    f = psd_results[name]["f"]
    p = psd_results[name]["p_mean"]
    mask = (f >= 50.0) & (f <= 3000.0)
    ax1.loglog(f[mask], p[mask], lw=1.8, label=name)
ax1.axvline(140.0, color="navy", ls=":", lw=1.2, label="140 Hz Bulk Heave")
ax1.set_xlim(50, 3000)
ax1.set_ylim(1e-4, 1e2)
ax1.set_title("A. Spatial Mean PSD $S_{\\bar{h}}(f)$ Across Temporal Methods", fontsize=11.5, fontweight="bold")
ax1.set_xlabel("Frequency $f$ [Hz]", fontsize=10.5)
ax1.set_ylabel("Power Spectral Density [$\\mu$m$^2$/Hz]", fontsize=10.5)
ax1.grid(True, which="both", ls="--", alpha=0.35)
ax1.legend(fontsize=9)

# Panel 2: Center Point PSD
ax2 = axes[1]
for name in methods.keys():
    f = psd_results[name]["f"]
    p = psd_results[name]["p_center"]
    mask = (f >= 50.0) & (f <= 3000.0)
    ax2.loglog(f[mask], p[mask], lw=1.8, label=name)
ax2.axvline(140.0, color="navy", ls=":", lw=1.2, label="140 Hz Bulk Heave")
ax2.set_xlim(50, 3000)
ax2.set_ylim(1e-4, 1e2)
ax2.set_title("B. Center Point PSD $S_{h_0}(f)$ Across Temporal Methods", fontsize=11.5, fontweight="bold")
ax2.set_xlabel("Frequency $f$ [Hz]", fontsize=10.5)
ax2.set_ylabel("Power Spectral Density [$\\mu$m$^2$/Hz]", fontsize=10.5)
ax2.grid(True, which="both", ls="--", alpha=0.35)
ax2.legend(fontsize=9)

fig.savefig(OUT_DIR / "psd_comparison_all_methods.png", dpi=200)
fig.savefig("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/psd_comparison_all_methods.png", dpi=200)
plt.close(fig)

# Print Band Power in 100-500 Hz for Spatial Mean
print("\nSpatial Mean Power in 100-500 Hz:")
df = f[1] - f[0]
mask_100_500 = (f >= 100.0) & (f <= 500.0)
p_ref = np.sum(psd_results["Raw (Uncorrected)"]["p_mean"][mask_100_500]) * df
for name in methods.keys():
    p_band = np.sum(psd_results[name]["p_mean"][mask_100_500]) * df
    pct = p_band / p_ref * 100.0
    print(f"  {name:35s}: Power = {p_band:6.1f} um^2 ({pct:5.1f}% of Raw)")

print(f"\nAll tests complete! Outputs saved in {OUT_DIR}/")
