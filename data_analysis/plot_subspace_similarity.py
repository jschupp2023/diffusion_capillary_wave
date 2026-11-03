"""Generate publication figure comparing the 10-mode POD subspaces of 0.08 Vpp and 0.20 Vpp."""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load precomputed data
p20 = np.load('runs/pod_modes_comparison_50hz/pod_results.npz')
modes_20 = p20['cln_modes'][:10].reshape(10, -1)  # (10, 2500)

# Compute 0.08 POD modes
import h5py
from scipy.signal import butter, sosfiltfilt
from data_analysis.pod.pod_2d import _frame_key, inspect_input, randomized_spatial_subspace, refine_pod, stratified_training_positions

path_08 = Path('/home/jonas/ucsd_thesis/11222025_h_0.08vpp_data_roi-none_cal-true.hdf5')
info = inspect_input(path_08, 0, None)
y_idx = np.arange(0, 200, 4)
x_idx = np.arange(0, 200, 4)
n_space = len(y_idx) * len(x_idx)
fs = 1.0 / np.mean(np.diff(info.time))

raw_traj = np.empty((100001, n_space), dtype=np.float32)
with h5py.File(path_08, 'r') as f:
    main = f['main']
    for i in range(100001):
        raw_traj[i] = np.asarray(main[_frame_key(i)], dtype=np.float32)[np.ix_(y_idx, x_idx)].reshape(-1)

sos = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
hp_traj = sosfiltfilt(sos, raw_traj.astype(np.float64), axis=0).astype(np.float32)

rng = np.random.default_rng(12345)
pos = stratified_training_positions(100001, 5000, rng)
snaps = hp_traj[pos]
snaps -= snaps.mean(axis=0, dtype=np.float64).astype(np.float32)[None, :]
tot_e = float(np.einsum('ij,ij->', snaps, snaps, dtype=np.float64))

cand_basis, _ = randomized_spatial_subspace(snaps, 25, 1, rng)
cand_coeff = snaps @ cand_basis
red_cov = cand_coeff.T.astype(np.float64) @ cand_coeff.astype(np.float64)
pod_08 = refine_pod(cand_basis, red_cov, tot_e, 10)
modes_08 = pod_08.modes[:, :10].T  # (10, 2500)

# Overlap matrix
M = modes_08.astype(np.float64) @ modes_20.astype(np.float64).T  # (10, 10)
M_abs = np.abs(M)
s = np.linalg.svd(M, compute_uv=False)
angles_deg = np.arccos(np.clip(s, 0, 1)) * 180.0 / np.pi
similarity = np.mean(s**2)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)

# 1. Heatmap
im = axes[0].imshow(M_abs, cmap="Blues", vmin=0, vmax=1.0)
axes[0].set_title(f"A. Cross-Mode Overlap $|\\langle \\phi_i^{{(0.08)}}, \\phi_j^{{(0.20)}} \\rangle|$ (Grassmann Overlap: {similarity*100:.1f}%)", fontsize=11.5, fontweight="bold")
axes[0].set_xlabel("0.20 Vpp POD Mode Index $j$", fontsize=11)
axes[0].set_ylabel("0.08 Vpp POD Mode Index $i$", fontsize=11)
axes[0].set_xticks(np.arange(10))
axes[0].set_xticklabels([f"M{k+1}" for k in range(10)])
axes[0].set_yticks(np.arange(10))
axes[0].set_yticklabels([f"M{k+1}" for k in range(10)])
cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label("Absolute Correlation $|M_{ij}|$", fontsize=10.5)

# Annotate values
for i in range(10):
    for j in range(10):
        val = M_abs[i, j]
        if val >= 0.20:
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.6 else "black", fontsize=8.5, fontweight="bold" if val > 0.7 else "normal")

# 2. Principal Angles
k_idx = np.arange(1, 11)
ax2 = axes[1]
ax2.plot(k_idx, s, "bo-", lw=2.0, ms=7, label="$\\cos(\\theta_k)$ (Singular Values of $M$)")
ax2.set_ylabel("Principal Cosine $\\cos(\\theta_k)$", color="blue", fontsize=11)
ax2.tick_params(axis="y", labelcolor="blue")
ax2.set_ylim(0.85, 1.01)
ax2.set_xticks(k_idx)
ax2.set_xlabel("Principal Angle Index $k$", fontsize=11)
ax2.grid(True, ls="--", alpha=0.35)

ax2_twin = ax2.twinx()
ax2_twin.plot(k_idx, angles_deg, "rs--", lw=1.8, ms=6, label="Principal Angle $\\theta_k$ [deg]")
ax2_twin.set_ylabel("Principal Angle $\\theta_k$ [degrees]", color="crimson", fontsize=11)
ax2_twin.tick_params(axis="y", labelcolor="crimson")
ax2_twin.set_ylim(-1, 25)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=10)
ax2.set_title(f"B. Principal Subspace Angles (9 of 10 angles $< 7^\\circ$, mean similarity = {similarity*100:.2f}%)", fontsize=11.5, fontweight="bold")

out_path = Path("runs/validation_0p04_0p08_cleaning/subspace_similarity_0p08_vs_0p20.png")
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved {out_path}")

