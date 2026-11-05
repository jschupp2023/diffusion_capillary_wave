"""Diagnostic analysis of Phase Breakdown, Singularities, and Poisson Leakage for Frame 1840.

Demonstrates:
1. Raw height field and the broken boundary region.
2. Reconstructed wrapped phase psi(x,y) in [-pi, pi).
3. Phase residues (vortices) where curl != 0.
4. Poisson harmonic potential drift and the 0.5 cycle threshold crossing.
5. Zoomed contrast between clean fringes and fractured optical noise.
6. Row cross-section showing why the integer step is created.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from data_analysis.pod.phase_unwrap_gradient_poisson import (
    DEFAULT_Q,
    DEFAULT_TAU,
    restore_frame_poisson,
)

RAW_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
FIG_DIR = Path("thesis_notes/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "frame_1840_phase_breakdown_analysis.png"


def analyze_frame_1840():
    with h5py.File(RAW_H5, "r") as f:
        f1840 = f["main/01840"][:]
        x = f["meta/x"][:]
        y = f["meta/y"][:]

    q = DEFAULT_Q
    tau = DEFAULT_TAU
    M, N = f1840.shape

    # 1. Wrapped phase in [-pi, pi)
    phase = ((f1840 / q * 2 * np.pi + np.pi) % (2 * np.pi)) - np.pi

    # 2. Phase differences
    d_phase_x = ((np.diff(phase, axis=1) + np.pi) % (2 * np.pi)) - np.pi
    d_phase_y = ((np.diff(phase, axis=0) + np.pi) % (2 * np.pi)) - np.pi

    # 3. Residues (phase singularities where loop integral != 0)
    res_plaquette = (
        d_phase_x[:-1, :] 
        + d_phase_y[:, 1:] 
        - d_phase_x[1:, :] 
        - d_phase_y[:, :-1]
    )
    residues = np.round(res_plaquette / (2 * np.pi)).astype(int)
    pos_res = residues > 0
    neg_res = residues < 0

    # 4. Restorations
    res_int = restore_frame_poisson(f1840, q=q, tau=tau, method="integer")
    res_poi = restore_frame_poisson(f1840, q=q, tau=tau, method="poisson")
    diff_q = (f1840 - res_poi.restored) / q

    # Row 141 for cross-section
    row_idx = 141
    row_y_um = float(y[row_idx])

    # Plot 6-panel layout (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    extent = [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]

    # Panel (a): Raw Height
    im_a = axes[0, 0].imshow(f1840, extent=extent, cmap="viridis", aspect="equal")
    axes[0, 0].axhline(row_y_um, color="crimson", linestyle="--", linewidth=1.5, label=f"Row {row_idx} ({row_y_um:.1f} um)")
    # Draw rectangle over broken zone
    rect_a = Rectangle((float(x[150]), float(y[199])), float(x[199]-x[150]), float(y[0]-y[199]),
                       linewidth=2, edgecolor="orange", facecolor="none", linestyle=":")
    axes[0, 0].add_patch(rect_a)
    axes[0, 0].set_title("(a) Raw Surface Elevation $h_{\\text{raw}}(x, y)$", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("x [$\\mu$m]")
    axes[0, 0].set_ylabel("y [$\\mu$m]")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    cb_a = fig.colorbar(im_a, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cb_a.set_label("$h$ [$\\mu$m]")

    # Panel (b): Wrapped Phase
    im_b = axes[0, 1].imshow(phase, extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
    axes[0, 1].axhline(row_y_um, color="crimson", linestyle="--", linewidth=1.5)
    rect_b = Rectangle((float(x[150]), float(y[199])), float(x[199]-x[150]), float(y[0]-y[199]),
                       linewidth=2, edgecolor="orange", facecolor="none", linestyle=":")
    axes[0, 1].add_patch(rect_b)
    axes[0, 1].set_title(r"(b) Wrapped Phase $\psi(x, y) = \text{wrap}(2\pi h / q)$", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("x [$\\mu$m]")
    axes[0, 1].set_ylabel("y [$\\mu$m]")
    cb_b = fig.colorbar(im_b, ax=axes[0, 1], fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])
    cb_b.set_ticklabels([r"$-\pi$", r"$0$", r"$+\pi$"])
    cb_b.set_label(r"Phase $\psi$ [rad]")

    # Panel (c): Phase Singularities (Residues)
    # Background: grayscale gradient magnitude
    grad_mag = np.sqrt(d_phase_x[:-1, :]**2 + d_phase_y[:, :-1]**2)
    axes[0, 2].imshow(grad_mag, extent=extent, cmap="Greys", alpha=0.35, aspect="equal")
    # Overlay residues
    ry_p, rx_p = np.where(pos_res)
    ry_n, rx_n = np.where(neg_res)
    axes[0, 2].scatter(x[rx_p], y[ry_p], color="red", s=8, alpha=0.8, label=f"+1 Residue (n={len(rx_p)})")
    axes[0, 2].scatter(x[rx_n], y[ry_n], color="blue", s=8, alpha=0.8, label=f"-1 Residue (n={len(rx_n)})")
    axes[0, 2].axhline(row_y_um, color="crimson", linestyle="--", linewidth=1.5)
    axes[0, 2].set_title(r"(c) Singularity Residue Cloud ($\oint \nabla \psi \cdot d\mathbf{l} \neq 0$)", fontsize=12, fontweight="bold")
    axes[0, 2].set_xlabel("x [$\\mu$m]")
    axes[0, 2].set_ylabel("y [$\\mu$m]")
    axes[0, 2].legend(loc="upper left", fontsize=9)

    # Panel (d): Continuous Poisson Discrepancy & Threshold Contour
    im_d = axes[1, 0].imshow(diff_q, extent=extent, cmap="coolwarm", vmin=-2, vmax=2, aspect="equal")
    # Draw contour where diff_q crosses -0.5
    cs = axes[1, 0].contour(x, y, diff_q, levels=[-0.5, 0.5], colors=["black", "magenta"], linewidths=1.8)
    axes[1, 0].clabel(cs, inline=True, fmt=r"$\pm 0.5 q$", fontsize=9)
    axes[1, 0].axhline(row_y_um, color="crimson", linestyle="--", linewidth=1.5)
    axes[1, 0].set_title(r"(d) Poisson Error $(h_{\text{raw}} - h_{\text{poiss}})/q$ with $\pm 0.5$ Contour", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("x [$\\mu$m]")
    axes[1, 0].set_ylabel("y [$\\mu$m]")
    cb_d = fig.colorbar(im_d, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb_d.set_label(r"Cycles error $[(h - h_{\text{poiss}})/q]$")

    # Panel (e): Row 141 Cross-Section
    x_row = x[:]
    raw_slice = f1840[row_idx, :]
    poi_slice = res_poi.restored[row_idx, :]
    int_slice = res_int.restored[row_idx, :]

    axes[1, 1].plot(x_row, raw_slice, label="Raw $h_{\\text{raw}}$", color="gray", linewidth=1.2, alpha=0.85)
    axes[1, 1].plot(x_row, poi_slice, label="Continuous Poisson $h_{\\text{poiss}}$", color="darkorange", linewidth=1.5, linestyle=":")
    axes[1, 1].plot(x_row, int_slice, label="Integer Restored $h_{\\text{int}}$", color="navy", linewidth=1.6)

    # Annotate artificial step at col 73
    x_jump = float(x[73])
    axes[1, 1].annotate(
        r"Artificial $1\cdot q$ step" + "\n" + r"(diff_q crosses $-0.5$)",
        xy=(x_jump, int_slice[73]),
        xytext=(x_jump - 500, int_slice[73] + 8.0),
        arrowprops=dict(facecolor="crimson", shrink=0.05, width=1.5, headwidth=6),
        fontsize=9, fontweight="bold", color="crimson",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="crimson", alpha=0.9)
    )

    # Annotate optical breakdown zone
    axes[1, 1].axvspan(float(x[150]), float(x[199]), color="orange", alpha=0.15, label="Optical breakdown zone")
    axes[1, 1].set_title(f"(e) Row {row_idx} Height Profiles Across Domain", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("x [$\\mu$m]")
    axes[1, 1].set_ylabel("Height [$\\mu$m]")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="lower left", fontsize=8.5)

    # Panel (f): Fringes: Clean coherent vs Broken noise
    # Split panel into two sub-crops:
    # Left crop: x in [40, 90], y in [115, 165] (Clean wave fringes)
    # Right crop: x in [150, 200], y in [115, 165] (Optical noise)
    crop_clean = phase[115:165, 40:90]
    crop_broken = phase[115:165, 150:200]

    # Stitch side-by-side with a separator
    sep = np.full((50, 4), np.nan)
    stitched = np.hstack([crop_clean, sep, crop_broken])
    im_f = axes[1, 2].imshow(stitched, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1, 2].axvline(52, color="white", linewidth=3)
    axes[1, 2].text(25, -4, "Clean Wave Region\n(Parallel Fringes, 0 Singularities)", color="darkgreen", fontsize=9.5, fontweight="bold", ha="center")
    axes[1, 2].text(79, -4, "Broken Optical Zone\n(Fringe Dropout, 1000+ Residues)", color="crimson", fontsize=9.5, fontweight="bold", ha="center")
    axes[1, 2].set_xticks([25, 79])
    axes[1, 2].set_xticklabels(["Clean Crop", "Broken Crop"])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title(r"(f) Microscopic Wrapped Phase Morphology", fontsize=12, fontweight="bold")
    cb_f = fig.colorbar(im_f, ax=axes[1, 2], fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])
    cb_f.set_ticklabels([r"$-\pi$", r"$0$", r"$+\pi$"])
    cb_f.set_label(r"Phase $\psi$ [rad]")

    plt.suptitle("Frame 1840: Diagnostic Analysis of Optical Phase Breakdown & Poisson Harmonic Leakage", fontsize=14, fontweight="bold")
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    analyze_frame_1840()
