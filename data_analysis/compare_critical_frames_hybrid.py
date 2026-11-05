"""Compare Critical Frames: Raw vs Current Cleaned vs New Hybrid Cleaned.

Evaluates 4 critical frames:
- Frame 1840: Right-edge phase breakdown & row 141 artificial step elimination.
- Frame 1740: Severe multi-cycle diagonal tear (row 80).
- Frame 1553: Extreme shear tear (row 100).
- Frame 55278: Gate-skipped edge defect (largest cliff component = 27 < 30).

Produces a multi-panel figure saved to thesis_notes/figures and conversation artifacts.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.export_cleaned_dataset import _restore_frame_hybrid, HYBRID_PARAMETERS
from data_analysis.pod.phase_unwrap_gradient_poisson import (
    DEFAULT_Q,
    DEFAULT_TAU,
    restore_frame_poisson,
)

RAW_PATH = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
CLN_PATH = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5")
HYB_PATH = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hybrid_hp50hz.hdf5")

OUT_FIG = Path("thesis_notes/figures/critical_frames_hybrid_comparison.png")
OUT_ARTIFACT = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/critical_frames_hybrid_comparison.png")


def plot_critical_frames():
    frames_to_plot = [
        (1840, 141, "Frame 1840: Right-Edge Phase Collapse & Step Elimination"),
        (1740, 80, "Frame 1740: Severe Diagonal Step Cliff Repair"),
        (1553, 100, "Frame 1553: Extreme Diagonal Shear Tear"),
        (55278, 188, "Frame 55278: Gate-Skipped Edge Defect (Component = 27 < 30)"),
    ]

    with h5py.File(RAW_PATH, "r") as f_r, h5py.File(CLN_PATH, "r") as f_c, h5py.File(HYB_PATH, "r") as f_h:
        x = np.asarray(f_r["meta/x"], dtype=float)
        y = np.asarray(f_r["meta/y"], dtype=float)
        extent = [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]

        fig, axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)

        for row_idx, (fn, slice_row, title_prefix) in enumerate(frames_to_plot):
            k = f"{fn:05d}"
            r = np.asarray(f_r["main"][k], dtype=float)
            c = np.asarray(f_c["main"][k], dtype=float)
            h = np.asarray(f_h["main"][k], dtype=float)

            # Pure spatial unwrap for un-filtered visual baseline
            res_cur_spatial = restore_frame_poisson(r, q=DEFAULT_Q, tau=4.0, method="integer")
            h_cur_sp = res_cur_spatial.restored
            h_hyb_sp, qc = _restore_frame_hybrid(r, DEFAULT_Q, 4.0)

            # Color limits based on spatial surfaces
            vmin = float(min(r.min(), h_cur_sp.min(), h_hyb_sp.min()))
            vmax = float(max(r.max(), h_cur_sp.max(), h_hyb_sp.max()))

            row_y_um = float(y[slice_row])

            # Col 0: Raw Surface
            im0 = axes[row_idx, 0].imshow(r, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
            axes[row_idx, 0].axhline(row_y_um, color="crimson", linestyle="--", linewidth=1.2)
            axes[row_idx, 0].set_title(f"Raw Surface ({fn})\nMax grad = {np.max(np.abs(np.diff(r, axis=1))):.1f} $\\mu$m", fontsize=10.5, fontweight="bold")
            axes[row_idx, 0].set_ylabel("y [$\\mu$m]")
            fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

            # Col 1: Current Cleaned Surface
            im1 = axes[row_idx, 1].imshow(h_cur_sp, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
            axes[row_idx, 1].axhline(row_y_um, color="crimson", linestyle="--", linewidth=1.2)
            mod_c = float(np.mean(h_cur_sp != r) * 100.0)
            axes[row_idx, 1].set_title(f"Current Cleaned (Unweighted Poisson)\nMax grad = {np.max(np.abs(np.diff(h_cur_sp, axis=1))):.2f} $\\mu$m | Mod: {mod_c:.1f}%", fontsize=10.5, fontweight="bold")
            fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

            # Col 2: New Hybrid Cleaned Surface
            im2 = axes[row_idx, 2].imshow(h_hyb_sp, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
            axes[row_idx, 2].axhline(row_y_um, color="crimson", linestyle="--", linewidth=1.2)
            mod_h = float(np.mean(h_hyb_sp != r) * 100.0)
            axes[row_idx, 2].set_title(f"New Hybrid Cleaned (Masked Harmonic)\nMax grad = {np.max(np.abs(np.diff(h_hyb_sp, axis=1))):.2f} $\\mu$m | Mod: {mod_h:.1f}%", fontsize=10.5, fontweight="bold")
            fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

            # Col 3: Cross Section Profile along slice_row
            ax_line = axes[row_idx, 3]
            ax_line.plot(x, r[slice_row], color="gray", linewidth=1.2, alpha=0.8, label="Raw $h_{\\text{raw}}$")
            ax_line.plot(x, h_cur_sp[slice_row], color="tab:blue", linewidth=1.4, linestyle="--", label="Current Cleaned")
            ax_line.plot(x, h_hyb_sp[slice_row], color="tab:green", linewidth=1.6, label="New Hybrid Cleaned")
            ax_line.set_title(f"Profile at Row {slice_row} ({row_y_um:.1f} $\\mu$m)", fontsize=10.5, fontweight="bold")
            ax_line.set_xlabel("x [$\\mu$m]")
            ax_line.set_ylabel("Height [$\\mu$m]")
            ax_line.grid(True, alpha=0.3)
            ax_line.legend(loc="best", fontsize=8)

            if row_idx == 3:
                for c_i in range(3):
                    axes[row_idx, c_i].set_xlabel("x [$\\mu$m]")

        plt.suptitle("Critical Frames Surface & Cross-Section Evaluation: Raw vs. Current vs. New Hybrid Cleaner", fontsize=14, fontweight="bold")
        
        for out_p in [OUT_FIG, OUT_ARTIFACT]:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=200)
            print(f"Saved figure to {out_p}")
        
        plt.close(fig)


if __name__ == "__main__":
    plot_critical_frames()

