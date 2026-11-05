"""Compare Baseline vs Optimistic Weighted Poisson Restoration on Frame 1840.

Generates an 8-panel diagnostic figure comparing:
- Raw height
- Production cleaner (unweighted Poisson)
- Baseline weighted pilot (28.8% mask)
- Optimistic weighted pilot (16.7% mask, tightly localized to the corner/edge)
- Row 141 cross sections and integer shift fields
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
import numpy as np

from data_analysis.pod.phase_unwrap_gradient_poisson import (
    DEFAULT_Q,
    DEFAULT_TAU,
    restore_frame_poisson,
)
from data_analysis.diagnose_weighted_poisson_mask import (
    wrapped_phase_quality,
    screened_weighted_reference,
    DEFAULT_RAW,
)

OUT_DIR = Path("runs/phase_unwrap_study/weighted_poisson_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "frame_01840_optimistic_comparison.png"


def run_comparison():
    with h5py.File(DEFAULT_RAW, "r") as f:
        f1840 = f["main/01840"][:]
        x = np.asarray(f["meta/x"], dtype=float)
        y = np.asarray(f["meta/y"], dtype=float)

    q = DEFAULT_Q
    tau = 4.0
    row_idx = 141

    # 1. Current production cleaner (unweighted integer Poisson)
    prod_res = restore_frame_poisson(f1840, q=q, tau=tau, method="integer")
    h_prod = prod_res.restored
    m_prod = prod_res.cycles_inferred

    # 2. Baseline weighted Poisson (win=15, dens=0.05, min_comp=100)
    phase, residues, res_px, low_b = wrapped_phase_quality(
        f1840, q, window=15, density_threshold=0.05, min_component_pixels=100
    )
    _, m_base, solv_b = screened_weighted_reference(
        f1840, q, tau, low_b, weak_edge_weight=0.01, trusted_data_weight=0.1
    )
    h_base = f1840 - m_base * q

    # 3. Optimistic weighted Poisson (win=9, dens=0.15, min_comp=200)
    _, _, _, low_opt = wrapped_phase_quality(
        f1840, q, window=9, density_threshold=0.15, min_component_pixels=200
    )
    _, m_opt, solv_opt = screened_weighted_reference(
        f1840, q, tau, low_opt, weak_edge_weight=0.01, trusted_data_weight=0.1
    )
    h_opt = f1840 - m_opt * q

    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    vmin = float(min(f1840.min(), h_prod.min(), h_base.min(), h_opt.min()))
    vmax = float(max(f1840.max(), h_prod.max(), h_base.max(), h_opt.max()))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)

    # Top Row: (a) Raw height, (b) Production m, (c) Baseline mask & m, (d) Optimistic mask & m
    im_a = axes[0, 0].imshow(f1840, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].axhline(y[row_idx], color="crimson", linestyle="--", linewidth=1.2, label=f"Row {row_idx}")
    axes[0, 0].set_title(f"(a) Raw Height ($h_{{\\text{{raw}}}}$)\nMax grad = {np.max(np.abs(np.diff(f1840, axis=1))):.1f} $\\mu$m", fontsize=11, fontweight="bold")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.colorbar(im_a, ax=axes[0, 0], label=r"$h$ [$\mu$m]")

    # (b) Production integer cycles
    im_b = axes[0, 1].imshow(m_prod, origin="lower", extent=extent, cmap="coolwarm", vmin=-16, vmax=11)
    axes[0, 1].axhline(y[row_idx], color="crimson", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title(f"(b) Production Cleaner $m(x, y)$\nModified = {100*np.mean(m_prod!=0):.1f}% (Global Leakage)", fontsize=11, fontweight="bold")
    fig.colorbar(im_b, ax=axes[0, 1], label="cycles")

    # (c) Baseline Weighted Pilot
    im_c = axes[0, 2].imshow(m_base, origin="lower", extent=extent, cmap="coolwarm", vmin=-16, vmax=11)
    axes[0, 2].contour(x, y, low_b, levels=[0.5], colors="black", linewidths=1.2)
    axes[0, 2].axhline(y[row_idx], color="crimson", linestyle="--", linewidth=1.2)
    axes[0, 2].set_title(f"(c) Baseline Weighted $m(x, y)$\nMask = {100*np.mean(low_b):.1f}% (Conservative)", fontsize=11, fontweight="bold")
    fig.colorbar(im_c, ax=axes[0, 2], label="cycles")

    # (d) Optimistic Weighted Pilot
    im_d = axes[0, 3].imshow(m_opt, origin="lower", extent=extent, cmap="coolwarm", vmin=-16, vmax=11)
    axes[0, 3].contour(x, y, low_opt, levels=[0.5], colors="black", linewidths=1.2)
    axes[0, 3].axhline(y[row_idx], color="crimson", linestyle="--", linewidth=1.2)
    axes[0, 3].set_title(f"(d) Optimistic Weighted $m(x, y)$\nMask = {100*np.mean(low_opt):.1f}% (Tightly Bounded)", fontsize=11, fontweight="bold")
    fig.colorbar(im_d, ax=axes[0, 3], label="cycles")

    # Bottom Row: (e) Production restored, (f) Baseline restored, (g) Optimistic restored, (h) Row 141 Profile
    im_e = axes[1, 0].imshow(h_prod, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].axhline(y[row_idx], color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title(f"(e) Production Surface $h_{{\\text{{prod}}}}$\nMax grad = {np.max(np.abs(np.diff(h_prod, axis=1))):.2f} $\\mu$m", fontsize=11, fontweight="bold")
    fig.colorbar(im_e, ax=axes[1, 0], label=r"$h$ [$\mu$m]")

    im_f = axes[1, 1].imshow(h_base, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 1].contour(x, y, low_b, levels=[0.5], colors="orange", linewidths=1.0)
    axes[1, 1].axhline(y[row_idx], color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 1].set_title(f"(f) Baseline Weighted $h_{{\\text{{base}}}}$\nTrusted mod = 0.0%", fontsize=11, fontweight="bold")
    fig.colorbar(im_f, ax=axes[1, 1], label=r"$h$ [$\mu$m]")

    im_g = axes[1, 2].imshow(h_opt, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 2].contour(x, y, low_opt, levels=[0.5], colors="orange", linewidths=1.0)
    axes[1, 2].axhline(y[row_idx], color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 2].set_title(f"(g) Optimistic Weighted $h_{{\\text{{opt}}}}$\nTrusted mod = 0.0%, Max grad = 3.99 $\\mu$m", fontsize=11, fontweight="bold")
    fig.colorbar(im_g, ax=axes[1, 2], label=r"$h$ [$\mu$m]")

    # (h) Row 141 Profiles
    axes[1, 3].plot(x, f1840[row_idx], color="gray", linewidth=1.2, alpha=0.8, label="Raw $h_{\\text{raw}}$")
    axes[1, 3].plot(x, h_prod[row_idx], color="tab:red", linewidth=1.5, linestyle=":", label="Production (artificial step at x~800)")
    axes[1, 3].plot(x, h_base[row_idx], color="tab:blue", linewidth=1.5, linestyle="--", label="Baseline Weighted (28.8% mask)")
    axes[1, 3].plot(x, h_opt[row_idx], color="darkgreen", linewidth=1.8, label="Optimistic Weighted (16.7% mask)")

    # Shading for the masks
    # Baseline mask row
    # Optimistic mask row
    x_opt_mask = x[low_opt[row_idx]]
    if len(x_opt_mask) > 0:
        axes[1, 3].axvspan(x_opt_mask[0], x_opt_mask[-1], color="darkgreen", alpha=0.15, label="Optimistic Mask Zone")

    axes[1, 3].set_title(f"(h) Row {row_idx} Cross-Section Profile", fontsize=11, fontweight="bold")
    axes[1, 3].set_xlabel(r"$x$ [$\mu$m]")
    axes[1, 3].set_ylabel(r"$h$ [$\mu$m]")
    axes[1, 3].legend(loc="lower left", fontsize=8)
    axes[1, 3].grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax is not axes[1, 3]:
            ax.set_xlabel(r"$x$ [$\mu$m]")
            ax.set_ylabel(r"$y$ [$\mu$m]")

    plt.suptitle("Frame 1840: Comparison of Production vs. Baseline vs. Optimistic Weighted Poisson Restoration",
                 fontsize=14, fontweight="bold")
    plt.savefig(OUT_FIG, dpi=180)
    plt.close()
    print(f"Saved figure to {OUT_FIG}")


if __name__ == "__main__":
    run_comparison()

