"""Generate 4-panel publication-quality diagnostic figures for spatial integer cleaning.

Layout (2x2):
- (a) Raw surface h_raw(x, y) with marked cross-section row
- (b) Cleaned surface h_clean(x, y) (integer-cycle restored)
- (c) Cross-section height profile: Raw vs Cleaned along marked row
- (d) Discrete integer shift map: |h_clean - h_raw| = |k_int(x, y)| * q
"""

from __future__ import annotations

import argparse
import json
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

from data_analysis.pod.phase_unwrap_gradient_poisson import (
    DEFAULT_Q,
    DEFAULT_TAU,
    restore_frame_poisson,
)
from data_analysis.pod.pod_2d import _frame_key

RAW_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
OUT_DIR = Path("runs/frame_visualizations_4panel")


def render_4panel_frame(
    raw_frame: np.ndarray,
    fn: int,
    x_um: np.ndarray,
    y_um: np.ndarray,
    row: int | None = None,
    tau: float = DEFAULT_TAU,
    q: float = DEFAULT_Q,
    save_path: Path | None = None,
    description: str = "",
) -> dict:
    res = restore_frame_poisson(raw_frame, q=q, tau=tau, method="integer")
    cln_frame = res.restored
    diff = np.abs(cln_frame - raw_frame)

    raw_dx = np.abs(np.diff(raw_frame, axis=1))
    raw_dy = np.abs(np.diff(raw_frame, axis=0))
    raw_max_grad = float(max(np.max(raw_dx), np.max(raw_dy)))

    cln_dx = np.abs(np.diff(cln_frame, axis=1))
    cln_dy = np.abs(np.diff(cln_frame, axis=0))
    cln_max_grad = float(max(np.max(cln_dx), np.max(cln_dy)))

    # Auto-detect most dramatic cross-section row if not provided
    if row is None:
        if res.was_modified and np.max(diff) > 0:
            # Pick row with highest gradient or highest difference transition
            diff_y = np.abs(np.diff(diff, axis=0))
            row = int(np.argmax(np.sum(diff_y, axis=1)))
        else:
            # Clean frame: pick row with steepest natural wave gradient
            row = int(np.argmax(np.max(raw_dx, axis=1)))

    y_slice_um = float(y_um[row])

    # Shared color limits for raw and cleaned
    vmin = float(min(np.min(raw_frame), np.min(cln_frame)))
    vmax = float(max(np.max(raw_frame), np.max(cln_frame)))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), constrained_layout=True)

    # 1. Top-Left: Raw Surface
    im0 = axes[0, 0].imshow(
        raw_frame,
        origin="lower",
        cmap="viridis",
        extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    title_a = f"(a) Raw Surface (Frame {fn})\nmax |grad| = {raw_max_grad:.1f} um"
    axes[0, 0].set_title(title_a, fontweight="bold", fontsize=11)
    axes[0, 0].set_ylabel(r"y [$\mu$m]", fontsize=10)
    axes[0, 0].axhline(y_slice_um, color="crimson", ls="--", lw=1.5, label=f"Row {row} ({y_slice_um:.1f} um)")
    axes[0, 0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.colorbar(im0, ax=axes[0, 0], label=r"Elevation $h$ [$\mu$m]")

    # 2. Top-Right: Cleaned Surface
    im1 = axes[0, 1].imshow(
        cln_frame,
        origin="lower",
        cmap="viridis",
        extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    mod_text = f"Restored (Modified: {res.pct_pixels_modified:.1f}%)" if res.was_modified else "Untouched (100% Bit-for-Bit)"
    title_b = f"(b) Cleaned Surface (Integer-Restored, tau={tau:.1f} um)\nmax |grad| = {cln_max_grad:.1f} um — {mod_text}"
    axes[0, 1].set_title(title_b, fontweight="bold", fontsize=11)
    fig.colorbar(im1, ax=axes[0, 1], label=r"Elevation $h_{\mathrm{clean}}$ [$\mu$m]")

    # 3. Bottom-Left: Height Cross-Section
    axes[1, 0].plot(x_um, raw_frame[row], color="crimson", lw=1.8, label=r"Raw $h_{\mathrm{obs}}$", alpha=0.85)
    axes[1, 0].plot(
        x_um,
        cln_frame[row],
        color="royalblue",
        lw=2.0,
        ls="--",
        label=r"Cleaned $h_{\mathrm{clean}}$",
        alpha=0.95,
    )
    title_c = f"(c) Height Cross-Section at y = {y_slice_um:.1f} um (Row {row})"
    axes[1, 0].set_title(title_c, fontweight="bold", fontsize=11)
    axes[1, 0].set_xlabel(r"x [$\mu$m]", fontsize=10)
    axes[1, 0].set_ylabel(r"Elevation [$\mu$m]", fontsize=10)
    axes[1, 0].legend(fontsize=9.5, loc="best", framealpha=0.9)
    axes[1, 0].grid(True, ls=":", alpha=0.6)

    # 4. Bottom-Right: Integer Difference
    vmax_diff = max(2.0, float(np.max(diff)))
    im3 = axes[1, 1].imshow(
        diff,
        origin="lower",
        cmap="plasma",
        extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
        vmin=0.0,
        vmax=vmax_diff,
    )
    diff_max = float(np.max(diff))
    diff_title = (
        f"(d) Integer Shift |h_clean - h_raw| = m(x,y) * q\n"
        f"Max = {diff_max:.1f} um | Altered: {res.pct_pixels_modified:.1f}% pixels"
    )
    axes[1, 1].set_title(diff_title, fontweight="bold", fontsize=11)
    axes[1, 1].set_xlabel("x [$\\mu$m]", fontsize=10)
    fig.colorbar(im3, ax=axes[1, 1], label="|Difference| [$\\mu$m]")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")

    plt.close(fig)

    return {
        "fn": fn,
        "row": row,
        "y_slice_um": y_slice_um,
        "raw_max_grad": raw_max_grad,
        "cln_max_grad": cln_max_grad,
        "was_modified": bool(res.was_modified),
        "pct_pixels_modified": float(res.pct_pixels_modified),
        "max_diff_um": float(np.max(diff)),
        "description": description,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-h5", type=Path, default=RAW_H5)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        (2300, 97, "Clean control frame (steep physical wave, zero pixels altered)"),
        (1553, 53, "Catastrophic 46 um phase tear crossing the domain"),
        (1840, 95, "51% domain phase slip (16-cycle step cliff)"),
        (1740, 80, "45% domain tear with 40 um cliff"),
        (1009, 100, "24% quadrant phase slip (16 um cliff)"),
        (1240, 110, "12% localized edge tear (16 um cliff)"),
        (2760, 115, "41% domain phase cliff (36 um tear)"),
        (2200, 60, "39% domain phase cut (28 um tear)"),
    ]

    summary = []
    with h5py.File(args.raw_h5, "r") as f:
        main_grp = f["main"]
        x_um = np.asarray(f["meta"]["x"], dtype=np.float64)
        y_um = np.asarray(f["meta"]["y"], dtype=np.float64)

        for fn, row, desc in frames:
            key = _frame_key(fn)
            raw = np.asarray(main_grp[key], dtype=np.float32)
            out_file = args.out_dir / f"frame_comparison_{fn}_4panel.png"
            info = render_4panel_frame(
                raw_frame=raw,
                fn=fn,
                x_um=x_um,
                y_um=y_um,
                row=row,
                save_path=out_file,
                description=desc,
            )
            summary.append(info)

    summary_file = args.out_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\nAll {len(frames)} frames generated. Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
