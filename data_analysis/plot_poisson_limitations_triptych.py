"""Plot three 0.20 Vpp frames motivating a masked harmonic reference.

Each row shows the raw height, height-implied wrapped phase, the current
Poisson-guided integer restoration, and its inferred integer-cycle field.
The figure is intended for ``thesis_notes/data_cleaning_notes.tex``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from data_analysis.pod.phase_unwrap_gradient_poisson import restore_frame_poisson
from data_analysis.pod.pod_2d import _frame_key


DEFAULT_RAW = Path(
    "/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5"
)
DEFAULT_OUTPUT = Path("thesis_notes/figures/poisson_limitations_three_frames.png")
DEFAULT_FRAMES = (1840, 23230, 67230)
DEFAULT_Q = 1.9969696
DEFAULT_TAU = 4.0

FRAME_DESCRIPTIONS = {
    1840: "localized optical breakdown",
    23230: "extended corrupted region",
    67230: "dense but coherent fringes",
}


def wrapped_phase(height: np.ndarray, q: float) -> np.ndarray:
    """Return the height-implied phase in (-pi, pi]."""
    return np.angle(np.exp(2j * np.pi * height / q))


def render(
    raw_path: Path,
    output: Path,
    frames: tuple[int, ...],
    q: float,
    tau: float,
) -> None:
    fig, axes = plt.subplots(
        len(frames),
        4,
        figsize=(13.8, 9.2),
        constrained_layout=True,
        squeeze=False,
    )

    phase_image = None
    with h5py.File(raw_path, "r") as handle:
        x = np.asarray(handle["meta/x"], dtype=np.float64)
        y = np.asarray(handle["meta/y"], dtype=np.float64)
        extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

        for row, frame_number in enumerate(frames):
            raw = np.asarray(
                handle["main"][_frame_key(frame_number)], dtype=np.float32
            )
            result = restore_frame_poisson(raw, q=q, tau=tau, method="integer")
            cleaned = np.asarray(result.restored, dtype=np.float64)
            cycles = np.asarray(result.cycles_inferred, dtype=np.int32)
            phase = wrapped_phase(raw, q)

            height_min = float(min(np.min(raw), np.min(cleaned)))
            height_max = float(max(np.max(raw), np.max(cleaned)))
            cycle_limit = max(1, int(np.max(np.abs(cycles))))

            raw_image = axes[row, 0].imshow(
                raw,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=height_min,
                vmax=height_max,
                interpolation="nearest",
            )
            phase_image = axes[row, 1].imshow(
                phase,
                origin="lower",
                extent=extent,
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="nearest",
            )
            axes[row, 2].imshow(
                cleaned,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=height_min,
                vmax=height_max,
                interpolation="nearest",
            )
            cycle_image = axes[row, 3].imshow(
                cycles,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                norm=TwoSlopeNorm(vmin=-cycle_limit, vcenter=0, vmax=cycle_limit),
                interpolation="nearest",
            )

            description = FRAME_DESCRIPTIONS.get(frame_number, "diagnostic frame")
            axes[row, 0].text(
                -0.32,
                0.5,
                f"Frame {frame_number}\n{description}",
                transform=axes[row, 0].transAxes,
                ha="center",
                va="center",
                rotation=90,
                fontsize=10,
                fontweight="bold",
            )
            axes[row, 2].text(
                0.02,
                0.03,
                f"modified: {result.pct_pixels_modified:.1f}%",
                transform=axes[row, 2].transAxes,
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
            axes[row, 3].text(
                0.02,
                0.03,
                rf"$m\in[-{cycle_limit},{cycle_limit}]$ cycles",
                transform=axes[row, 3].transAxes,
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

            height_bar = fig.colorbar(
                raw_image,
                ax=[axes[row, 0], axes[row, 2]],
                fraction=0.025,
                pad=0.012,
            )
            height_bar.set_label(r"$h$ [$\mu$m]", fontsize=9)
            height_bar.ax.tick_params(labelsize=8)

            cycle_bar = fig.colorbar(
                cycle_image,
                ax=axes[row, 3],
                fraction=0.046,
                pad=0.025,
            )
            cycle_bar.set_label("cycles", fontsize=9)
            cycle_bar.ax.tick_params(labelsize=8)

            for col in range(4):
                axes[row, col].set_aspect("equal")
                axes[row, col].tick_params(labelsize=8)
                if row == len(frames) - 1:
                    axes[row, col].set_xlabel(r"$x$ [$\mu$m]", fontsize=9)
                if col == 0:
                    axes[row, col].set_ylabel(r"$y$ [$\mu$m]", fontsize=9)

    column_titles = (
        r"(a) Raw height $h_{\mathrm{obs}}$",
        r"(b) Reconstructed wrapped phase $\phi_{\mathrm{w}}^{(h)}$",
        r"(c) Poisson-guided integer restoration",
        r"(d) Inferred integer field $m$",
    )
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10.5, fontweight="bold", pad=18)

    assert phase_image is not None
    phase_bar = fig.colorbar(
        phase_image,
        ax=axes[:, 1],
        fraction=0.025,
        pad=0.012,
        ticks=[-np.pi, 0.0, np.pi],
    )
    phase_bar.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    phase_bar.set_label("phase [rad]", fontsize=9)

    fig.suptitle(
        rf"Limitations of the global Poisson reference ($q={q:.6f}\,\mu$m, "
        rf"$\tau={tau:.1f}\,\mu$m)",
        fontsize=13,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = parser.parse_args()
    render(
        raw_path=args.raw_input,
        output=args.output,
        frames=tuple(args.frames),
        q=args.q,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
