"""Plot cumulative captured POD energy from a ``pod_2d.py`` output file.

Example
-------
    conda run -n capillarywave python plot_pod_energy.py \
        pod_2d_r5000_0p20.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="POD HDF5 file to plot.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image (default: <input_stem>_energy.png).",
    )
    parser.add_argument(
        "--title",
        help="Optional plot title (default: derived from the input filename).",
    )
    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Use a linear mode-number axis instead of the default logarithmic axis.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster output resolution (default: 200 dpi).",
    )
    return parser.parse_args()


def load_cumulative_energy(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load and validate cumulative POD energy and useful metadata."""
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"POD file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        if "pod/cumulative_energy_fraction" in handle:
            cumulative = np.asarray(
                handle["pod/cumulative_energy_fraction"], dtype=np.float64
            )
        elif "pod/singular_values" in handle:
            singular_values = np.asarray(
                handle["pod/singular_values"], dtype=np.float64
            )
            total_energy = handle["pod"].attrs.get(
                "total_preprocessed_snapshot_energy"
            )
            if total_energy is None:
                raise KeyError(
                    "The file has singular values but no total-energy metadata."
                )
            cumulative = np.cumsum(singular_values**2) / float(total_energy)
        else:
            raise KeyError(
                "The file contains neither pod/cumulative_energy_fraction "
                "nor pod/singular_values."
            )

        metadata: dict[str, object] = {
            "rank_attribute": handle.attrs.get("rank"),
            "candidate_rank": handle.attrs.get("candidate_rank"),
            "training_frames": handle.attrs.get("training_frames"),
            "spatial_mean_removed": handle.attrs.get(
                "instantaneous_spatial_mean_removed"
            ),
            "temporal_mean_removed": handle.attrs.get(
                "temporal_mean_field_removed"
            ),
        }

    if cumulative.ndim != 1 or len(cumulative) == 0:
        raise ValueError(
            f"Cumulative energy must be a nonempty 1-D array; got {cumulative.shape}."
        )
    if not np.isfinite(cumulative).all():
        raise ValueError("Cumulative energy contains NaN or infinite values.")
    if np.any(np.diff(cumulative) < -1e-12):
        raise ValueError("Cumulative energy is not monotonically nondecreasing.")
    if cumulative[0] < -1e-12 or cumulative[-1] > 1 + 1e-8:
        raise ValueError(
            f"Cumulative energy lies outside [0, 1]: "
            f"{cumulative[0]:.6g} to {cumulative[-1]:.6g}."
        )
    return cumulative, metadata


def _default_title(path: Path) -> str:
    match = re.search(r"_(0p\d+)\b", path.stem)
    if match:
        power = match.group(1).replace("p", ".")
        return f"POD cumulative energy, {power} Vpp"
    return "POD cumulative captured energy"


def _reported_filename_rank(path: Path) -> int | None:
    match = re.search(r"(?:^|_)r(\d+)(?:_|$)", path.stem)
    return int(match.group(1)) if match else None


def print_summary(
    path: Path,
    cumulative: np.ndarray,
    metadata: dict[str, object],
) -> None:
    n_modes = len(cumulative)
    rank_attribute = metadata["rank_attribute"]
    filename_rank = _reported_filename_rank(path)

    print(f"File: {path.resolve()}")
    print(f"Modes stored: {n_modes:,}")
    if rank_attribute is not None and int(rank_attribute) != n_modes:
        print(
            f"Warning: root rank attribute is {int(rank_attribute):,}, "
            f"but the energy array contains {n_modes:,} entries."
        )
    if filename_rank is not None and filename_rank != n_modes:
        print(
            f"Note: filename says r{filename_rank}, but the file contains "
            f"{n_modes:,} retained modes."
        )

    for rank in (1, 5, 10, 20, 50, 100, 500, 1_000, 2_000, 5_000):
        if rank <= n_modes:
            print(f"  r={rank:>5,}: {cumulative[rank - 1]:.6%}")

    for threshold in (0.80, 0.90, 0.95, 0.99):
        indices = np.flatnonzero(cumulative >= threshold)
        if len(indices):
            print(f"  {threshold:.0%} first reached at r={indices[0] + 1:,}")
        else:
            print(
                f"  {threshold:.0%} not reached "
                f"(r={n_modes:,}: {cumulative[-1]:.6%})"
            )


def plot_cumulative_energy(
    cumulative: np.ndarray,
    output_path: Path,
    title: str,
    logarithmic_x: bool,
    dpi: int,
) -> None:
    """Create and save the cumulative-energy figure."""
    ranks = np.arange(1, len(cumulative) + 1)
    energy_percent = 100 * cumulative

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ranks, energy_percent, color="C0", linewidth=2)
    if logarithmic_x:
        ax.set_xscale("log")
    ax.set_xlim(1, len(cumulative))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Number of retained POD modes, $r$")
    ax.set_ylabel("Cumulative captured energy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    if logarithmic_x:
        ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.2)

    # Mark standard thresholds only when the saved modes actually reach them.
    summary_lines: list[str] = []
    for threshold, color in ((80, "C2"), (90, "C3"), (95, "C4"), (99, "C5")):
        indices = np.flatnonzero(energy_percent >= threshold)
        if not len(indices):
            continue
        threshold_rank = int(indices[0] + 1)
        ax.scatter(
            threshold_rank,
            energy_percent[threshold_rank - 1],
            s=28,
            color=color,
            zorder=3,
        )
        summary_lines.append(f"{threshold}% first reached: $r={threshold_rank:,}$")

    ax.scatter(ranks[-1], energy_percent[-1], color="black", s=24, zorder=3)
    summary_lines.append(
        f"Final $r={len(cumulative):,}$: {energy_percent[-1]:.2f}%"
    )
    ax.text(
        0.97,
        0.05,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_energy.png")
    )
    cumulative, metadata = load_cumulative_energy(input_path)
    print_summary(input_path, cumulative, metadata)
    plot_cumulative_energy(
        cumulative,
        output_path,
        args.title or _default_title(input_path),
        logarithmic_x=not args.linear_x,
        dpi=args.dpi,
    )
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
