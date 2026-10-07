"""Render title-free normalized-coordinate panels from saved trajectory arrays.

The default outputs show time steps 0 through 100 for repetition 9 at rank 20
for 0p08, 0p20, and 0p30.  Saved normalized coordinates are used directly;
normalization and shared-POD projection are not recomputed.

Example
-------
    python -m data_analysis.plot_normalized_coordinates_presentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


DEFAULT_ROOT = Path(
    "/home/jonas/ucsd_thesis/reduced_data/coordinate_trajectories"
)
DEFAULT_POWERS = ("0p08", "0p20", "0p30")
DEFAULT_MODES = (0, 1, 2, 3, 4, 10, 20)
FIGURE_SIZE = (6.0, 8.5)
TICK_LABEL_SIZE = 15
AXIS_LABEL_SIZE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Coordinate-trajectory root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--powers",
        nargs="+",
        default=list(DEFAULT_POWERS),
        help="Power labels to render (default: 0p08 0p20 0p30).",
    )
    parser.add_argument("--rep", type=int, default=9)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--first-step", type=int, default=0)
    parser.add_argument("--last-step", type=int, default=100)
    parser.add_argument(
        "--source-timesteps",
        type=int,
        default=200,
        help="Saved trajectory window used as the source (default: 200).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    return parser.parse_args()


def load_window(
    source_directory: Path,
    power: str,
    repetition: int,
    rank: int,
    first_step: int,
    last_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate an inclusive step window from a saved trajectory."""
    summary = json.loads((source_directory / "summary.json").read_text())
    expected = {
        "experiment": power,
        "repetition": repetition,
        "pod_rank": rank,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"Expected {key}={value!r}, found {summary.get(key)!r} in "
                f"{source_directory / 'summary.json'}"
            )

    archive_path = source_directory / "normalized_coordinates.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        steps = np.asarray(archive["step"], dtype=np.int64)
        modes = np.asarray(archive["modes"], dtype=np.int64)
        coordinates = np.asarray(archive["coordinates"], dtype=np.float64)

    if steps.ndim != 1 or coordinates.shape != (len(steps), len(modes)):
        raise ValueError(f"Unexpected coordinate-array shape in {archive_path}")
    if not np.array_equal(modes, np.asarray(DEFAULT_MODES)):
        raise ValueError(f"Expected modes {DEFAULT_MODES}, found {modes.tolist()}")
    if not np.array_equal(steps, np.arange(steps[0], steps[-1] + 1)):
        raise ValueError(f"Saved steps are not contiguous in {archive_path}")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Coordinates contain nonfinite values in {archive_path}")

    selected = (steps >= first_step) & (steps <= last_step)
    expected_steps = np.arange(first_step, last_step + 1)
    if not np.array_equal(steps[selected], expected_steps):
        raise ValueError(
            f"Saved trajectory does not contain every step {first_step}-{last_step}: "
            f"{archive_path}"
        )
    return steps[selected], modes, coordinates[selected]


def plot_window(
    steps: np.ndarray,
    coordinates: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    """Save one title-free seven-row panel without coordinate-name y labels."""
    figure, axes = plt.subplots(
        coordinates.shape[1],
        1,
        figsize=FIGURE_SIZE,
        sharex=True,
        squeeze=False,
    )
    for axis, series in zip(axes[:, 0], coordinates.T, strict=True):
        axis.plot(steps, series, color="#4C9FDE", linewidth=2.0)
        axis.axhline(0, color="0.35", linewidth=1.2, alpha=0.8)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=TICK_LABEL_SIZE,
            width=1.4,
            length=5,
        )
        axis.grid(True, linewidth=0.9, alpha=0.20)
        for spine in axis.spines.values():
            spine.set_linewidth(1.3)

    axes[-1, 0].set_xlim(int(steps[0]), int(steps[-1]))
    axes[-1, 0].set_xticks(np.arange(steps[0], steps[-1] + 1, 20))
    axes[-1, 0].set_xlabel("Time step", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    figure.subplots_adjust(
        left=0.16,
        right=0.96,
        bottom=0.11,
        top=0.99,
        hspace=0.12,
    )
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def render(
    root: Path,
    powers: list[str],
    repetition: int,
    rank: int,
    source_timesteps: int,
    first_step: int,
    last_step: int,
    dpi: int,
) -> list[Path]:
    output_directory = root / "presentation"
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for power in powers:
        source_directory = root / power / (
            f"r{rank}_rep{repetition}_t{source_timesteps}"
        )
        steps, modes, coordinates = load_window(
            source_directory,
            power,
            repetition,
            rank,
            first_step,
            last_step,
        )
        output_path = output_directory / (
            f"normalized_coordinates_{power}_rep{repetition}_"
            f"steps_{first_step}_{last_step}_presentation.png"
        )
        plot_window(steps, coordinates, output_path, dpi)
        outputs.append(output_path)
        print(
            f"Saved {output_path} "
            f"({len(steps)} samples, modes {modes.tolist()})"
        )
    return outputs


def main() -> None:
    args = parse_args()
    if args.rep < 1 or args.rank < 1 or args.source_timesteps < 1:
        raise SystemExit("ERROR: --rep, --rank, and --source-timesteps must be positive.")
    if args.first_step < 0 or args.last_step < args.first_step:
        raise SystemExit("ERROR: require 0 <= --first-step <= --last-step.")
    if args.dpi < 1:
        raise SystemExit("ERROR: --dpi must be positive.")
    if len(set(args.powers)) != len(args.powers):
        raise SystemExit("ERROR: --powers contains duplicate labels.")
    try:
        render(
            args.root.expanduser().resolve(),
            args.powers,
            args.rep,
            args.rank,
            args.source_timesteps,
            args.first_step,
            args.last_step,
            args.dpi,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
