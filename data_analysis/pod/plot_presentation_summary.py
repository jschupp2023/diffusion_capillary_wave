"""Create presentation-ready combined POD summary plots.

This is a lightweight alternative to rerunning ``pod_analysis.py``: it loads
only the saved cumulative-energy and spatial-mean arrays, and writes one pair
of title-free 16:9 figures per condition.

Example
-------
    python -m data_analysis.pod.plot_presentation_summary 0p08 0p20 0p35
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from data_analysis.pod.plot_pod_energy import load_cumulative_energy
from data_analysis.pod.pod_analysis import (
    DEFAULT_ROOT,
    MeanSeries,
    PodResult,
    _axis_label,
    _common_units,
    _line_colors,
    discover_results,
    load_spatial_mean,
    resolve_condition_directory,
)


FIGURE_SIZE = (13.333, 7.5)
AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "conditions",
        nargs="+",
        type=Path,
        help="Condition names below --root (for example 0p08), or full paths.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Reduced-data root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--rank", type=int, default=1_000, help="POD file rank (default: 1000)."
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    return parser.parse_args()


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.6,
        length=7,
    )
    ax.tick_params(axis="both", which="minor", width=1.2, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def _presentation_label(name: str, units: str) -> str:
    """Use compact unit notation so large labels fit cleanly on a slide."""
    normalized_units = units.strip().lower()
    if normalized_units in {"second", "seconds"}:
        units = "s"
    elif normalized_units in {"micron", "microns", "micrometer", "micrometers"}:
        units = r"$\mu$m"
    return _axis_label(name, units)


def plot_energy(
    energy_by_result: list[tuple[PodResult, np.ndarray]], output_path: Path, dpi: int
) -> None:
    """Save a title-free, presentation-sized cumulative-energy figure."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = _line_colors(len(energy_by_result))
    maximum_rank = 1
    for color, (_, cumulative) in zip(colors, energy_by_result, strict=True):
        ranks = np.arange(1, len(cumulative) + 1)
        maximum_rank = max(maximum_rank, len(cumulative))
        ax.plot(
            ranks,
            100 * cumulative,
            color=color,
            linewidth=2.6,
        )

    ax.set_xscale("log")
    ax.set_xlim(1, maximum_rank)
    ax.set_ylim(0, 100)
    ax.set_xlabel(
        "Number of retained POD modes, $r$",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=10,
    )
    ax.set_ylabel(
        "Cumulative captured energy", fontsize=AXIS_LABEL_SIZE, labelpad=10
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, which="major", linestyle="--", linewidth=1.2, alpha=0.45)
    ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.9, alpha=0.24)
    _style_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.17, top=0.97)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_spatial_mean(
    series_by_result: list[tuple[PodResult, MeanSeries]], output_path: Path, dpi: int
) -> None:
    """Save a title-free, presentation-sized spatial-mean figure."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    colors = _line_colors(len(series_by_result))
    for color, (_, series) in zip(colors, series_by_result, strict=True):
        ax.plot(
            series.time,
            series.spatial_mean,
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )

    time_units = _common_units(series_by_result, "time_units")
    mean_units = _common_units(series_by_result, "mean_units")
    ax.set_xlabel(
        _presentation_label("Time", time_units),
        fontsize=AXIS_LABEL_SIZE,
        labelpad=10,
    )
    ax.set_ylabel(
        _presentation_label("Spatial mean", mean_units),
        fontsize=AXIS_LABEL_SIZE,
        labelpad=10,
    )
    ax.grid(True, linestyle="--", linewidth=1.2, alpha=0.38)
    _style_axes(ax)
    # Leave room for five-digit negative tick labels at high forcing amplitude.
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.97)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_condition(condition: Path, root: Path, rank: int, dpi: int) -> list[Path]:
    condition_directory = resolve_condition_directory(condition, root)
    results, issues = discover_results(condition_directory, rank)
    for issue in issues:
        print(f"WARNING: {issue}", file=sys.stderr)
    if not results:
        raise ValueError(
            f"No usable rank-{rank} POD repetitions found in {condition_directory}."
        )

    energy_by_result: list[tuple[PodResult, np.ndarray]] = []
    mean_by_result: list[tuple[PodResult, MeanSeries]] = []
    for result in results:
        try:
            cumulative, _ = load_cumulative_energy(result.path)
            energy_by_result.append((result, cumulative))
            mean_by_result.append((result, load_spatial_mean(result.path)))
        except Exception as exc:
            print(
                f"WARNING: repetition {result.repetition_number} was skipped: {exc}",
                file=sys.stderr,
            )

    if not energy_by_result or not mean_by_result:
        raise ValueError(f"No complete plot data could be loaded for {condition_directory}.")

    output_directory = condition_directory / "pod_analysis"
    output_directory.mkdir(parents=True, exist_ok=True)
    condition_name = condition_directory.name
    energy_path = output_directory / (
        f"all_repetitions_pod_energy_presentation_{condition_name}.png"
    )
    mean_path = output_directory / (
        f"all_repetitions_spatial_mean_over_time_presentation_{condition_name}.png"
    )
    plot_energy(energy_by_result, energy_path, dpi)
    plot_spatial_mean(mean_by_result, mean_path, dpi)
    print(f"Saved {len(results)} repetitions: {energy_path}")
    print(f"Saved {len(results)} repetitions: {mean_path}")
    return [energy_path, mean_path]


def main() -> None:
    args = parse_args()
    if args.rank < 1 or args.dpi < 1:
        raise SystemExit("ERROR: --rank and --dpi must be positive.")
    try:
        for condition in args.conditions:
            render_condition(condition, args.root.expanduser(), args.rank, args.dpi)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
