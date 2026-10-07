"""Plot high-pass-filtered spatial means across repetitions of one condition.

Example::

    python -m data_analysis.pod.plot_highpass_spatial_mean 0p35 20

The full spatial-mean trajectory from each repetition is filtered before it is
plotted. By default the figure is written to ``<condition>/pod_analysis``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from data_analysis.energy.pod_gravitational_energy import highpass_spatial_mean
from data_analysis.pod.pod_analysis import (
    DEFAULT_ROOT,
    MeanSeries,
    discover_results,
    load_spatial_mean,
    plot_combined_spatial_mean,
    resolve_condition_directory,
)


def cutoff_tag(cutoff_hz: float) -> str:
    return f"{cutoff_hz:.12g}".replace(".", "p").replace("+", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "condition",
        type=Path,
        help="Forcing-amplitude folder below --root, for example 0p35, or its full path.",
    )
    parser.add_argument("cutoff_hz", type=float, help="High-pass cutoff frequency [Hz].")
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help=f"Reduced-data root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: <condition>/pod_analysis).",
    )
    parser.add_argument("--rank", type=int, default=1000, help="POD file rank (default: 1000).")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution (default: 200 dpi).")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Write a title-free, legend-free presentation version with enlarged text.",
    )
    parser.add_argument(
        "--max-repetitions", type=int,
        help="Use only the first N repetitions; useful for a smoke test.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if not np.isfinite(args.cutoff_hz) or args.cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be finite and positive.")
    if args.rank < 1 or args.dpi < 1:
        raise ValueError("--rank and --dpi must be positive.")
    if args.max_repetitions is not None and args.max_repetitions < 1:
        raise ValueError("--max-repetitions must be positive.")

    condition_directory = resolve_condition_directory(args.condition, args.root)
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else condition_directory / "pod_analysis"
    )
    results, issues = discover_results(condition_directory, args.rank)
    if args.max_repetitions is not None:
        results = results[: args.max_repetitions]
    for issue in issues:
        print(f"WARNING: {issue}", file=sys.stderr)
    if not results:
        raise ValueError(f"No usable rank-{args.rank} POD repetitions found in {condition_directory}.")

    filtered_by_result = []
    for result in results:
        try:
            series = load_spatial_mean(result.path)
            filtered = highpass_spatial_mean(
                series.spatial_mean, series.time, args.cutoff_hz
            )
            filtered_by_result.append(
                (
                    result,
                    MeanSeries(
                        series.time,
                        filtered,
                        series.time_units,
                        series.mean_units,
                    ),
                )
            )
            print(f"Filtered repetition {result.repetition_number}: {result.path.name}")
        except Exception as exc:
            print(
                f"WARNING: repetition {result.repetition_number} was skipped: {exc}",
                file=sys.stderr,
            )

    if not filtered_by_result:
        raise ValueError("No spatial-mean trajectory could be filtered.")

    output_directory.mkdir(parents=True, exist_ok=True)
    tag = cutoff_tag(args.cutoff_hz)
    if getattr(args, "presentation", False):
        from data_analysis.pod.plot_presentation_summary import plot_spatial_mean

        output_path = output_directory / (
            f"all_repetitions_spatial_mean_highpass_{tag}Hz_"
            f"presentation_{condition_directory.name}.png"
        )
        plot_spatial_mean(filtered_by_result, output_path, args.dpi)
    else:
        output_path = output_directory / (
            f"all_repetitions_spatial_mean_highpass_{tag}Hz_{condition_directory.name}.png"
        )
        plot_combined_spatial_mean(
            filtered_by_result,
            output_path,
            condition_directory.name,
            args.dpi,
            title=(
                f"Spatial mean high-pass filtered at {args.cutoff_hz:g} Hz "
                f"across repetitions — {condition_directory.name}"
            ),
        )
    print(f"Saved {len(filtered_by_result)} filtered repetitions: {output_path}")
    return output_path


def main() -> None:
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
