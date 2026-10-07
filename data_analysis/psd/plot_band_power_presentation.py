"""Render title-free presentation panels from saved PSD band-power summaries.

The rank-10, rank-100, and rank-1000 CSV summaries are replotted with common
axes. A separate shared legend avoids repeating the same legend in every panel.

Example
-------
    python -m data_analysis.psd.plot_band_power_presentation
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from data_analysis.forcing_conditions import ca_ac_label


DEFAULT_COMPARISON_DIR = Path(
    "/home/jonas/ucsd_thesis/reduced_data/center_psd_comparisons"
)
RANKS = (10, 100, 1_000)
FIGURE_SIZE = (8.0, 7.0)
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 20
DEFAULT_MIN_FREQUENCY_HZ = 70.0


@dataclass(frozen=True)
class BandPowerSummary:
    conditions: tuple[str, ...]
    edges: np.ndarray
    ratio_mean: np.ndarray
    ratio_standard_deviation: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help=f"Band-power result root (default: {DEFAULT_COMPARISON_DIR}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    parser.add_argument(
        "--min-frequency-hz",
        type=float,
        default=DEFAULT_MIN_FREQUENCY_HZ,
        help="Left edge of the logarithmic frequency axis (default: 70 Hz).",
    )
    parser.add_argument(
        "--legend-labels",
        choices=("vpp", "ca-ac"),
        default="vpp",
        help="Labels for the shared legend (default: vpp).",
    )
    parser.add_argument(
        "--legend-only",
        action="store_true",
        help="Render only the shared legend.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> BandPowerSummary:
    """Load and validate one rank's saved aggregate band-power summary."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Band-power summary is empty: {path}")

    conditions = tuple(dict.fromkeys(row["input_power"] for row in rows))
    by_condition: dict[str, list[dict[str, str]]] = {name: [] for name in conditions}
    for row in rows:
        by_condition[row["input_power"]].append(row)

    reference_rows = by_condition[conditions[0]]
    lower = np.asarray(
        [float(row["lower_frequency_hz"]) for row in reference_rows]
    )
    upper = np.asarray(
        [float(row["upper_frequency_hz"]) for row in reference_rows]
    )
    if len(lower) == 0 or not np.allclose(lower[1:], upper[:-1]):
        raise ValueError(f"Frequency bands are empty or noncontiguous in {path}")
    edges = np.concatenate((lower[:1], upper))

    means = []
    standard_deviations = []
    for condition in conditions:
        condition_rows = by_condition[condition]
        condition_lower = np.asarray(
            [float(row["lower_frequency_hz"]) for row in condition_rows]
        )
        condition_upper = np.asarray(
            [float(row["upper_frequency_hz"]) for row in condition_rows]
        )
        if not (
            np.allclose(condition_lower, lower)
            and np.allclose(condition_upper, upper)
        ):
            raise ValueError(f"Frequency bands differ for {condition} in {path}")
        means.append([float(row["ratio_mean"]) for row in condition_rows])
        standard_deviations.append(
            [float(row["ratio_standard_deviation"]) for row in condition_rows]
        )

    ratio_mean = np.asarray(means, dtype=np.float64)
    ratio_standard_deviation = np.asarray(standard_deviations, dtype=np.float64)
    if not np.isfinite(ratio_mean).all() or not np.isfinite(
        ratio_standard_deviation
    ).all():
        raise ValueError(f"Band-power summary contains nonfinite values: {path}")
    return BandPowerSummary(
        conditions=conditions,
        edges=edges,
        ratio_mean=ratio_mean,
        ratio_standard_deviation=ratio_standard_deviation,
    )


def _power_label(condition: str, label_kind: str = "vpp") -> str:
    if label_kind == "ca-ac":
        return ca_ac_label(condition)
    return f"{float(condition.replace('p', '.')):.2f} Vpp"


def _colors(count: int) -> np.ndarray:
    return plt.get_cmap("turbo")(np.linspace(0.02, 0.98, count))


def plot_rank(
    summary: BandPowerSummary,
    output_path: Path,
    dpi: int,
    min_frequency_hz: float,
) -> None:
    """Save one title-free band-power panel without a redundant legend."""
    centers = np.sqrt(summary.edges[:-1] * summary.edges[1:])
    colors = _colors(len(summary.conditions))
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    for index, color in enumerate(colors):
        mean = summary.ratio_mean[index]
        standard_deviation = summary.ratio_standard_deviation[index]
        axis.plot(
            centers,
            mean,
            color=color,
            marker="o",
            markersize=6,
            linewidth=2.2,
        )
        axis.fill_between(
            centers,
            np.maximum(0.0, mean - standard_deviation),
            mean + standard_deviation,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    axis.axhline(1.0, color="black", linestyle="--", linewidth=2.2)
    for edge in summary.edges:
        axis.axvline(
            edge,
            color="0.25",
            linestyle="-",
            linewidth=1.25,
            alpha=0.50,
            zorder=0,
        )
    axis.set_xscale("log")
    axis.set_xlim(max(summary.edges[0], min_frequency_hz), summary.edges[-1])
    axis.set_ylim(0.0, 1.08)
    axis.set_xlabel(
        "Frequency band center [Hz]", fontsize=AXIS_LABEL_SIZE, labelpad=10
    )
    axis.set_ylabel(
        "Band-power ratio\n(reconstructed / experimental)",
        fontsize=22,
        labelpad=10,
    )
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.6,
        length=7,
    )
    axis.tick_params(axis="both", which="minor", width=1.1, length=4)
    axis.grid(True, which="both", linestyle="--", linewidth=0.9, alpha=0.28)
    for spine in axis.spines.values():
        spine.set_linewidth(1.4)
    figure.subplots_adjust(left=0.23, right=0.98, bottom=0.17, top=0.98)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def save_shared_legend(
    conditions: tuple[str, ...],
    output_path: Path,
    dpi: int,
    label_kind: str = "vpp",
) -> None:
    """Save a standalone legend shared by all three rank panels."""
    colors = _colors(len(conditions))
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=7,
            linewidth=2.5,
            label=_power_label(condition, label_kind),
        )
        for condition, color in zip(conditions, colors, strict=True)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2.5,
            label="ideal = 1",
        )
    )
    figure, axis = plt.subplots(figsize=(4.4, 5.8))
    axis.axis("off")
    axis.legend(
        handles=handles,
        title=r"$Ca_{ac}$" if label_kind == "ca-ac" else "Input power",
        loc="center",
        frameon=False,
        fontsize=19,
        title_fontsize=21,
        handlelength=2.8,
        labelspacing=0.75,
    )
    figure.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def render(
    comparison_directory: Path,
    dpi: int,
    legend_labels: str = "vpp",
    legend_only: bool = False,
    min_frequency_hz: float = DEFAULT_MIN_FREQUENCY_HZ,
) -> list[Path]:
    presentation_directory = comparison_directory / "presentation/band_power"
    presentation_directory.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    reference_summary: BandPowerSummary | None = None
    for rank in RANKS:
        summary = load_summary(
            comparison_directory
            / f"band_power_rank_{rank}/band_power_summary.csv"
        )
        if reference_summary is None:
            reference_summary = summary
        elif (
            summary.conditions != reference_summary.conditions
            or not np.allclose(summary.edges, reference_summary.edges)
        ):
            raise ValueError("Rank summaries do not use common powers and bands.")
        if not legend_only:
            output_path = presentation_directory / (
                f"all_powers__mean_band_power_ratio_r{rank}_presentation.png"
            )
            if min_frequency_hz >= summary.edges[-1]:
                raise ValueError(
                    "--min-frequency-hz must be below the largest band edge "
                    f"({summary.edges[-1]:g} Hz)."
                )
            plot_rank(summary, output_path, dpi, min_frequency_hz)
            outputs.append(output_path)
            print(f"Saved {output_path}")

    assert reference_summary is not None
    legend_path = presentation_directory / (
        "band_power_shared_legend_ca_ac_presentation.png"
        if legend_labels == "ca-ac"
        else "band_power_shared_legend_presentation.png"
    )
    save_shared_legend(
        reference_summary.conditions, legend_path, dpi, legend_labels
    )
    outputs.append(legend_path)
    print(f"Saved {legend_path}")
    return outputs


def main() -> None:
    args = parse_args()
    if args.dpi < 1 or not np.isfinite(args.min_frequency_hz) or args.min_frequency_hz <= 0:
        raise SystemExit("ERROR: --dpi and --min-frequency-hz must be positive.")
    try:
        render(
            args.comparison_dir.expanduser().resolve(),
            args.dpi,
            args.legend_labels,
            args.legend_only,
            args.min_frequency_hz,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
