"""Render the saved raw-versus-POD energy summary for presentation slides.

The plotting data are read from ``power_rank_summary.csv``.  No surface fields
are loaded and no energy values are recomputed.

Example
-------
    python -m data_analysis.energy.plot_raw_pod_energy_presentation
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.forcing_conditions import CA_AC_TIMES_1E3


DEFAULT_COMPARISON_DIR = Path(
    "/home/jonas/ucsd_thesis/reduced_data/capillary_energy/"
    "raw_vs_native_pod_energy_overview"
)
COLORS = {
    "raw": "#222222",
    10: "#0072B2",
    100: "#E69F00",
    200: "#009E73",
    1000: "#CC79A7",
}
FIGURE_SIZE = (12.0, 7.2)
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 18


@dataclass(frozen=True)
class EnergySummary:
    powers: tuple[str, ...]
    ranks: tuple[int, ...]
    repetitions: np.ndarray
    raw_mean_nj: np.ndarray
    raw_sd_nj: np.ndarray
    pod_mean_nj: np.ndarray
    pod_sd_nj: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help=f"Saved comparison directory (default: {DEFAULT_COMPARISON_DIR}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    parser.add_argument(
        "--x-axis",
        choices=("vpp", "ca-ac"),
        default="vpp",
        help="Horizontal-axis quantity (default: vpp).",
    )
    return parser.parse_args()


def load_summary(path: Path) -> EnergySummary:
    """Load and validate the saved per-power, per-rank mean and SD table."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Energy summary is empty: {path}")

    powers = tuple(
        sorted(
            {row["power"] for row in rows},
            key=lambda value: float(value.replace("p", ".")),
        )
    )
    ranks = tuple(sorted({int(row["rank"]) for row in rows}))
    lookup = {(row["power"], int(row["rank"])): row for row in rows}
    if len(lookup) != len(rows) or len(rows) != len(powers) * len(ranks):
        raise ValueError(f"Energy summary is incomplete or has duplicates: {path}")

    repetitions = np.empty((len(ranks), len(powers)), dtype=int)
    raw_mean = np.empty((len(ranks), len(powers)), dtype=np.float64)
    raw_sd = np.empty_like(raw_mean)
    pod_mean = np.empty_like(raw_mean)
    pod_sd = np.empty_like(raw_mean)
    for rank_index, rank in enumerate(ranks):
        for power_index, power in enumerate(powers):
            row = lookup[power, rank]
            repetitions[rank_index, power_index] = int(row["repetitions"])
            raw_mean[rank_index, power_index] = float(row["raw_mean_J"])
            raw_sd[rank_index, power_index] = float(row["raw_sd_J"])
            pod_mean[rank_index, power_index] = float(row["pod_mean_J"])
            pod_sd[rank_index, power_index] = float(row["pod_sd_J"])

    arrays = (raw_mean, raw_sd, pod_mean, pod_sd)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"Energy summary contains nonfinite values: {path}")
    if any(np.any(array < 0) for array in arrays):
        raise ValueError(f"Energy means and SDs must be nonnegative: {path}")
    if np.any(repetitions < 1):
        raise ValueError(f"Repetition counts must be positive: {path}")
    if not (
        np.allclose(raw_mean, raw_mean[:1], rtol=1e-12, atol=0)
        and np.allclose(raw_sd, raw_sd[:1], rtol=1e-12, atol=0)
    ):
        raise ValueError("Raw statistics differ between duplicated rank rows.")

    return EnergySummary(
        powers=powers,
        ranks=ranks,
        repetitions=repetitions,
        raw_mean_nj=raw_mean[0] * 1e9,
        raw_sd_nj=raw_sd[0] * 1e9,
        pod_mean_nj=pod_mean * 1e9,
        pod_sd_nj=pod_sd * 1e9,
    )


def plot_summary(
    summary: EnergySummary,
    output_path: Path,
    dpi: int,
    x_axis: str = "vpp",
) -> None:
    """Save a title-free presentation plot while preserving all error bars."""
    if x_axis == "ca-ac":
        missing = [power for power in summary.powers if power not in CA_AC_TIMES_1E3]
        if missing:
            raise ValueError(
                "Missing Ca_ac conversion for condition(s): " + ", ".join(missing)
            )
        x = np.asarray(
            [CA_AC_TIMES_1E3[power] for power in summary.powers],
            dtype=np.float64,
        )
        x_tick_labels = [f"{value:.2f}" for value in x]
        x_label = r"$Ca_{ac}\ [10^{-3}]$"
    else:
        x = np.asarray(
            [float(power.replace("p", ".")) for power in summary.powers],
            dtype=np.float64,
        )
        x_tick_labels = [f"{value:.2f}" for value in x]
        x_label = "Input amplitude [Vpp]"

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)

    errorbar_style = {
        "fmt": "o-",
        "linewidth": 2.8,
        "markersize": 8,
        "capsize": 5,
        "capthick": 1.8,
        "elinewidth": 1.8,
    }
    axis.errorbar(
        x,
        summary.raw_mean_nj,
        yerr=summary.raw_sd_nj,
        color=COLORS["raw"],
        label="Raw measured surface",
        **errorbar_style,
    )
    for rank_index, rank in enumerate(summary.ranks):
        axis.errorbar(
            x,
            summary.pod_mean_nj[rank_index],
            yerr=summary.pod_sd_nj[rank_index],
            color=COLORS.get(rank, f"C{rank_index}"),
            label=f"POD rank {rank}",
            **errorbar_style,
        )

    axis.set_xticks(x, x_tick_labels)
    axis.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE, labelpad=10)
    axis.set_ylabel(
        "Mean exact capillary excess\nenergy [nJ]",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=12,
    )
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.6,
        length=7,
    )
    axis.grid(True, linewidth=1.0, alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.5)
    axis.spines["bottom"].set_linewidth(1.5)
    axis.legend(
        loc="upper left",
        fontsize=LEGEND_SIZE,
        frameon=True,
        framealpha=0.93,
        borderpad=0.55,
        labelspacing=0.45,
        handlelength=2.0,
    )

    figure.subplots_adjust(left=0.15, right=0.985, bottom=0.17, top=0.98)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def render(comparison_directory: Path, dpi: int, x_axis: str = "vpp") -> Path:
    summary = load_summary(comparison_directory / "power_rank_summary.csv")
    output_name = (
        "energy_over_ca_ac_presentation.png"
        if x_axis == "ca-ac"
        else "energy_over_power_presentation.png"
    )
    output_path = comparison_directory / output_name
    plot_summary(summary, output_path, dpi, x_axis=x_axis)
    print(
        f"Saved {output_path} "
        f"({len(summary.powers)} powers, {len(summary.ranks)} POD ranks, "
        f"{summary.repetitions.min()}-{summary.repetitions.max()} repetitions per point)"
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise SystemExit("ERROR: --dpi must be positive.")
    try:
        render(
            args.comparison_dir.expanduser().resolve(),
            args.dpi,
            x_axis=args.x_axis,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
