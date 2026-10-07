"""Plot mean POD rank required for 95% energy in presentation style.

The means are read from the companion CSV written by
``pod_analysis_report.py``.  Conditions that did not reach the threshold are
omitted.  The horizontal positions use the experimental Ca_ac conversion,
not equally spaced input-amplitude categories.

Example
-------
    python -m data_analysis.pod.plot_energy_95_presentation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.forcing_conditions import CA_AC_TIMES_1E3


DEFAULT_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
DEFAULT_INPUT = DEFAULT_ROOT / "pod_analysis_overview_energy_95_ranks.csv"
DEFAULT_OUTPUT = DEFAULT_ROOT / (
    "pod_analysis_overview_energy_95_mean_ca_ac_presentation.png"
)
FIGURE_SIZE = (13.333, 7.5)
AXIS_LABEL_SIZE = 27
TICK_LABEL_SIZE = 13
ANNOTATION_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive.")
    return args


def load_means(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Energy-threshold table is empty: {path}")

    conditions: list[str] = []
    means: list[float] = []
    for row in rows:
        condition = row["power"]
        raw_mean = row["mean"].strip()
        if raw_mean.upper() == "N/A" or not raw_mean:
            continue
        if condition not in CA_AC_TIMES_1E3:
            raise ValueError(f"Missing Ca_ac conversion for {condition!r}.")
        value = float(raw_mean)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid mean rank for {condition}: {raw_mean!r}")
        conditions.append(condition)
        means.append(value)

    if not means:
        raise ValueError(f"No finite mean ranks were found in {path}")
    order = np.argsort([CA_AC_TIMES_1E3[name] for name in conditions])
    return [conditions[index] for index in order], np.asarray(means)[order]


def render(input_path: Path, output_path: Path, dpi: int) -> Path:
    conditions, means = load_means(input_path)
    x = np.asarray([CA_AC_TIMES_1E3[name] for name in conditions])

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(
        x,
        means,
        color="#0072B2",
        marker="o",
        markersize=10,
        linewidth=3.0,
    )
    axis.set_yscale("log")
    axis.set_xticks(x, [f"{value:.2f}" for value in x])
    axis.set_xlabel(r"$Ca_{ac}\ [10^{-3}]$", fontsize=AXIS_LABEL_SIZE, labelpad=12)
    axis.set_ylabel(
        "Mean POD modes required\nfor 95% captured energy",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=14,
    )
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.6,
        length=7,
    )
    axis.tick_params(axis="y", which="minor", width=1.1, length=4)
    axis.grid(True, which="major", linestyle="--", linewidth=1.2, alpha=0.4)
    axis.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.9, alpha=0.25)
    for spine in axis.spines.values():
        spine.set_linewidth(1.5)

    for x_value, mean in zip(x, means, strict=True):
        axis.annotate(
            f"{mean:.1f}",
            (x_value, mean),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )

    axis.set_xlim(x.min() - 0.10, x.max() + 0.12)
    axis.set_ylim(2.7, max(230.0, 1.35 * means.max()))
    figure.subplots_adjust(left=0.15, right=0.975, bottom=0.17, top=0.96)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Saved {output_path} ({len(means)} condition means)")
    return output_path


def main() -> None:
    args = parse_args()
    try:
        render(
            args.input.expanduser().resolve(),
            args.output.expanduser().resolve(),
            args.dpi,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
