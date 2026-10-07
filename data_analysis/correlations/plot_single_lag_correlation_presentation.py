"""Render one saved single-lag correlation-distance matrix for a slide.

The saved pairwise-distance CSV is replotted directly.  No POD projection,
correlation matrix, or pairwise distance is recomputed.

Example
-------
    python -m data_analysis.correlations.plot_single_lag_correlation_presentation
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


DEFAULT_ANALYSIS_DIR = Path(
    "/home/jonas/ucsd_thesis/reduced_data/center_psd_comparisons/"
    "lagged_covariance/single_lag_rank_10"
)
FIGURE_SIZE = (9.0, 7.6)
TICK_LABEL_SIZE = 19
COLORBAR_TICK_SIZE = 17
COLORBAR_LABEL_SIZE = 20
CONDITION_LABEL_SIZE = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help=f"Saved single-lag analysis directory (default: {DEFAULT_ANALYSIS_DIR}).",
    )
    parser.add_argument(
        "--reference",
        type=int,
        default=1,
        help="Reference basis number to render (default: 1).",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=10,
        help="Saved single-lag distance in samples (default: 10).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    return parser.parse_args()


def load_distance_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Load and validate a labeled symmetric distance matrix."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Distance CSV is empty or malformed: {path}")

    column_labels = rows[0][1:]
    row_labels = [row[0] for row in rows[1:]]
    try:
        distance = np.asarray(
            [[float(value) for value in row[1:]] for row in rows[1:]],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError(f"Distance CSV contains a nonnumeric value: {path}") from exc

    if row_labels != column_labels:
        raise ValueError(f"Row and column labels differ in {path}")
    if distance.shape != (len(row_labels), len(row_labels)):
        raise ValueError(f"Distance matrix is not square in {path}")
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError(f"Distances must be finite and nonnegative in {path}")
    if not np.allclose(distance, distance.T, rtol=0, atol=1e-10):
        raise ValueError(f"Distance matrix is not symmetric in {path}")
    if not np.allclose(np.diag(distance), 0, rtol=0, atol=1e-10):
        raise ValueError(f"Distance-matrix diagonal is not zero in {path}")
    return row_labels, distance


def grouped_conditions(labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Extract contiguous forcing-condition groups from recording labels."""
    conditions = [label.split("/rep", maxsplit=1)[0] for label in labels]
    if any(condition not in CA_AC_TIMES_1E3 for condition in conditions):
        missing = sorted(set(conditions) - set(CA_AC_TIMES_1E3))
        raise ValueError(f"Missing Ca_ac conversions for: {', '.join(missing)}")
    unique = list(dict.fromkeys(conditions))
    groups = np.asarray([unique.index(condition) for condition in conditions])
    if np.any(np.diff(groups) < 0):
        raise ValueError("Recordings are not grouped contiguously by condition.")
    return unique, groups


def plot_distance(
    distance: np.ndarray,
    conditions: list[str],
    groups: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    """Save a title-free matrix with compact Ca_ac axis notation."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    image = axis.imshow(distance, cmap="viridis", vmin=0, interpolation="nearest")

    centers = [float(np.flatnonzero(groups == index).mean()) for index in range(len(conditions))]
    tick_labels = [f"{CA_AC_TIMES_1E3[condition]:.2f}" for condition in conditions]
    axis.set_xticks(centers, tick_labels, rotation=45, ha="right")
    axis.set_yticks(centers, tick_labels)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.5,
        length=6,
    )
    for boundary in np.flatnonzero(np.diff(groups)) + 0.5:
        axis.axhline(boundary, color="white", linewidth=1.0, alpha=0.75)
        axis.axvline(boundary, color="white", linewidth=1.0, alpha=0.75)
    for spine in axis.spines.values():
        spine.set_linewidth(1.5)

    colorbar = figure.colorbar(image, ax=axis, pad=0.035, fraction=0.055)
    colorbar.set_label(
        "RMS difference of two\ncorrelation matrices",
        fontsize=COLORBAR_LABEL_SIZE,
        labelpad=12,
    )
    colorbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        width=1.4,
        length=6,
    )
    colorbar.outline.set_linewidth(1.3)

    figure.text(
        0.13,
        0.965,
        r"$Ca_{ac}\ [10^{-3}]$",
        ha="left",
        va="top",
        fontsize=CONDITION_LABEL_SIZE,
    )
    figure.subplots_adjust(left=0.13, right=0.83, bottom=0.14, top=0.91)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def render(analysis_directory: Path, reference: int, lag: int, dpi: int) -> Path:
    source_path = analysis_directory / f"reference_{reference}_lag_{lag}_distances.csv"
    labels, distance = load_distance_csv(source_path)
    conditions, groups = grouped_conditions(labels)
    output_directory = analysis_directory / "presentation"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"reference_{reference}_lag_{lag}_distance_matrix_ca_ac_presentation.png"
    )
    plot_distance(distance, conditions, groups, output_path, dpi)
    print(
        f"Saved {output_path} "
        f"({len(labels)} recordings, conditions {conditions})"
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.reference < 1 or args.lag < 0 or args.dpi < 1:
        raise SystemExit("ERROR: --reference and --dpi must be positive; --lag cannot be negative.")
    try:
        render(
            args.analysis_dir.expanduser().resolve(),
            args.reference,
            args.lag,
            args.dpi,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
