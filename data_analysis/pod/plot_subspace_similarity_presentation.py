"""Render title-free, presentation-ready POD subspace-similarity matrices.

The saved CSV matrices are used directly, so this command does not reload POD
bases or recompute any overlaps.

Example
-------
    python -m data_analysis.pod.plot_subspace_similarity_presentation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.forcing_conditions import ca_ac_labels


DEFAULT_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
REPETITION_PATTERN = re.compile(r".*_rep(?P<number>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Reduced-data root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    parser.add_argument(
        "--cross-labels",
        choices=("vpp", "ca-ac"),
        default="vpp",
        help="Labels for the cross-power matrix (default: vpp).",
    )
    parser.add_argument(
        "--cross-only",
        action="store_true",
        help="Render only the cross-power matrix.",
    )
    return parser.parse_args()


def load_similarity_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Load and validate a labeled symmetric similarity matrix."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Similarity CSV is empty or malformed: {path}")

    column_labels = rows[0][1:]
    row_labels = [row[0] for row in rows[1:]]
    try:
        matrix = np.asarray(
            [[float(value) for value in row[1:]] for row in rows[1:]],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError(f"Similarity CSV contains a nonnumeric value: {path}") from exc

    expected_shape = (len(row_labels), len(column_labels))
    if matrix.shape != expected_shape or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Similarity matrix is not square: {matrix.shape} in {path}")
    if row_labels != column_labels:
        raise ValueError(f"Row and column labels do not match in {path}")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(f"Similarity values must be finite and lie in [0, 1]: {path}")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-9):
        raise ValueError(f"Similarity matrix is not symmetric: {path}")
    return row_labels, matrix


def repetition_labels(labels: list[str]) -> list[str]:
    output = []
    for label in labels:
        match = REPETITION_PATTERN.fullmatch(label)
        if match is None:
            raise ValueError(f"Cannot extract repetition number from {label!r}.")
        output.append(match.group("number"))
    return output


def power_labels(labels: list[str]) -> list[str]:
    return [f"{float(label.replace('p', '.')):.2f} Vpp" for label in labels]


def plot_similarity(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    tick_size: float,
    text_size: float,
    rotate_x: float,
    left_margin: float,
    top_margin: float,
    dpi: int,
) -> None:
    """Save a title-free lower-triangular similarity heatmap."""
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    masked = np.ma.array(matrix, mask=mask)
    color_map = plt.get_cmap("YlGnBu").copy()
    color_map.set_bad("white")

    figure, axis = plt.subplots(figsize=(9, 9))
    axis.imshow(masked, vmin=0, vmax=1, cmap=color_map, interpolation="nearest")

    count = len(labels)
    axis.set_xticks(np.arange(count), labels=labels)
    axis.set_yticks(np.arange(count), labels=labels)
    axis.tick_params(
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        labelsize=tick_size,
        pad=6,
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=rotate_x,
        ha="left",
        rotation_mode="anchor",
    )

    axis.set_xticks(np.arange(-0.5, count, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, count, 1), minor=True)
    axis.grid(which="minor", color="0.72", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(count):
        for column in range(row + 1):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=text_size,
                color="white" if value >= 0.55 else "black",
            )

    for spine in axis.spines.values():
        spine.set_linewidth(1.5)
    figure.subplots_adjust(
        left=left_margin,
        right=0.985,
        bottom=0.035,
        top=top_margin,
    )
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def render(
    root: Path, dpi: int, cross_labels: str = "vpp", cross_only: bool = False
) -> list[Path]:
    within_csv = root / "0p20/pod_analysis/pairwise_subspace_similarity_k10.csv"
    cross_csv = root / "pod_analysis_overview_cross_power_similarity_k10.csv"
    within_output = root / (
        "0p20/pod_analysis/"
        "pairwise_subspace_similarity_k10_presentation_0p20.png"
    )
    cross_output = root / (
        "pod_analysis_overview_cross_power_similarity_k10_"
        + ("ca_ac_presentation.png" if cross_labels == "ca-ac" else "presentation.png")
    )

    outputs = []
    if not cross_only:
        within_labels, within_matrix = load_similarity_csv(within_csv)
        plot_similarity(
            within_matrix,
            repetition_labels(within_labels),
            within_output,
            tick_size=15,
            text_size=11,
            rotate_x=0,
            left_margin=0.075,
            top_margin=0.94,
            dpi=dpi,
        )
        outputs.append(within_output)
        print(f"Saved {within_output}")

    source_labels, cross_matrix = load_similarity_csv(cross_csv)
    display_labels = (
        ca_ac_labels(source_labels)
        if cross_labels == "ca-ac"
        else power_labels(source_labels)
    )
    plot_similarity(
        cross_matrix,
        display_labels,
        cross_output,
        tick_size=15,
        text_size=17,
        rotate_x=45,
        left_margin=0.22 if cross_labels == "ca-ac" else 0.17,
        top_margin=0.73 if cross_labels == "ca-ac" else 0.80,
        dpi=dpi,
    )
    print(f"Saved {cross_output}")
    outputs.append(cross_output)
    return outputs


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise SystemExit("ERROR: --dpi must be positive.")
    try:
        render(
            args.root.expanduser().resolve(),
            args.dpi,
            args.cross_labels,
            args.cross_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
