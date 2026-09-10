"""Build a compact PDF overview of all completed 2-D POD analyses.

The report contains:

1. two power levels per landscape page, with the combined spatial-mean plot,
   combined cumulative-energy plot, and a 16 x 16 rank-k subspace table;
2. the number of modes needed to capture 95% energy for every repetition and
   the corresponding trend across input powers; and
3. a cross-power subspace-similarity table using one reproducibly random
   repetition from each power.

Example
-------
    python pod_analysis_report.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from plot_pod_energy import load_cumulative_energy


DEFAULT_DATA_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
CONDITION_PATTERN = re.compile(r"0p\d+")
REPETITION_PATTERN = re.compile(r".*_rep(?P<number>\d+)$")


@dataclass(frozen=True)
class ConditionSummary:
    name: str
    value: float
    directory: Path
    pod_files: tuple[Path, ...]
    repetition_numbers: tuple[int, ...]
    energy_95_ranks: tuple[int | None, ...]
    within_similarity: np.ndarray
    selected_file: Path
    selected_repetition: int
    selected_basis: np.ndarray
    x: np.ndarray
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Reduced-data root (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PDF (default: <data-root>/pod_analysis_overview.pdf).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1_000,
        help="Stored POD rank to use (default: 1000).",
    )
    parser.add_argument(
        "--subspace-rank",
        type=int,
        default=10,
        help="Leading modes used for all subspace comparisons (default: 10).",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.95,
        help="Captured-energy target (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12_345,
        help="Seed for cross-power repetition selection (default: 12345).",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=2,
        help="Power rows on each overview page (default: 2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing report and companion CSV files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rank < 1:
        raise ValueError("--rank must be positive.")
    if args.subspace_rank < 1 or args.subspace_rank > args.rank:
        raise ValueError("--subspace-rank must lie between 1 and --rank.")
    if not 0 < args.energy_threshold <= 1:
        raise ValueError("--energy-threshold must lie in (0, 1].")
    if args.rows_per_page < 1:
        raise ValueError("--rows-per-page must be positive.")


def _condition_value(name: str) -> float:
    return float(name.replace("p", "."))


def _power_label(name: str) -> str:
    return f"{_condition_value(name):.2f} Vpp"


def _repetition_number(path: Path) -> int:
    match = REPETITION_PATTERN.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(f"Cannot determine repetition number from {path.parent.name}.")
    return int(match.group("number"))


def discover_conditions(data_root: Path, rank: int) -> list[tuple[Path, list[Path]]]:
    conditions: list[tuple[Path, list[Path]]] = []
    for directory in data_root.iterdir():
        if not directory.is_dir() or not CONDITION_PATTERN.fullmatch(directory.name):
            continue
        pod_files = sorted(
            directory.glob(f"Ca_ac*_rep*/pod_2d_r{rank}.h5"),
            key=_repetition_number,
        )
        if pod_files:
            conditions.append((directory, pod_files))
    conditions.sort(key=lambda item: _condition_value(item[0].name))
    if not conditions:
        raise FileNotFoundError(
            f"No condition folders containing rank-{rank} POD files found in "
            f"{data_root}."
        )
    return conditions


def load_basis(path: Path, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        for name in ("pod/modes", "grid/x", "grid/y"):
            if name not in handle:
                raise KeyError(f"Missing dataset {name!r} in {path}.")
        mode_dataset = handle["pod/modes"]
        if mode_dataset.ndim != 3 or len(mode_dataset) < rank:
            raise ValueError(
                f"Need {rank} modes in {path}, but pod/modes has shape "
                f"{mode_dataset.shape}."
            )
        basis = np.asarray(mode_dataset[:rank], dtype=np.float64).reshape(rank, -1)
        x = np.asarray(handle["grid/x"], dtype=np.float64)
        y = np.asarray(handle["grid/y"], dtype=np.float64)

    if not np.isfinite(basis).all():
        raise ValueError(f"POD basis contains nonfinite values in {path}.")
    gram_error = float(np.max(np.abs(basis @ basis.T - np.eye(rank))))
    if gram_error > 5e-5:
        raise ValueError(
            f"Leading modes are not orthonormal in {path}; maximum error "
            f"{gram_error:.3e}."
        )
    return basis, x, y


def _check_grid(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    context: str,
) -> None:
    if (
        reference_x.shape != x.shape
        or reference_y.shape != y.shape
        or not np.allclose(reference_x, x, rtol=1e-10, atol=1e-12)
        or not np.allclose(reference_y, y, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(f"Spatial grid mismatch for {context}.")


def subspace_similarity_matrix(bases: list[np.ndarray]) -> np.ndarray:
    """Compute normalized projection similarities in one stacked BLAS call."""
    if not bases:
        raise ValueError("At least one basis is required.")
    dimensions = {basis.shape for basis in bases}
    if len(dimensions) != 1:
        raise ValueError("All bases must have identical dimensions.")
    rank = bases[0].shape[0]
    stacked = np.concatenate(bases, axis=0)
    all_overlaps = stacked @ stacked.T
    count = len(bases)
    similarity = np.empty((count, count), dtype=np.float64)
    for row in range(count):
        row_slice = slice(row * rank, (row + 1) * rank)
        for column in range(row, count):
            column_slice = slice(column * rank, (column + 1) * rank)
            overlap = all_overlaps[row_slice, column_slice]
            score = float(np.sum(overlap**2) / rank)
            if score < -1e-10 or score > 1 + 1e-5:
                raise ValueError(
                    f"Invalid subspace similarity {score:.8g} at "
                    f"({row}, {column})."
                )
            score = float(np.clip(score, 0.0, 1.0))
            similarity[row, column] = score
            similarity[column, row] = score
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _energy_rank(path: Path, threshold: float) -> int | None:
    cumulative, _ = load_cumulative_energy(path)
    indices = np.flatnonzero(cumulative >= threshold)
    return int(indices[0] + 1) if len(indices) else None


def build_summaries(
    discovered: list[tuple[Path, list[Path]]],
    subspace_rank: int,
    energy_threshold: float,
    seed: int,
) -> list[ConditionSummary]:
    rng = np.random.default_rng(seed)
    summaries: list[ConditionSummary] = []
    global_x: np.ndarray | None = None
    global_y: np.ndarray | None = None

    for condition_directory, pod_files in discovered:
        repetition_numbers = tuple(_repetition_number(path) for path in pod_files)
        if len(repetition_numbers) != len(set(repetition_numbers)):
            raise ValueError(
                f"Duplicate repetition numbers in {condition_directory}."
            )

        selected_index = int(rng.integers(len(pod_files)))
        bases: list[np.ndarray] = []
        reference_x: np.ndarray | None = None
        reference_y: np.ndarray | None = None
        for path in pod_files:
            basis, x, y = load_basis(path, subspace_rank)
            if reference_x is None:
                reference_x, reference_y = x, y
            else:
                assert reference_y is not None
                _check_grid(reference_x, reference_y, x, y, str(path))
            bases.append(basis)

        assert reference_x is not None and reference_y is not None
        if global_x is None:
            global_x, global_y = reference_x, reference_y
        else:
            assert global_y is not None
            _check_grid(
                global_x,
                global_y,
                reference_x,
                reference_y,
                condition_directory.name,
            )

        energy_ranks = tuple(
            _energy_rank(path, energy_threshold) for path in pod_files
        )
        within_similarity = subspace_similarity_matrix(bases)
        summaries.append(
            ConditionSummary(
                name=condition_directory.name,
                value=_condition_value(condition_directory.name),
                directory=condition_directory,
                pod_files=tuple(pod_files),
                repetition_numbers=repetition_numbers,
                energy_95_ranks=energy_ranks,
                within_similarity=within_similarity,
                selected_file=pod_files[selected_index],
                selected_repetition=repetition_numbers[selected_index],
                selected_basis=bases[selected_index],
                x=reference_x,
                y=reference_y,
            )
        )
        print(
            f"Prepared {condition_directory.name}: {len(pod_files)} repetitions; "
            f"selected rep {repetition_numbers[selected_index]} for cross-power "
            "comparison.",
            flush=True,
        )
    return summaries


def _analysis_image(summary: ConditionSummary, filename: str) -> np.ndarray:
    path = summary.directory / "pod_analysis" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required overview image does not exist: {path}")
    return plt.imread(path)


def draw_similarity_table(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    *,
    title: str,
    text_size: float,
) -> None:
    """Draw the lower triangle of a symmetric matrix as a labeled table."""
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    masked = np.ma.array(matrix, mask=mask)
    color_map = plt.get_cmap("YlGnBu").copy()
    color_map.set_bad("white")
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
        labelsize=text_size,
        pad=1,
    )
    axis.set_xticks(np.arange(-0.5, count, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, count, 1), minor=True)
    axis.grid(which="minor", color="0.78", linewidth=0.35)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(count):
        for column in range(row + 1):
            value = matrix[row, column]
            color = "white" if value >= 0.55 else "black"
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=text_size,
                color=color,
            )
    if title:
        axis.set_title(title, fontsize=text_size + 2, pad=4)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)


def add_overview_pages(
    pdf: PdfPages,
    summaries: list[ConditionSummary],
    rows_per_page: int,
    subspace_rank: int,
) -> None:
    page_count = (len(summaries) + rows_per_page - 1) // rows_per_page
    for page_number in range(page_count):
        first = page_number * rows_per_page
        page_summaries = summaries[first : first + rows_per_page]
        figure, axes = plt.subplots(
            rows_per_page,
            3,
            figsize=(11.69, 8.27),
            squeeze=False,
            gridspec_kw={"wspace": 0.035, "hspace": 0.18},
        )
        figure.subplots_adjust(left=0.045, right=0.99, bottom=0.035, top=0.91)
        figure.suptitle(
            f"2-D POD overview ({page_number + 1}/{page_count})",
            fontsize=15,
            fontweight="bold",
        )
        figure.text(0.19, 0.925, "Spatial mean over time", ha="center", fontsize=10)
        figure.text(0.505, 0.925, "Cumulative POD energy", ha="center", fontsize=10)
        figure.text(
            0.825,
            0.925,
            f"Within-power subspace similarity ($k={subspace_rank}$)",
            ha="center",
            fontsize=10,
        )

        for row in range(rows_per_page):
            if row >= len(page_summaries):
                for axis in axes[row]:
                    axis.axis("off")
                continue
            summary = page_summaries[row]
            mean_axis, energy_axis, similarity_axis = axes[row]
            mean_axis.imshow(
                _analysis_image(summary, "all_repetitions_spatial_mean_over_time.png")
            )
            energy_axis.imshow(
                _analysis_image(summary, "all_repetitions_pod_energy.png")
            )
            mean_axis.axis("off")
            energy_axis.axis("off")
            draw_similarity_table(
                similarity_axis,
                summary.within_similarity,
                [str(number) for number in summary.repetition_numbers],
                title="",
                text_size=4.1,
            )
            row_center = 0.91 - (row + 0.5) * (0.875 / rows_per_page)
            figure.text(
                0.014,
                row_center,
                _power_label(summary.name),
                ha="center",
                va="center",
                rotation=90,
                fontsize=11,
                fontweight="bold",
            )
        pdf.savefig(figure)
        plt.close(figure)


def _energy_statistics(
    summary: ConditionSummary,
) -> tuple[float | None, float | None, int]:
    reached = np.asarray(
        [value for value in summary.energy_95_ranks if value is not None],
        dtype=np.float64,
    )
    if not len(reached):
        return None, None, 0
    return float(np.mean(reached)), float(np.std(reached)), len(reached)


def add_energy_page(
    pdf: PdfPages,
    summaries: list[ConditionSummary],
    energy_threshold: float,
    rank: int,
) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.35),
        left=0.055,
        right=0.98,
        bottom=0.08,
        top=0.91,
        hspace=0.30,
    )
    trend_axis = figure.add_subplot(grid[0])
    table_axis = figure.add_subplot(grid[1])
    threshold_percent = 100 * energy_threshold
    figure.suptitle(
        f"Modes required to capture {threshold_percent:g}% of snapshot energy",
        fontsize=15,
        fontweight="bold",
    )

    x_positions = np.arange(len(summaries))
    means: list[float] = []
    standard_deviations: list[float] = []
    valid_x: list[int] = []
    reached_counts: list[int] = []
    for index, summary in enumerate(summaries):
        mean, standard_deviation, reached_count = _energy_statistics(summary)
        reached_counts.append(reached_count)
        if mean is None or standard_deviation is None:
            continue
        valid_x.append(index)
        means.append(mean)
        standard_deviations.append(standard_deviation)

    trend_axis.errorbar(
        valid_x,
        means,
        yerr=standard_deviations,
        color="C0",
        marker="o",
        markersize=5,
        linewidth=1.5,
        capsize=3,
        label="mean ± population standard deviation",
    )
    trend_axis.set_yscale("log")
    trend_axis.set_xticks(
        x_positions,
        [_power_label(summary.name) for summary in summaries],
        rotation=35,
        ha="right",
    )
    trend_axis.set_ylabel("Required POD modes")
    trend_axis.set_xlabel("Input amplitude")
    trend_axis.grid(True, which="both", linestyle="--", alpha=0.35)
    trend_axis.legend(loc="upper left", fontsize=8)
    for index, summary in enumerate(summaries):
        mean, _, reached_count = _energy_statistics(summary)
        if mean is None:
            trend_axis.text(
                index,
                2.2,
                "N/A\n(0/{})".format(len(summary.energy_95_ranks)),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        else:
            trend_axis.annotate(
                f"{mean:.1f}\n({reached_count}/{len(summary.energy_95_ranks)})",
                (index, mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    repetition_columns = sorted(
        {number for summary in summaries for number in summary.repetition_numbers}
    )
    column_labels = [
        "Power",
        *(f"r{number}" for number in repetition_columns),
        "Mean*",
        "Reached",
    ]
    table_rows: list[list[str]] = []
    for summary, reached_count in zip(summaries, reached_counts, strict=True):
        by_repetition = dict(
            zip(summary.repetition_numbers, summary.energy_95_ranks, strict=True)
        )
        mean, _, _ = _energy_statistics(summary)
        table_rows.append(
            [
                _power_label(summary.name),
                *(
                    "N/A"
                    if by_repetition.get(number) is None
                    else str(by_repetition[number])
                    for number in repetition_columns
                ),
                "N/A" if mean is None else f"{mean:.1f}",
                f"{reached_count}/{len(summary.energy_95_ranks)}",
            ]
        )

    table_axis.axis("off")
    table = table_axis.table(
        cellText=table_rows,
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
        bbox=(0, 0.08, 1, 0.90),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(fontweight="bold")
        elif cell.get_text().get_text() == "N/A":
            cell.set_facecolor("#fbe3df")
        elif column in (0, len(column_labels) - 2, len(column_labels) - 1):
            cell.set_facecolor("#eef4f8")
    table_axis.set_title("Per-repetition threshold rank", fontsize=10, pad=1)
    figure.text(
        0.055,
        0.035,
        f"N/A means {threshold_percent:g}% was not reached by the {rank} stored "
        "modes. *Means use only repetitions that reached the threshold; the "
        "fraction in parentheses/Reached reports how many contributed.",
        fontsize=7.5,
    )
    pdf.savefig(figure)
    plt.close(figure)


def add_cross_power_page(
    pdf: PdfPages,
    summaries: list[ConditionSummary],
    matrix: np.ndarray,
    subspace_rank: int,
    seed: int,
) -> None:
    figure = plt.figure(figsize=(11.69, 8.27))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(2.35, 1.0),
        left=0.06,
        right=0.96,
        bottom=0.10,
        top=0.88,
        wspace=0.18,
    )
    table_axis = figure.add_subplot(grid[0])
    details_axis = figure.add_subplot(grid[1])
    labels = [_power_label(summary.name) for summary in summaries]
    draw_similarity_table(
        table_axis,
        matrix,
        labels,
        title=f"Cross-power projection similarity ($k={subspace_rank}$)",
        text_size=7.4,
    )
    table_axis.tick_params(axis="x", labelrotation=45)

    details_axis.axis("off")
    details_axis.set_title("Randomly selected basis", fontsize=11, pad=12)
    detail_lines = [
        f"Random seed: {seed}",
        "",
        *(
            f"{_power_label(summary.name):>8}: rep {summary.selected_repetition}"
            for summary in summaries
        ),
    ]
    details_axis.text(
        0.04,
        0.95,
        "\n".join(detail_lines),
        transform=details_axis.transAxes,
        va="top",
        family="monospace",
        fontsize=9.5,
        linespacing=1.45,
    )
    details_axis.text(
        0.04,
        0.28,
        "Similarity definition:\n"
        r"$S_{ij}=\|U_i^T U_j\|_F^2/k$"
        "\n\n0.0 = orthogonal subspaces\n1.0 = identical subspaces",
        transform=details_axis.transAxes,
        va="top",
        fontsize=9,
        linespacing=1.4,
    )
    figure.suptitle(
        "Similarity between input-power POD subspaces",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.045,
        "One rank-k basis is selected independently and reproducibly from each "
        "power. Values are shown to one decimal; the full-precision matrix is "
        "saved beside this PDF.",
        fontsize=8,
    )
    pdf.savefig(figure)
    plt.close(figure)


def _companion_paths(output_path: Path, subspace_rank: int) -> dict[str, Path]:
    prefix = output_path.with_suffix("")
    return {
        "cross": prefix.with_name(
            f"{prefix.name}_cross_power_similarity_k{subspace_rank}.csv"
        ),
        "selected": prefix.with_name(f"{prefix.name}_selected_repetitions.csv"),
        "energy": prefix.with_name(f"{prefix.name}_energy_95_ranks.csv"),
    }


def save_companion_csvs(
    paths: dict[str, Path],
    summaries: list[ConditionSummary],
    cross_power_similarity: np.ndarray,
) -> None:
    labels = [summary.name for summary in summaries]
    with paths["cross"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["power", *labels])
        for label, row in zip(labels, cross_power_similarity, strict=True):
            writer.writerow([label, *(f"{value:.10f}" for value in row)])

    with paths["selected"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["power", "repetition", "pod_file"])
        for summary in summaries:
            writer.writerow(
                [summary.name, summary.selected_repetition, summary.selected_file]
            )

    repetition_columns = sorted(
        {number for summary in summaries for number in summary.repetition_numbers}
    )
    with paths["energy"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["power", *(f"rep_{number}" for number in repetition_columns), "mean"]
        )
        for summary in summaries:
            by_repetition = dict(
                zip(summary.repetition_numbers, summary.energy_95_ranks, strict=True)
            )
            mean, _, _ = _energy_statistics(summary)
            writer.writerow(
                [
                    summary.name,
                    *(
                        "N/A"
                        if by_repetition.get(number) is None
                        else by_repetition[number]
                        for number in repetition_columns
                    ),
                    "N/A" if mean is None else f"{mean:.6f}",
                ]
            )


def make_report(args: argparse.Namespace) -> Path:
    validate_args(args)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Reduced-data root does not exist: {data_root}")
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_root / "pod_analysis_overview.pdf"
    )
    companion_paths = _companion_paths(output_path, args.subspace_rank)
    existing = [
        path for path in (output_path, *companion_paths.values()) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    discovered = discover_conditions(data_root, args.rank)
    summaries = build_summaries(
        discovered,
        args.subspace_rank,
        args.energy_threshold,
        args.seed,
    )
    cross_power_similarity = subspace_similarity_matrix(
        [summary.selected_basis for summary in summaries]
    )

    with PdfPages(output_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "2-D POD analysis overview"
        metadata["Author"] = "CapillaryWaveTurbulence analysis"
        metadata["Subject"] = (
            "POD energy, temporal means, and subspace similarities"
        )
        metadata["CreationDate"] = datetime.now()
        add_overview_pages(
            pdf,
            summaries,
            args.rows_per_page,
            args.subspace_rank,
        )
        add_energy_page(pdf, summaries, args.energy_threshold, args.rank)
        add_cross_power_page(
            pdf,
            summaries,
            cross_power_similarity,
            args.subspace_rank,
            args.seed,
        )

    save_companion_csvs(companion_paths, summaries, cross_power_similarity)
    print(f"Saved {output_path}")
    for path in companion_paths.values():
        print(f"Saved {path}")
    return output_path


def main() -> None:
    make_report(parse_args())


if __name__ == "__main__":
    main()
