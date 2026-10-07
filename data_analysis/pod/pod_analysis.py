"""Create individual and across-repetition plots for one POD condition.

Expected input layout::

    REDUCED_DATA_ROOT/
      0p04/
        Ca_ac_..._rep1/
          pod_2d_r1000.h5
        Ca_ac_..._rep2/
          pod_2d_r1000.h5

The default output is ``<condition>/pod_analysis``. Each repetition receives
its own directory containing a cumulative-energy plot, a spatial-mean plot,
and images of the leading spatial POD modes. Two additional figures compare
the energy curves and spatial means from every successfully loaded repetition.
A labeled CSV stores pairwise projection similarities between the subspaces
spanned by the same leading modes that are plotted.

Examples
--------
Analyze ``/home/jonas/ucsd_thesis/reduced_data/0p04`` using the rank-1000 POD
file in each repetition::

    python pod_analysis.py 0p04

Select rank 1000 explicitly and save the first ten modes::

    python pod_analysis.py 0p04 --rank 1000 --modes 10

Also create a short middle-segment reconstruction video for every repetition::

    python pod_analysis.py 0p04 --videos

An explicit condition path can be supplied instead of a condition name::

    python pod_analysis.py /path/to/reduced_data/0p04
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from data_analysis.pod.plot_pod_energy import load_cumulative_energy, plot_cumulative_energy
from data_analysis.pod.plot_pod_modes import _validate_pod_file, save_mode_figure
from data_analysis.pod.pod_reconstruction_video import make_reconstruction_video


DEFAULT_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
POD_FILE_PATTERN = "pod_2d_r*.h5"
REPETITION_PATTERN = re.compile(r"Ca_ac.*_rep(?P<number>\d+)$")


@dataclass(frozen=True)
class PodResult:
    """One selected POD result and the repetition it represents."""

    repetition: str
    repetition_number: int
    path: Path
    rank: int


@dataclass(frozen=True)
class MeanSeries:
    """One saved spatial-mean time series."""

    time: np.ndarray
    spatial_mean: np.ndarray
    time_units: str
    mean_units: str


@dataclass(frozen=True)
class SubspaceBasis:
    """Leading POD modes on the grid used for subspace comparison."""

    result: PodResult
    vectors: np.ndarray
    x: np.ndarray
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "condition",
        type=Path,
        help=(
            "Condition name (for example 0p04) below --root, or the full path "
            "to a condition directory."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Reduced-data root used for a condition name (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <condition>/pod_analysis).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1_000,
        help="POD rank to analyze (default: 1000).",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=3,
        help="Number of leading spatial modes to plot per repetition (default: 3).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster figure resolution (default: 200 dpi).",
    )
    parser.add_argument(
        "--linear-energy-x",
        action="store_true",
        help="Use a linear mode-number axis instead of the default logarithmic axis.",
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Create a short centered reconstruction video for every repetition.",
    )
    parser.add_argument(
        "--video-seconds",
        type=float,
        default=10.0,
        help="Playback duration of each reconstruction video (default: 10 seconds).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Reconstruction video frame rate (default: 30 fps).",
    )
    parser.add_argument(
        "--video-stride",
        type=int,
        default=5,
        help="Source-frame step in reconstruction videos (default: 5).",
    )
    parser.add_argument(
        "--video-rank",
        type=int,
        help="Video reconstruction rank (default: all stored modes).",
    )
    parser.add_argument(
        "--max-repetitions",
        type=int,
        help="Analyze only the first N repetitions; useful for a test run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected inputs and output locations without creating plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rank is not None and args.rank < 1:
        raise ValueError("--rank must be positive.")
    if args.modes < 1:
        raise ValueError("--modes must be positive.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")
    if args.max_repetitions is not None and args.max_repetitions < 1:
        raise ValueError("--max-repetitions must be positive.")
    if args.video_seconds <= 0:
        raise ValueError("--video-seconds must be positive.")
    if args.video_fps < 1:
        raise ValueError("--video-fps must be positive.")
    if args.video_stride < 1:
        raise ValueError("--video-stride must be positive.")
    if args.video_rank is not None and args.video_rank < 1:
        raise ValueError("--video-rank must be positive.")


def resolve_condition_directory(condition: Path, root: Path) -> Path:
    """Resolve either an explicit directory or a condition name below root."""
    condition = condition.expanduser()
    if condition.is_dir():
        return condition.resolve()

    candidate = root.expanduser() / condition
    if candidate.is_dir():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Condition directory does not exist: tried {condition} and {candidate}."
    )


def _read_stored_rank(path: Path) -> int:
    """Read and validate the rank metadata needed during result discovery."""
    with h5py.File(path, "r") as handle:
        if "pod/modes" not in handle:
            raise KeyError(f"Missing dataset 'pod/modes' in {path}.")
        modes = handle["pod/modes"]
        if modes.ndim != 3 or len(modes) < 1:
            raise ValueError(f"Invalid pod/modes shape {modes.shape} in {path}.")
        stored_rank = int(handle.attrs.get("rank", len(modes)))
        if stored_rank != len(modes):
            raise ValueError(
                f"Root rank attribute is {stored_rank}, but pod/modes contains "
                f"{len(modes)} modes in {path}."
            )
    return stored_rank


def discover_results(
    condition_directory: Path,
    requested_rank: int | None,
) -> tuple[list[PodResult], list[str]]:
    """Select exactly one valid POD result from each repetition directory."""
    results: list[PodResult] = []
    issues: list[str] = []
    repetition_directories: list[tuple[int, Path]] = []

    for path in condition_directory.iterdir():
        if not path.is_dir():
            continue
        match = REPETITION_PATTERN.fullmatch(path.name)
        if match:
            repetition_directories.append((int(match.group("number")), path))

    repetition_directories.sort(key=lambda item: (item[0], item[1].name))
    for repetition_number, repetition_directory in repetition_directories:
        candidates: list[tuple[int, Path]] = []
        pod_files = sorted(repetition_directory.glob(POD_FILE_PATTERN))
        if not pod_files:
            issues.append(
                f"{repetition_directory.name}: no {POD_FILE_PATTERN!r} file found"
            )
            continue

        for path in pod_files:
            try:
                stored_rank = _read_stored_rank(path)
            except Exception as exc:
                issues.append(f"{repetition_directory.name}/{path.name}: {exc}")
                continue
            if requested_rank is None or stored_rank == requested_rank:
                candidates.append((stored_rank, path))

        if not candidates:
            rank_description = (
                f"rank {requested_rank}" if requested_rank is not None else "a valid rank"
            )
            issues.append(
                f"{repetition_directory.name}: no valid POD file with {rank_description}"
            )
            continue

        highest_rank = max(rank for rank, _ in candidates)
        selected = [path for rank, path in candidates if rank == highest_rank]
        if len(selected) != 1:
            issues.append(
                f"{repetition_directory.name}: found {len(selected)} POD files at "
                f"rank {highest_rank}; selection is ambiguous"
            )
            continue

        results.append(
            PodResult(
                repetition=repetition_directory.name,
                repetition_number=repetition_number,
                path=selected[0],
                rank=highest_rank,
            )
        )

    return results, issues


def _attribute_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_spatial_mean(path: Path) -> MeanSeries:
    """Load and validate time and instantaneous spatial mean from one POD file."""
    with h5py.File(path, "r") as handle:
        for dataset_name in ("grid/time", "preprocessing/frame_spatial_mean"):
            if dataset_name not in handle:
                raise KeyError(f"Missing dataset {dataset_name!r} in {path}.")

        time_dataset = handle["grid/time"]
        mean_dataset = handle["preprocessing/frame_spatial_mean"]
        time = np.asarray(time_dataset, dtype=np.float64)
        spatial_mean = np.asarray(mean_dataset, dtype=np.float64)
        time_units = _attribute_text(time_dataset.attrs.get("units"), "s")
        mean_units = _attribute_text(mean_dataset.attrs.get("units"), "")

    if time.ndim != 1 or spatial_mean.ndim != 1:
        raise ValueError(
            "grid/time and preprocessing/frame_spatial_mean must be one-dimensional; "
            f"got {time.shape} and {spatial_mean.shape} in {path}."
        )
    if len(time) != len(spatial_mean) or len(time) == 0:
        raise ValueError(
            f"Time and spatial mean lengths do not match in {path}: "
            f"{len(time)} versus {len(spatial_mean)}."
        )
    if not np.isfinite(time).all() or not np.isfinite(spatial_mean).all():
        raise ValueError(f"Time or spatial mean contains nonfinite values in {path}.")
    if np.any(np.diff(time) < 0):
        raise ValueError(f"Time values are not monotonically nondecreasing in {path}.")

    return MeanSeries(time, spatial_mean, time_units, mean_units)


def _axis_label(name: str, units: str) -> str:
    return f"{name} [{units}]" if units else name


def plot_spatial_mean(
    series: MeanSeries,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(series.time, series.spatial_mean, linewidth=0.8)
    ax.set_xlabel(_axis_label("Time", series.time_units))
    ax.set_ylabel(_axis_label("Spatial mean", series.mean_units))
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_modes(
    result: PodResult,
    output_directory: Path,
    count: int,
    dpi: int,
) -> tuple[int, SubspaceBasis]:
    """Save leading mode images and retain their flattened basis vectors."""
    with h5py.File(result.path, "r") as handle:
        modes, singular_values, x, y, x_units, y_units = _validate_pod_file(
            handle, result.path
        )
        saved_count = min(count, len(modes))
        width = max(4, len(str(len(modes))))
        basis_vectors = np.empty(
            (saved_count, modes.shape[1] * modes.shape[2]), dtype=np.float64
        )
        for mode_index in range(saved_count):
            mode_number = mode_index + 1
            mode = np.asarray(modes[mode_index], dtype=np.float32)
            basis_vectors[mode_index] = mode.reshape(-1)
            save_mode_figure(
                mode,
                float(singular_values[mode_index]),
                mode_number,
                x,
                y,
                x_units,
                y_units,
                output_directory / f"mode_{mode_number:0{width}d}.png",
                dpi,
            )
    return saved_count, SubspaceBasis(result, basis_vectors, x, y)


def _grids_match(reference: SubspaceBasis, candidate: SubspaceBasis) -> bool:
    return (
        reference.x.shape == candidate.x.shape
        and reference.y.shape == candidate.y.shape
        and np.allclose(reference.x, candidate.x, rtol=1e-10, atol=1e-12)
        and np.allclose(reference.y, candidate.y, rtol=1e-10, atol=1e-12)
    )


def pairwise_subspace_similarity(
    bases: list[SubspaceBasis],
) -> np.ndarray:
    """Return normalized projection similarity for every pair of bases.

    For orthonormal bases with ``k`` rows, the score is
    ``||U_i.T @ U_j||_F^2 / k``. It is invariant to mode signs and rotations
    within each retained subspace and ranges from zero to one.
    """
    if not bases:
        raise ValueError("At least one POD basis is required.")

    mode_counts = {basis.vectors.shape[0] for basis in bases}
    vector_counts = {basis.vectors.shape[1] for basis in bases}
    if len(mode_counts) != 1 or len(vector_counts) != 1:
        raise ValueError("All POD bases must have the same dimensions.")
    mode_count = mode_counts.pop()
    if mode_count < 1:
        raise ValueError("POD bases must contain at least one mode.")

    matrix = np.empty((len(bases), len(bases)), dtype=np.float64)
    identity = np.eye(mode_count)
    for index, basis in enumerate(bases):
        gram = basis.vectors @ basis.vectors.T
        orthogonality_error = float(np.max(np.abs(gram - identity)))
        if orthogonality_error > 5e-5:
            raise ValueError(
                f"The leading modes for {basis.result.repetition} are not "
                f"orthonormal (maximum Gram-matrix error "
                f"{orthogonality_error:.3e})."
            )
        matrix[index, index] = 1.0

        for other_index in range(index + 1, len(bases)):
            overlap = basis.vectors @ bases[other_index].vectors.T
            score = float(np.sum(overlap**2) / mode_count)
            if score < -1e-10 or score > 1 + 1e-5:
                raise ValueError(
                    f"Invalid subspace similarity {score:.8g} between "
                    f"{basis.result.repetition} and "
                    f"{bases[other_index].result.repetition}."
                )
            score = float(np.clip(score, 0.0, 1.0))
            matrix[index, other_index] = score
            matrix[other_index, index] = score

    return matrix


def save_similarity_csv(
    bases: list[SubspaceBasis],
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Save a labeled, full symmetric similarity matrix as CSV."""
    labels = [basis.result.repetition for basis in bases]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["repetition", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *(f"{value:.10f}" for value in row)])


def _line_colors(count: int) -> np.ndarray:
    color_map = plt.get_cmap("turbo")
    if count == 1:
        return np.asarray([color_map(0.5)])
    return color_map(np.linspace(0.02, 0.98, count))


def plot_combined_energy(
    energy_by_result: list[tuple[PodResult, np.ndarray]],
    output_path: Path,
    condition_name: str,
    logarithmic_x: bool,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    colors = _line_colors(len(energy_by_result))
    maximum_rank = 1
    for color, (result, cumulative) in zip(colors, energy_by_result, strict=True):
        ranks = np.arange(1, len(cumulative) + 1)
        maximum_rank = max(maximum_rank, len(cumulative))
        ax.plot(
            ranks,
            100 * cumulative,
            color=color,
            linewidth=1.4,
            label=f"rep {result.repetition_number} (r={result.rank})",
        )

    if logarithmic_x:
        ax.set_xscale("log")
    ax.set_xlim(1, maximum_rank)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Number of retained POD modes, $r$")
    ax.set_ylabel("Cumulative captured energy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_title(f"POD cumulative energy across repetitions — {condition_name}")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    if logarithmic_x:
        ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _common_units(series_by_result: list[tuple[PodResult, MeanSeries]], field: str) -> str:
    units = {getattr(series, field) for _, series in series_by_result}
    return units.pop() if len(units) == 1 else ""


def plot_combined_spatial_mean(
    series_by_result: list[tuple[PodResult, MeanSeries]],
    output_path: Path,
    condition_name: str,
    dpi: int,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    colors = _line_colors(len(series_by_result))
    for color, (result, series) in zip(colors, series_by_result, strict=True):
        ax.plot(
            series.time,
            series.spatial_mean,
            color=color,
            linewidth=0.75,
            alpha=0.85,
            label=f"rep {result.repetition_number}",
        )

    time_units = _common_units(series_by_result, "time_units")
    mean_units = _common_units(series_by_result, "mean_units")
    ax.set_xlabel(_axis_label("Time", time_units))
    ax.set_ylabel(_axis_label("Spatial mean", mean_units))
    ax.set_title(title or f"Spatial mean over time across repetitions — {condition_name}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> int:
    validate_args(args)
    condition_directory = resolve_condition_directory(args.condition, args.root)
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else condition_directory / "pod_analysis"
    )

    results, issues = discover_results(condition_directory, args.rank)
    if args.max_repetitions is not None:
        results = results[: args.max_repetitions]

    print(f"Condition: {condition_directory}")
    print(f"Output:    {output_directory}")
    print(f"Selection: rank {args.rank}; {len(results)} repetition(s)")
    if args.videos:
        video_rank = args.video_rank if args.video_rank is not None else "all stored"
        print(
            f"Videos: {args.video_seconds:g} s at {args.video_fps} fps, "
            f"stride {args.video_stride}, rank {video_rank}"
        )
    for issue in issues:
        print(f"WARNING: {issue}", file=sys.stderr)
    for result in results:
        print(f"  rep {result.repetition_number}: {result.path.name} (rank {result.rank})")

    if not results:
        print("ERROR: no usable POD results were found.", file=sys.stderr)
        return 1
    if args.dry_run:
        print("Dry run complete; no output was created.")
        return 0

    output_directory.mkdir(parents=True, exist_ok=True)
    energy_by_result: list[tuple[PodResult, np.ndarray]] = []
    mean_by_result: list[tuple[PodResult, MeanSeries]] = []
    comparison_bases: list[SubspaceBasis] = []
    failures: list[str] = []

    for index, result in enumerate(results, start=1):
        repetition_output = output_directory / result.repetition
        repetition_output.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(results)}] Analyzing {result.repetition} from "
            f"{result.path.name}",
            flush=True,
        )

        try:
            cumulative, _ = load_cumulative_energy(result.path)
            plot_cumulative_energy(
                cumulative,
                repetition_output / "pod_energy.png",
                f"POD cumulative energy — {result.repetition}",
                logarithmic_x=not args.linear_energy_x,
                dpi=args.dpi,
            )
            energy_by_result.append((result, cumulative))
            print("  Saved pod_energy.png", flush=True)
        except Exception as exc:
            message = f"{result.repetition} energy: {exc}"
            failures.append(message)
            print(f"  WARNING: {message}", file=sys.stderr, flush=True)

        try:
            mean_series = load_spatial_mean(result.path)
            plot_spatial_mean(
                mean_series,
                repetition_output / "spatial_mean_over_time.png",
                f"Spatial mean over time — {result.repetition}",
                args.dpi,
            )
            mean_by_result.append((result, mean_series))
            print("  Saved spatial_mean_over_time.png", flush=True)
        except Exception as exc:
            message = f"{result.repetition} spatial mean: {exc}"
            failures.append(message)
            print(f"  WARNING: {message}", file=sys.stderr, flush=True)

        try:
            saved_modes, basis = plot_modes(
                result, repetition_output, args.modes, args.dpi
            )
            print(f"  Saved {saved_modes} mode image(s)", flush=True)
            if comparison_bases and not _grids_match(comparison_bases[0], basis):
                message = (
                    f"{result.repetition} subspace similarity: spatial grid does "
                    "not match the other repetitions"
                )
                failures.append(message)
                print(f"  WARNING: {message}", file=sys.stderr, flush=True)
            else:
                comparison_bases.append(basis)
        except Exception as exc:
            message = f"{result.repetition} modes: {exc}"
            failures.append(message)
            print(f"  WARNING: {message}", file=sys.stderr, flush=True)

        if args.videos:
            try:
                make_reconstruction_video(
                    result.path,
                    output_path=repetition_output / "reconstructed_dynamics.mp4",
                    seconds=args.video_seconds,
                    fps=args.video_fps,
                    stride=args.video_stride,
                    rank=args.video_rank,
                    overwrite=True,
                )
            except Exception as exc:
                message = f"{result.repetition} video: {exc}"
                failures.append(message)
                print(f"  WARNING: {message}", file=sys.stderr, flush=True)

    if energy_by_result:
        combined_energy_path = output_directory / "all_repetitions_pod_energy.png"
        plot_combined_energy(
            energy_by_result,
            combined_energy_path,
            condition_directory.name,
            logarithmic_x=not args.linear_energy_x,
            dpi=args.dpi,
        )
        print(f"Saved combined energy plot: {combined_energy_path}")
    else:
        failures.append("combined energy: no valid energy series were available")

    if mean_by_result:
        combined_mean_path = (
            output_directory / "all_repetitions_spatial_mean_over_time.png"
        )
        plot_combined_spatial_mean(
            mean_by_result,
            combined_mean_path,
            condition_directory.name,
            args.dpi,
        )
        print(f"Saved combined spatial-mean plot: {combined_mean_path}")
    else:
        failures.append("combined spatial mean: no valid mean series were available")

    if comparison_bases:
        try:
            mode_count = comparison_bases[0].vectors.shape[0]
            similarity = pairwise_subspace_similarity(comparison_bases)
            similarity_path = (
                output_directory
                / f"pairwise_subspace_similarity_k{mode_count}.csv"
            )
            save_similarity_csv(comparison_bases, similarity, similarity_path)
            print(f"Saved subspace-similarity matrix: {similarity_path}")
        except Exception as exc:
            message = f"subspace similarity: {exc}"
            failures.append(message)
            print(f"WARNING: {message}", file=sys.stderr, flush=True)
    else:
        failures.append("subspace similarity: no valid POD bases were available")

    print(
        f"Finished {len(results)} repetition(s) with {len(failures)} analysis "
        f"failure(s). Results: {output_directory.resolve()}"
    )
    if failures:
        print("Warnings:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    return 0


def main() -> None:
    try:
        status = analyze(parse_args())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    main()
