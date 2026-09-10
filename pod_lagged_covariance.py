"""Compare lagged reduced-coordinate statistics across repetitions at one power.

The lowest-numbered repetition (normally rep1) supplies the common POD basis.
Every other coefficient trajectory is mapped into that basis through the mode
overlap matrix. The instantaneous spatial mean is coordinate zero.

Example:
    python pod_lagged_covariance.py 0p20 --rank 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import time

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pod_center_psd import infer_sampling_frequency


DEFAULT_REDUCED_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
POWER_PATTERN = re.compile(r"0p\d{1,2}")
REPETITION_PATTERN = re.compile(r"Ca_ac.*_rep(?P<number>\d+)$")
DEFAULT_LAGS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000)


@dataclass(frozen=True)
class Repetition:
    name: str
    number: int
    pod_path: Path


@dataclass(frozen=True)
class Result:
    repetition: Repetition
    covariance: np.ndarray
    correlation: np.ndarray
    reference_scaled: np.ndarray
    basis_overlap: np.ndarray
    principal_cosines: np.ndarray
    projection_energy_ratio: float
    sampling_frequency_hz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("power", help="Power folder, for example 0p20.")
    parser.add_argument(
        "--rank", type=int, required=True, help="Number of leading POD modes."
    )
    parser.add_argument(
        "--reduced-root",
        type=Path,
        default=DEFAULT_REDUCED_ROOT,
        help=f"Reduced-data root (default: {DEFAULT_REDUCED_ROOT}).",
    )
    parser.add_argument(
        "--pod-rank",
        type=int,
        default=1_000,
        help="Rank in each stored POD filename (default: 1000).",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAGS),
        metavar="SAMPLES",
        help="Non-negative sample lags (default: 0 1 2 5 10 20 50 100 200 500 1000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8_192,
        help="Coefficient rows transformed per batch (default: 8192).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output folder (default: <reduced-root>/center_psd_comparisons/"
            "lagged_covariance/<power>_rank_<rank>)."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Plot resolution (default: 200)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing results."
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if POWER_PATTERN.fullmatch(args.power) is None:
        raise ValueError("power must look like 0p0, 0p04, or 0p20.")
    if args.rank < 1:
        raise ValueError("--rank must be positive.")
    if args.pod_rank < args.rank:
        raise ValueError("--pod-rank must be at least --rank.")
    if args.batch_size < 1 or args.dpi < 1:
        raise ValueError("--batch-size and --dpi must be positive.")
    if not args.lags or any(lag < 0 for lag in args.lags):
        raise ValueError("--lags must contain non-negative integers.")
    args.lags = sorted(set(args.lags))


def discover_repetitions(power_dir: Path, pod_rank: int) -> list[Repetition]:
    repetitions: list[Repetition] = []
    for directory in power_dir.iterdir():
        match = REPETITION_PATTERN.fullmatch(directory.name)
        if not directory.is_dir() or match is None:
            continue
        pod_path = directory / f"pod_2d_r{pod_rank}.h5"
        if not pod_path.is_file():
            print(f"Skipping {directory.name}: missing {pod_path.name}", flush=True)
            continue
        repetitions.append(
            Repetition(directory.name, int(match.group("number")), pod_path.resolve())
        )
    return sorted(repetitions, key=lambda repetition: repetition.number)


def load_modes(pod_path: Path, rank: int) -> tuple[np.ndarray, tuple[int, int]]:
    with h5py.File(pod_path, "r") as handle:
        if "pod/modes" not in handle:
            raise KeyError(f"Missing pod/modes in {pod_path}.")
        modes = handle["pod/modes"]
        if modes.ndim != 3 or len(modes) < rank:
            raise ValueError(
                f"Mode shape {modes.shape} in {pod_path} cannot supply rank {rank}."
            )
        shape = tuple(modes.shape[1:])
        flattened = np.asarray(modes[:rank], dtype=np.float64).reshape(rank, -1)
    if not np.isfinite(flattened).all():
        raise ValueError(f"Modes contain nonfinite values in {pod_path}.")
    return flattened, shape


def load_coordinates(
    repetition: Repetition,
    reference_modes: np.ndarray,
    reference_shape: tuple[int, int],
    rank: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    modes, shape = load_modes(repetition.pod_path, rank)
    if shape != reference_shape:
        raise ValueError(f"Spatial shape {shape} differs from {reference_shape}.")
    overlap = reference_modes @ modes.T
    principal_cosines = np.linalg.svd(overlap, compute_uv=False)

    with h5py.File(repetition.pod_path, "r") as handle:
        required = (
            "reduced/coefficients",
            "preprocessing/frame_spatial_mean",
            "grid/time",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"Missing {', '.join(missing)} in {repetition.pod_path}.")
        coefficients = handle["reduced/coefficients"]
        spatial_mean_data = handle["preprocessing/frame_spatial_mean"]
        if coefficients.ndim != 2 or coefficients.shape[1] < rank:
            raise ValueError(
                f"Coefficient shape {coefficients.shape} cannot supply rank {rank}."
            )
        sample_count = len(coefficients)
        if len(spatial_mean_data) != sample_count:
            raise ValueError("Coefficient and spatial-mean lengths differ.")
        time_values = np.asarray(handle["grid/time"], dtype=np.float64)
        if len(time_values) != sample_count:
            raise ValueError("Coefficient and time-grid lengths differ.")
        sampling_frequency_hz, _ = infer_sampling_frequency(time_values)

        source = np.empty((sample_count, rank), dtype=np.float64)
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            source[start:stop] = coefficients[start:stop, :rank]
        spatial_mean = np.asarray(spatial_mean_data, dtype=np.float64)

    if not np.isfinite(source).all() or not np.isfinite(spatial_mean).all():
        raise ValueError("Coefficients or spatial mean contain nonfinite values.")
    source -= np.mean(source, axis=0, keepdims=True)
    aligned = source @ overlap.T
    source_energy = float(np.sum(source * source))
    projection_energy_ratio = (
        float(np.sum(aligned * aligned)) / source_energy
        if source_energy > 0
        else np.nan
    )
    spatial_mean -= np.mean(spatial_mean)
    coordinates = np.column_stack((spatial_mean, aligned))
    return (
        coordinates,
        overlap,
        principal_cosines,
        projection_energy_ratio,
        sampling_frequency_hz,
    )


def lagged_covariances(coordinates: np.ndarray, lags: np.ndarray) -> np.ndarray:
    sample_count, coordinate_count = coordinates.shape
    output = np.empty(
        (len(lags), coordinate_count, coordinate_count), dtype=np.float64
    )
    for index, lag in enumerate(lags):
        pair_count = sample_count - int(lag)
        if pair_count < 2:
            raise ValueError(f"Lag {lag} leaves only {pair_count} sample pairs.")
        output[index] = (
            coordinates[:pair_count].T @ coordinates[int(lag) :]
        ) / pair_count
    return output


def normalize(
    covariance: np.ndarray, standard_deviation: np.ndarray
) -> np.ndarray:
    denominator = np.outer(standard_deviation, standard_deviation)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("At least one coordinate has zero or invalid variance.")
    return covariance / denominator[None, :, :]


def mean_variance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    variance = (
        np.var(values, axis=0, ddof=1)
        if len(values) > 1
        else np.full(values.shape[1:], np.nan)
    )
    return mean, variance


def relative_distance(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(values - mean[None, ...], axis=(-2, -1))
    denominator = np.linalg.norm(mean, axis=(-2, -1))
    return np.divide(
        numerator,
        denominator[None, :],
        out=np.full_like(numerator, np.nan),
        where=denominator[None, :] > 0,
    )


def compressed(group: h5py.Group, name: str, values: np.ndarray) -> None:
    group.create_dataset(name, data=values, compression="gzip", shuffle=True)


def save_hdf5(
    path: Path,
    power: str,
    rank: int,
    pod_rank: int,
    lags: np.ndarray,
    results: list[Result],
    failures: list[str],
) -> dict[str, np.ndarray]:
    covariance = np.stack([result.covariance for result in results])
    correlation = np.stack([result.correlation for result in results])
    reference_scaled = np.stack([result.reference_scaled for result in results])
    covariance_mean, covariance_variance = mean_variance(covariance)
    correlation_mean, correlation_variance = mean_variance(correlation)
    reference_mean, reference_variance = mean_variance(reference_scaled)
    covariance_distance = relative_distance(covariance, covariance_mean)
    correlation_distance = relative_distance(correlation, correlation_mean)
    reference_distance = relative_distance(reference_scaled, reference_mean)

    sampling_frequency = results[0].sampling_frequency_hz
    lag_seconds = lags / sampling_frequency
    labels = ["spatial_mean"] + [f"mode_{index}" for index in range(1, rank + 1)]
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary_path, "w") as handle:
        handle.attrs["input_power"] = power
        handle.attrs["rank"] = rank
        handle.attrs["stored_pod_rank"] = pod_rank
        handle.attrs["reference_repetition"] = results[0].repetition.name
        handle.attrs["coordinate_zero"] = "instantaneous spatial mean"
        handle.attrs["centering"] = "each coordinate centered by its time mean"
        handle.attrs["covariance_denominator"] = "number of paired samples"
        handle.attrs["basis_mapping"] = (
            "aligned = coefficients @ (U_reference.T @ U_repetition).T"
        )
        handle.attrs["variance_definition"] = "sample variance across repetitions"
        handle.create_dataset("lag_samples", data=lags)
        handle.create_dataset("lag_seconds", data=lag_seconds)
        handle.create_dataset(
            "coordinate_labels",
            data=np.asarray(labels, dtype=object),
            dtype=string_type,
        )

        repetitions = handle.create_group("repetitions")
        repetitions.create_dataset(
            "name",
            data=np.asarray(
                [result.repetition.name for result in results], dtype=object
            ),
            dtype=string_type,
        )
        repetitions.create_dataset(
            "sampling_frequency_hz",
            data=np.asarray(
                [result.sampling_frequency_hz for result in results]
            ),
        )
        repetitions.create_dataset(
            "projection_energy_ratio",
            data=np.asarray(
                [result.projection_energy_ratio for result in results]
            ),
        )
        compressed(
            repetitions,
            "basis_overlap",
            np.stack([result.basis_overlap for result in results]),
        )
        compressed(
            repetitions,
            "principal_cosines",
            np.stack([result.principal_cosines for result in results]),
        )
        compressed(repetitions, "lagged_covariance", covariance)
        compressed(repetitions, "lagged_correlation", correlation)
        compressed(
            repetitions, "reference_scaled_lagged_covariance", reference_scaled
        )

        summary = handle.create_group("summary_across_repetitions")
        for name, values in (
            ("covariance_mean", covariance_mean),
            ("covariance_variance", covariance_variance),
            ("correlation_mean", correlation_mean),
            ("correlation_variance", correlation_variance),
            ("reference_scaled_mean", reference_mean),
            ("reference_scaled_variance", reference_variance),
            ("covariance_relative_frobenius_distance", covariance_distance),
            ("correlation_relative_frobenius_distance", correlation_distance),
            ("reference_scaled_relative_frobenius_distance", reference_distance),
        ):
            compressed(summary, name, values)
        handle.create_dataset(
            "failures", data=np.asarray(failures, dtype=object), dtype=string_type
        )
    os.replace(temporary_path, path)
    return {
        "lag_seconds": lag_seconds,
        "correlation_mean": correlation_mean,
        "correlation_variance": correlation_variance,
        "covariance_distance": covariance_distance,
        "correlation_distance": correlation_distance,
    }


def save_distance_plot(
    path: Path,
    lags: np.ndarray,
    names: list[str],
    covariance_distance: np.ndarray,
    correlation_distance: np.ndarray,
    power: str,
    rank: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=True)
    for index, name in enumerate(names):
        axes[0].plot(lags, covariance_distance[index], marker="o", ms=3, label=name)
        axes[1].plot(
            lags, correlation_distance[index], marker="o", ms=3, label=name
        )
    axes[0].set_title("Raw covariance")
    axes[1].set_title("Normalized correlation")
    for axis in axes:
        axis.set_xlabel("Lag [samples]")
        axis.set_ylabel("Relative Frobenius distance from mean")
        axis.grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(
        title="Repetition",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False,
        fontsize=7,
    )
    figure.suptitle(f"Lagged-statistics consistency — {power}, rank {rank}")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_heatmaps(
    path: Path,
    correlation_mean: np.ndarray,
    lags: np.ndarray,
    power: str,
    rank: int,
    dpi: int,
) -> None:
    columns = min(4, len(lags))
    rows = int(np.ceil(len(lags) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 3.0 * rows), squeeze=False
    )
    limit = max(1.0, float(np.nanmax(np.abs(correlation_mean))))
    plotted_image = None
    for index, axis in enumerate(axes.flat):
        if index >= len(lags):
            axis.axis("off")
            continue
        plotted_image = axis.imshow(
            correlation_mean[index],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"lag = {lags[index]} samples")
        axis.set_xlabel("lagged coordinate")
        axis.set_ylabel("initial coordinate")
    assert plotted_image is not None
    figure.colorbar(
        plotted_image,
        ax=axes.ravel().tolist(),
        shrink=0.72,
        label="correlation",
    )
    figure.suptitle(
        f"Mean lagged correlation across repetitions — {power}, rank {rank}"
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_standard_deviation_heatmaps(
    path: Path,
    correlation_variance: np.ndarray,
    lags: np.ndarray,
    power: str,
    rank: int,
    dpi: int,
) -> None:
    standard_deviation = np.sqrt(np.maximum(0.0, correlation_variance))
    columns = min(4, len(lags))
    rows = int(np.ceil(len(lags) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 3.0 * rows), squeeze=False
    )
    limit = float(np.nanmax(standard_deviation))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    plotted_image = None
    for index, axis in enumerate(axes.flat):
        if index >= len(lags):
            axis.axis("off")
            continue
        plotted_image = axis.imshow(
            standard_deviation[index],
            cmap="magma",
            vmin=0.0,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"lag = {lags[index]} samples")
        axis.set_xlabel("lagged coordinate")
        axis.set_ylabel("initial coordinate")
    assert plotted_image is not None
    figure.colorbar(
        plotted_image,
        ax=axes.ravel().tolist(),
        shrink=0.72,
        label="correlation standard deviation",
    )
    figure.suptitle(
        f"Standard deviation of lagged correlation — {power}, rank {rank}"
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    reduced_root = args.reduced_root.expanduser().resolve()
    power_dir = reduced_root / args.power
    if not power_dir.is_dir():
        raise FileNotFoundError(f"Power directory does not exist: {power_dir}")
    repetitions = discover_repetitions(power_dir, args.pod_rank)
    if not repetitions:
        raise RuntimeError(f"No valid repetitions found in {power_dir}.")
    reference = repetitions[0]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else reduced_root
        / "center_psd_comparisons"
        / "lagged_covariance"
        / f"{args.power}_rank_{args.rank}"
    )
    output_path = output_dir / "lagged_covariance.h5"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Results exist: {output_path}. Use --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference basis: {reference.name}", flush=True)
    print(f"Repetitions:     {len(repetitions)}", flush=True)
    print(f"Rank:            {args.rank}", flush=True)
    print(f"Lags [samples]:  {' '.join(map(str, args.lags))}", flush=True)
    reference_modes, reference_shape = load_modes(reference.pod_path, args.rank)
    lags = np.asarray(args.lags, dtype=np.int64)
    started = time.perf_counter()
    results: list[Result] = []
    failures: list[str] = []
    reference_standard_deviation: np.ndarray | None = None

    for index, repetition in enumerate(repetitions, start=1):
        job_started = time.perf_counter()
        try:
            (
                coordinates,
                overlap,
                principal_cosines,
                projection_energy_ratio,
                sampling_frequency,
            ) = load_coordinates(
                repetition,
                reference_modes,
                reference_shape,
                args.rank,
                args.batch_size,
            )
            covariance = lagged_covariances(coordinates, lags)
            own_standard_deviation = np.sqrt(
                np.mean(coordinates * coordinates, axis=0)
            )
            if reference_standard_deviation is None:
                reference_standard_deviation = own_standard_deviation
            correlation = normalize(covariance, own_standard_deviation)
            reference_scaled = normalize(
                covariance, reference_standard_deviation
            )
            results.append(
                Result(
                    repetition,
                    covariance,
                    correlation,
                    reference_scaled,
                    overlap,
                    principal_cosines,
                    projection_energy_ratio,
                    sampling_frequency,
                )
            )
        except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as error:
            if repetition is reference:
                raise RuntimeError(
                    f"Reference repetition {reference.name} failed: {error}"
                ) from error
            failures.append(f"{repetition.name}: {error}")
            print(
                f"FAILED [{index}/{len(repetitions)}] {repetition.name}: {error}",
                flush=True,
            )
            continue
        print(
            f"DONE [{index}/{len(repetitions)}] {repetition.name}; "
            f"projection energy={projection_energy_ratio:.5f} "
            f"({time.perf_counter() - job_started:.2f} s)",
            flush=True,
        )

    arrays = save_hdf5(
        output_path,
        args.power,
        args.rank,
        args.pod_rank,
        lags,
        results,
        failures,
    )
    names = [result.repetition.name for result in results]
    distance_plot = output_dir / "distance_from_repetition_mean.png"
    save_distance_plot(
        distance_plot,
        lags,
        names,
        arrays["covariance_distance"],
        arrays["correlation_distance"],
        args.power,
        args.rank,
        args.dpi,
    )
    heatmap_plot = output_dir / "mean_lagged_correlation.png"
    save_heatmaps(
        heatmap_plot,
        arrays["correlation_mean"],
        lags,
        args.power,
        args.rank,
        args.dpi,
    )
    standard_deviation_plot = output_dir / "std_lagged_correlation.png"
    save_standard_deviation_heatmaps(
        standard_deviation_plot,
        arrays["correlation_variance"],
        lags,
        args.power,
        args.rank,
        args.dpi,
    )
    print()
    print(
        f"Completed {len(results)}/{len(repetitions)} repetitions in "
        f"{time.perf_counter() - started:.1f} s."
    )
    print(f"Saved {output_path}")
    print(f"Saved {distance_plot}")
    print(f"Saved {heatmap_plot}")
    print(f"Saved {standard_deviation_plot}")
    if failures:
        print(f"Skipped {len(failures)} failed repetitions; details are in the HDF5.")
    return 1 if failures else 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
