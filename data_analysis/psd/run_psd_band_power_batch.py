"""Compute and aggregate PSD band-power comparisons for all experiments.

Each POD center reconstruction uses the requested number of leading modes and
has its saved instantaneous spatial mean restored. Results for every
repetition, plus per-power means and sample variances, are stored in one HDF5
file. A CSV summary and mean +/- standard-deviation plots are also produced.

Example
-------
    python run_psd_band_power_batch.py /path/to/reduced_data --rank 10
"""

from __future__ import annotations

import argparse
import csv
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

from data_analysis.psd.compare_psd_band_power import (
    integrate_band,
    load_raw_signal,
    logarithmic_edges,
)
from data_analysis.psd.pod_center_psd import (
    compute_welch,
    infer_sampling_frequency,
    reconstruct_point_signals,
)


CONDITION_PATTERN = re.compile(r"0p\d{1,2}")
REALIZATION_PATTERN = re.compile(r"Ca_ac.*")


@dataclass(frozen=True)
class Job:
    condition: str
    realization: str
    pod_path: Path
    raw_cache_path: Path


@dataclass(frozen=True)
class Result:
    job: Job
    experimental_power: np.ndarray
    reconstructed_power: np.ndarray
    ratio: np.ndarray


@dataclass(frozen=True)
class Summary:
    conditions: list[str]
    counts: np.ndarray
    experimental_mean: np.ndarray
    experimental_variance: np.ndarray
    reconstructed_mean: np.ndarray
    reconstructed_variance: np.ndarray
    ratio_mean: np.ndarray
    ratio_variance: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reduced_root", type=Path, help="Reduced-data tree root.")
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help=(
            "Number of leading POD modes used; 0 compares the saved "
            "instantaneous spatial mean by itself."
        ),
    )
    parser.add_argument(
        "--pod-rank",
        type=int,
        default=1_000,
        help="Rank in each stored POD filename (default: 1000).",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        help=(
            "Directory containing raw_psd_cache (default: "
            "<reduced-root>/center_psd_comparisons)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory (default: "
            "<comparison-dir>/band_power_rank_<rank>)."
        ),
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=8_192,
        help="Samples per Welch segment (default: 8192).",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=0.5,
        help="Fractional Welch overlap (default: 0.5).",
    )
    parser.add_argument(
        "--bins-per-decade",
        type=int,
        choices=(1, 2),
        default=2,
        help="Use whole- or half-decade frequency bands (default: 2).",
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=50_000.0,
        help=(
            "Upper edge of the final band in Hz (default: 50000); using a "
            "fixed edge keeps all repetitions directly comparable."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8_192,
        help="Coefficient rows reconstructed per batch (default: 8192).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Process only the first N repetitions; useful for testing.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Plot resolution (default: 200)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing rank-specific result directory.",
    )
    return parser.parse_args()


def natural_key(text: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.rank < 0:
        raise ValueError("--rank must be non-negative.")
    if args.pod_rank < 1:
        raise ValueError("--pod-rank must be positive.")
    if args.nperseg < 2:
        raise ValueError("--nperseg must be at least 2.")
    if not 0 <= args.overlap_fraction < 1:
        raise ValueError("--overlap-fraction must lie in [0, 1).")
    if args.batch_size < 1 or args.dpi < 1:
        raise ValueError("--batch-size and --dpi must be positive.")
    if args.maximum_frequency <= 10.0:
        raise ValueError("--maximum-frequency must be greater than 10 Hz.")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive.")


def discover_jobs(
    reduced_root: Path,
    comparison_dir: Path,
    pod_rank: int,
) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    issues: list[str] = []
    condition_dirs = sorted(
        (
            path
            for path in reduced_root.iterdir()
            if path.is_dir() and CONDITION_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: natural_key(path.name),
    )
    for condition_dir in condition_dirs:
        realization_dirs = sorted(
            (
                path
                for path in condition_dir.iterdir()
                if path.is_dir() and REALIZATION_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: natural_key(path.name),
        )
        for realization_dir in realization_dirs:
            pod_path = realization_dir / f"pod_2d_r{pod_rank}.h5"
            raw_cache_path = (
                comparison_dir
                / "raw_psd_cache"
                / f"{condition_dir.name}__{realization_dir.name}__raw_center_psd.npz"
            )
            missing = []
            if not pod_path.is_file():
                missing.append(pod_path.name)
            if not raw_cache_path.is_file():
                missing.append(raw_cache_path.name)
            if missing:
                issues.append(
                    f"{condition_dir.name}/{realization_dir.name}: missing "
                    + ", ".join(missing)
                )
                continue
            jobs.append(
                Job(
                    condition_dir.name,
                    realization_dir.name,
                    pod_path.resolve(),
                    raw_cache_path.resolve(),
                )
            )
    return jobs, issues


def compute_result(job: Job, args: argparse.Namespace) -> tuple[Result, np.ndarray]:
    raw_time, raw_signal = load_raw_signal(job.raw_cache_path)
    (
        pod_time,
        _preprocessed_signal,
        _spatial_mean,
        reconstructed_signal,
        _selected,
        _used_rank,
        _signal_units,
    ) = reconstruct_point_signals(
        job.pod_path,
        args.rank,
        None,
        None,
        args.batch_size,
        show_progress=False,
    )
    raw_fs, _ = infer_sampling_frequency(raw_time)
    pod_fs, _ = infer_sampling_frequency(pod_time)
    raw_psd, _, _ = compute_welch(
        raw_signal, raw_fs, args.nperseg, args.overlap_fraction
    )
    reconstructed_psd, _, _ = compute_welch(
        reconstructed_signal, pod_fs, args.nperseg, args.overlap_fraction
    )
    available_frequency = min(
        float(raw_psd.frequency[-1]), float(reconstructed_psd.frequency[-1])
    )
    if available_frequency < args.maximum_frequency:
        raise ValueError(
            f"PSD ends at {available_frequency:g} Hz, below requested maximum "
            f"{args.maximum_frequency:g} Hz."
        )
    edges = logarithmic_edges(
        10.0, args.maximum_frequency, args.bins_per_decade
    )
    experimental_power = np.asarray(
        [
            integrate_band(raw_psd.frequency, raw_psd.density, lower, upper)
            for lower, upper in zip(edges[:-1], edges[1:], strict=True)
        ],
        dtype=np.float64,
    )
    reconstructed_power = np.asarray(
        [
            integrate_band(
                reconstructed_psd.frequency,
                reconstructed_psd.density,
                lower,
                upper,
            )
            for lower, upper in zip(edges[:-1], edges[1:], strict=True)
        ],
        dtype=np.float64,
    )
    if np.any(experimental_power <= 0):
        raise ValueError("Experimental band power must be positive.")
    ratio = reconstructed_power / experimental_power
    return Result(job, experimental_power, reconstructed_power, ratio), edges


def mean_and_sample_variance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    variance = (
        np.var(values, axis=0, ddof=1)
        if len(values) > 1
        else np.full(values.shape[1], np.nan)
    )
    return mean, variance


def aggregate(results: list[Result]) -> Summary:
    conditions = sorted({result.job.condition for result in results}, key=natural_key)
    counts = []
    experimental_means = []
    experimental_variances = []
    reconstructed_means = []
    reconstructed_variances = []
    ratio_means = []
    ratio_variances = []
    for condition in conditions:
        selected = [result for result in results if result.job.condition == condition]
        experimental = np.stack(
            [result.experimental_power for result in selected], axis=0
        )
        reconstructed = np.stack(
            [result.reconstructed_power for result in selected], axis=0
        )
        ratios = np.stack([result.ratio for result in selected], axis=0)
        experimental_mean, experimental_variance = mean_and_sample_variance(
            experimental
        )
        reconstructed_mean, reconstructed_variance = mean_and_sample_variance(
            reconstructed
        )
        ratio_mean, ratio_variance = mean_and_sample_variance(ratios)
        counts.append(len(selected))
        experimental_means.append(experimental_mean)
        experimental_variances.append(experimental_variance)
        reconstructed_means.append(reconstructed_mean)
        reconstructed_variances.append(reconstructed_variance)
        ratio_means.append(ratio_mean)
        ratio_variances.append(ratio_variance)
    return Summary(
        conditions,
        np.asarray(counts, dtype=np.int64),
        np.stack(experimental_means),
        np.stack(experimental_variances),
        np.stack(reconstructed_means),
        np.stack(reconstructed_variances),
        np.stack(ratio_means),
        np.stack(ratio_variances),
    )


def compressed_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    group.create_dataset(name, data=values, compression="gzip", shuffle=True)


def save_hdf5(
    output_path: Path,
    results: list[Result],
    summary: Summary,
    edges: np.ndarray,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary_path, "w") as handle:
        handle.attrs["reconstruction_rank"] = args.rank
        handle.attrs["stored_pod_rank"] = args.pod_rank
        handle.attrs["nperseg"] = args.nperseg
        handle.attrs["overlap_fraction"] = args.overlap_fraction
        handle.attrs["bins_per_decade"] = args.bins_per_decade
        handle.attrs["maximum_frequency_hz"] = args.maximum_frequency
        handle.attrs["ratio_definition"] = (
            "reconstructed_band_power / experimental_band_power"
        )
        handle.attrs["reconstruction_includes_instantaneous_spatial_mean"] = True
        bands = handle.create_group("bands")
        bands.create_dataset("lower_frequency_hz", data=edges[:-1])
        bands.create_dataset("upper_frequency_hz", data=edges[1:])

        repetitions = handle.create_group("repetitions")
        repetitions.create_dataset(
            "condition",
            data=np.asarray(
                [result.job.condition for result in results], dtype=object
            ),
            dtype=string_type,
        )
        repetitions.create_dataset(
            "realization",
            data=np.asarray(
                [result.job.realization for result in results], dtype=object
            ),
            dtype=string_type,
        )
        compressed_dataset(
            repetitions,
            "experimental_band_power",
            np.stack([result.experimental_power for result in results]),
        )
        compressed_dataset(
            repetitions,
            "reconstructed_band_power",
            np.stack([result.reconstructed_power for result in results]),
        )
        compressed_dataset(
            repetitions,
            "reconstructed_over_experimental",
            np.stack([result.ratio for result in results]),
        )

        aggregate_group = handle.create_group("summary_by_power")
        aggregate_group.attrs["variance_definition"] = "sample variance (ddof=1)"
        aggregate_group.create_dataset(
            "condition",
            data=np.asarray(summary.conditions, dtype=object),
            dtype=string_type,
        )
        aggregate_group.create_dataset("repetition_count", data=summary.counts)
        for name, values in (
            ("experimental_mean", summary.experimental_mean),
            ("experimental_variance", summary.experimental_variance),
            ("reconstructed_mean", summary.reconstructed_mean),
            ("reconstructed_variance", summary.reconstructed_variance),
            ("ratio_mean", summary.ratio_mean),
            ("ratio_variance", summary.ratio_variance),
        ):
            compressed_dataset(aggregate_group, name, values)
        handle.create_dataset(
            "failures", data=np.asarray(failures, dtype=object), dtype=string_type
        )
    os.replace(temporary_path, output_path)


def save_summary_csv(path: Path, summary: Summary, edges: np.ndarray) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "input_power",
                "repetition_count",
                "lower_frequency_hz",
                "upper_frequency_hz",
                "experimental_mean",
                "experimental_sample_variance",
                "reconstructed_mean",
                "reconstructed_sample_variance",
                "ratio_mean",
                "ratio_sample_variance",
                "ratio_standard_deviation",
            )
        )
        for power_index, condition in enumerate(summary.conditions):
            for band_index, (lower, upper) in enumerate(
                zip(edges[:-1], edges[1:], strict=True)
            ):
                ratio_variance = summary.ratio_variance[power_index, band_index]
                writer.writerow(
                    (
                        condition,
                        int(summary.counts[power_index]),
                        lower,
                        upper,
                        summary.experimental_mean[power_index, band_index],
                        summary.experimental_variance[power_index, band_index],
                        summary.reconstructed_mean[power_index, band_index],
                        summary.reconstructed_variance[power_index, band_index],
                        summary.ratio_mean[power_index, band_index],
                        ratio_variance,
                        np.sqrt(ratio_variance),
                    )
                )
    os.replace(temporary_path, path)


def configure_ratio_axis(axis: plt.Axes, rank: int, edges: np.ndarray) -> None:
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal = 1")
    for edge in edges:
        axis.axvline(
            edge,
            color="0.25",
            linestyle="-",
            linewidth=1.15,
            alpha=0.55,
            zorder=0,
        )
    axis.set_xscale("log")
    axis.set_xlim(edges[0], edges[-1])
    axis.set_xlabel("Frequency band center [Hz]")
    axis.set_ylabel("Reconstructed / experimental band power")
    axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axis.set_title(f"PSD band-power agreement — reconstruction rank {rank}")


def save_plots(
    output_dir: Path,
    summary: Summary,
    edges: np.ndarray,
    rank: int,
    dpi: int,
) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    outputs = []
    for index, condition in enumerate(summary.conditions):
        mean = summary.ratio_mean[index]
        standard_deviation = np.sqrt(summary.ratio_variance[index])
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        axis.errorbar(
            centers,
            mean,
            yerr=standard_deviation,
            color="C0",
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=3,
            label="mean +/- 1 std",
        )
        configure_ratio_axis(axis, rank, edges)
        axis.set_title(
            f"PSD band-power agreement — {condition}, rank {rank}\n"
            f"{summary.counts[index]} repetitions"
        )
        axis.legend(frameon=False)
        figure.tight_layout()
        output_path = plot_dir / f"{condition}__mean_band_power_ratio_r{rank}.png"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output_path)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(summary.conditions)))
    for index, (condition, color) in enumerate(
        zip(summary.conditions, colors, strict=True)
    ):
        mean = summary.ratio_mean[index]
        standard_deviation = np.sqrt(summary.ratio_variance[index])
        axis.plot(
            centers,
            mean,
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=condition,
        )
        axis.fill_between(
            centers,
            np.maximum(0.0, mean - standard_deviation),
            mean + standard_deviation,
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    configure_ratio_axis(axis, rank, edges)
    axis.legend(
        title="Input power",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
    )
    figure.tight_layout()
    combined_path = plot_dir / f"all_powers__mean_band_power_ratio_r{rank}.png"
    figure.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    outputs.append(combined_path)
    return outputs


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    reduced_root = args.reduced_root.expanduser().resolve()
    if not reduced_root.is_dir():
        raise FileNotFoundError(f"Reduced-data root does not exist: {reduced_root}")
    comparison_dir = (
        args.comparison_dir.expanduser().resolve()
        if args.comparison_dir is not None
        else reduced_root / "center_psd_comparisons"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else comparison_dir / f"band_power_rank_{args.rank}"
    )
    result_path = output_dir / "band_power_metrics.h5"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Results already exist: {result_path}. Use --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs, discovery_issues = discover_jobs(
        reduced_root, comparison_dir, args.pod_rank
    )
    if args.max_files is not None:
        jobs = jobs[: args.max_files]
    print(
        f"Discovered {len(jobs)} valid repetition(s); "
        f"{len(discovery_issues)} invalid folder(s).",
        flush=True,
    )
    if not jobs:
        raise RuntimeError("No valid repetitions were found.")

    started = time.perf_counter()
    results: list[Result] = []
    failures = list(discovery_issues)
    common_edges: np.ndarray | None = None
    for index, job in enumerate(jobs, start=1):
        label = f"{job.condition}/{job.realization}"
        job_started = time.perf_counter()
        try:
            result, edges = compute_result(job, args)
            if common_edges is None:
                common_edges = edges
            elif not np.allclose(edges, common_edges, rtol=1e-9, atol=1e-9):
                raise ValueError("Frequency-band edges differ from earlier jobs.")
            results.append(result)
        except (OSError, KeyError, ValueError) as error:
            failure = f"{label}: {error}"
            failures.append(failure)
            print(f"FAILED [{index}/{len(jobs)}] {failure}", flush=True)
            continue
        print(
            f"DONE [{index}/{len(jobs)}] {label} "
            f"({time.perf_counter() - job_started:.2f} s)",
            flush=True,
        )

    if not results or common_edges is None:
        raise RuntimeError("Every repetition failed; no results were saved.")
    summary = aggregate(results)
    save_hdf5(result_path, results, summary, common_edges, args, failures)
    csv_path = output_dir / "band_power_summary.csv"
    save_summary_csv(csv_path, summary, common_edges)
    plots = save_plots(output_dir, summary, common_edges, args.rank, args.dpi)
    print()
    print(
        f"Completed {len(results)}/{len(jobs)} repetition(s) in "
        f"{time.perf_counter() - started:.1f} s."
    )
    print(f"Saved numerical results: {result_path}")
    print(f"Saved summary table:      {csv_path}")
    print(f"Saved {len(plots)} plot(s):          {output_dir / 'plots'}")
    if failures:
        print(f"Skipped {len(failures)} invalid/failed repetition(s); see HDF5 failures.")
    return 1 if len(results) < len(jobs) else 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
