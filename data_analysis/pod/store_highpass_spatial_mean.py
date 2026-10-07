"""Append high-pass-filtered spatial-mean trajectories to POD HDF5 files.

The original datasets are read but never replaced or deleted. By default this
processes every rank-1000 POD file for one condition and stores 10, 20, 30,
40, 50, and 100 Hz variants::

    python -m data_analysis.pod.store_highpass_spatial_mean 0p20

Use ``--dry-run`` to validate every input and report the planned additions
without opening any file for writing. Existing matching datasets are skipped;
an existing dataset with different data or metadata is treated as an error.
For every repetition and cutoff, the preflight also reports the fraction
``rho = mean(filtered_spatial_mean**2) / mean(spatial_mean**2)`` of the
spatial-mean squared-norm energy retained by the filter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
import numpy as np

from data_analysis.energy.pod_gravitational_energy import highpass_spatial_mean
from data_analysis.pod.pod_analysis import (
    DEFAULT_ROOT,
    discover_results,
    resolve_condition_directory,
)


DEFAULT_CUTOFFS_HZ = (10.0, 20.0, 30.0, 40.0, 50.0, 100.0)
SOURCE_DATASET = "preprocessing/frame_spatial_mean"
TIME_DATASET = "grid/time"
OUTPUT_GROUP = "preprocessing/frame_spatial_mean_highpass"


@dataclass(frozen=True)
class FilePlan:
    path: Path
    existing: tuple[float, ...]
    missing: tuple[float, ...]
    energy_retained: tuple[tuple[float, float], ...]


def cutoff_tag(cutoff_hz: float) -> str:
    """Return a stable HDF5-safe representation of a positive cutoff."""
    return f"{cutoff_hz:.12g}".replace(".", "p").replace("+", "")


def dataset_name(cutoff_hz: float) -> str:
    return f"cutoff_{cutoff_tag(cutoff_hz)}_hz"


def _validated_cutoffs(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    cutoffs = tuple(float(value) for value in values)
    if not cutoffs or any(not np.isfinite(value) or value <= 0 for value in cutoffs):
        raise ValueError("High-pass cutoffs must be finite and positive.")
    names = [dataset_name(value) for value in cutoffs]
    if len(set(names)) != len(names):
        raise ValueError("High-pass cutoffs must be unique at the stored-name precision.")
    return cutoffs


def spatial_mean_energy_retained(
    spatial_mean: np.ndarray, filtered_spatial_mean: np.ndarray
) -> float:
    """Return mean(filtered**2) / mean(original**2) for two trajectories.

    The common number of spatial points in the squared-norm energy cancels
    from this ratio. The ratio is undefined when the original trajectory has
    zero mean-square energy.
    """
    original = np.asarray(spatial_mean, dtype=np.float64)
    filtered = np.asarray(filtered_spatial_mean, dtype=np.float64)
    if original.ndim != 1 or filtered.shape != original.shape:
        raise ValueError(
            "Original and filtered spatial means must be matching "
            "one-dimensional trajectories."
        )
    if not np.isfinite(original).all() or not np.isfinite(filtered).all():
        raise ValueError("Original and filtered spatial means must be finite.")

    original_mean_square = float(np.mean(np.square(original)))
    if original_mean_square == 0.0:
        raise ValueError(
            "Spatial-mean energy-retention ratio is undefined because the "
            "original spatial mean has zero mean-square energy."
        )
    fraction = float(np.mean(np.square(filtered)) / original_mean_square)
    if not np.isfinite(fraction):
        raise ValueError("Spatial-mean energy-retention ratio is nonfinite.")
    return fraction


def _read_source(handle: h5py.File, path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    for name in (TIME_DATASET, SOURCE_DATASET):
        if name not in handle:
            raise KeyError(f"Missing dataset {name!r} in {path}.")
    time_dataset = handle[TIME_DATASET]
    mean_dataset = handle[SOURCE_DATASET]
    if not isinstance(time_dataset, h5py.Dataset) or not isinstance(mean_dataset, h5py.Dataset):
        raise TypeError(f"Time and spatial mean must be datasets in {path}.")
    time = np.asarray(time_dataset, dtype=np.float64)
    spatial_mean = np.asarray(mean_dataset, dtype=np.float64)
    units = str(mean_dataset.attrs.get("units", ""))
    if not units:
        raise ValueError(f"Missing units on {SOURCE_DATASET} in {path}.")
    if time.shape != spatial_mean.shape or time.ndim != 1:
        raise ValueError(
            f"Time and spatial mean must be matching one-dimensional arrays in {path}; "
            f"got {time.shape} and {spatial_mean.shape}."
        )
    return time, spatial_mean, units


def _expected_attributes(cutoff_hz: float, units: str, sampling_hz: float) -> dict[str, object]:
    return {
        "units": units,
        "source_dataset": f"/{SOURCE_DATASET}",
        "filter_family": "Butterworth",
        "filter_type": "highpass",
        "filter_order": np.int64(4),
        "cutoff_hz": np.float64(cutoff_hz),
        "sampling_frequency_hz": np.float64(sampling_hz),
        "phase_treatment": "zero-phase forward-backward",
        "implementation": "scipy.signal.sosfiltfilt",
    }


def _attribute_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, (float, np.floating)):
        try:
            return bool(np.isclose(float(actual), float(expected), rtol=1e-12, atol=0.0))
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _validate_existing(
    dataset: h5py.Dataset,
    expected_values: np.ndarray,
    expected_attributes: dict[str, object],
    path: Path,
) -> None:
    if dataset.shape != expected_values.shape or dataset.dtype != np.dtype("float64"):
        raise ValueError(
            f"Existing {dataset.name} in {path} has shape/dtype "
            f"{dataset.shape}/{dataset.dtype}, expected {expected_values.shape}/float64; "
            "refusing to overwrite it."
        )
    for name, expected in expected_attributes.items():
        if name not in dataset.attrs or not _attribute_matches(dataset.attrs[name], expected):
            raise ValueError(
                f"Existing {dataset.name} in {path} has incompatible attribute {name!r}; "
                "refusing to overwrite it."
            )
    stored = np.asarray(dataset, dtype=np.float64)
    if not np.array_equal(stored, expected_values):
        raise ValueError(
            f"Existing {dataset.name} in {path} differs from the recomputed trajectory; "
            "refusing to overwrite it."
        )


def inspect_file(path: Path, cutoffs_hz: tuple[float, ...]) -> FilePlan:
    """Validate one file read-only and determine which datasets are missing."""
    with h5py.File(path, "r") as handle:
        time, spatial_mean, units = _read_source(handle, path)
        sampling_hz = 1.0 / float(np.mean(np.diff(time)))
        output = handle.get(OUTPUT_GROUP)
        if output is not None and not isinstance(output, h5py.Group):
            raise TypeError(f"{OUTPUT_GROUP!r} exists but is not a group in {path}.")

        existing: list[float] = []
        missing: list[float] = []
        energy_retained: list[tuple[float, float]] = []
        for cutoff_hz in cutoffs_hz:
            filtered = highpass_spatial_mean(spatial_mean, time, cutoff_hz)
            energy_retained.append(
                (
                    cutoff_hz,
                    spatial_mean_energy_retained(spatial_mean, filtered),
                )
            )
            name = dataset_name(cutoff_hz)
            if output is None or name not in output:
                missing.append(cutoff_hz)
                continue
            dataset = output[name]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"{output.name}/{name} is not a dataset in {path}.")
            _validate_existing(
                dataset,
                filtered,
                _expected_attributes(cutoff_hz, units, sampling_hz),
                path,
            )
            existing.append(cutoff_hz)
    return FilePlan(
        path=path,
        existing=tuple(existing),
        missing=tuple(missing),
        energy_retained=tuple(energy_retained),
    )


def append_missing(plan: FilePlan) -> int:
    """Append planned datasets without replacing or deleting any HDF5 object."""
    if not plan.missing:
        return 0
    # Recompute and validate all requested additions before opening the file writable.
    with h5py.File(plan.path, "r") as handle:
        time, spatial_mean, units = _read_source(handle, plan.path)
    sampling_hz = 1.0 / float(np.mean(np.diff(time)))
    filtered = {
        cutoff_hz: highpass_spatial_mean(spatial_mean, time, cutoff_hz)
        for cutoff_hz in plan.missing
    }

    with h5py.File(plan.path, "r+") as handle:
        output = handle.require_group(OUTPUT_GROUP)
        output.attrs.setdefault(
            "description",
            "Full frame_spatial_mean trajectories after zero-phase high-pass filtering",
        )
        output.attrs.setdefault("source_dataset", f"/{SOURCE_DATASET}")
        for cutoff_hz in plan.missing:
            name = dataset_name(cutoff_hz)
            if name in output:
                raise FileExistsError(
                    f"{output.name}/{name} appeared after validation in {plan.path}; "
                    "refusing to overwrite it."
                )
            values = filtered[cutoff_hz]
            dataset = output.create_dataset(
                name,
                data=values,
                dtype="float64",
                chunks=(min(len(values), 16_384),),
                compression="lzf",
                shuffle=True,
                track_times=False,
            )
            for attribute, value in _expected_attributes(
                cutoff_hz, units, sampling_hz
            ).items():
                dataset.attrs[attribute] = value
        handle.flush()
    return len(plan.missing)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "condition",
        type=Path,
        help="Condition below --root, for example 0p20, or its full path.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Reduced-data root (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--rank", type=int, default=1_000, help="POD file rank (default: 1000)."
    )
    parser.add_argument(
        "--cutoffs-hz",
        type=float,
        nargs="+",
        default=list(DEFAULT_CUTOFFS_HZ),
        metavar="HZ",
        help="Cutoffs to store (default: 10 20 30 40 50 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report additions without modifying files.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[FilePlan]:
    if args.rank < 1:
        raise ValueError("--rank must be positive.")
    cutoffs_hz = _validated_cutoffs(args.cutoffs_hz)
    condition_directory = resolve_condition_directory(args.condition, args.root)
    results, issues = discover_results(condition_directory, args.rank)
    if issues:
        raise ValueError(
            "POD discovery was incomplete; no files were modified:\n  "
            + "\n  ".join(issues)
        )
    if not results:
        raise ValueError(
            f"No rank-{args.rank} POD repetitions found in {condition_directory}."
        )

    # Preflight every repetition read-only before modifying the first file.
    plans = [inspect_file(result.path, cutoffs_hz) for result in results]
    for result, plan in zip(results, plans):
        present = ", ".join(f"{value:g}" for value in plan.existing) or "none"
        missing = ", ".join(f"{value:g}" for value in plan.missing) or "none"
        print(
            f"{result.repetition}: existing [{present}] Hz; "
            f"{'would add' if args.dry_run else 'to add'} [{missing}] Hz"
        )
        retained = ", ".join(
            f"{cutoff_hz:g} Hz: rho={fraction:.6g} ({100.0 * fraction:.6g}%)"
            for cutoff_hz, fraction in plan.energy_retained
        )
        print(f"  spatial-mean squared-norm energy retained: {retained}")

    if args.dry_run:
        print(f"Dry run complete: validated {len(plans)} POD files; no files modified.")
        return plans

    added = 0
    for result, plan in zip(results, plans):
        count = append_missing(plan)
        added += count
        print(f"{result.repetition}: appended {count} dataset(s) to {plan.path.name}")
    print(
        f"Complete: appended {added} dataset(s) across {len(plans)} POD files; "
        "no existing HDF5 object was replaced or deleted."
    )
    return plans


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
