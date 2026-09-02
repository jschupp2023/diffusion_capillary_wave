"""Compute an efficient POD reduction of the calibrated 2-D wave data.

The input file stores one 200 x 200 HDF5 dataset per time step. This script:

1. streams over every frame to compute preprocessing statistics;
2. learns a candidate spatial subspace from stratified-random snapshots;
3. projects every frame into that subspace;
4. rotates the candidate modes using the full-data reduced covariance; and
5. saves the leading modes, singular values, and complete coefficient history.

The full snapshot matrix is never held in memory.

Example
-------
    conda run -n capillarywave python pod_2d.py \
        --rank 100 --training-frames 5000 --output pod_2d_r100.h5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np


DEFAULT_DATA_PATH = Path(
    "/home/jonas/ucsd_thesis/11222025_j_0.04vpp_data_roi-none_cal-true.hdf5"
)
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class InputInfo:
    path: Path
    source_positions: np.ndarray
    frame_numbers: np.ndarray
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frame_shape: tuple[int, int]
    units: dict[str, str]

    @property
    def n_frames(self) -> int:
        return len(self.frame_numbers)

    @property
    def n_space(self) -> int:
        return int(np.prod(self.frame_shape))


@dataclass(frozen=True)
class PreprocessingStatistics:
    frame_spatial_mean: np.ndarray
    temporal_mean_field: np.ndarray


@dataclass(frozen=True)
class RefinedPOD:
    modes: np.ndarray
    rotation: np.ndarray
    singular_values: np.ndarray
    projected_candidate_singular_values: np.ndarray
    cumulative_energy_fraction: np.ndarray
    candidate_energy_fraction: float
    total_snapshot_energy: float
    orthonormality_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Input HDF5 file (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HDF5 file (default: pod_2d_r<RANK>.h5).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=100,
        help="Number of final POD modes to save (default: 100).",
    )
    parser.add_argument(
        "--training-frames",
        type=int,
        default=5_000,
        help="Number of stratified-random training frames (default: 5000).",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=20,
        help="Candidate modes in addition to --rank (default: 20).",
    )
    parser.add_argument(
        "--power-iterations",
        type=int,
        default=1,
        help="Randomized-SVD power iterations (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Frames read and projected at once (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12_345,
        help="Random seed for sampling and randomized SVD (default: 12345).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First source-array position to include (default: 0).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="Exclusive source-array stop position (default: all frames).",
    )
    parser.add_argument(
        "--keep-spatial-mean",
        action="store_true",
        help="Keep each frame's instantaneous spatial mean/piston motion.",
    )
    parser.add_argument(
        "--no-temporal-centering",
        action="store_true",
        help="Do not subtract the temporal mean field before POD.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Storage type for modes and coefficients (default: float32).",
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="HDF5 compression for large output arrays (default: lzf).",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        choices=range(1, 10),
        default=4,
        metavar="1-9",
        help="gzip level when --compression=gzip (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def _compression_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.compression == "none":
        return {}
    if args.compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {
        "compression": "gzip",
        "compression_opts": args.gzip_level,
        "shuffle": True,
    }


def inspect_input(path: Path, start: int, stop: int | None) -> InputInfo:
    """Validate the keyed-frame HDF5 input and load its small metadata arrays."""
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        for object_name in ("main", "meta/t", "meta/x", "meta/y"):
            if object_name not in handle:
                raise KeyError(f"Missing HDF5 object {object_name!r} in {path}.")

        frame_group = handle["main"]
        meta = handle["meta"]
        n_total = len(meta["t"])
        if "frames" in meta:
            all_frame_numbers = meta["frames"]
            if len(all_frame_numbers) != n_total:
                raise ValueError("meta/frames and meta/t have different lengths.")
        else:
            all_frame_numbers = None

        if stop is None:
            stop = n_total
        if not 0 <= start < stop <= n_total:
            raise ValueError(
                f"Require 0 <= start < stop <= {n_total:,}; got "
                f"start={start:,}, stop={stop:,}."
            )

        if all_frame_numbers is None:
            frame_numbers = np.arange(start, stop, dtype=np.int64)
        else:
            frame_numbers = np.asarray(
                all_frame_numbers[start:stop], dtype=np.int64
            )
        source_positions = np.arange(start, stop, dtype=np.int64)
        time_values = np.asarray(meta["t"][start:stop], dtype=np.float64)
        x = np.asarray(meta["x"], dtype=np.float64)
        y = np.asarray(meta["y"], dtype=np.float64)
        units = {
            "time": str(meta.attrs.get("t_units", "")),
            "x": str(meta.attrs.get("x_units", "")),
            "y": str(meta.attrs.get("y_units", "")),
            "z": str(meta.attrs.get("z_units", "")),
        }

        first_key = _frame_key(frame_numbers[0])
        last_key = _frame_key(frame_numbers[-1])
        for key in (first_key, last_key):
            if key not in frame_group:
                raise KeyError(f"Missing HDF5 frame /main/{key}.")
        frame_shape = tuple(int(value) for value in frame_group[first_key].shape)
        if len(frame_shape) != 2:
            raise ValueError(
                f"Expected two-dimensional frames; got shape {frame_shape}."
            )
        if frame_shape != (len(y), len(x)):
            raise ValueError(
                f"Frame shape {frame_shape} does not match y/x grid "
                f"lengths {(len(y), len(x))}."
            )

    return InputInfo(
        path=path,
        source_positions=source_positions,
        frame_numbers=frame_numbers,
        time=time_values,
        x=x,
        y=y,
        frame_shape=frame_shape,
        units=units,
    )


def _frame_key(frame_number: int) -> str:
    return f"{int(frame_number):05d}"


def _read_frame_block(
    frame_group: h5py.Group,
    frame_numbers: np.ndarray,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    block = np.empty((len(frame_numbers), int(np.prod(frame_shape))), dtype=np.float32)
    for row, frame_number in enumerate(frame_numbers):
        key = _frame_key(frame_number)
        if key not in frame_group:
            raise KeyError(f"Missing HDF5 frame /main/{key}.")
        frame = np.asarray(frame_group[key], dtype=np.float32)
        if frame.shape != frame_shape:
            raise ValueError(
                f"Frame /main/{key} has shape {frame.shape}; expected {frame_shape}."
            )
        if not np.isfinite(frame).all():
            raise ValueError(f"Frame /main/{key} contains NaN or infinite values.")
        block[row] = frame.reshape(-1)
    return block


def _iter_frame_blocks(
    frame_group: h5py.Group,
    info: InputInfo,
    batch_size: int,
    stage: str,
):
    n_blocks = (info.n_frames + batch_size - 1) // batch_size
    report_every = max(1, n_blocks // 10)
    started = time.perf_counter()
    for block_number, local_start in enumerate(
        range(0, info.n_frames, batch_size), start=1
    ):
        local_stop = min(local_start + batch_size, info.n_frames)
        block = _read_frame_block(
            frame_group,
            info.frame_numbers[local_start:local_stop],
            info.frame_shape,
        )
        yield local_start, block
        if block_number % report_every == 0 or block_number == n_blocks:
            elapsed = time.perf_counter() - started
            print(
                f"{stage}: {local_stop:,}/{info.n_frames:,} frames "
                f"({100 * local_stop / info.n_frames:.0f}%, {elapsed:.1f} s)",
                flush=True,
            )


def compute_preprocessing_statistics(
    info: InputInfo,
    batch_size: int,
    remove_spatial_mean: bool,
) -> PreprocessingStatistics:
    """Stream all frames to compute piston offsets and the temporal mean field."""
    frame_spatial_mean = np.empty(info.n_frames, dtype=np.float64)
    temporal_sum = np.zeros(info.n_space, dtype=np.float64)

    with h5py.File(info.path, "r") as handle:
        frame_group = handle["main"]
        for local_start, block in _iter_frame_blocks(
            frame_group, info, batch_size, "Mean pass"
        ):
            local_stop = local_start + len(block)
            spatial_means = block.mean(axis=1, dtype=np.float64)
            frame_spatial_mean[local_start:local_stop] = spatial_means
            if remove_spatial_mean:
                block -= spatial_means.astype(np.float32)[:, None]
            temporal_sum += block.sum(axis=0, dtype=np.float64)

    temporal_mean_field = temporal_sum / info.n_frames
    return PreprocessingStatistics(
        frame_spatial_mean=frame_spatial_mean,
        temporal_mean_field=temporal_mean_field,
    )


def _preprocess_block(
    block: np.ndarray,
    spatial_means: np.ndarray,
    temporal_mean_field: np.ndarray,
    remove_spatial_mean: bool,
    temporal_center: bool,
) -> np.ndarray:
    if remove_spatial_mean:
        block -= spatial_means.astype(np.float32, copy=False)[:, None]
    if temporal_center:
        block -= temporal_mean_field.astype(np.float32, copy=False)[None, :]
    return block


def stratified_training_positions(
    n_frames: int,
    n_training: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose one random frame from each of equally spaced temporal bins."""
    edges = np.linspace(0, n_frames, n_training + 1, dtype=np.int64)
    positions = np.empty(n_training, dtype=np.int64)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        positions[index] = rng.integers(left, right)
    return positions


def load_training_matrix(
    info: InputInfo,
    statistics: PreprocessingStatistics,
    training_positions: np.ndarray,
    remove_spatial_mean: bool,
    temporal_center: bool,
) -> np.ndarray:
    """Load and preprocess only the selected training snapshots."""
    print(f"Loading {len(training_positions):,} training frames...", flush=True)
    started = time.perf_counter()
    with h5py.File(info.path, "r") as handle:
        training = _read_frame_block(
            handle["main"],
            info.frame_numbers[training_positions],
            info.frame_shape,
        )
    _preprocess_block(
        training,
        statistics.frame_spatial_mean[training_positions],
        statistics.temporal_mean_field,
        remove_spatial_mean,
        temporal_center,
    )
    print(
        f"Loaded training matrix {training.shape} in "
        f"{time.perf_counter() - started:.1f} s.",
        flush=True,
    )
    return training


def randomized_spatial_subspace(
    training: np.ndarray,
    candidate_rank: int,
    power_iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute approximate leading right singular vectors of the training data."""
    print(
        f"Randomized SVD: {candidate_rank} candidate modes, "
        f"{power_iterations} power iteration(s).",
        flush=True,
    )
    started = time.perf_counter()
    omega = rng.standard_normal(
        (training.shape[1], candidate_rank), dtype=np.float32
    )
    sample_range = training @ omega
    del omega

    for iteration in range(power_iterations):
        left_basis, _ = np.linalg.qr(sample_range, mode="reduced")
        spatial_range = training.T @ left_basis
        spatial_basis, _ = np.linalg.qr(spatial_range, mode="reduced")
        sample_range = training @ spatial_basis
        print(
            f"Randomized SVD power iteration {iteration + 1}/{power_iterations}",
            flush=True,
        )

    left_basis, _ = np.linalg.qr(sample_range, mode="reduced")
    small_matrix = left_basis.T @ training
    del sample_range, left_basis
    _, training_singular_values, right_vectors = np.linalg.svd(
        small_matrix, full_matrices=False
    )
    candidate_basis = np.ascontiguousarray(
        right_vectors[:candidate_rank].T, dtype=np.float32
    )
    training_singular_values = np.asarray(
        training_singular_values[:candidate_rank], dtype=np.float64
    )
    print(
        f"Randomized SVD completed in {time.perf_counter() - started:.1f} s.",
        flush=True,
    )
    return candidate_basis, training_singular_values


def project_full_dataset(
    info: InputInfo,
    statistics: PreprocessingStatistics,
    candidate_basis: np.ndarray,
    batch_size: int,
    remove_spatial_mean: bool,
    temporal_center: bool,
    work_directory: Path,
) -> tuple[np.memmap, Path, np.ndarray, float]:
    """Project all frames and accumulate the reduced covariance and total energy."""
    candidate_rank = candidate_basis.shape[1]
    coefficient_file = tempfile.NamedTemporaryFile(
        prefix=".pod_candidate_coefficients.",
        suffix=".dat",
        dir=work_directory,
        delete=False,
    )
    coefficient_path = Path(coefficient_file.name)
    coefficient_file.close()
    coefficients = np.memmap(
        coefficient_path,
        mode="w+",
        dtype=np.float32,
        shape=(info.n_frames, candidate_rank),
    )
    reduced_covariance = np.zeros(
        (candidate_rank, candidate_rank), dtype=np.float64
    )
    total_snapshot_energy = 0.0

    try:
        with h5py.File(info.path, "r") as handle:
            frame_group = handle["main"]
            for local_start, block in _iter_frame_blocks(
                frame_group, info, batch_size, "Projection pass"
            ):
                local_stop = local_start + len(block)
                _preprocess_block(
                    block,
                    statistics.frame_spatial_mean[local_start:local_stop],
                    statistics.temporal_mean_field,
                    remove_spatial_mean,
                    temporal_center,
                )
                total_snapshot_energy += float(
                    np.einsum(
                        "ij,ij->",
                        block,
                        block,
                        dtype=np.float64,
                        optimize=True,
                    )
                )
                block_coefficients = block @ candidate_basis
                coefficients[local_start:local_stop] = block_coefficients
                coefficients64 = np.asarray(block_coefficients, dtype=np.float64)
                reduced_covariance += coefficients64.T @ coefficients64
        coefficients.flush()
    except BaseException:
        del coefficients
        coefficient_path.unlink(missing_ok=True)
        raise

    if not np.isfinite(total_snapshot_energy) or total_snapshot_energy <= 0:
        del coefficients
        coefficient_path.unlink(missing_ok=True)
        raise ValueError("Preprocessed snapshots have no finite, positive energy.")
    return (
        coefficients,
        coefficient_path,
        reduced_covariance,
        total_snapshot_energy,
    )


def refine_pod(
    candidate_basis: np.ndarray,
    reduced_covariance: np.ndarray,
    total_snapshot_energy: float,
    rank: int,
) -> RefinedPOD:
    """Rotate the candidate basis using covariance from every source frame."""
    eigenvalues, rotation = np.linalg.eigh(reduced_covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    rotation = rotation[:, order]
    projected_candidate_singular_values = np.sqrt(eigenvalues)
    cumulative_candidate_energy = np.cumsum(eigenvalues) / total_snapshot_energy

    retained_rotation = np.asarray(rotation[:, :rank], dtype=np.float32)
    modes = np.asarray(candidate_basis @ retained_rotation, dtype=np.float32)

    # Fix arbitrary POD signs and apply the same signs to the coefficient rotation.
    for mode_index in range(rank):
        pivot = int(np.argmax(np.abs(modes[:, mode_index])))
        if modes[pivot, mode_index] < 0:
            modes[:, mode_index] *= -1
            retained_rotation[:, mode_index] *= -1

    gram_error = modes.T.astype(np.float64) @ modes.astype(np.float64)
    gram_error -= np.eye(rank)
    orthonormality_error = float(np.max(np.abs(gram_error)))

    return RefinedPOD(
        modes=modes,
        rotation=retained_rotation,
        singular_values=projected_candidate_singular_values[:rank],
        projected_candidate_singular_values=projected_candidate_singular_values,
        cumulative_energy_fraction=cumulative_candidate_energy[:rank],
        candidate_energy_fraction=float(cumulative_candidate_energy[-1]),
        total_snapshot_energy=total_snapshot_energy,
        orthonormality_error=orthonormality_error,
    )


def write_output(
    output_path: Path,
    args: argparse.Namespace,
    info: InputInfo,
    statistics: PreprocessingStatistics,
    training_positions: np.ndarray,
    training_singular_values: np.ndarray,
    candidate_coefficients: np.memmap,
    pod: RefinedPOD,
    remove_spatial_mean: bool,
    temporal_center: bool,
) -> None:
    """Write a self-contained POD result to an HDF5 file atomically."""
    storage_dtype = np.dtype(args.storage_dtype)
    compression = _compression_kwargs(args)
    rank = pod.modes.shape[1]
    candidate_rank = candidate_coefficients.shape[1]
    coefficient_chunk = (min(args.batch_size, info.n_frames), rank)

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    try:
        with h5py.File(temporary_path, "w", track_order=True) as output:
            output.attrs["schema_version"] = SCHEMA_VERSION
            output.attrs["description"] = "Hybrid sampled/streaming POD of 2-D data"
            output.attrs["generator"] = Path(__file__).name
            output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            output.attrs["source_file"] = str(info.path.resolve())
            output.attrs["source_start_position"] = int(info.source_positions[0])
            output.attrs["source_stop_position_exclusive"] = int(
                info.source_positions[-1] + 1
            )
            output.attrs["n_frames"] = info.n_frames
            output.attrs["frame_shape"] = info.frame_shape
            output.attrs["n_space"] = info.n_space
            output.attrs["rank"] = rank
            output.attrs["candidate_rank"] = candidate_rank
            output.attrs["training_frames"] = len(training_positions)
            output.attrs["random_seed"] = args.seed
            output.attrs["power_iterations"] = args.power_iterations
            output.attrs["instantaneous_spatial_mean_removed"] = remove_spatial_mean
            output.attrs["temporal_mean_field_removed"] = temporal_center
            output.attrs["storage_dtype"] = args.storage_dtype

            source = output.create_group("source")
            position_dataset = source.create_dataset(
                "position", data=info.source_positions, track_times=False
            )
            position_dataset.attrs["description"] = (
                "Position in the source meta arrays"
            )
            frame_dataset = source.create_dataset(
                "frame_number", data=info.frame_numbers, track_times=False
            )
            frame_dataset.attrs["description"] = (
                "Frame number used as the keyed /main dataset name"
            )

            grid = output.create_group("grid")
            time_dataset = grid.create_dataset(
                "time", data=info.time, track_times=False
            )
            time_dataset.attrs["units"] = info.units["time"]
            x_dataset = grid.create_dataset("x", data=info.x, track_times=False)
            x_dataset.attrs["units"] = info.units["x"]
            y_dataset = grid.create_dataset("y", data=info.y, track_times=False)
            y_dataset.attrs["units"] = info.units["y"]

            preprocessing = output.create_group("preprocessing")
            spatial_mean_dataset = preprocessing.create_dataset(
                "frame_spatial_mean",
                data=statistics.frame_spatial_mean,
                track_times=False,
            )
            spatial_mean_dataset.attrs["units"] = info.units["z"]
            spatial_mean_dataset.attrs["subtracted_before_pod"] = (
                remove_spatial_mean
            )
            mean_field_dataset = preprocessing.create_dataset(
                "temporal_mean_field",
                data=statistics.temporal_mean_field.reshape(info.frame_shape),
                track_times=False,
            )
            mean_field_dataset.attrs["axis_order"] = "y, x"
            mean_field_dataset.attrs["units"] = info.units["z"]
            mean_field_dataset.attrs["subtracted_before_pod"] = temporal_center

            training = output.create_group("training")
            training.create_dataset(
                "source_position",
                data=info.source_positions[training_positions],
                track_times=False,
            )
            training.create_dataset(
                "frame_number",
                data=info.frame_numbers[training_positions],
                track_times=False,
            )
            sampled_singular_dataset = training.create_dataset(
                "sample_singular_values",
                data=training_singular_values,
                track_times=False,
            )
            sampled_singular_dataset.attrs["description"] = (
                "Randomized-SVD singular values of the sampled training matrix; "
                "not the final full-data singular values"
            )
            training.attrs["sampling"] = (
                "one uniformly random frame from each equal temporal bin"
            )

            pod_group = output.create_group("pod")
            modes_dataset = pod_group.create_dataset(
                "modes",
                data=pod.modes.T.reshape(rank, *info.frame_shape).astype(
                    storage_dtype, copy=False
                ),
                chunks=(1, *info.frame_shape),
                track_times=False,
                **compression,
            )
            modes_dataset.attrs["axis_order"] = "mode, y, x"
            modes_dataset.attrs["description"] = "Orthonormal spatial POD modes"
            singular_dataset = pod_group.create_dataset(
                "singular_values", data=pod.singular_values, track_times=False
            )
            singular_dataset.attrs["description"] = (
                "Singular values from the full-data covariance projected into "
                "the sampled candidate subspace"
            )
            energy_dataset = pod_group.create_dataset(
                "cumulative_energy_fraction",
                data=pod.cumulative_energy_fraction,
                track_times=False,
            )
            energy_dataset.attrs["description"] = (
                "Cumulative retained energy divided by total preprocessed "
                "snapshot energy"
            )
            pod_group.create_dataset(
                "projected_candidate_singular_values",
                data=pod.projected_candidate_singular_values,
                track_times=False,
            )
            pod_group.attrs["method"] = (
                "stratified sampled randomized SVD followed by full-data "
                "projected covariance rotation"
            )
            pod_group.attrs["spatial_inner_product"] = (
                "unweighted Euclidean sum over pixels; x and y spacing are uniform"
            )
            pod_group.attrs["retained_energy_fraction"] = float(
                pod.cumulative_energy_fraction[-1]
            )
            pod_group.attrs["candidate_subspace_energy_fraction"] = (
                pod.candidate_energy_fraction
            )
            pod_group.attrs["total_preprocessed_snapshot_energy"] = (
                pod.total_snapshot_energy
            )
            pod_group.attrs["max_orthonormality_error"] = (
                pod.orthonormality_error
            )

            reduced = output.create_group("reduced")
            coefficient_dataset = reduced.create_dataset(
                "coefficients",
                shape=(info.n_frames, rank),
                dtype=storage_dtype,
                chunks=coefficient_chunk,
                track_times=False,
                **compression,
            )
            coefficient_dataset.attrs["axis_order"] = "time, mode"
            coefficient_dataset.attrs["units"] = info.units["z"]
            coefficient_dataset.attrs["definition"] = (
                "preprocessed_flattened_snapshots @ flattened_modes.T"
            )
            for local_start in range(0, info.n_frames, args.batch_size):
                local_stop = min(local_start + args.batch_size, info.n_frames)
                coefficient_dataset[local_start:local_stop] = (
                    candidate_coefficients[local_start:local_stop]
                    @ pod.rotation
                ).astype(storage_dtype, copy=False)
            reduced.attrs["processed_reconstruction"] = (
                "coefficients[t] contracted with pod/modes, plus "
                "preprocessing/temporal_mean_field when that field was removed"
            )
            reduced.attrs["absolute_reconstruction"] = (
                "processed reconstruction plus frame_spatial_mean[t] when "
                "instantaneous spatial means were removed"
            )

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_args(args: argparse.Namespace, info: InputInfo) -> int:
    for name in ("rank", "training_frames", "batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.oversampling < 0:
        raise ValueError("--oversampling must be non-negative.")
    if args.power_iterations < 0:
        raise ValueError("--power-iterations must be non-negative.")
    if args.training_frames > info.n_frames:
        raise ValueError(
            f"--training-frames={args.training_frames:,} exceeds the selected "
            f"{info.n_frames:,} frames."
        )
    candidate_rank = args.rank + args.oversampling
    maximum_rank = min(args.training_frames, info.n_space)
    if candidate_rank > maximum_rank:
        raise ValueError(
            f"rank + oversampling = {candidate_rank} exceeds the sampled "
            f"maximum rank {maximum_rank}."
        )
    return candidate_rank


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else Path(f"pod_2d_r{args.rank}.h5")
    )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = inspect_input(input_path, args.start, args.stop)
    candidate_rank = _validate_args(args, info)
    remove_spatial_mean = not args.keep_spatial_mean
    temporal_center = not args.no_temporal_centering

    training_mib = args.training_frames * info.n_space * 4 / 2**20
    coefficient_mib = info.n_frames * candidate_rank * 4 / 2**20
    print(
        f"Input: {info.path.resolve()}\n"
        f"Selected {info.n_frames:,} frames of shape {info.frame_shape}; "
        f"rank {args.rank}, candidate rank {candidate_rank}.\n"
        f"Training matrix: {args.training_frames:,} stratified frames "
        f"(~{training_mib:.1f} MiB float32).\n"
        f"Temporary candidate coefficients: ~{coefficient_mib:.1f} MiB.\n"
        f"Remove instantaneous spatial mean: {remove_spatial_mean}.\n"
        f"Subtract temporal mean field: {temporal_center}.",
        flush=True,
    )
    if not remove_spatial_mean:
        print(
            "Note: piston/global-height motion is retained and may dominate "
            "the leading POD mode.",
            flush=True,
        )

    seed_sequence = np.random.SeedSequence(args.seed)
    sampling_seed, svd_seed = seed_sequence.spawn(2)
    training_positions = stratified_training_positions(
        info.n_frames,
        args.training_frames,
        np.random.default_rng(sampling_seed),
    )

    statistics = compute_preprocessing_statistics(
        info,
        args.batch_size,
        remove_spatial_mean,
    )
    training = load_training_matrix(
        info,
        statistics,
        training_positions,
        remove_spatial_mean,
        temporal_center,
    )
    candidate_basis, training_singular_values = randomized_spatial_subspace(
        training,
        candidate_rank,
        args.power_iterations,
        np.random.default_rng(svd_seed),
    )
    del training

    candidate_coefficients: np.memmap | None = None
    coefficient_path: Path | None = None
    try:
        (
            candidate_coefficients,
            coefficient_path,
            reduced_covariance,
            total_snapshot_energy,
        ) = project_full_dataset(
            info,
            statistics,
            candidate_basis,
            args.batch_size,
            remove_spatial_mean,
            temporal_center,
            output_path.parent,
        )
        pod = refine_pod(
            candidate_basis,
            reduced_covariance,
            total_snapshot_energy,
            args.rank,
        )
        print(
            f"Candidate subspace captures {pod.candidate_energy_fraction:.6%} "
            f"of total preprocessed energy.\n"
            f"Rank-{args.rank} captures "
            f"{pod.cumulative_energy_fraction[-1]:.6%}.\n"
            f"Maximum mode orthonormality error: "
            f"{pod.orthonormality_error:.3e}.",
            flush=True,
        )
        write_output(
            output_path,
            args,
            info,
            statistics,
            training_positions,
            training_singular_values,
            candidate_coefficients,
            pod,
            remove_spatial_mean,
            temporal_center,
        )
    finally:
        if candidate_coefficients is not None:
            candidate_coefficients.flush()
            del candidate_coefficients
        if coefficient_path is not None:
            coefficient_path.unlink(missing_ok=True)

    size_mib = output_path.stat().st_size / 2**20
    print(f"Saved {output_path.resolve()} ({size_mib:.1f} MiB).", flush=True)


if __name__ == "__main__":
    main()
