"""Compute and save compact POD coordinates for one forcing power.

The POD convention matches ``main.ipynb``: the snapshot matrix is not
mean-centered. Each repeated experiment is kept as one uninterrupted time
series; the data are never split into realizations.

Example
-------
    python save_reduced_data.py 0p10
    python save_reduced_data.py 0p10 --rank 23 --output my_reduced_data.h5

The saved reduced state has shape ``(rank, experiment, time)`` and can be
loaded without loading the full-order measurements::

    with h5py.File("reduced_data/0p10/reduced_dynamics_0p10_r23.h5") as f:
        Xr = f["reduced/state"][:]
        Vr = f["pod/basis"][:]
        t = f["grid/time"][:]

        # Reconstruct complete experiment i if needed:
        Xi_approx = Vr @ Xr[:, i, :]

Small time blocks are used internally for I/O, but they have no meaning in the
saved data. The POD basis is obtained from the incrementally accumulated
spatial Gram matrix X X.T, avoiding the multi-gigabyte concatenated snapshot
matrix used in the notebook.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


POWER_LABELS = {
    "0p04": list("abcdefghijklmnop"),
    "0p07": list("abcdefghijklnop"),
    "0p08": list("abcefghijklnop"),
    "0p10": list("bcdefhijklmnopq"),
    "0p15": list("abcdefghijklmnop"),
    "0p18": list("abcdefghijkmnop"),
    "0p20": list("abcdefghijklmnop"),
    "0p25": list("abcdefghijklmno"),
    "0p30": list("abcdefghijklmnop"),
    "0p35": list("abcdefhijklmnop"),
}

SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class SourceFile:
    label: str
    path: Path
    n_time: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a memory-efficient POD basis and save reduced dynamics "
            "for one power level."
        )
    )
    parser.add_argument("power", choices=POWER_LABELS, help="Power label, e.g. 0p10.")
    parser.add_argument(
        "--rank",
        type=int,
        default=23,
        help="Number of POD modes to retain (default: 23).",
    )
    parser.add_argument(
        "--io-block-size",
        type=int,
        default=8192,
        help=(
            "Time samples read at once internally (default: 8192). This does "
            "not segment the saved experiments."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jonas/ucsd_thesis/DHM_new_1Dcenter"),
        help="Directory containing one subdirectory per power.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output HDF5 file (default: "
            "reduced_data/<power>/reduced_dynamics_<power>_r<rank>.h5)."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Storage type for reduced coordinates (default: float32).",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 compression for reduced coordinates (default: gzip).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        choices=range(1, 10),
        metavar="1-9",
        help="gzip compression level (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def _allclose_or_raise(
    actual: np.ndarray, reference: np.ndarray, name: str, path: Path
) -> None:
    if actual.shape != reference.shape or not np.allclose(
        actual, reference, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(f"{name} in {path} does not match the first source file.")


def inspect_sources(
    data_dir: Path, power: str
) -> tuple[list[SourceFile], np.ndarray, np.ndarray, int, int]:
    """Validate inputs and return their ordered manifest and common grids."""
    sources: list[SourceFile] = []
    reference_x: np.ndarray | None = None
    reference_time: np.ndarray | None = None
    n_space: int | None = None

    for label in POWER_LABELS[power]:
        path = data_dir / power / f"Q_1D_{power}vpp_{label}.h5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")

        with h5py.File(path, "r") as handle:
            for dataset_name in ("Q_1D", "t", "x"):
                if dataset_name not in handle:
                    raise KeyError(f"Dataset {dataset_name!r} is missing from {path}.")

            state = handle["Q_1D"]
            if state.ndim != 2:
                raise ValueError(
                    f"Q_1D in {path} must be two-dimensional; got {state.shape}."
                )
            file_n_space, n_time = state.shape
            time = np.asarray(handle["t"])
            x = np.asarray(handle["x"])
            if time.ndim != 1 or len(time) != n_time:
                raise ValueError(
                    f"t in {path} must be one-dimensional with {n_time} entries."
                )
            if x.ndim != 1 or len(x) != file_n_space:
                raise ValueError(
                    f"x in {path} must be one-dimensional with {file_n_space} entries."
                )
            if reference_x is None:
                reference_x = x.astype(np.float64, copy=False)
                reference_time = time.astype(np.float64, copy=False)
                n_space = file_n_space
            else:
                _allclose_or_raise(x, reference_x, "x", path)
                _allclose_or_raise(time, reference_time, "t", path)

            sources.append(
                SourceFile(
                    label=label,
                    path=path,
                    n_time=n_time,
                )
            )

    assert reference_x is not None
    assert reference_time is not None
    assert n_space is not None
    return sources, reference_x, reference_time, n_space, len(reference_time)


def compute_pod_basis(
    sources: list[SourceFile], io_block_size: int, rank: int, n_space: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the uncentered POD using an incrementally accumulated X X.T."""
    gram = np.zeros((n_space, n_space), dtype=np.float64)

    for experiment, source in enumerate(sources, start=1):
        with h5py.File(source.path, "r") as handle:
            state = handle["Q_1D"]
            for start in range(0, source.n_time, io_block_size):
                stop = min(start + io_block_size, source.n_time)
                snapshot_block = np.asarray(
                    state[:, start:stop], dtype=np.float64
                )
                gram += snapshot_block @ snapshot_block.T
        print(
            f"POD pass: experiment {experiment}/{len(sources)} "
            f"({source.label})",
            flush=True,
        )

    eigenvalues, basis = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = basis[:, order]

    # Roundoff can make theoretically nonnegative tail eigenvalues slightly negative.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues)
    total_energy = eigenvalues.sum()
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("The snapshot matrix has no finite, nonzero POD energy.")

    # Fix the otherwise arbitrary eigenvector signs for reproducible files.
    for mode in range(basis.shape[1]):
        pivot = np.argmax(np.abs(basis[:, mode]))
        if basis[pivot, mode] < 0.0:
            basis[:, mode] *= -1.0

    retained_basis = basis[:, :rank]
    energy_fraction = np.cumsum(eigenvalues) / total_energy
    return retained_basis, singular_values, energy_fraction


def _compression_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.compression == "none":
        return {}
    if args.compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {
        "compression": "gzip",
        "compression_opts": args.compression_level,
        "shuffle": True,
    }


def write_reduced_file(
    temporary_path: Path,
    args: argparse.Namespace,
    sources: list[SourceFile],
    x: np.ndarray,
    time: np.ndarray,
    basis: np.ndarray,
    singular_values: np.ndarray,
    energy_fraction: np.ndarray,
) -> None:
    """Project complete experiments and write the self-describing HDF5 file."""
    rank = basis.shape[1]
    n_experiments = len(sources)
    n_time = len(time)
    total_snapshots = n_experiments * n_time
    storage_dtype = np.dtype(args.dtype)
    chunk_time = min(n_time, args.io_block_size)

    with h5py.File(temporary_path, "w", track_order=True) as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["description"] = "Uncentered POD reduced dynamics"
        output.attrs["power"] = args.power
        output.attrs["rank"] = rank
        output.attrs["n_experiments"] = n_experiments
        output.attrs["n_space"] = len(x)
        output.attrs["n_time"] = n_time
        output.attrs["n_snapshots"] = total_snapshots
        output.attrs["centered"] = False
        output.attrs["axis_order"] = "mode, experiment, time"
        output.attrs["coordinate_dtype"] = args.dtype
        output.attrs["source_data_dir"] = str(args.data_dir.resolve())
        output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        output.attrs["generator"] = Path(__file__).name

        grid = output.create_group("grid")
        x_dataset = grid.create_dataset("x", data=x, track_times=False)
        x_dataset.attrs["axis"] = "space"
        time_dataset = grid.create_dataset("time", data=time, track_times=False)
        time_dataset.attrs["axis"] = "experiment time"

        pod = output.create_group("pod")
        basis_dataset = pod.create_dataset(
            "basis", data=basis, track_times=False
        )
        basis_dataset.attrs["axis_order"] = "space, mode"
        singular_dataset = pod.create_dataset(
            "singular_values", data=singular_values, track_times=False
        )
        singular_dataset.attrs["description"] = "Full spatial POD spectrum"
        energy_dataset = pod.create_dataset(
            "cumulative_energy_fraction", data=energy_fraction, track_times=False
        )
        energy_dataset.attrs["description"] = (
            "Cumulative squared-singular-value fraction"
        )
        pod.attrs["retained_energy_fraction"] = float(energy_fraction[rank - 1])
        pod.attrs["method"] = "eigendecomposition of incrementally accumulated X X.T"
        pod.attrs["mean_centered"] = False

        reduced = output.create_group("reduced")
        state = reduced.create_dataset(
            "state",
            shape=(rank, n_experiments, n_time),
            dtype=storage_dtype,
            chunks=(rank, 1, chunk_time),
            track_times=False,
            **_compression_kwargs(args),
        )
        state.attrs["axis_order"] = "mode, experiment, time"
        state.attrs["definition"] = "basis.T @ full_state"

        string_dtype = h5py.string_dtype(encoding="utf-8")
        experiment_group = output.create_group("experiments")
        experiment_group.create_dataset(
            "label",
            data=np.asarray([source.label for source in sources], dtype=object),
            dtype=string_dtype,
            track_times=False,
        )
        experiment_group.create_dataset(
            "source_file",
            data=np.asarray([source.path.name for source in sources], dtype=object),
            dtype=string_dtype,
            track_times=False,
        )

        for experiment, source in enumerate(sources):
            with h5py.File(source.path, "r") as handle:
                source_state = handle["Q_1D"]
                for start in range(0, n_time, args.io_block_size):
                    stop = min(start + args.io_block_size, n_time)
                    snapshot_block = np.asarray(
                        source_state[:, start:stop], dtype=np.float64
                    )
                    state[:, experiment, start:stop] = (
                        basis.T @ snapshot_block
                    ).astype(storage_dtype, copy=False)
            print(
                f"Projection pass: experiment {experiment + 1}/{n_experiments} "
                f"({source.label})",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.rank < 1:
        raise ValueError("--rank must be positive.")
    if args.io_block_size < 1:
        raise ValueError("--io-block-size must be positive.")

    output_path = args.output or (
        Path("reduced_data")
        / args.power
        / f"reduced_dynamics_{args.power}_r{args.rank}.h5"
    )
    output_path = output_path.expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources, x, time, n_space, n_time = inspect_sources(
        args.data_dir.expanduser(), args.power
    )
    args.data_dir = args.data_dir.expanduser()
    n_experiments = len(sources)
    total_snapshots = n_experiments * n_time
    if args.rank > min(n_space, total_snapshots):
        raise ValueError(
            f"--rank={args.rank} exceeds the maximum possible rank "
            f"{min(n_space, total_snapshots)}."
        )

    print(
        f"Power {args.power}: {n_experiments} complete experiments, "
        f"{n_time:,} time samples each, {total_snapshots:,} snapshots, "
        f"rank {args.rank}. No segmentation will be applied.",
        flush=True,
    )
    basis, singular_values, energy_fraction = compute_pod_basis(
        sources, args.io_block_size, args.rank, n_space
    )

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        write_reduced_file(
            temporary_path,
            args,
            sources,
            x,
            time,
            basis,
            singular_values,
            energy_fraction,
        )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    size_mib = output_path.stat().st_size / (1024**2)
    print(
        f"Saved {output_path.resolve()} ({size_mib:.1f} MiB). "
        f"Rank-{args.rank} retained energy: "
        f"{energy_fraction[args.rank - 1]:.8%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
