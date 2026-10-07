"""Compute a shared spatial POD basis for all repetitions of one experiment.

For X_r = U_r S_r V_r.T (space x time), decompose
B = [U_1 S_1, ..., U_R S_R]. Since B B.T = sum_r X_r X_r.T,
its left singular vectors are the pooled spatial POD modes. With truncated
input PODs this identity applies to their retained reconstructions only.
Each recording keeps its original preprocessing: removed repetition means
are NOT restored or replaced with a global mean. Longer recordings naturally
contribute more snapshot energy; there is no per-repetition normalization.

Examples (in the capillarywave environment)::

    python shared_pod_basis.py 0p20 --dry-run
    python shared_pod_basis.py 0p20 --rank 100 --solver randomized
    python shared_pod_basis.py 0p20 --input-rank 100 --rank 50

The default solver is an exact economy SVD. The optional randomized solver
approximates the leading modes with much less SVD work. Both assemble B in
float64 RAM; --dry-run reports its size and Linux memory availability.
Exact SVD computes ALL min(B.shape) modes before truncation, regardless of
--rank. Its estimated main allocations (including LAPACK workspace) are
reported separately. For large inputs use --solver randomized, which works
with a subspace of size --rank + --oversampling.
Output contains pod/modes, pod/singular_values, grid/x, grid/y, energy
fractions, and per-repetition projection matrices. For saved coefficients
A_r with shape (time, local_mode), shared coefficients of the retained
reconstruction are A_r[:, :input_rank] @ repetitions/<name>/projection.T.
No temporal coefficients or single shared temporal mean are fabricated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
from scipy.linalg import qr, svd
from scipy.linalg.lapack import get_lapack_funcs


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "reduced_data"
PREPROCESSING_FLAGS = (
    "instantaneous_spatial_mean_removed", "temporal_mean_field_removed",
)


def report_solver_memory(n_space: int, n_columns: int, solver: str) -> None:
    """Report major exact-SVD allocations, not a guaranteed process peak."""
    gib = 2**30
    if solver == "exact":
        k = min(n_space, n_columns)
        query = get_lapack_funcs("gesdd_lwork", dtype=np.float64)
        lwork, info = query(n_space, n_columns, compute_uv=1, full_matrices=0)
        if info == 0 and np.isfinite(lwork) and lwork > 0:
            # Original matrix, LAPACK input copy, economy U and Vh, workspace.
            elements = 2 * n_space * n_columns + (n_space + n_columns) * k + lwork
            estimate = elements * 8 / gib
            print(f"Exact SVD computes all {k} modes before truncation. "
                  f"Estimated main allocations: {estimate:.2f} GiB "
                  "(input, copy, U, Vh, LAPACK workspace; other overhead additional).",
                  flush=True)
        print("For lower memory use, select --solver randomized; "
              "reducing --rank alone does not shrink the exact SVD.", flush=True)
    else:
        print("Randomized SVD uses rank + oversampling columns; "
              "the full input matrix and smaller solver arrays still need RAM.", flush=True)
    try:
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemTotal", "MemAvailable"):
                memory[name] = int(value.split()[0]) * 1024 / gib
        if "MemTotal" in memory and "MemAvailable" in memory:
            print(f"Linux RAM: {memory['MemTotal']:.2f} GiB total, "
                  f"{memory['MemAvailable']:.2f} GiB currently available "
                  "(may be limited by WSL/container settings).", flush=True)
    except OSError:
        pass  # /proc is not present on all supported platforms.


def discover_repetitions(directory: Path, pod_filename: str) -> list[Path]:
    """Require one selected POD file in every repetition directory."""
    if Path(pod_filename).name != pod_filename:
        raise ValueError("--pod-filename must be a filename, not a path.")
    directories = sorted(
        (p for p in directory.iterdir()
         if p.is_dir() and re.fullmatch(r"Ca_ac.*_rep\d+", p.name)),
        key=lambda p: (int(p.name.rsplit("_rep", 1)[1]), p.name),
    )
    if not directories:
        raise ValueError(f"No Ca_ac*_repN directories found in {directory}.")
    # Do not silently pool different acquisition conditions within a power.
    if len({p.name.rsplit("_rep", 1)[0] for p in directories}) != 1:
        raise ValueError(f"Multiple Ca_ac conditions found in {directory}.")
    paths = [p / pod_filename for p in directories]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing repetition POD files:\n" + "\n".join(missing))
    return paths


def inspect_inputs(paths: list[Path], input_rank: int | None):
    """Check matching grids, physical units, and preprocessing before pooling."""
    if not paths or (input_rank is not None and input_rank < 1):
        raise ValueError("Inputs must be nonempty and input rank positive.")
    records = []
    reference = None
    for path in paths:
        with h5py.File(path, "r") as f:
            modes = f["pod/modes"]
            singular = np.asarray(f["pod/singular_values"], dtype=np.float64)
            x, y = np.asarray(f["grid/x"]), np.asarray(f["grid/y"])
            if (x.ndim != 1 or y.ndim != 1 or not np.isfinite(x).all()
                    or not np.isfinite(y).all() or modes.ndim != 3
                    or modes.shape[1:] != (len(y), len(x))
                    or singular.shape != (len(modes),)):
                raise ValueError(f"Invalid mode, singular-value, or grid shape in {path}.")
            if (not np.isfinite(singular).all() or np.any(singular < 0)
                    or np.any(np.diff(singular) > 0)):
                raise ValueError(f"Invalid or unordered singular values in {path}.")
            rank = len(modes) if input_rank is None else input_rank
            if not 1 <= rank <= len(modes):
                raise ValueError(f"Requested {rank} input modes; {path} has {len(modes)}.")
            signature = {
                **{name: bool(f.attrs[name]) for name in PREPROCESSING_FLAGS},
                "x_units": str(f["grid/x"].attrs["units"]),
                "y_units": str(f["grid/y"].attrs["units"]),
                "z_units": str(f["reduced/coefficients"].attrs["units"]),
                "spatial_inner_product": str(f["pod"].attrs["spatial_inner_product"]),
            }
            if not signature["spatial_inner_product"].startswith("unweighted Euclidean"):
                raise ValueError(f"Unsupported spatial inner product in {path}.")
            if reference is None:
                reference = dict(signature, x=x, y=y, frame_shape=modes.shape[1:])
            elif (any(reference[k] != v for k, v in signature.items())
                  or not np.array_equal(reference["x"], x)
                  or not np.array_equal(reference["y"], y)):
                raise ValueError(f"Grid, units, or preprocessing mismatch in {path}.")
            n_frames = int(f.attrs["n_frames"])
            if f["reduced/coefficients"].shape != (n_frames, len(modes)) or n_frames < 1:
                raise ValueError(f"Invalid coefficient dimensions in {path}.")
            total = float(f["pod"].attrs["total_preprocessed_snapshot_energy"])
            if not np.isfinite(total) or total <= 0:
                raise ValueError(f"Invalid total snapshot energy in {path}.")
            records.append(dict(path=path, rank=rank, n_frames=n_frames,
                                singular_values=singular[:rank], total_energy=total))
    return records, reference


def assemble_weighted_modes(records, n_space: int) -> np.ndarray:
    matrix = np.empty((n_space, sum(r["rank"] for r in records)),
                      dtype=np.float64, order="F")
    offset = 0
    for record in records:
        print(f"Loading {record['path'].parent.name}: {record['rank']} modes", flush=True)
        with h5py.File(record["path"], "r") as f:
            block = np.asarray(f["pod/modes"][:record["rank"]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite modes in {record['path']}.")
        stop = offset + record["rank"]
        matrix[:, offset:stop] = block.reshape(record["rank"], n_space).T
        matrix[:, offset:stop] *= record["singular_values"]
        offset = stop
    return matrix


def compute_shared_basis(matrix, rank, solver="exact", oversampling=20,
                         power_iterations=2, seed=12345):
    """Return leading left singular vectors and values of weighted modes."""
    if matrix.ndim != 2 or not 1 <= rank <= min(matrix.shape):
        raise ValueError("Shared rank must be between 1 and min(matrix.shape).")
    if oversampling < 0 or power_iterations < 0:
        raise ValueError("Oversampling and power iterations must be nonnegative.")
    # Check in blocks to avoid a matrix-sized temporary boolean allocation.
    finite = True
    nonzero = False
    for start in range(0, matrix.shape[1], 256):
        block = matrix[:, start:start + 256]
        finite = finite and np.isfinite(block).all()
        nonzero = nonzero or np.any(block)
    if not finite or not nonzero:
        raise ValueError("Weighted modes must be finite and have positive energy.")
    if solver == "exact":
        u, s, _ = svd(matrix, full_matrices=False, check_finite=False)
    elif solver == "randomized":
        width = min(rank + oversampling, min(matrix.shape))
        rng = np.random.default_rng(seed)
        q, _ = qr(matrix @ rng.standard_normal((matrix.shape[1], width)),
                  mode="economic", check_finite=False)
        for _ in range(power_iterations):
            z, _ = qr(matrix.T @ q, mode="economic", check_finite=False)
            q, _ = qr(matrix @ z, mode="economic", check_finite=False)
        small_u, s, _ = svd(q.T @ matrix, full_matrices=False, check_finite=False)
        u = q @ small_u[:, :rank]
    else:
        raise ValueError(f"Unknown solver: {solver}")
    u, s = u[:, :rank].copy(), s[:rank].copy()
    # A deterministic sign makes plots and stored projection matrices consistent.
    signs = np.where(u[np.argmax(np.abs(u), axis=0), np.arange(rank)] < 0, -1, 1)
    u *= signs
    return u, s


def write_output(path, records, reference, modes, singular, pooled_energy, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with h5py.File(temporary_path, "w") as f:
            f.attrs.update(schema_version="shared_pod_1.0", generator=Path(__file__).name,
                           created_utc=datetime.now(timezone.utc).isoformat(),
                           experiment=args.experiment, rank=modes.shape[1],
                           frame_shape=reference["frame_shape"], n_space=modes.shape[0],
                           n_repetitions=len(records), solver=args.solver,
                           oversampling=args.oversampling, power_iterations=args.power_iterations,
                           random_seed=args.seed, weighting="singular values; pooled snapshots",
                           description="Shared POD of retained repetition reconstructions")
            for name in PREPROCESSING_FLAGS:
                f.attrs[name] = reference[name]
            for axis in ("x", "y"):
                d = f.create_dataset(f"grid/{axis}", data=reference[axis])
                d.attrs["units"] = reference[f"{axis}_units"]
            d = f.create_dataset("pod/modes", data=modes.T.reshape(
                len(singular), *reference["frame_shape"]), compression="lzf")
            d.attrs["axis_order"] = "mode, y, x"
            f.create_dataset("pod/singular_values", data=singular)
            total_energy = sum(r["total_energy"] for r in records)
            cumulative = np.cumsum(singular**2)
            d = f.create_dataset("pod/cumulative_energy_fraction", data=cumulative / total_energy)
            d.attrs["description"] = "Fraction of total original preprocessed snapshot energy"
            d = f.create_dataset("pod/cumulative_retained_input_energy_fraction",
                                 data=cumulative / pooled_energy)
            d.attrs["description"] = "Fraction of energy in concatenated weighted input modes"
            f["pod"].attrs.update(
                spatial_inner_product=reference["spatial_inner_product"],
                total_preprocessed_snapshot_energy=total_energy,
                pooled_retained_input_energy=pooled_energy,
                retained_energy_fraction=cumulative[-1] / total_energy,
                max_orthonormality_error=np.max(np.abs(modes.T @ modes - np.eye(len(singular)))),
            )
            for record in records:
                group = f.create_group(f"repetitions/{record['path'].parent.name}")
                group.attrs.update(source_pod_file=str(record["path"].resolve()),
                                   input_rank=record["rank"], n_frames=record["n_frames"],
                                   signal_units=reference["z_units"])
                group.create_dataset("input_singular_values", data=record["singular_values"])
                with h5py.File(record["path"], "r") as source:
                    local = np.asarray(source["pod/modes"][:record["rank"]], dtype=np.float64)
                projection = modes.T @ local.reshape(record["rank"], -1).T
                d = group.create_dataset("projection", data=projection)
                d.attrs["axis_order"] = "shared_mode, local_mode"
                d.attrs["definition"] = "shared_coefficients = local_coefficients @ projection.T"
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", help="Power folder, e.g. 0p20.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pod-filename", default="pod_2d_r1000.h5")
    parser.add_argument("--input-rank", type=int, help="Leading modes per repetition (default: all saved).")
    parser.add_argument("--rank", type=int, default=100, help="Shared output rank (default: 100).")
    parser.add_argument("--solver", choices=("exact", "randomized"), default="exact")
    parser.add_argument("--oversampling", type=int, default=20)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report matrix size only.")
    args = parser.parse_args(argv)
    if args.rank < 1 or args.oversampling < 0 or args.power_iterations < 0:
        parser.error("Rank must be positive; oversampling and power iterations nonnegative.")
    if Path(args.experiment).name != args.experiment or args.experiment in (".", ".."):
        parser.error("Experiment must be a single folder name.")
    directory = args.data_root.expanduser() / args.experiment
    paths = discover_repetitions(directory, args.pod_filename)
    records, reference = inspect_inputs(paths, args.input_rank)
    n_space = int(np.prod(reference["frame_shape"]))
    n_columns = sum(r["rank"] for r in records)
    if args.rank > min(n_space, n_columns):
        parser.error(f"Shared rank exceeds matrix dimensions {(n_space, n_columns)}.")
    output = (args.output or directory / "shared_pod" / f"shared_pod_r{args.rank}.h5").expanduser()
    if output.resolve() in {p.resolve() for p in paths}:
        raise ValueError("Output must not replace an input POD file.")
    print(f"{len(records)} repetitions; weighted matrix {n_space} x {n_columns}; "
          f"{n_space * n_columns * 8 / 2**30:.2f} GiB plus solver workspace.", flush=True)
    print(f"Solver: {args.solver}; shared rank: {args.rank}; output: {output}", flush=True)
    report_solver_memory(n_space, n_columns, args.solver)
    if args.dry_run:
        return
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
    matrix = assemble_weighted_modes(records, n_space)
    pooled_energy = float(np.einsum("ij,ij->", matrix, matrix))
    print("Computing shared SVD...", flush=True)
    modes, singular = compute_shared_basis(matrix, args.rank, args.solver,
                                          args.oversampling, args.power_iterations, args.seed)
    del matrix
    write_output(output, records, reference, modes, singular, pooled_energy, args)
    print(f"Saved {output}; retained {np.sum(singular**2) / pooled_energy:.4%} "
          "of pooled input-mode energy.", flush=True)


if __name__ == "__main__":
    main()
