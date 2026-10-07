"""Plot Q(k dt) for one normalized shared-POD trajectory.

    python -m data_analysis.correlations.increment_covariance 0p30 --rep 1 --rank 50

POD rank r gives r+1 model coordinates: sqrt(N) times the spatial mean, then
the first r shared POD coefficients. By default, the same training-only state
normalization as the neural SDE is fitted directly from the prepared data.
An optional checkpoint can supply its saved normalization instead.
All valid increments are used; increments are not centered or drift-corrected.
Use --unscaled for M(h) = h Q(h), the raw increment second moment.
Use --trim-percentiles 1 99 to keep increments whose normalized vector norm
lies within those percentiles, computed separately at each lag.
With trimming, the terminal reports the excluded share of total squared-L2
increment energy at each lag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from modelling.acdm.conditional_edm.checkpoint import load_checkpoint
from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData


def estimate_increment_covariance(z, dt, min_lag=1, max_lag=10, batch_size=4096,
                                  trim_percentiles=None, return_details=False, lags=None):
    """Return uncentered Q matrices; optional trimming uses vector norms per lag."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 1 or not np.isfinite(z).all():
        raise ValueError("z must be a finite time-by-coordinate array.")
    if not np.isfinite(dt) or dt <= 0 or batch_size < 1:
        raise ValueError("Require positive dt and batch size.")
    if lags is None:
        if not 1 <= min_lag <= max_lag < len(z):
            raise ValueError("Require 1 <= min_lag <= max_lag < sample count.")
        lags = np.arange(min_lag, max_lag + 1)
    else:
        lags = np.asarray(lags)
        if (lags.ndim != 1 or len(lags) == 0 or not np.issubdtype(lags.dtype, np.integer)
                or np.any(lags < 1) or np.any(lags >= len(z))):
            raise ValueError("Lags must be positive integer steps smaller than the sample count.")
        lags = np.array(sorted(set(lags.tolist())), dtype=int)
    if trim_percentiles is not None:
        if (len(trim_percentiles) != 2 or not np.isfinite(trim_percentiles).all()
                or not 0 <= trim_percentiles[0] < trim_percentiles[1] <= 100):
            raise ValueError("Trim percentiles must satisfy 0 <= lower < upper <= 100.")
    q = np.empty((len(lags), z.shape[1], z.shape[1]), dtype=np.float64)
    retained_counts = np.empty(len(lags), dtype=np.int64)
    norm_thresholds = np.full((len(lags), 2), np.nan)
    excluded_energy_percent = np.zeros(len(lags), dtype=np.float64)
    for index, k in enumerate(lags):
        gram = np.zeros((z.shape[1], z.shape[1]), dtype=np.float64)
        count = len(z) - k
        norms = None
        if trim_percentiles is not None:
            norms = np.empty(count, dtype=np.float64)
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                delta = z[start + k:stop + k] - z[start:stop]
                norms[start:stop] = np.linalg.norm(delta, axis=1)
            norm_thresholds[index] = np.percentile(norms, trim_percentiles)
            lower, upper = norm_thresholds[index]
            excluded = norms[(norms < lower) | (norms > upper)]
            total_energy = norms @ norms
            excluded_energy_percent[index] = (100 * (excluded @ excluded) / total_energy
                                              if total_energy > 0 else np.nan)
        retained = 0
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            delta = z[start + k:stop + k] - z[start:stop]
            if norms is not None:
                lower, upper = norm_thresholds[index]
                delta = delta[(norms[start:stop] >= lower) & (norms[start:stop] <= upper)]
            retained += len(delta)
            gram += delta.T @ delta
        if retained == 0:
            raise ValueError(f"Percentile trimming retained no increments at lag {k}.")
        retained_counts[index] = retained
        q[index] = gram / (retained * k * dt)
    h = lags * dt
    if return_details:
        return h, q, retained_counts, norm_thresholds, excluded_energy_percent
    return h, q


def load_normalized_trajectory(power, rep, rank, checkpoint_path=None, *, data_root=None,
                               shared_basis=None, validation_reps=None, test_reps=None,
                               batch_size=4096, prepared_data=None):
    """Read one repetition in training-normalized shared-POD coordinates."""
    if not power or Path(power).name != power or power in (".", ".."):
        raise ValueError("Forcing must be one folder name, such as 0p30.")
    if rep < 1 or rank < 1 or batch_size < 1:
        raise ValueError("Repetition, rank, and batch size must be positive.")
    if prepared_data is not None:
        if (checkpoint_path is not None or data_root is not None or shared_basis is not None
                or validation_reps is not None or test_reps is not None
                or prepared_data.experiment != power or prepared_data.rank != rank
                or prepared_data.mean is None):
            raise ValueError("Expected fitted shared-POD data for the requested experiment and rank.")
        data = prepared_data
        mean = np.asarray(data.mean, dtype=np.float32).astype(np.float64)
        std = np.asarray(data.scale, dtype=np.float32).astype(np.float64)
        normalization = "training-split shared-POD state mean/std (float32 model-buffer precision)"
        checkpoint_name = None
    elif checkpoint_path is None:
        options = dict(shared_basis=shared_basis, validation_reps=validation_reps,
                       test_reps=test_reps, batch_size=batch_size)
        if data_root is not None:
            options["data_root"] = data_root
        data = SharedPODTrainingData(power, rank, **options).fit_standardization()
        # SDE fit_normalization uses the same training moments and stores float32 buffers.
        mean = np.asarray(data.mean, dtype=np.float32).astype(np.float64)
        std = np.asarray(data.scale, dtype=np.float32).astype(np.float64)
        normalization = "training-split shared-POD state mean/std (float32 model-buffer precision)"
        checkpoint_name = None
    else:
        if validation_reps is not None or test_reps is not None:
            raise ValueError("Split overrides cannot be combined with --checkpoint.")
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = load_checkpoint(checkpoint_path, "cpu")
        config = checkpoint["model_config"]
        source_config = checkpoint["data_config"]
        if "dt" not in config or "diffusion_model.raw_diagonal" not in checkpoint["model_state"]:
            raise ValueError("Expected a neural SDE checkpoint with saved state normalization.")
        if (source_config["experiment"] != power or int(source_config["rank"]) != rank
                or not bool(source_config.get("include_spatial_mean", True))
                or int(config["reduced_dim"]) != rank + 1):
            raise ValueError("Checkpoint forcing, POD rank, or spatial-mean convention differs from request.")
        data = checkpoint_data(checkpoint, data_root=data_root, shared_basis=shared_basis)
        if not np.isclose(float(config["dt"]), data.native_dt, rtol=1e-4, atol=0):
            raise ValueError("Checkpoint timestep differs from source timestamps.")
        state = checkpoint["model_state"]
        mean = np.asarray(state["state_mean"], dtype=np.float64)
        std = np.asarray(state["state_std"], dtype=np.float64)
        normalization = "(physical coordinate - checkpoint state_mean) / checkpoint state_std"
        checkpoint_name = str(checkpoint_path)
    names = [name for name in data.frame_counts if name.endswith(f"_rep{rep}")]
    if len(names) != 1:
        raise ValueError(f"Expected one repetition {rep} in shared-POD data; found {len(names)}.")
    name = names[0]
    if (mean.shape != (rank + 1,) or std.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("Invalid training state normalization.")
    z = np.empty((data.frame_counts[name], rank + 1), dtype=np.float64)
    if len(z) < 2:
        raise ValueError("Need at least two samples for increment covariance.")
    for start in range(0, len(z), batch_size):
        stop = min(start + batch_size, len(z))
        _, physical = data.read_coordinates(name, start, stop)
        z[start:stop] = (physical - mean) / std
    if not np.isfinite(z).all():
        raise ValueError("Normalized trajectory contains nonfinite values.")
    metadata = dict(forcing=power, repetition=rep, repetition_name=name,
                    pod_rank=rank, coordinate_count=rank + 1,
                    coordinate_order="sqrt(N)*spatial_mean, shared POD 1..rank",
                    normalization=normalization, checkpoint=checkpoint_name,
                    normalization_train_repetitions=list(data.train_repetitions),
                    normalization_state_mean=mean.tolist(), normalization_state_std=std.tolist(),
                    shared_basis=str(data.basis_path),
                    source=str(data.source_paths[name]), physical_coordinate_units=data.signal_units,
                    normalized_coordinate_units="dimensionless", n_samples=len(z),
                    dt_seconds=data.native_dt)
    return z, data.native_dt, metadata


def save_diagnostic(h, q, metadata, output, *, unscaled=False, overwrite=False,
                    retained_counts=None, norm_thresholds=None, trim_percentiles=None,
                    excluded_energy_percent=None):
    output = Path(output)
    if trim_percentiles is not None and (retained_counts is None or norm_thresholds is None
                                        or excluded_energy_percent is None):
        raise ValueError("Trimmed diagnostics require counts, norm thresholds, and excluded energy percentages.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Nonempty output directory {output}; use --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    lags = np.rint(h / metadata["dt_seconds"]).astype(int)
    if retained_counts is None:
        retained_counts = metadata["n_samples"] - lags
    values = q * h[:, None, None] if unscaled else q
    name = "second_moment" if unscaled else "Q"
    units = "normalized²" if unscaled else "normalized²/s"
    trace = np.trace(values, axis1=1, axis2=2)
    diagonal = np.diagonal(values, axis1=1, axis2=2).copy()
    arrays = dict(k=lags, h_seconds=h, **{name: values}, trace=trace, diagonal=diagonal,
                  increment_counts=metadata["n_samples"] - lags,
                  retained_increment_counts=retained_counts)
    if trim_percentiles is not None:
        arrays["norm_thresholds"] = norm_thresholds
    if excluded_energy_percent is not None:
        arrays["excluded_increment_energy_percent"] = excluded_energy_percent
    np.savez(output / f"{name}.npz", **arrays)
    report = dict(metadata, statistic=name,
                  estimator=("uncentered mean of retained outer-product increments"
                             if trim_percentiles is not None else
                             "uncentered mean of all valid outer-product increments")
                  + ("" if unscaled else " divided by k*dt"),
                  statistic_units=units, min_lag=int(lags[0]), max_lag=int(lags[-1]),
                  lags=lags.tolist())
    method = ("upper_tail_normalized_increment_vector_norm_per_lag"
              if trim_percentiles is not None and "upper_cutoff_percent" in metadata else
              "normalized_increment_vector_norm_per_lag")
    report["selection"] = (dict(method=method,
                                lower_percentile=float(trim_percentiles[0]),
                                upper_percentile=float(trim_percentiles[1]),
                                inclusive=True, retained_increment_counts=retained_counts.tolist(),
                                excluded_increment_energy_definition=(
                                    "100 * sum_excluded ||delta z||_2^2 / sum_all ||delta z||_2^2, per lag"),
                                excluded_increment_energy_percent=[float(value) if np.isfinite(value) else None
                                                                   for value in excluded_energy_percent])
                           if trim_percentiles is not None else dict(method="all_valid_increments"))
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(h, trace, "o-", lw=1.3)
    axes[0].set_ylabel(f"tr {'M' if unscaled else 'Q'}(h) [{units}]")
    axes[0].grid(alpha=.25)
    colors = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max(1, diagonal.shape[1] - 1))
    for i in range(diagonal.shape[1]):
        axes[1].plot(h, diagonal[:, i], "o-", ms=2, lw=.8, color=colors(norm(i)), alpha=.7)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colors), ax=axes[1], label="Coordinate index (0 = spatial mean)")
    axes[1].set(xlabel="Lag h [s]", ylabel=f"{'M' if unscaled else 'Q'}ᵢᵢ(h) [{units}]")
    axes[1].grid(alpha=.25)
    fig.savefig(output / ("increment_second_moment.png" if unscaled else "increment_covariance.png"), dpi=160)
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("power", help="Forcing folder, e.g. 0p30.")
    parser.add_argument("--rep", type=int, required=True, help="Repetition index.")
    parser.add_argument("--rank", type=int, required=True, help="Shared POD rank.")
    parser.add_argument("--min-lag", type=int, default=1,
                        help="First lag k to include (default 1).")
    parser.add_argument("--max-lag", type=int, default=10,
                        help="Last lag k to include (default 10).")
    parser.add_argument("--unscaled", action="store_true",
                        help="Plot and save the unscaled increment second moment M(h)=h Q(h).")
    parser.add_argument("--trim-percentiles", type=float, nargs=2, metavar=("LOW", "HIGH"),
                        help="Keep increments within these per-lag vector-norm percentiles, e.g. 1 99.")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional neural SDE checkpoint for its saved state normalization.")
    parser.add_argument("--data-root", type=Path, help="Optional relocated reduced-data root.")
    parser.add_argument("--shared-basis", type=Path, help="Optional relocated shared basis.")
    parser.add_argument("--validation-reps", type=int, nargs="+",
                        help="Validation repetitions excluded when fitting normalization without a checkpoint.")
    parser.add_argument("--test-reps", type=int, nargs="*",
                        help="Test repetitions excluded when fitting normalization without a checkpoint.")
    parser.add_argument("--output", type=Path, help="Output directory; default runs/increment_covariance/...")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.min_lag <= args.max_lag:
        parser.error("Require 1 <= --min-lag <= --max-lag.")
    if args.trim_percentiles is not None and not (0 <= args.trim_percentiles[0] < args.trim_percentiles[1] <= 100):
        parser.error("Require 0 <= LOW < HIGH <= 100 for --trim-percentiles.")
    suffix = f"{args.power}_rep{args.rep}_r{args.rank}_normalized"
    if args.min_lag != 1:
        suffix += f"_k{args.min_lag}-{args.max_lag}"
    elif args.max_lag != 10:
        suffix += f"_k{args.max_lag}"
    if args.unscaled:
        suffix += "_unscaled"
    if args.trim_percentiles is not None:
        suffix += f"_trim{args.trim_percentiles[0]:g}-{args.trim_percentiles[1]:g}"
    output = args.output or Path("runs/increment_covariance") / suffix
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"Nonempty output directory {output}; use --overwrite.")
    z, dt, metadata = load_normalized_trajectory(
        args.power, args.rep, args.rank, args.checkpoint,
        data_root=args.data_root, shared_basis=args.shared_basis,
        validation_reps=args.validation_reps, test_reps=args.test_reps)
    h, q, retained, thresholds, excluded_energy_percent = estimate_increment_covariance(
        z, dt, min_lag=args.min_lag, max_lag=args.max_lag,
        trim_percentiles=args.trim_percentiles, return_details=True)
    print(save_diagnostic(h, q, metadata, output, unscaled=args.unscaled,
                          overwrite=args.overwrite, retained_counts=retained,
                          norm_thresholds=thresholds, trim_percentiles=args.trim_percentiles,
                          excluded_energy_percent=excluded_energy_percent))
    if args.trim_percentiles is not None:
        print("Lag k | increments cut | cut % | excluded L2 energy %")
        for k, kept, energy_percent in zip(range(args.min_lag, args.max_lag + 1),
                                           retained, excluded_energy_percent):
            total = len(z) - k
            cut = total - kept
            energy_text = f"{energy_percent:.2f}" if np.isfinite(energy_percent) else "n/a"
            print(f"{k:5d} | {cut:14d} | {100 * cut / total:5.2f} | {energy_text}")


if __name__ == "__main__":
    main()
