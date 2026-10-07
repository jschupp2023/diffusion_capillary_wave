"""Plot training-normalized shared-POD states and optional increments.

    python -m data_analysis.plot_normalized_coordinates 0p15 \
        --rep 9 --rank 20 --timesteps 10000 --modes 0 1 2 3 4 10 20 \
        --increment-lag 3

Coordinate 0 is sqrt(number of spatial points) times the frame spatial mean;
coordinates 1..rank are the shared POD coefficients. No model is required.
An increment is the difference of two normalized coordinates; it is not
standardized again by the increment standard deviation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT


def _save_plot(path, elapsed, values, modes, title, *, increments=False, start_step=0):
    figure, axes = plt.subplots(len(modes), 1, figsize=(11, max(3, 2 * len(modes))),
                                sharex=True, squeeze=False, constrained_layout=True)
    for axis, mode, series in zip(axes[:, 0], modes, values.T):
        axis.plot(elapsed, series, lw=.8)
        axis.axhline(0, color="0.5", lw=.6)
        label = "√N × mean (0)" if mode == 0 else f"POD {mode}"
        axis.set_ylabel(f"Δ {label}" if increments else label)
        axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel(
        f"Time since frame {start_step} [s]"
        + (" (increment end)" if increments else "")
    )
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def increment_mahalanobis(normalized, lag, percentile=95.0, eigenvalue_rcond=1e-10):
    """Score all lagged increments using their full-trajectory mean and covariance."""
    normalized = np.asarray(normalized, dtype=np.float64)
    if (normalized.ndim != 2 or len(normalized) <= lag or lag < 1
            or not np.isfinite(normalized).all()):
        raise ValueError("Need a finite state trajectory longer than the positive increment lag.")
    if not np.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError("Mahalanobis percentile must lie strictly between 0 and 100.")
    increments = normalized[lag:] - normalized[:-lag]
    if len(increments) < 2:
        raise ValueError("Need at least two full-trajectory increments for Mahalanobis distance.")
    increment_mean = increments.mean(axis=0)
    centered = increments - increment_mean
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(eigenvalues[-1])
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Full-trajectory increment covariance has no positive variance.")
    tolerance = eigenvalue_rcond * maximum
    retained = eigenvalues > tolerance
    projected = centered @ eigenvectors[:, retained]
    squared_distance = np.sum(projected**2 / eigenvalues[retained], axis=1)
    distance = np.sqrt(np.maximum(squared_distance, 0))
    threshold = float(np.percentile(distance, percentile))
    return dict(distance=distance, squared_distance=squared_distance,
                increment_mean=increment_mean, covariance=covariance,
                covariance_eigenvalues=eigenvalues,
                covariance_eigenvalue_tolerance=tolerance,
                effective_covariance_rank=int(np.count_nonzero(retained)),
                threshold=threshold, percentile=float(percentile))


def _save_increment_score_plot(path, elapsed, score, threshold, percentile, title,
                               start_step, label):
    figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    axis.plot(elapsed, score, lw=.8, color="C0", label=label)
    exceed = score > threshold
    axis.scatter(elapsed[exceed], score[exceed], s=12, color="C3", zorder=3,
                 label=f"Above full-trajectory {percentile:g}th percentile")
    axis.axhline(threshold, color="C3", ls="--", lw=1,
                 label=f"{percentile:g}th percentile = {threshold:.3g}")
    axis.set_xlabel(f"Time since frame {start_step} [s] (increment end)")
    axis.set_ylabel(label)
    axis.set_title(title)
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_normalized_coordinates(experiment, rep, rank, timesteps, modes=None, *,
                                increment_lag=None, start=0,
                                increment_score="mahalanobis", increment_score_percentile=95.0,
                                data_root=DEFAULT_DATA_ROOT, shared_basis=None,
                                validation_reps=None, test_reps=None,
                                output=None, overwrite=False):
    """Plot normalized states and optional lagged differences of those states."""
    if rep < 1 or timesteps < 1 or start < 0:
        raise ValueError("--rep and --timesteps must be positive; --start must be nonnegative.")
    if increment_lag is not None and not 1 <= increment_lag <= timesteps:
        raise ValueError("--increment-lag must be between 1 and --timesteps.")
    if increment_score not in ("mahalanobis", "l2"):
        raise ValueError("--increment-score must be 'mahalanobis' or 'l2'.")
    if not np.isfinite(increment_score_percentile) or not 0 < increment_score_percentile < 100:
        raise ValueError("--increment-score-percentile must lie strictly between 0 and 100.")
    data = SharedPODTrainingData(experiment, rank, data_root=data_root,
                                 shared_basis=shared_basis, validation_reps=validation_reps,
                                 test_reps=test_reps).fit_standardization()
    chosen = ([mode for mode in (0, 1, 2, 3, 4, 10, 20) if mode <= rank]
              if modes is None else list(dict.fromkeys(modes)))
    if not chosen or any(mode < 0 or mode > rank for mode in chosen):
        raise ValueError(f"--modes must be coordinate indices from 0 to {rank}.")
    names = [name for name in data.frame_counts if name.endswith(f"_rep{rep}")]
    if len(names) != 1:
        raise ValueError(f"Expected one repetition {rep}; found {len(names)}.")
    name = names[0]
    last_step = start + timesteps
    if last_step >= data.frame_counts[name]:
        raise ValueError(
            f"Repetition {rep} has {data.frame_counts[name]} frames; with --start {start}, "
            f"max --timesteps is {data.frame_counts[name] - 1 - start}."
        )
    time, physical = data.read_coordinates(name, start, last_step + 1)
    mean = np.asarray(data.mean, dtype=np.float32)
    std = np.asarray(data.scale, dtype=np.float32)
    # Training batches become float32 before the neural SDE standardizes them.
    normalized = (physical.astype(np.float32) - mean) / std
    output = (Path(output) if output is not None else
              Path(data.source_config["data_root"]) / "coordinate_trajectories" / experiment
              / (f"r{rank}_rep{rep}_t{timesteps}" if start == 0 else
                 f"r{rank}_rep{rep}_s{start}_t{timesteps}"))
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Nonempty output directory {output}; use --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    elapsed = time - time[0]
    selected = normalized[:, chosen]
    _save_plot(output / "normalized_coordinates.png", elapsed, selected, chosen,
               f"{experiment} rep {rep}: normalized shared-POD state, frames {start}–{last_step}",
               start_step=start)
    np.savez_compressed(output / "normalized_coordinates.npz",
                        step=np.arange(start, last_step + 1), time_seconds=elapsed,
                        recording_time_seconds=time,
                        modes=np.asarray(chosen), coordinates=selected)
    if increment_lag is not None:
        end_step = np.arange(start + increment_lag, last_step + 1)
        increments = selected[increment_lag:] - selected[:-increment_lag]
        suffix = f"lag{increment_lag}"
        _save_plot(output / f"normalized_increments_{suffix}.png", elapsed[increment_lag:],
                   increments, chosen,
                   f"{experiment} rep {rep}: normalized shared-POD increments, lag {increment_lag}",
                   increments=True, start_step=start)
        np.savez_compressed(output / f"normalized_increments_{suffix}.npz",
                            start_step=end_step - increment_lag, end_step=end_step,
                            time_seconds=elapsed[increment_lag:],
                            recording_time_seconds=time[increment_lag:],
                            modes=np.asarray(chosen),
                            lag_steps=increment_lag, increments=increments)
        full_normalized = np.empty((data.frame_counts[name], rank + 1), dtype=np.float32)
        for batch_start in range(0, len(full_normalized), 4096):
            batch_stop = min(batch_start + 4096, len(full_normalized))
            _, full_physical = data.read_coordinates(name, batch_start, batch_stop)
            full_normalized[batch_start:batch_stop] = (
                full_physical.astype(np.float32) - mean
            ) / std
        if increment_score == "mahalanobis":
            score_details = increment_mahalanobis(
                full_normalized, increment_lag, increment_score_percentile
            )
            full_score = score_details["distance"]
            score_stem, score_label = "increment_mahalanobis", "Mahalanobis distance"
        else:
            full_increments = full_normalized[increment_lag:] - full_normalized[:-increment_lag]
            full_score = np.linalg.norm(full_increments, axis=1)
            score_details = dict(
                threshold=float(np.percentile(full_score, increment_score_percentile)),
                percentile=float(increment_score_percentile),
            )
            score_stem, score_label = "increment_l2_norm", "Normalized increment L2 norm"
        window_slice = slice(start, last_step + 1 - increment_lag)
        window_score = full_score[window_slice]
        above_threshold = window_score > score_details["threshold"]
        _save_increment_score_plot(
            output / f"{score_stem}_{suffix}.png",
            elapsed[increment_lag:], window_score, score_details["threshold"],
            score_details["percentile"],
            (f"{experiment} rep {rep}: lag-{increment_lag} {score_label.lower()} "
             f"over all {rank + 1} coordinates"), start, score_label,
        )
        score_arrays = dict(
            start_step=end_step - increment_lag, end_step=end_step,
            time_seconds=elapsed[increment_lag:],
            recording_time_seconds=time[increment_lag:],
            score=window_score, squared_score=window_score**2,
            above_threshold=above_threshold,
            threshold=score_details["threshold"],
            threshold_percentile=score_details["percentile"],
            score_kind=increment_score,
            coordinate_indices=np.arange(rank + 1),
            full_trajectory_score=full_score,
            full_trajectory_start_step=np.arange(len(full_score)),
            full_trajectory_end_step=np.arange(increment_lag, len(full_normalized)),
            lag_steps=increment_lag,
        )
        if increment_score == "mahalanobis":
            score_arrays.update(
                distance=window_score, squared_distance=window_score**2,
                full_trajectory_distance=full_score,
                increment_mean=score_details["increment_mean"],
                increment_covariance=score_details["covariance"],
                covariance_eigenvalues=score_details["covariance_eigenvalues"],
                covariance_eigenvalue_tolerance=score_details["covariance_eigenvalue_tolerance"],
                effective_covariance_rank=score_details["effective_covariance_rank"],
            )
        else:
            score_arrays.update(norm=window_score, squared_norm=window_score**2,
                                full_trajectory_norm=full_score)
        np.savez_compressed(output / f"{score_stem}_{suffix}.npz", **score_arrays)
    summary = dict(experiment=experiment, repetition=rep, repetition_name=name, pod_rank=rank,
                   modes=chosen, samples=timesteps + 1, first_step=start, last_step=last_step,
                   dt_seconds=data.native_dt, shared_basis=str(data.basis_path),
                   source=str(data.source_paths[name]),
                   normalization="float32 physical states, then (state - training mean) / training scale",
                   coordinate_zero="sqrt(number of spatial points) * frame spatial mean",
                   normalization_train_repetitions=list(data.train_repetitions),
                   state_mean=mean.tolist(), state_std=std.tolist())
    if increment_lag is not None:
        summary.update(increment_lag_steps=increment_lag,
                       increment_samples=timesteps + 1 - increment_lag,
                       increment_definition="normalized_state[t + lag] - normalized_state[t]",
                       increment_time_label="ending frame",
                       increment_score=increment_score,
                       increment_score_coordinates=f"all model-visible coordinates 0..{rank}",
                       increment_score_reference="all valid lagged increments in the complete repetition",
                       increment_score_threshold_percentile=score_details["percentile"],
                       increment_score_threshold=score_details["threshold"],
                       increment_score_window_exceedance_count=int(np.count_nonzero(above_threshold)),
                       increment_score_window_increment_count=len(window_score))
        if increment_score == "mahalanobis":
            summary.update(
                mahalanobis_covariance="centered sample covariance; eigenvalue pseudoinverse",
                mahalanobis_covariance_eigenvalue_rcond=1e-10,
                mahalanobis_effective_covariance_rank=score_details["effective_covariance_rank"],
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Forcing folder, e.g. 0p15.")
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True,
                        help="Number of native steps after --start; includes both endpoints.")
    parser.add_argument("--start", type=int, default=0,
                        help="First native timestep to plot (default: 0).")
    parser.add_argument("--modes", nargs="+", type=int,
                        help="Coordinate indices, with 0 = spatial mean; default 0 1 2 3 4 10 20 within rank.")
    parser.add_argument("--increment-lag", type=int,
                        help="Also plot normalized-coordinate increments over this many native steps.")
    parser.add_argument("--increment-score", choices=("mahalanobis", "l2"),
                        default="mahalanobis",
                        help="Full-vector increment anomaly score (default: mahalanobis).")
    parser.add_argument("--increment-score-percentile", "--mahalanobis-percentile",
                        dest="increment_score_percentile", type=float, default=95.0,
                        help="Full-trajectory score percentile shown as the threshold (default: 95).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--validation-reps", nargs="+", type=int)
    parser.add_argument("--test-reps", nargs="*", type=int)
    parser.add_argument("--output", type=Path, help="Override reduced_data/coordinate_trajectories/... output.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(plot_normalized_coordinates(
        args.experiment, args.rep, args.rank, args.timesteps, args.modes,
        increment_lag=args.increment_lag, start=args.start,
        increment_score=args.increment_score,
        increment_score_percentile=args.increment_score_percentile,
        data_root=args.data_root, shared_basis=args.shared_basis,
        validation_reps=args.validation_reps, test_reps=args.test_reps,
        output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
