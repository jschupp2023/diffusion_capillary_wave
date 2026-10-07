"""Training-only, trim-aware decorrelation of normalized model increments.

The checkpoint selects the exact shared-POD coordinates, training repetitions,
50 Hz spatial-mean treatment, normalization, and training trim. Correlations
are computed within repetitions and intact retained runs; gaps are never joined.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.neural_sde.train import load_model


def clean_runs(good):
    """Return half-open runs of retained native increment edges."""
    changes = np.diff(np.r_[False, good, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def segmented_correlation(segments, max_lag, min_pairs=1000):
    """Exact pooled Pearson correlation at each lag across intact segments."""
    if not segments:
        raise ValueError("At least one retained increment segment is required.")
    dimension = segments[0].shape[1]
    count = np.zeros(max_lag + 1, dtype=np.int64)
    sums = np.zeros((5, max_lag + 1, dimension), dtype=np.float64)
    for values in segments:
        n = len(values)
        last = min(max_lag, n - 1)
        if last < 0:
            continue
        lag = np.arange(last + 1)
        prefix = np.vstack((np.zeros((1, dimension)), np.cumsum(values, axis=0)))
        squares = np.vstack((np.zeros((1, dimension)), np.cumsum(values**2, axis=0)))
        size = next_fast_len(2 * n - 1)
        spectrum = rfft(values, n=size, axis=0)
        products = irfft(spectrum * spectrum.conj(), n=size, axis=0)[:last + 1]
        sums[:, :last + 1] += np.stack((
            prefix[n - lag], prefix[n] - prefix[lag],
            squares[n - lag], squares[n] - squares[lag], products))
        count[:last + 1] += n - lag
    n = np.maximum(count[:, None], 1)
    sx, sy, sxx, syy, sxy = sums
    vx = np.maximum(sxx - sx**2 / n, 0.)
    vy = np.maximum(syy - sy**2 / n, 0.)
    denominator = np.sqrt(vx * vy)
    valid = (count[:, None] >= min_pairs) & (denominator > 0)
    correlation = np.divide(
        sxy - sx * sy / n, denominator,
        out=np.full_like(sxy, np.nan), where=valid)
    return np.clip(correlation, -1., 1.), count


def decorrelation_summary(correlation, threshold=.1, consecutive=20):
    """First crossing and first K-lag low-correlation run for one curve."""
    values = np.asarray(correlation, dtype=float)
    supported = np.flatnonzero(np.isfinite(values))
    last = int(supported[-1]) if len(supported) else 0
    values = values[:last + 1]
    below = np.isfinite(values) & (np.abs(values) < threshold)
    crossings = np.flatnonzero(below[1:]) + 1
    first = int(crossings[0]) if len(crossings) else None
    persistent = None
    if len(below) > consecutive:
        runs = np.convolve(below[1:].astype(np.int64), np.ones(consecutive, dtype=np.int64),
                           mode="valid")
        hits = np.flatnonzero(runs == consecutive)
        persistent = int(hits[0] + 1) if len(hits) else None
    rebound = (None if persistent is None else
               bool(np.any(~below[persistent + consecutive:last + 1])))
    return dict(first_crossing_frames=first, persistent_frames=persistent,
                supported_max_lag=last, later_rebound_after_K=rebound,
                status="established" if persistent is not None else "not_established")


def analyze_increment_path(values, good, target_mean, target_std, max_lag, min_pairs):
    """Correlate the exact normalized native increments supplied to the model."""
    state = np.asarray(values, dtype=np.float32)
    normalized = (np.diff(state, axis=0) - target_mean) / target_std
    runs = clean_runs(good)
    segments = [normalized[start:stop].astype(np.float64)
                for start, stop in runs if stop > start]
    if not segments:
        shape = (max_lag + 1, state.shape[1])
        return np.full(shape, np.nan), np.zeros(max_lag + 1, dtype=np.int64), runs
    correlation, count = segmented_correlation(segments, max_lag, min_pairs)
    return correlation, count, runs


def plot_results(output, correlations, counts, candidates, labels, dt, threshold):
    median = np.ma.median(np.ma.masked_invalid(correlations), axis=0).filled(np.nan)
    lag_us = np.arange(median.shape[0]) * dt * 1e6
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    image = axes[0].imshow(
        median.T, aspect="auto", origin="upper",
        extent=[0, lag_us[-1], len(labels) - .5, -.5],
        cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set(xlabel="Increment lag (µs)", ylabel="Model coordinate",
                title="Median training-repetition correlation")
    fig.colorbar(image, ax=axes[0], label="Correlation")
    for coordinate in (0, 1, 2, 3, 10, 20):
        if coordinate < len(labels):
            axes[1].plot(lag_us, median[:, coordinate], label=labels[coordinate])
    axes[1].axhline(threshold, color="gray", ls=":")
    axes[1].axhline(-threshold, color="gray", ls=":")
    axes[1].set(xlabel="Increment lag (µs)", ylabel="Correlation",
                title="Representative normalized increments")
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "increment_correlations.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    lag_us = np.arange(counts.shape[1]) * dt * 1e6
    for index in range(len(counts)):
        ax.plot(lag_us, 100 * counts[index] / np.maximum(candidates[index], 1), alpha=.7)
    ax.set(xlabel="Increment lag (µs)", ylabel="Retained candidate pairs (%)",
           title="Complete-window retention by training repetition")
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output / "retained_pairs.png", dpi=160); plt.close(fig)


def analyze(args):
    torch.set_num_threads(4)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path /= "best.pt"
    model, checkpoint = load_model(checkpoint_path, "cpu")
    if model.config.state_variable != "velocity" or model.config.lag_steps != 1:
        raise ValueError("This diagnostic requires a native-lag velocity checkpoint.")
    training = checkpoint.get("training", {})
    cutoff = training.get("train_trim_norm_cutoff")
    trim_scale = np.asarray(training.get("train_trim_state_std"), dtype=np.float32)
    if (cutoff is None or trim_scale.shape != (model.config.reduced_dim,)
            or not np.isfinite(trim_scale).all() or np.any(trim_scale <= 0)):
        raise ValueError("Checkpoint lacks a valid saved training trim rule.")
    target_mean = model.target_mean.detach().cpu().numpy().astype(np.float32)
    target_std = model.target_std.detach().cpu().numpy().astype(np.float32)
    data = checkpoint_data(checkpoint, data_root=args.data_root, shared_basis=args.shared_basis)
    repetitions = data.train_repetitions[:args.max_repetitions]
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if data.include_spatial_mean:
        mean_label = ("sqrt(N)*spatial_mean" if data.spatial_mean_highpass_hz is None else
                      f"sqrt(N)*spatial_mean_highpass_{data.spatial_mean_highpass_hz:g}Hz")
    else:
        mean_label = None
    labels = ([mean_label] if mean_label else []) + [f"POD_{mode}" for mode in range(1, data.rank + 1)]
    correlations, counts, candidates, retention, per_rep = [], [], [], [], []
    for name in repetitions:
        print(f"Reading {name}", flush=True)
        _, values = data.read_coordinates(name, 0, data.frame_counts[name])
        state = torch.as_tensor(values, dtype=torch.float32)
        increments = torch.diff(state, dim=0)
        good = (torch.linalg.vector_norm(increments / torch.from_numpy(trim_scale), dim=1)
                <= float(cutoff)).numpy()
        correlation, count, runs = analyze_increment_path(
            values, good, target_mean, target_std, args.max_lag, args.min_pairs)
        correlations.append(correlation); counts.append(count)
        candidates.append(np.maximum(len(good) - np.arange(args.max_lag + 1), 0))
        retention.append(dict(repetition=name, total_native_increments=len(good),
                              retained_native_increments=int(good.sum()),
                              retained_percent=100 * float(good.mean()),
                              longest_clean_run=int(max(
                                  (stop-start for start, stop in runs), default=0))))
        for coordinate, label in enumerate(labels):
            for threshold in args.thresholds:
                result = decorrelation_summary(correlation[:, coordinate], threshold, args.consecutive)
                for key in ("first_crossing_frames", "persistent_frames"):
                    value = result[key]
                    result[key.replace("_frames", "_microseconds")] = (
                        None if value is None else value * data.native_dt * 1e6)
                per_rep.append(dict(repetition=name, coordinate=label,
                                    threshold=threshold, **result))
        print(f"  retained {100*good.mean():.2f}% of native increments; "
              f"supported pairs at max lag {count[-1]:,}", flush=True)
    correlations = np.stack(correlations); counts = np.stack(counts)
    candidates = np.stack(candidates)
    rows = []
    for label in labels:
        for threshold in args.thresholds:
            selected = [row for row in per_rep
                        if row["coordinate"] == label and row["threshold"] == threshold]
            row = dict(coordinate=label, threshold=threshold, repetitions=len(selected))
            for metric in ("first_crossing", "persistent"):
                values = [item[metric + "_frames"] for item in selected
                          if item[metric + "_frames"] is not None]
                row[metric + "_established_repetitions"] = len(values)
                for statistic, function in (("median", np.median), ("min", np.min), ("max", np.max)):
                    value = float(function(values)) if values else None
                    row[f"{metric}_{statistic}_frames"] = value
                    row[f"{metric}_{statistic}_microseconds"] = (
                        None if value is None else value * data.native_dt * 1e6)
            rows.append(row)
    with (output / "horizons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(
        checkpoint=str(checkpoint_path), checkpoint_epoch=(
            None if checkpoint.get("epoch") is None else int(checkpoint["epoch"])),
        experiment=data.experiment, rank=data.rank, training_repetitions=list(repetitions),
        partial_training_subset=len(repetitions) != len(data.train_repetitions),
        native_dt_seconds=float(data.native_dt), coordinate_labels=labels,
        coordinate_convention=(f"coordinate 0 = {mean_label}; "
                               "coordinates 1..rank = shared-POD coefficients"
                               if mean_label else "coordinates 0..rank-1 = shared-POD coefficients"),
        increment_convention=("(a[t]-a[t-1]-checkpoint target_mean)/checkpoint target_std; "
                              "the exact normalized native-frame velocity seen by the model"),
        normalization_note=("Pearson correlation is invariant to nonzero per-coordinate affine "
                            "normalization; normalization is nevertheless applied explicitly."),
        trim=dict(percent=training.get("train_trim_percent"), cutoff=float(cutoff),
                  scale=trim_scale.tolist(), rule=("both increment endpoints and every intervening "
                  "native increment lie in one retained run"), interpretation=("upper-tail amplitude "
                  "heuristic, not an identified discontinuity fraction")),
        max_lag=args.max_lag, thresholds=args.thresholds, consecutive=args.consecutive,
        min_pairs_per_repetition=args.min_pairs,
        definitions=dict(first_crossing="first lag with abs(correlation) below threshold",
                         persistent="first start of K consecutive supported lags below threshold",
                         later_rebound="correlation exceeds threshold again after that K-lag run",
                         aggregate="median and range of per-repetition horizons; missing is not zero",
                         limits="finite-range marginal correlation, not independence or conditional predictability"),
        retention=retention, horizons=rows, per_repetition=per_rep)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "correlations.npz", correlations=correlations,
                        pair_counts=counts, candidate_pair_counts=candidates,
                        repetitions=np.asarray(repetitions), coordinates=np.asarray(labels),
                        lag_frames=np.arange(args.max_lag + 1), native_dt=data.native_dt,
                        target_mean=target_mean, target_std=target_std)
    plot_results(output, correlations, counts, candidates, labels, data.native_dt, .1)
    print(output, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--max-lag", type=int, default=256, help="Native increment frames.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.3, .1, .01])
    parser.add_argument("--consecutive", type=int, default=20)
    parser.add_argument("--min-pairs", type=int, default=1000)
    parser.add_argument("--max-repetitions", type=int,
                        help="First N training repetitions, for a smoke check only.")
    args = parser.parse_args(argv)
    if (args.max_lag < 2 or not 1 <= args.consecutive <= args.max_lag
            or args.min_pairs < 2 or any(not 0 < value < 1 for value in args.thresholds)
            or (args.max_repetitions is not None and args.max_repetitions < 1)):
        parser.error("Require valid lag, consecutive count, pair count, and thresholds in (0,1).")
    analyze(args)


if __name__ == "__main__":
    main()
