"""Trim-aware temporal diagnostic of Neural-SDE one-step innovations.

For every repetition in the selected split, this computes the exact velocity transition used
by a native-lag Neural SDE, subtracts its learned conditional mean, and saves
both marginally standardized and jointly whitened innovations. Correlations and
Welch spectra are evaluated only inside contiguous runs whose complete model
stencil passes the checkpoint's saved training trim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import torch

from data_analysis.correlations.training_pod_decorrelation import (
    clean_runs, decorrelation_summary, segmented_correlation)
from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.neural_sde.train import load_model


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def eligible_velocity_transitions(edge_good: np.ndarray, history_steps: int) -> np.ndarray:
    """Return eligibility for current indices history_steps .. n_frames-2.

    A transition at current frame ``t`` uses the native increment edges from
    ``t-history_steps`` through ``t`` inclusive: history, current velocity, and
    following velocity. Every edge must pass the frozen training trim.
    """
    edge_good = np.asarray(edge_good, dtype=bool)
    if edge_good.ndim != 1 or history_steps < 1 or len(edge_good) <= history_steps:
        raise ValueError("Need a one-dimensional edge mask longer than history_steps.")
    current = np.arange(history_steps, len(edge_good))
    bad_prefix = np.r_[0, np.cumsum(~edge_good)]
    return (bad_prefix[current + 1] - bad_prefix[current - history_steps]) == 0


def transition_innovations(model, values: np.ndarray, edge_good: np.ndarray, *,
                           batch_size: int, device: torch.device
                           ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Return clean marginal and joint-whitened innovation segments."""
    state = np.asarray(values, dtype=np.float32)
    history_steps = model.config.history_steps
    eligible = eligible_velocity_transitions(edge_good, history_steps)
    current_indices = np.arange(history_steps, len(state) - 1)
    if len(eligible) != len(current_indices):
        raise RuntimeError("Transition eligibility and current indices disagree.")
    marginal = np.full((len(current_indices), state.shape[1]), np.nan, dtype=np.float32)
    whitened = np.full_like(marginal, np.nan)
    selected = np.flatnonzero(eligible)
    history_offsets = np.arange(1, history_steps + 1)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            local = selected[start:start + batch_size]
            indices = current_indices[local]
            current = torch.from_numpy(state[indices]).to(device)
            history = torch.from_numpy(state[indices[:, None] - history_offsets]).to(device)
            velocity = current - history[:, 0]
            next_velocity = torch.from_numpy(state[indices + 1] - state[indices]).to(device)
            mean, factor = model.velocity_step_distribution(current, history, velocity)
            if factor.ndim == 2:
                factor = factor.expand(len(current), -1, -1)
            residual = next_velocity - mean
            joint = torch.linalg.solve_triangular(
                factor, residual.unsqueeze(-1), upper=False).squeeze(-1)
            marginal_scale = torch.sqrt(torch.sum(factor.square(), dim=-1))
            marginal[local] = (residual / marginal_scale).cpu().numpy()
            whitened[local] = joint.cpu().numpy()
    runs = clean_runs(eligible)
    marginal_segments = [marginal[a:b].astype(np.float64) for a, b in runs if b > a]
    whitened_segments = [whitened[a:b].astype(np.float64) for a, b in runs if b > a]
    if not marginal_segments or not all(np.isfinite(x).all() for x in marginal_segments + whitened_segments):
        raise ValueError("No finite clean innovation segments were produced.")
    return marginal_segments, whitened_segments, eligible


def segmented_welch(segments: list[np.ndarray], fs: float, nperseg: int
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Window-count-weighted Welch mean without windows crossing trim gaps."""
    densities, weights, frequency = [], [], None
    overlap = nperseg // 2
    step = nperseg - overlap
    for segment in segments:
        if len(segment) < nperseg:
            continue
        f, density = welch(
            segment, fs=fs, window="hann", nperseg=nperseg, noverlap=overlap,
            detrend="constant", scaling="density", axis=0)
        windows = 1 + (len(segment) - nperseg) // step
        frequency = f
        densities.append(density)
        weights.append(windows)
    if not densities:
        raise ValueError("No clean segment is long enough for the requested Welch nperseg.")
    return frequency, np.average(densities, axis=0, weights=weights), int(sum(weights))


def pooled_update(accumulator: dict, segments: list[np.ndarray]) -> None:
    for values in segments:
        accumulator["count"] += len(values)
        accumulator["sum"] += values.sum(axis=0)
        accumulator["cross"] += values.T @ values


def pooled_finish(accumulator: dict) -> tuple[np.ndarray, np.ndarray]:
    count, total, cross = (accumulator[key] for key in ("count", "sum", "cross"))
    mean = total / count
    covariance = (cross - np.outer(total, total) / count) / (count - 1)
    return mean, covariance


def white_psd(frequency: np.ndarray, fs: float) -> np.ndarray:
    reference = np.full_like(frequency, 2. / fs)
    reference[0] = 1. / fs
    if len(frequency) > 1 and math.isclose(frequency[-1], fs / 2):
        reference[-1] = 1. / fs
    return reference


def plot_acf(output: Path, correlations: dict[str, np.ndarray], labels: list[str],
             dt: float, threshold: float) -> None:
    selected = [index for index in (0, 1, 2, 3, 10, 20) if index < len(labels)]
    lag = np.arange(next(iter(correlations.values())).shape[1])
    lag_us = lag * dt * 1e6
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, kind in enumerate(("marginal", "whitened")):
        values = correlations[kind]
        median = np.nanmedian(values, axis=0)
        image = axes[0, column].imshow(
            median.T, aspect="auto", origin="upper",
            extent=[lag_us[0], lag_us[-1], len(labels) - .5, -.5],
            cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0, column].set(title=f"{kind.capitalize()} innovation ACF",
                            xlabel="Lag [microseconds]", ylabel="Innovation coordinate")
        figure.colorbar(image, ax=axes[0, column], label="Pearson correlation")
        for index in selected:
            label = labels[index] if kind == "marginal" else f"whitened {index}"
            axes[1, column].plot(lag_us, median[:, index], label=label)
        axes[1, column].axhline(threshold, color="gray", linestyle=":")
        axes[1, column].axhline(-threshold, color="gray", linestyle=":")
        axes[1, column].axhline(0, color="gray", linewidth=.5)
        axes[1, column].set(xlabel="Lag [microseconds]", ylabel="Correlation")
        axes[1, column].grid(alpha=.2)
        axes[1, column].legend(fontsize=7, ncol=2)
    figure.savefig(output / "innovation_acf.png", dpi=170)
    plt.close(figure)


def plot_psd(output: Path, frequency: np.ndarray, densities: dict[str, np.ndarray],
             labels: list[str], fs: float) -> None:
    selected = [index for index in (0, 1, 2, 3, 10, 20) if index < len(labels)]
    reference = white_psd(frequency, fs)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, kind in zip(axes, ("marginal", "whitened"), strict=True):
        ratio = densities[kind] / reference[None, :, None]
        median = np.nanmedian(ratio, axis=0)
        for index in selected:
            label = labels[index] if kind == "marginal" else f"whitened {index}"
            axis.semilogx(frequency[1:], 10 * np.log10(median[1:, index]), label=label)
        population = ratio[:, 1:, :].reshape(-1, ratio.shape[1] - 1)
        axis.fill_between(
            frequency[1:], 10 * np.log10(np.nanpercentile(population, 10, axis=0)),
            10 * np.log10(np.nanpercentile(population, 90, axis=0)),
            color="gray", alpha=.15, label="10--90% all coordinates")
        axis.axhline(0, color="black", linestyle=":", linewidth=.8, label="unit white noise")
        axis.set(title=f"{kind.capitalize()} innovation PSD / white PSD",
                 xlabel="Frequency [Hz]", ylabel="Power ratio [dB]")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, ncol=2)
    figure.savefig(output / "innovation_psd.png", dpi=170)
    plt.close(figure)


def analyze(args) -> Path:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path /= "best.pt"
    device = torch.device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    if model.config.state_variable != "velocity" or model.config.lag_steps != 1:
        raise ValueError("This diagnostic requires a native-lag velocity checkpoint.")
    data = checkpoint_data(checkpoint, data_root=args.data_root, shared_basis=args.shared_basis)
    training = checkpoint.get("training", {})
    cutoff = training.get("train_trim_norm_cutoff")
    trim_scale = training.get("train_trim_state_std")
    if cutoff is not None:
        trim_scale = np.asarray(trim_scale, dtype=np.float32)
        if trim_scale.shape != (model.config.reduced_dim,) or np.any(trim_scale <= 0):
            raise ValueError("Checkpoint has an invalid saved training trim scale.")
    repetitions = data.splits[args.split][:args.max_repetitions]
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mean_label = None
    if data.include_spatial_mean:
        mean_label = ("sqrt(N)*spatial_mean" if data.spatial_mean_highpass_hz is None else
                      f"sqrt(N)*spatial_mean_highpass_{data.spatial_mean_highpass_hz:g}Hz")
    labels = ([mean_label] if mean_label else []) + [f"POD_{mode}" for mode in range(1, data.rank + 1)]
    dimension = model.config.reduced_dim
    pooled = {kind: {"count": 0, "sum": np.zeros(dimension),
                     "cross": np.zeros((dimension, dimension))}
              for kind in ("marginal", "whitened")}
    correlations = {kind: [] for kind in pooled}
    spectra = {kind: [] for kind in pooled}
    counts, retention = [], []
    frequency = None
    for name in repetitions:
        print(f"Reading and evaluating {name}", flush=True)
        _, values = data.read_coordinates(name, 0, data.frame_counts[name])
        values32 = np.asarray(values, dtype=np.float32)
        if cutoff is None:
            edge_good = np.ones(len(values32) - 1, dtype=bool)
        else:
            edge_good = (np.linalg.norm(np.diff(values32, axis=0) / trim_scale, axis=1)
                         <= float(cutoff))
        marginal, whitened, eligible = transition_innovations(
            model, values32, edge_good, batch_size=args.batch_size, device=device)
        record = dict(repetition=name, native_edges=len(edge_good),
                      retained_edges=int(edge_good.sum()), transitions=len(eligible),
                      retained_transitions=int(eligible.sum()),
                      retained_transition_percent=100 * float(eligible.mean()),
                      clean_runs=len(clean_runs(eligible)),
                      longest_clean_run=max((b - a for a, b in clean_runs(eligible)), default=0))
        for kind, segments in (("marginal", marginal), ("whitened", whitened)):
            correlation, count = segmented_correlation(segments, args.max_lag, args.min_pairs)
            f, density, windows = segmented_welch(segments, 1. / data.native_dt, args.nperseg)
            if frequency is not None and not np.array_equal(frequency, f):
                raise RuntimeError("Welch frequency grids differ across repetitions.")
            frequency = f
            correlations[kind].append(correlation)
            spectra[kind].append(density)
            pooled_update(pooled[kind], segments)
            if kind == "whitened":
                counts.append(count)
                record["welch_windows"] = windows
        retention.append(record)
        print(f"  retained {eligible.sum():,}/{len(eligible):,} transitions "
              f"({100*eligible.mean():.2f}%)", flush=True)

    correlations = {key: np.stack(value) for key, value in correlations.items()}
    spectra = {key: np.stack(value) for key, value in spectra.items()}
    pooled_stats = {}
    for kind in pooled:
        mean, covariance = pooled_finish(pooled[kind])
        pooled_stats[kind] = dict(
            samples=int(pooled[kind]["count"]), mean=mean.tolist(),
            standard_deviation=np.sqrt(np.diag(covariance)).tolist(),
            covariance=covariance.tolist())
    white_covariance = np.asarray(pooled_stats["whitened"]["covariance"])
    identity_error = float(np.linalg.norm(white_covariance - np.eye(dimension))
                           / np.linalg.norm(np.eye(dimension)))
    white_mean = np.asarray(pooled_stats["whitened"]["mean"])
    white_std = np.asarray(pooled_stats["whitened"]["standard_deviation"])

    selected_lags = [lag for lag in (1, 2, 5, 10, 20, 50, 100, 256, 512)
                     if lag <= args.max_lag]
    acf_summary = {}
    horizons = {}
    for kind, values in correlations.items():
        median = np.nanmedian(values, axis=0)
        acf_summary[kind] = {
            str(lag): {
                "median_absolute_across_coordinates": float(np.median(np.abs(median[lag]))),
                "maximum_absolute_across_coordinates": float(np.max(np.abs(median[lag]))),
            } for lag in selected_lags}
        horizons[kind] = []
        for coordinate in range(dimension):
            item = decorrelation_summary(
                median[:, coordinate], args.threshold, args.consecutive)
            horizons[kind].append(dict(coordinate=(labels[coordinate] if kind == "marginal"
                                                   else f"whitened_{coordinate}"), **item))

    reference = white_psd(frequency, 1. / data.native_dt)
    psd_summary = {}
    nyquist = frequency[-1]
    bands = ((0, 112.5), (112.5, 1000), (1000, 10000), (10000, nyquist + 1))
    for kind, density in spectra.items():
        ratio = density / reference[None, :, None]
        psd_summary[kind] = []
        for low, high in bands:
            mask = (frequency > max(low, 0)) & (frequency < high)
            if mask.any():
                values = ratio[:, mask, :]
                psd_summary[kind].append(dict(
                    lower_hz=low, upper_hz=min(high, nyquist),
                    median_power_over_unit_white=float(np.median(values)),
                    p10=float(np.percentile(values, 10)),
                    p90=float(np.percentile(values, 90))))

    report = dict(
        checkpoint=str(checkpoint_path), experiment=data.experiment, rank=data.rank,
        split=args.split, repetitions=list(repetitions),
        partial_split_subset=len(repetitions) != len(data.splits[args.split]),
        coordinate_labels=labels, native_dt_seconds=float(data.native_dt),
        innovation_definition=dict(
            velocity="v_t=a_t-a_(t-1)",
            conditional_residual="v_(t+1) - E_theta[v_(t+1)|state/history]",
            marginal="residual divided by sqrt(diag(L L^T)); retains original coordinate labels",
            whitened="solve(L, residual); exact joint Gaussian whitening",
            expected="temporally independent N(0,I) under a correct one-step model"),
        trim=dict(percent=training.get("train_trim_percent", 0), cutoff=cutoff,
                  state_scale=trim_scale.tolist() if cutoff is not None else None,
                  rule="complete velocity-transition stencil must lie in one retained native-edge run",
                  gaps="never joined for ACF or Welch windows"),
        settings=dict(max_lag=args.max_lag, threshold=args.threshold,
                      consecutive=args.consecutive, min_pairs=args.min_pairs,
                      nperseg=args.nperseg, device=str(device)),
        retention=retention, pooled=pooled_stats,
        whitened_calibration=dict(mean_rmse=float(np.sqrt(np.mean(white_mean**2))),
                                  std_median=float(np.median(white_std)),
                                  std_min=float(np.min(white_std)),
                                  std_max=float(np.max(white_std)),
                                  covariance_relative_error_to_identity=identity_error),
        acf=acf_summary, decorrelation=horizons, psd_over_unit_white=psd_summary)
    (output / "summary.json").write_text(json.dumps(
        report, indent=2, allow_nan=False, default=json_default) + "\n")
    np.savez_compressed(
        output / "curves.npz", frequency_hz=frequency,
        lag_frames=np.arange(args.max_lag + 1), repetitions=np.asarray(repetitions),
        labels=np.asarray(labels), pair_counts=np.stack(counts),
        marginal_acf=correlations["marginal"], whitened_acf=correlations["whitened"],
        marginal_psd=spectra["marginal"], whitened_psd=spectra["whitened"])
    plot_acf(output, correlations, labels, data.native_dt, args.threshold)
    plot_psd(output, frequency, spectra, labels, 1. / data.native_dt)

    lines = [
        f"# {args.split.capitalize()} innovation diagnostic", "",
        f"All {len(repetitions)} {args.split} repetitions; "
        f"{sum(item['retained_transitions'] for item in retention):,}/"
        f"{sum(item['transitions'] for item in retention):,} complete-stencil transitions retained.", "",
        "| Joint-whitened calibration | Value |", "|---|---:|",
        f"| Mean RMSE from zero | {report['whitened_calibration']['mean_rmse']:.4g} |",
        f"| Median coordinate standard deviation | {report['whitened_calibration']['std_median']:.4g} |",
        f"| Standard-deviation range | {report['whitened_calibration']['std_min']:.4g}--{report['whitened_calibration']['std_max']:.4g} |",
        f"| Covariance relative error from identity | {identity_error:.4g} |", "",
        "| Lag [frames] | Median abs ACF | Maximum abs ACF |", "|---:|---:|---:|",
    ]
    for lag in selected_lags:
        item = acf_summary["whitened"][str(lag)]
        lines.append(f"| {lag} | {item['median_absolute_across_coordinates']:.4g} | "
                     f"{item['maximum_absolute_across_coordinates']:.4g} |")
    lines.extend(["", "Whitened innovations should have zero temporal ACF and a flat unit-white PSD. "
                  "Large systematic departures indicate unresolved temporal state or colored forcing."])
    (output / "quick_summary.md").write_text("\n".join(lines) + "\n")
    print(output, flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="train")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-lag", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--consecutive", type=int, default=20)
    parser.add_argument("--min-pairs", type=int, default=1000)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--max-repetitions", type=int,
                        help="First N repetitions of the selected split, for a smoke test only.")
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.max_lag < 2 or args.nperseg < 4
            or not 0 < args.threshold < 1 or not 1 <= args.consecutive <= args.max_lag
            or args.min_pairs < 2
            or (args.max_repetitions is not None and args.max_repetitions < 1)):
        parser.error("Invalid batch, lag, threshold, consecutive, pair, PSD, or repetition setting.")
    analyze(args)


if __name__ == "__main__":
    main()
