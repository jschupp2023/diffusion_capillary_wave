"""Plot shared-POD states and their model-step changes from data or a saved rollout.

Example:
  python -m modelling.acdm.experiments.plot_coordinate_dynamics \
    --checkpoint runs/0p20_r50_lag1_acdm_edm_incr_joint/best.pt \
    --split train --repetition Ca_ac_0p001660_rep1 --steps 1000

Pass --rollout PATH to also compare one generated trajectory with its matching
reference. The rollout must have been made by the specified checkpoint. For
future-state models, increments are differences of adjacent saved states and
their normalizing mean/std are estimated from the sampled training windows.
For increment models, the checkpoint's training-target mean/std are reused.
POD-only checkpoints display POD 1 as coordinate zero without mean scaling.
Traces show stored reduced coordinates before the model's internal standardization;
coordinate zero, when present, is sqrt(N) times the physical spatial mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelling.acdm.conditional_edm.checkpoint import load_checkpoint
from modelling.acdm.conditional_edm.data import checkpoint_data


def coordinate_labels(rank, include_spatial_mean=True):
    pod = [f"POD {k}" for k in range(1, rank + 1)]
    return (["√N × spatial mean"] + pod) if include_spatial_mean else pod


def training_ranges(data, lag, increment_mean=None, increment_std=None,
                    windows_per_rep=4, window_steps=512):
    """Sample training windows and derive change scaling when target is a future state."""
    states, increments = [], []
    for name in data.train_repetitions:
        width = min(window_steps, (data.frame_counts[name] - 1) // lag)
        if width < 1:
            continue
        last = data.frame_counts[name] - 1 - width * lag
        for start in np.unique(np.linspace(0, last, windows_per_rep, dtype=int)):
            _, values = data.read_coordinates(name, int(start), int(start + width * lag + 1))
            values = values[::lag]
            states.append(values)
            increments.append(np.diff(values, axis=0))
    if not states:
        raise ValueError("No training transitions available for reference ranges.")
    state = np.concatenate(states)
    increment = np.concatenate(increments)
    if increment_mean is None or increment_std is None:
        if increment_mean is not None or increment_std is not None:
            raise ValueError("Supply both increment mean and standard deviation, or neither.")
        increment_mean = increment.mean(axis=0)
        increment_std = increment.std(axis=0)
        increment_std = np.where(increment_std > 0, increment_std, 1.)
    state_q = np.quantile(state, [.005, .995], axis=0)
    increment_q = np.quantile(increment, [.005, .995], axis=0)
    normalized_q = np.quantile((increment - increment_mean) / increment_std, [.005, .995], axis=0)
    return state_q, increment_q, normalized_q, len(increment), increment_mean, increment_std


def make_series(states, increment_mean, increment_std):
    states = np.asarray(states, dtype=np.float64)
    increment = np.diff(states, axis=0)
    return states, increment, (increment - increment_mean) / increment_std


def first_steps(series, count):
    """Keep count transitions and their count + 1 states."""
    return series[0][:count + 1], series[1][:count], series[2][:count]


def plot_traces(path, reference, generated, ranges, labels, modes, unit, dt,
                include_spatial_mean=True):
    rows = ([0] + modes) if include_spatial_mean else [mode - 1 for mode in modes]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 2.15 * len(rows)),
                             sharex="col", squeeze=False, constrained_layout=True)
    names = [f"state [{unit}]", f"increment [{unit}]", "normalized increment [σ]"]
    for col, (ref, gen, limits) in enumerate(zip(reference, generated or (None,) * 3, ranges)):
        for row, coordinate in enumerate(rows):
            ax = axes[row, col]
            ax.axhspan(*limits[:, coordinate], color="0.90", zorder=0)
            ax.plot(np.arange(len(ref)), ref[:, coordinate], color="0.15", lw=.8,
                    label="measured" if row == 0 and col == 0 else None)
            if gen is not None:
                ax.plot(np.arange(len(gen)), gen[:, coordinate], color="C3", lw=.8,
                        alpha=.85, label="ROM" if row == 0 and col == 0 else None)
            if col == 0:
                ax.set_ylabel(labels[coordinate])
            if row == 0:
                ax.set_title(names[col])
            ax.grid(alpha=.18)
    for ax in axes[-1]:
        ax.set_xlabel(f"model step (Δt = {dt * 1e6:.3g} µs)")
    if generated is not None:
        axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Gray band: sampled training 0.5–99.5% per-coordinate range")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overview(path, series, normalized_range, state_mean, state_std,
                  increment_std, labels, dt, include_spatial_mean=True,
                  increment_scale_source="target normalization"):
    state, increment, normalized = series
    state_z = (state - state_mean) / state_std
    finite = np.isfinite(normalized)
    outside = finite & ((normalized < normalized_range[0]) |
                        (normalized > normalized_range[1]))
    nonfinite = ~finite
    outside_fraction = float(outside.sum() / finite.sum()) if finite.any() else None
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    coordinate_axis = "coordinate (0 = √N × spatial mean)" if include_spatial_mean else "coordinate (0 = POD 1)"
    for ax, array, title in ((axes[0], state_z, "State / training state standard deviation"),
                             (axes[1], normalized, f"Increment / {increment_scale_source} standard deviation")):
        image = ax.imshow(array.T, origin="lower", aspect="auto", interpolation="nearest",
                          cmap="coolwarm", vmin=-5, vmax=5)
        ax.set(title=title, ylabel=coordinate_axis)
        fig.colorbar(image, ax=ax, label="training σ", pad=.01)
    axes[1].imshow(outside.T, origin="lower", aspect="auto", interpolation="nearest",
                   cmap="gray_r", vmin=0, vmax=1, alpha=outside.T.astype(float) * .8)
    axes[1].set_title("Normalized increment; black = outside sampled training range; white = nonfinite")
    axes[1].set_xlabel(f"model step (Δt = {dt * 1e6:.3g} µs)")
    pooled_ratio = increment_std / state_std
    window_ratio = np.divide(increment.std(axis=0), state.std(axis=0),
                             out=np.full(state.shape[1], np.nan), where=state.std(axis=0) > 0)
    axes[2].semilogy(np.arange(len(pooled_ratio)), pooled_ratio, ".-", ms=3,
                     label=increment_scale_source)
    axes[2].semilogy(np.arange(len(window_ratio)), window_ratio, ".-", ms=3,
                     label="displayed window")
    axes[2].set(xlabel=coordinate_axis,
                ylabel="increment std / state std",
                title="One-step change scale (a proxy, not a fitted decorrelation time)")
    axes[2].legend()
    axes[2].grid(alpha=.25)
    fig.suptitle(f"All {len(labels)} coordinates; "
                 f"{outside_fraction:.2%} of finite increments outside sampled training range; "
                 f"{nonfinite.mean():.2%} nonfinite" if outside_fraction is not None
                 else f"All {len(labels)} coordinates; every increment is nonfinite")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    count = outside.sum(axis=1)
    nonfinite_steps = np.flatnonzero(nonfinite.any(axis=1))
    selected = ([0] if include_spatial_mean else []) + [int(include_spatial_mean) + mode - 1
                                                         for mode in (1, 2, 5, 10, 20, len(labels) - int(include_spatial_mean))
                                                         if mode <= len(labels) - int(include_spatial_mean)]
    return dict(outside_fraction=outside_fraction,
                nonfinite_fraction=float(nonfinite.mean()),
                first_nonfinite_step=int(nonfinite_steps[0] + 1) if len(nonfinite_steps) else None,
                first_outside_step=int(np.flatnonzero(count)[0] + 1) if count.any() else None,
                maximum_outside_coordinates=int(count.max()) if len(count) else 0,
                window_change_scale_ratio={labels[i]: (float(window_ratio[i])
                                                      if np.isfinite(window_ratio[i]) else None)
                                           for i in dict.fromkeys(selected)})


def read_rollout(path, checkpoint_path, data, lag, condition, ensemble, steps):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "rollout.npz"
    metadata = json.loads(path.with_name("metrics.json").read_text())
    if Path(metadata["checkpoint"]).resolve() != checkpoint_path:
        raise ValueError("Rollout checkpoint differs from --checkpoint; its normalization may differ.")
    if (int(metadata["rank"]) != data.rank or int(metadata["lag_steps"]) != lag
            or bool(metadata.get("include_spatial_mean", True)) != data.include_spatial_mean
            or Path(metadata["source_shared_basis"]).resolve() != data.basis_path):
        raise ValueError("Rollout rank, lag, coordinate convention, or shared basis differs from checkpoint data.")
    with np.load(path, allow_pickle=False) as saved:
        generated = saved["trajectories"]
        reference = saved["reference"]
        if (generated.ndim != 4 or reference.ndim != 3
                or generated.shape[-1] != data.n_coordinates
                or reference.shape != (generated.shape[0], generated.shape[2], generated.shape[3])):
            raise ValueError("Rollout arrays do not match checkpoint coordinates.")
        if not 0 <= condition < len(generated) or not 0 <= ensemble < generated.shape[1]:
            raise ValueError("Requested rollout condition or ensemble is unavailable.")
        name = str(saved["repetition"][condition])
        stop = min(steps + 1, generated.shape[2])
        return (reference[condition, :stop], generated[condition, ensemble, :stop],
                name, int(stop - 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Optional relocated reduced-data root")
    parser.add_argument("--shared-basis", type=Path, help="Optional relocated shared basis")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--repetition", help="Name of a repetition in the selected split; defaults to first")
    parser.add_argument("--start", type=int, default=0, help="Native-frame index in the measured repetition")
    parser.add_argument("--steps", type=int, default=1000, help="Number of model-step increments")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50],
                        help="POD mode numbers to show as individual traces")
    parser.add_argument("--rollout", type=Path, help="Saved rollout directory or rollout.npz")
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--ensemble", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Output directory; defaults beside checkpoint")
    args = parser.parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    target_mode = checkpoint["model_config"]["target_mode"]
    if target_mode not in {"increment", "future_state"}:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    lag = int(checkpoint["model_config"]["lag_steps"])
    data = checkpoint_data(checkpoint, data_root=args.data_root, shared_basis=args.shared_basis)
    choices = data.splits[args.split]
    name = args.repetition or (choices[0] if choices else None)
    if name not in choices or args.steps < 1 or args.start < 0:
        raise ValueError("Choose a repetition in the selected split, nonnegative start, and positive steps.")
    if args.start + args.steps * lag >= data.frame_counts[name]:
        raise ValueError("Requested measured window extends past the repetition.")
    modes = sorted(set(args.modes))
    if any(mode < 1 or mode > data.rank for mode in modes):
        raise ValueError(f"Modes must lie between 1 and {data.rank}.")
    normalization = checkpoint["normalization"]
    state_mean, state_std = (np.asarray(normalization[key], dtype=np.float64)
                             for key in ("state_mean", "state_std"))
    if target_mode == "increment":
        target_mean, target_std = (np.asarray(normalization[key], dtype=np.float64)
                                   for key in ("target_mean", "target_std"))
        ranges = training_ranges(data, lag, target_mean, target_std)
        increment_scale_source = "training target"
    else:
        ranges = training_ranges(data, lag)
        increment_scale_source = "sampled training increment"
    increment_mean, increment_std = ranges[4:]
    time, measured = data.read_coordinates(name, args.start, args.start + args.steps * lag + 1)
    measured = measured[::lag]
    dt = float(np.mean(np.diff(time[::lag])))
    output = (args.output or checkpoint_path.parent / "coordinate_diagnostics").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = coordinate_labels(data.rank, data.include_spatial_mean)
    measured_series = make_series(measured, increment_mean, increment_std)
    plot_traces(output / "measured_traces.png", measured_series, None, ranges[:3], labels,
                modes, data.signal_units, dt, data.include_spatial_mean)
    summary = {"checkpoint": str(checkpoint_path), "training_reference_transitions": ranges[3],
               "target_mode": target_mode, "include_spatial_mean": data.include_spatial_mean,
               "coordinate_convention": ("stored reduced coordinates; coordinate 0 = sqrt(N) * spatial mean"
                                         if data.include_spatial_mean else "shared POD coefficients only"),
               "increment_scale_source": increment_scale_source,
               "measured_repetition": name, "measured_start": args.start, "lag_steps": lag,
               "measured": plot_overview(output / "measured_overview.png", measured_series, ranges[2],
                                          state_mean, state_std, increment_std, labels, dt,
                                          data.include_spatial_mean, increment_scale_source)}
    if args.rollout:
        reference, generated, rollout_name, rollout_steps = read_rollout(
            args.rollout, checkpoint_path, data, lag, args.condition, args.ensemble, args.steps)
        reference_series = make_series(reference, increment_mean, increment_std)
        generated_series = make_series(generated, increment_mean, increment_std)
        plot_traces(output / "rollout_traces.png", reference_series, generated_series, ranges[:3],
                    labels, modes, data.signal_units, dt, data.include_spatial_mean)
        for zoom in (100, 300):
            if rollout_steps > zoom:
                plot_traces(output / f"rollout_traces_first_{zoom}.png",
                            first_steps(reference_series, zoom), first_steps(generated_series, zoom),
                            ranges[:3], labels, modes, data.signal_units, dt,
                            data.include_spatial_mean)
        summary["rollout"] = {"repetition": rollout_name, "condition": args.condition,
                              "ensemble": args.ensemble, "steps": rollout_steps,
                              "generated": plot_overview(output / "rollout_overview.png", generated_series,
                                                         ranges[2], state_mean, state_std, increment_std,
                                                         labels, dt, data.include_spatial_mean,
                                                         increment_scale_source)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
