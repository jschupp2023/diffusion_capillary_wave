"""Analyze normalized shared-POD increments for every repetition of one experiment.

    python -m data_analysis.correlations.increment_covariance_batch 0p15 \
        --rank 20 --lags 1 2 5 10 20 50 --cutoff 5

The cutoff removes only the largest percentage of increment-vector L2 norms,
separately for each repetition and lag. Results go to the reduced-data tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT
from .increment_covariance import (estimate_increment_covariance,
                                   load_normalized_trajectory, save_diagnostic)


def _lag_label(lags):
    if len(lags) == 1:
        return f"k{lags[0]}"
    if lags == tuple(range(lags[0], lags[-1] + 1)):
        return f"k{lags[0]}-{lags[-1]}"
    digest = hashlib.sha256(",".join(map(str, lags)).encode()).hexdigest()[:8]
    return f"k{lags[0]}-{lags[-1]}_n{len(lags)}_{digest}"


def _plot_combined(h, curves, repetitions, *, ylabel, title, output, nonnegative=False):
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = plt.get_cmap("tab20")
    for index, (rep, curve) in enumerate(zip(repetitions, curves)):
        axis.plot(h, curve, "o-", ms=2.5, lw=1, color=colors(index % 20), label=f"rep {rep}")
    axis.set(xlabel="Lag h [s]", ylabel=ylabel, title=title)
    if nonnegative:
        axis.set_ylim(bottom=0)
    axis.grid(alpha=.25)
    axis.legend(ncol=4, fontsize=8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def analyze_batch(experiment, rank, lags, cutoff=0., *, data_root=DEFAULT_DATA_ROOT,
                  shared_basis=None, validation_reps=None, test_reps=None,
                  output=None, overwrite=False):
    """Fit one training scaler, then save individual and combined Q diagnostics."""
    lags = tuple(sorted(set(lags)))
    if not lags or any(not isinstance(k, (int, np.integer)) or k < 1 for k in lags):
        raise ValueError("--lags must contain positive integer steps.")
    if not np.isfinite(cutoff) or not 0 <= cutoff < 100:
        raise ValueError("--cutoff must be in [0, 100).")
    data = SharedPODTrainingData(experiment, rank, data_root=data_root,
                                 shared_basis=shared_basis, validation_reps=validation_reps,
                                 test_reps=test_reps).fit_standardization()
    names = sorted(data.frame_counts, key=lambda name: int(name.rsplit("_rep", 1)[1]))
    if any(data.frame_counts[name] <= lags[-1] for name in names):
        raise ValueError(f"All repetitions must have more than {lags[-1]} frames.")
    root = Path(data.source_config["data_root"])
    output = Path(output) if output is not None else (root / "increments_analysis" / experiment
              / f"r{rank}_upper{cutoff:g}_{_lag_label(lags)}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Nonempty output directory {output}; use --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    trim = (0., 100. - cutoff) if cutoff else None
    traces, energies, retained_counts, total_counts, repetitions = [], [], [], [], []
    h = None
    for name in names:
        rep = int(name.rsplit("_rep", 1)[1])
        z, dt, metadata = load_normalized_trajectory(experiment, rep, rank, prepared_data=data)
        metadata["upper_cutoff_percent"] = float(cutoff)
        h, q, retained, thresholds, energy = estimate_increment_covariance(
            z, dt, lags=lags, trim_percentiles=trim, return_details=True)
        destination = save_diagnostic(h, q, metadata, output / f"rep{rep}",
                                      overwrite=overwrite, retained_counts=retained,
                                      norm_thresholds=thresholds, trim_percentiles=trim,
                                      excluded_energy_percent=energy)
        print(destination, flush=True)
        repetitions.append(rep)
        traces.append(np.trace(q, axis1=1, axis2=2))
        energies.append(energy)
        retained_counts.append(retained)
        total_counts.append(len(z) - np.asarray(lags))
    traces = np.stack(traces)
    energies = np.stack(energies)
    np.savez_compressed(output / "combined.npz", repetition=np.asarray(repetitions),
                        k=np.asarray(lags), h_seconds=h, trace_Q=traces,
                        excluded_increment_energy_percent=energies,
                        retained_increment_counts=np.stack(retained_counts),
                        increment_counts=np.stack(total_counts))
    _plot_combined(h, traces, repetitions, ylabel="tr Q(h) [normalized²/s]",
                   title=f"{experiment}: shared POD rank {rank}", output=output / "combined_trace.png")
    _plot_combined(h, energies, repetitions, ylabel="Excluded increment L2 energy [%]",
                   title=f"{experiment}: upper {cutoff:g}% norm cutoff",
                   output=output / "combined_excluded_energy.png", nonnegative=True)
    report = dict(experiment=experiment, pod_rank=rank, shared_basis=str(data.basis_path),
                  normalization="training-split shared-POD state mean/std (float32 model-buffer precision)",
                  normalization_train_repetitions=list(data.train_repetitions),
                  validation_repetitions=list(data.validation_repetitions),
                  test_repetitions=list(data.test_repetitions),
                  state_mean=np.asarray(data.mean, dtype=np.float32).tolist(),
                  state_std=np.asarray(data.scale, dtype=np.float32).tolist(),
                  dt_seconds=data.native_dt, lags=list(lags), h_seconds=h.tolist(),
                  upper_cutoff_percent=float(cutoff),
                  selection="remove increments with L2 norm above the per-repetition, per-lag upper percentile",
                  excluded_energy_definition="100 * sum_excluded ||delta z||_2^2 / sum_all ||delta z||_2^2",
                  repetitions=repetitions)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Forcing folder, e.g. 0p15.")
    parser.add_argument("--rank", type=int, required=True, help="Shared POD rank.")
    parser.add_argument("--lags", nargs="+", type=int, required=True,
                        help="Native-step lags, e.g. 1 2 5 10 20 50.")
    parser.add_argument("--cutoff", type=float, default=0.,
                        help="Remove the largest P percent of increment norms (default 0).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--validation-reps", nargs="+", type=int)
    parser.add_argument("--test-reps", nargs="*", type=int)
    parser.add_argument("--output", type=Path, help="Override reduced_data/increments_analysis/... output.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(analyze_batch(args.experiment, args.rank, args.lags, args.cutoff,
                        data_root=args.data_root, shared_basis=args.shared_basis,
                        validation_reps=args.validation_reps, test_reps=args.test_reps,
                        output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
