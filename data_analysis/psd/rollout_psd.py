"""Compare rollout/reference center-point and available spatial-mean PSDs using Welch.

    python -m data_analysis.psd.rollout_psd runs/0p20_r100/validation_rollout

Each trajectory is transformed separately. Plots show ensemble mean and
10–90% trajectory spread (not confidence intervals). Reference windows are
not duplicated for each generated ensemble member. Constant temporal mean
fields remain omitted, consistently with the saved reference coefficients.
POD-only rollouts provide the center fluctuation PSD without a spatial mean.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory
from .pod_center_psd import compute_welch, select_point


def point_signals(coordinates, modes, n_space, point):
    spatial_mean = coordinates[..., 0] / np.sqrt(n_space)
    fluctuation = coordinates[..., 1:] @ modes[:, point.y_index, point.x_index]
    return dict(center=fluctuation + spatial_mean,
                center_without_spatial_mean=fluctuation, spatial_mean=spatial_mean)


def fluctuation_signals(coordinates, modes, point):
    return dict(center_without_spatial_mean=coordinates @ modes[:, point.y_index, point.x_index])


def ensemble_psd(signals, fs, nperseg, overlap):
    densities = []
    for signal in signals.reshape(-1, signals.shape[-1]):
        result, segment, n_overlap = compute_welch(signal, fs, nperseg, overlap)
        densities.append(result.density)
    return result.frequency, np.stack(densities).reshape(*signals.shape[:-1], -1), segment, n_overlap


def analyze(rollout, *, point=None, nperseg=1024, overlap=.5, output=None, overwrite=False):
    if nperseg < 4 or not 0 <= overlap < 1:
        raise ValueError("nperseg must be >=4 and overlap in [0, 1).")
    selected = select_point(rollout.modes.shape[1:], rollout.x, rollout.y,
                            rollout.x_units, rollout.y_units, point, None)
    if rollout.include_spatial_mean:
        generated = point_signals(rollout.generated, rollout.modes, rollout.n_space, selected)
        reference = point_signals(rollout.reference, rollout.modes, rollout.n_space, selected)
    else:
        generated = fluctuation_signals(rollout.generated, rollout.modes, selected)
        reference = fluctuation_signals(rollout.reference, rollout.modes, selected)
    report = rollout.provenance()
    report.update(point=dict(y_index=selected.y_index, x_index=selected.x_index,
                             x=selected.x_coordinate, y=selected.y_coordinate),
                  estimator="Welch: Hann window, constant detrend, density scaling",
                  density_units=f"{rollout.units}^2/Hz", signals={},
                  spatial_mean_available=rollout.include_spatial_mean)
    arrays = {}
    figure, axes = plt.subplots(len(generated), 1, figsize=(8, 3.5 * len(generated)),
                               squeeze=False, constrained_layout=True)
    for axis, name in zip(axes[:, 0], generated):
        frequency, g, segment, n_overlap = ensemble_psd(generated[name], rollout.fs, nperseg, overlap)
        _, r, _, _ = ensemble_psd(reference[name], rollout.fs, nperseg, overlap)
        arrays.update({f"generated_{name}": g, f"reference_{name}": r})
        for values, label, color in ((g, "Generated", "C0"), (r, "Reference", "C1")):
            values = values.reshape(-1, len(frequency))
            positive = frequency > 0
            average = values.mean(0)
            axis.loglog(frequency[positive], np.maximum(average[positive], np.finfo(float).tiny), label=label, color=color)
            lo, hi = np.quantile(values, [.1, .9], axis=0)
            axis.fill_between(frequency[positive], np.maximum(lo[positive], np.finfo(float).tiny),
                              np.maximum(hi[positive], np.finfo(float).tiny), color=color, alpha=.18)
        # Density bins are discrete; their sum times df is integrated Welch power.
        g_power, r_power = g.sum(-1) * (frequency[1] - frequency[0]), r.sum(-1) * (frequency[1] - frequency[0])
        report["signals"][name] = dict(generated_mean_power=float(g_power.mean()), reference_mean_power=float(r_power.mean()),
                                       generated_over_reference=None if r_power.mean() == 0 else float(g_power.mean() / r_power.mean()))
        axis.set(title=name.replace("_", " "), xlabel="Frequency [Hz]", ylabel=f"PSD [{rollout.units}²/Hz]")
        axis.legend()
    report.update(nperseg=segment, noverlap=n_overlap, frequency_spacing_hz=float(frequency[1] - frequency[0]))
    out = output_directory(rollout, "psd", output, overwrite)
    np.savez_compressed(out / "psd.npz", frequency_hz=frequency, **arrays)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    figure.savefig(out / "psd.png", dpi=160)
    plt.close(figure)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--point", nargs=2, type=int, metavar=("Y", "X"), help="Pixel indices; default grid center.")
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=.5)
    args = parser.parse_args(argv)
    rollout = Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard)
    print(analyze(rollout, point=args.point, nperseg=args.nperseg, overlap=args.overlap,
                  output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
