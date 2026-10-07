"""Compare shared-POD rollout and reference capillary energies in joules.

    python -m data_analysis.energy.rollout_energy runs/0p20_r100/validation_rollout
    python -m data_analysis.energy.rollout_energy runs/0p20_r100/validation_rollout --exact

Default: quadratic (small-slope) capillary energy, using the existing gradient
stiffness matrix. --exact also evaluates gamma * integral(sqrt(1+|grad h|²)-1)
on every frame in bounded reconstruction batches. Uniform spatial mean has
zero gradient and contributes no deformation energy; coordinate 0 is omitted.
Removed temporal mean surfaces are not restored in either generated/reference
fields. Comparisons therefore describe the shared-rank fluctuating surface.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory
from .capillary_energy import capillary_stiffness, energy_series, energy_statistics, field_energies, length_scale


def compute_energies(coordinates, modes, x_m, y_m, z_scale, gamma, *, exact=False,
                     batch_size=32, stiffness=None, include_spatial_mean=True):
    if batch_size < 1 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Batch size and surface tension must be positive.")
    if stiffness is None:
        stiffness, _ = capillary_stiffness(modes, x_m, y_m, gamma=gamma)
    shape = coordinates.shape[:-1]
    flat = coordinates.reshape(-1, coordinates.shape[-1])
    quadratic = np.empty(len(flat))
    geometric = np.empty(len(flat)) if exact else None
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        coefficients_m = np.asarray(flat[start:stop, int(include_spatial_mean):], dtype=float) * z_scale
        quadratic[start:stop] = energy_series(coefficients_m, stiffness)
        if exact:
            field = (coefficients_m @ modes.reshape(len(modes), -1)).reshape(-1, len(y_m), len(x_m))
            _, value = field_energies(field, x_m, y_m)
            geometric[start:stop] = gamma * value
    result = {"quadratic": quadratic.reshape(shape)}
    if exact:
        result["exact"] = geometric.reshape(shape)
    return result


def analyze(rollout, *, gamma=.0728, exact=False, batch_size=32, output=None, overwrite=False):
    if batch_size < 1 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Batch size and surface tension must be positive.")
    x_m, y_m = rollout.x * length_scale(rollout.x_units), rollout.y * length_scale(rollout.y_units)
    stiffness, _ = capillary_stiffness(rollout.modes, x_m, y_m, gamma=gamma)
    options = dict(exact=exact, batch_size=batch_size, stiffness=stiffness,
                   include_spatial_mean=rollout.include_spatial_mean)
    generated = compute_energies(rollout.generated, rollout.modes, x_m, y_m, length_scale(rollout.units), gamma, **options)
    reference = compute_energies(rollout.reference, rollout.modes, x_m, y_m, length_scale(rollout.units), gamma, **options)
    report = rollout.provenance()
    report.update(energy_units="J", gamma_N_per_m=gamma, geometry="surface",
                  treatment="shared-rank fluctuations; spatial mean has zero deformation energy", energy={})
    arrays = dict(reference_time_seconds=rollout.time, generated_time_seconds=np.arange(rollout.generated.shape[2]) / rollout.fs)
    figure, axes = plt.subplots(len(generated), 1, figsize=(8, 3.5 * len(generated)), squeeze=False, constrained_layout=True)
    for axis, name in zip(axes[:, 0], generated):
        g, r = generated[name], reference[name]
        g_stats = [[energy_statistics(np.arange(g.shape[-1]) / rollout.fs, g[c, e]) for e in range(g.shape[1])] for c in range(g.shape[0])]
        r_stats = [energy_statistics(rollout.time[c], r[c]) for c in range(len(r))]
        g_mean = np.mean([s["time_averaged_energy"] for condition in g_stats for s in condition])
        r_mean = np.mean([s["time_averaged_energy"] for s in r_stats])
        report["energy"][name] = dict(generated=g_stats, reference=r_stats,
                                      generated_mean_J=float(g_mean), reference_mean_J=float(r_mean),
                                      generated_over_reference=None if r_mean == 0 else float(g_mean / r_mean))
        arrays.update({f"generated_{name}_J": g, f"reference_{name}_J": r})
        time = np.arange(g.shape[-1]) / rollout.fs
        for values, label, color in ((g, "Generated", "C0"), (r, "Reference", "C1")):
            values = values.reshape(-1, len(time))
            axis.plot(time, values.mean(0), label=label, color=color)
            lo, hi = np.quantile(values, [.1, .9], axis=0)
            axis.fill_between(time, lo, hi, color=color, alpha=.18)
        axis.set(title=f"{name.capitalize()} capillary energy", xlabel="Time since selected window start [s]", ylabel="Energy [J]")
        if not rollout.include_spatial_mean:
            axis.set_yscale("log")
        axis.legend()
    out = output_directory(rollout, "energy", output, overwrite)
    np.savez_compressed(out / "energy.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    figure.savefig(out / "energy.png", dpi=160)
    plt.close(figure)
    if not rollout.include_spatial_mean:
        selected = [i for i in (1, 2, 5, 10, 20, 50) if i <= rollout.rank]
        fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.1 * len(selected)),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        first_reference = rollout.reference[0]
        first_generated = rollout.generated[0, 0]
        time = np.arange(len(first_generated)) / rollout.fs
        for axis, mode in zip(axes, selected):
            axis.plot(time, first_reference[:, mode - 1], color="0.2", lw=.8,
                      label="measured" if mode == selected[0] else None)
            axis.plot(time, first_generated[:, mode - 1], color="C3", lw=.8,
                      label="ROM" if mode == selected[0] else None)
            axis.set_ylabel(f"POD {mode}")
            axis.grid(alpha=.2)
        axes[0].legend()
        axes[-1].set_xlabel("Time since selected window start [s]")
        fig.savefig(out / "traces.png", dpi=160)
        cut = min(101, len(time))
        axes[-1].set_xlim(time[0], time[cut - 1])
        for axis, mode in zip(axes, selected):
            values = np.r_[first_reference[:cut, mode - 1], first_generated[:cut, mode - 1]]
            low, high = float(np.min(values)), float(np.max(values))
            pad = max(.05 * (high - low), 1.)
            axis.set_ylim(low - pad, high + pad)
        fig.savefig(out / "traces_first_100.png", dpi=160)
        plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--gamma", type=float, default=.0728, help="Surface tension [N/m]; project default 0.0728.")
    parser.add_argument("--exact", action="store_true", help="Also compute geometric energy for every frame.")
    parser.add_argument("--batch-size", type=int, default=32, help="Frame batch size for energy computation.")
    args = parser.parse_args(argv)
    rollout = Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard)
    print(analyze(rollout, gamma=args.gamma, exact=args.exact, batch_size=args.batch_size,
                  output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
