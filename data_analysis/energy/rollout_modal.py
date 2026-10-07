"""Attribute rollout amplitude and quadratic capillary energy mode by mode.

The amplitude diagnostic is

    <a_i^2>_generated / <a_i^2>_reference,

whose square root is the generated/reference RMS-amplitude ratio.  The exact
additive attribution of the quadratic energy is 0.5*a_i*(K@a)_i; unlike the
diagonal self-energy, it includes cross terms and sums to 0.5*a.T@K@a.

When the checkpoint records a training-increment trim, the primary plot and
summary compare complete generated trajectories against reference frames
retained by that exact rule. The untrimmed comparison is retained explicitly.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory
from data_analysis.rollout_metrics import reference_training_trim_population
from .capillary_energy import capillary_stiffness, length_scale


def _ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator != 0)


def _moments(coordinates, keep_mask=None, chunk_size=65536):
    """Accumulate first/second moments in float64 with bounded temporaries."""
    flat = coordinates.reshape(-1, coordinates.shape[-1])
    mask = None if keep_mask is None else np.asarray(keep_mask, dtype=bool).reshape(-1)
    if mask is not None and len(mask) != len(flat):
        raise ValueError("Moment mask must match all non-coordinate sample axes.")
    total = np.zeros(flat.shape[1], dtype=float)
    second = np.zeros((flat.shape[1], flat.shape[1]), dtype=float)
    count = 0
    for start in range(0, len(flat), chunk_size):
        batch = np.asarray(flat[start:start + chunk_size], dtype=float)
        if mask is not None:
            batch = batch[mask[start:start + chunk_size]]
        if not len(batch):
            continue
        if not np.isfinite(batch).all():
            raise ValueError("Rollout coordinates must be finite.")
        total += batch.sum(axis=0)
        second += batch.T @ batch
        count += len(batch)
    if count < 1:
        raise ValueError("Moment mask retained no samples.")
    mean = total / count
    second /= count
    variance = np.maximum(np.diag(second) - mean**2, 0)
    return mean, variance, second


def modal_diagnostics(generated, reference, stiffness, coefficient_scale=1.,
                      reference_mask=None):
    """Return ensemble/sample-averaged POD amplitude and energy diagnostics.

    ``generated`` is [condition, ensemble, time, mode] and ``reference`` is
    [condition, time, mode]. The arrays must contain POD coefficients only,
    without the optional spatial-mean coordinate.
    """
    generated, reference = np.asarray(generated), np.asarray(reference)
    stiffness = np.asarray(stiffness, dtype=float)
    if (generated.ndim != 4 or reference.ndim != 3
            or generated.shape[0] != reference.shape[0]
            or generated.shape[2:] != reference.shape[1:]):
        raise ValueError("Expected generated [condition, ensemble, time, mode] and matching reference.")
    rank = generated.shape[-1]
    if stiffness.shape != (rank, rank) or not np.isfinite(stiffness).all():
        raise ValueError("Stiffness must be a finite square matrix matching the POD rank.")
    if not np.isfinite(coefficient_scale) or coefficient_scale <= 0:
        raise ValueError("Coefficient scale must be positive and finite.")

    if (reference_mask is not None
            and np.asarray(reference_mask).shape != reference.shape[:-1]):
        raise ValueError("Reference mask must match condition and time axes.")
    generated_mean, generated_variance, generated_second_moment = _moments(generated)
    reference_mean, reference_variance, reference_second_moment = _moments(
        reference, reference_mask)
    generated_mean_square = np.diag(generated_second_moment)
    reference_mean_square = np.diag(reference_second_moment)
    mean_square_ratio = _ratio(generated_mean_square, reference_mean_square)

    # E[0.5*a_i*(K@a)_i] = 0.5*sum_j K_ij E[a_i*a_j].  Working
    # from the small second-moment matrices avoids rollout-sized temporaries.
    scale_squared = coefficient_scale**2
    generated_energy = .5 * scale_squared * np.sum(stiffness * generated_second_moment, axis=1)
    reference_energy = .5 * scale_squared * np.sum(stiffness * reference_second_moment, axis=1)
    energy_excess = generated_energy - reference_energy
    total_excess = float(energy_excess.sum())

    return dict(
        generated_mean=generated_mean,
        reference_mean=reference_mean,
        generated_variance=generated_variance,
        reference_variance=reference_variance,
        variance_ratio=_ratio(generated_variance, reference_variance),
        generated_mean_square=generated_mean_square,
        reference_mean_square=reference_mean_square,
        mean_square_ratio=mean_square_ratio,
        rms_amplitude_ratio=np.sqrt(mean_square_ratio),
        generated_energy_attribution_J=generated_energy,
        reference_energy_attribution_J=reference_energy,
        energy_excess_attribution_J=energy_excess,
        excess_fraction=_ratio(energy_excess, np.full(rank, total_excess)),
    )


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _comparison_report(values, rank):
    generated_total = float(values["generated_energy_attribution_J"].sum())
    reference_total = float(values["reference_energy_attribution_J"].sum())
    excess_total = generated_total - reference_total
    order = np.argsort(np.abs(values["energy_excess_attribution_J"]))[::-1]
    return dict(
        generated_mean_energy_J=generated_total,
        reference_mean_energy_J=reference_total,
        generated_over_reference=(
            None if reference_total == 0 else generated_total / reference_total),
        total_energy_excess_J=excess_total,
        modes=[dict(
            mode=int(i + 1),
            mean_square_ratio=_finite_or_none(values["mean_square_ratio"][i]),
            rms_amplitude_ratio=_finite_or_none(values["rms_amplitude_ratio"][i]),
            generated_mean=float(values["generated_mean"][i]),
            reference_mean=float(values["reference_mean"][i]),
            generated_variance=float(values["generated_variance"][i]),
            reference_variance=float(values["reference_variance"][i]),
            variance_ratio=_finite_or_none(values["variance_ratio"][i]),
            generated_energy_attribution_J=float(
                values["generated_energy_attribution_J"][i]),
            reference_energy_attribution_J=float(
                values["reference_energy_attribution_J"][i]),
            energy_excess_attribution_J=float(
                values["energy_excess_attribution_J"][i]),
            excess_fraction=_finite_or_none(values["excess_fraction"][i]),
        ) for i in range(rank)],
        modes_by_absolute_energy_excess=[int(i + 1) for i in order],
    )


def analyze(rollout, *, gamma=.0728, output=None, overwrite=False):
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Surface tension must be positive and finite.")
    offset = int(rollout.include_spatial_mean)
    generated = rollout.generated[..., offset:]
    reference = rollout.reference[..., offset:]
    x_m = rollout.x * length_scale(rollout.x_units)
    y_m = rollout.y * length_scale(rollout.y_units)
    stiffness, _ = capillary_stiffness(rollout.modes, x_m, y_m, gamma=gamma)
    coefficient_scale = length_scale(rollout.units)
    untrimmed_values = modal_diagnostics(
        generated, reference, stiffness, coefficient_scale)
    reference_mask, trim_report = reference_training_trim_population(rollout)
    values = (untrimmed_values if reference_mask is None else
              modal_diagnostics(generated, reference, stiffness, coefficient_scale,
                                reference_mask=reference_mask))
    modes = np.arange(1, rollout.rank + 1)

    primary = _comparison_report(values, rollout.rank)
    untrimmed = _comparison_report(untrimmed_values, rollout.rank)
    excess_total = primary["total_energy_excess_J"]
    comparison = ("untrimmed" if reference_mask is None else
                  "generated_untrimmed_vs_reference_training_trimmed")
    report = rollout.provenance()
    report.update(
        amplitude_definition=(
            "mean square is averaged over conditions, ensembles, and time; "
            "RMS ratio = sqrt(generated/reference mean square); generated "
            "trajectories are never trimmed"),
        energy_definition="quadratic capillary energy; modal attribution = 0.5 * a_i * (K @ a)_i and sums exactly to total energy",
        energy_units="J",
        gamma_N_per_m=gamma,
        comparison_used_for_plot_and_primary_fields=comparison,
        reference_training_trim=trim_report,
        untrimmed=untrimmed,
        generated_untrimmed_vs_reference_training_trimmed=(
            None if reference_mask is None else dict(
                **primary, reference_trim=trim_report)),
        **primary,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    if reference_mask is not None:
        axes[0].plot(modes, untrimmed_values["rms_amplitude_ratio"], "o--",
                     ms=3, lw=1, color="0.55", label="Untrimmed reference")
    axes[0].plot(
        modes, values["rms_amplitude_ratio"], "o-", ms=3, lw=1, color="C3",
        label=("Training-trimmed reference" if reference_mask is not None
               else "Reference"))
    axes[0].axhline(1, color="0.25", lw=1, ls="--")
    axes[0].set(ylabel="RMS generated / reference", title="Per-mode coefficient amplitude")
    axes[0].legend()

    width = .4
    axes[1].bar(modes - width / 2, values["reference_energy_attribution_J"], width,
                label=("Reference (training-trimmed)" if reference_mask is not None
                       else "Reference"), color="C1")
    axes[1].bar(modes + width / 2, values["generated_energy_attribution_J"], width,
                label="Generated", color="C0")
    axes[1].set(ylabel="Mean attributed energy [J]", title="Additive quadratic-energy attribution")
    axes[1].legend()

    colors = np.where(values["energy_excess_attribution_J"] >= 0, "C3", "C0")
    axes[2].bar(modes, values["energy_excess_attribution_J"], color=colors)
    axes[2].axhline(0, color="0.25", lw=.8)
    axes[2].set(xlabel="POD mode", ylabel="Generated - reference [J]",
                title=f"Modal energy excess (sum = {excess_total:.3g} J)")
    for axis in axes:
        axis.set_xlim(.25, rollout.rank + .75)
        axis.grid(axis="y", alpha=.2)
    if trim_report is not None:
        fig.suptitle(
            "Generated trajectories untrimmed; reference uses training-increment "
            f"trim ({trim_report['removed_reference_frame_percent']:.2f}% frames removed)")

    out = output_directory(rollout, "modal", output, overwrite)
    arrays = dict(mode=modes, **values)
    if reference_mask is not None:
        arrays.update({f"untrimmed_{name}": value
                       for name, value in untrimmed_values.items()})
        arrays["reference_training_trim_mask"] = reference_mask
    np.savez_compressed(out / "modal_diagnostics.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig.savefig(out / "modal_diagnostics.png", dpi=160)
    plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--gamma", type=float, default=.0728,
                        help="Surface tension [N/m]; project default 0.0728.")
    args = parser.parse_args(argv)
    rollout = Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard)
    print(analyze(rollout, gamma=args.gamma, output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
