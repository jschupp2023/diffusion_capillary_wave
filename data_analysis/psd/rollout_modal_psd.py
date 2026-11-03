"""Compare generated and reference Welch PSDs for individual POD coordinates."""
import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory
from data_analysis.rollout_metrics import _band_metrics
from .rollout_psd import ensemble_psd


def _validated_modes(rank, modes):
    selected = tuple(range(1, rank + 1)) if modes is None else tuple(modes)
    if (not selected or any(not isinstance(mode, int) or mode < 1 or mode > rank
                            for mode in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError(f"POD modes must be unique integers between 1 and {rank}.")
    return selected


def _quick_summary(report):
    rows = sorted(report["modes"], key=lambda row: row[
        "band_power_comparison"]["mean_absolute_log10_band_power_error_decades"],
        reverse=True)
    lines = [
        "# Modal PSD: quick summary",
        "",
        "Each POD coefficient is transformed separately for every trajectory. The table is "
        "ordered from the largest to the smallest half-decade band-power mismatch.",
        "",
        "| POD mode | PSD error [decades] | Approx. factor | Total power ratio |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        error = row["band_power_comparison"][
            "mean_absolute_log10_band_power_error_decades"]
        ratio = row["generated_over_reference_power"]
        ratio_text = "undefined" if ratio is None else f"{ratio:.3g}"
        lines.append(
            f"| {row['mode']} | {error:.3g} | {10 ** error:.3g}x | {ratio_text}x |")
    lines.extend([
        "",
        "PSD error is the equal-weight mean absolute log10 error of integrated band "
        "powers and has no over/under direction. Total power ratio is generated/reference.",
        "PSDs use complete, uniformly sampled trajectories without trimming.",
    ])
    return "\n".join(lines) + "\n"


def analyze(rollout, *, modes=None, nperseg=1024, overlap=.5,
            lower_frequency=10., bins_per_decade=2, output=None,
            overwrite=False):
    if nperseg < 4 or not 0 <= overlap < 1 or lower_frequency <= 0:
        raise ValueError(
            "Modal PSD settings require nperseg >= 4, overlap in [0, 1), and "
            "positive lower frequency.")
    if bins_per_decade not in (1, 2):
        raise ValueError("Bins per decade must be 1 or 2.")
    selected_modes = _validated_modes(rollout.rank, modes)
    offset = int(rollout.include_spatial_mean)

    generated_psds, reference_psds, rows = [], [], []
    frequency = None
    segment = n_overlap = None
    for mode in selected_modes:
        coordinate = offset + mode - 1
        frequency, generated, segment, n_overlap = ensemble_psd(
            rollout.generated[..., coordinate], rollout.fs, nperseg, overlap)
        _, reference, _, _ = ensemble_psd(
            rollout.reference[..., coordinate], rollout.fs, nperseg, overlap)
        generated_psds.append(generated)
        reference_psds.append(reference)
        spacing = frequency[1] - frequency[0]
        generated_power = generated.sum(axis=-1) * spacing
        reference_power = reference.sum(axis=-1) * spacing
        reference_mean_power = float(reference_power.mean())
        rows.append(dict(
            mode=mode,
            coordinate_index=coordinate,
            generated_mean_power=float(generated_power.mean()),
            reference_mean_power=reference_mean_power,
            generated_over_reference_power=(
                None if reference_mean_power == 0 else
                float(generated_power.mean() / reference_mean_power)),
            band_power_comparison=_band_metrics(
                frequency, generated, reference, lower_frequency, bins_per_decade),
        ))

    generated_psds = np.stack(generated_psds)
    reference_psds = np.stack(reference_psds)
    report = rollout.provenance()
    report.update(
        purpose="per-mode temporal-spectrum diagnostic",
        estimator=("Welch: Hann window, constant detrend, density scaling; each trajectory "
                   "transformed separately and trajectory PSDs averaged for metrics"),
        trimming="none; PSD requires complete uniformly sampled trajectories",
        density_units=f"{rollout.units}^2/Hz",
        nperseg=int(segment), noverlap=int(n_overlap),
        frequency_spacing_hz=float(frequency[1] - frequency[0]),
        selected_modes=list(selected_modes), modes=rows,
    )

    columns = min(4, len(selected_modes))
    plot_rows = math.ceil(len(selected_modes) / columns)
    figure, axes = plt.subplots(
        plot_rows, columns, figsize=(4.1 * columns, 3.1 * plot_rows),
        squeeze=False, constrained_layout=True)
    positive = frequency > 0
    tiny = np.finfo(float).tiny
    for axis, mode, generated, reference, row in zip(
            axes.flat, selected_modes, generated_psds, reference_psds, rows):
        for values, label, color in (
                (generated, "Generated", "C0"), (reference, "Reference", "C1")):
            flattened = values.reshape(-1, len(frequency))
            average = flattened.mean(axis=0)
            lo, hi = np.quantile(flattened, [.1, .9], axis=0)
            axis.loglog(frequency[positive], np.maximum(average[positive], tiny),
                        color=color, label=label, linewidth=1.)
            axis.fill_between(
                frequency[positive], np.maximum(lo[positive], tiny),
                np.maximum(hi[positive], tiny), color=color, alpha=.15)
        error = row["band_power_comparison"][
            "mean_absolute_log10_band_power_error_decades"]
        ratio = row["generated_over_reference_power"]
        ratio_text = "undefined" if ratio is None else f"{ratio:.2g}x"
        axis.set_title(f"POD {mode} | power {ratio_text} | PSD error {error:.2g} dec")
        axis.grid(True, which="both", alpha=.2)
        axis.set_xlabel("Frequency [Hz]")
        axis.set_ylabel(f"PSD [{rollout.units}$^2$/Hz]")
    for axis in axes.flat[len(selected_modes):]:
        axis.remove()
    axes.flat[0].legend(frameon=False, fontsize=9)

    out = output_directory(rollout, "modal_psd", output, overwrite)
    np.savez_compressed(
        out / "psd.npz", frequency_hz=frequency,
        modes=np.asarray(selected_modes, dtype=np.int64),
        generated_psd=generated_psds, reference_psd=reference_psds)
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "quick_summary.md").write_text(_quick_summary(report))
    figure.savefig(out / "modal_psd.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--modes", nargs="+", type=int,
                        help="POD mode numbers; default: all modes.")
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=.5)
    parser.add_argument("--psd-lower-frequency", type=float, default=10.)
    parser.add_argument("--bins-per-decade", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    rollout = Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard)
    print(analyze(
        rollout, modes=args.modes, nperseg=args.nperseg, overlap=args.overlap,
        lower_frequency=args.psd_lower_frequency,
        bins_per_decade=args.bins_per_decade,
        output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
