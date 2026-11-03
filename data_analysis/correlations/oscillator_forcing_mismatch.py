"""Explain hybrid oscillator failures through the spectrum of its forcing.

This is a second-stage analysis of ``oscillator_conditional_compatibility``.
It compares the exact residual forcing required by measured trajectories with
the forcing produced by autonomous hybrid rollouts.  In particular, it tests
whether a scalar diffusion calibration can repair the model, or whether the
frequency distribution of the forcing is itself wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_toeplitz

from modelling.neural_sde.train import load_model
from .oscillator_conditional_compatibility import modal_sequences, spectral_budget


def interp_modes(frequency, spectrum, points):
    return np.asarray([
        np.interp(points[mode], frequency, spectrum[:, mode])
        for mode in range(len(points))
    ])


def local_factors(frequency, spectrum, poles, inner=50., outer=150.):
    factors = []
    for mode, pole in enumerate(poles):
        neighbors = ((np.abs(frequency - pole) >= inner)
                     & (np.abs(frequency - pole) <= outer))
        factors.append(np.interp(pole, frequency, spectrum[:, mode])
                       / np.median(spectrum[neighbors, mode]))
    return np.asarray(factors)


def neighborhood_ratios(generated_f, generated_s, reference_f, reference_s,
                        poles, half_width):
    ratios = []
    for mode, pole in enumerate(poles):
        generated = np.abs(generated_f - pole) <= half_width
        reference = np.abs(reference_f - pole) <= half_width
        ratios.append(
            np.trapezoid(generated_s[generated, mode], generated_f[generated])
            / np.trapezoid(reference_s[reference, mode], reference_f[reference]))
    return np.asarray(ratios)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(median=float(np.median(values)),
                q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)),
                minimum=float(values.min()), maximum=float(values.max()))


def truncated_spectral_factor(acf, poles, dt, lag):
    """PSD/white-level implied by the signed ACF through ``lag``."""
    lag = min(lag, len(acf) - 1)
    indices = np.arange(1, lag + 1)[:, None]
    phase = 2 * np.pi * poles[None, :] * dt
    return 1 + 2 * np.sum(acf[1:lag + 1] * np.cos(indices * phase), axis=0)


def ar_notch_order_scan(acf, poles, dt, orders):
    """Yule-Walker AR baselines showing the memory needed for local notches."""
    frequency = np.linspace(0, 1500, 12001)
    theta = 2 * np.pi * frequency * dt
    rows = []
    for order in orders:
        indices = np.arange(1, order + 1)
        factors = []
        pole_over_innovation = []
        for mode, pole in enumerate(poles):
            correlation = acf[:order + 1, mode]
            coefficients = solve_toeplitz(
                (correlation[:-1], correlation[:-1]), correlation[1:])
            denominator = np.abs(
                1 - np.exp(-1j * theta[:, None] * indices) @ coefficients
            ) ** 2
            spectrum = 1 / denominator
            at_pole = np.interp(pole, frequency, spectrum)
            neighbors = ((np.abs(frequency - pole) >= 50)
                         & (np.abs(frequency - pole) <= 150))
            factors.append(at_pole / np.median(spectrum[neighbors]))
            pole_over_innovation.append(at_pole)
        rows.append(dict(order=order,
                         pole_over_local=quantiles(factors),
                         pole_over_innovation=quantiles(pole_over_innovation)))
    return rows


def analyze_saved_scale(path, model, transform, poles, nperseg):
    with np.load(path / "rollout.npz", allow_pickle=False) as saved:
        trajectories = np.asarray(saved["trajectories"], dtype=float)
        references = np.asarray(saved["reference"], dtype=float)
    equilibrium = model.drift_model.equilibrium.detach().double().numpy()
    stiffness = model.drift_model.physical_stiffness.detach().double().numpy()
    damping = model.drift_model.physical_damping.detach().double().numpy()
    generated_spectra = []
    reference_spectra = []
    generated_variance = []
    reference_variance = []
    second_half_ratios = []
    frequency = None
    for condition in range(trajectories.shape[0]):
        reference = references[condition]
        reference_variance.append(np.var(reference, axis=0).sum())
        for path_index in range(trajectories.shape[1]):
            generated = trajectories[condition, path_index]
            generated_variance.append(np.var(generated, axis=0).sum())
            middle = len(generated) // 2
            second_half_ratios.append(
                np.var(generated[middle:], axis=0).sum()
                / np.var(reference[middle:], axis=0).sum())
            generated_sequence = modal_sequences(
                generated, transform, equilibrium, stiffness, damping, model.config.dt)
            reference_sequence = modal_sequences(
                reference, transform, equilibrium, stiffness, damping, model.config.dt)
            segment = min(nperseg, len(generated_sequence["q"]))
            frequency, generated_spectrum, _ = spectral_budget(
                generated_sequence, 1 / model.config.dt, segment)
            _, reference_spectrum, _ = spectral_budget(
                reference_sequence, 1 / model.config.dt, segment)
            generated_spectra.append(generated_spectrum)
            reference_spectra.append(reference_spectrum)
    generated = {key: np.mean([item[key] for item in generated_spectra], axis=0)
                 for key in generated_spectra[0]}
    reference = {key: np.mean([item[key] for item in reference_spectra], axis=0)
                 for key in reference_spectra[0]}
    q_ratio = (interp_modes(frequency, generated["q"], poles)
               / interp_modes(frequency, reference["q"], poles))
    forcing_ratio = (interp_modes(frequency, generated["residual"], poles)
                     / interp_modes(frequency, reference["residual"], poles))
    return dict(
        shape=list(trajectories.shape),
        physical_trace_ratio=float(np.mean(generated_variance)
                                   / np.mean(reference_variance)),
        physical_trace_ratio_second_half=float(np.mean(second_half_ratios)),
        q_pole_ratio=quantiles(q_ratio),
        forcing_pole_ratio=quantiles(forcing_ratio),
        forcing_pole_over_local=quantiles(local_factors(
            frequency, generated["residual"], poles)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    report = json.loads((args.audit / "report.json").read_text())
    checkpoint_path = args.checkpoint or Path(report["checkpoint"])
    arrays = np.load(args.audit / "spectral_budget.npz")
    poles = arrays["poles_hz"]
    transform = arrays["transform"]
    validation_f = arrays["validation_frequency_hz"]
    validation_e = arrays["validation_residual_psd"]
    validation_q = arrays["validation_q_psd"]
    model, _ = load_model(checkpoint_path, "cpu")
    torch.set_num_threads(2)

    result = dict(
        checkpoint=str(checkpoint_path.resolve()),
        definition=("Residual forcing is the exact modal velocity change left after "
                    "subtracting the frozen oscillator update."),
        validation={}, generated={}, diffusion={}, scale_controls={},
        forcing_memory_order_scan=[],
        caveats=[
            "Pointwise pole ratios compare density estimates with different Welch resolutions; the 50 and 100 Hz neighborhood integrals are more robust.",
            "The 5,000-step scale controls retain substantial information from their measured initial history.",
            "The 90,000-step scale control contains one path and is a mechanism diagnostic, not an uncertainty estimate.",
            "Diffusion scaling is inference-only and does not produce a retrained calibrated model.",
        ])

    validation_local = local_factors(validation_f, validation_e, poles)
    result["validation"] = dict(
        forcing_pole_over_local=quantiles(validation_local),
        acf_spectral_factor_lag63=quantiles(truncated_spectral_factor(
            arrays["validation_residual_acf"], poles, report["dt_seconds"], 63)),
        acf_spectral_factor_lag2000=quantiles(truncated_spectral_factor(
            arrays["validation_residual_acf"], poles, report["dt_seconds"], 2000)),
    )
    result["forcing_memory_order_scan"] = ar_notch_order_scan(
        arrays["train_residual_acf"], poles, report["dt_seconds"],
        (5, 10, 20, 32, 63, 100, 200, 500, 1000, 2000))

    validation_std = np.asarray(
        report["populations"]["validation"]["residual_modal_std"])
    for label, generated_report in report["generated"].items():
        prefix = f"generated_{label}"
        frequency = arrays[f"{prefix}_frequency_hz"]
        forcing = arrays[f"{prefix}_residual_psd"]
        q_spectrum = arrays[f"{prefix}_q_psd"]
        forcing_pole_ratio = (interp_modes(frequency, forcing, poles)
                              / interp_modes(validation_f, validation_e, poles))
        q_pole_ratio = (interp_modes(frequency, q_spectrum, poles)
                        / interp_modes(validation_f, validation_q, poles))
        generated_std = np.asarray(generated_report["residual_modal_std"])
        result["generated"][label] = dict(
            residual_std_ratio=quantiles(generated_std / validation_std),
            total_residual_power_ratio=float(
                np.trapezoid(forcing, frequency, axis=0).sum()
                / np.trapezoid(validation_e, validation_f, axis=0).sum()),
            forcing_pole_ratio=quantiles(forcing_pole_ratio),
            q_pole_ratio=quantiles(q_pole_ratio),
            forcing_neighborhood_50hz_ratio=quantiles(neighborhood_ratios(
                frequency, forcing, validation_f, validation_e, poles, 50)),
            forcing_neighborhood_100hz_ratio=quantiles(neighborhood_ratios(
                frequency, forcing, validation_f, validation_e, poles, 100)),
            forcing_pole_over_local=quantiles(local_factors(
                frequency, forcing, poles)),
            acf_spectral_factor_lag63=quantiles(truncated_spectral_factor(
                arrays[f"{prefix}_residual_acf"], poles,
                report["dt_seconds"], 63)),
        )

    inverse = np.linalg.inv(transform)
    diffusion = model.diffusion_matrix().detach().double().numpy()
    modal_covariance = inverse @ diffusion @ diffusion.T @ inverse.T
    white_floor = 2 * np.diag(modal_covariance) / report["sampling_frequency_hz"]
    observed_at_poles = interp_modes(validation_f, validation_e, poles)
    result["diffusion"] = dict(
        modal_noise_std=np.sqrt(np.diag(modal_covariance)).tolist(),
        white_noise_floor_over_measured_forcing_at_poles=quantiles(
            white_floor / observed_at_poles),
        common_amplitude_reduction_to_match_median_pole_floor=float(
            np.sqrt(np.median(observed_at_poles / white_floor))),
    )

    controls = {
        "scale_0": args.audit / "diffusion_scale_0",
        "scale_0p01": args.audit / "diffusion_scale_0p01",
        "scale_0p03": args.audit / "diffusion_scale_0p03",
        "scale_0p1": args.audit / "diffusion_scale_0p1",
        "scale_0p01_long": args.audit / "diffusion_scale_0p01_rollout_90000_1x1",
    }
    for label, path in controls.items():
        if (path / "rollout.npz").exists():
            result["scale_controls"][label] = analyze_saved_scale(
                path, model, transform, poles,
                16384 if label.endswith("long") else 4096)

    (args.audit / "forcing_mismatch.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    make_plot(args.audit, arrays, result)
    write_interpretation(args.audit, result)
    print(args.audit / "interpretation.md")


def make_plot(output, arrays, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poles = arrays["poles_hz"]
    validation_f = arrays["validation_frequency_hz"]
    validation_e = arrays["validation_residual_psd"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for mode, ax in enumerate(axes.flat):
        pole = poles[mode]
        mask = (validation_f >= max(0, pole - 200)) & (validation_f <= pole + 200)
        local = np.median(validation_e[
            (np.abs(validation_f - pole) >= 50)
            & (np.abs(validation_f - pole) <= 150), mode])
        ax.semilogy(validation_f[mask], validation_e[mask, mode] / local,
                    label="Measured required forcing", lw=1.5)
        for label, style in (("parent_e5", "--"), ("nll_e2", ":")):
            frequency = arrays[f"generated_{label}_frequency_hz"]
            forcing = arrays[f"generated_{label}_residual_psd"]
            generated_mask = ((frequency >= max(0, pole - 200))
                              & (frequency <= pole + 200))
            generated_local = np.median(forcing[
                (np.abs(frequency - pole) >= 50)
                & (np.abs(frequency - pole) <= 150), mode])
            ax.semilogy(frequency[generated_mask],
                        forcing[generated_mask, mode] / generated_local,
                        style, label=label)
        ax.axvline(pole, color=".4", lw=.7)
        ax.set(title="Spatial mean direction" if mode == 0 else f"Oscillator mode {mode}",
               xlabel="Frequency [Hz]", ylabel="Forcing PSD / local background")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "forcing_notches.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             layout="constrained")
    for label, values in result["generated"].items():
        frequency = arrays[f"generated_{label}_frequency_hz"]
        forcing = arrays[f"generated_{label}_residual_psd"]
        q_spectrum = arrays[f"generated_{label}_q_psd"]
        forcing_ratio = (interp_modes(frequency, forcing, poles)
                         / interp_modes(validation_f, validation_e, poles))
        q_ratio = (interp_modes(frequency, q_spectrum, poles)
                   / interp_modes(validation_f, arrays["validation_q_psd"], poles))
        axes[0].semilogy(poles, forcing_ratio, "o-", label=label)
        axes[1].semilogy(poles, q_ratio, "o-", label=label)
    axes[0].set(ylabel="Generated / measured forcing PSD",
                title="Absolute mismatch at fitted poles")
    axes[1].set(xlabel="Fitted peak [Hz]",
                ylabel="Generated / measured modal PSD")
    for ax in axes:
        ax.axhline(1, color="k", lw=.7)
        ax.legend(fontsize=8)
    fig.savefig(output / "forcing_pole_mismatch.png", dpi=180)
    plt.close(fig)

    rows = result["forcing_memory_order_scan"]
    orders = np.asarray([row["order"] for row in rows])
    median = np.asarray([row["pole_over_local"]["median"] for row in rows])
    q25 = np.asarray([row["pole_over_local"]["q25"] for row in rows])
    q75 = np.asarray([row["pole_over_local"]["q75"] for row in rows])
    observed = result["validation"]["forcing_pole_over_local"]["median"]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.loglog(orders, median, "o-", label="Yule-Walker AR fit")
    ax.fill_between(orders, q25, q75, alpha=.2, label="Mode quartiles")
    ax.axhline(observed, color="k", ls="--", lw=1,
               label="Measured forcing notch")
    ax.axvline(63, color=".4", ls=":", lw=1, label="Current TCN context")
    ax.set(xlabel="Forcing-history order [native frames]",
           ylabel="Median pole / local-background forcing PSD",
           title="Narrow forcing notches require long memory")
    ax.legend()
    fig.savefig(output / "forcing_memory_order.png", dpi=180)
    plt.close(fig)


def write_interpretation(output, result):
    validation = result["validation"]
    parent = result["generated"]["parent_e5"]
    diffusion = result["diffusion"]
    controls = result["scale_controls"]
    lines = [
        "# Why the forcing-history hybrid still fails", "",
        "[Forcing notches](forcing_notches.png) · "
        "[Pole mismatch](forcing_pole_mismatch.png) · "
        "[Memory order](forcing_memory_order.png) · "
        "[Full numbers](forcing_mismatch.json) · "
        "[Conditional budget](README.md)", "",
        "## Finding", "",
        "The frozen oscillator is conditionally compatible with the measured peak motion. "
        "The failure comes from the spectrum of the forcing placed on top of it, rather than "
        "from a neural residual that cancels the core at the fitted resonances.", "",
        f"Measured required forcing has deep notches at the 20 fitted frequencies: median "
        f"pole/local-background power is **{validation['forcing_pole_over_local']['median']:.4g}**. "
        f"The epoch-5 rollout has no such notches: **{parent['forcing_pole_over_local']['median']:.4g}**. "
        f"Its total residual power is only **{parent['total_residual_power_ratio']:.3g}x** measured, "
        f"but its median pole power is **{parent['forcing_pole_ratio']['median']:.0f}x** measured and "
        f"its integrated power in +/-100 Hz pole neighborhoods is **"
        f"{parent['forcing_neighborhood_100hz_ratio']['median']:.0f}x** measured.", "",
        f"The trained constant diffusion by itself has a median white-noise floor **"
        f"{diffusion['white_noise_floor_over_measured_forcing_at_poles']['median']:.0f}x** above the "
        "measured forcing at the poles. The TCN would therefore need to turn approximately white "
        "one-step innovations into a sharply notched colored process during autonomous recursion. "
        "One-step likelihood rewards conditional prediction and innovation variance, but does not "
        "make that autonomous spectral factorization reliable.", "",
        "This notch is a structural consequence of using one observed state. In oscillator modal "
        "coordinate `j`, the discrete equation can be written `A_j(L) q_j = e_j`, so "
        "`S_e,j(f) = |A_j(exp(-i 2 pi f dt))|^2 S_q,j(f)`. The oscillator operator is smallest near "
        "its resonance. Therefore the forcing that produces a broadband background in `q_j` must be "
        "suppressed there; otherwise that background is resonantly amplified as well.", "",
        "## The two hybrid failures are complementary", "",
        "The [earlier raw-state residual audit](../imported_oscillator_contribution_0p10_v1/README.md) "
        "found that an unrestricted TCN added roughly 100 times the fitted modal damping and erased the "
        "peaks. The present core-innovation input prevents that direct state feedback, but leaves a "
        "second failure: short-memory forcing plus constant white diffusion has no resonance notches and "
        "overexcites the preserved high-Q core. This explains why the baseline TCN can match broadband "
        "shape under stronger effective damping while the same one-step calibration is incompatible with "
        "the imported oscillator.", "",
        "## What the history result means", "",
        f"Through 63 frames, the measured signed ACF gives a pole/white spectral factor of "
        f"**{validation['acf_spectral_factor_lag63']['median']:.4g}**, while the generated process gives "
        f"**{parent['acf_spectral_factor_lag63']['median']:.4g}**. That comparison captures broad "
        "suppression but does not establish a narrow local notch.", "",
        "A Yule-Walker baseline makes the resolution requirement explicit:", "",
        "| History order | Median pole/local-background PSD |", "|---:|---:|",
    ]
    for row in result["forcing_memory_order_scan"]:
        if row["order"] in {20, 63, 200, 500, 1000, 2000}:
            lines.append(f"| {row['order']} | {row['pole_over_local']['median']:.4g} |")
    lines += ["",
        f"The measured value is **{validation['forcing_pole_over_local']['median']:.4g}**. "
        "The small long-lag correlations are individually negligible but add coherently over many "
        "cycles. The 63-frame TCN is therefore too short for the sharp separation demanded by this "
        "parameterization. A context of 1000-2000 frames starts to resolve it, but would be expensive "
        "and would retain the autonomous teacher-forcing problem.", "",
        "## Diffusion controls", "",
    ]
    for label, values in controls.items():
        lines.append(
            f"- `{label}`: physical covariance-trace ratio **{values['physical_trace_ratio']:.3g}**; "
            f"second-half ratio **{values['physical_trace_ratio_second_half']:.3g}**; "
            f"median modal pole ratio **{values['q_pole_ratio']['median']:.3g}**.")
    lines += ["",
        "The short controls show that reducing diffusion restores bounded, visibly resonant motion. "
        "The long 0.01 control loses much of its initial energy and remains underforced in its second "
        "half, while its forcing still lacks the measured notches. A scalar diffusion rescaling can "
        "move total amplitude but cannot match broadband power and resonant excitation together.", "",
        "## Recommended model change", "",
        "Keep the frozen oscillator and the ordinary observed POD state. Give the neural branch an "
        "explicit forcing contract. In oscillator modal coordinates use `e = e_peak + H_notch b_theta`, "
        "where `e_peak` is a small unfiltered excitation that sustains the resonances and `H_notch` "
        "contains one stable second-order band-stop section per modal frequency. Apply the same "
        "background filter to its stochastic innovation. Initialize notch depth and width from the "
        "measured required-forcing spectrum, then freeze them for the proof of concept or allow tightly "
        "bounded refinement. The TCN models `b_theta`, the remaining broadband colored forcing; it no "
        "longer has to discover twenty sharp zeros through teacher-forced recursion.", "",
        "This is the single-state form of the necessary factorization. If the notch were chosen exactly "
        "as the oscillator difference operator `A`, then `A q = e_peak + A b_theta`, hence "
        "`q = A^-1 e_peak + b_theta`. Any same-state residual that preserves the peaks while adding an "
        "independent broadband component must realize approximately this cancellation, explicitly or "
        "implicitly. The current TCN was being asked to learn it implicitly.", "",
        "Train the one-step conditional likelihood as before, but evaluate the filtered forcing "
        "recursively. Add a cheap forcing-domain target: match the pole/local-background ratios and "
        "a sparse set of signed residual correlations extending through the notch time scale. This "
        "avoids differentiating a long state rollout. "
        "Use a modest 500-2000-step covariance or energy check only for amplitude calibration.", "",
        "Before implementing the filter, a smaller ablation can fix diffusion near 1-2% of its current "
        "amplitude and retrain the TCN. It tests whether deterministic forcing can rebuild broadband variance, but it "
        "should be treated as a mechanism check because scalar calibration cannot create the missing "
        "frequency-selective notches.", "",
        "No checkpoint was modified or retrained in this analysis. Scale controls change diffusion "
        "only during rollout; the long control contains one path."
    ]
    (output / "interpretation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
