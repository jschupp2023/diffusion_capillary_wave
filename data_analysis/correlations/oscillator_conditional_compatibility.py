"""Audit whether a PSD-fitted oscillator is compatible with conditional dynamics.

For native positions x[n], define u[n] = x[n]-x[n-1] and split the measured
velocity change into the frozen imported core and its exactly required residual:

    a[n] = u[n+1]-u[n] = c[n] + e[n]
    c[n] = -dt^2 K (x[n]-x_eq) - dt D u[n].

The diagnostic works in the fitted oscillator modal coordinates, averages
spectra across independent blocks without concatenating repetitions, and reports
the signed core/residual cross-spectrum.  Frequency-domain associations are
descriptive least-squares quantities, not causal drift identification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import csd, welch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.neural_sde.train import load_model


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def modal_sequences(values, transform, equilibrium, stiffness, damping, dt):
    """Return q,u,a,c,e at centers n=1..N-2 in fitted modal coordinates."""
    values = np.asarray(values, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    q_all = (values - equilibrium) @ inverse.T
    q = q_all[1:-1]
    u = q_all[1:-1] - q_all[:-2]
    a = q_all[2:] - 2 * q_all[1:-1] + q_all[:-2]
    physical_u = values[1:-1] - values[:-2]
    physical_core = (-dt**2 * (values[1:-1] - equilibrium) @ stiffness.T
                     -dt * physical_u @ damping.T)
    core = physical_core @ inverse.T
    residual = a - core
    return dict(q=q, u=u, acceleration=a, core=core, residual=residual)


def spectral_budget(sequences, fs, nperseg):
    """One-sided modal auto/cross spectra with A=C+E budget convention."""
    segment = min(int(nperseg), len(sequences["q"]))
    if segment < 16:
        raise ValueError("Spectral blocks require at least 16 samples")
    overlap = segment // 2
    options = dict(fs=fs, nperseg=segment, noverlap=overlap,
                   detrend="constant", axis=0, scaling="density")
    result = {}
    frequency = None
    for name in ("q", "u", "acceleration", "core", "residual"):
        frequency, result[name] = welch(sequences[name], **options)
    _, result["core_residual"] = csd(
        sequences["core"], sequences["residual"], **options)
    _, result["q_residual"] = csd(
        sequences["q"], sequences["residual"], **options)
    _, result["u_residual"] = csd(
        sequences["u"], sequences["residual"], **options)
    windows = 1 + (len(sequences["q"]) - segment) // (segment - overlap)
    return frequency, result, windows


def autocorrelation(values, max_lag):
    """Unbiased per-coordinate ACF using FFT, with no cross-block pairs."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    lag = min(int(max_lag), n - 1)
    centered = values - values.mean(axis=0)
    size = 1 << int(math.ceil(math.log2(2 * n - 1)))
    transform = np.fft.rfft(centered, n=size, axis=0)
    covariance = np.fft.irfft(transform.conj() * transform, n=size, axis=0)[:lag + 1]
    covariance /= (n - np.arange(lag + 1))[:, None]
    scale = np.maximum(covariance[0], np.finfo(float).tiny)
    return covariance / scale


class Accumulator:
    def __init__(self, max_lag):
        self.max_lag = max_lag
        self.frequency = None
        self.spectra = {}
        self.spectral_weight = 0
        self.acf_sum = None
        self.acf_weight = 0
        self.blocks = []
        self.residual_sum = None
        self.residual_square_sum = None
        self.samples = 0

    def add(self, sequences, fs, nperseg, population):
        weight = 0
        if len(sequences["q"]) >= nperseg:
            frequency, spectra, weight = spectral_budget(sequences, fs, nperseg)
            if self.frequency is None:
                self.frequency = frequency
                self.spectra = {key: weight * value for key, value in spectra.items()}
            else:
                np.testing.assert_allclose(self.frequency, frequency)
                for key, value in spectra.items():
                    self.spectra[key] += weight * value
            self.spectral_weight += weight
        acf = autocorrelation(sequences["residual"], self.max_lag)
        acf_weight = len(sequences["residual"]) - np.arange(len(acf))
        if self.acf_sum is None:
            self.acf_sum = acf * acf_weight[:, None]
        else:
            self.acf_sum[:len(acf)] += acf * acf_weight[:, None]
        self.acf_weight += len(sequences["residual"])
        residual = sequences["residual"]
        total = residual.sum(axis=0)
        square = (residual * residual).sum(axis=0)
        self.residual_sum = total if self.residual_sum is None else self.residual_sum + total
        self.residual_square_sum = (square if self.residual_square_sum is None
                                    else self.residual_square_sum + square)
        self.samples += len(residual)
        self.blocks.append({**population, "centers": len(residual),
                            "welch_windows": weight})

    def finish(self):
        if not self.spectral_weight:
            raise ValueError("No block was long enough for the requested Welch segment")
        spectra = {key: value / self.spectral_weight for key, value in self.spectra.items()}
        # Exact pair weighting is retained approximately across blocks; each
        # block ACF is normalized by its own lag-zero variance before averaging.
        weights = np.maximum(1, np.array([
            sum(max(0, block["centers"] - lag) for block in self.blocks)
            for lag in range(len(self.acf_sum))]))
        acf = self.acf_sum / weights[:, None]
        mean = self.residual_sum / self.samples
        variance = self.residual_square_sum / self.samples - mean * mean
        return self.frequency, spectra, acf, mean, np.sqrt(np.maximum(variance, 0))


def resonance_rows(frequency, spectra, poles, omega, gamma, dt):
    rows = []
    eps = np.finfo(float).tiny
    for mode, pole in enumerate(poles):
        index = int(np.argmin(abs(frequency - pole)))
        sa = float(spectra["acceleration"][index, mode])
        sc = float(spectra["core"][index, mode])
        se = float(spectra["residual"][index, mode])
        sce = complex(spectra["core_residual"][index, mode])
        sq = float(spectra["q"][index, mode])
        sqe = complex(spectra["q_residual"][index, mode])
        theta = 2 * np.pi * frequency[index] * dt
        core_transfer = (-dt**2 * omega[mode]**2
                         -2 * dt * gamma[mode] * (1 - np.exp(-1j * theta)))
        associated_residual_transfer = sqe / max(sq, eps)
        rows.append(dict(
            mode=mode, fitted_peak_hz=float(pole), bin_hz=float(frequency[index]),
            core_over_acceleration=sc / max(sa, eps),
            residual_over_acceleration=se / max(sa, eps),
            signed_cross_over_acceleration=2 * sce.real / max(sa, eps),
            budget_sum_over_acceleration=(sc + se + 2 * sce.real) / max(sa, eps),
            core_residual_coherence=float(abs(sce)**2 / max(sc * se, eps)),
            core_residual_phase_deg=float(np.angle(sce, deg=True)),
            associated_residual_over_core_real=float(
                (associated_residual_transfer / core_transfer).real),
            associated_residual_over_core_imag=float(
                (associated_residual_transfer / core_transfer).imag),
        ))
    return rows


def summarize_rows(rows):
    def values(key):
        return np.asarray([row[key] for row in rows], dtype=float)
    cross = values("signed_cross_over_acceleration")
    associated = values("associated_residual_over_core_real")
    return dict(
        modes=len(rows),
        median_core_over_acceleration=float(np.median(values("core_over_acceleration"))),
        median_residual_over_acceleration=float(np.median(values("residual_over_acceleration"))),
        median_signed_cross_over_acceleration=float(np.median(cross)),
        modes_with_negative_cross=int(np.sum(cross < 0)),
        modes_with_cross_below_minus_half_acceleration=int(np.sum(cross < -.5)),
        median_core_residual_coherence=float(np.median(values("core_residual_coherence"))),
        median_associated_residual_over_core_real=float(np.median(associated)),
        modes_with_associated_residual_opposing_core=int(np.sum(associated < 0)),
        maximum_budget_identity_absolute_error=float(np.max(abs(
            values("budget_sum_over_acceleration") - 1))),
    )


def acf_summary(acf, dt, context=63):
    requested = (1, 2, 5, 10, 20, 50, 63, 100, 200, 500, 1000, 2000)
    lags = [lag for lag in requested if lag < len(acf)]
    median_abs = np.median(abs(acf), axis=1)
    maximum_abs = np.max(abs(acf), axis=1)
    tail = median_abs[min(context + 1, len(acf)):]
    return dict(
        values={str(lag): dict(
            time_ms=1000 * lag * dt,
            median_absolute=float(median_abs[lag]),
            maximum_absolute=float(maximum_abs[lag])) for lag in lags},
        median_absolute_after_context_max=(float(tail.max()) if len(tail) else None),
        first_lag_after_context_below_0p1=(next((int(i) for i in range(context + 1, len(acf))
                                               if median_abs[i] < .1), None)),
        context_frames=context, context_ms=1000 * context * dt,
    )


def analyze_paths(paths, transform, equilibrium, stiffness, damping, dt, nperseg, max_lag):
    accumulator = Accumulator(max_lag)
    for label, values in paths:
        sequences = modal_sequences(values, transform, equilibrium, stiffness, damping, dt)
        accumulator.add(sequences, 1 / dt, nperseg,
                        {"source": label, "start": 0, "stop": len(values)})
    return accumulator.finish(), accumulator.blocks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout", nargs=2, action="append", default=[],
                        metavar=("LABEL", "ROLLOUT_NPZ"))
    parser.add_argument("--nperseg", type=int, default=16384)
    parser.add_argument("--block-centers", type=int, default=32768)
    parser.add_argument("--max-lag", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    model, checkpoint = load_model(args.checkpoint, "cpu")
    if (model.config.drift_type != "imported_oscillator_residual"
            or model.config.residual_input != "core_innovation"):
        raise ValueError("Checkpoint must use the core-innovation imported oscillator")
    data = checkpoint_data(checkpoint)
    artifact = Path(checkpoint["training"]["imported_oscillator"]["source_artifact"])
    with np.load(artifact, allow_pickle=False) as saved:
        scale = np.asarray(saved["scale"], dtype=float)
        mixing = np.asarray(saved["T"], dtype=float)
        omega = np.asarray(saved["omega_per_second"], dtype=float)
        gamma = np.asarray(saved["gamma_per_second"], dtype=float)
    transform = np.sqrt(data.n_space) * scale[:, None] * mixing
    stiffness = model.drift_model.physical_stiffness.detach().double().numpy()
    damping = model.drift_model.physical_damping.detach().double().numpy()
    equilibrium = model.drift_model.equilibrium.detach().double().numpy()
    dt = model.config.dt
    poles = np.sqrt(np.maximum(0, omega**2 - 2 * gamma**2)) / (2 * np.pi)
    inverse = np.linalg.inv(transform)
    np.testing.assert_allclose(inverse @ stiffness @ transform,
                               np.diag(omega**2), rtol=2e-8, atol=1e-5)
    np.testing.assert_allclose(inverse @ damping @ transform,
                               np.diag(2 * gamma), rtol=2e-8, atol=1e-6)

    report = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        artifact=str(artifact.resolve()),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        definition="a[n]=x[n+1]-2x[n]+x[n-1]=core[n]+residual[n] in fitted oscillator modal coordinates",
        core="-dt^2 K (x[n]-equilibrium) - dt D (x[n]-x[n-1])",
        dt_seconds=dt, sampling_frequency_hz=1 / dt,
        nperseg=args.nperseg, frequency_spacing_hz=1 / dt / args.nperseg,
        block_centers=args.block_centers, max_lag=args.max_lag,
        context_frames=model.config.tcn_context_frames,
        context_ms=1000 * model.config.tcn_context_frames * dt,
        populations={}, generated={}, caveats=[
            "Spectra are averaged across independent contiguous blocks; signals and repetitions are never concatenated.",
            "Ratios at resonance bins can be large when measured acceleration power is small; inspect signed cross terms and coherence together.",
            "S_qe/S_qq is a frequency-domain least-squares association, not an identified causal residual drift.",
            "Generated screens contain four 5001-frame paths and have coarser Welch resolution than the observed-population analysis.",
            "No test repetition is read.",
        ])
    arrays = dict(poles_hz=poles, omega_per_second=omega,
                  gamma_per_second=gamma, transform=transform)

    for split in ("train", "validation"):
        accumulator = Accumulator(args.max_lag)
        names = data.splits[split]
        for name in names:
            count = data.frame_counts[name]
            for center_start in range(1, count - 1, args.block_centers):
                center_stop = min(count - 1, center_start + args.block_centers)
                _, values = data.read_coordinates(name, center_start - 1, center_stop + 1)
                sequences = modal_sequences(
                    values, transform, equilibrium, stiffness, damping, dt)
                accumulator.add(sequences, 1 / dt, args.nperseg,
                                {"repetition": name, "center_start": center_start,
                                 "center_stop": center_stop})
            print("processed", split, name, flush=True)
        frequency, spectra, acf, mean, std = accumulator.finish()
        rows = resonance_rows(frequency, spectra, poles, omega, gamma, dt)
        report["populations"][split] = dict(
            repetitions=list(names), samples=accumulator.samples,
            blocks=accumulator.blocks, residual_modal_mean=mean.tolist(),
            residual_modal_std=std.tolist(), resonance_summary=summarize_rows(rows),
            resonances=rows, residual_acf=acf_summary(
                acf, dt, model.config.tcn_context_frames))
        arrays.update({f"{split}_frequency_hz": frequency,
                       f"{split}_residual_acf": acf,
                       **{f"{split}_{key}_psd": value for key, value in spectra.items()}})
        save_json(args.output / "report.json", report)

    for label, path_value in args.rollout:
        path = Path(path_value)
        with np.load(path, allow_pickle=False) as saved:
            trajectories = np.asarray(saved["trajectories"], dtype=float)
        paths = [(f"condition_{i}_path_{j}", trajectories[i, j])
                 for i in range(trajectories.shape[0])
                 for j in range(trajectories.shape[1])]
        (frequency, spectra, acf, mean, std), blocks = analyze_paths(
            paths, transform, equilibrium, stiffness, damping, dt,
            min(4096, trajectories.shape[2] - 2), min(args.max_lag, 2000))
        rows = resonance_rows(frequency, spectra, poles, omega, gamma, dt)
        report["generated"][label] = dict(
            rollout=str(path.resolve()), shape=list(trajectories.shape),
            residual_modal_mean=mean.tolist(), residual_modal_std=std.tolist(),
            resonance_summary=summarize_rows(rows), resonances=rows,
            residual_acf=acf_summary(acf, dt, model.config.tcn_context_frames),
            blocks=blocks)
        arrays.update({f"generated_{label}_frequency_hz": frequency,
                       f"generated_{label}_residual_acf": acf,
                       **{f"generated_{label}_{key}_psd": value
                          for key, value in spectra.items()}})
        save_json(args.output / "report.json", report)

    np.savez_compressed(args.output / "spectral_budget.npz", **arrays)
    make_plots(args.output, report, arrays)
    write_readme(args.output, report)
    print(args.output / "README.md")


def make_plots(output, report, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = arrays["validation_frequency_hz"]
    mask = (f >= 100) & (f <= 1500)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for mode, ax in enumerate(axes.flat):
        sa = arrays["validation_acceleration_psd"][:, mode]
        for key, label in (("acceleration", "Measured acceleration"),
                           ("core", "Frozen core"), ("residual", "Required residual")):
            ax.semilogy(f[mask], arrays[f"validation_{key}_psd"][mask, mode],
                        label=label, lw=1)
        cross = 2 * arrays["validation_core_residual_psd"][:, mode].real
        ax.plot(f[mask], abs(cross[mask]), label="|signed cross term|", lw=.8, ls="--")
        ax.axvline(report["populations"]["validation"]["resonances"][mode]["fitted_peak_hz"],
                   color=".5", lw=.6)
        ax.set(title="Spatial mean direction" if mode == 0 else f"Oscillator mode {mode}",
               xlabel="Frequency [Hz]", ylabel="Modal increment PSD")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output / "spectral_budget.png", dpi=170)
    plt.close(fig)

    rows = report["populations"]["validation"]["resonances"]
    frequency = np.array([row["fitted_peak_hz"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    axes[0].plot(frequency, [row["core_over_acceleration"] for row in rows], "o-", label="core")
    axes[0].plot(frequency, [row["residual_over_acceleration"] for row in rows], "o-", label="residual")
    axes[0].plot(frequency, [row["signed_cross_over_acceleration"] for row in rows], "o-", label="signed cross")
    axes[0].axhline(0, color="k", lw=.5); axes[0].set(ylabel="Contribution / measured acceleration", yscale="symlog", title="Validation resonance-bin budget")
    axes[0].legend()
    axes[1].plot(frequency, [row["core_residual_coherence"] for row in rows], "o-", label="core-residual coherence")
    axes[1].plot(frequency, [row["associated_residual_over_core_real"] for row in rows], "o-", label="Re[associated residual/core]")
    axes[1].axhline(0, color="k", lw=.5); axes[1].set(xlabel="Fitted peak [Hz]", ylabel="Coherence or ratio", yscale="symlog")
    axes[1].legend()
    fig.savefig(output / "resonance_budget.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for split, style in (("train", "-"), ("validation", "--")):
        acf = arrays[f"{split}_residual_acf"]
        ax.semilogy(np.arange(len(acf)) * report["dt_seconds"] * 1000,
                    np.maximum(np.median(abs(acf), axis=1), 1e-5), style,
                    label=f"Measured {split}")
    for label in report["generated"]:
        acf = arrays[f"generated_{label}_residual_acf"]
        ax.semilogy(np.arange(len(acf)) * report["dt_seconds"] * 1000,
                    np.maximum(np.median(abs(acf), axis=1), 1e-5),
                    label=f"Generated {label}", alpha=.8)
    ax.axvline(report["context_ms"], color="k", lw=.7, ls=":", label="TCN context")
    ax.set(xlabel="Lag [ms]", ylabel="Median |modal residual ACF|",
           title="Required forcing memory", ylim=(1e-4, 1.1))
    ax.legend(fontsize=8)
    fig.savefig(output / "residual_acf.png", dpi=170)
    plt.close(fig)


def write_readme(output, report):
    validation = report["populations"]["validation"]
    train = report["populations"]["train"]
    v = validation["resonance_summary"]
    lines = [
        "# Conditional compatibility of the frozen oscillator", "",
        "[Spectral budget](spectral_budget.png) · [Resonance summary](resonance_budget.png) · "
        "[Residual memory](residual_acf.png) · [Full numbers](report.json)", "",
        "This audit decomposes every measured native-frame velocity change exactly as", "",
        "`measured acceleration = frozen oscillator core + required residual`.", "",
        f"It uses {len(train['repetitions'])} complete training repetitions and "
        f"{len(validation['repetitions'])} complete validation repetitions, processed in "
        "independent contiguous blocks. Test repetitions are not read.", "",
        "## Main validation findings", "",
        f"- Median core/acceleration power at the 20 fitted resonance bins: **{v['median_core_over_acceleration']:.3g}**.",
        f"- Median residual/acceleration power: **{v['median_residual_over_acceleration']:.3g}**.",
        f"- Median signed cross contribution: **{v['median_signed_cross_over_acceleration']:.3g}**; "
        f"{v['modes_with_negative_cross']}/20 modes are negative.",
        f"- Median core-residual coherence: **{v['median_core_residual_coherence']:.3g}**.",
        f"- Frequency-domain residual association opposes the core in "
        f"**{v['modes_with_associated_residual_opposing_core']}/20** modes; median real ratio "
        f"**{v['median_associated_residual_over_core_real']:.3g}**.",
        f"- Maximum exact spectral-budget identity error at resonance bins: "
        f"**{v['maximum_budget_identity_absolute_error']:.2e}**.", "",
        "The signed cross-spectrum is the important quantity: large core and residual powers do not "
        "add independently when their cross contribution is negative. The associated residual/core "
        "ratio is descriptive, because forcing and state are jointly generated.", "",
        "## Residual memory", "",
        f"The TCN sees {report['context_frames']} forcing tokens ({report['context_ms']:.3f} ms).", "",
        "| Population | Median |ACF| at lag 63 | Lag 100 | Lag 200 | Maximum after context |", "|---|---:|---:|---:|---:|"
    ]
    for label, population in [("Train", train), ("Validation", validation)]:
        summary = population["residual_acf"]
        values = summary["values"]
        def value(lag):
            return values.get(str(lag), {}).get("median_absolute")
        lines.append(f"| {label} | {value(63):.4g} | {value(100):.4g} | {value(200):.4g} | {summary['median_absolute_after_context_max']:.4g} |")
    for label, population in report["generated"].items():
        summary = population["residual_acf"]; values = summary["values"]
        def value(lag):
            return values.get(str(lag), {}).get("median_absolute")
        lines.append(f"| Generated {label} | {value(63):.4g} | {value(100):.4g} | {value(200):.4g} | {summary['median_absolute_after_context_max']:.4g} |")
    lines += ["", "## Interpretation rule", "",
              "Strong negative, coherent cross terms mean that a PSD-fitted core cannot be treated as an "
              "independent conditional drift: the measured residual supplies phase-dependent compensation. "
              "Weak cross terms with long residual ACF instead support a colored-forcing model with longer memory. "
              "Generated comparisons show which part the current forcing-history model fails to reproduce.", "",
              "The audit diagnoses compatibility; it does not refit an operator or establish a causal physical force."]
    (output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
