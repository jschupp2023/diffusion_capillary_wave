"""Compare cached experimental and rank-r POD center bispectra/bicoherence.

Both instantaneous-spatial-mean treatments are computed. No raw image file is
required. Example (use the capillarywave Python environment)::

    python compare_center_bispectrum.py RAW_CACHE.npz POD_FILE.h5 \
        --rank 100 --output-dir OUTPUT

Use --replot to regenerate figures from OUTPUT/bispectral_results.npz without
repeating the transforms. The original paper's normalization is retained.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pywt
from scipy.signal import welch

from bis_bic import wavelet_bispectral_metrics
from data_analysis.bispectrum.compare_bispectral_repetitions import distance_matrices, log_area_weights
from data_analysis.psd.pod_center_psd import infer_sampling_frequency, reconstruct_point_signals, resolve_pod_file
from data_analysis.psd.raw_center_psd import read_center_signal
from data_analysis.bispectrum.sparse_bispectrum import sparse_grid


TREATMENTS = {
    "with_mean": "Instantaneous spatial mean retained",
    "without_mean": "Instantaneous spatial mean removed",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("pod_input", type=Path)
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--raw-input", type=Path,
                        help="Read this local raw HDF5 and verify every center sample against the cache before analysis.")
    parser.add_argument("--score-axis-samples", type=int, default=64,
                        help="Also score a subset of the dense maps matching repetition comparisons (default: 64 targets).")
    parser.add_argument("--score-max-sum-frequency", type=float, default=10000)
    parser.add_argument("--stored-rank", type=int, default=1000, help="POD filename rank when input is a folder.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fb-low", type=float, default=150.0)
    parser.add_argument("--fb-high", type=float, default=57000.0, help="Maximum transform/sum frequency; plotted axes end at half this value.")
    parser.add_argument("--num-scales", type=int, default=1024)
    parser.add_argument("--wavelet", default="cgau1")
    parser.add_argument("--profile-frequency", type=float, default=2500.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replot", action="store_true", help="Regenerate figures and summary using the saved numerical results.")
    return parser.parse_args()


def load_signals(raw_cache, pod_path, rank, raw_input=None):
    with np.load(raw_cache, allow_pickle=False) as cache:
        raw_time = np.asarray(cache["time"], dtype=np.float64)
        raw_signal = np.asarray(cache["center_signal"], dtype=np.float64)
        point_indices = (int(cache["center_y_index"]), int(cache["center_x_index"]))
        coordinates = (float(cache["center_y_coordinate"]), float(cache["center_x_coordinate"]))
        raw_units = str(cache["signal_units"])
        raw_source = str(cache["source_file"]) if "source_file" in cache else None
    raw_validation = None
    if raw_input is not None:
        raw_input = raw_input.expanduser().resolve()
        local_t, local_z, iy, ix, x, y, local_units = read_center_signal(raw_input)
        if not np.array_equal(local_t, raw_time) or not np.array_equal(local_z, raw_signal):
            raise ValueError("Local raw HDF5 center samples/timestamps differ from the experimental cache.")
        if (iy, ix) != point_indices or not np.allclose((y, x), coordinates) or local_units["z"] != raw_units:
            raise ValueError("Local raw HDF5 point coordinates/units differ from the experimental cache.")
        raw_signal = local_z
        stat = raw_input.stat()
        raw_validation = {"path": str(raw_input), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                          "all_center_samples_exactly_match_cache": True, "samples_checked": len(local_z)}
    t, preprocessed, spatial_mean, restored, point, rank, units = reconstruct_point_signals(
        pod_path, rank, point_indices, None, 8192, show_progress=False
    )
    if not np.array_equal(t, raw_time) or raw_signal.shape != t.shape:
        raise ValueError("Experimental and POD samples/timestamps do not match.")
    if not np.allclose(coordinates, (point.y_coordinate, point.x_coordinate), rtol=1e-10, atol=1e-10):
        raise ValueError("Experimental and POD center coordinates do not match.")
    if raw_units != units:
        raise ValueError(f"Signal units differ: {raw_units!r} vs {units!r}.")
    if not np.isfinite(raw_signal).all():
        raise ValueError("The cached experimental center signal is not finite.")
    with h5py.File(pod_path, "r") as handle:
        source = str(handle.attrs.get("source_file", ""))
        if raw_source and source and raw_source != source:
            raise ValueError("Cache and POD metadata identify different experimental source files.")
        mean_field = handle["preprocessing/temporal_mean_field"]
        removed = bool(mean_field.attrs.get("subtracted_before_pod", handle.attrs.get("temporal_mean_field_removed", False)))
        static_offset = float(mean_field[point.y_index, point.x_index]) if removed else 0.0
    fs, jitter = infer_sampling_frequency(t)
    signals = {
        "with_mean_experiment": raw_signal,
        "with_mean_pod": restored + static_offset,
        "without_mean_experiment": raw_signal - spatial_mean,
        "without_mean_pod": preprocessed + static_offset,
    }
    metadata = {
        "raw_cache": str(raw_cache), "pod_file": str(pod_path),
        "source_file": source, "experiment": pod_path.parent.name,
        "power": pod_path.parent.parent.name, "rank": rank,
        "center_y_index": point.y_index, "center_x_index": point.x_index,
        "center_x_coordinate": point.x_coordinate, "center_y_coordinate": point.y_coordinate,
        "signal_units": units, "sampling_frequency_hz": fs,
        "maximum_relative_timestamp_step_variation": jitter,
        "n_samples": len(t), "duration_seconds": float(t[-1] - t[0]),
        "restored_static_center_offset": static_offset,
        "local_raw_validation": raw_validation,
    }
    return t, spatial_mean, signals, metadata


def finite_limits(arrays):
    values = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if not len(values):
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)


def log_magnitude(values):
    result = np.full_like(values, np.nan, dtype=np.float64)
    np.log10(values, out=result, where=values > 0)
    return result


def pair_arrays(data, treatment, metric):
    if metric == "bispectrum":
        a, b = [log_magnitude(np.abs(data[f"{treatment}_{source}_complex_bispectrum"])) for source in ("experiment", "pod")]
    else:
        a, b = [data[f"{treatment}_{source}_bicoherence"] for source in ("experiment", "pod")]
    return a, b, np.abs(a - b)


def plot_metric(data, metadata, treatment, metric, limits, error_max, profile_frequency):
    f = data["frequency_hz"] / 1000
    profile = int(np.argmin(abs(f * 1000 - profile_frequency)))
    arrays = pair_arrays(data, treatment, metric)
    figure, axes = plt.subplots(
        3, 3, figsize=(15, 10), sharex=True,
        gridspec_kw={"height_ratios": [1, 3.7, 1]}, layout="constrained",
    )
    colors = ("#ae2683", "#dc9215")
    main_label = r"$\log_{10}|B|$" if metric == "bispectrum" else "Bicoherence b"
    error_label = r"$|\log_{10}(|B_{exp}|/|B_r|)|$ [decades]" if metric == "bispectrum" else r"$|b_{exp}-b_r|$"
    main_trace_limits = finite_limits([arrays[0][profile], arrays[1][profile], np.diag(arrays[0]), np.diag(arrays[1])])
    if metric == "bicoherence":
        main_trace_limits = (0, 1)
    for col, (title, values) in enumerate(zip(("Experiment", f"POD rank {metadata['rank']}", "Absolute difference"), arrays)):
        axes[0, col].set_title(title, fontsize=13)
        axes[0, col].plot(f, values[profile], color=colors[0], lw=1)
        axes[2, col].plot(f, np.diag(values), color=colors[1], lw=1)
        lo, hi = limits if col < 2 else (0, error_max)
        mesh = axes[1, col].pcolormesh(
            f, f, np.ma.masked_invalid(values), shading="auto", rasterized=True,
            cmap="magma" if col < 2 else "Reds", vmin=lo, vmax=hi,
        )
        axes[1, col].set_yscale("log")
        axes[1, col].plot(f, f, "--", color=colors[1], lw=1)
        axes[1, col].axhline(f[profile], ls="--", color=colors[0], lw=1.2)
        axes[1, col].set_ylim(f[0], f[-1])
        axes[2, col].set_xlabel(r"$f_2$ [kHz]")
        for row in (0, 2):
            axes[row, col].set_ylim(main_trace_limits if col < 2 else (0, error_max))
            axes[row, col].grid(True, which="both", alpha=0.2)
        for row in range(3):
            axes[row, col].set_xscale("log")
            axes[row, col].set_xlim(f[0], f[-1])
            axes[row, col].xaxis.set_major_formatter(ScalarFormatter())
        if col == 1:
            figure.colorbar(mesh, ax=axes[1, :2], orientation="horizontal", pad=0.03, label=main_label)
        elif col == 2:
            figure.colorbar(mesh, ax=axes[1, 2], orientation="horizontal", pad=0.03, label=error_label)
    axes[0, 0].set_ylabel(f"Slice at {f[profile]:.3f} kHz")
    axes[1, 0].set_ylabel(r"$f_1$ [kHz]")
    axes[2, 0].set_ylabel(r"Diagonal: $f_1=f_2$")
    name = "Bispectrum magnitude" if metric == "bispectrum" else "Bicoherence (original normalization)"
    figure.suptitle(
        f"{name} — {metadata['power']} / {metadata['experiment']}\n"
        f"{TREATMENTS[treatment]} · same color scales across both treatments\n"
        f"Temporal mean removed; wavelet edges excluded; exact sum frequencies; {metadata['wavelet']}",
        fontsize=13,
    )
    return figure


def plot_signals(data, metadata):
    t = data["time"]
    fs = metadata["sampling_frequency_hz"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    # Display subsampling is only for the time plot. Analysis always uses every sample.
    stride = max(1, len(t) // 12000)
    start, stop = metadata["sample_start"], metadata["sample_stop"]
    for col, treatment in enumerate(TREATMENTS):
        for source, color in (("experiment", "#244d8e"), ("pod", "#e28728")):
            z = data[f"{treatment}_{source}_signal"]
            z = z - z.mean()
            axes[0, col].plot(t[::stride], z[::stride], color=color, lw=0.65, alpha=0.85, label=source)
            f, p = welch(z[start:stop], fs=fs, nperseg=min(8192, stop-start), noverlap=min(8192, stop-start)//2)
            axes[1, col].loglog(f[1:], p[1:], color=color, lw=1, label=source)
        axes[0, col].axvspan(t[0], t[start], color="0.5", alpha=0.15)
        axes[0, col].axvspan(t[stop - 1], t[-1], color="0.5", alpha=0.15)
        axes[0, col].set_title(TREATMENTS[treatment])
        axes[0, col].set_xlabel("Time [s]")
        axes[0, col].set_ylabel(f"Centered height [{metadata['signal_units']}]")
        axes[1, col].set_xlabel("Frequency [Hz]")
        axes[1, col].set_ylabel(f"PSD [{metadata['signal_units']}²/Hz]")
        for row in range(2):
            axes[row, col].grid(True, which="both", alpha=0.2)
            axes[row, col].legend()
    fig.suptitle(
        f"Center signals — {metadata['power']} / {metadata['experiment']} / rank {metadata['rank']}\n"
        "Shaded record ends excluded from bispectral averages; PSDs use the retained interval",
        fontsize=14,
    )
    return fig


def summarize(data, metadata):
    summary = {}
    upper = np.triu_indices(len(data["frequency_hz"]))
    start, stop = metadata["sample_start"], metadata["sample_stop"]
    for treatment in TREATMENTS:
        a, b = [data[f"{treatment}_{s}_signal"] for s in ("experiment", "pod")]
        a, b = (a - a.mean())[start:stop], (b - b.mean())[start:stop]
        denom = np.linalg.norm(a)
        item = {
            "center_signal_relative_l2_error": float(np.linalg.norm(a-b) / denom) if denom else None,
            "experimental_center_signal_std": float(np.std(a)),
        }
        for metric in ("bispectrum", "bicoherence"):
            values = pair_arrays(data, treatment, metric)[2][upper]
            values = values[np.isfinite(values)]
            item[f"{metric}_mean_absolute_difference"] = float(np.mean(values)) if len(values) else None
            item[f"{metric}_median_absolute_difference"] = float(np.median(values)) if len(values) else None
            item[f"{metric}_p95_absolute_difference"] = float(np.quantile(values, 0.95)) if len(values) else None
        summary[treatment] = item
    return summary


def weighted_scores(data, metadata, axis_samples=64, maximum_sum=10000):
    """Use the repetition code's common mask, log-area weights and centering.

    Dense scores describe the plotted maps. Subset scores match the existing
    sparse repetition reports exactly; never interpolate bispectral values.
    """
    frequency = data["frequency_hz"]
    selections = {"dense": np.arange(len(frequency))}
    if axis_samples:
        grid = sparse_grid(metadata["sampling_frequency_hz"], metadata["fb_low_hz"],
                           metadata["fb_high_hz"], metadata["num_scales_requested"],
                           metadata["wavelet"], axis_samples)
        np.testing.assert_allclose(frequency[grid.output_positions], grid.frequency_hz, rtol=1e-12)
        selections[f"sparse{axis_samples}"] = grid.output_positions
    scores = {}
    for name, selected in selections.items():
        for treatment in TREATMENTS:
            logs = np.asarray(pair_arrays(data, treatment, "bispectrum")[:2])[:, selected][:, :, selected]
            b = np.asarray(pair_arrays(data, treatment, "bicoherence")[:2])[:, selected][:, :, selected]
            for domain, maximum in (("full", None), ("limited_sum", maximum_sum)):
                upper, weights = log_area_weights(frequency[selected], maximum)
                distances, info = distance_matrices(logs[:, upper[0], upper[1]], b[:, upper[0], upper[1]], weights)
                scores[f"{name}__{treatment}__{domain}"] = {
                    "axis_samples": len(selected), "maximum_sum_frequency_hz": maximum,
                    **{metric: float(matrix[0, 1]) for metric, matrix in distances.items()}, **info,
                }
    return scores


def render_raw_maps(data, metadata, output, dpi):
    f = data["frequency_hz"] / 1000
    with PdfPages(output / "raw_bispectrum_bicoherence.pdf") as pdf:
        for treatment, title in TREATMENTS.items():
            log = log_magnitude(np.abs(data[f"{treatment}_experiment_complex_bispectrum"]))
            b = data[f"{treatment}_experiment_bicoherence"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
            for ax, values, label, limits in zip(axes, (log, b),
                    (r"$\log_{10}|B|$", "Bicoherence b"), (finite_limits([log]), (0, 1))):
                mesh = ax.pcolormesh(f, f, values, shading="auto", cmap="magma", rasterized=True,
                                     vmin=limits[0], vmax=limits[1])
                ax.set(xscale="log", yscale="log", xlabel="f2 [kHz]", ylabel="f1 [kHz]", title=label)
                fig.colorbar(mesh, ax=ax)
            fig.suptitle(f"Raw experiment: {metadata['power']} / {metadata['experiment']}\n"
                         f"{title}; center point; temporal mean removed")
            fig.savefig(output / f"raw_{treatment}.png", dpi=dpi)
            pdf.savefig(fig)
            plt.close(fig)


def render(data, metadata, output, dpi, profile_frequency, score_axis_samples=64, score_max_sum=10000):
    metadata = dict(metadata)
    metadata["profile_frequency_requested_hz"] = profile_frequency
    metadata["profile_frequency_actual_hz"] = float(
        data["frequency_hz"][np.argmin(abs(data["frequency_hz"] - profile_frequency))]
    )
    summary = summarize(data, metadata)
    weighted = weighted_scores(data, metadata, score_axis_samples, score_max_sum)
    render_raw_maps(data, metadata, output, dpi)
    with (output / "weighted_distances.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grid_treatment_domain", "bicoherence_mae", "log_bispectrum_mae_decades",
                         "centered_log_bispectrum_mae_decades", "retained_log_area_fraction"])
        for key, item in weighted.items():
            writer.writerow([key, *[item[k] for k in ("bicoherence", "bispectrum_log", "bispectrum_shape",
                                                     "retained_log_area_fraction")]])
    with PdfPages(output / "center_bispectral_comparison.pdf") as pdf:
        fig = plot_signals(data, metadata)
        fig.savefig(output / "center_signals_and_psds.png", dpi=dpi)
        pdf.savefig(fig)
        plt.close(fig)
        for metric in ("bispectrum", "bicoherence"):
            pairs = [pair_arrays(data, tr, metric) for tr in TREATMENTS]
            limits = finite_limits([a for pair in pairs for a in pair[:2]]) if metric == "bispectrum" else (0, 1)
            error_max = max(1e-12, finite_limits([pair[2] for pair in pairs])[1])
            for treatment in TREATMENTS:
                fig = plot_metric(data, metadata, treatment, metric, limits, error_max, profile_frequency)
                fig.savefig(output / f"{treatment}_{metric}.png", dpi=dpi)
                pdf.savefig(fig)
                plt.close(fig)
    (output / "summary.json").write_text(json.dumps({"metadata": metadata, "comparison": summary,
                                                    "repetition_weighted_comparison": weighted}, indent=2, allow_nan=False) + "\n")
    f = data["frequency_hz"]
    lines = [
        "# Center-point bispectrum and bicoherence comparison", "",
        f"Case: **{metadata['power']} / {metadata['experiment']} / POD rank {metadata['rank']}**.",
        "", "Open `center_bispectral_comparison.pdf` for the signal diagnostics and all four metric comparisons.",
        "Open `raw_bispectrum_bicoherence.pdf` for standalone measured center-point maps.",
        "The bispectral_repetition_comparisons reports already use raw experimental center signals, not POD reconstructions.",
        "", "## Calculation", "",
        "- Experimental input is the cached center time series; POD input is the first r measured coefficients and center mode values.",
        "- With mean: compare the raw experiment with POD + saved instantaneous spatial mean + static center offset.",
        "- Without mean: subtract the saved instantaneous spatial mean from the experiment; compare with POD + static center offset.",
        "- Subtract each signal's temporal mean before transforming. No linear detrending, resampling, or averaging of repetitions.",
        f"- Wavelet: `{metadata['wavelet']}`; sampling frequency: {metadata['sampling_frequency_hz']:.6f} Hz.",
        f"- Frequency spacing: {metadata['frequency_step_hz']:.6f} Hz. Axes: {f[0]:.6f}–{f[-1]:.6f} Hz ({len(f)} points).",
        f"- Sum frequencies are selected exactly. Transform extends to {metadata['fb_high_hz']:.1f} Hz, below Nyquist.",
        f"- Common retained time interval: samples [{metadata['sample_start']}:{metadata['sample_stop']}] (Python slicing); wavelet-support margins excluded.",
        "- Q = W(f1,t) W(f2,t) conjugate(W(f1+f2,t)); B = mean(Q); b = abs(B)/mean(abs(Q)).",
        "- Original normalization: b, not b squared. Zero denominators are undefined (NaN). Complex bispectrum and denominators are saved.",
        "", "## Reading the figures", "",
        "The horizontal axis is f2 and the vertical axis is f1. A pixel tests the triad (f1, f2, f1+f2), not a pair of spatial coordinates.",
        "The upper trace is a fixed-f1 slice; the lower trace is the f1=f2 diagonal (f+f=2f).",
        "Bispectrum color is log10(abs(B)); bicoherence color is b on [0,1]. Shared scales span both mean treatments.",
        "Bispectrum difference is abs(log10(abs(B_exp)/abs(B_r))) in decades: 0.301 means a factor of two and 1 means a factor of ten. It is not log10(abs(B_exp-B_r)).",
        "Bicoherence difference is abs(b_exp-b_r). Neither magnitude plot compares biphase; the saved complex B permits that later.",
        "", "## Numerical comparison", "",
        "Differences below weight unique frequency pairs equally on the linear frequency grid. They are descriptive, not significance tests.",
        "The additional `weighted_distances.csv` and `repetition_weighted_comparison` in summary.json use the exact repetition-comparison functions: normalized log-frequency cell area, folded symmetry, common finite mask, and weighted MAE.",
        "The shape distance subtracts each log10(abs(B)) map's weighted mean before calculating MAE. It removes uniform amplitude scaling. Bicoherence distance is in [0,1]; both log distances are in decades. Lower is closer.",
        f"Both dense and sparse{score_axis_samples} scores are reported, for the full domain and f1+f2 <= {score_max_sum:g} Hz. The sparse scores select original grid cells and match repetition reports with the same settings. These are center-point temporal spectra, not spatially averaged field spectra.",
    ]
    for treatment, item in summary.items():
        lines += ["", f"### {TREATMENTS[treatment]}", ""]
        lines += [f"- {key}: {value:.6g}" if value is not None else f"- {key}: undefined" for key, value in item.items()]
    lines += [
        "", "## Reproduce", "", "```bash",
        shlex.join([
            sys.executable, str(Path(__file__).resolve()), metadata["raw_cache"], metadata["pod_file"],
            "--rank", str(metadata["rank"]), "--output-dir", str(output),
            "--fb-low", str(metadata["fb_low_hz"]), "--fb-high", str(metadata["fb_high_hz"]),
            "--num-scales", str(metadata["num_scales_requested"]), "--wavelet", metadata["wavelet"],
            "--profile-frequency", str(profile_frequency),
            "--score-axis-samples", str(score_axis_samples), "--score-max-sum-frequency", str(score_max_sum),
            *(["--raw-input", metadata["local_raw_validation"]["path"]] if metadata.get("local_raw_validation") else []),
        ]),
        "```", "",
        "Add `--replot` to redraw the saved results, or `--overwrite` to recompute this case. Change `--rank` and `--output-dir` for a different truncation.",
        "", "## Interpretation limits", "",
        "Retaining the same spatial mean in both signals can make agreement appear stronger than agreement of the POD fluctuations alone.",
        "Bicoherence measures amplitude-weighted phase consistency, not an energy-transfer rate or proof of a spatial resonance.",
        "Finite records, weak-amplitude regions, wavelet bandwidth, and discretization can affect the maps. No surrogate significance threshold has been applied.",
        "The cgau1 wavelet and upper band match the previous method, but the highest sum frequencies use short, coarsely sampled wavelets near Nyquist; interpret those values cautiously.",
        "Bispectrum amplitudes use the PyWavelets sample-domain CWT convention and the saved height units (cubed); they are not a Fourier bispectral density per Hz squared.",
        "This is a single measured repetition and its POD projection, whereas the old notebook used ensemble-mean experimental and simulated ROM signals.",
        "", "## References", "",
        "- Orosco, Connacher & Friend (2023), Eq. 11: https://doi.org/10.1016/j.chaos.2023.113615",
        "- PyWavelets CWT frequency/scale convention: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse_args()
    if args.rank < 0 or args.dpi < 1 or args.profile_frequency <= 0 or args.num_scales < 4:
        raise ValueError("Rank must be nonnegative, dpi/profile frequency positive, and num-scales >= 4.")
    if args.score_axis_samples < 0 or args.score_axis_samples == 1 or args.score_max_sum_frequency <= 0:
        raise ValueError("Scoring samples must be 0 or >= 2; maximum sum frequency must be positive.")
    pod_path = resolve_pod_file(args.pod_input, args.stored_rank)
    output = (args.output_dir or pod_path.parent / f"bispectrum_r{args.rank}").expanduser().resolve()
    result_path = output / "bispectral_results.npz"
    if args.replot:
        with np.load(result_path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files if k != "metadata_json"}
            metadata = json.loads(str(archive["metadata_json"]))
    else:
        if result_path.exists() and not args.overwrite:
            raise FileExistsError(f"{result_path} exists. Use --replot or --overwrite.")
        started = time.perf_counter()
        raw_cache = args.raw_cache.expanduser().resolve()
        t, spatial_mean, signals, metadata = load_signals(raw_cache, pod_path, args.rank, args.raw_input)
        metadata.update({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "wavelet": args.wavelet, "pywavelets_version": pywt.__version__,
            "fb_low_hz": args.fb_low, "fb_high_hz": args.fb_high,
            "num_scales_requested": args.num_scales,
            "frequency_step_hz": args.fb_high / args.num_scales,
            "normalization": "abs(mean(Q))/mean(abs(Q))",
            "demean": True, "trim_wavelet_support_edges": True,
        })
        data = {"time": t, "frame_spatial_mean": spatial_mean}
        for name, signal in signals.items():
            def progress(message):
                print(f"[{time.perf_counter() - started:.1f}s] {name}: {message}", flush=True)
            progress("Starting.")
            result = wavelet_bispectral_metrics(
                signal, t, metadata["sampling_frequency_hz"], args.fb_low, args.fb_high,
                Ns=args.num_scales, wavelet=args.wavelet, progress=progress,
            )
            if "frequency_hz" in data and not np.array_equal(data["frequency_hz"], result.frequency_hz):
                raise ValueError("Frequency grids unexpectedly differ between signals.")
            data.update({
                "frequency_hz": result.frequency_hz,
                "transform_frequency_hz": result.transform_frequency_hz,
                "scales": result.scales,
                f"{name}_signal": signal,
                f"{name}_complex_bispectrum": result.complex_bispectrum,
                f"{name}_normalization_magnitude": result.normalization_magnitude,
                f"{name}_bicoherence": result.bicoherence,
                f"{name}_removed_temporal_mean": np.float64(result.removed_temporal_mean),
            })
            metadata.update({"sample_start": result.sample_start, "sample_stop": result.sample_stop})
        metadata["computation_seconds"] = time.perf_counter() - started
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(result_path, **data, metadata_json=np.asarray(json.dumps(metadata)))
        print(f"Saved {result_path}", flush=True)
    render(data, metadata, output, args.dpi, args.profile_frequency, args.score_axis_samples, args.score_max_sum_frequency)
    print(f"Saved figures, PDF, and summary in {output}", flush=True)


if __name__ == "__main__":
    main()
