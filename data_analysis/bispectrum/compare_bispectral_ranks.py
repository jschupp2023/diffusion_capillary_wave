"""Summarize saved raw-versus-POD center bispectra across reconstruction ranks.

Example: python compare_bispectral_ranks.py OUTPUT_R10 OUTPUT_R100 OUTPUT_R1000
         --output-dir RANK_OVERVIEW
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from data_analysis.bispectrum.compare_bispectral_repetitions import METRICS, distance_matrices, log_area_weights
from data_analysis.bispectrum.compare_center_bispectrum import TREATMENTS, finite_limits, log_magnitude
from data_analysis.bispectrum.sparse_bispectrum import sparse_grid


def load_cases(paths):
    cases = []
    for path in paths:
        with np.load(path / "bispectral_results.npz", allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files if k != "metadata_json"}
            metadata = json.loads(str(archive["metadata_json"]))
        cases.append((path.resolve(), metadata, data))
    cases.sort(key=lambda case: case[1]["rank"])
    if len({c[1]["rank"] for c in cases}) != len(cases):
        raise ValueError("Supply distinct reconstruction ranks.")
    reference = cases[0]
    for _, meta, data in cases[1:]:
        for key in ("source_file", "pod_file", "center_y_index", "center_x_index", "signal_units",
                    "wavelet", "pywavelets_version", "num_scales_requested", "fb_low_hz", "fb_high_hz",
                    "sample_start", "sample_stop", "sampling_frequency_hz", "demean",
                    "trim_wavelet_support_edges", "normalization"):
            if meta[key] != reference[1][key]:
                raise ValueError(f"Rank cases differ in {key}.")
        for key in ("frequency_hz", "time", "frame_spatial_mean",
                    *[f"{tr}_experiment_{quantity}" for tr in TREATMENTS
                      for quantity in ("signal", "complex_bispectrum", "bicoherence")]):
            np.testing.assert_array_equal(data[key], reference[2][key], err_msg=key)
    return cases


def maps(cases, treatment, quantity):
    values = [cases[0][2][f"{treatment}_experiment_{quantity}"]]
    values.extend(c[2][f"{treatment}_pod_{quantity}"] for c in cases)
    return np.asarray(values)


def score_cases(cases, axis_samples, maximum_sum):
    meta, data = cases[0][1:]
    frequency = data["frequency_hz"]
    grid = sparse_grid(meta["sampling_frequency_hz"], meta["fb_low_hz"], meta["fb_high_hz"],
                       meta["num_scales_requested"], meta["wavelet"], axis_samples)
    np.testing.assert_allclose(frequency[grid.output_positions], grid.frequency_hz, rtol=1e-12)
    rows = []
    for grid_name, indices in (("dense", np.arange(len(frequency))),
                               (f"sparse{axis_samples}", grid.output_positions)):
        for treatment in TREATMENTS:
            logs = log_magnitude(np.abs(maps(cases, treatment, "complex_bispectrum")))[:, indices][:, :, indices]
            b = maps(cases, treatment, "bicoherence")[:, indices][:, :, indices]
            for domain, maximum in (("full", None), ("limited_sum", maximum_sum)):
                upper, weights = log_area_weights(frequency[indices], maximum)
                distances, info = distance_matrices(logs[:, upper[0], upper[1]], b[:, upper[0], upper[1]], weights)
                for i, (_, meta, _) in enumerate(cases, 1):
                    rows.append({"grid": grid_name, "treatment": treatment, "domain": domain,
                                 "rank": meta["rank"], "axis_samples": len(indices),
                                 "maximum_sum_frequency_hz": maximum,
                                 **{metric: float(matrix[0, i]) for metric, matrix in distances.items()},
                                 "retained_log_area_fraction": info["retained_log_area_fraction"]})
    return rows


def plot_scores(rows, grid_name):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for row, treatment in enumerate(TREATMENTS):
        for ax, metric in zip(axes[row], METRICS):
            for domain, label in (("full", "Full domain"), ("limited_sum", "Limited sum domain")):
                selected = [r for r in rows if (r["grid"], r["treatment"], r["domain"]) ==
                            (grid_name, treatment, domain)]
                ax.plot([r["rank"] for r in selected], [r[metric] for r in selected], "o-", label=label)
            ax.set(xscale="log", xlabel="POD rank", ylabel=METRICS[metric], title=TREATMENTS[treatment])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle(f"Raw versus POD center spectra — {grid_name} repetition scoring\n"
                 "Log-frequency area weighted errors; lower = closer to raw")
    return fig


def plot_maps(cases, treatment, quantity):
    f = cases[0][2]["frequency_hz"] / 1000
    values = maps(cases, treatment, quantity)
    if quantity == "complex_bispectrum":
        values = log_magnitude(np.abs(values))
        limits, label = finite_limits(values), r"$\log_{10}|B|$"
    else:
        limits, label = (0, 1), "Bicoherence b"
    errors = np.abs(values[1:] - values[0])
    error_max = max(finite_limits(errors)[1], 1e-12)
    fig, axes = plt.subplots(2, len(values), figsize=(4 * len(values), 8), layout="constrained")
    names = ["Raw experiment", *[f"POD rank {c[1]['rank']}" for c in cases]]
    for i, (name, value) in enumerate(zip(names, values)):
        mesh = axes[0, i].pcolormesh(f, f, value, shading="auto", rasterized=True,
                                    cmap="magma", vmin=limits[0], vmax=limits[1])
        axes[0, i].set_title(name)
        if i:
            error_mesh = axes[1, i].pcolormesh(f, f, errors[i-1], shading="auto", rasterized=True,
                                               cmap="Reds", vmin=0, vmax=error_max)
            axes[1, i].set_title("Absolute difference from raw")
        for ax in axes[:, i]:
            ax.set(xscale="log", yscale="log", xlabel="f2 [kHz]", ylabel="f1 [kHz]")
    axes[1, 0].axis("off")
    axes[1, 0].text(.05, .6, "Shared color limits across ranks.\n\nEach map uses the same center point,\ntime interval and frequency grid.",
                    transform=axes[1, 0].transAxes, va="top")
    fig.colorbar(mesh, ax=axes[0, :], label=label, shrink=.8)
    fig.colorbar(error_mesh, ax=axes[1, 1:], label="Absolute difference" + (" [decades]" if quantity == "complex_bispectrum" else ""), shrink=.8)
    meta = cases[0][1]
    fig.suptitle(f"{meta['power']} / {meta['experiment']} — {TREATMENTS[treatment]}\n"
                 "Temporal mean removed; wavelet support edges excluded")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axis-samples", type=int, default=64)
    parser.add_argument("--maximum-sum", type=float, default=10000)
    args = parser.parse_args()
    if args.axis_samples < 2 or args.maximum_sum <= 0:
        parser.error("axis-samples must be >= 2 and maximum-sum must be positive")
    cases = load_cases(args.cases)
    rows = score_cases(cases, args.axis_samples, args.maximum_sum)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "rank_distances.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"cases": [{"path": str(path), "metadata": meta} for path, meta, _ in cases], "scores": rows}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with PdfPages(output / "rank_comparison.pdf") as pdf:
        fig = plot_scores(rows, f"sparse{args.axis_samples}")
        fig.savefig(output / "rank_errors.png", dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        for treatment in TREATMENTS:
            for quantity in ("complex_bispectrum", "bicoherence"):
                fig = plot_maps(cases, treatment, quantity)
                fig.savefig(output / f"{treatment}_{quantity}.png", dpi=160)
                pdf.savefig(fig)
                plt.close(fig)
    validations = [meta["local_raw_validation"] for _, meta, _ in cases if meta.get("local_raw_validation")]
    validation_note = " ".join(
        f"The input metadata records verification of {v['samples_checked']:,} center samples against {v['path']}."
        for v in validations
    )
    (output / "README.md").write_text(
        "# POD rank comparison with raw center spectra\n\n"
        "Open rank_comparison.pdf for error curves and maps. rank_distances.csv contains all scores.\n\n"
        "Scores reuse the repetition comparison's log-frequency area weights, common finite mask across raw and all ranks, and weighted MAE. "
        "The centered log-bispectrum score removes each map's weighted mean log magnitude; the other log score retains amplitude differences. "
        "Bicoherence is abs(mean(Q))/mean(abs(Q)), not its square. Lower distances mean closer agreement.\n\n"
        f"Dense maps and sparse{args.axis_samples} scores share the original frequency grid. "
        f"The limited sum domain is f1+f2 <= {args.maximum_sum:g} Hz. Both mean treatments are temporally demeaned. "
        "The retained spatial mean is shared between raw and POD, so inspect the mean-removed results to assess fluctuation reconstruction.\n\n"
        "These are descriptive errors for one measured repetition and its POD projection, not unseen-data prediction or statistical significance tests. "
        "Adding POD modes optimizes representation of the signal; these nonlinear spectral errors need not decrease monotonically.\n\n"
        "All inputs were checked for identical raw signals, raw spectra, timestamps, preprocessing and transform settings. "
        + validation_note + "\n\n"
        "Reproduce this overview:\n\n```bash\n" + shlex.join([sys.executable, str(Path(__file__).resolve()),
        *[str(c[0]) for c in cases], "--output-dir", str(output), "--axis-samples", str(args.axis_samples),
        "--maximum-sum", str(args.maximum_sum)]) + "\n```\n")
    print(f"Saved {output / 'rank_comparison.pdf'}")
    for row in rows:
        if row["grid"] == f"sparse{args.axis_samples}" and row["domain"] == "full":
            print(row)


if __name__ == "__main__":
    main()
