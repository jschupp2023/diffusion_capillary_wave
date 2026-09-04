"""Compare raw and POD-reconstructed center-point PSDs for one experiment.

The saved figure contains the raw experimental center PSD, the POD center
reconstruction with its instantaneous spatial mean restored, and the spatial
mean PSD by itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pod_center_psd import (
    compute_welch,
    infer_sampling_frequency,
    reconstruct_point_signals,
)
from raw_center_psd import read_center_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_input", type=Path, help="Raw experimental HDF5 file.")
    parser.add_argument("pod_input", type=Path, help="Reduced POD HDF5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--label",
        help="Experiment label used in the title (default: POD parent folder).",
    )
    parser.add_argument(
        "--reconstruction-rank",
        type=int,
        help="Number of POD modes to use (default: all stored modes).",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=8_192,
        help="Samples per Welch segment (default: 8192).",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=0.5,
        help="Fractional Welch overlap (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8_192,
        help="POD coefficient rows reconstructed per batch (default: 8192).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output resolution (default: 200 dpi).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output image.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.reconstruction_rank is not None and args.reconstruction_rank < 1:
        raise ValueError("--reconstruction-rank must be positive.")
    if args.nperseg < 2:
        raise ValueError("--nperseg must be at least 2.")
    if not 0 <= args.overlap_fraction < 1:
        raise ValueError("--overlap-fraction must lie in [0, 1).")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    raw_path = args.raw_input.expanduser().resolve()
    pod_path = args.pod_input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for label, path in (("Raw input", raw_path), ("POD input", pod_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        raw_time,
        raw_signal,
        raw_center_y,
        raw_center_x,
        raw_x_coordinate,
        raw_y_coordinate,
        raw_units,
    ) = read_center_signal(raw_path)
    (
        pod_time,
        _preprocessed_signal,
        spatial_mean,
        restored_signal,
        selected,
        reconstruction_rank,
        pod_signal_units,
    ) = reconstruct_point_signals(
        pod_path,
        args.reconstruction_rank,
        None,
        None,
        args.batch_size,
    )

    raw_frequency, raw_jitter = infer_sampling_frequency(raw_time)
    pod_frequency, pod_jitter = infer_sampling_frequency(pod_time)
    raw_psd, raw_nperseg, raw_noverlap = compute_welch(
        raw_signal, raw_frequency, args.nperseg, args.overlap_fraction
    )
    restored_psd, pod_nperseg, pod_noverlap = compute_welch(
        restored_signal, pod_frequency, args.nperseg, args.overlap_fraction
    )
    mean_psd, mean_nperseg, mean_noverlap = compute_welch(
        spatial_mean, pod_frequency, args.nperseg, args.overlap_fraction
    )
    if (mean_nperseg, mean_noverlap) != (pod_nperseg, pod_noverlap):
        raise ValueError("POD and spatial-mean Welch settings do not match.")

    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    curves = (
        (raw_psd, "black", "raw experimental center"),
        (restored_psd, "C0", "POD center + spatial mean"),
        (mean_psd, "C1", "spatial mean only"),
    )
    for result, color, curve_label in curves:
        valid = (
            (result.frequency > 0)
            & (result.density > 0)
            & np.isfinite(result.density)
        )
        if not np.any(valid):
            raise ValueError(f"{curve_label} PSD contains no positive values.")
        axis.loglog(
            result.frequency[valid],
            result.density[valid],
            color=color,
            linewidth=1.25,
            label=curve_label,
        )

    signal_units = pod_signal_units or raw_units.get("z", "")
    density_units = f"{signal_units}$^2$/Hz" if signal_units else "value$^2$/Hz"
    experiment_label = args.label or pod_path.parent.name
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title(
        "Raw versus POD-reconstructed center-point PSD\n"
        f"{experiment_label}"
    )
    axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axis.legend(loc="best", frameon=False)
    details = (
        f"rank = {reconstruction_rank:,}; center (y, x) = "
        f"({selected.y_index}, {selected.x_index})\n"
        f"$f_s$ raw/POD = {raw_frequency:,.3f}/{pod_frequency:,.3f} Hz\n"
        f"Welch nperseg = {pod_nperseg:,}, overlap = {pod_noverlap:,}"
    )
    axis.text(
        0.02,
        0.03,
        details,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(
        f"Raw center: (y, x) = ({raw_center_y}, {raw_center_x}), "
        f"coordinate = ({raw_x_coordinate:.3f}, {raw_y_coordinate:.3f}); "
        f"time jitter = {raw_jitter:.3%}.",
        flush=True,
    )
    print(
        f"POD center: (y, x) = ({selected.y_index}, {selected.x_index}); "
        f"time jitter = {pod_jitter:.3%}.",
        flush=True,
    )
    print(
        f"Raw Welch: nperseg={raw_nperseg:,}, overlap={raw_noverlap:,}.",
        flush=True,
    )
    print(f"Saved {output_path}", flush=True)
    return output_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
