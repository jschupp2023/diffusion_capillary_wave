"""Create title-free presentation versions of the selected center-PSD plots.

The all-power figure is drawn from the existing raw PSD caches. The selected
raw/POD comparison reuses its raw cache and reconstructs only the requested
rank-10 POD center signal for 0p20 repetition 13.

Example
-------
    python -m data_analysis.psd.plot_psd_presentation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.psd.plot_raw_psds_by_power import discover, load_psd
from data_analysis.psd.pod_center_psd import (
    compute_welch,
    infer_sampling_frequency,
    reconstruct_point_signals,
)


DEFAULT_REDUCED_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
FIGURE_SIZE = (13.333, 7.5)
AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 22
LEGEND_SIZE = 16
DEFAULT_MIN_FREQUENCY_HZ = 70.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_REDUCED_ROOT,
        help=f"Reduced-data root (default: {DEFAULT_REDUCED_ROOT}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    parser.add_argument(
        "--min-frequency-hz",
        type=float,
        default=DEFAULT_MIN_FREQUENCY_HZ,
        help="Left edge of the logarithmic frequency axis (default: 70 Hz).",
    )
    return parser.parse_args()


def _style_axes(axis: plt.Axes, min_frequency_hz: float) -> None:
    axis.set_xlim(left=min_frequency_hz)
    axis.set_xlabel("Frequency [Hz]", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    axis.set_ylabel(
        r"PSD [$\mu$m$^2$/Hz]", fontsize=AXIS_LABEL_SIZE, labelpad=10
    )
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.6,
        length=7,
    )
    axis.tick_params(axis="both", which="minor", width=1.1, length=4)
    axis.grid(True, which="both", linestyle="--", linewidth=0.9, alpha=0.28)
    for spine in axis.spines.values():
        spine.set_linewidth(1.4)


def _power_label(power: str) -> str:
    return f"{float(power.replace('p', '.')):.2f} Vpp"


def plot_all_powers(
    cache_directory: Path,
    output_path: Path,
    dpi: int,
    min_frequency_hz: float,
) -> int:
    """Render all cached raw PSDs and the unchanged -17/6 reference guide."""
    grouped = discover(cache_directory)
    if not grouped:
        raise FileNotFoundError(f"No recognized raw PSD caches in {cache_directory}.")

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(grouped)))
    plotted = 0
    reference_levels: list[float] = []
    for (power, entries), color in zip(grouped.items(), colors, strict=True):
        power_plotted = False
        for entry in entries:
            frequency, density, _ = load_psd(entry.path)
            positive = (frequency > 0) & (density > 0)
            if not np.any(positive):
                continue
            axis.loglog(
                frequency[positive],
                density[positive],
                color=color,
                linewidth=1.1,
                alpha=0.52,
                label=_power_label(power) if not power_plotted else None,
            )
            power_plotted = True
            plotted += 1
            anchor_band = positive & (frequency >= 900) & (frequency <= 1100)
            if np.any(anchor_band):
                reference_levels.append(float(np.median(density[anchor_band])))

    if not plotted:
        plt.close(figure)
        raise ValueError("No positive raw PSD values were available.")
    if reference_levels:
        reference_amplitude = 20.0 * max(reference_levels)
        reference_frequency = np.geomspace(500.0, 10_000.0, 200)
        reference_density = reference_amplitude * (
            reference_frequency / 1000.0
        ) ** (-17 / 6)
        axis.loglog(
            reference_frequency,
            reference_density,
            color="black",
            linestyle="--",
            linewidth=3.0,
            zorder=10,
        )
        axis.annotate(
            r"$f^{-17/6}$ reference",
            xy=(1500.0, reference_amplitude * 1.5 ** (-17 / 6)),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=19,
            ha="left",
            va="bottom",
            zorder=11,
        )

    _style_axes(axis, min_frequency_hz)
    axis.legend(
        title="Input power",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE + 1,
        handlelength=2.5,
    )
    figure.subplots_adjust(left=0.13, right=0.81, bottom=0.17, top=0.97)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return plotted


def _load_raw_comparison_cache(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, int, int, str]:
    with np.load(path, allow_pickle=False) as cache:
        required = (
            "frequency_hz",
            "power_spectral_density",
            "sampling_frequency_hz",
            "nperseg",
            "noverlap",
            "signal_units",
        )
        missing = [name for name in required if name not in cache]
        if missing:
            raise KeyError(f"Raw cache {path} is missing: {', '.join(missing)}")
        return (
            np.asarray(cache["frequency_hz"], dtype=np.float64),
            np.asarray(cache["power_spectral_density"], dtype=np.float64),
            float(cache["sampling_frequency_hz"]),
            int(cache["nperseg"]),
            int(cache["noverlap"]),
            str(cache["signal_units"]),
        )


def plot_raw_pod_comparison(
    raw_cache_path: Path,
    pod_path: Path,
    output_path: Path,
    dpi: int,
    min_frequency_hz: float,
) -> None:
    """Render the selected rank-10 comparison with its original conventions."""
    (
        raw_frequency_axis,
        raw_density,
        raw_sampling_frequency,
        raw_nperseg,
        raw_noverlap,
        raw_units,
    ) = _load_raw_comparison_cache(raw_cache_path)

    (
        pod_time,
        _preprocessed_signal,
        spatial_mean,
        restored_signal,
        selected,
        reconstruction_rank,
        pod_units,
    ) = reconstruct_point_signals(pod_path, 10, None, None, 8192)
    pod_sampling_frequency, _ = infer_sampling_frequency(pod_time)
    restored_psd, pod_nperseg, pod_noverlap = compute_welch(
        restored_signal, pod_sampling_frequency, 8192, 0.5
    )
    mean_psd, mean_nperseg, mean_noverlap = compute_welch(
        spatial_mean, pod_sampling_frequency, 8192, 0.5
    )
    if (mean_nperseg, mean_noverlap) != (pod_nperseg, pod_noverlap):
        raise ValueError("POD and spatial-mean Welch settings do not match.")
    if (raw_nperseg, raw_noverlap) != (pod_nperseg, pod_noverlap):
        raise ValueError("Raw and POD Welch settings do not match.")
    if (pod_units or raw_units).strip().lower() not in {
        "micron",
        "microns",
        "micrometer",
        "micrometers",
    }:
        raise ValueError(f"Expected micron-valued center signals, got {pod_units!r}.")

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    curves = (
        (raw_frequency_axis, raw_density, "black", "raw experimental center"),
        (
            restored_psd.frequency,
            restored_psd.density,
            "C0",
            "POD center + spatial mean",
        ),
        (mean_psd.frequency, mean_psd.density, "C1", "spatial mean only"),
    )
    for frequency, density, color, label in curves:
        valid = (frequency > 0) & (density > 0) & np.isfinite(density)
        if not np.any(valid):
            raise ValueError(f"{label} PSD contains no positive values.")
        axis.loglog(
            frequency[valid],
            density[valid],
            color=color,
            linewidth=2.4,
            label=label,
        )

    _style_axes(axis, min_frequency_hz)
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.8,
    )
    details = (
        f"rank = {reconstruction_rank:,}; center (y, x) = "
        f"({selected.y_index}, {selected.x_index})\n"
        f"$f_s$ raw/POD = {raw_sampling_frequency:,.3f}/"
        f"{pod_sampling_frequency:,.3f} Hz\n"
        f"Welch nperseg = {pod_nperseg:,}, overlap = {pod_noverlap:,}"
    )
    axis.text(
        0.02,
        0.03,
        details,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.92},
    )
    figure.subplots_adjust(left=0.18, right=0.97, bottom=0.17, top=0.97)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def render(root: Path, dpi: int, min_frequency_hz: float = DEFAULT_MIN_FREQUENCY_HZ) -> list[Path]:
    comparison_directory = root / "center_psd_comparisons"
    raw_cache_directory = comparison_directory / "raw_psd_cache"
    presentation_directory = comparison_directory / "presentation"
    presentation_directory.mkdir(parents=True, exist_ok=True)

    all_powers_output = presentation_directory / (
        "all_powers__all_repetitions_raw_center_psd_presentation.png"
    )
    comparison_output = presentation_directory / (
        "0p20__Ca_ac_0p001660_rep13__r10__raw_vs_pod_center_psd_presentation.png"
    )
    plotted = plot_all_powers(
        raw_cache_directory,
        all_powers_output,
        dpi,
        min_frequency_hz,
    )
    plot_raw_pod_comparison(
        raw_cache_directory
        / "0p20__Ca_ac_0p001660_rep13__raw_center_psd.npz",
        root / "0p20/Ca_ac_0p001660_rep13/pod_2d_r1000.h5",
        comparison_output,
        dpi,
        min_frequency_hz,
    )
    print(f"Saved {all_powers_output} ({plotted} cached PSDs)")
    print(f"Saved {comparison_output}")
    return [all_powers_output, comparison_output]


def main() -> None:
    args = parse_args()
    if args.dpi < 1 or not np.isfinite(args.min_frequency_hz) or args.min_frequency_hz <= 0:
        raise SystemExit("ERROR: --dpi and --min-frequency-hz must be positive.")
    try:
        render(
            args.root.expanduser().resolve(),
            args.dpi,
            args.min_frequency_hz,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
