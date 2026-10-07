"""Overlay all cached raw center-point PSDs separately for each input power.

The input is the ``center_psd_comparisons`` directory produced by
``run_center_psd_comparison_batch.py``. Numerical PSDs are loaded from its
``raw_psd_cache`` subdirectory, grouped by the leading power label, and saved
as one log-log figure per power.

Example
-------
    python plot_raw_psds_by_power.py /path/to/center_psd_comparisons
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CACHE_PATTERN = re.compile(
    r"(?P<power>0p\d{1,2})__(?P<realization>Ca_ac.+)__raw_center_psd\.npz"
)
REPETITION_PATTERN = re.compile(r"_rep(?P<number>\d+)$")


@dataclass(frozen=True)
class CachedPsd:
    power: str
    realization: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_dir",
        type=Path,
        help="Directory containing the raw_psd_cache subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: <comparison-dir>/raw_psds_by_power).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Plot resolution (default: 200 dpi).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing figures.",
    )
    return parser.parse_args()


def _natural_key(text: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def discover(cache_directory: Path) -> dict[str, list[CachedPsd]]:
    grouped: defaultdict[str, list[CachedPsd]] = defaultdict(list)
    for path in cache_directory.glob("*.npz"):
        match = CACHE_PATTERN.fullmatch(path.name)
        if match is None:
            print(f"Skipping unrecognized cache filename: {path.name}", flush=True)
            continue
        cached = CachedPsd(
            power=match.group("power"),
            realization=match.group("realization"),
            path=path,
        )
        grouped[cached.power].append(cached)
    return {
        power: sorted(entries, key=lambda entry: _natural_key(entry.realization))
        for power, entries in sorted(
            grouped.items(), key=lambda item: _natural_key(item[0])
        )
    }


def repetition_label(realization: str) -> str:
    match = REPETITION_PATTERN.search(realization)
    return f"rep{match.group('number')}" if match else realization


def load_psd(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as cache:
        for name in ("frequency_hz", "power_spectral_density"):
            if name not in cache:
                raise KeyError(f"{path} is missing {name!r}.")
        frequency = np.asarray(cache["frequency_hz"], dtype=np.float64)
        density = np.asarray(cache["power_spectral_density"], dtype=np.float64)
        signal_units = str(cache["signal_units"]) if "signal_units" in cache else ""
    if frequency.ndim != 1 or density.shape != frequency.shape:
        raise ValueError(
            f"Invalid frequency/PSD shapes in {path}: "
            f"{frequency.shape}, {density.shape}."
        )
    if not np.isfinite(frequency).all() or not np.isfinite(density).all():
        raise ValueError(f"Nonfinite frequency or PSD values in {path}.")
    return frequency, density, signal_units


def plot_power_group(
    power: str,
    entries: list[CachedPsd],
    output_path: Path,
    dpi: int,
) -> int:
    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(entries)))
    plotted = 0
    signal_units = ""
    for entry, color in zip(entries, colors, strict=True):
        try:
            frequency, density, current_units = load_psd(entry.path)
        except (KeyError, OSError, ValueError) as error:
            print(f"Skipping {entry.path.name}: {error}", flush=True)
            continue
        positive = (frequency > 0) & (density > 0)
        if not np.any(positive):
            print(f"Skipping {entry.path.name}: no positive PSD values", flush=True)
            continue
        axis.loglog(
            frequency[positive],
            density[positive],
            color=color,
            linewidth=1.0,
            alpha=0.82,
            label=repetition_label(entry.realization),
        )
        signal_units = signal_units or current_units
        plotted += 1

    if not plotted:
        plt.close(figure)
        raise ValueError(f"No valid PSDs were available for power {power}.")
    density_units = f"{signal_units}$^2$/Hz" if signal_units else "value$^2$/Hz"
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title(f"Raw experimental center-point PSDs — input power {power}")
    axis.grid(True, which="both", linestyle="--", alpha=0.28)
    axis.legend(
        title="Repetition",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=1,
        fontsize=8,
        title_fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return plotted


def plot_all_power_groups(
    grouped: dict[str, list[CachedPsd]],
    output_path: Path,
    dpi: int,
) -> int:
    """Plot every repetition with a reference slope of -17/6."""
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    colors = plt.get_cmap("turbo")(np.linspace(0.02, 0.98, len(grouped)))
    plotted = 0
    signal_units = ""
    reference_levels: list[float] = []
    for (power, entries), color in zip(grouped.items(), colors, strict=True):
        power_plotted = False
        for entry in entries:
            try:
                frequency, density, current_units = load_psd(entry.path)
            except (KeyError, OSError, ValueError) as error:
                print(f"Skipping {entry.path.name}: {error}", flush=True)
                continue
            positive = (frequency > 0) & (density > 0)
            if not np.any(positive):
                print(
                    f"Skipping {entry.path.name}: no positive PSD values",
                    flush=True,
                )
                continue
            axis.loglog(
                frequency[positive],
                density[positive],
                color=color,
                linewidth=0.75,
                alpha=0.48,
                label=power if not power_plotted else None,
            )
            power_plotted = True
            signal_units = signal_units or current_units
            plotted += 1
            anchor_band = positive & (frequency >= 900) & (frequency <= 1100)
            if np.any(anchor_band):
                reference_levels.append(float(np.median(density[anchor_band])))

    if not plotted:
        plt.close(figure)
        raise ValueError("No valid PSDs were available for the combined plot.")
    if reference_levels:
        # Position the guide above the spectra; its amplitude is not a fit.
        reference_amplitude = 20.0 * max(reference_levels)
        reference_frequency = np.geomspace(500.0, 10_000.0, 200)
        reference_density = reference_amplitude * (reference_frequency / 1000.0) ** (-17 / 6)
        axis.loglog(
            reference_frequency, reference_density,
            color="black", linestyle="--", linewidth=2.0, zorder=10,
        )
        axis.annotate(
            r"$f^{-17/6}$ reference",
            xy=(1500.0, reference_amplitude * 1.5 ** (-17 / 6)),
            xytext=(0, 10), textcoords="offset points",
            fontsize=11, ha="left", va="bottom", zorder=11,
        )
    density_units = f"{signal_units}$^2$/Hz" if signal_units else "value$^2$/Hz"
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title("Raw experimental center-point PSDs — all experiments")
    axis.grid(True, which="both", linestyle="--", alpha=0.24)
    axis.legend(
        title="Input power",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return plotted


def run(args: argparse.Namespace) -> list[Path]:
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")
    comparison_directory = args.comparison_dir.expanduser().resolve()
    cache_directory = comparison_directory / "raw_psd_cache"
    if not cache_directory.is_dir():
        raise FileNotFoundError(f"Raw PSD cache directory does not exist: {cache_directory}")
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else comparison_directory / "raw_psds_by_power"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    grouped = discover(cache_directory)
    if not grouped:
        raise FileNotFoundError(f"No recognized raw PSD caches found in {cache_directory}")
    print(
        f"Discovered {sum(map(len, grouped.values()))} PSD cache(s) across "
        f"{len(grouped)} input power(s).",
        flush=True,
    )
    outputs: list[Path] = []
    for power, entries in grouped.items():
        output_path = output_directory / f"{power}__all_repetitions_raw_center_psd.png"
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path}", flush=True)
            outputs.append(output_path)
            continue
        plotted = plot_power_group(power, entries, output_path, args.dpi)
        print(f"Saved {output_path} ({plotted} repetitions)", flush=True)
        outputs.append(output_path)

    combined_path = output_directory / "all_powers__all_repetitions_raw_center_psd.png"
    if combined_path.exists() and not args.overwrite:
        print(f"Skipping existing {combined_path}", flush=True)
    else:
        plotted = plot_all_power_groups(grouped, combined_path, args.dpi)
        print(
            f"Saved {combined_path} ({plotted} curves, {len(grouped)} powers)",
            flush=True,
        )
    outputs.append(combined_path)
    return outputs


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
