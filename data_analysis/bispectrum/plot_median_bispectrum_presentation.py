"""Render title-free presentation panels from cached repetition bispectra.

The script reproduces the pointwise median ``log10(abs(B))`` panel from the
saved repetition-comparison caches.  It does not recompute wavelet transforms
or bispectra.

Example
-------
    python -m data_analysis.bispectrum.plot_median_bispectrum_presentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COMPARISON_DIR = Path(
    "/home/jonas/ucsd_thesis/reduced_data/"
    "bispectral_repetition_comparisons/"
    "all_powers_ns512_without_mean_sparse32"
)
DEFAULT_POWERS = ("0p08", "0p35")
FIGURE_SIZE = (7.4, 6.2)
AXIS_LABEL_SIZE = 25
TICK_LABEL_SIZE = 20
COLORBAR_LABEL_SIZE = 22
COLORBAR_TICK_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help=f"Saved repetition-comparison directory (default: {DEFAULT_COMPARISON_DIR}).",
    )
    parser.add_argument(
        "--powers",
        nargs="+",
        default=list(DEFAULT_POWERS),
        help="Power labels to render (default: 0p08 0p35).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300 dpi)."
    )
    return parser.parse_args()


def load_median_log_bispectrum(
    manifest_path: Path, power: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one power's caches and return its pointwise median log magnitude."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"Manifest must contain a list: {manifest_path}")

    cases = [entry for entry in manifest if entry.get("power") == power]
    cases.sort(key=lambda entry: int(entry["rep"]))
    if not cases:
        raise ValueError(f"No entries for {power} in {manifest_path}")

    frequency: np.ndarray | None = None
    log_maps: list[np.ndarray] = []
    repetitions: set[int] = set()
    for case in cases:
        repetition = int(case["rep"])
        if repetition in repetitions:
            raise ValueError(f"Duplicate repetition {repetition} for {power}")
        repetitions.add(repetition)

        cache_path = Path(case["spectra"]["without_mean"])
        with np.load(cache_path, allow_pickle=False) as archive:
            this_frequency = np.asarray(archive["frequency_hz"], dtype=np.float64)
            magnitude = np.abs(archive["complex_bispectrum"])

        if frequency is None:
            frequency = this_frequency
        elif not np.allclose(frequency, this_frequency, rtol=1e-12, atol=1e-9):
            raise ValueError(f"Frequency grids differ for {power}: {cache_path}")
        if magnitude.shape != (len(this_frequency), len(this_frequency)):
            raise ValueError(f"Unexpected bispectrum shape in {cache_path}")

        log_magnitude = np.full_like(magnitude, np.nan, dtype=np.float64)
        np.log10(magnitude, out=log_magnitude, where=magnitude > 0)
        log_maps.append(log_magnitude)

    assert frequency is not None
    median = np.nanmedian(np.asarray(log_maps), axis=0)
    if not np.isfinite(median).any():
        raise ValueError(f"Median bispectrum contains no finite values for {power}")
    return frequency, median, len(cases)


def plot_median(
    frequency_hz: np.ndarray,
    median_log_bispectrum: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    """Save one title-free median-bispectrum panel with presentation typography."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    frequency_khz = frequency_hz / 1_000.0
    mesh = axis.pcolormesh(
        frequency_khz,
        frequency_khz,
        median_log_bispectrum,
        shading="auto",
        rasterized=True,
        cmap="magma",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$f_2$ [kHz]", fontsize=AXIS_LABEL_SIZE, labelpad=9)
    axis.set_ylabel(r"$f_1$ [kHz]", fontsize=AXIS_LABEL_SIZE, labelpad=9)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.6,
        length=7,
    )
    axis.tick_params(axis="both", which="minor", width=1.1, length=4)
    for spine in axis.spines.values():
        spine.set_linewidth(1.4)

    colorbar = figure.colorbar(mesh, ax=axis, pad=0.035, fraction=0.06)
    colorbar.set_label(
        r"Median $\log_{10}|B|$",
        fontsize=COLORBAR_LABEL_SIZE,
        labelpad=12,
    )
    colorbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        width=1.4,
        length=6,
    )
    colorbar.outline.set_linewidth(1.3)

    figure.subplots_adjust(left=0.17, right=0.83, bottom=0.18, top=0.98)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def render(comparison_directory: Path, powers: list[str], dpi: int) -> list[Path]:
    manifest_path = comparison_directory / "manifest.json"
    output_directory = comparison_directory.parent / "presentation/median_bispectrum"
    output_directory.mkdir(parents=True, exist_ok=True)

    outputs = []
    for power in powers:
        frequency, median, repetition_count = load_median_log_bispectrum(
            manifest_path, power
        )
        output_path = output_directory / (
            f"without_mean_{power}_median_log_bispectrum_presentation.png"
        )
        plot_median(frequency, median, output_path, dpi)
        outputs.append(output_path)
        print(
            f"Saved {output_path} "
            f"({repetition_count} repetitions, {len(frequency)} frequency samples)"
        )
    return outputs


def main() -> None:
    args = parse_args()
    if args.dpi < 1:
        raise SystemExit("ERROR: --dpi must be positive.")
    if len(set(args.powers)) != len(args.powers):
        raise SystemExit("ERROR: --powers contains duplicate labels.")
    try:
        render(
            args.comparison_dir.expanduser().resolve(),
            args.powers,
            args.dpi,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
