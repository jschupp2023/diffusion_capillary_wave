"""Compare experimental and rank-truncated POD PSDs using band power.

The positional input is one reduced-data repetition folder. The script loads
its cached raw center signal, reconstructs the POD center using the requested
number of modes, restores the instantaneous spatial mean, and integrates both
Welch PSDs over logarithmic frequency bands.

Example
-------
    python compare_psd_band_power.py \
        /path/to/reduced_data/0p20/Ca_ac_0p001660_rep13 --rank 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_analysis.psd.pod_center_psd import (
    compute_welch,
    infer_sampling_frequency,
    reconstruct_point_signals,
    resolve_pod_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Reduced-data folder for one Ca_ac... repetition.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help=(
            "Number of leading POD modes used; 0 compares the saved "
            "instantaneous spatial mean by itself."
        ),
    )
    parser.add_argument(
        "--pod-rank",
        type=int,
        default=1_000,
        help="Rank in the stored POD filename (default: 1000).",
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        help="Raw center-signal NPZ (default: inferred from the folder).",
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
        "--bins-per-decade",
        type=int,
        choices=(1, 2),
        default=2,
        help="Use whole- or half-decade frequency bands (default: 2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8_192,
        help="Coefficient rows reconstructed per batch (default: 8192).",
    )
    return parser.parse_args()


def infer_raw_cache(experiment_dir: Path) -> Path:
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment folder does not exist: {experiment_dir}")
    condition = experiment_dir.parent.name
    reduced_root = experiment_dir.parent.parent
    return (
        reduced_root
        / "center_psd_comparisons"
        / "raw_psd_cache"
        / f"{condition}__{experiment_dir.name}__raw_center_psd.npz"
    )


def load_raw_signal(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Raw center cache does not exist: {cache_path}\n"
            "Pass its location with --raw-cache."
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        missing = [name for name in ("time", "center_signal") if name not in cache]
        if missing:
            raise KeyError(f"Raw cache is missing: {', '.join(missing)}")
        time = np.asarray(cache["time"], dtype=np.float64)
        signal = np.asarray(cache["center_signal"], dtype=np.float64)
    if time.ndim != 1 or signal.shape != time.shape:
        raise ValueError(
            f"Raw time/signal shapes are incompatible: {time.shape}, {signal.shape}."
        )
    return time, signal


def logarithmic_edges(
    lower_frequency: float,
    upper_frequency: float,
    bins_per_decade: int,
) -> np.ndarray:
    exponent_step = 1.0 / bins_per_decade
    exponents = np.arange(
        np.log10(lower_frequency),
        np.log10(upper_frequency) + exponent_step,
        exponent_step,
    )
    edges = np.power(10.0, exponents)
    edges = edges[edges < upper_frequency]
    return np.append(edges, upper_frequency)


def integrate_band(
    frequency: np.ndarray,
    density: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Trapezoidal integral including interpolated values at both boundaries."""
    interior = (frequency > lower) & (frequency < upper)
    band_frequency = np.concatenate(([lower], frequency[interior], [upper]))
    band_density = np.concatenate(
        (
            [np.interp(lower, frequency, density)],
            density[interior],
            [np.interp(upper, frequency, density)],
        )
    )
    return float(np.trapezoid(band_density, band_frequency))


def validate_args(args: argparse.Namespace) -> None:
    if args.rank < 0:
        raise ValueError("--rank must be non-negative.")
    if args.pod_rank < 1:
        raise ValueError("--pod-rank must be positive.")
    if args.nperseg < 2:
        raise ValueError("--nperseg must be at least 2.")
    if not 0 <= args.overlap_fraction < 1:
        raise ValueError("--overlap-fraction must lie in [0, 1).")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    experiment_dir = args.experiment_dir.expanduser().resolve()
    pod_path = resolve_pod_file(experiment_dir, args.pod_rank)
    raw_cache = (
        args.raw_cache.expanduser().resolve()
        if args.raw_cache is not None
        else infer_raw_cache(experiment_dir)
    )
    raw_time, raw_signal = load_raw_signal(raw_cache)

    (
        pod_time,
        _preprocessed_signal,
        _spatial_mean,
        reconstructed_signal,
        selected,
        used_rank,
        _signal_units,
    ) = reconstruct_point_signals(
        pod_path,
        args.rank,
        None,
        None,
        args.batch_size,
    )
    raw_sampling_frequency, _ = infer_sampling_frequency(raw_time)
    pod_sampling_frequency, _ = infer_sampling_frequency(pod_time)
    raw_psd, raw_nperseg, raw_noverlap = compute_welch(
        raw_signal,
        raw_sampling_frequency,
        args.nperseg,
        args.overlap_fraction,
    )
    reconstructed_psd, pod_nperseg, pod_noverlap = compute_welch(
        reconstructed_signal,
        pod_sampling_frequency,
        args.nperseg,
        args.overlap_fraction,
    )

    maximum_frequency = min(
        float(raw_psd.frequency[-1]), float(reconstructed_psd.frequency[-1])
    )
    if maximum_frequency <= 10.0:
        raise ValueError("The PSD does not extend above the 10 Hz lower bound.")
    edges = logarithmic_edges(10.0, maximum_frequency, args.bins_per_decade)

    print()
    print(f"Experiment:       {experiment_dir.parent.name}/{experiment_dir.name}")
    print(f"POD file:         {pod_path}")
    print(f"Raw cache:        {raw_cache}")
    print(f"Reconstruction:   rank {used_rank} + instantaneous spatial mean")
    print(f"Center (y, x):    ({selected.y_index}, {selected.x_index})")
    print(
        f"Welch:            nperseg={pod_nperseg}, overlap={pod_noverlap}; "
        f"raw nperseg={raw_nperseg}, overlap={raw_noverlap}"
    )
    print("Ratio definition: reconstructed band power / experimental band power")
    print()
    print(
        f"{'Frequency band [Hz]':>23}  {'Experimental':>14}  "
        f"{'Reconstructed':>14}  {'Ratio':>9}  {'Error':>9}"
    )
    print("-" * 79)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        experimental_power = integrate_band(
            raw_psd.frequency, raw_psd.density, lower, upper
        )
        reconstructed_power = integrate_band(
            reconstructed_psd.frequency,
            reconstructed_psd.density,
            lower,
            upper,
        )
        ratio = reconstructed_power / experimental_power
        relative_error = 100.0 * (ratio - 1.0)
        band_label = f"{lower:8.1f}–{upper:8.1f}"
        print(
            f"{band_label:>23}  {experimental_power:14.6e}  "
            f"{reconstructed_power:14.6e}  {ratio:9.4f}  "
            f"{relative_error:+8.1f}%"
        )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
