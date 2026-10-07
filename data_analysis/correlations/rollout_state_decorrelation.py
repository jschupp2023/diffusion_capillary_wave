"""Compare generated and reference POD-state decorrelation in a saved rollout.

The diagnostic demeans every path separately, averages its biased FFT
autocorrelation across paths, and reports the first start of ``K`` consecutive
lags satisfying ``abs(ACF) < threshold``.  Coordinate zero (the spatial mean)
is excluded when present, so the aggregate contains POD modes only.

Example
-------
python -m data_analysis.correlations.rollout_state_decorrelation \
    runs/comparison/velocity_lag1/rollout_50000_native
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from data_analysis.rollout import Rollout


def mean_path_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Return the path-mean ACF for a finite [path, time, coordinate] array."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("ACF input must be a finite [path, time, coordinate] array.")
    if not 1 <= max_lag < values.shape[1]:
        raise ValueError("max_lag must lie between 1 and the record length minus 1.")
    centered = values - values.mean(axis=1, keepdims=True)
    nfft = next_fast_len(2 * values.shape[1] - 1)
    spectrum = rfft(centered, n=nfft, axis=1)
    products = irfft(spectrum * spectrum.conj(), n=nfft, axis=1)[:, :max_lag + 1]
    energy = np.sum(centered * centered, axis=1)
    acf = np.divide(
        products, energy[:, None, :], out=np.full_like(products, np.nan),
        where=energy[:, None, :] > 0,
    )
    return np.nanmean(acf, axis=0)


def persistent_crossing(values: np.ndarray, threshold: float, consecutive: int) -> int | None:
    """First lag starting ``consecutive`` values with absolute ACF below threshold."""
    good = np.abs(np.asarray(values)[1:]) < threshold
    if len(good) < consecutive:
        return None
    counts = np.convolve(good.astype(np.int64), np.ones(consecutive, dtype=np.int64), mode="valid")
    hits = np.flatnonzero(counts == consecutive)
    return int(hits[0] + 1) if len(hits) else None


def population_summary(acf: np.ndarray, threshold: float, consecutive: int,
                       max_lag: int) -> dict:
    crossings = [persistent_crossing(acf[:, index], threshold, consecutive)
                 for index in range(acf.shape[1])]
    censored = np.asarray([max_lag + 1 if value is None else value for value in crossings])
    median = float(np.median(censored))
    return {
        "persistent_crossing_frames_by_mode": crossings,
        "median_persistent_crossing_frames": None if median > max_lag else median,
        "median_is_right_censored": bool(median > max_lag),
        "unresolved_mode_count": int(sum(value is None for value in crossings)),
    }


def analyze(rollout_path: Path, *, frames: int = 5001, max_lag: int = 2500,
            threshold: float = .1, consecutive: int = 100,
            output: Path | None = None, overwrite: bool = False) -> Path:
    rollout = Rollout(rollout_path)
    available = min(rollout.generated.shape[-2], rollout.reference.shape[-2])
    if frames < 2 or frames > available:
        raise ValueError(f"Requested {frames} frames, but rollout provides {available}.")
    if max_lag + consecutive > frames:
        raise ValueError("max_lag + consecutive must not exceed the analyzed frame count.")

    coordinate_start = 1 if rollout.include_spatial_mean else 0
    generated = rollout.generated[..., :frames, coordinate_start:].reshape(
        -1, frames, rollout.generated.shape[-1] - coordinate_start)
    reference = rollout.reference[..., :frames, coordinate_start:]
    if generated.shape[-1] != rollout.rank or reference.shape[-1] != rollout.rank:
        raise ValueError("Selected rollout coordinates do not match the saved POD rank.")

    destination = output or rollout.path.parent / f"state_decorrelation_k{consecutive}"
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Nonempty output directory {destination}; use --overwrite.")
    destination.mkdir(parents=True, exist_ok=True)

    acfs = {
        "reference": mean_path_acf(reference, max_lag),
        "generated": mean_path_acf(generated, max_lag),
    }
    populations = {
        "reference": population_summary(acfs["reference"], threshold, consecutive, max_lag),
        "generated": population_summary(acfs["generated"], threshold, consecutive, max_lag),
    }
    generated_median = populations["generated"]["median_persistent_crossing_frames"]
    reference_median = populations["reference"]["median_persistent_crossing_frames"]
    ratio = None if generated_median is None or reference_median in (None, 0) else generated_median / reference_median
    report = {
        "rollout": str(rollout.path.parent.resolve()),
        "definition": (
            "POD-state ACF; each path demeaned separately; biased FFT lag products divided by "
            "lag-zero energy; path ACFs averaged before threshold crossing"
        ),
        "coordinate_population": "POD modes only; spatial-mean coordinate excluded",
        "frames": frames,
        "max_lag_frames": max_lag,
        "threshold_absolute_acf": threshold,
        "required_consecutive_lags": consecutive,
        "sampling_frequency_hz": rollout.fs,
        "generated_path_count": len(generated),
        "reference_path_count": len(reference),
        "populations": populations,
        "generated_over_reference_median_crossing": ratio,
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(destination / "acf.npz", lag_frames=np.arange(max_lag + 1), **acfs)

    selected_modes = [mode for mode in (1, 2, 3, 5, 10, 20) if mode <= rollout.rank]
    lines = [
        "# POD-state decorrelation", "",
        f"Persistent crossing: {consecutive} consecutive lags with "
        f"`abs(ACF) < {threshold:g}`; first {frames} frames used.", "",
        "| Population | Median crossing [frames] | Unresolved modes |",
        "|---|---:|---:|",
    ]
    for name in ("reference", "generated"):
        item = populations[name]
        value = item["median_persistent_crossing_frames"]
        rendered = f">{max_lag}" if value is None else f"{value:g}"
        lines.append(f"| {name.capitalize()} | {rendered} | {item['unresolved_mode_count']} |")
    lines.extend(["", f"**Generated/reference:** {('unresolved' if ratio is None else f'{ratio:.4g}')}×", "",
                  "| Population | " + " | ".join(f"POD {mode}" for mode in selected_modes) + " |",
                  "|---|" + "---:|" * len(selected_modes)])
    for name in ("reference", "generated"):
        values = populations[name]["persistent_crossing_frames_by_mode"]
        rendered = [f">{max_lag}" if values[mode - 1] is None else str(values[mode - 1])
                    for mode in selected_modes]
        lines.append(f"| {name.capitalize()} | " + " | ".join(rendered) + " |")
    (destination / "quick_summary.md").write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(len(selected_modes[:3]), 1, figsize=(8, 8), sharex=True,
                               constrained_layout=True, squeeze=False)
    for axis, mode in zip(axes[:, 0], selected_modes[:3], strict=True):
        for name, color in (("reference", "black"), ("generated", "tab:red")):
            axis.plot(np.arange(max_lag + 1), acfs[name][:, mode - 1], color=color, label=name)
        axis.axhline(threshold, color="gray", linestyle=":", linewidth=.7)
        axis.axhline(-threshold, color="gray", linestyle=":", linewidth=.7)
        axis.axhline(0, color="gray", linewidth=.5)
        axis.set(title=f"POD {mode}", ylabel="ACF", xlim=(0, min(1200, max_lag)))
        axis.grid(alpha=.15)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Lag [frames]")
    figure.suptitle(f"POD-state decorrelation: {rollout.path.parent.name}")
    figure.savefig(destination / "acf_modes1_3.png", dpi=160)
    plt.close(figure)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--frames", type=int, default=5001)
    parser.add_argument("--max-lag", type=int, default=2500)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--consecutive", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(analyze(args.rollout, frames=args.frames, max_lag=args.max_lag,
                  threshold=args.threshold, consecutive=args.consecutive,
                  output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
