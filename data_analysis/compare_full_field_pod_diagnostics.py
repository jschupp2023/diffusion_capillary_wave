"""Compare matched full-field diagnostics with rank-10 and rank-200 POD fields.

All three populations represent the same production preprocessing: 50 Hz
full-field temporal high-pass filtering, temporal-mean-field subtraction, and
retention of the instantaneous spatial mean.  Welch PSDs are computed within
each recording and repetitions are aggregated only afterward.

The full-field archive contains sampled exact and quadratic capillary energy.
POD quadratic energy is evaluated from the complete coefficient trajectory;
POD exact energy is the full quadratic mean multiplied by the paired sampled
exact/quadratic ratio.  Lines are medians and bands are the 10th--90th
percentiles across the same 16 repetitions per forcing condition.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.energy.plot_clean_reduced_capillary_energy import read_energy_rows


METHODS = ("full", "rank200", "rank10")
METHOD_LABELS = {
    "full": "Full field",
    "rank200": "Rank 200",
    "rank10": "Rank 10",
}
METHOD_COLORS = {
    "full": "#202020",
    "rank200": "#2878B5",
    "rank10": "#E07A1F",
}
METHOD_MARKERS = {"full": "o", "rank200": "s", "rank10": "^"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-diagnostics", type=Path, required=True)
    parser.add_argument("--rank200-psd", type=Path, required=True)
    parser.add_argument("--rank10-psd", type=Path, required=True)
    parser.add_argument("--rank200-energy-dir", type=Path, required=True)
    parser.add_argument("--rank10-energy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-frequency-hz", type=float, default=50.0)
    parser.add_argument("--max-frequency-hz", type=float)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.min_frequency_hz <= 0 or args.dpi < 1:
        parser.error("Minimum frequency and DPI must be positive.")
    if (
        args.max_frequency_hz is not None
        and args.max_frequency_hz <= args.min_frequency_hz
    ):
        parser.error("Maximum frequency must exceed minimum frequency.")
    return args


def _power_value(power: str) -> float:
    return float(power.replace("p", "."))


def _quantiles(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return q10, median, q90 in that order."""
    return np.quantile(np.asarray(values, dtype=float), [0.1, 0.5, 0.9], axis=axis)


def load_full_field(
    path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[tuple[str, int], dict[str, float]], dict]:
    psds: dict[str, list[np.ndarray]] = defaultdict(list)
    energy: dict[tuple[str, int], dict[str, float]] = {}
    metadata: dict[str, object] = {"recordings": 0, "preprocessing_modes": {}}
    reference_frequency: np.ndarray | None = None
    modes: defaultdict[str, int] = defaultdict(int)
    with h5py.File(path, "r") as archive:
        configuration = json.loads(archive.attrs["analysis_configuration"])
        for power, power_group in archive["recordings"].items():
            ordered = sorted(
                power_group.values(),
                key=lambda group: int(json.loads(group.attrs["summary_json"])["repetition"]),
            )
            for group in ordered:
                summary = json.loads(group.attrs["summary_json"])
                repetition = int(summary["repetition"])
                psd = group["center_psd"]
                frequency = np.asarray(psd["frequency_hz"], dtype=float)
                density = np.asarray(psd["power_spectral_density"], dtype=float)
                if reference_frequency is None:
                    reference_frequency = frequency
                elif not np.allclose(reference_frequency, frequency, rtol=0, atol=1e-10):
                    raise ValueError(f"Full-field frequency mismatch: {power}/rep{repetition}")
                if not np.isfinite(density).all() or np.any(density < 0):
                    raise ValueError(f"Invalid full-field PSD: {power}/rep{repetition}")
                psds[power].append(density)
                capillary = group["capillary_energy"]
                energy[(power, repetition)] = {
                    "quadratic_J": float(capillary.attrs["Ebar_quadratic_J"]),
                    "exact_J": float(capillary.attrs["Ebar_exact_J"]),
                }
                modes[str(summary["preprocessing_mode"])] += 1
                metadata["recordings"] = int(metadata["recordings"]) + 1
    if reference_frequency is None:
        raise ValueError(f"No full-field recordings in {path}")
    metadata["preprocessing_modes"] = dict(modes)
    metadata["configuration"] = configuration
    return reference_frequency, {key: np.stack(value) for key, value in psds.items()}, energy, metadata


def load_rank_psd(path: Path, expected_rank: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as archive:
        rank = int(archive["reconstruction_rank"])
        if rank != expected_rank:
            raise ValueError(f"{path} contains rank {rank}, expected {expected_rank}.")
        frequency = np.asarray(archive["frequency_hz"], dtype=float)
        conditions = [str(value) for value in archive["conditions"]]
        values = np.asarray(archive["psd_by_condition"], dtype=float)
    if values.shape[:2] != (len(conditions), 16) or values.shape[2:] != frequency.shape:
        raise ValueError(f"Unexpected rank-{expected_rank} PSD shape: {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid rank-{expected_rank} PSD values.")
    return frequency, dict(zip(conditions, values, strict=True))


def load_rank_energy(path: Path, rank: int) -> dict[tuple[str, int], dict[str, float]]:
    rows = read_energy_rows(path, rank)
    result = {
        (str(row["power"]), int(row["repetition"])): {
            "quadratic_J": float(row["quadratic_J"]),
            "exact_J": float(row["exact_J"]),
        }
        for row in rows
    }
    if len(result) != 128:
        raise ValueError(f"Expected 128 rank-{rank} energy rows, found {len(result)}.")
    return result


def validate_populations(
    frequency: dict[str, np.ndarray],
    psd: dict[str, dict[str, np.ndarray]],
    energy: dict[str, dict[tuple[str, int], dict[str, float]]],
) -> list[str]:
    conditions = sorted(psd["full"], key=_power_value)
    expected = {condition: 16 for condition in conditions}
    for method in METHODS:
        if sorted(psd[method], key=_power_value) != conditions:
            raise ValueError(f"PSD conditions differ for {method}.")
        counts = {condition: len(psd[method][condition]) for condition in conditions}
        if counts != expected:
            raise ValueError(f"PSD repetition counts differ for {method}: {counts}")
        if not np.allclose(frequency["full"], frequency[method], rtol=0, atol=1e-10):
            raise ValueError(f"Frequency grid differs for {method}.")
    reference_keys = set(energy["full"])
    for method in METHODS:
        if set(energy[method]) != reference_keys:
            raise ValueError(f"Energy recording population differs for {method}.")
    if len(reference_keys) != 128:
        raise ValueError(f"Expected 128 paired energy recordings, found {len(reference_keys)}.")
    return conditions


def plot_psd(
    frequency: np.ndarray,
    data: dict[str, dict[str, np.ndarray]],
    conditions: list[str],
    output: Path,
    minimum: float,
    maximum: float | None,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16.8, 8.2), sharex=True, layout="constrained")
    valid_frequency = (frequency >= minimum) & (frequency > 0)
    if maximum is not None:
        valid_frequency &= frequency <= maximum
    for axis, condition in zip(axes.flat, conditions, strict=True):
        for method in METHODS:
            q10, median, q90 = _quantiles(data[method][condition], axis=0)
            valid = valid_frequency & np.isfinite(median) & (median > 0)
            color = METHOD_COLORS[method]
            axis.loglog(
                frequency[valid],
                median[valid],
                color=color,
                linewidth=1.65,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                frequency[valid],
                np.maximum(q10[valid], np.finfo(float).tiny),
                np.maximum(q90[valid], np.finfo(float).tiny),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(f"{_power_value(condition):.2f} Vpp")
        axis.grid(True, which="both", linestyle="--", alpha=0.23)
    for axis in axes[-1]:
        axis.set_xlabel("Frequency [Hz]")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"PSD [$\mu$m$^2$/Hz]")
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=9)
    figure.suptitle("Center-point PSD: matched full field and POD reconstructions", fontsize=14)
    figure.text(
        0.5,
        -0.018,
        "line: median; shading: 10th--90th percentile across 16 repetitions; "
        "Hann Welch, nperseg=8192, 50% overlap",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _energy_array(
    source: dict[tuple[str, int], dict[str, float]], condition: str, field: str
) -> np.ndarray:
    return np.asarray([source[(condition, repetition)][field] for repetition in range(1, 17)])


def plot_energy(
    energy: dict[str, dict[tuple[str, int], dict[str, float]]],
    conditions: list[str],
    output: Path,
    dpi: int,
) -> None:
    x = np.asarray([_power_value(condition) for condition in conditions])
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.15), layout="constrained")
    for axis, field, title in zip(
        axes[:2],
        ("exact_J", "quadratic_J"),
        ("Exact geometric energy", "Quadratic energy"),
        strict=True,
    ):
        for method in METHODS:
            values = np.stack(
                [_energy_array(energy[method], condition, field) for condition in conditions]
            ) * 1e12
            q10, median, q90 = _quantiles(values, axis=1)
            color = METHOD_COLORS[method]
            axis.plot(
                x,
                median,
                color=color,
                marker=METHOD_MARKERS[method],
                linewidth=1.9,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(x, q10, q90, color=color, alpha=0.15, linewidth=0)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel("Capillary excess energy [pJ]")

    for method in ("rank200", "rank10"):
        ratios = []
        for condition in conditions:
            numerator = _energy_array(energy[method], condition, "exact_J")
            denominator = _energy_array(energy["full"], condition, "exact_J")
            ratios.append(100.0 * numerator / denominator)
        q10, median, q90 = _quantiles(np.stack(ratios), axis=1)
        color = METHOD_COLORS[method]
        axes[2].plot(
            x,
            median,
            color=color,
            marker=METHOD_MARKERS[method],
            linewidth=1.9,
            label=METHOD_LABELS[method],
        )
        axes[2].fill_between(x, q10, q90, color=color, alpha=0.15, linewidth=0)
    axes[2].axhline(100.0, color="0.3", linestyle="--", linewidth=1)
    axes[2].set_title("Paired exact-energy retention")
    axes[2].set_ylabel("POD / full field [%]")

    for axis in axes:
        axis.set_xlabel("Forcing amplitude [Vpp]")
        axis.set_xticks(x)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    figure.suptitle("Capillary energy: matched full field and POD reconstructions", fontsize=14)
    figure.text(
        0.5,
        -0.035,
        "line: median; shading: 10th--90th percentile across 16 matched repetitions. "
        "Full field: 2000 sampled frames; POD: full quadratic mean and 256-frame exact correction.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_statistics(
    path: Path,
    energy: dict[str, dict[tuple[str, int], dict[str, float]]],
    conditions: list[str],
) -> None:
    rows: list[dict[str, object]] = []
    for condition in conditions:
        full_exact = _energy_array(energy["full"], condition, "exact_J")
        for method in METHODS:
            exact = _energy_array(energy[method], condition, "exact_J")
            quadratic = _energy_array(energy[method], condition, "quadratic_J")
            eq = _quantiles(exact * 1e12)
            qq = _quantiles(quadratic * 1e12)
            ratios = _quantiles(100.0 * exact / full_exact)
            rows.append(
                {
                    "power": condition,
                    "forcing_vpp": _power_value(condition),
                    "method": method,
                    "repetitions": 16,
                    "exact_pJ_q10": eq[0],
                    "exact_pJ_median": eq[1],
                    "exact_pJ_q90": eq[2],
                    "quadratic_pJ_q10": qq[0],
                    "quadratic_pJ_median": qq[1],
                    "quadratic_pJ_q90": qq[2],
                    "paired_exact_retention_percent_q10": ratios[0],
                    "paired_exact_retention_percent_median": ratios[1],
                    "paired_exact_retention_percent_q90": ratios[2],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_psd_band_statistics(
    path: Path,
    frequency: np.ndarray,
    psd: dict[str, dict[str, np.ndarray]],
    conditions: list[str],
) -> None:
    bands = (
        ("50_to_2000_hz", 50.0, 2_000.0),
        ("2000_to_10000_hz", 2_000.0, 10_000.0),
        ("10000_to_nyquist_hz", 10_000.0, float(frequency[-1])),
        ("50_to_nyquist_hz", 50.0, float(frequency[-1])),
    )
    rows: list[dict[str, object]] = []
    for condition in conditions:
        for band, lower, upper in bands:
            selected = (frequency >= lower) & (frequency <= upper)
            full_power = np.trapezoid(psd["full"][condition][:, selected], frequency[selected], axis=1)
            for method in METHODS:
                power = np.trapezoid(psd[method][condition][:, selected], frequency[selected], axis=1)
                ratios = _quantiles(100.0 * power / full_power)
                rows.append(
                    {
                        "power": condition,
                        "forcing_vpp": _power_value(condition),
                        "band": band,
                        "minimum_hz": lower,
                        "maximum_hz": upper,
                        "method": method,
                        "paired_psd_power_retention_percent_q10": ratios[0],
                        "paired_psd_power_retention_percent_median": ratios[1],
                        "paired_psd_power_retention_percent_q90": ratios[2],
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output_dir.expanduser().resolve()
    outputs = [
        output_dir / "center_psd_full_vs_rank200_vs_rank10.png",
        output_dir / "capillary_energy_full_vs_rank200_vs_rank10.png",
        output_dir / "capillary_energy_comparison.csv",
        output_dir / "psd_band_retention.csv",
        output_dir / "comparison_summary.json",
    ]
    existing = next((path for path in outputs if path.exists()), None)
    if existing is not None and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing}; use --overwrite.")

    full_frequency, full_psd, full_energy, full_metadata = load_full_field(
        args.full_diagnostics.expanduser().resolve()
    )
    rank200_frequency, rank200_psd = load_rank_psd(
        args.rank200_psd.expanduser().resolve(), 200
    )
    rank10_frequency, rank10_psd = load_rank_psd(
        args.rank10_psd.expanduser().resolve(), 10
    )
    frequency = {
        "full": full_frequency,
        "rank200": rank200_frequency,
        "rank10": rank10_frequency,
    }
    psd = {"full": full_psd, "rank200": rank200_psd, "rank10": rank10_psd}
    energy = {
        "full": full_energy,
        "rank200": load_rank_energy(args.rank200_energy_dir.expanduser().resolve(), 200),
        "rank10": load_rank_energy(args.rank10_energy_dir.expanduser().resolve(), 10),
    }
    conditions = validate_populations(frequency, psd, energy)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_psd(
        full_frequency,
        psd,
        conditions,
        outputs[0],
        args.min_frequency_hz,
        args.max_frequency_hz,
        args.dpi,
    )
    plot_energy(energy, conditions, outputs[1], args.dpi)
    save_statistics(outputs[2], energy, conditions)
    save_psd_band_statistics(outputs[3], full_frequency, psd, conditions)
    summary = {
        "description": "Matched full-field, rank-200, and rank-10 comparison",
        "conditions": conditions,
        "repetitions_per_condition": 16,
        "aggregation": "median and 10th/90th percentiles across repetitions",
        "preprocessing": (
            "50 Hz full-field high-pass; temporal mean field removed; "
            "instantaneous spatial mean retained"
        ),
        "psd": {
            "observable": "center-point frequency PSD",
            "window": "hann",
            "nperseg": 8192,
            "overlap": 4096,
            "detrend": "constant",
            "scaling": "density",
        },
        "energy": {
            "gamma_N_per_m": 0.0728,
            "full_field_estimator": "2000 uniformly sampled frames with inclusion-probability weights",
            "pod_quadratic_estimator": "complete coefficient trajectory",
            "pod_exact_estimator": "full quadratic mean times paired 256-frame exact/quadratic ratio",
            "temporal_mean_restored": False,
        },
        "full_field_archive": full_metadata,
        "inputs": {
            "full_diagnostics": str(args.full_diagnostics.expanduser().resolve()),
            "rank200_psd": str(args.rank200_psd.expanduser().resolve()),
            "rank10_psd": str(args.rank10_psd.expanduser().resolve()),
            "rank200_energy_dir": str(args.rank200_energy_dir.expanduser().resolve()),
            "rank10_energy_dir": str(args.rank10_energy_dir.expanduser().resolve()),
        },
    }
    outputs[4].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for output in outputs:
        print(f"Saved {output}", flush=True)
    return outputs


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
