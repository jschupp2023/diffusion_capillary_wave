"""Summarize rank-limited capillary excess energy across forcing conditions.

This consumes the per-recording output of ``pod_capillary_energy.py``.  The
quadratic energy is the exact full-trajectory POD result.  The exact geometric
energy is estimated by multiplying that result by the paired exact/quadratic
ratio from the sampled reconstructed fields.  This ratio correction removes
the leading random-frame sampling error without assuming the small-slope
formula is exact.

A derivative-order diagnostic reconstructs a nested subset of the archived
frames and compares second-, fourth-, and sixth-order centered differences on
the same interior pixels.  The comparison therefore changes only the spatial
derivative stencil; it does not mix in boundary-stencil or domain-area effects.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POWER_PATTERN = re.compile(r"0p\d+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.0728)
    parser.add_argument(
        "--stability-frames",
        type=int,
        default=64,
        help="Nested archived frames per recording for derivative-order checks.",
    )
    parser.add_argument("--stability-seed", type=int, default=20261108)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if min(args.rank, args.stability_frames, args.batch_size, args.dpi) < 1:
        parser.error("Rank, frame count, batch size, and DPI must be positive.")
    if args.gamma <= 0 or not np.isfinite(args.gamma) or args.stability_seed < 0:
        parser.error("Gamma must be positive and the seed nonnegative.")
    return args


def _power_value(power: str) -> float:
    if POWER_PATTERN.fullmatch(power) is None:
        raise ValueError(f"Invalid power label: {power}")
    return float(power.replace("p", "."))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights) / np.sum(weights))


def read_energy_rows(energy_dir: Path, rank: int) -> list[dict[str, object]]:
    csv_path = energy_dir / "repetition_energy.csv"
    rows: list[dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            if int(raw["rank"]) != rank:
                continue
            full_quad = float(raw["time_averaged_energy"])
            sampled_quad = float(raw["Ebar_quad"])
            sampled_exact = float(raw["Ebar_exact"])
            if min(full_quad, sampled_quad, sampled_exact) <= 0:
                raise ValueError(f"Nonpositive capillary energy in {csv_path}.")
            exact_ratio = sampled_exact / sampled_quad
            if not 0 < exact_ratio <= 1 + 1e-12:
                raise ValueError("Expected paired exact energy not to exceed quadratic energy.")
            rows.append(
                {
                    "power": raw["power"],
                    "repetition": int(raw["repetition"]),
                    "rank": rank,
                    "quadratic_J": full_quad,
                    "exact_ratio": exact_ratio,
                    "exact_J": full_quad * exact_ratio,
                    "quadratic_overestimate_percent": 100.0
                    * (1.0 / exact_ratio - 1.0),
                    "sampled_quadratic_J": sampled_quad,
                    "sampled_exact_J": sampled_exact,
                    "quadratic_sampling_error_percent": 100.0
                    * (sampled_quad / full_quad - 1.0),
                    "rms_slope": float(raw["rms_spatial_slope"]),
                    "source_file": raw["source_file"],
                }
            )
    if not rows:
        raise ValueError(f"No rank-{rank} rows found in {csv_path}.")
    return rows


def centered_gradients(
    field: np.ndarray, dx: float, dy: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centered derivatives on the common three-pixel interior."""
    if order == 2:
        gx = (field[:, 3:-3, 4:-2] - field[:, 3:-3, 2:-4]) / (2.0 * dx)
        gy = (field[:, 4:-2, 3:-3] - field[:, 2:-4, 3:-3]) / (2.0 * dy)
    elif order == 4:
        gx = (
            field[:, 3:-3, 1:-5]
            - 8.0 * field[:, 3:-3, 2:-4]
            + 8.0 * field[:, 3:-3, 4:-2]
            - field[:, 3:-3, 5:-1]
        ) / (12.0 * dx)
        gy = (
            field[:, 1:-5, 3:-3]
            - 8.0 * field[:, 2:-4, 3:-3]
            + 8.0 * field[:, 4:-2, 3:-3]
            - field[:, 5:-1, 3:-3]
        ) / (12.0 * dy)
    elif order == 6:
        gx = (
            -field[:, 3:-3, :-6]
            + 9.0 * field[:, 3:-3, 1:-5]
            - 45.0 * field[:, 3:-3, 2:-4]
            + 45.0 * field[:, 3:-3, 4:-2]
            - 9.0 * field[:, 3:-3, 5:-1]
            + field[:, 3:-3, 6:]
        ) / (60.0 * dx)
        gy = (
            -field[:, :-6, 3:-3]
            + 9.0 * field[:, 1:-5, 3:-3]
            - 45.0 * field[:, 2:-4, 3:-3]
            + 45.0 * field[:, 4:-2, 3:-3]
            - 9.0 * field[:, 5:-1, 3:-3]
            + field[:, 6:, 3:-3]
        ) / (60.0 * dy)
    else:
        raise ValueError(f"Unsupported derivative order: {order}")
    return gx, gy


def interior_energies(
    field_m: np.ndarray, dx_m: float, dy_m: float, gamma: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = centered_gradients(field_m, dx_m, dy_m, order)
    slope_squared = gx * gx + gy * gy
    pixel_area = dx_m * dy_m
    quadratic = 0.5 * gamma * pixel_area * np.sum(slope_squared, axis=(1, 2))
    exact = gamma * pixel_area * np.sum(
        slope_squared / (np.sqrt(1.0 + slope_squared) + 1.0), axis=(1, 2)
    )
    return quadratic, exact


def gradient_stability(
    rows: list[dict[str, object]],
    cache_path: Path,
    rank: int,
    frame_count: int,
    seed: int,
    batch_size: int,
    gamma: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    with h5py.File(cache_path, "r") as cache:
        first = rows[0]
        first_group = cache[
            f"recordings/{first['power']}_rep{first['repetition']}"
        ]
        candidates = [
            name for name in first_group
            if name.startswith("sampled_trajectories_n")
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one sampled trajectory archive, found {candidates}."
            )
        series_name = candidates[0]
        match = re.fullmatch(r"sampled_trajectories_n(\d+)_seed(\d+)", series_name)
        if match is None:
            raise ValueError(f"Could not parse sampled archive name: {series_name}")
        archived_count, archived_seed = map(int, match.groups())
        if frame_count > archived_count:
            raise ValueError(
                f"Requested {frame_count} stability frames, but archive has {archived_count}."
            )
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(archived_count, frame_count, replace=False))

        for index, row in enumerate(rows, start=1):
            power = str(row["power"])
            repetition = int(row["repetition"])
            archived = cache[
                f"recordings/{power}_rep{repetition}/{series_name}"
            ]
            frame_indices = np.asarray(archived["frame_indices"][positions], dtype=int)
            weights = np.asarray(archived["mean_weights"][positions], dtype=float)
            pod_path = Path(str(row["source_file"]))
            with h5py.File(pod_path, "r") as pod:
                x = np.asarray(pod["grid/x"], dtype=float) * 1e-6
                y = np.asarray(pod["grid/y"], dtype=float) * 1e-6
                dx = float(np.mean(np.diff(x)))
                dy = float(np.mean(np.diff(y)))
                if (
                    np.max(np.abs(np.diff(x) / dx - 1.0)) > 1e-4
                    or np.max(np.abs(np.diff(y) / dy - 1.0)) > 1e-4
                ):
                    raise ValueError(f"Derivative-order check requires a uniform grid: {pod_path}")
                modes = np.asarray(pod["pod/modes"][:rank], dtype=np.float64).reshape(
                    rank, -1
                )
                coefficient_dataset = pod["reduced/coefficients"]
                accumulated = {
                    order: {"quadratic": [], "exact": []} for order in (2, 4, 6)
                }
                for start in range(0, frame_count, batch_size):
                    stop = min(start + batch_size, frame_count)
                    coefficients = np.asarray(
                        coefficient_dataset[frame_indices[start:stop], :rank],
                        dtype=np.float64,
                    )
                    field = (coefficients @ modes).reshape(
                        stop - start, len(y), len(x)
                    ) * 1e-6
                    for order in (2, 4, 6):
                        quadratic, exact = interior_energies(field, dx, dy, gamma, order)
                        accumulated[order]["quadratic"].append(quadratic)
                        accumulated[order]["exact"].append(exact)
            means: dict[tuple[int, str], float] = {}
            for order in (2, 4, 6):
                for observable in ("quadratic", "exact"):
                    values = np.concatenate(accumulated[order][observable])
                    means[(order, observable)] = _weighted_mean(values, weights)
            results.append(
                {
                    "power": power,
                    "repetition": repetition,
                    "frames": frame_count,
                    "second_exact_J": means[(2, "exact")],
                    "fourth_exact_J": means[(4, "exact")],
                    "sixth_exact_J": means[(6, "exact")],
                    "fourth_vs_second_exact_percent": 100.0
                    * (means[(4, "exact")] / means[(2, "exact")] - 1.0),
                    "sixth_vs_second_exact_percent": 100.0
                    * (means[(6, "exact")] / means[(2, "exact")] - 1.0),
                    "fourth_vs_second_quadratic_percent": 100.0
                    * (means[(4, "quadratic")] / means[(2, "quadratic")] - 1.0),
                    "sixth_vs_second_quadratic_percent": 100.0
                    * (means[(6, "quadratic")] / means[(2, "quadratic")] - 1.0),
                    "source_file": str(pod_path),
                }
            )
            print(
                f"Gradient stability {index}/{len(rows)}: {power}/rep{repetition}",
                flush=True,
            )
    return results


def _group(rows: list[dict[str, object]], field: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["power"])].append(float(row[field]))
    return {key: np.asarray(value) for key, value in grouped.items()}


def _band(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.5, 0.1, 0.9]))


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_energy(rows: list[dict[str, object]], path: Path, rank: int, dpi: int) -> None:
    powers = sorted({str(row["power"]) for row in rows}, key=_power_value)
    x = np.asarray([_power_value(power) for power in powers])
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), layout="constrained")
    for field, label, color, marker in (
        ("quadratic_J", "Quadratic", "#2864b7", "o"),
        ("exact_J", "Exact geometric", "#d1495b", "s"),
    ):
        grouped = _group(rows, field)
        bands = np.asarray([_band(grouped[power] * 1e12) for power in powers])
        axes[0].plot(x, bands[:, 0], marker=marker, color=color, linewidth=2, label=label)
        axes[0].fill_between(x, bands[:, 1], bands[:, 2], color=color, alpha=0.18)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Forcing amplitude [Vpp]", ylabel="Capillary excess energy [pJ]")
    axes[0].legend(frameon=False)

    grouped = _group(rows, "quadratic_overestimate_percent")
    bands = np.asarray([_band(grouped[power]) for power in powers])
    axes[1].plot(x, bands[:, 0], "o-", color="#6a3d9a", linewidth=2)
    axes[1].fill_between(x, bands[:, 1], bands[:, 2], color="#6a3d9a", alpha=0.2)
    axes[1].set(
        xlabel="Forcing amplitude [Vpp]",
        ylabel="Quadratic excess over exact [%]",
    )
    for axis in axes:
        axis.set_xticks(x)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle(f"Rank-{rank} POD capillary surface-energy estimate")
    figure.text(
        0.5,
        -0.015,
        "line: median across 16 repetitions; shading: 10th--90th percentile",
        ha="center",
        fontsize=9,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_stability(
    rows: list[dict[str, object]], path: Path, rank: int, frames: int, dpi: int
) -> None:
    powers = sorted({str(row["power"]) for row in rows}, key=_power_value)
    x = np.asarray([_power_value(power) for power in powers])
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), layout="constrained")
    for axis, observable in zip(axes, ("exact", "quadratic"), strict=True):
        for order, label, color in (
            ("fourth", "Fourth vs second order", "#1b9e77"),
            ("sixth", "Sixth vs second order", "#d95f02"),
        ):
            field = f"{order}_vs_second_{observable}_percent"
            grouped = _group(rows, field)
            bands = np.asarray([_band(grouped[power]) for power in powers])
            axis.plot(x, bands[:, 0], "o-", color=color, linewidth=2, label=label)
            axis.fill_between(x, bands[:, 1], bands[:, 2], color=color, alpha=0.18)
        axis.axhline(0, color="0.25", linewidth=1)
        axis.set(
            xlabel="Forcing amplitude [Vpp]",
            ylabel=f"{observable.title()}-energy change [%]",
        )
        axis.legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"Rank-{rank} spatial-gradient stability ({frames} matched frames per repetition)"
    )
    figure.text(
        0.5,
        -0.015,
        "common interior pixels only; line: median across 16 repetitions; shading: 10th--90th percentile",
        ha="center",
        fontsize=9,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def summarize(
    energy_rows: list[dict[str, object]], stability_rows: list[dict[str, object]]
) -> dict[str, object]:
    powers = sorted({str(row["power"]) for row in energy_rows}, key=_power_value)
    summary: dict[str, object] = {}
    for power in powers:
        selected = [row for row in energy_rows if row["power"] == power]
        stable = [row for row in stability_rows if row["power"] == power]
        summary[power] = {
            "repetitions": len(selected),
            "quadratic_pJ_median_q10_q90": _band(
                np.asarray([float(row["quadratic_J"]) for row in selected]) * 1e12
            ),
            "exact_pJ_median_q10_q90": _band(
                np.asarray([float(row["exact_J"]) for row in selected]) * 1e12
            ),
            "quadratic_overestimate_percent_median_q10_q90": _band(
                np.asarray(
                    [float(row["quadratic_overestimate_percent"]) for row in selected]
                )
            ),
            "rms_slope_median_q10_q90": _band(
                np.asarray([float(row["rms_slope"]) for row in selected])
            ),
            "fourth_vs_second_exact_percent_median_q10_q90": _band(
                np.asarray(
                    [float(row["fourth_vs_second_exact_percent"]) for row in stable]
                )
            ),
            "sixth_vs_second_exact_percent_median_q10_q90": _band(
                np.asarray(
                    [float(row["sixth_vs_second_exact_percent"]) for row in stable]
                )
            ),
        }
    return summary


def run(args: argparse.Namespace) -> list[Path]:
    energy_dir = args.energy_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_path = energy_dir / "capillary_energy.h5"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"capillary_energy_rank{args.rank}_all_conditions.png",
        output_dir / f"capillary_gradient_stability_rank{args.rank}.png",
        output_dir / "capillary_energy_repetitions.csv",
        output_dir / "capillary_gradient_stability.csv",
        output_dir / "summary.json",
    ]
    if not args.overwrite:
        existing = next((path for path in outputs if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Output exists: {existing}; use --overwrite.")

    energy_rows = read_energy_rows(energy_dir, args.rank)
    stability_rows = gradient_stability(
        energy_rows,
        cache_path,
        args.rank,
        args.stability_frames,
        args.stability_seed,
        args.batch_size,
        args.gamma,
    )
    save_csv(outputs[2], energy_rows)
    save_csv(outputs[3], stability_rows)
    plot_energy(energy_rows, outputs[0], args.rank, args.dpi)
    plot_stability(stability_rows, outputs[1], args.rank, args.stability_frames, args.dpi)
    report = {
        "rank": args.rank,
        "gamma_N_per_m": args.gamma,
        "energy_field": "temporally centered rank-200 POD reconstruction; temporal mean not restored",
        "quadratic_time_average": "full 100001-frame coefficient second moment",
        "exact_estimator": "full quadratic mean times paired sampled exact/quadratic ratio",
        "exact_sample_frames_per_recording": 256,
        "gradient_stability_frames_per_recording": args.stability_frames,
        "gradient_stability_seed": args.stability_seed,
        "gradient_stability_domain": "common three-pixel interior; identical pixels for orders 2, 4, 6",
        "aggregation": "per-recording estimates; condition median and 10th/90th percentiles across repetitions",
        "conditions": summarize(energy_rows, stability_rows),
    }
    outputs[4].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_dir}", flush=True)
    return outputs


if __name__ == "__main__":
    run(parse_args())
