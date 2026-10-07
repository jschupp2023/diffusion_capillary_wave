"""Compare second-, fourth-, and sixth-order spatial differences in POD energy.

The current energy pipeline uses ``numpy.gradient(..., edge_order=2)``: a
three-point, second-order approximation in the interior and second-order
one-sided differences at the boundary.  This diagnostic holds the sampled
frames, POD reconstruction, nonlinear energy formula, and trapezoidal
quadrature fixed, changing only the spatial derivative stencil.

On a uniform grid, the alternatives use five-point fourth-order and seven-point
sixth-order centered interior stencils, with matching one-sided accuracy at the
boundaries. The implementation obtains the weights from the saved coordinates;
this accounts for their negligible floating-point spacing variation while
retaining the physically uniform grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.energy.pod_capillary_energy import quadrature_weights
from data_analysis.forcing_conditions import CA_AC_TIMES_1E3


DEFAULT_DATA_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
DEFAULT_ENERGY_CACHE = DEFAULT_DATA_ROOT / (
    "capillary_energy/"
    "surface_per_recording_ranks_10_100_200_1000_frames_2000/"
    "capillary_energy.h5"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / (
    "capillary_energy/gradient_order_sensitivity"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--energy-cache", type=Path, default=DEFAULT_ENERGY_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--powers", nargs="+", default=["0p08", "0p20", "0p35"])
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--ranks", nargs="+", type=int, default=[10, 100, 200, 1000])
    parser.add_argument(
        "--frames",
        type=int,
        default=256,
        help="Nested subset of the archived 2,000 sampled frames (default: 256).",
    )
    parser.add_argument("--subset-seed", type=int, default=20260929)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.ranks = sorted(set(args.ranks))
    if min(args.ranks) < 1 or min(args.frames, args.batch_size, args.repetition) < 1:
        parser.error("Ranks, frames, batch size, and repetition must be positive.")
    if args.subset_seed < 0:
        parser.error("--subset-seed must be nonnegative.")
    return args


def derivative_matrix(grid: np.ndarray, stencil_size: int) -> np.ndarray:
    """Return coordinate-aware first-derivative weights for an odd stencil."""
    grid = np.asarray(grid, dtype=np.float64)
    if stencil_size < 3 or stencil_size % 2 != 1:
        raise ValueError("The derivative stencil size must be odd and at least three.")
    if grid.ndim != 1 or len(grid) < stencil_size:
        raise ValueError(
            f"A one-dimensional grid with at least {stencil_size} points is required."
        )
    if not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("The derivative grid must be finite and strictly increasing.")

    matrix = np.zeros((len(grid), len(grid)), dtype=np.float64)
    half_width = stencil_size // 2
    for index in range(len(grid)):
        start = min(max(index - half_width, 0), len(grid) - stencil_size)
        stencil = np.arange(start, start + stencil_size)
        offsets = grid[stencil] - grid[index]
        scale = np.max(np.abs(offsets))
        normalized = offsets / scale
        vandermonde = np.vstack(
            [normalized**degree for degree in range(stencil_size)]
        )
        rhs = np.zeros(stencil_size)
        rhs[1] = 1.0 / scale
        matrix[index, stencil] = np.linalg.solve(vandermonde, rhs)
    return matrix


def matrix_gradient(
    field: np.ndarray,
    derivative_x: np.ndarray,
    derivative_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a ``(time, y, x)`` field using saved derivative matrices."""
    gx = np.einsum("tyx,kx->tyk", field, derivative_x, optimize=True)
    gy = np.einsum("ky,tyx->tkx", derivative_y, field, optimize=True)
    return gx, gy


def surface_energies(
    field_m: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    scheme: str,
    derivative_x: np.ndarray | None = None,
    derivative_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return quadratic and exact geometric excess areas for each frame."""
    if scheme == "second":
        gx = np.gradient(field_m, x_m, axis=-1, edge_order=2)
        gy = np.gradient(field_m, y_m, axis=-2, edge_order=2)
    elif scheme in {"fourth", "sixth"}:
        if derivative_x is None or derivative_y is None:
            raise ValueError(f"{scheme.title()}-order derivatives require matrices.")
        gx, gy = matrix_gradient(field_m, derivative_x, derivative_y)
    else:
        raise ValueError(f"Unknown finite-difference scheme: {scheme}")

    slopes_squared = gx * gx + gy * gy
    weights = np.outer(quadrature_weights(y_m), quadrature_weights(x_m))
    quadratic = 0.5 * np.einsum("tyx,yx->t", slopes_squared, weights)
    exact = np.einsum(
        "tyx,yx->t",
        slopes_squared / (np.sqrt(1.0 + slopes_squared) + 1.0),
        weights,
    )
    return quadratic, exact


def locate_pod(data_root: Path, power: str, repetition: int) -> Path:
    matches = sorted((data_root / power).glob(f"*_rep{repetition}/pod_2d_r1000.h5"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one rank-1000 POD for {power} repetition {repetition}; "
            f"found {len(matches)}."
        )
    return matches[0]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights) / np.sum(weights))


def percent_change(new: np.ndarray | float, old: np.ndarray | float):
    return 100.0 * (np.asarray(new) / np.asarray(old) - 1.0)


def analyze_case(
    pod_path: Path,
    cache: h5py.File,
    power: str,
    repetition: int,
    ranks: list[int],
    frames: int,
    subset_seed: int,
    batch_size: int,
    cached_ranks: list[int],
    gamma: float,
) -> list[dict[str, float | int | str]]:
    group_name = f"recordings/{power}_rep{repetition}/sampled_trajectories_n2000_seed12345"
    if group_name not in cache:
        raise KeyError(f"Missing archived trajectory group: {group_name}")
    archived = cache[group_name]
    archived_indices = np.asarray(archived["frame_indices"], dtype=np.int64)
    archived_weights = np.asarray(archived["mean_weights"], dtype=np.float64)
    if frames > len(archived_indices):
        raise ValueError(
            f"Requested {frames} frames but {group_name} contains {len(archived_indices)}."
        )
    rng = np.random.default_rng(subset_seed)
    subset_positions = np.sort(rng.choice(len(archived_indices), frames, replace=False))
    frame_indices = archived_indices[subset_positions]
    sample_weights = archived_weights[subset_positions]

    with h5py.File(pod_path) as pod:
        x_m = np.asarray(pod["grid/x"], dtype=np.float64) * 1e-6
        y_m = np.asarray(pod["grid/y"], dtype=np.float64) * 1e-6
        if pod["pod/modes"].shape[0] < max(ranks):
            raise ValueError(f"POD does not contain rank {max(ranks)}: {pod_path}")
        modes = np.asarray(pod["pod/modes"][: max(ranks)], dtype=np.float64)
        coefficients_m = (
            np.asarray(
                pod["reduced/coefficients"][frame_indices, : max(ranks)],
                dtype=np.float64,
            )
            * 1e-6
        )

    fourth_derivative_x = derivative_matrix(x_m, 5)
    fourth_derivative_y = derivative_matrix(y_m, 5)
    sixth_derivative_x = derivative_matrix(x_m, 7)
    sixth_derivative_y = derivative_matrix(y_m, 7)
    rows: list[dict[str, float | int | str]] = []
    for rank in ranks:
        second_quad = np.empty(frames)
        second_exact = np.empty(frames)
        fourth_quad = np.empty(frames)
        fourth_exact = np.empty(frames)
        sixth_quad = np.empty(frames)
        sixth_exact = np.empty(frames)
        flat_modes = modes[:rank].reshape(rank, -1)
        for start in range(0, frames, batch_size):
            stop = min(start + batch_size, frames)
            field = (coefficients_m[start:stop, :rank] @ flat_modes).reshape(
                stop - start, len(y_m), len(x_m)
            )
            second_quad[start:stop], second_exact[start:stop] = surface_energies(
                field, x_m, y_m, "second"
            )
            fourth_quad[start:stop], fourth_exact[start:stop] = surface_energies(
                field,
                x_m,
                y_m,
                "fourth",
                derivative_x=fourth_derivative_x,
                derivative_y=fourth_derivative_y,
            )
            sixth_quad[start:stop], sixth_exact[start:stop] = surface_energies(
                field,
                x_m,
                y_m,
                "sixth",
                derivative_x=sixth_derivative_x,
                derivative_y=sixth_derivative_y,
            )

        second_quad *= gamma
        second_exact *= gamma
        fourth_quad *= gamma
        fourth_exact *= gamma
        sixth_quad *= gamma
        sixth_exact *= gamma
        rank_position = cached_ranks.index(rank)
        cached_exact = np.asarray(
            archived["E_exact"][0, rank_position, subset_positions], dtype=np.float64
        )
        relative_validation = np.abs(second_exact - cached_exact) / np.maximum(
            np.abs(cached_exact), np.finfo(float).tiny
        )
        if not np.allclose(second_exact, cached_exact, rtol=2e-6, atol=1e-20):
            raise ValueError(
                f"Current-scheme recomputation did not match the archive for {power}, "
                f"rank {rank}; maximum relative error={relative_validation.max():.3g}."
            )

        second_exact_mean = weighted_mean(second_exact, sample_weights)
        fourth_exact_mean = weighted_mean(fourth_exact, sample_weights)
        sixth_exact_mean = weighted_mean(sixth_exact, sample_weights)
        second_quad_mean = weighted_mean(second_quad, sample_weights)
        fourth_quad_mean = weighted_mean(fourth_quad, sample_weights)
        sixth_quad_mean = weighted_mean(sixth_quad, sample_weights)
        paired_exact_change = percent_change(fourth_exact, second_exact)
        paired_sixth_exact_change = percent_change(sixth_exact, second_exact)
        paired_quad_change = percent_change(fourth_quad, second_quad)
        paired_sixth_quad_change = percent_change(sixth_quad, second_quad)
        rows.append(
            {
                "power": power,
                "ca_ac_times_1e3": CA_AC_TIMES_1E3[power],
                "repetition": repetition,
                "rank": rank,
                "sampled_frames": frames,
                "second_order_exact_nJ": second_exact_mean * 1e9,
                "fourth_order_exact_nJ": fourth_exact_mean * 1e9,
                "exact_change_percent": float(
                    percent_change(fourth_exact_mean, second_exact_mean)
                ),
                "sixth_order_exact_nJ": sixth_exact_mean * 1e9,
                "sixth_order_exact_change_percent": float(
                    percent_change(sixth_exact_mean, second_exact_mean)
                ),
                "exact_frame_change_median_percent": float(
                    np.median(paired_exact_change)
                ),
                "exact_frame_change_q05_percent": float(
                    np.quantile(paired_exact_change, 0.05)
                ),
                "exact_frame_change_q95_percent": float(
                    np.quantile(paired_exact_change, 0.95)
                ),
                "sixth_order_exact_frame_change_median_percent": float(
                    np.median(paired_sixth_exact_change)
                ),
                "sixth_order_exact_frame_change_q05_percent": float(
                    np.quantile(paired_sixth_exact_change, 0.05)
                ),
                "sixth_order_exact_frame_change_q95_percent": float(
                    np.quantile(paired_sixth_exact_change, 0.95)
                ),
                "second_order_quadratic_nJ": second_quad_mean * 1e9,
                "fourth_order_quadratic_nJ": fourth_quad_mean * 1e9,
                "quadratic_change_percent": float(
                    percent_change(fourth_quad_mean, second_quad_mean)
                ),
                "sixth_order_quadratic_nJ": sixth_quad_mean * 1e9,
                "sixth_order_quadratic_change_percent": float(
                    percent_change(sixth_quad_mean, second_quad_mean)
                ),
                "quadratic_frame_change_median_percent": float(
                    np.median(paired_quad_change)
                ),
                "sixth_order_quadratic_frame_change_median_percent": float(
                    np.median(paired_sixth_quad_change)
                ),
                "max_archive_validation_relative_error": float(
                    relative_validation.max()
                ),
                "pod_file": str(pod_path),
            }
        )
        print(
            f"{power} rep {repetition}, rank {rank}: exact "
            f"second/fourth/sixth = {second_exact_mean * 1e9:.6g}/"
            f"{fourth_exact_mean * 1e9:.6g}/{sixth_exact_mean * 1e9:.6g} nJ "
            f"(fourth {rows[-1]['exact_change_percent']:+.3f}%, "
            f"sixth {rows[-1]['sixth_order_exact_change_percent']:+.3f}%)",
            flush=True,
        )
    return rows


def save_outputs(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    settings: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "gradient_order_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "configuration.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )

    powers = list(dict.fromkeys(str(row["power"]) for row in rows))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.2), sharex=True)
    panels = (
        (axes[0, 0], "exact_change_percent", "Fourth vs second: exact"),
        (
            axes[0, 1],
            "sixth_order_exact_change_percent",
            "Sixth vs second: exact",
        ),
        (axes[1, 0], "quadratic_change_percent", "Fourth vs second: quadratic"),
        (
            axes[1, 1],
            "sixth_order_quadratic_change_percent",
            "Sixth vs second: quadratic",
        ),
    )
    for axis, field, title in panels:
        for power in powers:
            selected = [row for row in rows if row["power"] == power]
            ranks = [int(row["rank"]) for row in selected]
            label = rf"$Ca_{{ac}}={CA_AC_TIMES_1E3[power]:.2f}\times10^{{-3}}$"
            axis.plot(
                ranks,
                [float(row[field]) for row in selected],
                "o-",
                linewidth=2.2,
                markersize=7,
                label=label,
            )
        axis.axhline(0.0, color="0.2", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_xticks(settings["ranks"], [str(rank) for rank in settings["ranks"]])
        axis.set_title(title, fontsize=15)
        axis.tick_params(labelsize=12)
        axis.grid(True, alpha=0.25)
    for axis in axes[1]:
        axis.set_xlabel("POD reconstruction rank", fontsize=14)
    axes[0, 0].set_ylabel("Exact-energy change [%]", fontsize=14)
    axes[1, 0].set_ylabel("Quadratic-energy change [%]", fontsize=14)
    axes[0, 1].legend(fontsize=10.5, frameon=True)
    figure.suptitle(
        "Finite-difference sensitivity of capillary energy",
        fontsize=17,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "gradient_order_sensitivity.png", dpi=220)
    plt.close(figure)

    lines = [
        "# Finite-difference-order sensitivity",
        "",
        "The existing three-point second-order spatial derivative was compared "
        "with five-point fourth-order and seven-point sixth-order derivatives. "
        "The physical grid is uniform; weights were calculated from the saved "
        "coordinates only to accommodate their negligible floating-point spacing "
        "variation. The interior stencils are centered and the boundary stencils "
        "use matching one-sided accuracy.",
        "",
        f"This focused test used repetition {settings['repetition']} at powers "
        f"{', '.join(settings['powers'])}, ranks {settings['ranks']}, and "
        f"{settings['frames']} matched frames per case. The POD quantity is the "
        "same retained fluctuating field used by the existing comparison: no "
        "temporal mean or instantaneous spatial mean was restored.",
        "",
        "Only the spatial derivative approximation changed. The POD coefficients, "
        "modes, frames, surface tension, exact nonlinear integrand, and trapezoidal "
        "spatial quadrature were identical. Recomputed second-order per-frame "
        "energies were checked against the existing archive.",
        "",
        "| Power | Rank | Second order exact [nJ] | Fourth order exact [nJ] | Fourth change | Sixth order exact [nJ] | Sixth change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['power']} | {row['rank']} | "
            f"{float(row['second_order_exact_nJ']):.6g} | "
            f"{float(row['fourth_order_exact_nJ']):.6g} | "
            f"{float(row['exact_change_percent']):+.3f}% | "
            f"{float(row['sixth_order_exact_nJ']):.6g} | "
            f"{float(row['sixth_order_exact_change_percent']):+.3f}% |"
        )
    lines.extend(
        [
            "",
            "This is a derivative-order sensitivity check on selected POD "
            "reconstructions, not a full repetition-ensemble recomputation and not "
            "a raw-height-map test.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | int | str]] = []
    with h5py.File(args.energy_cache.expanduser().resolve()) as cache:
        configuration = json.loads(cache.attrs["projection_configuration"])
        cached_ranks = [int(rank) for rank in configuration["ranks"]]
        missing_ranks = [rank for rank in args.ranks if rank not in cached_ranks]
        if missing_ranks:
            raise ValueError(f"Ranks absent from energy archive: {missing_ranks}")
        gamma = float(cache.attrs["gamma"])
        for power in args.powers:
            if power not in CA_AC_TIMES_1E3:
                raise ValueError(f"Missing Ca_ac conversion for {power}")
            pod_path = locate_pod(args.data_root.expanduser().resolve(), power, args.repetition)
            rows.extend(
                analyze_case(
                    pod_path,
                    cache,
                    power,
                    args.repetition,
                    args.ranks,
                    args.frames,
                    args.subset_seed,
                    args.batch_size,
                    cached_ranks,
                    gamma,
                )
            )

    settings = {
        "powers": args.powers,
        "repetition": args.repetition,
        "ranks": args.ranks,
        "frames": args.frames,
        "subset_seed": args.subset_seed,
        "batch_size": args.batch_size,
        "gamma_N_per_m": gamma,
        "energy_cache": str(args.energy_cache.expanduser().resolve()),
        "current_scheme": "numpy.gradient edge_order=2",
        "fourth_order_scheme": (
            "five-point coordinate-aware first derivative; fourth-order on a "
            "uniform grid, including one-sided boundary stencils"
        ),
        "sixth_order_scheme": (
            "seven-point coordinate-aware first derivative; sixth-order on a "
            "uniform grid, including one-sided boundary stencils"
        ),
        "unchanged_spatial_quadrature": "trapezoidal",
        "unchanged_energy_integrand": "s/(sqrt(1+s)+1)",
    }
    save_outputs(rows, args.output_dir.expanduser().resolve(), settings)
    print(f"Saved {args.output_dir.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
