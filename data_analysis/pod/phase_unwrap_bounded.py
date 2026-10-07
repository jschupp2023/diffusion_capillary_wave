"""Experimental bounded, minimum-change integer-cycle correction.

The corrected field is restricted to ``height + q * k`` with integer ``k``.
Already plausible spatial and detrended-temporal edges preferentially request
zero relative correction.  Edges exceeding a user-supplied hard bound request
the nearest integer-cycle adjustment.  A joint weighted graph solve reconciles
those requests between two fixed clean endpoint frames.

This is an explicit physical-prior experiment, not recovery of uniquely known
phase branches.  The source HDF5 is opened read-only and all candidates are
written separately.

Example
-------
python data_analysis/pod/phase_unwrap_bounded.py \
    --raw-input /home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5 \
    --start-anchor 1548 --stop-anchor 1578 --target 1553 --q 2.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

if __package__:
    from .pod_2d import InputInfo, _frame_key, inspect_input
else:
    from pod_2d import InputInfo, _frame_key, inspect_input


OUTPUT_FILENAMES = (
    "target_comparison.png",
    "window_diagnostics.png",
    "bounded_candidate.npz",
    "summary.json",
)


@dataclass(frozen=True)
class BoundedResult:
    corrected: np.ndarray
    correction_cycles: np.ndarray
    temporal_carrier: np.ndarray
    solver_info: int | None
    gate_preserved_original: bool


@dataclass(frozen=True)
class BoundedWindow:
    info: InputInfo
    fields: np.ndarray
    invalid: np.ndarray
    target_local_index: int


def load_raw_interval(
    raw_input: Path, start_anchor: int, stop_anchor: int, target: int
) -> BoundedWindow:
    """Load an inclusive anchored interval through the repository input schema."""
    if start_anchor < 0 or stop_anchor <= start_anchor:
        raise ValueError("Require 0 <= start_anchor < stop_anchor.")
    if not start_anchor <= target <= stop_anchor:
        raise ValueError("target must lie inside the inclusive anchor interval.")
    source = Path(raw_input).expanduser().resolve()
    info = inspect_input(source, start_anchor, stop_anchor + 1)
    fields = np.empty((info.n_frames, *info.frame_shape), dtype=np.float64)
    with h5py.File(source, "r") as handle:
        main = handle["main"]
        for index, frame_number in enumerate(info.frame_numbers):
            fields[index] = np.asarray(main[_frame_key(int(frame_number))], dtype=float)
    invalid = ~np.isfinite(fields)
    if np.any(np.all(invalid, axis=(1, 2))):
        raise ValueError("The anchored interval contains an entirely invalid frame.")
    return BoundedWindow(info, fields, invalid, target - start_anchor)


def _polynomial_design(shape: tuple[int, int], degree: int) -> np.ndarray:
    if degree < 0:
        raise ValueError("carrier_degree must be nonnegative.")
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0]),
        np.linspace(-1.0, 1.0, shape[1]),
        indexing="ij",
    )
    powers = [(i, order - i) for order in range(degree + 1) for i in range(order + 1)]
    return np.column_stack([(x**i * y**j).ravel() for i, j in powers])


def robust_temporal_carrier(
    fields: np.ndarray,
    invalid: np.ndarray,
    *,
    degree: int = 5,
    iterations: int = 4,
) -> np.ndarray:
    """Fit robust low-order motion to each raw adjacent-frame difference."""
    if iterations < 1:
        raise ValueError("carrier_iterations must be positive.")
    design = _polynomial_design(fields.shape[1:], degree)
    carriers = np.empty((len(fields) - 1, *fields.shape[1:]), dtype=np.float64)
    for index in range(len(fields) - 1):
        difference = fields[index + 1] - fields[index]
        valid = ~(invalid[index + 1] | invalid[index])
        matrix = design[valid.ravel()]
        values = difference[valid]
        if len(values) < matrix.shape[1]:
            raise ValueError("Too few valid temporal samples for the carrier fit.")
        weights = np.ones(len(values), dtype=float)
        coefficients = np.zeros(matrix.shape[1], dtype=float)
        for _ in range(iterations):
            square_root = np.sqrt(weights)
            coefficients = np.linalg.lstsq(
                matrix * square_root[:, None], values * square_root, rcond=None
            )[0]
            residual = values - matrix @ coefficients
            median = np.median(residual)
            scale = 1.4826 * np.median(np.abs(residual - median)) + 1e-12
            standardized = np.abs(residual - median) / (1.5 * scale)
            weights = np.ones_like(standardized)
            outside = standardized > 1.0
            weights[outside] = 1.0 / standardized[outside]
        carriers[index] = (design @ coefficients).reshape(fields.shape[1:])
    return carriers


def _valid_edges(invalid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        ~(invalid[:, :, 1:] | invalid[:, :, :-1]),
        ~(invalid[:, 1:, :] | invalid[:, :-1, :]),
        ~(invalid[1:] | invalid[:-1]),
    )


def _edge_requests(
    difference: np.ndarray,
    valid: np.ndarray,
    q: float,
    soft_bound: float,
    hard_bound: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(difference)
    request = np.zeros(difference.shape, dtype=np.float64)
    violates = valid & (magnitude >= hard_bound)
    request[violates] = -np.rint(difference[violates] / q)

    nearest_residual = difference + q * request
    weight = np.where(
        magnitude < soft_bound,
        2.0,
        np.where(
            magnitude < hard_bound,
            0.75,
            2.0 + np.minimum(magnitude / q - 1.0, 8.0),
        ),
    )
    # Edges whose nearest-cycle residual lies at the q/2 decision boundary
    # carry little branch information.  Keep a floor so they are still connected.
    reliability = np.maximum(
        0.1,
        (1.0 - 2.0 * np.minimum(np.abs(nearest_residual), q / 2) / q) ** 2,
    )
    weight = np.where(valid, scale * weight * reliability, 0.0)
    return request, weight


def _spatial_values(field: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    valid_x = ~(invalid[:, 1:] | invalid[:, :-1])
    valid_y = ~(invalid[1:, :] | invalid[:-1, :])
    return np.concatenate(
        (
            np.abs(np.diff(field, axis=1))[valid_x],
            np.abs(np.diff(field, axis=0))[valid_y],
        )
    )


def spatial_metrics(
    fields: np.ndarray, invalid: np.ndarray, soft_bound: float, hard_bound: float
) -> dict[str, np.ndarray]:
    names = ("p50", "p95", "p99", "p999", "fraction_above_soft", "fraction_above_hard")
    output = {name: np.empty(len(fields), dtype=float) for name in names}
    for index, field in enumerate(fields):
        values = _spatial_values(field, invalid[index])
        percentiles = np.percentile(values, (50, 95, 99, 99.9))
        for name, value in zip(names[:4], percentiles):
            output[name][index] = value
        output["fraction_above_soft"][index] = np.mean(values > soft_bound)
        output["fraction_above_hard"][index] = np.mean(values > hard_bound)
    return output


def temporal_residual_metrics(
    fields: np.ndarray,
    invalid: np.ndarray,
    carrier: np.ndarray,
    soft_bound: float,
    hard_bound: float,
) -> dict[str, np.ndarray]:
    names = ("p50", "p95", "p99", "fraction_above_soft", "fraction_above_hard")
    output = {name: np.empty(len(fields) - 1, dtype=float) for name in names}
    for index in range(len(fields) - 1):
        valid = ~(invalid[index + 1] | invalid[index])
        values = np.abs(fields[index + 1] - fields[index] - carrier[index])[valid]
        percentiles = np.percentile(values, (50, 95, 99))
        for name, value in zip(names[:3], percentiles):
            output[name][index] = value
        output["fraction_above_soft"][index] = np.mean(values > soft_bound)
        output["fraction_above_hard"][index] = np.mean(values > hard_bound)
    return output


def target_is_already_smooth(
    fields: np.ndarray,
    invalid: np.ndarray,
    carrier: np.ndarray,
    target_index: int,
    soft_bound: float,
    hard_bound: float,
) -> bool:
    values = _spatial_values(fields[target_index], invalid[target_index])
    spatially_smooth = bool(
        np.percentile(values, 95) <= soft_bound
        and np.percentile(values, 99) <= hard_bound
    )
    temporal_checks: list[bool] = []
    for earlier in (target_index - 1, target_index):
        if 0 <= earlier < len(fields) - 1:
            valid = ~(invalid[earlier + 1] | invalid[earlier])
            residual = np.abs(
                fields[earlier + 1] - fields[earlier] - carrier[earlier]
            )[valid]
            temporal_checks.append(
                bool(
                    np.percentile(residual, 95) <= hard_bound
                    and np.mean(residual > hard_bound) <= 0.05
                )
            )
    return spatially_smooth and all(temporal_checks)


def solve_bounded_correction(
    fields: np.ndarray,
    invalid: np.ndarray,
    carrier: np.ndarray,
    q: float,
    *,
    soft_bound: float,
    hard_bound: float,
    preservation_weight: float = 0.05,
    spatial_weight: float = 4.0,
    temporal_weight: float = 1.0,
    anchor_weight: float = 1e4,
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int = 300,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve a weighted graph relaxation and round to integer cycle labels."""
    if fields.shape != invalid.shape:
        raise ValueError("fields and invalid must have identical shapes.")
    if carrier.shape != (len(fields) - 1, *fields.shape[1:]):
        raise ValueError("carrier has the wrong shape.")
    if not 0 < soft_bound < hard_bound:
        raise ValueError("Require 0 < soft_bound < hard_bound.")
    if min(preservation_weight, spatial_weight, temporal_weight, anchor_weight) <= 0:
        raise ValueError("All solver weights must be positive.")

    safe = np.where(invalid, 0.0, fields)
    valid_x, valid_y, valid_t = _valid_edges(invalid)
    differences = (
        safe[:, :, 1:] - safe[:, :, :-1],
        safe[:, 1:, :] - safe[:, :-1, :],
        safe[1:] - safe[:-1] - carrier,
    )
    requests_and_weights = (
        _edge_requests(differences[0], valid_x, q, soft_bound, hard_bound, spatial_weight),
        _edge_requests(differences[1], valid_y, q, soft_bound, hard_bound, spatial_weight),
        _edge_requests(differences[2], valid_t, q, soft_bound, hard_bound, temporal_weight),
    )
    requests = tuple(item[0] for item in requests_and_weights)
    weights = tuple(item[1] for item in requests_and_weights)

    unary = np.full(fields.shape, preservation_weight, dtype=np.float64)
    unary[invalid] = anchor_weight
    unary[0] = anchor_weight
    unary[-1] = anchor_weight
    right_hand_side = np.zeros(fields.shape, dtype=np.float64)
    diagonal = unary.copy()

    for request, weight, axis in zip(requests, weights, (2, 1, 0)):
        contribution = weight * request
        if axis == 2:
            right_hand_side[:, :, :-1] -= contribution
            right_hand_side[:, :, 1:] += contribution
            diagonal[:, :, :-1] += weight
            diagonal[:, :, 1:] += weight
        elif axis == 1:
            right_hand_side[:, :-1, :] -= contribution
            right_hand_side[:, 1:, :] += contribution
            diagonal[:, :-1, :] += weight
            diagonal[:, 1:, :] += weight
        else:
            right_hand_side[:-1] -= contribution
            right_hand_side[1:] += contribution
            diagonal[:-1] += weight
            diagonal[1:] += weight

    def matrix_vector(values: np.ndarray) -> np.ndarray:
        labels = values.reshape(fields.shape)
        output = unary * labels
        for weight, axis in zip(weights, (2, 1, 0)):
            if axis == 2:
                delta = labels[:, :, 1:] - labels[:, :, :-1]
                contribution = weight * delta
                output[:, :, :-1] -= contribution
                output[:, :, 1:] += contribution
            elif axis == 1:
                delta = labels[:, 1:, :] - labels[:, :-1, :]
                contribution = weight * delta
                output[:, :-1, :] -= contribution
                output[:, 1:, :] += contribution
            else:
                delta = labels[1:] - labels[:-1]
                contribution = weight * delta
                output[:-1] -= contribution
                output[1:] += contribution
        return output.ravel()

    size = fields.size
    operator = LinearOperator((size, size), matvec=matrix_vector, dtype=np.float64)
    preconditioner = LinearOperator(
        (size, size), matvec=lambda values: values / diagonal.ravel(), dtype=np.float64
    )
    relaxed, solver_info = cg(
        operator,
        right_hand_side.ravel(),
        M=preconditioner,
        rtol=cg_tolerance,
        atol=0.0,
        maxiter=cg_max_iterations,
    )
    if solver_info < 0:
        raise RuntimeError(f"Conjugate-gradient solver failed with info={solver_info}.")
    cycles = np.rint(relaxed.reshape(fields.shape)).astype(np.int16)
    cycles[invalid] = 0
    cycles[0] = 0
    cycles[-1] = 0
    corrected = safe + q * cycles
    corrected[invalid] = np.nan
    return corrected, cycles, int(solver_info)


def bounded_correction(
    window: BoundedWindow,
    q: float,
    *,
    soft_bound: float | None = None,
    hard_bound: float | None = None,
    preservation_weight: float = 0.05,
    spatial_weight: float = 4.0,
    temporal_weight: float = 1.0,
    carrier_degree: int = 5,
    carrier_iterations: int = 4,
    force: bool = False,
) -> BoundedResult:
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be finite and positive.")
    soft = q / 2 if soft_bound is None else soft_bound
    hard = q if hard_bound is None else hard_bound
    carrier = robust_temporal_carrier(
        window.fields,
        window.invalid,
        degree=carrier_degree,
        iterations=carrier_iterations,
    )
    preserve = target_is_already_smooth(
        window.fields,
        window.invalid,
        carrier,
        window.target_local_index,
        soft,
        hard,
    ) and not force
    if preserve:
        cycles = np.zeros(window.fields.shape, dtype=np.int16)
        corrected = window.fields.copy()
        solver_info = None
    else:
        corrected, cycles, solver_info = solve_bounded_correction(
            window.fields,
            window.invalid,
            carrier,
            q,
            soft_bound=soft,
            hard_bound=hard,
            preservation_weight=preservation_weight,
            spatial_weight=spatial_weight,
            temporal_weight=temporal_weight,
        )
    return BoundedResult(corrected, cycles, carrier, solver_info, preserve)


def _json_arrays(metrics: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {name: [float(value) for value in values] for name, values in metrics.items()}


def _save_target_plot(
    path: Path, window: BoundedWindow, result: BoundedResult, q: float, dpi: int
) -> None:
    index = window.target_local_index
    original = window.fields[index]
    candidate = result.corrected[index]
    cycles = result.correction_cycles[index]
    valid_values = np.concatenate((original[np.isfinite(original)], candidate[np.isfinite(candidate)]))
    norm = Normalize(float(np.min(valid_values)), float(np.max(valid_values)))
    cycle_bound = max(1.0, float(np.max(np.abs(cycles))))
    cycle_norm = TwoSlopeNorm(vmin=-cycle_bound, vcenter=0.0, vmax=cycle_bound)
    row, column = original.shape[0] // 2, original.shape[1] // 2
    extent = (window.info.x[0], window.info.x[-1], window.info.y[0], window.info.y[-1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axes_item, field, title in zip(
        axes[0, :2], (original, candidate), ("Original", "Bounded candidate")
    ):
        image = axes_item.imshow(
            field, origin="lower", extent=extent, aspect="auto",
            cmap="viridis", norm=norm, interpolation="none",
        )
        axes_item.set_title(title)
    fig.colorbar(image, ax=axes[0, :2], label=window.info.units["z"] or "height")
    cycle_image = axes[0, 2].imshow(
        cycles, origin="lower", extent=extent, aspect="auto",
        cmap="coolwarm", norm=cycle_norm, interpolation="none",
    )
    axes[0, 2].set_title("Integer correction cycles")
    fig.colorbar(cycle_image, ax=axes[0, 2], label="(candidate - original) / q")
    axes[1, 0].plot(window.info.x, original[row], label="original")
    axes[1, 0].plot(window.info.x, candidate[row], label="candidate")
    axes[1, 0].set_title(f"Center row {row}")
    axes[1, 1].plot(window.info.y, original[:, column], label="original")
    axes[1, 1].plot(window.info.y, candidate[:, column], label="candidate")
    axes[1, 1].set_title(f"Center column {column}")
    original_edges = _spatial_values(original, window.invalid[index])
    candidate_edges = _spatial_values(candidate, window.invalid[index])
    axes[1, 2].hist(original_edges, bins=100, density=True, alpha=0.5, label="original")
    axes[1, 2].hist(candidate_edges, bins=100, density=True, alpha=0.5, label="candidate")
    axes[1, 2].axvline(q / 2, color="black", linestyle=":", label="q/2")
    axes[1, 2].axvline(q, color="black", linestyle="--", label="q")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Spatial increment distribution")
    for axes_item in axes[1]:
        axes_item.grid(alpha=0.2)
        axes_item.legend()
    fig.suptitle(
        f"Target {int(window.info.source_positions[index])}; q={q:g}; "
        f"smooth gate={'no-op' if result.gate_preserved_original else 'solve'}"
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _save_window_plot(
    path: Path,
    window: BoundedWindow,
    result: BoundedResult,
    original_spatial: dict[str, np.ndarray],
    candidate_spatial: dict[str, np.ndarray],
    original_temporal: dict[str, np.ndarray],
    candidate_temporal: dict[str, np.ndarray],
    q: float,
    dpi: int,
) -> None:
    positions = window.info.source_positions
    changed = np.mean(result.correction_cycles != 0, axis=(1, 2))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, title, axes_item in (
        ("p95", "Spatial increment p95", axes[0, 0]),
        ("p999", "Spatial increment p99.9", axes[0, 1]),
    ):
        axes_item.plot(positions, original_spatial[name], label="original")
        axes_item.plot(positions, candidate_spatial[name], label="candidate")
        axes_item.axhline(q, color="black", linestyle="--", label="q")
        axes_item.set_title(title)
    axes[1, 0].plot(
        positions[1:], original_temporal["fraction_above_hard"], label="original"
    )
    axes[1, 0].plot(
        positions[1:], candidate_temporal["fraction_above_hard"], label="candidate"
    )
    axes[1, 0].set_title("Detrended temporal fraction above q")
    axes[1, 1].plot(positions, changed, label="candidate")
    axes[1, 1].set_title("Changed-pixel fraction")
    for axes_item in axes.ravel():
        axes_item.grid(alpha=0.2)
        axes_item.legend()
    axes[1, 0].set_xlabel("source timestep position")
    axes[1, 1].set_xlabel("source timestep position")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _default_output_dir(
    source: Path, start_anchor: int, stop_anchor: int, target: int, q: float
) -> Path:
    q_label = f"{q:.9g}".replace(".", "p")
    return (
        Path("runs/phase_unwrap_bounded")
        / source.stem
        / f"anchors_{start_anchor}_{stop_anchor}_target_{target}_q_{q_label}"
    )


def run_bounded_diagnostic(
    raw_input: Path,
    start_anchor: int,
    stop_anchor: int,
    target: int,
    q: float,
    *,
    output_dir: Path | None = None,
    preservation_weight: float = 0.05,
    spatial_weight: float = 4.0,
    temporal_weight: float = 1.0,
    carrier_degree: int = 5,
    carrier_iterations: int = 4,
    force: bool = False,
    dpi: int = 160,
    overwrite: bool = False,
) -> Path:
    source = Path(raw_input).expanduser().resolve()
    output = Path(
        output_dir
        or _default_output_dir(source, start_anchor, stop_anchor, target, q)
    ).expanduser().resolve()
    existing = [output / name for name in OUTPUT_FILENAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Bounded diagnostic already exists ({existing[0]}); pass --overwrite."
        )
    window = load_raw_interval(source, start_anchor, stop_anchor, target)
    result = bounded_correction(
        window,
        q,
        preservation_weight=preservation_weight,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
        carrier_degree=carrier_degree,
        carrier_iterations=carrier_iterations,
        force=force,
    )
    soft, hard = q / 2, q
    original_spatial = spatial_metrics(window.fields, window.invalid, soft, hard)
    candidate_spatial = spatial_metrics(result.corrected, window.invalid, soft, hard)
    original_temporal = temporal_residual_metrics(
        window.fields, window.invalid, result.temporal_carrier, soft, hard
    )
    candidate_temporal = temporal_residual_metrics(
        result.corrected, window.invalid, result.temporal_carrier, soft, hard
    )

    output.mkdir(parents=True, exist_ok=True)
    _save_target_plot(output / "target_comparison.png", window, result, q, dpi)
    _save_window_plot(
        output / "window_diagnostics.png",
        window,
        result,
        original_spatial,
        candidate_spatial,
        original_temporal,
        candidate_temporal,
        q,
        dpi,
    )
    np.savez_compressed(
        output / "bounded_candidate.npz",
        source_positions=window.info.source_positions,
        frame_numbers=window.info.frame_numbers,
        time=window.info.time,
        x=window.info.x,
        y=window.info.y,
        corrected_candidate=result.corrected,
        correction_cycles=result.correction_cycles,
        temporal_carrier=result.temporal_carrier,
        invalid_mask=window.invalid,
        q=np.float64(q),
    )
    index = window.target_local_index
    cycle_values = result.correction_cycles[index][~window.invalid[index]]
    integer_error = np.abs(
        (result.corrected[index][~window.invalid[index]] - window.fields[index][~window.invalid[index]])
        / q
        - cycle_values
    )
    summary: dict[str, Any] = {
        "source": str(source),
        "source_opened_read_only": True,
        "start_anchor": int(start_anchor),
        "stop_anchor": int(stop_anchor),
        "target_timestep_position": int(target),
        "q": float(q),
        "soft_bound": float(soft),
        "hard_bound": float(hard),
        "assumption": (
            "Prefer corrected spatial and detrended-temporal increments below q/2; "
            "treat increments at or above q as strong evidence for a cycle adjustment."
        ),
        "endpoint_cycle_labels_fixed_to_zero": True,
        "smooth_target_gate": {
            "criterion": (
                "original spatial p95 <= q/2 and p99 <= q; adjacent detrended-"
                "temporal p95 <= q and no more than 5% of residuals above q"
            ),
            "preserved_original": result.gate_preserved_original,
            "force_override": bool(force),
        },
        "solver": {
            "type": "weighted graph least-squares relaxation followed by integer rounding",
            "conjugate_gradient_info": result.solver_info,
            "preservation_weight": preservation_weight,
            "spatial_weight": spatial_weight,
            "temporal_weight": temporal_weight,
            "carrier_degree": carrier_degree,
            "carrier_iterations": carrier_iterations,
        },
        "processing": (
            "The output changes heights only by integer multiples of q. The robust "
            "polynomial temporal carrier is used only in the objective and is not "
            "applied to or subtracted from the saved candidate."
        ),
        "target": {
            "source_position": int(target),
            "changed_valid_pixel_fraction": float(np.mean(cycle_values != 0)),
            "correction_cycle_min": int(np.min(cycle_values)),
            "correction_cycle_max": int(np.max(cycle_values)),
            "max_integer_cycle_error": float(np.max(integer_error)),
            "original_spatial": {
                name: float(values[index]) for name, values in original_spatial.items()
            },
            "candidate_spatial": {
                name: float(values[index]) for name, values in candidate_spatial.items()
            },
        },
        "window_metrics": {
            "original_spatial": _json_arrays(original_spatial),
            "candidate_spatial": _json_arrays(candidate_spatial),
            "original_detrended_temporal": _json_arrays(original_temporal),
            "candidate_detrended_temporal": _json_arrays(candidate_temporal),
            "changed_pixel_fraction": [
                float(value)
                for value in np.mean(result.correction_cycles != 0, axis=(1, 2))
            ],
        },
        "interpretation": (
            "This is a candidate conditional on the stated bounds. Remaining above-bound "
            "edges are explicit assumption violations. Lower discontinuity metrics do not "
            "prove that the selected phase branches are physical."
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(
        f"Target {target}: gate={'preserved original' if result.gate_preserved_original else 'solved'}; "
        f"changed={np.mean(cycle_values != 0):.3%}; "
        f"spatial fraction > q: {original_spatial['fraction_above_hard'][index]:.3%} -> "
        f"{candidate_spatial['fraction_above_hard'][index]:.3%}.\n"
        f"Saved bounded candidate diagnostics to {output}"
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--start-anchor", type=int, required=True)
    parser.add_argument("--stop-anchor", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--spatial-weight", type=float, default=4.0)
    parser.add_argument("--temporal-weight", type=float, default=1.0)
    parser.add_argument("--carrier-degree", type=int, default=5)
    parser.add_argument("--carrier-iterations", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Bypass the smooth-target no-op gate.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_bounded_diagnostic(
        args.raw_input,
        args.start_anchor,
        args.stop_anchor,
        args.target,
        args.q,
        output_dir=args.output_dir,
        preservation_weight=args.preservation_weight,
        spatial_weight=args.spatial_weight,
        temporal_weight=args.temporal_weight,
        carrier_degree=args.carrier_degree,
        carrier_iterations=args.carrier_iterations,
        force=args.force,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
