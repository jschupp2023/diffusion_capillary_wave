"""Compare the production spatial cleaner with conservative masked-reference pilots.

This is a diagnostic only.  It does not modify the production cleaner or any HDF5
file.  The preferred ``hybrid`` pilot uses height-gradient inconsistency as strong
evidence and grows only into phase-incoherent, residue-rich components.  It then
chooses integer cycles relative to a minimum-gradient-energy harmonic reference.
The older residue-only ``harmonic`` and ``weighted`` pilots remain available for
comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label, uniform_filter
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr, spsolve

from data_analysis.pod.phase_unwrap_gradient_poisson import (
    restore_frame_poisson,
)
from data_analysis.pod.pod_2d import _frame_key


DEFAULT_RAW = Path(
    "/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5"
)
DEFAULT_OUTPUT = Path("runs/phase_unwrap_study/hybrid_poisson_pilot")


def wrapped_phase_quality(
    frame: np.ndarray,
    q: float,
    window: int,
    density_threshold: float,
    min_component_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return wrapped phase, residues, residue pixels, and a low-confidence mask."""
    phase = np.angle(np.exp(2j * np.pi * frame / q))
    dx = np.angle(np.exp(1j * np.diff(phase, axis=1)))
    dy = np.angle(np.exp(1j * np.diff(phase, axis=0)))
    loop = dx[:-1] + dy[:, 1:] - dx[1:] - dy[:, :-1]
    residues = np.rint(loop / (2 * np.pi)).astype(np.int8)

    residue_pixels = np.zeros_like(frame, dtype=bool)
    active = residues != 0
    residue_pixels[:-1, :-1] |= active
    residue_pixels[1:, :-1] |= active
    residue_pixels[:-1, 1:] |= active
    residue_pixels[1:, 1:] |= active

    density = uniform_filter(residue_pixels.astype(float), size=window, mode="nearest")
    low_confidence = density >= density_threshold
    if min_component_pixels > 1:
        components, count = label(low_confidence)
        sizes = np.bincount(components.ravel())
        keep = sizes >= min_component_pixels
        keep[0] = False
        low_confidence = keep[components] if count else np.zeros_like(low_confidence)
    return phase, residues, residue_pixels, low_confidence


def phase_gradient_coherence(phase: np.ndarray, window: int) -> np.ndarray:
    """Circular concentration of wrapped x/y phase increments in a local window."""
    dpx = np.angle(np.exp(1j * np.diff(phase, axis=1)))
    dpy = np.angle(np.exp(1j * np.diff(phase, axis=0)))

    def concentration(delta: np.ndarray) -> np.ndarray:
        real = uniform_filter(np.cos(delta), size=window, mode="nearest")
        imag = uniform_filter(np.sin(delta), size=window, mode="nearest")
        return np.hypot(real, imag)

    coherence_x = concentration(dpx)
    coherence_y = concentration(dpy)
    total = np.zeros_like(phase, dtype=np.float64)
    count = np.zeros_like(phase, dtype=np.float64)
    total[:, :-1] += coherence_x
    count[:, :-1] += 1
    total[:, 1:] += coherence_x
    count[:, 1:] += 1
    total[:-1, :] += coherence_y
    count[:-1, :] += 1
    total[1:, :] += coherence_y
    count[1:, :] += 1
    return total / count


def height_gradient_dispersion(
    frame: np.ndarray,
    q: float,
    window: int,
) -> np.ndarray:
    """Local gradient-vector standard deviation, normalized by one cycle q."""
    gy, gx = np.gradient(frame.astype(np.float64, copy=False))
    mean_gx = uniform_filter(gx, size=window, mode="nearest")
    mean_gy = uniform_filter(gy, size=window, mode="nearest")
    var_gx = np.maximum(
        0.0,
        uniform_filter(gx * gx, size=window, mode="nearest") - mean_gx**2,
    )
    var_gy = np.maximum(
        0.0,
        uniform_filter(gy * gy, size=window, mode="nearest") - mean_gy**2,
    )
    return np.sqrt(var_gx + var_gy) / q


def conservative_hybrid_mask(
    frame: np.ndarray,
    phase: np.ndarray,
    residue_pixels: np.ndarray,
    q: float,
    residue_window: int,
    residue_candidate_threshold: float,
    coherence_window: int,
    phase_coherence_threshold: float,
    height_window: int,
    height_seed_threshold: float,
    seed_dilation_pixels: int,
    min_component_pixels: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Return candidate components supported by height-domain inconsistency seeds."""
    residue_density = uniform_filter(
        residue_pixels.astype(float), size=residue_window, mode="nearest"
    )
    phase_coherence = phase_gradient_coherence(phase, coherence_window)
    height_score = height_gradient_dispersion(frame, q, height_window)
    height_seeds = height_score >= height_seed_threshold
    seed_support = (
        binary_dilation(
            height_seeds,
            structure=np.ones((3, 3), dtype=bool),
            iterations=seed_dilation_pixels,
        )
        if seed_dilation_pixels > 0
        else height_seeds.copy()
    )
    phase_candidates = (
        (residue_density >= residue_candidate_threshold)
        & (phase_coherence <= phase_coherence_threshold)
    )

    structure = np.ones((3, 3), dtype=np.int8)
    components, count = label(phase_candidates, structure=structure)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    supported = np.bincount(
        components[seed_support].ravel(), minlength=count + 1
    ) > 0
    keep = (sizes >= min_component_pixels) & supported
    keep[0] = False
    mask = keep[components] if count else np.zeros_like(frame, dtype=bool)
    diagnostics: dict[str, np.ndarray | int] = {
        "residue_density": residue_density,
        "phase_coherence": phase_coherence,
        "height_score": height_score,
        "height_seeds": height_seeds,
        "seed_support": seed_support,
        "phase_candidates": phase_candidates,
        "candidate_component_count": int(count),
        "retained_component_count": int(np.count_nonzero(keep)),
    }
    return mask, diagnostics


def screened_weighted_reference(
    frame: np.ndarray,
    q: float,
    tau: float,
    low_confidence: np.ndarray,
    weak_edge_weight: float,
    trusted_data_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Solve a weighted gradient fit with a data term on trusted pixels."""
    ny, nx = frame.shape
    n_pixels = ny * nx

    dx = np.diff(frame, axis=1)
    dy = np.diff(frame, axis=0)
    gx = dx.copy()
    gy = dy.copy()
    jump_x = np.abs(dx) >= tau
    jump_y = np.abs(dy) >= tau
    gx[jump_x] -= q * np.rint(dx[jump_x] / q)
    gy[jump_y] -= q * np.rint(dy[jump_y] / q)

    indices = np.arange(n_pixels).reshape(ny, nx)
    left = indices[:, :-1].ravel()
    right = left + 1
    upper = indices[:-1, :].ravel()
    lower = upper + nx

    wx = np.where(
        (~low_confidence[:, :-1] & ~low_confidence[:, 1:]).ravel(),
        1.0,
        weak_edge_weight,
    )
    wy = np.where(
        (~low_confidence[:-1, :] & ~low_confidence[1:, :]).ravel(),
        1.0,
        weak_edge_weight,
    )
    sx = np.sqrt(wx)
    sy = np.sqrt(wy)

    n_x = left.size
    n_y = upper.size
    rows = np.concatenate(
        [np.arange(n_x), np.arange(n_x), n_x + np.arange(n_y), n_x + np.arange(n_y)]
    )
    cols = np.concatenate([left, right, upper, lower])
    values = np.concatenate([-sx, sx, -sy, sy])
    gradient_matrix = coo_matrix(
        (values, (rows, cols)), shape=(n_x + n_y, n_pixels)
    ).tocsr()
    rhs = np.concatenate([sx * gx.ravel(), sy * gy.ravel()])

    trusted = np.flatnonzero(~low_confidence.ravel())
    data_scale = np.sqrt(trusted_data_weight)
    data_matrix = coo_matrix(
        (
            np.full(trusted.size, data_scale),
            (np.arange(trusted.size), trusted),
        ),
        shape=(trusted.size, n_pixels),
    ).tocsr()
    system = vstack([gradient_matrix, data_matrix], format="csr")
    rhs = np.concatenate([rhs, data_scale * frame.ravel()[trusted]])

    fit = lsqr(system, rhs, atol=1e-7, btol=1e-7, iter_lim=1500)
    reference = fit[0].reshape(frame.shape)
    cycles = np.rint((frame - reference) / q).astype(np.int32)
    solver = {
        "iterations": int(fit[2]),
        "residual_norm": float(fit[3]),
        "condition_estimate": float(fit[6]),
    }
    return reference, cycles, solver


def harmonic_masked_reference(
    frame: np.ndarray,
    low_confidence: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Harmonically continue trusted heights through the low-confidence mask.

    The unknown values minimize the sum of squared nearest-neighbor gradients
    while trusted pixels remain fixed.  Integer-cycle corrections are then chosen
    independently as the branch nearest to that reference.  This is a diagnostic
    prior, not a recovery of unresolved sub-cycle structure.
    """
    if frame.shape != low_confidence.shape:
        raise ValueError("frame and low_confidence must have identical shapes")
    if not np.all(np.isfinite(frame)):
        raise ValueError("harmonic pilot currently requires finite frame values")

    reference = frame.astype(np.float64, copy=True)
    unknown_flat = np.flatnonzero(low_confidence.ravel())
    if unknown_flat.size == 0:
        return reference, np.zeros(frame.shape, dtype=np.int32), {
            "unknown_pixels": 0,
            "linear_residual_norm": 0.0,
        }
    if unknown_flat.size == frame.size:
        raise ValueError("harmonic reference requires at least one trusted pixel")

    ny, nx = frame.shape
    unknown_index = np.full(frame.size, -1, dtype=np.int64)
    unknown_index[unknown_flat] = np.arange(unknown_flat.size)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros(unknown_flat.size, dtype=np.float64)

    for equation, flat in enumerate(unknown_flat):
        iy, ix = divmod(int(flat), nx)
        degree = 0
        for jy, jx in (
            (iy - 1, ix),
            (iy + 1, ix),
            (iy, ix - 1),
            (iy, ix + 1),
        ):
            if not (0 <= jy < ny and 0 <= jx < nx):
                continue
            degree += 1
            neighbor_flat = jy * nx + jx
            if low_confidence[jy, jx]:
                rows.append(equation)
                cols.append(int(unknown_index[neighbor_flat]))
                values.append(-1.0)
            else:
                rhs[equation] += frame[jy, jx]
        rows.append(equation)
        cols.append(equation)
        values.append(float(degree))

    laplacian = coo_matrix(
        (values, (rows, cols)),
        shape=(unknown_flat.size, unknown_flat.size),
    ).tocsr()
    solution = spsolve(laplacian, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("harmonic reference solve did not return finite values")

    reference.ravel()[unknown_flat] = solution
    cycles = np.zeros(frame.shape, dtype=np.int32)
    cycles[low_confidence] = np.rint(
        (frame[low_confidence] - reference[low_confidence]) / q
    ).astype(np.int32)
    residual = laplacian @ solution - rhs
    solver: dict[str, float | int] = {
        "unknown_pixels": int(unknown_flat.size),
        "linear_residual_norm": float(np.linalg.norm(residual)),
    }
    return reference, cycles, solver


def boundary_violation_outside_pixels(
    field: np.ndarray,
    mask: np.ndarray,
    limit: float,
) -> np.ndarray:
    """Trusted-side endpoints of mask-boundary edges exceeding ``limit``."""
    outside = np.zeros_like(mask, dtype=bool)

    dx = np.abs(np.diff(field, axis=1))
    bad_x = (mask[:, :-1] ^ mask[:, 1:]) & (dx > limit)
    outside[:, :-1] |= bad_x & ~mask[:, :-1]
    outside[:, 1:] |= bad_x & ~mask[:, 1:]

    dy = np.abs(np.diff(field, axis=0))
    bad_y = (mask[:-1, :] ^ mask[1:, :]) & (dy > limit)
    outside[:-1, :] |= bad_y & ~mask[:-1, :]
    outside[1:, :] |= bad_y & ~mask[1:, :]
    return outside


def max_gradient_crossing_mask(field: np.ndarray, mask: np.ndarray) -> float:
    """Maximum neighbor difference on edges crossing a binary-mask boundary."""
    dx = np.abs(np.diff(field, axis=1))
    dy = np.abs(np.diff(field, axis=0))
    values = []
    cross_x = mask[:, :-1] ^ mask[:, 1:]
    cross_y = mask[:-1, :] ^ mask[1:, :]
    if np.any(cross_x):
        values.append(float(np.max(dx[cross_x])))
    if np.any(cross_y):
        values.append(float(np.max(dy[cross_y])))
    return max(values, default=0.0)


def harmonic_reference_with_boundary_growth(
    frame: np.ndarray,
    initial_mask: np.ndarray,
    envelope: np.ndarray,
    q: float,
    boundary_gradient_limit: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Grow a hybrid mask only where its own correction creates a boundary cliff.

    Growth is restricted to the broader residue-derived envelope.  This is a
    hysteresis/safety step, not a second general-purpose segmentation rule.
    """
    mask = initial_mask.copy()
    initial_count = int(np.count_nonzero(mask))
    growth_iterations = 0
    best: tuple[
        tuple[float, int, int],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, float | int],
        int,
    ] | None = None

    for _ in range(max_iterations + 1):
        reference, cycles, base_solver = harmonic_masked_reference(frame, mask, q)
        corrected = frame - q * cycles
        outside = boundary_violation_outside_pixels(
            corrected, mask, boundary_gradient_limit
        )
        violation_count = int(np.count_nonzero(outside))
        added_count = int(np.count_nonzero(mask)) - initial_count
        score = (
            max_gradient_crossing_mask(corrected, mask),
            violation_count,
            added_count,
        )
        if best is None or score < best[0]:
            best = (
                score,
                reference.copy(),
                cycles.copy(),
                mask.copy(),
                dict(base_solver),
                growth_iterations,
            )
        if violation_count == 0:
            break
        additions = outside & envelope & ~mask
        if not np.any(additions) or growth_iterations >= max_iterations:
            break
        if np.count_nonzero(mask | additions) == mask.size:
            break
        mask |= additions
        growth_iterations += 1

    assert best is not None
    _, reference, cycles, mask, base_solver, selected_iterations = best
    corrected = frame - q * cycles
    unresolved = boundary_violation_outside_pixels(
        corrected, mask, boundary_gradient_limit
    )
    solver: dict[str, float | int | bool] = dict(base_solver)
    solver.update(
        {
            "boundary_growth_iterations": selected_iterations,
            "boundary_growth_attempted_iterations": growth_iterations,
            "boundary_growth_added_pixels": int(np.count_nonzero(mask))
            - initial_count,
            "boundary_gradient_limit_um": float(boundary_gradient_limit),
            "unresolved_boundary_violation_pixels": int(
                np.count_nonzero(unresolved)
            ),
            "boundary_growth_converged": not np.any(unresolved),
        }
    )
    return reference, cycles, mask, solver


def max_gradient(frame: np.ndarray) -> float:
    return float(
        max(
            np.max(np.abs(np.diff(frame, axis=1))),
            np.max(np.abs(np.diff(frame, axis=0))),
        )
    )


def max_gradient_touching_mask(frame: np.ndarray, mask: np.ndarray) -> float:
    """Maximum neighbor difference over edges with at least one masked endpoint."""
    dx = np.abs(np.diff(frame, axis=1))
    dy = np.abs(np.diff(frame, axis=0))
    mask_x = mask[:, :-1] | mask[:, 1:]
    mask_y = mask[:-1, :] | mask[1:, :]
    candidates = []
    if np.any(mask_x):
        candidates.append(float(np.max(dx[mask_x])))
    if np.any(mask_y):
        candidates.append(float(np.max(dy[mask_y])))
    return max(candidates, default=0.0)


def shade_masked_segments(axis: plt.Axes, x: np.ndarray, mask: np.ndarray) -> None:
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in changes.reshape(-1, 2):
        axis.axvspan(x[start], x[min(stop - 1, len(x) - 1)], color="orange", alpha=0.14)


def contour_mask(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    **kwargs: object,
) -> None:
    """Draw a binary-mask boundary without warnings for constant masks."""
    if np.any(mask) and not np.all(mask):
        axis.contour(x, y, mask, levels=[0.5], **kwargs)


def analyze_frame(
    frame: np.ndarray,
    frame_number: int,
    x: np.ndarray,
    y: np.ndarray,
    output: Path,
    q: float,
    tau: float,
    window: int,
    density_threshold: float,
    min_component_pixels: int,
    weak_edge_weight: float,
    trusted_data_weight: float,
    pilot_method: str,
    min_cliff_edge_fraction: float,
    min_cliff_component_pixels: int,
    residue_candidate_threshold: float,
    coherence_window: int,
    phase_coherence_threshold: float,
    height_window: int,
    height_seed_threshold: float,
    seed_dilation_pixels: int,
    boundary_gradient_limit_factor: float,
    max_boundary_growth_iterations: int,
) -> dict[str, object]:
    phase, residues, residue_pixels, residue_mask = wrapped_phase_quality(
        frame, q, window, density_threshold, min_component_pixels
    )
    pilot_mask = residue_mask
    hybrid_diagnostics: dict[str, np.ndarray | int] | None = None
    if pilot_method == "hybrid":
        pilot_mask, hybrid_diagnostics = conservative_hybrid_mask(
            frame,
            phase,
            residue_pixels,
            q,
            window,
            residue_candidate_threshold,
            coherence_window,
            phase_coherence_threshold,
            height_window,
            height_seed_threshold,
            seed_dilation_pixels,
            min_component_pixels,
        )
    current_result = restore_frame_poisson(frame, q=q, tau=tau, method="integer")
    current_cycles = current_result.cycles_inferred.astype(np.int32)
    current = current_result.restored.astype(np.float64)
    raw_dx = np.abs(np.diff(frame, axis=1))
    raw_dy = np.abs(np.diff(frame, axis=0))
    cliff_edge_count = int(
        np.count_nonzero(raw_dx >= tau) + np.count_nonzero(raw_dy >= tau)
    )
    spatial_edge_count = int(raw_dx.size + raw_dy.size)
    cliff_edge_fraction = cliff_edge_count / spatial_edge_count
    cliff_seed = np.zeros(frame.shape, dtype=bool)
    cliff_x = raw_dx >= tau
    cliff_y = raw_dy >= tau
    cliff_seed[:, :-1] |= cliff_x
    cliff_seed[:, 1:] |= cliff_x
    cliff_seed[:-1, :] |= cliff_y
    cliff_seed[1:, :] |= cliff_y
    cliff_components, cliff_component_count = label(
        cliff_seed, structure=np.ones((3, 3), dtype=np.int8)
    )
    cliff_component_sizes = np.bincount(cliff_components.ravel())[1:]
    largest_cliff_component_pixels = int(
        np.max(cliff_component_sizes) if cliff_component_sizes.size else 0
    )
    uses_harmonic_reference = pilot_method in {"harmonic", "hybrid"}
    cliff_gate_applied = uses_harmonic_reference and (
        min_cliff_edge_fraction > 0 or min_cliff_component_pixels > 0
    )
    fraction_gate_passed = cliff_edge_fraction >= min_cliff_edge_fraction
    component_gate_passed = (
        largest_cliff_component_pixels >= min_cliff_component_pixels
    )
    cliff_gate_passed = (
        not cliff_gate_applied or (fraction_gate_passed and component_gate_passed)
    )
    if pilot_method == "weighted":
        pilot_reference, pilot_cycles, solver = screened_weighted_reference(
            frame,
            q,
            tau,
            pilot_mask,
            weak_edge_weight,
            trusted_data_weight,
        )
        pilot_title = "Screened weighted pilot"
        pilot_label = "weighted pilot"
    elif uses_harmonic_reference and cliff_gate_passed:
        if pilot_method == "hybrid":
            initial_pilot_mask = pilot_mask.copy()
            pilot_reference, pilot_cycles, pilot_mask, solver = (
                harmonic_reference_with_boundary_growth(
                    frame,
                    initial_pilot_mask,
                    residue_mask
                    | np.asarray(hybrid_diagnostics["seed_support"], dtype=bool),
                    q,
                    boundary_gradient_limit_factor * tau,
                    max_boundary_growth_iterations,
                )
            )
            pilot_title = "Conservative hybrid pilot"
            pilot_label = "hybrid pilot"
        else:
            pilot_reference, pilot_cycles, solver = harmonic_masked_reference(
                frame, pilot_mask, q
            )
            pilot_title = "Harmonic-reference pilot"
            pilot_label = "harmonic pilot"
    elif uses_harmonic_reference:
        pilot_reference = frame.astype(np.float64, copy=True)
        pilot_cycles = np.zeros(frame.shape, dtype=np.int32)
        solver = {
            "skipped": True,
            "reason": "raw cliff evidence did not pass all conservative gates",
        }
        pilot_title = f"{pilot_method.capitalize()} pilot skipped by cliff gate"
        pilot_label = f"{pilot_method} pilot (unchanged)"
    else:
        raise ValueError(f"unknown pilot method: {pilot_method}")
    pilot = frame - q * pilot_cycles

    distance = distance_transform_edt(~residue_pixels)
    current_modified = current_cycles != 0
    pilot_modified = pilot_cycles != 0
    trusted = ~pilot_mask
    outside_disagreement = trusted & (current_cycles != pilot_cycles)
    diagnostic_rows = {1740: 80, 1840: 141}
    row = diagnostic_rows.get(
        frame_number, int(np.argmax(np.count_nonzero(outside_disagreement, axis=1)))
    )

    metrics: dict[str, object] = {
        "frame": int(frame_number),
        "shape": list(frame.shape),
        "residue_count": int(np.count_nonzero(residues)),
        "residue_plaquette_fraction": float(np.mean(residues != 0)),
        "residue_mask_fraction": float(np.mean(residue_mask)),
        "pilot_mask_fraction": float(np.mean(pilot_mask)),
        "profile_row": row,
        "raw_max_gradient_um": max_gradient(frame),
        "raw_cliff_edge_count": cliff_edge_count,
        "spatial_edge_count": spatial_edge_count,
        "raw_cliff_edge_fraction": float(cliff_edge_fraction),
        "min_cliff_edge_fraction": float(min_cliff_edge_fraction),
        "cliff_component_count": int(cliff_component_count),
        "largest_cliff_component_pixels": largest_cliff_component_pixels,
        "min_cliff_component_pixels": int(min_cliff_component_pixels),
        "fraction_gate_passed": bool(fraction_gate_passed),
        "component_gate_passed": bool(component_gate_passed),
        "cliff_gate_applied": cliff_gate_applied,
        "cliff_gate_passed": cliff_gate_passed,
        "current_max_gradient_um": max_gradient(current),
        "current_max_gradient_touching_mask_um": max_gradient_touching_mask(
            current, pilot_mask
        ),
        "pilot_method": pilot_method,
        "pilot_max_gradient_um": max_gradient(pilot),
        "pilot_max_gradient_touching_mask_um": max_gradient_touching_mask(
            pilot, pilot_mask
        ),
        "current_modified_fraction": float(np.mean(current_modified)),
        "pilot_modified_fraction": float(np.mean(pilot_modified)),
        "current_trusted_modified_fraction": float(np.mean(current_modified[trusted])),
        "pilot_trusted_modified_fraction": float(np.mean(pilot_modified[trusted])),
        "current_modified_farther_than_15px_fraction": float(
            np.mean(current_modified & (distance > 15))
        ),
        "pilot_modified_farther_than_15px_fraction": float(
            np.mean(pilot_modified & (distance > 15))
        ),
        "solver": solver,
    }

    if hybrid_diagnostics is not None:
        height_seeds = np.asarray(hybrid_diagnostics["height_seeds"], dtype=bool)
        phase_candidates = np.asarray(
            hybrid_diagnostics["phase_candidates"], dtype=bool
        )
        phase_coherence = np.asarray(
            hybrid_diagnostics["phase_coherence"], dtype=float
        )
        height_score = np.asarray(hybrid_diagnostics["height_score"], dtype=float)
        metrics.update(
            {
                "height_seed_fraction": float(np.mean(height_seeds)),
                "phase_candidate_fraction": float(np.mean(phase_candidates)),
                "candidate_component_count": int(
                    hybrid_diagnostics["candidate_component_count"]
                ),
                "retained_component_count": int(
                    hybrid_diagnostics["retained_component_count"]
                ),
                "initial_pilot_mask_fraction": float(
                    np.mean(initial_pilot_mask)
                )
                if cliff_gate_passed
                else float(np.mean(pilot_mask)),
                "boundary_growth_added_pixels": int(
                    solver.get("boundary_growth_added_pixels", 0)
                ),
                "boundary_growth_converged": bool(
                    solver.get("boundary_growth_converged", True)
                ),
                "unresolved_boundary_violation_pixels": int(
                    solver.get("unresolved_boundary_violation_pixels", 0)
                ),
                "pilot_mask_phase_coherence_median": float(
                    np.median(phase_coherence[pilot_mask])
                )
                if np.any(pilot_mask)
                else None,
                "pilot_mask_height_score_median": float(
                    np.median(height_score[pilot_mask])
                )
                if np.any(pilot_mask)
                else None,
            }
        )

    if np.any(pilot_mask):
        boundary = np.zeros_like(pilot_mask)
        boundary[:-1, :] |= pilot_mask[1:, :] & trusted[:-1, :]
        boundary[1:, :] |= pilot_mask[:-1, :] & trusted[1:, :]
        boundary[:, :-1] |= pilot_mask[:, 1:] & trusted[:, :-1]
        boundary[:, 1:] |= pilot_mask[:, :-1] & trusted[:, 1:]
        if np.any(boundary):
            metrics["trusted_boundary_height_range_um"] = [
                float(np.min(frame[boundary])),
                float(np.max(frame[boundary])),
            ]
        metrics["pilot_reference_mask_range_um"] = [
            float(np.min(pilot_reference[pilot_mask])),
            float(np.max(pilot_reference[pilot_mask])),
        ]

    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    vmin = float(min(frame.min(), current.min(), pilot.min()))
    vmax = float(max(frame.max(), current.max(), pilot.max()))
    cycle_max = int(
        max(1, np.max(np.abs(current_cycles)), np.max(np.abs(pilot_cycles)))
    )

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=True)
    height_kwargs = dict(origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    im_raw = axes[0, 0].imshow(frame, **height_kwargs)
    axes[0, 0].set_title("(a) Raw height")
    fig.colorbar(im_raw, ax=axes[0, 0], label=r"$h$ [$\mu$m]")

    im_phase = axes[0, 1].imshow(
        phase, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    py, px = np.where(residues != 0)
    axes[0, 1].scatter(x[px], y[py], s=2, c="black", alpha=0.45)
    axes[0, 1].set_title(f"(b) Rewrapped phase and residues (n={len(px):,})")
    fig.colorbar(im_phase, ax=axes[0, 1], label="phase [rad]")

    axes[0, 2].imshow(
        pilot_mask,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    if cliff_gate_applied:
        gate_text = (
            f"cliff gate: {'PASS' if cliff_gate_passed else 'SKIP'}\n"
            f"edges {100*cliff_edge_fraction:.3f}% >= "
            f"{100*min_cliff_edge_fraction:.3f}%; connected support "
            f"{largest_cliff_component_pixels} >= {min_cliff_component_pixels} px"
        )
    elif uses_harmonic_reference:
        gate_text = "cliff gate: disabled"
    else:
        gate_text = "cliff gate: not applied to weighted pilot"
    mask_name = "Hybrid uncertainty mask" if pilot_method == "hybrid" else "Low-confidence mask"
    mask_details = ""
    if hybrid_diagnostics is not None:
        mask_details = (
            f"; seeds {100*np.mean(hybrid_diagnostics['height_seeds']):.1f}%"
        )
        contour_mask(
            axes[0, 2],
            x,
            y,
            np.asarray(hybrid_diagnostics["height_seeds"], dtype=bool),
            colors="cyan",
            linewidths=0.5,
        )
    axes[0, 2].set_title(
        f"(c) {mask_name} ({100*np.mean(pilot_mask):.1f}%{mask_details})\n"
        + gate_text
    )

    axes[0, 3].plot(x, frame[row], color="0.5", lw=1.2, label="raw")
    axes[0, 3].plot(x, current[row], color="tab:blue", lw=1.6, label="current")
    axes[0, 3].plot(x, pilot[row], color="tab:orange", lw=1.6, ls="--", label=pilot_label)
    shade_masked_segments(axes[0, 3], x, pilot_mask[row])
    axes[0, 3].set_title(f"(d) Row {row} profile")
    axes[0, 3].set_xlabel(r"$x$ [$\mu$m]")
    axes[0, 3].set_ylabel(r"$h$ [$\mu$m]")
    axes[0, 3].legend(fontsize=8)
    axes[0, 3].grid(alpha=0.25)

    im_current = axes[1, 0].imshow(current, **height_kwargs)
    contour_mask(
        axes[1, 0], x, y, pilot_mask, colors="orange", linewidths=0.8
    )
    axes[1, 0].set_title(
        f"(e) Current restoration\nmodified={100*np.mean(current_modified):.1f}%, "
        f"trusted modified={100*np.mean(current_modified[trusted]):.1f}%"
    )
    fig.colorbar(im_current, ax=axes[1, 0], label=r"$h$ [$\mu$m]")

    im_weighted = axes[1, 1].imshow(pilot, **height_kwargs)
    contour_mask(
        axes[1, 1], x, y, pilot_mask, colors="orange", linewidths=0.8
    )
    axes[1, 1].set_title(
        f"(f) {pilot_title}\nmodified={100*np.mean(pilot_modified):.1f}%, "
        f"trusted modified={100*np.mean(pilot_modified[trusted]):.1f}%"
    )
    fig.colorbar(im_weighted, ax=axes[1, 1], label=r"$h$ [$\mu$m]")

    cycle_kwargs = dict(
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-cycle_max,
        vmax=cycle_max,
    )
    im_mc = axes[1, 2].imshow(current_cycles, **cycle_kwargs)
    contour_mask(
        axes[1, 2], x, y, pilot_mask, colors="black", linewidths=0.7
    )
    axes[1, 2].set_title("(g) Current integer field $m$")
    fig.colorbar(im_mc, ax=axes[1, 2], label="cycles")

    im_mw = axes[1, 3].imshow(pilot_cycles, **cycle_kwargs)
    contour_mask(
        axes[1, 3], x, y, pilot_mask, colors="black", linewidths=0.7
    )
    axes[1, 3].set_title(f"(h) {pilot_title} integer field $m$")
    fig.colorbar(im_mw, ax=axes[1, 3], label="cycles")

    for axis in axes.flat:
        if axis is not axes[0, 3]:
            axis.set_xlabel(r"$x$ [$\mu$m]")
            axis.set_ylabel(r"$y$ [$\mu$m]")

    fig.suptitle(
        rf"Frame {frame_number}: current versus {pilot_method} masked-reference pilot" "\n"
        rf"diagnostic only; $q={q:.6f}\,\mu\mathrm{{m}}$, "
        rf"$\tau={tau:.1f}\,\mu\mathrm{{m}}$"
        + (
            rf", weak weight={weak_edge_weight:g}, $\lambda={trusted_data_weight:g}$"
            if pilot_method == "weighted"
            else ", minimum-gradient-energy continuation with cliff gate"
        ),
        fontsize=14,
        fontweight="bold",
    )
    figure_path = output / f"frame_{frame_number:05d}_{pilot_method}_poisson_pilot.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    metrics["figure"] = str(figure_path)
    review_flags: list[str] = []
    if int(metrics.get("boundary_growth_added_pixels", 0)) > 0:
        review_flags.append("boundary_mask_growth")
    if not bool(metrics.get("boundary_growth_converged", True)):
        review_flags.append("unresolved_boundary_violation")
    if cliff_gate_passed and max_gradient_touching_mask(
        pilot, pilot_mask
    ) > max_gradient_touching_mask(current, pilot_mask) + 0.25:
        review_flags.append("pilot_mask_gradient_worse_than_current")
    if np.mean(pilot_modified) > np.mean(current_modified) + 1e-12:
        review_flags.append("pilot_modifies_more_than_current")
    if np.mean(pilot_mask) > 0.70:
        review_flags.append("pilot_mask_above_70pct")
    if not cliff_gate_passed and max_gradient(frame) > 2 * tau:
        review_flags.append("gate_skip_despite_raw_gradient_above_2tau")
    metrics["review_flags"] = review_flags
    metrics["requires_review"] = bool(review_flags)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frames", type=int, nargs="+")
    selection.add_argument(
        "--random-frames",
        type=int,
        help="Sample this many available frames without replacement.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full per-frame JSON instead of one concise audit line.",
    )
    parser.add_argument(
        "--q",
        type=float,
        required=True,
        help="Height corresponding to one full 2*pi phase cycle, in height units.",
    )
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--residue-window", type=int, default=15)
    parser.add_argument("--residue-density-threshold", type=float, default=0.20)
    parser.add_argument("--min-mask-component-pixels", type=int, default=100)
    parser.add_argument("--weak-edge-weight", type=float, default=0.01)
    parser.add_argument("--trusted-data-weight", type=float, default=0.1)
    parser.add_argument(
        "--pilot",
        choices=("weighted", "harmonic", "hybrid"),
        default="hybrid",
        help="Reference used inside the residue-derived low-confidence mask.",
    )
    parser.add_argument("--residue-candidate-threshold", type=float, default=0.10)
    parser.add_argument("--coherence-window", type=int, default=7)
    parser.add_argument("--phase-coherence-threshold", type=float, default=0.70)
    parser.add_argument("--height-window", type=int, default=7)
    parser.add_argument("--height-seed-threshold", type=float, default=0.35)
    parser.add_argument("--seed-dilation-pixels", type=int, default=3)
    parser.add_argument(
        "--boundary-gradient-limit-factor",
        type=float,
        default=1.05,
        help=(
            "Grow the hybrid mask when its correction leaves a boundary jump "
            "larger than this multiple of tau (default: 1.05)."
        ),
    )
    parser.add_argument(
        "--max-boundary-growth-iterations",
        type=int,
        default=30,
        help="Maximum hybrid-mask safety-growth iterations (default: 30).",
    )
    parser.add_argument(
        "--min-cliff-edge-fraction",
        type=float,
        default=5e-4,
        help=(
            "Skip the harmonic pilot unless this fraction of raw spatial edges "
            "has magnitude >= tau (default: 0.0005 = 0.05%%; use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--min-cliff-component-pixels",
        type=int,
        default=30,
        help=(
            "Skip the harmonic pilot unless threshold-crossing edge endpoints "
            "contain an 8-connected component of at least this size "
            "(default: 30; use 0 to disable)."
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.min_cliff_edge_fraction <= 1:
        parser.error("--min-cliff-edge-fraction must lie in [0, 1]")
    if args.min_cliff_component_pixels < 0:
        parser.error("--min-cliff-component-pixels must be non-negative")
    if args.random_frames is not None and args.random_frames <= 0:
        parser.error("--random-frames must be positive")
    if args.q <= 0:
        parser.error("--q must be positive")
    if args.coherence_window <= 0 or args.height_window <= 0:
        parser.error("quality-map windows must be positive")
    if not 0 <= args.phase_coherence_threshold <= 1:
        parser.error("--phase-coherence-threshold must lie in [0, 1]")
    if args.height_seed_threshold < 0:
        parser.error("--height-seed-threshold must be non-negative")
    if args.seed_dilation_pixels < 0:
        parser.error("--seed-dilation-pixels must be non-negative")
    if args.boundary_gradient_limit_factor <= 0:
        parser.error("--boundary-gradient-limit-factor must be positive")
    if args.max_boundary_growth_iterations < 0:
        parser.error("--max-boundary-growth-iterations must be non-negative")

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(args.raw_input, "r") as h5:
        x = np.asarray(h5["meta/x"], dtype=float)
        y = np.asarray(h5["meta/y"], dtype=float)
        available_frames = np.array(
            sorted(int(key) for key in h5["main"].keys() if str(key).isdigit()),
            dtype=int,
        )
        if args.random_frames is not None:
            if args.random_frames > available_frames.size:
                parser.error(
                    f"--random-frames={args.random_frames} exceeds "
                    f"{available_frames.size} available frames"
                )
            rng = np.random.default_rng(args.random_seed)
            selected_frames = np.sort(
                rng.choice(available_frames, size=args.random_frames, replace=False)
            ).tolist()
        elif args.frames is not None:
            selected_frames = args.frames
        else:
            selected_frames = [1740, 1840]

        missing = sorted(set(selected_frames) - set(available_frames.tolist()))
        if missing:
            parser.error(f"frames not found in main group: {missing}")
        print(f"Selected frames: {selected_frames}")

        for frame_number in selected_frames:
            frame = np.asarray(h5[f"main/{_frame_key(frame_number)}"], dtype=float)
            record = analyze_frame(
                frame,
                frame_number,
                x,
                y,
                args.output,
                args.q,
                args.tau,
                args.residue_window,
                args.residue_density_threshold,
                args.min_mask_component_pixels,
                args.weak_edge_weight,
                args.trusted_data_weight,
                args.pilot,
                args.min_cliff_edge_fraction,
                args.min_cliff_component_pixels,
                args.residue_candidate_threshold,
                args.coherence_window,
                args.phase_coherence_threshold,
                args.height_window,
                args.height_seed_threshold,
                args.seed_dilation_pixels,
                args.boundary_gradient_limit_factor,
                args.max_boundary_growth_iterations,
            )
            records.append(record)
            if args.verbose:
                print(json.dumps(record, indent=2))
            else:
                flags = ",".join(record["review_flags"]) or "none"
                print(
                    f"frame={frame_number:05d} "
                    f"gate={'pass' if record['cliff_gate_passed'] else 'skip'} "
                    f"mask={100*record['pilot_mask_fraction']:.1f}% "
                    f"modified={100*record['pilot_modified_fraction']:.1f}% "
                    f"flags={flags}"
                )

    review_flag_counts: dict[str, int] = {}
    for record in records:
        for flag in record["review_flags"]:
            review_flag_counts[flag] = review_flag_counts.get(flag, 0) + 1
    summary = {
        "diagnostic_only": True,
        "raw_input": str(args.raw_input.resolve()),
        "parameters": {
            "q_um": args.q,
            "tau_um": args.tau,
            "residue_window": args.residue_window,
            "residue_density_threshold": args.residue_density_threshold,
            "min_mask_component_pixels": args.min_mask_component_pixels,
            "weak_edge_weight": args.weak_edge_weight,
            "trusted_data_weight": args.trusted_data_weight,
            "pilot": args.pilot,
            "residue_candidate_threshold": args.residue_candidate_threshold,
            "coherence_window": args.coherence_window,
            "phase_coherence_threshold": args.phase_coherence_threshold,
            "height_window": args.height_window,
            "height_seed_threshold_q": args.height_seed_threshold,
            "seed_dilation_pixels": args.seed_dilation_pixels,
            "boundary_gradient_limit_factor": args.boundary_gradient_limit_factor,
            "max_boundary_growth_iterations": args.max_boundary_growth_iterations,
            "min_cliff_edge_fraction": args.min_cliff_edge_fraction,
            "min_cliff_component_pixels": args.min_cliff_component_pixels,
        },
        "selection": {
            "random_frames": args.random_frames,
            "random_seed": args.random_seed if args.random_frames is not None else None,
            "selected_frames": selected_frames,
        },
        "review_summary": {
            "requires_review_count": int(
                sum(bool(record["requires_review"]) for record in records)
            ),
            "flag_counts": review_flag_counts,
        },
        "frames": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    review_fields = [
        "frame",
        "requires_review",
        "review_flags",
        "cliff_gate_passed",
        "raw_cliff_edge_fraction",
        "largest_cliff_component_pixels",
        "residue_mask_fraction",
        "height_seed_fraction",
        "phase_candidate_fraction",
        "initial_pilot_mask_fraction",
        "pilot_mask_fraction",
        "boundary_growth_added_pixels",
        "boundary_growth_converged",
        "unresolved_boundary_violation_pixels",
        "current_modified_fraction",
        "pilot_modified_fraction",
        "current_trusted_modified_fraction",
        "pilot_trusted_modified_fraction",
        "raw_max_gradient_um",
        "current_max_gradient_um",
        "current_max_gradient_touching_mask_um",
        "pilot_max_gradient_um",
        "pilot_max_gradient_touching_mask_um",
        "figure",
    ]
    with (args.output / "review_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in review_fields}
            row["review_flags"] = ";".join(record["review_flags"])
            writer.writerow(row)


if __name__ == "__main__":
    main()
