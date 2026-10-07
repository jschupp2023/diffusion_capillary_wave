"""Diagnose phase-unwrapping corrections in a local raw-height window.

The input is the original calibrated keyed-frame HDF5 recording.  Heights are
rewrapped with a user-supplied period ``q`` and unwrapped both frame by frame
and as one ``(time, y, x)`` volume.  No filtering, smoothing, interpolation, or
POD reconstruction is performed, and the source HDF5 file is opened read-only.

Example
-------
python data_analysis/pod/phase_unwrap_diagnostic.py \
    --raw-input /home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5 \
    --timestep 23000 --neighbors 3 --q 2.0

``q`` has no default: it must be supplied in the same units as the stored
height data.  ``--neighbors 3`` loads the target plus up to three frames on
each side, clipped at the recording boundaries.
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
from matplotlib.cm import ScalarMappable
import numpy as np

try:
    from skimage.restoration import unwrap_phase
except ImportError as error:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "phase_unwrap_diagnostic requires scikit-image>=0.25; install requirements.txt"
    ) from error

if __package__:
    from .pod_2d import InputInfo, _frame_key, inspect_input
else:
    from pod_2d import InputInfo, _frame_key, inspect_input


OUTPUT_FILENAMES = (
    "target_surfaces.png",
    "target_line_profiles.png",
    "target_correction_cycles.png",
    "window_diagnostics.png",
    "corrections.npz",
    "summary.json",
)


@dataclass(frozen=True)
class RawWindow:
    info: InputInfo
    fields: np.ndarray
    invalid: np.ndarray
    target_local_index: int


@dataclass(frozen=True)
class CorrectionResult:
    independent_2d: np.ndarray
    joint_3d: np.ndarray
    independent_alignment_cycles: np.ndarray
    joint_alignment_cycles: int


def load_raw_window(raw_input: Path, timestep: int, neighbors: int) -> RawWindow:
    """Load a bounded window with the repository's keyed-frame input loader."""
    if timestep < 0:
        raise ValueError("timestep must be nonnegative.")
    if neighbors < 0:
        raise ValueError("neighbors must be nonnegative.")

    source = Path(raw_input).expanduser().resolve()
    # A one-frame inspection establishes the recording length without loading data.
    target_info = inspect_input(source, timestep, timestep + 1)
    with h5py.File(source, "r") as handle:
        n_total = len(handle["meta/t"])
    start = max(0, timestep - neighbors)
    stop = min(n_total, timestep + neighbors + 1)
    info = inspect_input(source, start, stop)

    fields = np.empty((info.n_frames, *info.frame_shape), dtype=np.float64)
    with h5py.File(source, "r") as handle:
        frames = handle["main"]
        for index, frame_number in enumerate(info.frame_numbers):
            key = _frame_key(frame_number)
            field = np.asarray(frames[key], dtype=np.float64)
            if field.shape != info.frame_shape:
                raise ValueError(
                    f"Frame /main/{key} has shape {field.shape}; expected {info.frame_shape}."
                )
            fields[index] = field

    invalid = ~np.isfinite(fields)
    if np.any(np.all(invalid, axis=(1, 2))):
        bad = info.source_positions[np.all(invalid, axis=(1, 2))].tolist()
        raise ValueError(f"Entirely invalid frame(s) in local window: {bad}.")
    target_local_index = timestep - start
    assert int(target_info.source_positions[0]) == timestep
    return RawWindow(info, fields, invalid, target_local_index)


def rewrap_heights(fields: np.ndarray, invalid: np.ndarray, q: float) -> np.ma.MaskedArray:
    """Map heights to wrapped phase in ``[-pi, pi]`` without changing samples."""
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be a finite positive height period.")
    if fields.shape != invalid.shape:
        raise ValueError("fields and invalid mask must have the same shape.")
    safe = np.where(invalid, 0.0, fields)
    # Keep this expression explicit: q is the height represented by one 2*pi cycle.
    phase = np.angle(np.exp(2j * np.pi * safe / q))
    return np.ma.array(phase, mask=invalid, copy=False)


def _robust_integer_offset(
    reference: np.ndarray,
    candidate: np.ndarray,
    q: float,
    valid: np.ndarray,
    *,
    context: str,
) -> int:
    usable = valid & np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(usable):
        raise ValueError(f"No common valid pixels for {context} integer-cycle alignment.")
    cycle_difference = (reference[usable] - candidate[usable]) / q
    return int(np.rint(np.median(cycle_difference)))


def unwrap_independent_frames(
    phase: np.ma.MaskedArray,
    original: np.ndarray,
    invalid: np.ndarray,
    q: float,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap in 2-D, then temporally align whole-frame integer offsets.

    The target is first anchored by the robust median integer-cycle difference
    from the original target.  Other frames are aligned outward to an already
    aligned neighbor.  This assumes the median physical frame-to-frame height
    change over common valid pixels has magnitude below q/2; it does not set
    each frame's spatial mean to zero.
    """
    n_frames = len(original)
    candidates = np.empty_like(original, dtype=np.float64)
    for index in range(n_frames):
        unwrapped = unwrap_phase(
            phase[index], wrap_around=(False, False), rng=0
        )
        candidates[index] = np.ma.filled(unwrapped, np.nan) * q / (2 * np.pi)

    corrected = candidates.copy()
    offsets = np.zeros(n_frames, dtype=np.int64)
    target_valid = ~invalid[target_index]
    offsets[target_index] = _robust_integer_offset(
        original[target_index], candidates[target_index], q, target_valid,
        context="target-frame",
    )
    corrected[target_index] += offsets[target_index] * q

    for index in range(target_index - 1, -1, -1):
        common = ~(invalid[index] | invalid[index + 1])
        offsets[index] = _robust_integer_offset(
            corrected[index + 1], candidates[index], q, common,
            context=f"timesteps {index} and {index + 1}",
        )
        corrected[index] += offsets[index] * q
    for index in range(target_index + 1, n_frames):
        common = ~(invalid[index] | invalid[index - 1])
        offsets[index] = _robust_integer_offset(
            corrected[index - 1], candidates[index], q, common,
            context=f"timesteps {index - 1} and {index}",
        )
        corrected[index] += offsets[index] * q
    corrected[invalid] = np.nan
    return corrected, offsets


def unwrap_joint_volume(
    phase: np.ma.MaskedArray,
    original: np.ndarray,
    invalid: np.ndarray,
    q: float,
    target_index: int,
) -> tuple[np.ndarray, int]:
    """Unwrap the time-space volume with all periodic boundary links disabled."""
    unwrapped = unwrap_phase(
        phase, wrap_around=(False, False, False), rng=0
    )
    corrected = np.ma.filled(unwrapped, np.nan) * q / (2 * np.pi)
    offset = _robust_integer_offset(
        original[target_index], corrected[target_index], q,
        ~invalid[target_index], context="joint-volume target-frame",
    )
    corrected += offset * q
    corrected[invalid] = np.nan
    return corrected, offset


def correct_window(window: RawWindow, q: float) -> CorrectionResult:
    phase = rewrap_heights(window.fields, window.invalid, q)
    independent, independent_offsets = unwrap_independent_frames(
        phase, window.fields, window.invalid, q, window.target_local_index
    )
    joint, joint_offset = unwrap_joint_volume(
        phase, window.fields, window.invalid, q, window.target_local_index
    )
    return CorrectionResult(independent, joint, independent_offsets, joint_offset)


def integer_consistency(
    corrected: np.ndarray,
    original: np.ndarray,
    invalid: np.ndarray,
    q: float,
    tolerance: float = 1e-6,
) -> dict[str, float | int]:
    cycles = (corrected[~invalid] - original[~invalid]) / q
    error = np.abs(cycles - np.rint(cycles))
    return {
        "valid_sample_count": int(len(error)),
        "max_distance_to_integer_cycles": float(np.max(error)),
        "p99_distance_to_integer_cycles": float(np.percentile(error, 99)),
        "fraction_within_tolerance": float(np.mean(error <= tolerance)),
        "tolerance_cycles": float(tolerance),
    }


def _percentiles(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


def spatial_gradient_statistics(
    fields: np.ndarray, invalid: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame p50/p95 absolute adjacent-pixel physical gradients."""
    dx = np.diff(x)
    dy = np.diff(y)
    if np.any(~np.isfinite(dx)) or np.any(dx == 0) or np.any(~np.isfinite(dy)) or np.any(dy == 0):
        raise ValueError("x and y grids must have finite, distinct adjacent coordinates.")
    p50 = np.empty(len(fields))
    p95 = np.empty(len(fields))
    for index, field in enumerate(fields):
        valid_x = ~(invalid[index, :, 1:] | invalid[index, :, :-1])
        valid_y = ~(invalid[index, 1:, :] | invalid[index, :-1, :])
        gx = np.abs(np.diff(field, axis=1) / dx[None, :])[valid_x]
        gy = np.abs(np.diff(field, axis=0) / dy[:, None])[valid_y]
        p50[index], p95[index] = _percentiles(np.concatenate((gx, gy)))
    return p50, p95


def temporal_increment_statistics(
    fields: np.ndarray, invalid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return p50/p95 absolute height increments for adjacent window frames."""
    p50 = np.empty(max(0, len(fields) - 1))
    p95 = np.empty_like(p50)
    for index in range(len(fields) - 1):
        common = ~(invalid[index + 1] | invalid[index])
        values = np.abs(fields[index + 1] - fields[index])[common]
        p50[index], p95[index] = _percentiles(values)
    return p50, p95


def _finite_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if values.size == 0:
        raise ValueError("Cannot plot arrays without finite values.")
    lower, upper = float(np.min(values)), float(np.max(values))
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.01)
        lower, upper = lower - padding, upper + padding
    return lower, upper


def _save_surfaces(
    path: Path, window: RawWindow, result: CorrectionResult, q: float, dpi: int
) -> None:
    index = window.target_local_index
    fields = (window.fields[index], result.independent_2d[index], result.joint_3d[index])
    zmin, zmax = _finite_limits(*fields)
    norm = Normalize(zmin, zmax)
    x_grid, y_grid = np.meshgrid(window.info.x, window.info.y)
    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    titles = (
        "Original",
        "Candidate: independent 2-D",
        "Candidate: joint 3-D",
    )
    for panel, (field, title) in enumerate(zip(fields, titles), start=1):
        axes = fig.add_subplot(1, 3, panel, projection="3d")
        axes.plot_surface(
            x_grid, y_grid, field, cmap="viridis", norm=norm,
            rcount=field.shape[0], ccount=field.shape[1], linewidth=0,
            antialiased=False,
        )
        axes.set(xlabel=window.info.units["x"] or "x", ylabel=window.info.units["y"] or "y")
        axes.set_zlabel(window.info.units["z"] or "height")
        axes.set_zlim(zmin, zmax)
        axes.set_title(title)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"), ax=fig.axes,
        label=f"height [{window.info.units['z'] or 'source units'}]", shrink=0.7,
    )
    fig.suptitle(
        f"Timestep {int(window.info.source_positions[index])}; q={q:g} "
        f"{window.info.units['z'] or 'source units'} | identical height/color limits"
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _save_profiles(path: Path, window: RawWindow, result: CorrectionResult, dpi: int) -> None:
    index = window.target_local_index
    row, column = window.fields.shape[1] // 2, window.fields.shape[2] // 2
    fields = (window.fields[index], result.independent_2d[index], result.joint_3d[index])
    labels = ("original", "candidate: independent 2-D", "candidate: joint 3-D")
    zmin, zmax = _finite_limits(*fields)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, constrained_layout=True)
    for field, label in zip(fields, labels):
        axes[0].plot(window.info.x, field[row, :], label=label)
        axes[1].plot(window.info.y, field[:, column], label=label)
    axes[0].set_title(f"Horizontal profile: row {row}")
    axes[1].set_title(f"Vertical profile: column {column}")
    axes[0].set_xlabel(window.info.units["x"] or "x")
    axes[1].set_xlabel(window.info.units["y"] or "y")
    axes[0].set_ylabel(f"height [{window.info.units['z'] or 'source units'}]")
    for axes_item in axes:
        axes_item.set_ylim(zmin, zmax)
        axes_item.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _save_correction_maps(
    path: Path, window: RawWindow, result: CorrectionResult, q: float, dpi: int
) -> None:
    index = window.target_local_index
    maps = (
        (result.independent_2d[index] - window.fields[index]) / q,
        (result.joint_3d[index] - window.fields[index]) / q,
    )
    finite = np.concatenate([value[np.isfinite(value)] for value in maps])
    bound = max(1.0, float(np.max(np.abs(finite))))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    extent = (window.info.x[0], window.info.x[-1], window.info.y[0], window.info.y[-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    titles = ("Independent 2-D", "Joint (time, y, x)")
    image = None
    for axes_item, correction, title in zip(axes, maps, titles):
        image = axes_item.imshow(
            correction, origin="lower", extent=extent, aspect="auto",
            cmap="coolwarm", norm=norm, interpolation="none",
        )
        axes_item.set_title(title)
        axes_item.set_xlabel(window.info.units["x"] or "x")
        axes_item.set_ylabel(window.info.units["y"] or "y")
    assert image is not None
    fig.colorbar(image, ax=axes, label="(corrected - original) / q")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def compute_window_diagnostics(
    window: RawWindow, result: CorrectionResult
) -> dict[str, dict[str, np.ndarray]]:
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for label, fields in (
        ("original", window.fields),
        ("independent_2d", result.independent_2d),
        ("joint_3d", result.joint_3d),
    ):
        gradient_p50, gradient_p95 = spatial_gradient_statistics(
            fields, window.invalid, window.info.x, window.info.y
        )
        increment_p50, increment_p95 = temporal_increment_statistics(fields, window.invalid)
        diagnostics[label] = {
            "spatial_gradient_abs_p50": gradient_p50,
            "spatial_gradient_abs_p95": gradient_p95,
            "temporal_increment_abs_p50": increment_p50,
            "temporal_increment_abs_p95": increment_p95,
        }
    return diagnostics


def _save_window_diagnostics(
    path: Path,
    window: RawWindow,
    diagnostics: dict[str, dict[str, np.ndarray]],
    dpi: int,
) -> None:
    labels = {
        "original": "original",
        "independent_2d": "independent 2-D",
        "joint_3d": "joint 3-D",
    }
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for key, label in labels.items():
        axes[0].plot(
            window.info.source_positions,
            diagnostics[key]["spatial_gradient_abs_p95"], marker="o", label=label,
        )
        if len(window.info.time) > 1:
            axes[1].plot(
                window.info.source_positions[1:],
                diagnostics[key]["temporal_increment_abs_p95"], marker="o", label=label,
            )
    axes[0].set_title("Spatial diagnostic: p95 |adjacent height gradient|")
    axes[0].set_ylabel(
        f"height/{window.info.units['x'] or 'coordinate unit'}"
    )
    axes[1].set_title("Temporal diagnostic: p95 |adjacent-frame height increment|")
    axes[1].set_ylabel(window.info.units["z"] or "height")
    axes[1].set_xlabel("source timestep position")
    for axes_item in axes:
        axes_item.grid(alpha=0.25)
        axes_item.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _json_array(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _adjacent_height_increments(field: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    valid_x = ~(invalid[:, 1:] | invalid[:, :-1])
    valid_y = ~(invalid[1:, :] | invalid[:-1, :])
    horizontal = np.abs(np.diff(field, axis=1))[valid_x]
    vertical = np.abs(np.diff(field, axis=0))[valid_y]
    return np.concatenate((horizontal, vertical))


def target_candidate_assessment(
    window: RawWindow, result: CorrectionResult, q: float
) -> dict[str, Any]:
    """Quantify candidate changes without claiming physical correctness."""
    index = window.target_local_index
    original = window.fields[index]
    invalid = window.invalid[index]
    original_edges = _adjacent_height_increments(original, invalid)
    original_tv = float(np.sum(original_edges))
    assessment: dict[str, Any] = {
        "original": {
            "adjacent_height_increment_p95": float(np.percentile(original_edges, 95)),
            "adjacent_height_increment_p99": float(np.percentile(original_edges, 99)),
            "fraction_adjacent_increments_above_q_over_2": float(
                np.mean(original_edges > q / 2)
            ),
            "spatial_total_variation": original_tv,
        }
    }
    valid = ~invalid
    for label, candidate in (
        ("independent_2d", result.independent_2d[index]),
        ("joint_3d", result.joint_3d[index]),
    ):
        cycles = (candidate[valid] - original[valid]) / q
        edges = _adjacent_height_increments(candidate, invalid)
        tv = float(np.sum(edges))
        assessment[label] = {
            "changed_valid_pixel_fraction": float(np.mean(np.abs(cycles) > 0.5)),
            "correction_cycle_min": int(np.min(np.rint(cycles))),
            "correction_cycle_max": int(np.max(np.rint(cycles))),
            "adjacent_height_increment_p95": float(np.percentile(edges, 95)),
            "adjacent_height_increment_p99": float(np.percentile(edges, 99)),
            "fraction_adjacent_increments_above_q_over_2": float(
                np.mean(edges > q / 2)
            ),
            "spatial_total_variation": tv,
            "spatial_total_variation_ratio_to_original": (
                tv / original_tv if original_tv else None
            ),
        }
    assessment["automatic_action"] = (
        "preserve_original; these are branch-choice candidates for diagnosis only. "
        "A smoother candidate or an integer-multiple correction is not evidence that "
        "the selected phase branch is physical."
    )
    return assessment


def _default_output_dir(source: Path, timestep: int, neighbors: int, q: float) -> Path:
    q_label = f"{q:.9g}".replace(".", "p")
    return (
        Path("runs/phase_unwrap")
        / source.stem
        / f"timestep_{timestep}_neighbors_{neighbors}_q_{q_label}"
    )


def run_diagnostic(
    raw_input: Path,
    timestep: int,
    neighbors: int,
    q: float,
    *,
    output_dir: Path | None = None,
    dpi: int = 180,
    overwrite: bool = False,
) -> Path:
    """Run both unwrapping methods and save separate corrections and diagnostics."""
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be a finite positive height period.")
    if dpi < 1:
        raise ValueError("dpi must be positive.")
    source = Path(raw_input).expanduser().resolve()
    output = Path(
        output_dir or _default_output_dir(source, timestep, neighbors, q)
    ).expanduser().resolve()
    existing = [output / name for name in OUTPUT_FILENAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Diagnostic output already exists ({existing[0]}); pass --overwrite to replace it."
        )

    window = load_raw_window(source, timestep, neighbors)
    result = correct_window(window, q)
    diagnostics = compute_window_diagnostics(window, result)
    consistency = {
        "independent_2d": integer_consistency(
            result.independent_2d, window.fields, window.invalid, q
        ),
        "joint_3d": integer_consistency(
            result.joint_3d, window.fields, window.invalid, q
        ),
    }
    candidate_assessment = target_candidate_assessment(window, result, q)

    output.mkdir(parents=True, exist_ok=True)
    _save_surfaces(output / "target_surfaces.png", window, result, q, dpi)
    _save_profiles(output / "target_line_profiles.png", window, result, dpi)
    _save_correction_maps(
        output / "target_correction_cycles.png", window, result, q, dpi
    )
    _save_window_diagnostics(
        output / "window_diagnostics.png", window, diagnostics, dpi
    )
    np.savez_compressed(
        output / "corrections.npz",
        source_positions=window.info.source_positions,
        frame_numbers=window.info.frame_numbers,
        time=window.info.time,
        x=window.info.x,
        y=window.info.y,
        invalid_mask=window.invalid,
        corrected_independent_2d=result.independent_2d,
        corrected_joint_3d=result.joint_3d,
        correction_cycles_independent_2d=(result.independent_2d - window.fields) / q,
        correction_cycles_joint_3d=(result.joint_3d - window.fields) / q,
        independent_alignment_cycles=result.independent_alignment_cycles,
        joint_alignment_cycles=np.int64(result.joint_alignment_cycles),
        q=np.float64(q),
    )

    summary: dict[str, Any] = {
        "source": str(source),
        "source_opened_read_only": True,
        "target_timestep_position": int(timestep),
        "target_local_index": int(window.target_local_index),
        "window_start_inclusive": int(window.info.source_positions[0]),
        "window_stop_exclusive": int(window.info.source_positions[-1] + 1),
        "source_positions": window.info.source_positions.tolist(),
        "hdf5_frame_numbers": window.info.frame_numbers.tolist(),
        "q": float(q),
        "q_units": window.info.units["z"] or "same units as stored heights",
        "invalid_pixel_policy": "nonfinite samples are masked; no filling or interpolation",
        "periodic_boundary_connections": False,
        "processing": "rewrap only; no filtering, smoothing, interpolation, or mean removal",
        "global_cycle_alignment": {
            "independent_2d_alignment_cycles": result.independent_alignment_cycles.tolist(),
            "joint_3d_single_alignment_cycles": int(result.joint_alignment_cycles),
            "target_anchor": (
                "nearest integer to the median (original - unwrapped) / q over valid "
                "target pixels; assumes a majority of target pixels have the correct cycle"
            ),
            "independent_temporal_assumption": (
                "successive frames are aligned by robust median agreement over common valid "
                "pixels; assumes median physical frame-to-frame height change is below q/2"
            ),
            "spatial_mean_removed_per_frame": False,
        },
        "integer_cycle_consistency": consistency,
        "consistency_interpretation": (
            "Near-integer corrections are required by the rewrap/unwrap construction but "
            "are not proof that the selected branches are physically correct."
        ),
        "target_candidate_assessment": candidate_assessment,
        "automatic_action": candidate_assessment["automatic_action"],
        "ambiguity_warning": (
            "Noise or spatial/temporal undersampling can make phase unwrapping ambiguous; "
            "inspect the correction maps, profiles, and gradient/increment diagnostics."
        ),
        "diagnostic_definitions": {
            "spatial_gradient": (
                "p50 and p95 absolute forward height differences divided by x or y "
                "coordinate spacing, pooled over valid adjacent pixel pairs"
            ),
            "temporal_increment": (
                "p50 and p95 absolute height differences between adjacent loaded frames "
                "over common valid pixels; not divided by elapsed time"
            ),
        },
        "window_diagnostics": {
            label: {name: _json_array(values) for name, values in values_by_name.items()}
            for label, values_by_name in diagnostics.items()
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    for method, values in consistency.items():
        print(
            f"{method}: max distance to integer correction = "
            f"{values['max_distance_to_integer_cycles']:.3g} cycles; "
            f"fraction within {values['tolerance_cycles']:.1g} = "
            f"{values['fraction_within_tolerance']:.6f}"
        )
    print(
        "Consistency is necessary, not proof of physical correctness; noise or "
        "undersampling can make unwrapping ambiguous. The saved fields are candidates, "
        "not automatically accepted corrections; preserve the original unless the "
        "branch choice is externally validated.\n"
        f"Saved separate corrections and diagnostics to {output}"
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, required=True, help="Raw keyed-frame HDF5 recording.")
    parser.add_argument(
        "--timestep", type=int, required=True,
        help="Target zero-based position in /meta/t (not necessarily the HDF5 frame number).",
    )
    parser.add_argument(
        "--neighbors", type=int, required=True,
        help="Frames to load on each side of the target; clipped at recording boundaries.",
    )
    parser.add_argument(
        "--q", type=float, required=True,
        help="Height per 2*pi phase cycle, in the same units as stored heights (no default).",
    )
    parser.add_argument("--output-dir", type=Path, help="Separate output directory for corrections and plots.")
    parser.add_argument("--dpi", type=int, default=180, help="Plot resolution (default: 180).")
    parser.add_argument("--overwrite", action="store_true", help="Replace this tool's existing output files.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_diagnostic(
        args.raw_input, args.timestep, args.neighbors, args.q,
        output_dir=args.output_dir, dpi=args.dpi, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
