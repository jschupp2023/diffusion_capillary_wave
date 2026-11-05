"""Export a production cleaned HDF5 dataset using gradient-domain restoration,
advanced kinematic temporal alignment, and full-field 50 Hz high-pass filtering.

Pipeline stages:
1. Spatial integer-cycle unwrapping (tau=4.0 um, q=1.9969696 um, integer mode).
   Eliminates 2D phase-cut cliffs and tears without altering continuous gradients.
2. Advanced kinematic temporal cycle alignment with local-motion veto.
   Eliminates global optical reference slips while preserving real wave excursions
   and vetting large jumps against continuing local motion.
3. 50 Hz zero-phase 4th-order Butterworth high-pass filter applied along the time axis
   across the entire 200x200 spatial field.
   Eliminates slow thermal drift and low-frequency container baseline wander.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

# Prevent BLAS thread oversubscription in worker processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
from scipy.ndimage import label
from scipy.signal import butter, sosfiltfilt

from data_analysis.diagnose_weighted_poisson_mask import (
    conservative_hybrid_mask,
    harmonic_reference_with_boundary_growth,
    max_gradient,
    wrapped_phase_quality,
)
from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.pod.phase_unwrap_gradient_poisson import restore_frame_poisson
from data_analysis.temporal_alignment import (
    DEFAULT_Q,
    DEFAULT_TAU,
    extract_spatial_channels,
    select_temporal_kinematic,
    TemporalSelectionResult,
)

DEFAULT_INPUT = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
DEFAULT_OUTPUT = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5")

HYBRID_PARAMETERS: dict[str, float | int] = {
    "residue_window": 15,
    "residue_density_threshold": 0.20,
    "min_mask_component_pixels": 100,
    "residue_candidate_threshold": 0.10,
    "coherence_window": 7,
    "phase_coherence_threshold": 0.70,
    "height_window": 7,
    "height_seed_threshold_q": 0.35,
    "seed_dilation_pixels": 3,
    "min_cliff_edge_fraction": 5e-4,
    "min_cliff_component_pixels": 30,
    "boundary_gradient_limit_factor": 1.05,
    "max_boundary_growth_iterations": 30,
}

SPATIAL_QC_FIELDS = (
    "modified_fraction",
    "mask_fraction",
    "initial_mask_fraction",
    "residue_mask_fraction",
    "cliff_gate_passed",
    "raw_cliff_edge_fraction",
    "largest_cliff_component_pixels",
    "boundary_growth_added_pixels",
    "boundary_growth_converged",
    "unresolved_boundary_violation_pixels",
    "raw_max_gradient_um",
    "restored_max_gradient_um",
    "review_flag_bits",
)


def hybrid_cliff_gate_diagnostics(
    frame: np.ndarray,
    tau: float,
) -> dict[str, float | int | bool]:
    """Evaluate the cheap spatial gate used by the hybrid cleaner.

    This helper deliberately contains the authoritative gate implementation so
    that read-only audits can skip the expensive phase/mask calculation when a
    frame cannot be modified by the production cleaner.
    """
    p = HYBRID_PARAMETERS
    raw_dx = np.abs(np.diff(frame, axis=1))
    raw_dy = np.abs(np.diff(frame, axis=0))
    cliff_x = raw_dx >= tau
    cliff_y = raw_dy >= tau
    cliff_edge_count = int(np.count_nonzero(cliff_x) + np.count_nonzero(cliff_y))
    cliff_edge_fraction = cliff_edge_count / float(raw_dx.size + raw_dy.size)

    largest_component = 0
    # Component labelling is unnecessary unless the edge-fraction part of the
    # gate already passes. This is the common fast path for low forcing.
    if cliff_edge_fraction >= float(p["min_cliff_edge_fraction"]):
        cliff_seed = np.zeros(frame.shape, dtype=bool)
        cliff_seed[:, :-1] |= cliff_x
        cliff_seed[:, 1:] |= cliff_x
        cliff_seed[:-1, :] |= cliff_y
        cliff_seed[1:, :] |= cliff_y
        components, _ = label(cliff_seed, structure=np.ones((3, 3), dtype=np.int8))
        component_sizes = np.bincount(components.ravel())[1:]
        largest_component = int(component_sizes.max()) if component_sizes.size else 0

    gate_passed = bool(
        cliff_edge_fraction >= float(p["min_cliff_edge_fraction"])
        and largest_component >= int(p["min_cliff_component_pixels"])
    )
    raw_max = float(max(raw_dx.max(initial=0.0), raw_dy.max(initial=0.0)))
    return {
        "gate_passed": gate_passed,
        "cliff_edge_fraction": float(cliff_edge_fraction),
        "largest_component_pixels": largest_component,
        "raw_max_gradient_um": raw_max,
    }


def _restore_frame_hybrid(
    frame: np.ndarray,
    q: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the validated conservative hybrid spatial restoration to one frame."""
    p = HYBRID_PARAMETERS
    phase, _, residue_pixels, residue_mask = wrapped_phase_quality(
        frame,
        q,
        int(p["residue_window"]),
        float(p["residue_density_threshold"]),
        int(p["min_mask_component_pixels"]),
    )
    initial_mask, diagnostics = conservative_hybrid_mask(
        frame,
        phase,
        residue_pixels,
        q,
        int(p["residue_window"]),
        float(p["residue_candidate_threshold"]),
        int(p["coherence_window"]),
        float(p["phase_coherence_threshold"]),
        int(p["height_window"]),
        float(p["height_seed_threshold_q"]),
        int(p["seed_dilation_pixels"]),
        int(p["min_mask_component_pixels"]),
    )

    gate = hybrid_cliff_gate_diagnostics(frame, tau)
    cliff_edge_fraction = float(gate["cliff_edge_fraction"])
    largest_component = int(gate["largest_component_pixels"])
    gate_passed = bool(gate["gate_passed"])

    mask = initial_mask.copy()
    if gate_passed and np.any(mask) and not np.all(mask):
        _, cycles, mask, solver = harmonic_reference_with_boundary_growth(
            frame,
            mask,
            residue_mask | np.asarray(diagnostics["seed_support"], dtype=bool),
            q,
            float(p["boundary_gradient_limit_factor"]) * tau,
            int(p["max_boundary_growth_iterations"]),
        )
    elif np.all(mask):
        cycles = np.zeros(frame.shape, dtype=np.int32)
        solver = {
            "boundary_growth_added_pixels": 0,
            "boundary_growth_converged": False,
            "unresolved_boundary_violation_pixels": int(mask.size),
        }
    else:
        cycles = np.zeros(frame.shape, dtype=np.int32)
        solver = {
            "boundary_growth_added_pixels": 0,
            "boundary_growth_converged": True,
            "unresolved_boundary_violation_pixels": 0,
        }

    restored = (frame - q * cycles).astype(frame.dtype, copy=False)
    modified_fraction = float(np.mean(cycles != 0))
    growth_added = int(solver["boundary_growth_added_pixels"])
    growth_converged = bool(solver["boundary_growth_converged"])
    unresolved = int(solver["unresolved_boundary_violation_pixels"])

    # Compact bit field for unattended review: 1=growth, 2=unresolved,
    # 4=gate skipped despite a >2*tau raw gradient, 8=mask above 70 percent,
    # 16=cliff gate passed but no supported uncertainty component was found.
    review_bits = 0
    if growth_added > 0:
        review_bits |= 1
    if not growth_converged:
        review_bits |= 2
    raw_max = float(gate["raw_max_gradient_um"])
    if not gate_passed and raw_max > 2 * tau:
        review_bits |= 4
    if np.mean(mask) > 0.70:
        review_bits |= 8
    if gate_passed and not np.any(mask):
        review_bits |= 16

    qc_values = {
        "modified_fraction": modified_fraction,
        "mask_fraction": float(np.mean(mask)),
        "initial_mask_fraction": float(np.mean(initial_mask)),
        "residue_mask_fraction": float(np.mean(residue_mask)),
        "cliff_gate_passed": float(gate_passed),
        "raw_cliff_edge_fraction": float(cliff_edge_fraction),
        "largest_cliff_component_pixels": float(largest_component),
        "boundary_growth_added_pixels": float(growth_added),
        "boundary_growth_converged": float(growth_converged),
        "unresolved_boundary_violation_pixels": float(unresolved),
        "raw_max_gradient_um": raw_max,
        "restored_max_gradient_um": max_gradient(restored),
        "review_flag_bits": float(review_bits),
    }
    return restored, np.asarray(
        [qc_values[field] for field in SPATIAL_QC_FIELDS], dtype=np.float64
    )


def _process_chunk_worker(
    input_path_str: str,
    start_idx: int,
    end_idx: int,
    q: float,
    tau: float,
    method: str = "integer",
) -> tuple[int, int, np.ndarray, int, np.ndarray, np.ndarray]:
    """Worker to restore a chunk of frames and compute spatial tracking channels."""
    n_frames = end_idx - start_idx
    with h5py.File(input_path_str, "r") as h5_in:
        main_grp = h5_in["main"]
        first_key = _frame_key(start_idx)
        first_shape = main_grp[first_key].shape
        cleaned = np.empty((n_frames, *first_shape), dtype=np.float32)
        channels = np.empty((n_frames, 6), dtype=np.float64)
        spatial_qc = np.full(
            (n_frames, len(SPATIAL_QC_FIELDS)), np.nan, dtype=np.float64
        )
        n_modified = 0

        for i in range(n_frames):
            frame_num = start_idx + i
            key = _frame_key(frame_num)
            raw = np.asarray(main_grp[key], dtype=np.float32)
            if method == "hybrid":
                restored, frame_qc = _restore_frame_hybrid(raw, q=q, tau=tau)
                spatial_qc[i] = frame_qc
                was_modified = bool(frame_qc[SPATIAL_QC_FIELDS.index("modified_fraction")] > 0)
            else:
                res = restore_frame_poisson(raw, q=q, tau=tau, method=method)
                restored = res.restored
                was_modified = res.was_modified
                spatial_qc[i, SPATIAL_QC_FIELDS.index("modified_fraction")] = (
                    res.pct_pixels_modified / 100.0
                )
                spatial_qc[i, SPATIAL_QC_FIELDS.index("raw_max_gradient_um")] = (
                    res.orig_max_dx
                )
                spatial_qc[i, SPATIAL_QC_FIELDS.index("restored_max_gradient_um")] = (
                    res.rest_max_dx
                )
            cleaned[i] = restored
            channels[i] = extract_spatial_channels(restored)
            if was_modified:
                n_modified += 1

    return start_idx, end_idx, cleaned, n_modified, channels, spatial_qc


def export_cleaned_dataset(
    raw_input: Path,
    output_file: Path,
    q: float = DEFAULT_Q,
    tau: float = DEFAULT_TAU,
    method: str = "integer",
    temporal_align: bool = True,
    apply_highpass_50hz: bool = True,
    highpass_cutoff_hz: float = 50.0,
    workers: int = 8,
    chunk_size: int = 500,
    scratch_dir: Path | None = None,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> Path:
    raw_input = raw_input.expanduser().resolve()
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not raw_input.is_file():
        raise FileNotFoundError(f"Input file not found: {raw_input}")
    if method not in {"integer", "poisson", "hybrid"}:
        raise ValueError("method must be one of: integer, poisson, hybrid")

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. Use --overwrite to replace it."
        )

    started = time.perf_counter()
    print(f"=== Starting Production Data Cleaning Pipeline ===", flush=True)
    print(f"Input file:  {raw_input}")
    print(f"Output file: {output_file}")
    print(
        f"Settings: q = {q:.6f} um | tau = {tau:.2f} um | method = {method} | "
        f"temporal_align = {temporal_align} | hp50hz = {apply_highpass_50hz} | workers = {workers}"
    )

    # 1. Read metadata from raw file
    with h5py.File(raw_input, "r") as h5_in:
        meta_in = h5_in["meta"]
        t_arr = np.asarray(meta_in["t"], dtype=np.float64)
        x_arr = np.asarray(meta_in["x"])
        y_arr = np.asarray(meta_in["y"])
        frames_arr = (
            np.asarray(meta_in["frames"])
            if "frames" in meta_in
            else np.arange(len(t_arr), dtype=np.int32)
        )
        meta_attrs = dict(meta_in.attrs)
        total_frames = len(t_arr)

    if max_frames is not None and max_frames < total_frames:
        total_frames = max_frames
        t_arr = t_arr[:total_frames]
        frames_arr = frames_arr[:total_frames]
        print(f"Subsampling: running pilot on first {total_frames:,} frames only.")

    dt = float(np.mean(np.diff(t_arr))) if total_frames > 1 else 1.0 / 115200.0
    fs = 1.0 / dt
    ny, nx = len(y_arr), len(x_arr)
    print(f"Total frames: {total_frames:,} | Grid: {ny}x{nx} | fs: {fs:.1f} Hz", flush=True)

    # Allocate scratch memory map for full recording to minimize RAM footprint
    if scratch_dir is None:
        scratch_dir = output_file.parent / ".scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    mmap_path = scratch_dir / f"mmap_spatial_{output_file.stem}_{os.getpid()}.dat"

    try:
        mmap_spatial = np.memmap(
            mmap_path, dtype=np.float32, mode="w+", shape=(total_frames, ny, nx)
        )
        spatial_channels = np.zeros((total_frames, 6), dtype=np.float64)
        spatial_qc = np.full(
            (total_frames, len(SPATIAL_QC_FIELDS)), np.nan, dtype=np.float64
        )

        # -------------------------------------------------------------
        # Stage 1: Spatial Poisson/Integer Restoration (Parallel Chunks)
        # -------------------------------------------------------------
        print("\n--- Stage 1: Spatial Integer-Cycle Restoration ---", flush=True)
        t_stage1 = time.perf_counter()
        chunk_ranges = [
            (start, min(start + chunk_size, total_frames))
            for start in range(0, total_frames, chunk_size)
        ]
        n_chunks = len(chunk_ranges)
        max_in_flight = workers + 2

        total_modified = 0
        frames_written = 0
        completed_chunks = 0
        progress_interval = max(1, n_chunks // 10)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_chunk = {}
            chunk_iter = iter(chunk_ranges)

            for _ in range(min(max_in_flight, n_chunks)):
                start, end = next(chunk_iter)
                fut = executor.submit(
                    _process_chunk_worker,
                    str(raw_input),
                    start,
                    end,
                    q,
                    tau,
                    method,
                )
                future_to_chunk[fut] = (start, end)

            while future_to_chunk:
                done, _ = wait(future_to_chunk.keys(), return_when=FIRST_COMPLETED)

                for fut in done:
                    (
                        start,
                        end,
                        chunk_data,
                        n_mod,
                        chunk_channels,
                        chunk_qc,
                    ) = fut.result()
                    del future_to_chunk[fut]
                    total_modified += n_mod

                    # Store directly into memory-mapped scratch buffer
                    mmap_spatial[start:end] = chunk_data
                    spatial_channels[start:end] = chunk_channels
                    spatial_qc[start:end] = chunk_qc

                    n_chunk_frames = end - start
                    frames_written += n_chunk_frames
                    completed_chunks += 1
                    del chunk_data, chunk_channels, chunk_qc

                    try:
                        next_start, next_end = next(chunk_iter)
                        next_fut = executor.submit(
                            _process_chunk_worker,
                            str(raw_input),
                            next_start,
                            next_end,
                            q,
                            tau,
                            method,
                        )
                        future_to_chunk[next_fut] = (next_start, next_end)
                    except StopIteration:
                        pass

                    if (
                        completed_chunks % progress_interval == 0
                        or completed_chunks == n_chunks
                    ):
                        elapsed = time.perf_counter() - t_stage1
                        fps = frames_written / elapsed if elapsed > 0 else 0
                        pct = (frames_written / total_frames) * 100.0
                        print(
                            f"  Spatial Chunk {completed_chunks}/{n_chunks}: {frames_written:,}/{total_frames:,} ({pct:.1f}%) | {fps:.1f} fps",
                            flush=True,
                        )
                        gc.collect()

        mmap_spatial.flush()
        t_stage1_elapsed = time.perf_counter() - t_stage1
        pct_modified = (total_modified / total_frames) * 100.0
        print(
            f"Stage 1 finished in {t_stage1_elapsed:.1f} s: {total_modified:,} frames modified ({pct_modified:.2f}%).",
            flush=True,
        )
        if method == "hybrid":
            qc_index = {name: i for i, name in enumerate(SPATIAL_QC_FIELDS)}
            review_bits = spatial_qc[:, qc_index["review_flag_bits"]].astype(np.uint8)
            print(
                "  Hybrid QC: "
                f"gate passed {int(np.count_nonzero(spatial_qc[:, qc_index['cliff_gate_passed']])):,} | "
                f"boundary growth {int(np.count_nonzero(review_bits & 1)):,} | "
                f"unresolved {int(np.count_nonzero(review_bits & 2)):,} | "
                f"skipped severe gradients {int(np.count_nonzero(review_bits & 4)):,} | "
                f"mask >70% {int(np.count_nonzero(review_bits & 8)):,} | "
                f"empty supported mask {int(np.count_nonzero(review_bits & 16)):,}",
                flush=True,
            )

        # -------------------------------------------------------------
        # Stage 2: Advanced Kinematic Temporal Alignment
        # -------------------------------------------------------------
        temporal_res = None
        if temporal_align and total_frames > 1:
            print("\n--- Stage 2: Advanced Kinematic Temporal Alignment ---", flush=True)
            t_stage2 = time.perf_counter()
            dh_mean = np.r_[0.0, np.diff(spatial_channels[:, 0])]
            quad_diffs = np.vstack(
                [np.zeros((1, 4)), np.diff(spatial_channels[:, 2:], axis=0)]
            )

            temporal_res = select_temporal_kinematic(dh_mean, quad_diffs, q=q)
            offsets = temporal_res.offsets

            print(
                f"  Temporal selector: {temporal_res.parameters['method']} "
                f"| Corrections: {temporal_res.n_corrections:,} "
                f"(Large: {temporal_res.n_large_corrections:,}, Small: {temporal_res.n_small_corrections:,}, Flagged: {temporal_res.n_flagged_corrections:,})"
            )
            print(f"  Max residual increment: {temporal_res.max_residual_increment_um:.2f} um")

            # Apply integer offset uniformly across spatial field in chunks to minimize memory
            if temporal_res.n_corrections > 0:
                print("  Applying global integer cycle offsets to field...", flush=True)
                for start, end in chunk_ranges:
                    sub_offsets = offsets[start:end, None, None]
                    if np.any(sub_offsets != 0):
                        mmap_spatial[start:end] -= (sub_offsets * q).astype(np.float32)
                mmap_spatial.flush()

            t_stage2_elapsed = time.perf_counter() - t_stage2
            print(f"Stage 2 finished in {t_stage2_elapsed:.2f} s.", flush=True)
        else:
            offsets = np.zeros(total_frames, dtype=np.int64)
            temporal_flags = np.zeros(total_frames, dtype=bool)

        # -------------------------------------------------------------
        # Stage 3: Full-Field 50 Hz High-Pass Filter (Row Chunks)
        # -------------------------------------------------------------
        if apply_highpass_50hz and total_frames > 64:
            print(
                f"\n--- Stage 3: Full-Field {highpass_cutoff_hz:.1f} Hz High-Pass Filter ---",
                flush=True,
            )
            t_stage3 = time.perf_counter()
            sos = butter(4, highpass_cutoff_hz, btype="highpass", fs=fs, output="sos")
            chunk_rows = 4  # Keep row chunks small (640 MB in float64) to ensure peak RAM < 2 GB
            n_row_chunks = (ny + chunk_rows - 1) // chunk_rows

            for c in range(n_row_chunks):
                y_start = c * chunk_rows
                y_end = min(ny, (c + 1) * chunk_rows)
                # Read row chunk into float64
                row_data = np.asarray(mmap_spatial[:, y_start:y_end, :], dtype=np.float64)
                # Zero-phase Butterworth filter along time axis (axis 0)
                row_filt = sosfiltfilt(sos, row_data, axis=0).astype(np.float32)
                mmap_spatial[:, y_start:y_end, :] = row_filt
                del row_data, row_filt
                if (c + 1) % 5 == 0:
                    gc.collect()
                if (c + 1) % max(1, n_row_chunks // 5) == 0 or (c + 1) == n_row_chunks:
                    print(
                        f"  Filtered rows [{y_start:3d}:{y_end:3d}] ({c+1}/{n_row_chunks}) in {time.perf_counter()-t_stage3:.1f} s",
                        flush=True,
                    )

            mmap_spatial.flush()
            t_stage3_elapsed = time.perf_counter() - t_stage3
            print(f"Stage 3 finished in {t_stage3_elapsed:.1f} s.", flush=True)

        # -------------------------------------------------------------
        # Export Cleaned Output HDF5 Dataset (Atomic Write)
        # -------------------------------------------------------------
        print("\n--- Writing Final Output HDF5 Dataset ---", flush=True)
        t_write = time.perf_counter()
        temp_output_file = output_file.with_name(f"{output_file.name}.tmp")
        if temp_output_file.exists():
            temp_output_file.unlink()

        with h5py.File(temp_output_file, "w") as h5_out:
            # 1. Meta group
            meta_grp = h5_out.create_group("meta")
            meta_grp.create_dataset("t", data=t_arr)
            meta_grp.create_dataset("x", data=x_arr)
            meta_grp.create_dataset("y", data=y_arr)
            meta_grp.create_dataset("frames", data=frames_arr)
            meta_grp.create_dataset("temporal_offset_cycles", data=offsets.astype(np.int32))
            if temporal_res is not None:
                meta_grp.create_dataset(
                    "temporal_uncertainty_flags",
                    data=temporal_res.uncertainty_flags.astype(bool),
                )
            for k_attr, v_attr in meta_attrs.items():
                meta_grp.attrs[k_attr] = v_attr

            # 2. Root metadata attributes
            h5_out.attrs["cleaned"] = True
            h5_out.attrs["cleaning_version"] = (
                "v3_conservative_hybrid_advanced_kinematic_veto"
                if method == "hybrid"
                else "v2_advanced_kinematic_veto"
            )
            h5_out.attrs["spatial_method"] = (
                "conservative_hybrid_harmonic_integer"
                if method == "hybrid"
                else f"gradient_domain_{method}"
            )
            h5_out.attrs["q_microns"] = float(q)
            h5_out.attrs["tau_microns"] = float(tau)
            h5_out.attrs["source_file"] = str(raw_input)
            h5_out.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
            h5_out.attrs["temporal_dc_aligned"] = bool(temporal_align)
            h5_out.attrs["temporal_selector"] = "advanced_kinematic_veto_v1"
            h5_out.attrs["highpass_filter_applied"] = bool(apply_highpass_50hz)
            h5_out.attrs["highpass_cutoff_hz"] = float(highpass_cutoff_hz)
            h5_out.attrs["highpass_filter_order"] = 4
            h5_out.attrs["highpass_filter_type"] = "butterworth_zero_phase"
            if method == "hybrid":
                h5_out.attrs["hybrid_spatial_parameters_json"] = json.dumps(
                    HYBRID_PARAMETERS, sort_keys=True
                )

            if temporal_res is not None:
                h5_out.attrs["temporal_corrections_count"] = int(temporal_res.n_corrections)
                h5_out.attrs["temporal_large_corrections_count"] = int(
                    temporal_res.n_large_corrections
                )
                h5_out.attrs["temporal_small_corrections_count"] = int(
                    temporal_res.n_small_corrections
                )
                h5_out.attrs["temporal_flagged_count"] = int(temporal_res.n_flagged_corrections)
                h5_out.attrs["max_residual_increment_um"] = float(
                    temporal_res.max_residual_increment_um
                )

            # 3. Per-frame spatial-stage diagnostics. These describe the field
            # before temporal alignment and high-pass filtering.
            diagnostics_grp = h5_out.create_group("cleaning_diagnostics")
            spatial_grp = diagnostics_grp.create_group("spatial")
            fields_to_write = (
                SPATIAL_QC_FIELDS
                if method == "hybrid"
                else (
                    "modified_fraction",
                    "raw_max_gradient_um",
                    "restored_max_gradient_um",
                )
            )
            for field in fields_to_write:
                column = SPATIAL_QC_FIELDS.index(field)
                values = spatial_qc[:, column]
                if field in {
                    "cliff_gate_passed",
                    "boundary_growth_converged",
                }:
                    values = values.astype(bool)
                elif field in {
                    "largest_cliff_component_pixels",
                    "boundary_growth_added_pixels",
                    "unresolved_boundary_violation_pixels",
                    "review_flag_bits",
                }:
                    values = values.astype(np.int32)
                spatial_grp.create_dataset(field, data=values)
            spatial_grp.attrs["stage"] = "after spatial restoration, before temporal alignment and high-pass filtering"
            spatial_grp.attrs["review_flag_bits"] = (
                "1=boundary growth; 2=unresolved boundary violation; "
                "4=gate skipped despite raw max gradient > 2*tau; "
                "8=mask fraction > 0.70; "
                "16=cliff gate passed but hybrid mask is empty"
            )

            # 4. Main group frames
            main_grp = h5_out.create_group("main")
            write_report_step = max(1, total_frames // 5)
            for t_idx in range(total_frames):
                key = _frame_key(t_idx)
                main_grp.create_dataset(key, data=mmap_spatial[t_idx])
                if (t_idx + 1) % write_report_step == 0 or (t_idx + 1) == total_frames:
                    pct = (t_idx + 1) / total_frames * 100.0
                    print(f"  Written {t_idx+1:,}/{total_frames:,} frames ({pct:.0f}%)", flush=True)

        t_write_elapsed = time.perf_counter() - t_write
        print(f"HDF5 write finished in {t_write_elapsed:.1f} s.", flush=True)

        # 4. Validate output dataset before atomic replace
        print("Validating cleaned dataset with pod_2d.inspect_input...", flush=True)
        info = inspect_input(temp_output_file, start=0, stop=None)
        assert info.n_frames == total_frames, f"Frame count mismatch: {info.n_frames} != {total_frames}"
        assert info.frame_shape == (ny, nx), f"Grid shape mismatch: {info.frame_shape} != {(ny, nx)}"

        # 5. Atomically replace temp file with destination
        temp_output_file.replace(output_file)
        if method == "hybrid":
            qc_index = {name: i for i, name in enumerate(SPATIAL_QC_FIELDS)}
            review_bits = spatial_qc[:, qc_index["review_flag_bits"]].astype(np.uint8)
            modified = spatial_qc[:, qc_index["modified_fraction"]]
            review_positions = np.flatnonzero(review_bits != 0)
            summary_path = output_file.with_suffix(".cleaning_summary.json")
            summary_path.write_text(
                json.dumps(
                    {
                        "source_file": str(raw_input),
                        "output_file": str(output_file),
                        "n_frames": int(total_frames),
                        "q_um": float(q),
                        "tau_um": float(tau),
                        "hybrid_parameters": HYBRID_PARAMETERS,
                        "spatial_modified_frame_count": int(
                            np.count_nonzero(modified > 0)
                        ),
                        "spatial_modified_fraction_quantiles": {
                            str(prob): float(np.quantile(modified, prob))
                            for prob in (0.0, 0.5, 0.9, 0.99, 1.0)
                        },
                        "review_counts": {
                            "any": int(review_positions.size),
                            "boundary_growth": int(np.count_nonzero(review_bits & 1)),
                            "unresolved_boundary": int(np.count_nonzero(review_bits & 2)),
                            "gate_skipped_severe_gradient": int(
                                np.count_nonzero(review_bits & 4)
                            ),
                            "mask_above_70pct": int(np.count_nonzero(review_bits & 8)),
                            "gate_passed_empty_mask": int(
                                np.count_nonzero(review_bits & 16)
                            ),
                        },
                        "review_frame_positions": review_positions.astype(int).tolist(),
                        "review_frame_ids": frames_arr[review_positions].astype(int).tolist(),
                        "temporal_corrections_count": int(
                            temporal_res.n_corrections if temporal_res is not None else 0
                        ),
                        "highpass_filter_applied": bool(apply_highpass_50hz),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Hybrid cleaning summary: {summary_path}", flush=True)
        total_elapsed = time.perf_counter() - started
        size_gb = output_file.stat().st_size / (1024.0 ** 3)
        print(f"\nCleaned dataset exported successfully in {total_elapsed:.1f} s ({total_frames/total_elapsed:.1f} fps)!")
        print(f"Destination: {output_file} ({size_gb:.2f} GB)")

    finally:
        # Clean up scratch memory map
        if "mmap_spatial" in locals():
            del mmap_spatial
        if mmap_path.exists():
            try:
                mmap_path.unlink()
            except OSError:
                pass

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument(
        "--method",
        type=str,
        default="integer",
        choices=["integer", "poisson", "hybrid"],
        help="Spatial restoration method (integer, poisson, or conservative hybrid)",
    )
    parser.add_argument(
        "--no-temporal-align",
        dest="temporal_align",
        action="store_false",
        help="Disable temporal cycle alignment across frames",
    )
    parser.add_argument(
        "--no-highpass-50hz",
        dest="apply_highpass_50hz",
        action="store_false",
        help="Disable 50 Hz Butterworth high-pass filter across 200x200 field",
    )
    parser.add_argument("--highpass-cutoff-hz", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-frames", type=int, help="Run pilot test on first N frames only")
    parser.add_argument("--scratch-dir", type=Path, help="Directory for temporary memory maps")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if present",
    )
    args = parser.parse_args()

    export_cleaned_dataset(
        raw_input=args.raw_input,
        output_file=args.output_file,
        q=args.q,
        tau=args.tau,
        method=args.method,
        temporal_align=args.temporal_align,
        apply_highpass_50hz=args.apply_highpass_50hz,
        highpass_cutoff_hz=args.highpass_cutoff_hz,
        workers=args.workers,
        chunk_size=args.chunk_size,
        scratch_dir=args.scratch_dir,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
