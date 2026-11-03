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
from scipy.signal import butter, sosfiltfilt

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


def _process_chunk_worker(
    input_path_str: str,
    start_idx: int,
    end_idx: int,
    q: float,
    tau: float,
    method: str = "integer",
) -> tuple[int, int, np.ndarray, int, np.ndarray]:
    """Worker to restore a chunk of frames and compute spatial tracking channels."""
    n_frames = end_idx - start_idx
    with h5py.File(input_path_str, "r") as h5_in:
        main_grp = h5_in["main"]
        first_key = _frame_key(start_idx)
        first_shape = main_grp[first_key].shape
        cleaned = np.empty((n_frames, *first_shape), dtype=np.float32)
        channels = np.empty((n_frames, 6), dtype=np.float64)
        n_modified = 0

        for i in range(n_frames):
            frame_num = start_idx + i
            key = _frame_key(frame_num)
            raw = np.asarray(main_grp[key], dtype=np.float32)
            res = restore_frame_poisson(raw, q=q, tau=tau, method=method)
            restored = res.restored
            cleaned[i] = restored
            channels[i] = extract_spatial_channels(restored)
            if res.was_modified:
                n_modified += 1

    return start_idx, end_idx, cleaned, n_modified, channels


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
                    start, end, chunk_data, n_mod, chunk_channels = fut.result()
                    del future_to_chunk[fut]
                    total_modified += n_mod

                    # Store directly into memory-mapped scratch buffer
                    mmap_spatial[start:end] = chunk_data
                    spatial_channels[start:end] = chunk_channels

                    n_chunk_frames = end - start
                    frames_written += n_chunk_frames
                    completed_chunks += 1
                    del chunk_data, chunk_channels

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
            chunk_rows = 20
            n_row_chunks = (ny + chunk_rows - 1) // chunk_rows

            for c in range(n_row_chunks):
                y_start = c * chunk_rows
                y_end = min(ny, (c + 1) * chunk_rows)
                # Read row chunk into float64
                row_data = np.asarray(mmap_spatial[:, y_start:y_end, :], dtype=np.float64)
                # Zero-phase Butterworth filter along time axis (axis 0)
                row_filt = sosfiltfilt(sos, row_data, axis=0).astype(np.float32)
                mmap_spatial[:, y_start:y_end, :] = row_filt
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
            h5_out.attrs["cleaning_version"] = "v2_advanced_kinematic_veto"
            h5_out.attrs["spatial_method"] = "gradient_domain_integer"
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

            # 3. Main group frames
            main_grp = h5_out.create_group("main")
            for t_idx in range(total_frames):
                key = _frame_key(t_idx)
                main_grp.create_dataset(key, data=mmap_spatial[t_idx])

        t_write_elapsed = time.perf_counter() - t_write
        print(f"HDF5 write finished in {t_write_elapsed:.1f} s.", flush=True)

        # 4. Validate output dataset before atomic replace
        print("Validating cleaned dataset with pod_2d.inspect_input...", flush=True)
        info = inspect_input(temp_output_file, start=0, stop=None)
        assert info.n_frames == total_frames, f"Frame count mismatch: {info.n_frames} != {total_frames}"
        assert info.frame_shape == (ny, nx), f"Grid shape mismatch: {info.frame_shape} != {(ny, nx)}"

        # 5. Atomically replace temp file with destination
        temp_output_file.replace(output_file)
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
        choices=["integer", "poisson"],
        help="Restoration method ('integer' or 'poisson')",
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
