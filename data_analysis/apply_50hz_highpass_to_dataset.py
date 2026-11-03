"""Apply 50 Hz zero-phase high-pass filter across all 100,001 frames of the cleaned dataset.

Eliminates low-frequency (< 50 Hz) thermal drift and slow container sloshing at every pixel,
while preserving all physical wave dynamics (including 140 Hz acoustic/sloshing resonance).
Uses memory-mapped array chunks to ensure strictly bounded RAM usage (< 2 GB).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent BLAS thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

from data_analysis.pod.pod_2d import _frame_key, inspect_input

INPUT_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
OUTPUT_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5")
SCRATCH_DIR = Path("/home/jonas/ucsd_thesis/scratch")


def apply_50hz_filter(
    input_path: Path = INPUT_H5,
    output_path: Path = OUTPUT_H5,
    cutoff_hz: float = 50.0,
    chunk_rows: int = 20,
) -> None:
    t_total = time.perf_counter()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    info = inspect_input(input_path, 0, None)
    n_frames = info.n_frames
    ny, nx = info.frame_shape
    
    mean_dt = float(np.mean(np.diff(info.time)))
    fs = 1.0 / mean_dt
    
    print(f"Applying {cutoff_hz:.1f} Hz high-pass filter to {input_path.name}:")
    print(f"  Frames: {n_frames:,} | Resolution: {nx}x{ny} | fs: {fs:.1f} Hz")
    
    sos = butter(4, cutoff_hz, btype="highpass", fs=fs, output="sos")
    
    # 1. Allocate memory-mapped arrays on disk
    raw_mmap_path = SCRATCH_DIR / "mmap_raw_frames.dat"
    hp_mmap_path = SCRATCH_DIR / "mmap_hp_frames.dat"
    
    try:
        print("1. Reading all frames into memory-mapped buffer...")
        t0 = time.perf_counter()
        raw_mmap = np.memmap(raw_mmap_path, dtype=np.float32, mode="w+", shape=(n_frames, ny, nx))
        
        with h5py.File(input_path, "r") as fin:
            main_in = fin["main"]
            report_step = n_frames // 5
            for i in range(n_frames):
                raw_mmap[i] = np.asarray(main_in[_frame_key(i)], dtype=np.float32)
                if (i + 1) % report_step == 0 or (i + 1) == n_frames:
                    print(f"  Loaded {i+1:,}/{n_frames:,} frames ({(i+1)/n_frames*100:.0f}%, {time.perf_counter()-t0:.1f} s)")
        raw_mmap.flush()
        
        print("2. Filtering spatial chunks along time axis (zero-phase 4th-order Butterworth)...")
        t_flt = time.perf_counter()
        hp_mmap = np.memmap(hp_mmap_path, dtype=np.float32, mode="w+", shape=(n_frames, ny, nx))
        
        n_chunks = (ny + chunk_rows - 1) // chunk_rows
        for c in range(n_chunks):
            y_start = c * chunk_rows
            y_end = min(ny, (c + 1) * chunk_rows)
            t_chunk = time.perf_counter()
            # Read chunk (n_frames, chunk_height, nx)
            chunk_data = np.asarray(raw_mmap[:, y_start:y_end, :], dtype=np.float64)
            # Filter along time axis (axis 0)
            chunk_filtered = sosfiltfilt(sos, chunk_data, axis=0).astype(np.float32)
            hp_mmap[:, y_start:y_end, :] = chunk_filtered
            print(f"  Filtered rows [{y_start:3d}:{y_end:3d}] ({c+1}/{n_chunks}) in {time.perf_counter()-t_chunk:.1f} s")
            
        hp_mmap.flush()
        print(f"  Filtering complete in {time.perf_counter() - t_flt:.1f} s")
        
        # 3. Export to output HDF5
        print(f"3. Writing output dataset to {output_path.name}...")
        t_out = time.perf_counter()
        if output_path.exists():
            output_path.unlink()
            
        with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
            # Copy metadata and attributes
            fin.copy("meta", fout)
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            fout.attrs["highpass_filter_applied"] = True
            fout.attrs["highpass_cutoff_hz"] = cutoff_hz
            fout.attrs["highpass_filter_order"] = 4
            fout.attrs["highpass_filter_type"] = "butterworth_zero_phase"
            
            # Create main group
            main_out = fout.create_group("main")
            for i in range(n_frames):
                main_out.create_dataset(
                    _frame_key(i),
                    data=hp_mmap[i],
                    dtype="float32",
                )
                if (i + 1) % report_step == 0 or (i + 1) == n_frames:
                    print(f"  Written {i+1:,}/{n_frames:,} frames ({(i+1)/n_frames*100:.0f}%, {time.perf_counter()-t_out:.1f} s)")
                    
        print(f"  Export complete in {time.perf_counter() - t_out:.1f} s")
        print(f"  Output size: {output_path.stat().st_size / 1e9:.2f} GB")
        
    finally:
        # Clean up temporary memory-mapped scratch files
        if raw_mmap_path.exists():
            raw_mmap_path.unlink(missing_ok=True)
        if hp_mmap_path.exists():
            hp_mmap_path.unlink(missing_ok=True)
            
    print(f"\nAll operations completed successfully in {time.perf_counter() - t_total:.1f} s.")


if __name__ == "__main__":
    apply_50hz_filter()

