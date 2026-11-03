"""Render 3D surface videos for frames 90000 to 90100 of the 0.35 Vpp dataset.

Renders:
1. Cleaned 3-D surface video: videos/cleaned_surface_11252025_e_0.35vpp_data_roi-none_cal-true_90000_90100.mp4
2. Side-by-side 3-D comparison video: videos/comparison_3d_0p35_90000_90100.mp4
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np

from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.pod.pod_surface_video import make_surface_comparison_video, make_video

RAW_FILE = Path("/home/jonas/ucsd_thesis/11252025_e_0.35vpp_data_roi-none_cal-true.hdf5")
CLEANED_FILE = Path("/home/jonas/ucsd_thesis/11252025_e_0.35vpp_data_roi-none_cal-true_cleaned.hdf5")
START = 90000
STOP = 90100
SPATIAL_POINTS = 30
FPS = 30


def main() -> None:
    print(f"Rendering 3D videos for 0.35 Vpp frames [{START}, {STOP})...")

    # 1. Render single cleaned 3D surface video matching the raw video exactly
    cleaned_output = Path("videos") / f"cleaned_surface_{RAW_FILE.stem}_{START}_{STOP}.mp4"
    print(f"\n1. Generating Cleaned 3D Video -> {cleaned_output}...")
    make_video(
        experiment=None,
        rep=None,
        rank=None,
        start=START,
        stop=STOP,
        spatial_points=SPATIAL_POINTS,
        max_frames=STOP - START,
        fps=FPS,
        overwrite=True,
        raw_input=CLEANED_FILE,
        output=cleaned_output,
    )

    # 2. Render side-by-side 3D comparison video
    comparison_output = Path("videos") / f"comparison_3d_0p35_{START}_{STOP}.mp4"
    print(f"\n2. Generating Side-by-Side 3D Comparison Video -> {comparison_output}...")

    info = inspect_input(RAW_FILE, START, STOP)
    ny, nx = info.frame_shape
    ystep = max(1, round((ny - 1) / (min(SPATIAL_POINTS, ny) - 1)))
    xstep = max(1, round((nx - 1) / (min(SPATIAL_POINTS, nx) - 1)))
    yidx = np.arange(0, ny, ystep)
    xidx = np.arange(0, nx, xstep)

    n_frames = STOP - START
    raw_fields = np.empty((n_frames, len(yidx), len(xidx)), dtype=np.float32)
    cln_fields = np.empty((n_frames, len(yidx), len(xidx)), dtype=np.float32)

    with h5py.File(RAW_FILE, "r") as f_raw, h5py.File(CLEANED_FILE, "r") as f_cln:
        grp_raw = f_raw["main"]
        grp_cln = f_cln["main"]
        for i in range(n_frames):
            frame_num = START + i
            key = _frame_key(frame_num)
            raw_fields[i] = np.asarray(grp_raw[key], dtype=np.float32)[::ystep, ::xstep]
            cln_fields[i] = np.asarray(grp_cln[key], dtype=np.float32)[::ystep, ::xstep]

    x = info.x[xidx]
    y = info.y[yidx]
    times = info.time

    make_surface_comparison_video(
        raw_fields,
        cln_fields,
        x,
        y,
        times,
        output=comparison_output,
        x_unit=info.units["x"],
        y_unit=info.units["y"],
        z_unit=info.units["z"],
        fps=FPS,
        title="0.35 Vpp (0p30): Raw Surface (Left) vs Poisson Restored Surface (Right)",
        overwrite=True,
    )
    print("\nAll 3D videos rendered successfully!")


if __name__ == "__main__":
    main()

