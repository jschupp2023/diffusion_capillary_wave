"""Generate side-by-side comparison video between raw and cleaned HDF5 datasets.

Shows:
1. Left panel: Raw surface height field h_raw(x, y) with phase discontinuities.
2. Center panel: Cleaned surface field h_cln(x, y) restored via Poisson integration.
3. Right panel: Integer phase cycle correction |h_raw - h_cln| / q.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from data_analysis.pod.pod_2d import _frame_key
from data_analysis.pod.phase_unwrap_gradient_poisson import DEFAULT_Q

DEFAULT_RAW = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
DEFAULT_CLEANED = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
DEFAULT_OUT_DIR = Path("videos")


def create_comparison_video(
    raw_path: Path,
    cleaned_path: Path,
    output_path: Path,
    start_frame: int = 1500,
    n_frames: int = 200,
    fps: int = 25,
    q: float = DEFAULT_Q,
    dpi: int = 150,
) -> None:
    raw_path = raw_path.expanduser().resolve()
    cleaned_path = cleaned_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating comparison video:")
    print(f"  Raw:     {raw_path}")
    print(f"  Cleaned: {cleaned_path}")
    print(f"  Output:  {output_path}")
    print(f"  Frames:  [{start_frame}, {start_frame + n_frames}) ({n_frames} frames) @ {fps} fps")

    with h5py.File(raw_path, "r") as h5_raw, h5py.File(cleaned_path, "r") as h5_cln:
        raw_grp = h5_raw["main"]
        cln_grp = h5_cln["main"]
        meta = h5_raw["meta"]

        x_um = np.asarray(meta["x"])
        y_um = np.asarray(meta["y"])
        t_arr = np.asarray(meta["t"])

        extent = [x_um[0], x_um[-1], y_um[-1], y_um[0]]

        # Determine robust color scale from cleaned sequence
        print("Determining color scale...")
        sample_indices = np.linspace(start_frame, start_frame + n_frames - 1, min(20, n_frames), dtype=int)
        samples = []
        for s_idx in sample_indices:
            f_cln = np.asarray(cln_grp[_frame_key(s_idx)], dtype=np.float32)
            f_cln_detrend = f_cln - np.mean(f_cln)
            samples.append(f_cln_detrend)
        samples = np.stack(samples, axis=0)

        v_bound = float(np.percentile(np.abs(samples), 99.0))
        vmin = -v_bound
        vmax = +v_bound
        print(f"Color limits (detrended): [{vmin:.2f}, {vmax:.2f}] um")

        # Set up figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        ax_raw, ax_cln, ax_diff = axes

        # Read first frame
        k0 = _frame_key(start_frame)
        f0_raw = np.asarray(raw_grp[k0], dtype=np.float32)
        f0_cln = np.asarray(cln_grp[k0], dtype=np.float32)
        f0_raw_dt = f0_raw - np.mean(f0_raw)
        f0_cln_dt = f0_cln - np.mean(f0_cln)
        diff_cycles = (f0_raw - f0_cln) / q

        im_raw = ax_raw.imshow(f0_raw_dt, cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, origin="upper")
        cbar_raw = fig.colorbar(im_raw, ax=ax_raw, shrink=0.8, pad=0.02)
        cbar_raw.set_label("Height $h - \\bar{h}$ [$\\mu$m]", fontsize=11)
        ax_raw.set_title("Raw Surface (calibrated)", fontsize=13, fontweight="bold", color="darkred")
        ax_raw.set_xlabel("x [$\\mu$m]", fontsize=11)
        ax_raw.set_ylabel("y [$\\mu$m]", fontsize=11)

        im_cln = ax_cln.imshow(f0_cln_dt, cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent, origin="upper")
        cbar_cln = fig.colorbar(im_cln, ax=ax_cln, shrink=0.8, pad=0.02)
        cbar_cln.set_label("Height $h - \\bar{h}$ [$\\mu$m]", fontsize=11)
        ax_cln.set_title("Cleaned Surface (Poisson Restored)", fontsize=13, fontweight="bold", color="darkgreen")
        ax_cln.set_xlabel("x [$\\mu$m]", fontsize=11)

        im_diff = ax_diff.imshow(diff_cycles, cmap="PuOr", vmin=-5, vmax=5, extent=extent, origin="upper")
        cbar_diff = fig.colorbar(im_diff, ax=ax_diff, shrink=0.8, pad=0.02)
        cbar_diff.set_label("Correction $(h_{\\text{raw}} - h_{\\text{cln}})/q$ [cycles]", fontsize=11)
        ax_diff.set_title("Removed Phase Cycle Slip", fontsize=13, fontweight="bold", color="indigo")
        ax_diff.set_xlabel("x [$\\mu$m]", fontsize=11)

        suptitle = fig.suptitle("", fontsize=14, fontweight="bold")

        writer = FFMpegWriter(fps=fps, metadata=dict(artist="Antigravity"), extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])

        print(f"Rendering {n_frames} frames with FFMpegWriter...")
        t_start = time.perf_counter()
        with writer.saving(fig, str(output_path), dpi=dpi):
            for i in range(n_frames):
                frame_idx = start_frame + i
                key = _frame_key(frame_idx)
                f_raw = np.asarray(raw_grp[key], dtype=np.float32)
                f_cln = np.asarray(cln_grp[key], dtype=np.float32)

                f_raw_dt = f_raw - np.mean(f_raw)
                f_cln_dt = f_cln - np.mean(f_cln)
                diff = (f_raw - f_cln) / q

                im_raw.set_data(f_raw_dt)
                im_cln.set_data(f_cln_dt)
                im_diff.set_data(diff)

                t_ms = t_arr[frame_idx] * 1e3
                max_diff_cycles = np.max(np.abs(diff))
                suptitle.set_text(
                    f"Frame {frame_idx:05d} | t = {t_ms:.2f} ms | Max Correction: {max_diff_cycles:.1f} cycles ({max_diff_cycles * q:.1f} $\\mu$m)"
                )

                writer.grab_frame()

                if (i + 1) % 50 == 0 or (i + 1) == n_frames:
                    fps_render = (i + 1) / (time.perf_counter() - t_start)
                    print(f"  Rendered {i + 1}/{n_frames} frames ({fps_render:.1f} fps)")

        plt.close(fig)
        elapsed = time.perf_counter() - t_start
        print(f"Video saved successfully to: {output_path} in {elapsed:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cleaned-file", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "comparison_0p20_frame1450_1650.mp4")
    parser.add_argument("--start-frame", type=int, default=1450)
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--q", type=float, default=DEFAULT_Q)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    create_comparison_video(
        raw_path=args.raw_file,
        cleaned_path=args.cleaned_file,
        output_path=args.output,
        start_frame=args.start_frame,
        n_frames=args.n_frames,
        fps=args.fps,
        q=args.q,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

