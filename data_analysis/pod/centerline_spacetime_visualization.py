"""1D Centerline Space-Time Visualization Suite (Hovmöller Diagram, Waterfall Profiles, and MP4 Video).

Visualizes the spatial and temporal continuity of 1-D surface elevation profiles:
1. Space-Time Hovmöller Diagram:
   - Plots h(x, t) on a 2-D space-time grid (x on horizontal axis, time on vertical axis)
   - Propagating capillary waves appear as smooth diagonal streaks (dx/dt = c_phase)
   - Optical phase cliffs appear as sharp horizontal stripes (temporal frame slips)
     and vertical step shears (intra-frame tears)
   - Compares Raw vs Cleaned side-by-side with difference map
2. Centerline Multi-Line Waterfall Profile:
   - Visualizes the continuous evolution of h(x) across consecutive timesteps
   - Colored by time to show propagating wave packets and crest crest-line continuity
3. Synchronized Side-by-Side MP4 Video:
   - Top row: Dynamic 1-D profile h(x) (Raw on left, Cleaned on right)
   - Bottom row: Synchronized 2-D space-time map with scanning time cursor

Example:
    python centerline_spacetime_visualization.py --center-frame 1553 --window 120 --make-video
    python centerline_spacetime_visualization.py --center-frame 2300 --window 120 --make-video
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent BLAS thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from data_analysis.pod.pod_2d import _frame_key

DEFAULT_RAW = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
DEFAULT_CLEAN = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
DEFAULT_OUTPUT_DIR = Path("runs/spacetime_centerline")


def extract_centerline_spacetime(
    raw_h5_path: Path,
    clean_h5_path: Path,
    start_frame: int,
    end_frame: int,
    row: int = 100,
) -> dict[str, np.ndarray]:
    """Extract 1-D centerline space-time slice from raw and cleaned HDF5 files."""
    n_frames = end_frame - start_frame
    
    with h5py.File(raw_h5_path, "r") as f_raw, h5py.File(clean_h5_path, "r") as f_cln:
        raw_main = f_raw["main"]
        cln_main = f_cln["main"]
        
        t_all = np.asarray(f_raw["meta"]["t"])
        x_um = np.asarray(f_raw["meta"]["x"], dtype=np.float32)
        y_um = np.asarray(f_raw["meta"]["y"], dtype=np.float32)
        
        if row < 0 or row >= len(y_um):
            raise ValueError(f"Row {row} out of range [0, {len(y_um)})")
            
        t_sub = t_all[start_frame:end_frame]
        
        st_raw = np.empty((n_frames, len(x_um)), dtype=np.float32)
        st_cln = np.empty((n_frames, len(x_um)), dtype=np.float32)
        
        for i in range(n_frames):
            fn = start_frame + i
            k = _frame_key(fn)
            st_raw[i] = raw_main[k][row, :]
            st_cln[i] = cln_main[k][row, :]
            
    return {
        "time_s": t_sub,
        "time_ms": (t_sub - t_sub[0]) * 1e3,
        "time_abs_ms": t_sub * 1e3,
        "x_um": x_um,
        "y_um": y_um,
        "row_idx": row,
        "row_y_um": float(y_um[row]),
        "raw_st": st_raw,      # shape: (n_frames, n_x)
        "clean_st": st_cln,    # shape: (n_frames, n_x)
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def plot_spacetime_hovmoller(
    data: dict[str, np.ndarray],
    title_suffix: str,
    output_path: Path,
    dpi: int = 200,
) -> None:
    """Generate high-resolution 3-panel Hovmöller Space-Time diagram + 1-D profile slices."""
    raw = data["raw_st"]
    cln = data["clean_st"]
    diff = np.abs(cln - raw)
    
    x = data["x_um"]
    t_ms = data["time_ms"]
    start_fn = data["start_frame"]
    end_fn = data["end_frame"]
    row_y = data["row_y_um"]
    
    # Establish robust common color limits from the cleaned data
    p2, p98 = np.percentile(cln, [1, 99])
    v_margin = 0.1 * (p98 - p2)
    vmin, vmax = p2 - v_margin, p98 + v_margin
    
    extent = [x[0], x[-1], t_ms[0], t_ms[-1]]
    
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.8, 1.0])
    
    # Panel 0: Raw Space-Time
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(raw, origin="lower", cmap="viridis", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    ax0.set_title(f"Raw Surface $h_{{\\mathrm{{raw}}}}(x, t)$\n(Optical phase steps & tears)", fontsize=12, fontweight="bold")
    ax0.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Time elapsed [$\mathrm{ms}$]", fontsize=11)
    fig.colorbar(im0, ax=ax0, label=r"Elevation $h$ [$\mu\mathrm{m}$]")
    
    # Panel 1: Cleaned Space-Time
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(cln, origin="lower", cmap="viridis", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    ax1.set_title(f"Cleaned Surface $h_{{\\mathrm{{clean}}}}(x, t)$\n(Smooth propagating wave crests)", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Time elapsed [$\mathrm{ms}$]", fontsize=11)
    fig.colorbar(im1, ax=ax1, label=r"Elevation $h$ [$\mu\mathrm{m}$]")
    
    # Panel 2: Difference Space-Time Map
    ax2 = fig.add_subplot(gs[0, 2])
    diff_vmax = max(2.0, float(np.percentile(diff, 99.5)))
    im2 = ax2.imshow(diff, origin="lower", cmap="hot", extent=extent, aspect="auto", vmin=0, vmax=diff_vmax)
    ax2.set_title(f"Absolute Correction $|h_{{\\mathrm{{clean}}}} - h_{{\\mathrm{{raw}}}}|$\nMax diff = {np.max(diff):.2f} $\\mu\\mathrm{{m}}$", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=11)
    ax2.set_ylabel(r"Time elapsed [$\mathrm{ms}$]", fontsize=11)
    fig.colorbar(im2, ax=ax2, label=r"Discontinuity Step [$\mu\mathrm{m}$]")
    
    # Bottom Row: Centerline Profiles at sample timestamps & Waterfall
    # Panel 3: 1D Profile Comparison at 3 selected key timestamps
    ax3 = fig.add_subplot(gs[1, 0:2])
    n_frames = len(t_ms)
    sample_indices = [
        int(0.15 * n_frames),
        int(0.50 * n_frames),
        int(0.85 * n_frames),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for idx, col in zip(sample_indices, colors):
        fn = start_fn + idx
        t_val = t_ms[idx]
        ax3.plot(x, raw[idx], color=col, linestyle="--", lw=1.2, alpha=0.7, label=f"Raw (Frame {fn}, {t_val:.2f} ms)")
        ax3.plot(x, cln[idx], color=col, linestyle="-", lw=1.8, label=f"Cleaned (Frame {fn}, {t_val:.2f} ms)")
        
    ax3.set_title(f"1-D Centerline Profiles $h(x)$ Comparison (Cross-Section at $y = {row_y:.1f}\\,\\mu\\mathrm{{m}}$)", fontsize=12, fontweight="bold")
    ax3.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=11)
    ax3.set_ylabel(r"Elevation $h$ [$\mu\mathrm{m}$]", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(fontsize=9, loc="upper right", ncol=2)
    
    # Panel 4: Center-Point Time Trajectory
    ax4 = fig.add_subplot(gs[1, 2])
    cx_idx = len(x) // 2
    ax4.plot(t_ms, raw[:, cx_idx], color="crimson", lw=1.2, alpha=0.7, label=r"Raw Center $h(x_{\mathrm{mid}}, t)$")
    ax4.plot(t_ms, cln[:, cx_idx], color="forestgreen", lw=1.5, label=r"Cleaned Center")
    ax4.set_title(r"Center-Point Time History $h(x_{\mathrm{mid}}, t)$", fontsize=12, fontweight="bold")
    ax4.set_xlabel(r"Time elapsed [$\mathrm{ms}$]", fontsize=11)
    ax4.set_ylabel(r"Elevation $h$ [$\mu\mathrm{m}$]", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.35)
    ax4.legend(fontsize=9, loc="upper right")
    
    fig.suptitle(
        f"Space-Time Continuity Analysis on Centerline ($y = {row_y:.1f}\\,\\mu\\mathrm{{m}}$): {title_suffix}\n"
        f"Frames [{start_fn:,} to {end_fn:,}] | Total duration: {t_ms[-1]:.3f} ms | $\\Delta t = 8.68\\,\\mu\\mathrm{{s}}$",
        fontsize=14,
        fontweight="bold",
    )
    
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Hovmöller plot to {output_path}")


def plot_waterfall_evolution(
    data: dict[str, np.ndarray],
    title_suffix: str,
    output_path: Path,
    n_curves: int = 30,
    dpi: int = 200,
) -> None:
    """Generate 1-D Waterfall stacked curves showing continuous wave propagation."""
    raw = data["raw_st"]
    cln = data["clean_st"]
    x = data["x_um"]
    t_ms = data["time_ms"]
    start_fn = data["start_frame"]
    end_fn = data["end_frame"]
    row_y = data["row_y_um"]
    
    total_steps = len(t_ms)
    curve_indices = np.linspace(0, total_steps - 1, min(n_curves, total_steps), dtype=int)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True, constrained_layout=True)
    
    # Calculate vertical spacing offset for waterfall
    amp_range = float(np.percentile(cln, 98) - np.percentile(cln, 2))
    offset_step = amp_range * 0.35
    
    cmap = plt.get_cmap("viridis")
    
    for i, idx in enumerate(curve_indices):
        t_frac = i / (len(curve_indices) - 1)
        color = cmap(t_frac)
        y_offset = i * offset_step
        
        # Raw waterfall
        axes[0].plot(x, raw[idx] + y_offset, color=color, lw=1.2, alpha=0.85)
        # Cleaned waterfall
        axes[1].plot(x, cln[idx] + y_offset, color=color, lw=1.4, alpha=0.9)
        
    axes[0].set_title(f"Raw Centerline Waterfall\n(Notice jagged tears and sudden vertical step shifts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[0].set_ylabel(r"Stacked Elevation $h(x, t) + c\cdot t$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.25)
    
    axes[1].set_title(f"Cleaned Centerline Waterfall\n(Notice smooth propagating crests and wave phase continuity)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.25)
    
    # Add colorbar for time
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t_ms[0], vmax=t_ms[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], label=r"Time elapsed [$\mathrm{ms}$]")
    
    fig.suptitle(
        f"1-D Wave Packet Propagation Waterfall ($y = {row_y:.1f}\\,\\mu\\mathrm{{m}}$): {title_suffix}\n"
        f"Frames [{start_fn:,} to {end_fn:,}] | {len(curve_indices)} profiles spanning {t_ms[-1]:.3f} ms",
        fontsize=14,
        fontweight="bold",
    )
    
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Waterfall plot to {output_path}")


def generate_spacetime_video(
    data: dict[str, np.ndarray],
    output_path: Path,
    fps: int = 30,
) -> None:
    """Render synchronized side-by-side MP4 video comparing raw vs cleaned 1D profile evolution."""
    raw = data["raw_st"]
    cln = data["clean_st"]
    x = data["x_um"]
    t_ms = data["time_ms"]
    start_fn = data["start_frame"]
    end_fn = data["end_frame"]
    row_y = data["row_y_um"]
    n_frames = len(t_ms)
    
    print(f"Rendering side-by-side MP4 video ({n_frames} frames at {fps} fps)...")
    t0 = time.perf_counter()
    
    # Establish robust vertical axis limits
    p1, p99 = np.percentile(cln, [1, 99])
    y_min_plot = min(float(np.percentile(raw, 1)), p1) - 5.0
    y_max_plot = max(float(np.percentile(raw, 99)), p99) + 5.0
    
    # Common color scale for bottom space-time map
    vmin, vmax = p1 - 2.0, p99 + 2.0
    extent = [x[0], x[-1], t_ms[0], t_ms[-1]]
    
    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    
    # Top-Left: Raw 1D profile
    ax_top_raw = fig.add_subplot(gs[0, 0])
    ax_top_raw.set_xlim(x[0], x[-1])
    ax_top_raw.set_ylim(y_min_plot, y_max_plot)
    ax_top_raw.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_top_raw.set_ylabel(r"Elevation $h$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_top_raw.grid(True, linestyle="--", alpha=0.3)
    (line_raw,) = ax_top_raw.plot([], [], color="crimson", lw=2.0, label="Raw Profile")
    title_raw = ax_top_raw.set_title("", fontsize=11, fontweight="bold")
    ax_top_raw.legend(loc="upper right", fontsize=9)
    
    # Top-Right: Cleaned 1D profile
    ax_top_cln = fig.add_subplot(gs[0, 1])
    ax_top_cln.set_xlim(x[0], x[-1])
    ax_top_cln.set_ylim(y_min_plot, y_max_plot)
    ax_top_cln.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_top_cln.set_ylabel(r"Elevation $h$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_top_cln.grid(True, linestyle="--", alpha=0.3)
    (line_cln,) = ax_top_cln.plot([], [], color="forestgreen", lw=2.0, label="Cleaned Profile")
    title_cln = ax_top_cln.set_title("", fontsize=11, fontweight="bold")
    ax_top_cln.legend(loc="upper right", fontsize=9)
    
    # Bottom-Left: Raw Space-Time Map
    ax_bot_raw = fig.add_subplot(gs[1, 0])
    ax_bot_raw.imshow(raw, origin="lower", cmap="viridis", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    ax_bot_raw.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_bot_raw.set_ylabel(r"Time elapsed [$\mathrm{ms}$]", fontsize=10)
    ax_bot_raw.set_title(r"Raw Space-Time $h(x, t)$", fontsize=11, fontweight="bold")
    cursor_raw = ax_bot_raw.axhline(0, color="red", lw=1.5, ls="--")
    
    # Bottom-Right: Cleaned Space-Time Map
    ax_bot_cln = fig.add_subplot(gs[1, 1])
    im_bot = ax_bot_cln.imshow(cln, origin="lower", cmap="viridis", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    ax_bot_cln.set_xlabel(r"Spatial coordinate $x$ [$\mu\mathrm{m}$]", fontsize=10)
    ax_bot_cln.set_ylabel(r"Time elapsed [$\mathrm{ms}$]", fontsize=10)
    ax_bot_cln.set_title(r"Cleaned Space-Time $h(x, t)$", fontsize=11, fontweight="bold")
    cursor_cln = ax_bot_cln.axhline(0, color="lime", lw=1.5, ls="--")
    fig.colorbar(im_bot, ax=ax_bot_cln, label=r"Elevation [$\mu\mathrm{m}$]")
    
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="Antigravity"), bitrate=3000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with writer.saving(fig, str(output_path), dpi=160):
        for i in range(n_frames):
            fn = start_fn + i
            cur_t = t_ms[i]
            
            # Update 1D lines
            line_raw.set_data(x, raw[i])
            line_cln.set_data(x, cln[i])
            
            # Update titles
            title_raw.set_text(f"Raw Centerline (Frame {fn:,} | t = {cur_t:.3f} ms)")
            title_cln.set_text(f"Cleaned Centerline (Frame {fn:,} | t = {cur_t:.3f} ms)")
            
            # Update time cursor on space-time map
            cursor_raw.set_ydata([cur_t, cur_t])
            cursor_cln.set_ydata([cur_t, cur_t])
            
            writer.grab_frame()
            
    plt.close(fig)
    elapsed = time.perf_counter() - t0
    print(f"Video export complete in {elapsed:.1f} s! Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--clean-input", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--center-frame", type=int, default=1553, help="Center frame of interest")
    parser.add_argument("--window", type=int, default=120, help="Total frame window length")
    parser.add_argument("--row", type=int, default=100, help="Grid row index for centerline (default 100)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--make-video", action="store_true", help="Generate side-by-side MP4 video")
    parser.add_argument("--fps", type=int, default=30, help="Framerate for MP4 video")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    start_frame = max(0, args.center_frame - args.window // 2)
    end_frame = start_frame + args.window
    
    print(f"Extracting space-time centerline for Frame {args.center_frame} (Window [{start_frame}, {end_frame}])...")
    data = extract_centerline_spacetime(
        raw_h5_path=args.raw_input,
        clean_h5_path=args.clean_input,
        start_frame=start_frame,
        end_frame=end_frame,
        row=args.row,
    )
    
    title_suffix = f"Episode Around Frame {args.center_frame}"
    
    # 1. Hovmöller Space-Time Diagram
    hov_path = args.output_dir / f"spacetime_hovmoller_{args.center_frame}.png"
    plot_spacetime_hovmoller(data, title_suffix=title_suffix, output_path=hov_path)
    
    # 2. Waterfall Profile Plot
    wf_path = args.output_dir / f"centerline_waterfall_{args.center_frame}.png"
    plot_waterfall_evolution(data, title_suffix=title_suffix, output_path=wf_path)
    
    # 3. Optional Side-by-Side MP4 Video
    if args.make_video:
        video_path = args.output_dir / f"centerline_comparison_{args.center_frame}.mp4"
        generate_spacetime_video(data, output_path=video_path, fps=args.fps)
        
    print(f"\nAll visualizations for Frame {args.center_frame} completed successfully!")


if __name__ == "__main__":
    main()
