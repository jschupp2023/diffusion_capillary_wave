"""Render a mean-restored POD reconstruction as a fixed-camera 3-D surface MP4.

Example:
    python data_analysis/pod/pod_surface_video.py --experiment 0p20 --rep 1 \
        --rank 100 --start 1000 --stop 2000

    python data_analysis/pod/pod_surface_video.py \
        --raw-input /home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5 \
        --start 1000 --stop 2000 --spatial-points 30 --max-frames 300

``--stop`` is exclusive. Up to 300 evenly spaced timesteps are rendered by
default; set ``--max-frames`` to the window length to render every timestep.
Raw videos allow at most 1000 rendered frames and show calibrated heights as stored.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import h5py
import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np

if __package__:
    from .pod_2d import _frame_key, inspect_input
    from .surface_visualization import (SurfaceFigure, draw_surface,
                                        make_surface_figure, padded_limits)
else:
    from pod_2d import _frame_key, inspect_input
    from surface_visualization import (SurfaceFigure, draw_surface,
                                       make_surface_figure, padded_limits)


DATA_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
RAW_FRAME_LIMIT = 1000


def make_surface_comparison_video(
    reference_fields: np.ndarray,
    generated_fields: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    times_seconds: np.ndarray,
    *,
    output: Path,
    x_unit: str,
    y_unit: str,
    z_unit: str,
    fps: int = 30,
    title: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Render reference and generated surface sequences side by side."""
    reference = np.asarray(reference_fields, dtype=np.float32)
    generated = np.asarray(generated_fields, dtype=np.float32)
    x = np.asarray(x)
    y = np.asarray(y)
    times = np.asarray(times_seconds, dtype=float)
    expected = (len(times), len(y), len(x))
    if (reference.shape != expected or generated.shape != expected
            or not reference.size or fps < 1):
        raise ValueError(
            "Reference/generated fields must match finite [time, y, x] coordinates."
        )
    if (len(x) < 3 or len(y) < 3 or not np.all(np.diff(x) > 0)
            or not np.all(np.diff(y) > 0)
            or not np.isfinite(times).all() or not np.all(np.diff(times) >= 0)
            or not np.isfinite(reference).all() or not np.isfinite(generated).all()):
        raise ValueError("Surface video arrays must be finite with valid spatial/time grids.")
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write MP4 video.")

    began = time.perf_counter()
    zlim = padded_limits(np.stack((reference, generated)))
    norm = Normalize(vmin=zlim[0], vmax=zlim[1])
    cmap = matplotlib.colormaps["viridis"]
    figure = Figure(figsize=(11, 5), dpi=90, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = (
        figure.add_axes((0.02, 0.10, 0.40, 0.78), projection="3d"),
        figure.add_axes((0.46, 0.10, 0.40, 0.78), projection="3d"),
    )
    plots = []
    for axis in axes:
        axis.view_init(elev=30, azim=-60)
        axis.set_box_aspect((1, 1, .48))
        axis.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=zlim)
        axis.set_xlabel(f"x [{x_unit}]", labelpad=2)
        axis.set_ylabel(f"y [{y_unit}]", labelpad=2)
        axis.set_zlabel(f"height [{z_unit}]", labelpad=2)
        axis.tick_params(labelsize=7, pad=0)
        plots.append(SurfaceFigure(figure, canvas, axis, norm, cmap))
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=figure.add_axes((0.91, 0.20, 0.02, 0.58)),
    )
    colorbar.set_label(f"height [{z_unit}]", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    if title:
        figure.suptitle(title, fontsize=11, y=.97)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    width, height = canvas.get_width_height()
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
        "rgb24", "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(temporary_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    surfaces = [None, None]
    try:
        assert process.stdin is not None
        for index, (reference_field, generated_field) in enumerate(
                zip(reference, generated)):
            for surface_index, (plot, field, label) in enumerate(zip(
                    plots, (reference_field, generated_field),
                    ("Reference reconstruction", "Generated reconstruction"))):
                if surfaces[surface_index] is not None:
                    surfaces[surface_index].remove()
                surfaces[surface_index] = draw_surface(plot, x, y, field)
                plot.axes.set_title(
                    f"{label}\nstep {index} | t = {1e3 * times[index]:.3f} ms",
                    fontsize=9, pad=3,
                )
            canvas.draw()
            process.stdin.write(
                np.asarray(canvas.buffer_rgba())[:, :, :3].tobytes()
            )
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
        os.replace(temporary_path, output)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        figure.clear()
    print(f"Saved {output} in {time.perf_counter() - began:.1f} s.", flush=True)
    return output


def _pod_file(experiment: str, rep: int, rank: int, root: Path) -> Path:
    exp_dir = root / experiment
    matches = list(exp_dir.glob(f"*_rep{rep}/pod_2d_r*.h5"))
    candidates = []
    for path in matches:
        try:
            stored_rank = int(path.stem.removeprefix("pod_2d_r"))
        except ValueError:
            continue
        if stored_rank >= rank:
            candidates.append((stored_rank, path))
    if not candidates:
        raise FileNotFoundError(
            f"No POD file with at least rank {rank} for {experiment} rep {rep} in {exp_dir}"
        )
    return min(candidates)[1]


def make_video(
    experiment: str | None,
    rep: int | None,
    rank: int | None,
    start: int,
    stop: int,
    *,
    root: Path = DATA_ROOT,
    output: Path | None = None,
    spatial_points: int = 30,
    max_frames: int = 300,
    fps: int = 30,
    overwrite: bool = False,
    raw_input: Path | None = None,
) -> Path:
    if start < 0 or stop <= start:
        raise ValueError("Require 0 <= start < stop.")
    if spatial_points < 3 or fps < 1 or max_frames < 1:
        raise ValueError("Require spatial-points >= 3, fps >= 1, max-frames >= 1.")
    if raw_input is not None:
        if any(value is not None for value in (experiment, rep, rank)):
            raise ValueError("Raw input cannot be combined with experiment, rep, or rank.")
        if max_frames > RAW_FRAME_LIMIT:
            raise ValueError(f"Raw videos are limited to {RAW_FRAME_LIMIT} rendered frames.")
        source = Path(raw_input).expanduser().resolve()
        output = output or Path("videos") / f"raw_surface_{source.stem}_{start}_{stop}.mp4"
    else:
        if not experiment or rep is None or rank is None or rep < 1 or rank < 1:
            raise ValueError("POD video requires experiment, rep >= 1, and rank >= 1.")
        source = _pod_file(experiment, rep, rank, root)
        output = output or Path("videos") / (
            f"pod_surface_{experiment}_rep{rep}_r{rank}_{start}_{stop}.mp4"
        )
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write MP4 video.")

    began = time.perf_counter()
    indices = np.arange(start, stop)
    if len(indices) > max_frames:
        indices = indices[np.linspace(0, len(indices) - 1, max_frames, dtype=int)]
    if raw_input is not None:
        info = inspect_input(source, start, stop)
        ny, nx = info.frame_shape
        ystep = max(1, round((ny - 1) / (min(spatial_points, ny) - 1)))
        xstep = max(1, round((nx - 1) / (min(spatial_points, nx) - 1)))
        yidx = np.arange(0, ny, ystep)
        xidx = np.arange(0, nx, xstep)
        relative = indices - start
        with h5py.File(source, "r") as h5:
            frames = h5["main"]
            fields = np.empty((len(indices), len(yidx), len(xidx)), dtype=np.float32)
            for i, frame_number in enumerate(info.frame_numbers[relative]):
                fields[i] = frames[_frame_key(frame_number)][::ystep, ::xstep]
        x, y, times = info.x[xidx], info.y[yidx], info.time[relative]
        z_unit, x_unit, y_unit = info.units["z"], info.units["x"], info.units["y"]
        title = lambda i: f"raw | timestep {indices[i]} | t = {times[i]:.6f} s"
        description = "raw calibrated heights"
    else:
        with h5py.File(source, "r") as h5:
            modes_ds = h5["pod/modes"]
            coeff_ds = h5["reduced/coefficients"]
            if stop > len(coeff_ds) or rank > len(modes_ds):
                raise ValueError(
                    f"Requested [{start}, {stop}) and rank {rank}; file has "
                    f"{len(coeff_ds)} timesteps and {len(modes_ds)} modes."
                )
            ny, nx = modes_ds.shape[1:]
            ystep = max(1, round((ny - 1) / (min(spatial_points, ny) - 1)))
            xstep = max(1, round((nx - 1) / (min(spatial_points, nx) - 1)))
            yidx = np.arange(0, ny, ystep)
            xidx = np.arange(0, nx, xstep)
            modes = np.asarray(modes_ds[:rank, ::ystep, ::xstep], dtype=np.float32)
            modes = np.ascontiguousarray(modes.reshape(rank, -1))
            coefficients = np.asarray(coeff_ds[indices, :rank], dtype=np.float32)
            fields = (coefficients @ modes).reshape(len(indices), len(yidx), len(xidx))
            if h5.attrs.get("temporal_mean_field_removed", False):
                fields += np.asarray(h5["preprocessing/temporal_mean_field"])[
                    np.ix_(yidx, xidx)
                ]
            if h5.attrs.get("instantaneous_spatial_mean_removed", False):
                fields += np.asarray(h5["preprocessing/frame_spatial_mean"][indices])[
                    :, None, None
                ]
            x = np.asarray(h5["grid/x"])[xidx]
            y = np.asarray(h5["grid/y"])[yidx]
            times = np.asarray(h5["grid/time"][indices])
            z_unit = h5["reduced/coefficients"].attrs.get("units", "")
            x_unit = h5["grid/x"].attrs.get("units", "")
            y_unit = h5["grid/y"].attrs.get("units", x_unit)
        title = lambda i: (f"{experiment} rep {rep} | rank {rank} | timestep {indices[i]} "
                           f"| t = {times[i]:.6f} s")
        description = f"rank {rank} mean-restored reconstruction"
    plot = make_surface_figure(
        x, y, padded_limits(fields), x_unit=x_unit, y_unit=y_unit, z_unit=z_unit,
        figsize=(7, 5), dpi=90,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.", suffix=".mp4", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
        "rgb24", "-s:v", "630x450", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(temporary_path),
    ]
    print(
        f"{source}\n{description}, source timesteps [{start}, {stop}), "
        f"rendering {len(indices)} frames on a {len(yidx)}x{len(xidx)} grid.",
        flush=True,
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    surface = None
    try:
        assert process.stdin is not None
        for i, field in enumerate(fields):
            if surface is not None:
                surface.remove()
            surface = draw_surface(plot, x, y, field)
            plot.axes.set_title(title(i), fontsize=10, pad=4)
            plot.canvas.draw()
            process.stdin.write(np.asarray(plot.canvas.buffer_rgba())[:, :, :3].tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error}")
        os.replace(temporary_path, output)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Saved {output} in {time.perf_counter() - began:.1f} s.", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, help="Raw keyed-frame HDF5 file; uses raw heights as stored.")
    parser.add_argument("--experiment", help="POD directory, e.g. 0p0 or 0p20")
    parser.add_argument("--rep", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--start", required=True, type=int, help="First timestep (inclusive)")
    parser.add_argument("--stop", required=True, type=int, help="Last timestep (exclusive)")
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spatial-points", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Sample this many timesteps at most (default: 300; raw limit: 1000)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.raw_input is None and (args.experiment is None or args.rep is None or args.rank is None):
        parser.error("POD video requires --experiment, --rep, and --rank, or use --raw-input.")
    if args.raw_input is not None and any(value is not None for value in (args.experiment, args.rep, args.rank)):
        parser.error("--raw-input cannot be combined with --experiment, --rep, or --rank.")
    make_video(
        args.experiment, args.rep, args.rank, args.start, args.stop,
        root=args.root, output=args.output, spatial_points=args.spatial_points,
        max_frames=args.max_frames, fps=args.fps, overwrite=args.overwrite,
        raw_input=args.raw_input,
    )


if __name__ == "__main__":
    main()
