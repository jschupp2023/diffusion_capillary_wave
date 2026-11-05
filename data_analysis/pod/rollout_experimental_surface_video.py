"""Render matched experimental, rank-limited reference, and rollout surfaces."""

from __future__ import annotations

import argparse
import json
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

from data_analysis.pod.pod_2d import _frame_key, inspect_input
from data_analysis.pod.surface_visualization import (
    SurfaceFigure,
    draw_surface,
    padded_limits,
)
from data_analysis.rollout import Rollout
from modelling.data_preparation.prepare_shared_pod_training import (
    spatial_mean_dataset_path,
)


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_fields(
    rollout: Rollout,
    raw_path: Path,
    pod_path: Path,
    condition: int,
    ensemble: int,
    steps: int,
    spatial_points: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict
]:
    if rollout.metadata.get("temporal_preprocessing", {}).get("downsample_factor", 1) != 1:
        raise ValueError("Raw experimental video does not yet apply the saved temporal anti-alias filter. "
                         "Prepared rollout indices are not native source-frame indices.")
    if rollout.start_index is None:
        raise ValueError("Rollout does not contain source start indices.")
    if not rollout.spatial_mean_available:
        raise ValueError("Experimental surface comparison requires the physical spatial mean.")
    if not 0 <= condition < len(rollout.reference):
        raise ValueError("Condition is outside the rollout.")
    if not 0 <= ensemble < rollout.generated.shape[1]:
        raise ValueError("Ensemble member is outside the rollout.")
    if rollout.repetitions[condition] != pod_path.parent.name:
        raise ValueError(
            f"Condition {condition} uses {rollout.repetitions[condition]}, while "
            f"the POD input represents {pod_path.parent.name}."
        )
    if rollout.include_spatial_mean and rollout.spatial_mean_highpass_hz is None:
        raise ValueError("The rollout does not use a spatial-mean high-pass filter.")
    count = min(steps, rollout.generated.shape[2])
    ny, nx = rollout.modes.shape[1:]
    yidx = np.unique(np.linspace(0, ny - 1, min(spatial_points, ny), dtype=int))
    xidx = np.unique(np.linspace(0, nx - 1, min(spatial_points, nx), dtype=int))
    sampled_modes = np.asarray(rollout.modes[:, yidx][:, :, xidx], dtype=np.float32)

    reference = rollout.reconstruct(
        np.asarray(rollout.reference[condition, :count], dtype=np.float32),
        sampled_modes)
    generated = rollout.reconstruct(
        np.asarray(rollout.generated[condition, ensemble, :count], dtype=np.float32),
        sampled_modes)

    raw_info = inspect_input(raw_path, 0, None)
    with h5py.File(pod_path, "r") as pod:
        recorded_raw = Path(_text(pod.attrs.get("source_file", ""))).name
        if recorded_raw and recorded_raw != raw_path.name:
            raise ValueError(f"POD source is {recorded_raw}, not {raw_path.name}.")
        mean_path = (spatial_mean_dataset_path(rollout.spatial_mean_highpass_hz)
                     if rollout.include_spatial_mean else None)
        required = [
            "source/frame_number",
            "grid/time",
            "preprocessing/temporal_mean_field",
        ]
        if rollout.include_spatial_mean:
            required.extend(("preprocessing/frame_spatial_mean", mean_path))
        missing = [name for name in required if name not in pod]
        if missing:
            raise KeyError(f"POD file is missing: {', '.join(missing)}")
        frame_numbers = np.asarray(pod["source/frame_number"], dtype=np.int64)
        source_time = np.asarray(pod["grid/time"], dtype=np.float64)
        original_mean = (np.asarray(
            pod["preprocessing/frame_spatial_mean"], dtype=np.float32)
            if rollout.include_spatial_mean else None)
        filtered_mean = (np.asarray(pod[mean_path], dtype=np.float32)
                         if rollout.include_spatial_mean else None)
        temporal_mean = np.asarray(
            pod["preprocessing/temporal_mean_field"], dtype=np.float32
        )[np.ix_(yidx, xidx)]
    if raw_info.frame_shape != (ny, nx):
        raise ValueError("Raw, local POD, and shared-POD field shapes do not match.")

    positions = rollout.start_index[condition] + (
        np.arange(count, dtype=np.int64) * rollout.lag_steps
    )
    if positions[-1] >= len(frame_numbers):
        raise IndexError("Requested video extends beyond the experimental source.")
    if not np.allclose(
        rollout.time[condition, :count], source_time[positions], rtol=0, atol=2e-7
    ):
        raise ValueError("Rollout and experimental time windows do not align.")

    experimental = np.empty((count, len(yidx), len(xidx)), dtype=np.float32)
    with h5py.File(raw_path, "r") as raw:
        frames = raw["main"]
        for output_index, source_position in enumerate(positions):
            key = _frame_key(frame_numbers[source_position])
            if key not in frames:
                raise KeyError(f"Missing raw frame /main/{key}.")
            field = np.asarray(frames[key], dtype=np.float32)[np.ix_(yidx, xidx)]
            experimental[output_index] = field - temporal_mean
            if rollout.include_spatial_mean:
                experimental[output_index] += (
                    filtered_mean[source_position] - original_mean[source_position])

    if rollout.include_spatial_mean:
        expected_mean = filtered_mean[positions]
        reference_mean = rollout.spatial_mean(rollout.reference[condition, :count])
        if not np.allclose(reference_mean, expected_mean, rtol=2e-5, atol=2e-4):
            raise ValueError("Reference spatial mean does not match the experimental source.")

    relative_time = rollout.time[condition, :count] - rollout.time[condition, 0]
    report = {
        "rollout": str(rollout.path),
        "raw_input": str(raw_path),
        "pod_input": str(pod_path),
        "repetition": str(rollout.repetitions[condition]),
        "condition": condition,
        "ensemble": ensemble,
        "source_start_index": int(rollout.start_index[condition]),
        "frames": count,
        "spatial_grid": [int(len(yidx)), int(len(xidx))],
        "spatial_mean_highpass_hz": rollout.spatial_mean_highpass_hz,
        "spatial_mean_representation": rollout.spatial_mean_representation,
        "experimental_preprocessing": (
            "raw field - original 2-D frame spatial mean - saved temporal mean "
            f"+ saved {rollout.spatial_mean_highpass_hz:g} Hz high-pass 2-D spatial mean"
            if rollout.include_spatial_mean else
            "cleaned raw field - saved temporal mean; spatial mean retained in POD modes"
        ),
        "shared_color_and_height_limits": True,
        "temporal_mean_restored": False,
    }
    return experimental, reference, generated, rollout.x[xidx], rollout.y[yidx], relative_time, report


def _render(
    fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    output: Path,
    *,
    fps: int,
    units: tuple[str, str, str],
    title: str,
    generated_label: str,
    overwrite: bool,
) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write MP4 video.")
    zlim = padded_limits(np.stack(fields))
    norm = Normalize(vmin=zlim[0], vmax=zlim[1])
    cmap = matplotlib.colormaps["viridis"]
    figure = Figure(figsize=(14, 5), dpi=90, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axes = tuple(
        figure.add_axes((left, 0.12, 0.27, 0.74), projection="3d")
        for left in (0.01, 0.325, 0.64)
    )
    plots = []
    x_unit, y_unit, z_unit = units
    for axis in axes:
        axis.view_init(elev=30, azim=-60)
        axis.set_box_aspect((1, 1, 0.48))
        axis.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=zlim)
        axis.set_xlabel(f"x [{x_unit}]", labelpad=2)
        axis.set_ylabel(f"y [{y_unit}]", labelpad=2)
        axis.tick_params(labelsize=7, pad=0)
        plots.append(SurfaceFigure(figure, canvas, axis, norm, cmap))
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=figure.add_axes((0.925, 0.2, 0.012, 0.58)),
    )
    colorbar.set_label(f"height [{z_unit}]", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    figure.suptitle(title, fontsize=11, y=0.97)

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
    surfaces = [None, None, None]
    labels = ("Experimental", "rank-20 POD reference", generated_label)
    began = time.perf_counter()
    try:
        assert process.stdin is not None
        for frame_index, frame_fields in enumerate(zip(*fields)):
            for panel, (plot, field, label) in enumerate(zip(plots, frame_fields, labels)):
                if surfaces[panel] is not None:
                    surfaces[panel].remove()
                surfaces[panel] = draw_surface(plot, x, y, field)
                plot.axes.set_title(
                    f"{label}\nframe {frame_index} | t = {1e3 * times[frame_index]:.3f} ms",
                    fontsize=9,
                    pad=3,
                )
            canvas.draw()
            process.stdin.write(np.asarray(canvas.buffer_rgba())[:, :, :3].tobytes())
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--pod-input", type=Path, required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--ensemble", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--spatial-points", type=int, default=30)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--generated-label", default="Neural SDE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.spatial_points < 3 or args.fps < 1:
        parser.error("--steps/--fps must be positive and --spatial-points at least 3.")

    rollout = Rollout(args.rollout)
    raw_path = args.raw_input.expanduser().resolve()
    pod_path = args.pod_input.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else rollout.path.parent / "surface_video_experimental" / "comparison.mp4"
    )
    experimental, reference, generated, x, y, times, report = _load_fields(
        rollout,
        raw_path,
        pod_path,
        args.condition,
        args.ensemble,
        args.steps,
        args.spatial_points,
    )
    _render(
        (experimental, reference, generated),
        x,
        y,
        times,
        output,
        fps=args.fps,
        units=(rollout.x_units, rollout.y_units, rollout.units),
        title=(
            f"{report['repetition']} | condition {args.condition}, ensemble {args.ensemble} "
            f"| 50 Hz high-pass spatial mean"
        ),
        generated_label=args.generated_label,
        overwrite=args.overwrite,
    )
    report["fps"] = args.fps
    report["duration_seconds"] = report["frames"] / args.fps
    report["generated_label"] = args.generated_label
    report["output"] = str(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
