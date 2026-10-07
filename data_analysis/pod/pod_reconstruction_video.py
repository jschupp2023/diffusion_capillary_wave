"""Create a fast MP4 of reconstructed preprocessed POD dynamics.

The reconstruction is formed directly from the saved reduced coordinates,

    reconstructed = reduced/coefficients @ pod/modes,

without restoring either the temporal mean field or the instantaneous spatial
mean. The default selects 300 frames near the middle of the data, separated by
five source samples, and encodes them at 30 fps to make a 10-second video.

Examples
--------
    python pod_reconstruction_video.py pod_2d_r1000.h5

    python pod_reconstruction_video.py pod_2d_r1000.h5 \
        --seconds 10 --fps 30 --stride 5 --overwrite
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="POD HDF5 result file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 (default: <input_stem>_reconstruction.mp4).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Video playback duration (default: 10 seconds).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Encoded frames per second (default: 30).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Source-frame step between video frames (default: 5).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="First source index; default centers the selected segment.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="Reconstruction rank (default: all stored POD modes).",
    )
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=99.5,
        help=(
            "Symmetric color limit percentile of absolute displacement "
            "(default: 99.5)."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Integer nearest-neighbor output scaling (default: 2).",
    )
    parser.add_argument(
        "--cmap",
        default="RdBu_r",
        help="Matplotlib color map (default: RdBu_r).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality, where lower is higher quality (default: 18).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output video.",
    )
    return parser.parse_args()


def _validate_options(
    seconds: float,
    fps: int,
    stride: int,
    start_index: int | None,
    rank: int | None,
    color_percentile: float,
    scale: int,
    crf: int,
) -> None:
    if seconds <= 0:
        raise ValueError("Video duration must be positive.")
    if fps < 1:
        raise ValueError("FPS must be positive.")
    if stride < 1:
        raise ValueError("Frame stride must be positive.")
    if start_index is not None and start_index < 0:
        raise ValueError("Start index must be non-negative.")
    if rank is not None and rank < 1:
        raise ValueError("Reconstruction rank must be positive.")
    if not 50 < color_percentile <= 100:
        raise ValueError("Color percentile must lie in (50, 100].")
    if scale < 1:
        raise ValueError("Output scale must be positive.")
    if not 0 <= crf <= 51:
        raise ValueError("CRF must lie between 0 and 51.")


def _select_frame_range(
    n_frames: int,
    video_frame_count: int,
    stride: int,
    start_index: int | None,
) -> tuple[int, int]:
    source_span = (video_frame_count - 1) * stride + 1
    if source_span > n_frames:
        raise ValueError(
            f"A {video_frame_count}-frame video with stride {stride} needs "
            f"{source_span:,} source frames, but only {n_frames:,} are stored."
        )
    if start_index is None:
        start_index = (n_frames - source_span) // 2
    stop_index = start_index + source_span
    if stop_index > n_frames:
        raise ValueError(
            f"Selected source range [{start_index:,}, {stop_index:,}) exceeds "
            f"the {n_frames:,} stored frames."
        )
    return start_index, stop_index


def _encode_rgb_frames(
    reconstructed: np.ndarray,
    output_path: Path,
    fps: int,
    scale: int,
    color_limit: float,
    cmap_name: str,
    colorbar_label: str,
    crf: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to create MP4 files but was not found on PATH."
        )
    try:
        color_map = matplotlib.colormaps[cmap_name]
    except KeyError as exc:
        raise ValueError(f"Unknown Matplotlib color map: {cmap_name}") from exc

    frame_count, height, width = reconstructed.shape
    colorbar_panel = _make_colorbar_panel(
        height,
        color_map,
        color_limit,
        colorbar_label,
    )
    encoded_width = width + colorbar_panel.shape[1]
    scaled_width = encoded_width * scale
    scaled_height = height * scale
    video_filter = (
        f"scale={scaled_width}:{scaled_height}:flags=neighbor,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{encoded_width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    normalization_scale = 1.0 / (2.0 * color_limit)
    report_every = max(1, frame_count // 10)

    try:
        for frame_number, frame in enumerate(reconstructed, start=1):
            normalized = (frame + color_limit) * normalization_scale
            np.clip(normalized, 0.0, 1.0, out=normalized)
            rgb = color_map(normalized, bytes=True)[..., :3]
            frame_with_colorbar = np.concatenate(
                (np.flipud(rgb), colorbar_panel), axis=1
            )
            process.stdin.write(frame_with_colorbar.tobytes())
            if (
                frame_number == 1
                or frame_number % report_every == 0
                or frame_number == frame_count
            ):
                print(
                    f"Encoded {frame_number:,}/{frame_count:,} frames",
                    flush=True,
                )
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.stdin.close()
        process.terminate()
        process.wait()
        raise

    if return_code != 0:
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {error}")


def _make_colorbar_panel(
    height: int,
    color_map: matplotlib.colors.Colormap,
    color_limit: float,
    label: str,
) -> np.ndarray:
    """Render one compact colorbar panel that can be reused for every frame."""
    width = 110
    dpi = 100
    figure = Figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor="white",
    )
    canvas = FigureCanvasAgg(figure)
    colorbar_axis = figure.add_axes((0.12, 0.08, 0.20, 0.84))
    mappable = ScalarMappable(
        norm=Normalize(vmin=-color_limit, vmax=color_limit),
        cmap=color_map,
    )
    colorbar = figure.colorbar(
        mappable,
        cax=colorbar_axis,
        orientation="vertical",
        format="%.3g",
    )
    colorbar.ax.tick_params(labelsize=6, pad=2, length=2)
    colorbar.set_label(label, fontsize=6, labelpad=3)
    canvas.draw()
    panel = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    if panel.shape != (height, width, 3):
        raise RuntimeError(
            f"Unexpected rendered colorbar shape {panel.shape}; expected "
            f"{(height, width, 3)}."
        )
    return panel


def make_reconstruction_video(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    seconds: float = 10.0,
    fps: int = 30,
    stride: int = 5,
    start_index: int | None = None,
    rank: int | None = None,
    color_percentile: float = 99.5,
    scale: int = 2,
    cmap: str = "RdBu_r",
    crf: int = 18,
    overwrite: bool = False,
) -> Path:
    """Reconstruct selected centered POD states and encode them as an MP4."""
    _validate_options(
        seconds,
        fps,
        stride,
        start_index,
        rank,
        color_percentile,
        scale,
        crf,
    )
    input_path = Path(input_path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"POD file does not exist: {input_path}")
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_reconstruction.mp4"
        )
    else:
        output_path = Path(output_path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output video already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with h5py.File(input_path, "r") as handle:
        for dataset_name in ("pod/modes", "reduced/coefficients", "grid/time"):
            if dataset_name not in handle:
                raise KeyError(f"Missing dataset {dataset_name!r} in {input_path}.")
        mode_dataset = handle["pod/modes"]
        coefficient_dataset = handle["reduced/coefficients"]
        time_values = handle["grid/time"]
        units_value = coefficient_dataset.attrs.get("units", "")
        if isinstance(units_value, bytes):
            units = units_value.decode("utf-8", errors="replace")
        else:
            units = str(units_value)
        colorbar_label = f"Preprocessed [{units}]" if units else "Preprocessed value"
        if mode_dataset.ndim != 3:
            raise ValueError(
                f"pod/modes must have shape (mode, y, x); got {mode_dataset.shape}."
            )
        if coefficient_dataset.ndim != 2:
            raise ValueError(
                "reduced/coefficients must have shape (time, mode); got "
                f"{coefficient_dataset.shape}."
            )
        if coefficient_dataset.shape[1] != len(mode_dataset):
            raise ValueError(
                "Coefficient and mode counts differ: "
                f"{coefficient_dataset.shape[1]} versus {len(mode_dataset)}."
            )
        if len(time_values) != len(coefficient_dataset):
            raise ValueError(
                f"Time and coefficient lengths differ: {len(time_values)} versus "
                f"{len(coefficient_dataset)}."
            )

        reconstruction_rank = len(mode_dataset) if rank is None else rank
        if reconstruction_rank > len(mode_dataset):
            raise ValueError(
                f"Requested rank {reconstruction_rank}, but only "
                f"{len(mode_dataset)} modes are stored."
            )
        video_frame_count = max(1, round(seconds * fps))
        first, stop = _select_frame_range(
            len(coefficient_dataset),
            video_frame_count,
            stride,
            start_index,
        )
        source_indices = first + np.arange(video_frame_count) * stride

        # Read one contiguous HDF5 range for efficient chunk access, then apply
        # the requested stride in memory.
        coefficient_block = np.asarray(
            coefficient_dataset[first:stop, :reconstruction_rank],
            dtype=np.float32,
        )
        coefficients = np.ascontiguousarray(coefficient_block[::stride])
        modes = np.asarray(
            mode_dataset[:reconstruction_rank], dtype=np.float32
        ).reshape(reconstruction_rank, -1)
        if not np.isfinite(coefficients).all() or not np.isfinite(modes).all():
            raise ValueError("POD coefficients or modes contain nonfinite values.")

        physical_span = float(
            time_values[int(source_indices[-1])] - time_values[int(source_indices[0])]
        )
        print(
            f"Input: {input_path.resolve()}\n"
            f"Using source indices {first:,} through {int(source_indices[-1]):,} "
            f"with stride {stride} ({1e3 * physical_span:.3f} ms physical time).\n"
            f"Reconstructing {video_frame_count:,} frames at rank "
            f"{reconstruction_rank:,} for a {seconds:g} s, {fps} fps video.\n"
            "Mean fields are not restored.",
            flush=True,
        )

        reconstruction_started = time.perf_counter()
        reconstructed = (coefficients @ modes).reshape(
            video_frame_count, *mode_dataset.shape[1:]
        )
        print(
            f"Reconstruction completed in "
            f"{time.perf_counter() - reconstruction_started:.2f} s.",
            flush=True,
        )

    sample_count = min(64, video_frame_count)
    sample_indices = np.linspace(
        0, video_frame_count - 1, sample_count, dtype=np.int64
    )
    color_limit = float(
        np.percentile(
            np.abs(reconstructed[sample_indices]),
            color_percentile,
        )
    )
    if not np.isfinite(color_limit) or color_limit <= 0:
        raise ValueError("Could not determine a finite positive color limit.")
    print(
        f"Fixed symmetric color range: {-color_limit:.6g} to "
        f"{color_limit:.6g}",
        flush=True,
    )

    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.mp4",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        _encode_rgb_frames(
            reconstructed,
            temporary_path,
            fps,
            scale,
            color_limit,
            cmap,
            colorbar_label,
            crf,
        )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(
        f"Saved {output_path.resolve()} in {time.perf_counter() - started:.2f} s.",
        flush=True,
    )
    return output_path.resolve()


def main() -> None:
    args = parse_args()
    make_reconstruction_video(
        args.input,
        output_path=args.output,
        seconds=args.seconds,
        fps=args.fps,
        stride=args.stride,
        start_index=args.start_index,
        rank=args.rank,
        color_percentile=args.color_percentile,
        scale=args.scale,
        cmap=args.cmap,
        crf=args.crf,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
