"""Create an MP4 from a time segment of the calibrated 2-D HDF5 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


DEFAULT_DATA_PATH = Path(
    "/home/jonas/ucsd_thesis/11222025_j_0.04vpp_data_roi-none_cal-true.hdf5"
)


def _read_frame(
    frame_group: h5py.Group,
    frame_index: int,
    remove_spatial_mean: bool,
) -> np.ndarray:
    """Read one keyed HDF5 frame and optionally remove its spatial mean."""
    frame = np.asarray(frame_group[f"{frame_index:05d}"], dtype=np.float32)
    if remove_spatial_mean:
        frame = frame.copy()
        frame -= np.nanmean(frame, dtype=np.float64)
    return frame


def _color_limits(
    frame_group: h5py.Group,
    frame_indices: np.ndarray,
    remove_spatial_mean: bool,
    percentile: float,
    max_sample_frames: int,
) -> tuple[float, float]:
    """Estimate stable, robust color limits from frames across the segment."""
    sample_positions = np.linspace(
        0,
        len(frame_indices) - 1,
        min(max_sample_frames, len(frame_indices)),
        dtype=int,
    )
    sample_indices = frame_indices[sample_positions]
    first_sample = _read_frame(
        frame_group,
        int(sample_indices[0]),
        remove_spatial_mean,
    )
    sampled = np.empty((len(sample_indices), *first_sample.shape), dtype=np.float32)
    sampled[0] = first_sample
    for sample_number, frame_index in enumerate(sample_indices[1:], start=1):
        sampled[sample_number] = _read_frame(
            frame_group,
            int(frame_index),
            remove_spatial_mean,
        )
    if remove_spatial_mean:
        np.abs(sampled, out=sampled)
        color_limit = float(
            np.nanpercentile(sampled, percentile, overwrite_input=True)
        )
        vmin, vmax = -color_limit, color_limit
    else:
        lower_percentile = 100.0 - percentile
        vmin, vmax = np.nanpercentile(
            sampled,
            [lower_percentile, percentile],
            overwrite_input=True,
        )
        vmin, vmax = float(vmin), float(vmax)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError("Could not determine finite, ordered color limits.")
    return vmin, vmax


def make_capillary_video(
    start_index: int,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path | None = None,
    source_samples: int = 5_760,
    video_seconds: float = 60.0,
    fps: int = 30,
    remove_spatial_mean: bool = True,
    color_percentile: float = 99.0,
    color_sample_frames: int = 128,
    bitrate_kbps: int = 5_000,
    dpi: int = 120,
) -> Path:
    """Export a source-data segment as a time-stretched MP4.

    The HDF5 file stores each 200 x 200 frame as a separate dataset named
    ``/main/00000``, ``/main/00001``, and so on. Frames are therefore streamed
    one at a time instead of loading the full segment into memory.

    By default, 1,800 frames are selected approximately uniformly from the
    5,760-sample source window. At 30 fps this produces exactly 60 seconds of
    video. The experiment itself advances by only about 0.05 seconds.
    """
    data_path = Path(data_path).expanduser()
    start_index = int(start_index)
    source_samples = int(source_samples)
    fps = int(fps)

    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    if source_samples < 1:
        raise ValueError("source_samples must be at least 1.")
    if video_seconds <= 0 or fps < 1:
        raise ValueError("video_seconds and fps must both be positive.")
    if not 50 < color_percentile <= 100:
        raise ValueError("color_percentile must be in (50, 100].")
    if color_sample_frames < 1:
        raise ValueError("color_sample_frames must be at least 1.")

    stop_index = start_index + source_samples
    video_frame_count = max(1, round(video_seconds * fps))
    frame_indices = np.rint(
        np.linspace(start_index, stop_index - 1, video_frame_count)
    ).astype(np.int64)

    if output_path is None:
        output_path = Path(f"capillary_wave_start_{start_index:06d}.mp4")
    else:
        output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as h5_file:
        frame_group = h5_file["main"]
        time = np.asarray(h5_file["meta/t"], dtype=np.float64)
        x = np.asarray(h5_file["meta/x"], dtype=np.float32)
        y = np.asarray(h5_file["meta/y"], dtype=np.float32)

        if stop_index > len(time):
            raise IndexError(
                f"Requested [{start_index}, {stop_index}), but only "
                f"{len(time):,} frames are available."
            )
        for boundary in (start_index, stop_index - 1):
            if f"{boundary:05d}" not in frame_group:
                raise KeyError(f"Missing HDF5 frame /main/{boundary:05d}.")

        physical_seconds = float(time[stop_index - 1] - time[start_index])
        speed_factor = (
            video_seconds / physical_seconds if physical_seconds else np.inf
        )
        source_steps = np.diff(frame_indices)
        step_description = (
            f"{int(source_steps.min())}-{int(source_steps.max())}"
            if len(source_steps)
            else "0"
        )
        print(
            f"Source frames [{start_index:,}, {stop_index:,}) span "
            f"{1e3 * physical_seconds:.3f} ms.\n"
            f"Exporting {video_frame_count:,} frames at {fps} fps "
            f"({video_seconds:g} s), using source steps of {step_description}.\n"
            f"Playback is approximately {speed_factor:,.0f}x slower than the experiment."
        )

        vmin, vmax = _color_limits(
            frame_group,
            frame_indices,
            remove_spatial_mean,
            color_percentile,
            color_sample_frames,
        )
        print(f"Fixed color range: {vmin:.3f} to {vmax:.3f} microns")

        first_frame = _read_frame(
            frame_group,
            int(frame_indices[0]),
            remove_spatial_mean,
        )
        fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
        image = ax.imshow(
            first_frame,
            origin="lower",
            extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            interpolation="nearest",
        )
        ax.set_xlabel("x [microns]")
        ax.set_ylabel("y [microns]")
        ax.grid(False)
        title = ax.set_title("")
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label(
            "surface displacement - spatial mean [microns]"
            if remove_spatial_mean
            else "surface displacement [microns]"
        )
        fig.tight_layout()

        def update(video_frame: int):
            source_index = int(frame_indices[video_frame])
            image.set_data(
                _read_frame(frame_group, source_index, remove_spatial_mean)
            )
            experiment_ms = 1e3 * (time[source_index] - time[start_index])
            video_time = video_frame / fps
            title.set_text(
                f"source frame {source_index:,} | experiment +{experiment_ms:.3f} ms"
                f" | video {video_time:.1f} s"
            )
            return image, title

        animation = FuncAnimation(
            fig,
            update,
            frames=video_frame_count,
            interval=1_000 / fps,
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=bitrate_kbps,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        progress_interval = max(1, video_frame_count // 10)

        def show_progress(current_frame: int, total_frames: int) -> None:
            completed = current_frame + 1
            if (
                completed == 1
                or completed % progress_interval == 0
                or completed == total_frames
            ):
                print(f"Rendered {completed:,}/{total_frames:,} frames")

        try:
            animation.save(
                output_path,
                writer=writer,
                dpi=dpi,
                progress_callback=show_progress,
            )
        finally:
            plt.close(fig)

    print(f"Saved {output_path.resolve()}")
    return output_path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_index", type=int, help="first source frame")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-samples", type=int, default=5_760)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--keep-spatial-mean",
        action="store_true",
        help="do not subtract each frame's spatial mean",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_capillary_video(
        args.start_index,
        data_path=args.data,
        output_path=args.output,
        source_samples=args.source_samples,
        video_seconds=args.seconds,
        fps=args.fps,
        remove_spatial_mean=not args.keep_spatial_mean,
    )
