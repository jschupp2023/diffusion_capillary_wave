"""Save one raw calibrated capillary-wave frame as a high-resolution PNG.

The requested timestep is the zero-based position in ``/meta/t``, matching the
``--start`` convention in ``pod_surface_video.py``. The complete stored spatial
grid is rendered; no spatial downsampling is performed.

Example
-------
python data_analysis/pod/raw_surface_snapshot.py \\
    --raw-input /path/to/recording_cal-true.hdf5 --timestep 23000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np

if __package__:
    from .pod_2d import _frame_key, inspect_input
    from .surface_visualization import draw_surface, make_surface_figure, padded_limits
else:
    from pod_2d import _frame_key, inspect_input
    from surface_visualization import draw_surface, make_surface_figure, padded_limits


def save_raw_snapshot(
    raw_input: Path,
    timestep: int,
    *,
    output: Path | None = None,
    dpi: int = 300,
    width: float = 8.0,
    height: float = 6.0,
    elev: float = 30.0,
    azim: float = -60.0,
    zmin: float | None = None,
    zmax: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Read and render one full-resolution raw HDF5 frame."""
    if timestep < 0:
        raise ValueError("timestep must be nonnegative.")
    if dpi < 1 or width <= 0 or height <= 0:
        raise ValueError("dpi, width, and height must be positive.")

    source = Path(raw_input).expanduser().resolve()
    if output is None:
        output = Path("images") / f"raw_surface_{source.stem}_timestep_{timestep}.png"
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")

    info = inspect_input(source, timestep, timestep + 1)
    frame_number = int(info.frame_numbers[0])
    with h5py.File(source, "r") as h5:
        field = np.asarray(h5["main"][_frame_key(frame_number)], dtype=np.float64)
    if field.shape != info.frame_shape:
        raise ValueError(
            f"Frame /main/{_frame_key(frame_number)} has shape {field.shape}; "
            f"expected {info.frame_shape}."
        )

    automatic_min, automatic_max = padded_limits(field)
    limits = (
        automatic_min if zmin is None else float(zmin),
        automatic_max if zmax is None else float(zmax),
    )
    plot = make_surface_figure(
        info.x,
        info.y,
        limits,
        x_unit=info.units["x"],
        y_unit=info.units["y"],
        z_unit=info.units["z"],
        figsize=(width, height),
        dpi=dpi,
        elev=elev,
        azim=azim,
    )
    draw_surface(plot, info.x, info.y, field, full_resolution=True)
    time_unit = info.units["time"] or "s"
    frame_suffix = "" if frame_number == timestep else f" | frame {frame_number}"
    plot.axes.set_title(
        f"raw | timestep {timestep}{frame_suffix} | "
        f"t = {info.time[0]:.6f} {time_unit}",
        fontsize=10,
        pad=4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    plot.figure.savefig(output, dpi=dpi)
    print(
        f"Saved {output}\n"
        f"Source: {source}\n"
        f"Timestep: {timestep:,}; HDF5 frame: {_frame_key(frame_number)}; "
        f"time: {info.time[0]:.9g} {time_unit}\n"
        f"Spatial grid: {field.shape[1]} x {field.shape[0]} (full resolution); "
        f"image: {round(width * dpi)} x {round(height * dpi)} px"
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-input", type=Path, required=True, help="Raw keyed-frame HDF5 file."
    )
    parser.add_argument(
        "--timestep",
        type=int,
        required=True,
        help="Zero-based position in /meta/t (the same indexing used by the video CLI).",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300).")
    parser.add_argument(
        "--width", type=float, default=8.0, help="Figure width in inches (default: 8)."
    )
    parser.add_argument(
        "--height", type=float, default=6.0, help="Figure height in inches (default: 6)."
    )
    parser.add_argument(
        "--elev", type=float, default=30.0, help="Camera elevation in degrees."
    )
    parser.add_argument(
        "--azim", type=float, default=-60.0, help="Camera azimuth in degrees."
    )
    parser.add_argument(
        "--zmin", type=float, help="Lower color and z-axis limit (default: frame range)."
    )
    parser.add_argument(
        "--zmax", type=float, help="Upper color and z-axis limit (default: frame range)."
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    save_raw_snapshot(
        args.raw_input,
        args.timestep,
        output=args.output,
        dpi=args.dpi,
        width=args.width,
        height=args.height,
        elev=args.elev,
        azim=args.azim,
        zmin=args.zmin,
        zmax=args.zmax,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
