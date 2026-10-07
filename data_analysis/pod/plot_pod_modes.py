"""Save the first r spatial POD modes as individual image files.

Each mode is read directly from a ``pod_2d.py`` HDF5 output, plotted on the
saved physical x/y grid, and labeled with its singular value.

Example
-------
    conda run -n capillarywave python plot_pod_modes.py \
        pod_2d_r5000_0p20.h5 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="POD HDF5 file.")
    parser.add_argument(
        "count",
        type=int,
        help="Number of leading POD modes to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: directory containing the HDF5 file).",
    )
    parser.add_argument(
        "--prefix",
        help="Output filename prefix (default: input filename stem).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output image format (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster output resolution (default: 200 dpi).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace mode images that already exist.",
    )
    return parser.parse_args()


def _validate_pod_file(
    handle: h5py.File,
    path: Path,
) -> tuple[h5py.Dataset, h5py.Dataset, np.ndarray, np.ndarray, str, str]:
    for dataset_name in (
        "pod/modes",
        "pod/singular_values",
        "grid/x",
        "grid/y",
    ):
        if dataset_name not in handle:
            raise KeyError(f"Missing dataset {dataset_name!r} in {path}.")

    modes = handle["pod/modes"]
    singular_values = handle["pod/singular_values"]
    x_dataset = handle["grid/x"]
    y_dataset = handle["grid/y"]
    x = np.asarray(x_dataset, dtype=np.float64)
    y = np.asarray(y_dataset, dtype=np.float64)
    x_units = str(x_dataset.attrs.get("units", ""))
    y_units = str(y_dataset.attrs.get("units", ""))

    if modes.ndim != 3:
        raise ValueError(
            f"pod/modes must have shape (mode, y, x); got {modes.shape}."
        )
    if singular_values.ndim != 1 or len(singular_values) != len(modes):
        raise ValueError(
            "pod/singular_values must be one-dimensional and have one value "
            f"per mode; got {singular_values.shape} for {modes.shape}."
        )
    if modes.shape[1:] != (len(y), len(x)):
        raise ValueError(
            f"Mode shape {modes.shape[1:]} does not match y/x grid lengths "
            f"{(len(y), len(x))}."
        )
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("grid/x and grid/y must be one-dimensional.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("The spatial grids contain NaN or infinite values.")

    return modes, singular_values, x, y, x_units, y_units


def _axis_label(name: str, units: str) -> str:
    return f"{name} [{units}]" if units else name


def _scientific_mathtext(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return rf"{mantissa:.6f}\times 10^{{{exponent}}}"


def _output_paths(
    output_directory: Path,
    prefix: str,
    count: int,
    stored_modes: int,
    image_format: str,
) -> list[Path]:
    width = max(4, len(str(stored_modes)))
    return [
        output_directory / f"{prefix}_mode_{mode_number:0{width}d}.{image_format}"
        for mode_number in range(1, count + 1)
    ]


def save_mode_figure(
    mode: np.ndarray,
    singular_value: float,
    mode_number: int,
    x: np.ndarray,
    y: np.ndarray,
    x_units: str,
    y_units: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot and save one spatial POD mode with a symmetric color range."""
    if not np.isfinite(mode).all():
        raise ValueError(f"POD mode {mode_number} contains nonfinite values.")
    if not np.isfinite(singular_value) or singular_value < 0:
        raise ValueError(
            f"POD mode {mode_number} has invalid singular value {singular_value}."
        )

    color_limit = float(np.max(np.abs(mode)))
    if color_limit <= 0:
        raise ValueError(f"POD mode {mode_number} is identically zero.")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(
        mode,
        origin="lower",
        extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel(_axis_label("x", x_units))
    ax.set_ylabel(_axis_label("y", y_units))
    ax.set_title(
        f"POD mode {mode_number}\n"
        rf"singular value $\sigma_{{{mode_number}}}="
        rf"{_scientific_mathtext(singular_value)}$"
    )
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("POD mode value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"POD file does not exist: {input_path}")
    if args.count < 1:
        raise ValueError("count must be positive.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")

    output_directory = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else input_path.parent
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem

    with h5py.File(input_path, "r") as handle:
        modes, singular_values, x, y, x_units, y_units = _validate_pod_file(
            handle, input_path
        )
        stored_modes = len(modes)
        if args.count > stored_modes:
            raise ValueError(
                f"Requested {args.count:,} modes, but {input_path} contains "
                f"only {stored_modes:,}."
            )

        output_paths = _output_paths(
            output_directory,
            prefix,
            args.count,
            stored_modes,
            args.format,
        )
        existing_paths = [path for path in output_paths if path.exists()]
        if existing_paths and not args.overwrite:
            example = existing_paths[0]
            raise FileExistsError(
                f"{len(existing_paths):,} output file(s) already exist, including "
                f"{example}. Use --overwrite to replace them."
            )

        report_every = max(1, args.count // 10)
        for mode_index, output_path in enumerate(output_paths):
            mode_number = mode_index + 1
            save_mode_figure(
                np.asarray(modes[mode_index], dtype=np.float32),
                float(singular_values[mode_index]),
                mode_number,
                x,
                y,
                x_units,
                y_units,
                output_path,
                args.dpi,
            )
            if (
                mode_number == 1
                or mode_number % report_every == 0
                or mode_number == args.count
            ):
                print(
                    f"Saved {mode_number:,}/{args.count:,}: {output_path}",
                    flush=True,
                )

    print(
        f"Finished: {args.count:,} mode image(s) written to "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()
