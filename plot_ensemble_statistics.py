"""Plot ensemble mean and covariance evolution across repeated experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np


POWER_LABELS = {
    "0p04": list("abcdefghijklmnop"),
    "0p07": list("abcdefghijklnop"),
    "0p08": list("abcefghijklnop"),
    "0p10": list("bcdefhijklmnopq"),
    "0p15": list("abcdefghijklmnop"),
    "0p18": list("abcdefghijkmnop"),
    "0p20": list("abcdefghijklmnop"),
    "0p25": list("abcdefghijklmno"),
    "0p30": list("abcdefghijklmnop"),
    "0p35": list("abcdefhijklmnop"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute time-dependent ensemble statistics across all repeated "
            "experiments for one power."
        )
    )
    parser.add_argument("--power", default="0p10", choices=POWER_LABELS)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jonas/ucsd_thesis/DHM_new_1Dcenter"),
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=20,
        help="Use every Nth time sample over the full experiment (default: 20).",
    )
    parser.add_argument(
        "--covariance-snapshots",
        type=int,
        default=6,
        help="Number of covariance matrices sampled across time (default: 6).",
    )
    parser.add_argument(
        "--mean-vmin",
        type=float,
        help="Optional lower limit for the ensemble-mean color scale.",
    )
    parser.add_argument(
        "--mean-vmax",
        type=float,
        help="Optional upper limit for the ensemble-mean color scale.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: ensemble_statistics_<power>).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def dataset_path(data_dir: Path, power: str, label: str) -> Path:
    return data_dir / power / f"Q_1D_{power}vpp_{label}.h5"


def save_field_plot(
    field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    title: str,
    colorbar_label: str,
    output: Path,
    dpi: int,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    image = ax.pcolormesh(
        t,
        x,
        field,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Experiment time [s]")
    ax.set_ylabel(r"x [$\mu$m]")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.time_stride < 1 or args.covariance_snapshots < 1:
        raise ValueError("--time-stride and --covariance-snapshots must be positive.")
    if (
        args.mean_vmin is not None
        and args.mean_vmax is not None
        and args.mean_vmin >= args.mean_vmax
    ):
        raise ValueError("--mean-vmin must be smaller than --mean-vmax.")

    labels = POWER_LABELS[args.power]
    paths = [dataset_path(args.data_dir, args.power, label) for label in labels]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset file: {missing[0]}")

    lengths = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            lengths.append(handle["Q_1D"].shape[1])
    common_length = min(lengths)

    with h5py.File(paths[0], "r") as handle:
        x = np.asarray(handle["x"])
        t = np.asarray(handle["t"][:common_length:args.time_stride])
        nx = handle["Q_1D"].shape[0]

    # Shape: (experiment, spatial position, time). Float32 limits memory use.
    states = np.empty((len(paths), nx, len(t)), dtype=np.float32)
    for index, (label, path) in enumerate(zip(labels, paths)):
        with h5py.File(path, "r") as handle:
            states[index] = handle["Q_1D"][
                :, :common_length:args.time_stride
            ]
        print(f"Loaded experiment {label} ({index + 1}/{len(paths)})", flush=True)

    # These are the diagonal statistics of C(t) across the experiment ensemble.
    ensemble_mean = np.mean(states, axis=0)
    pointwise_variance = np.var(states, axis=0, ddof=1)
    covariance_trace = np.sum(pointwise_variance, axis=0)
    spatial_mean = np.mean(ensemble_mean, axis=0)
    mean_pointwise_std = np.mean(np.sqrt(pointwise_variance), axis=0)

    output_dir = args.output_dir or Path(f"ensemble_statistics_{args.power}")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_field_plot(
        ensemble_mean,
        t,
        x,
        f"Power {args.power}: ensemble mean across {len(labels)} experiments",
        r"Mean surface displacement [$\mu$m]",
        output_dir / "mean_evolution.png",
        args.dpi,
        cmap="jet",
        vmin=args.mean_vmin,
        vmax=args.mean_vmax,
    )
    save_field_plot(
        pointwise_variance,
        t,
        x,
        f"Power {args.power}: pointwise ensemble variance",
        r"Surface-displacement variance [$\mu$m$^2$]",
        output_dir / "variance_evolution.png",
        args.dpi,
        cmap="magma",
        vmin=0.0,
    )

    # Full nx-by-nx covariance matrices at evenly spaced times.
    snapshot_indices = np.unique(
        np.linspace(0, len(t) - 1, args.covariance_snapshots, dtype=int)
    )
    covariances = [
        np.cov(states[:, :, index], rowvar=False, ddof=1)
        for index in snapshot_indices
    ]
    covariance_limit = max(float(np.nanmax(np.abs(cov))) for cov in covariances)
    columns = min(3, len(covariances))
    rows = int(np.ceil(len(covariances) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 4.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for ax, covariance, index in zip(axes.flat, covariances, snapshot_indices):
        image = ax.imshow(
            covariance,
            origin="lower",
            extent=(x[0], x[-1], x[0], x[-1]),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-covariance_limit,
            vmax=covariance_limit,
            rasterized=True,
        )
        ax.set_title(f"t = {t[index]:.6g} s")
        ax.set_xlabel(r"x$_2$ [$\mu$m]")
        ax.set_ylabel(r"x$_1$ [$\mu$m]")
    for ax in axes.flat[len(covariances) :]:
        ax.set_visible(False)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
        colorbar.set_label(r"Covariance [$\mu$m$^2$]")
    fig.suptitle(f"Power {args.power}: spatial covariance evolution", fontsize=15)
    fig.savefig(output_dir / "covariance_snapshots.png", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, spatial_mean)
    axes[0].set_ylabel(r"Spatial mean [$\mu$m]")
    axes[1].plot(t, mean_pointwise_std)
    axes[1].set_ylabel(r"Mean pointwise std. [$\mu$m]")
    axes[2].plot(t, covariance_trace)
    axes[2].set_ylabel(r"trace($C$) [$\mu$m$^2$]")
    axes[2].set_xlabel("Experiment time [s]")
    fig.suptitle(f"Power {args.power}: ensemble-statistics summary")
    fig.savefig(output_dir / "statistics_summary.png", dpi=args.dpi)
    plt.close(fig)

    print(f"Saved statistics to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
