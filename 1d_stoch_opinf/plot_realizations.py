"""Plot 1D frames, complete recordings, or 5760-sample realizations."""

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
        description="Plot 1D frames, complete recordings, or segmented realizations."
    )
    parser.add_argument("--power", default="0p10", choices=POWER_LABELS)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jonas/ucsd_thesis/DHM_new_1Dcenter"),
    )
    parser.add_argument("--samples", type=int, default=5760)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--whole-experiment",
        action="store_true",
        help="Plot each complete recording instead of splitting it into segments.",
    )
    mode.add_argument(
        "--timestep",
        type=int,
        help="Plot one spatial frame at this zero-based timestep.",
    )
    parser.add_argument(
        "--experiment-label",
        default="a",
        help="Repeated-experiment label used with --timestep (default: a).",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=20,
        help="In whole-experiment mode, display every Nth time sample (default: 20).",
    )
    parser.add_argument("--vmin", type=float, default=50.0)
    parser.add_argument("--vmax", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default depends on the selected plotting mode).",
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vmin >= args.vmax:
        raise ValueError("--vmin must be smaller than --vmax.")
    if args.samples < 1 or args.time_stride < 1:
        raise ValueError("--samples and --time-stride must be positive.")
    if args.timestep is not None and args.timestep < 0:
        raise ValueError("--timestep must be nonnegative.")
    labels = POWER_LABELS[args.power]
    default_dir = (
        f"frames_{args.power}"
        if args.timestep is not None
        else (
            f"whole_experiments_{args.power}"
            if args.whole_experiment
            else f"realizations_{args.power}"
        )
    )
    output_dir = args.output_dir or Path(default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        label: args.data_dir / args.power / f"Q_1D_{args.power}vpp_{label}.h5"
        for label in labels
    }

    if args.timestep is not None:
        if args.experiment_label not in labels:
            raise ValueError(
                f"Experiment label {args.experiment_label!r} is not available for "
                f"power {args.power}; choose from {', '.join(labels)}."
            )
        path = paths[args.experiment_label]
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        with h5py.File(path, "r") as handle:
            data = handle["Q_1D"]
            if args.timestep >= data.shape[1]:
                raise ValueError(
                    f"--timestep {args.timestep} is outside the available range "
                    f"0 to {data.shape[1] - 1}."
                )
            state = np.asarray(data[:, args.timestep])
            time = float(handle["t"][args.timestep])
            x = np.asarray(handle["x"])

        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(x, state, color="tab:blue", linewidth=1.5)
        ax.set_title(
            f"Power {args.power} — experiment {args.experiment_label}, "
            f"timestep {args.timestep} (t = {time:.9g} s)"
        )
        ax.set_xlabel(r"$x$ [$\mu$m]")
        ax.set_ylabel(r"Surface displacement [$\mu$m]")
        ax.grid(alpha=0.25)

        filename = (
            f"frame_{args.power}_experiment_{args.experiment_label}_"
            f"timestep_{args.timestep:06d}.png"
        )
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved frame plot to {output_path.resolve()}")
        return

    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset file: {missing[0]}")

    if args.whole_experiment:
        total = len(labels)
        for experiment, (label, path) in enumerate(paths.items(), start=1):
            with h5py.File(path, "r") as handle:
                state = np.asarray(handle["Q_1D"][:, :: args.time_stride])
                t = np.asarray(handle["t"][:: args.time_stride])
                x = np.asarray(handle["x"])

            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            image = ax.pcolormesh(
                t,
                x,
                state,
                shading="auto",
                cmap="jet",
                vmin=args.vmin,
                vmax=args.vmax,
                rasterized=True,
            )
            ax.set_title(
                f"Power {args.power} — complete experiment {label} "
                f"({experiment}/{total})"
            )
            ax.set_xlabel("Experiment time [s]")
            ax.set_ylabel(r"x [$\mu$m]")
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label(r"Surface displacement [$\mu$m]")

            filename = f"experiment_{experiment:02d}_{label}_complete.png"
            fig.savefig(output_dir / filename, dpi=args.dpi)
            plt.close(fig)
            print(f"[{experiment}/{total}] {filename}", flush=True)

        print(f"Saved {total} complete-experiment plots to {output_dir.resolve()}")
        return

    segment_counts = {}
    for label, path in paths.items():
        with h5py.File(path, "r") as handle:
            data = handle["Q_1D"]
            segment_counts[label] = data.shape[1] // args.samples

    total = sum(segment_counts.values())
    realization = 0
    for label, path in paths.items():
        with h5py.File(path, "r") as handle:
            t = np.asarray(handle["t"][: args.samples])
            x = np.asarray(handle["x"])
            data = handle["Q_1D"]

            for segment in range(segment_counts[label]):
                realization += 1
                start = segment * args.samples
                stop = start + args.samples
                state = np.asarray(data[:, start:stop])

                fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
                image = ax.pcolormesh(
                    t,
                    x,
                    state,
                    shading="auto",
                    cmap="jet",
                    vmin=args.vmin,
                    vmax=args.vmax,
                    rasterized=True,
                )
                ax.set_title(
                    f"Power {args.power} — experiment {label}, segment {segment + 1} "
                    f"(realization {realization}/{total})"
                )
                ax.set_xlabel("Time within segment [s]")
                ax.set_ylabel(r"x [$\mu$m]")
                colorbar = fig.colorbar(image, ax=ax)
                colorbar.set_label(r"Surface displacement [$\mu$m]")

                filename = (
                    f"realization_{realization:03d}_experiment_{label}_"
                    f"segment_{segment + 1:02d}.png"
                )
                fig.savefig(output_dir / filename, dpi=args.dpi)
                plt.close(fig)
                print(f"[{realization}/{total}] {filename}", flush=True)

    print(f"Saved {total} plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
