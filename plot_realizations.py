"""Plot every 5760-sample realization used by the notebook."""

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
        description="Plot every complete segment from every repeated experiment."
    )
    parser.add_argument("--power", default="0p10", choices=POWER_LABELS)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jonas/ucsd_thesis/DHM_new_1Dcenter"),
    )
    parser.add_argument("--samples", type=int, default=5760)
    parser.add_argument(
        "--whole-experiment",
        action="store_true",
        help="Plot each complete recording instead of splitting it into segments.",
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
        help="Output directory (default: realizations_<power>).",
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vmin >= args.vmax:
        raise ValueError("--vmin must be smaller than --vmax.")
    if args.samples < 1 or args.time_stride < 1:
        raise ValueError("--samples and --time-stride must be positive.")
    labels = POWER_LABELS[args.power]
    default_dir = (
        f"whole_experiments_{args.power}"
        if args.whole_experiment
        else f"realizations_{args.power}"
    )
    output_dir = args.output_dir or Path(default_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        label: args.data_dir / args.power / f"Q_1D_{args.power}vpp_{label}.h5"
        for label in labels
    }
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
