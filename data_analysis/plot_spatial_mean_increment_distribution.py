"""Compare spatial-mean increments before and after production temporal cleaning."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


FINAL_CACHE = Path("runs/data_cleaning_production/center_signals_11272025_c_0.2vpp.npz")
SPATIAL_CACHE = Path(
    "runs/temporal_alignment_validation/step_events_0p20/full_recording_traces.npz"
)
OUTPUT = Path("thesis_notes/figures/spatial_mean_increment_distribution.png")
Q_MICRONS = 1.9969696
HIGHPASS_HZ = 50.0


def exceedance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.abs(values))
    probability = (ordered.size - np.arange(ordered.size)) / ordered.size
    return ordered, probability


def main() -> None:
    final_cache = np.load(FINAL_CACHE)
    spatial_cache = np.load(SPATIAL_CACHE)
    fs = float(final_cache["fs"])
    spatial_mean = np.asarray(spatial_cache["spatial"][:, 0], dtype=np.float64)
    final_mean = np.asarray(final_cache["cln_mean"], dtype=np.float64)

    sos = butter(4, HIGHPASS_HZ, btype="highpass", fs=fs, output="sos")
    before = np.diff(sosfiltfilt(sos, spatial_mean))
    after = np.diff(final_mean)
    abs_before = np.abs(before)
    abs_after = np.abs(after)

    bins = np.geomspace(1e-3, 1.5e2, 130)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    axes[0].hist(
        abs_before,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        color="#dd8452",
        label="Before temporal cleaning",
    )
    axes[0].hist(
        abs_after,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        color="#4c72b0",
        label="After temporal cleaning",
    )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel=r"Absolute increment $|\Delta\bar h|$ [$\mu$m]",
        ylabel="Probability density",
        title="(a) Increment distribution",
    )

    for values, color, label in (
        (before, "#dd8452", "Before temporal cleaning"),
        (after, "#4c72b0", "After temporal cleaning"),
    ):
        x, probability = exceedance(values)
        axes[1].loglog(x, probability, color=color, lw=1.5, label=label)
    axes[1].set(
        xlabel=r"Threshold $a$ [$\mu$m]",
        ylabel=r"$P(|\Delta\bar h|>a)$",
        title="(b) Tail exceedance",
    )

    for axis in axes:
        axis.axvline(Q_MICRONS, color="0.4", ls="--", lw=0.9, label=r"$q$")
        axis.axvline(3 * Q_MICRONS, color="0.4", ls=":", lw=0.9, label=r"$3q$")
        axis.grid(True, which="both", alpha=0.22, lw=0.55)
        axis.legend(frameon=False, fontsize=8.0)

    summary = (
        rf"max: {abs_before.max():.1f} $\rightarrow$ {abs_after.max():.1f} $\mu$m"
        "\n"
        rf"$|\Delta\bar h|>10q$: {np.count_nonzero(abs_before > 10*Q_MICRONS)}"
        rf" $\rightarrow$ {np.count_nonzero(abs_after > 10*Q_MICRONS)}"
    )
    axes[1].text(
        0.04,
        0.05,
        summary,
        transform=axes[1].transAxes,
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )

    fig.suptitle("Production temporal cleaning: spatial-mean increment tails")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
