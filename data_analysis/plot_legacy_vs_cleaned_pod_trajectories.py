"""Qualitative legacy-versus-cleaned 0.20 Vpp shared-POD trajectories."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData


REPETITION = "Ca_ac_0p001660_rep14"
START = 74_569
LENGTH = 5_001
RANK = 20
DISPLAY_MODES = (1, 2, 5, 10, 20)
OUTPUT = Path("thesis_notes/figures/legacy_vs_cleaned_pod_trajectories.png")

OLD_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
OLD_BASIS = OLD_ROOT / "0p20/shared_pod/shared_pod_r1000.h5"
NEW_ROOT = Path("/home/jonas/ucsd_thesis/clean_red_data")
NEW_BASIS = NEW_ROOT / "0p20/shared_pod/shared_pod_r200.h5"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = dict(validation_reps=[13, 14], test_reps=[15, 16])
    legacy = SharedPODTrainingData(
        "0p20",
        RANK,
        data_root=OLD_ROOT,
        shared_basis=OLD_BASIS,
        include_spatial_mean=True,
        spatial_mean_highpass_hz=50.0,
        **common,
    )
    cleaned = SharedPODTrainingData(
        "0p20",
        RANK,
        data_root=NEW_ROOT,
        shared_basis=NEW_BASIS,
        **common,
    )
    stop = START + LENGTH
    time_old, coordinates_old = legacy.read_coordinates(REPETITION, START, stop)
    time_new, coordinates_new = cleaned.read_coordinates(REPETITION, START, stop)
    if not np.allclose(time_old, time_new, rtol=0.0, atol=1e-12):
        raise RuntimeError("Legacy and cleaned trajectories are not time aligned.")
    return (
        time_old,
        legacy.spatial_mean(coordinates_old),
        legacy.pod_coefficients(coordinates_old),
        cleaned.spatial_mean(coordinates_new),
        cleaned.pod_coefficients(coordinates_new),
    )


def main() -> None:
    time, mean_old, pod_old, mean_new, pod_new = load_data()
    time_ms = 1e3 * (time - time[0])
    rows = 1 + len(DISPLAY_MODES)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(11.0, 10.8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0, 0].plot(time_ms, mean_old, color="0.22", lw=0.85)
    axes[0, 1].plot(time_ms, mean_new, color="#4c72b0", lw=0.85)
    axes[0, 0].set_title("Legacy raw-data basis\n(separate spatial mean)")
    axes[0, 1].set_title("Cleaned-data basis\n(spatial mean embedded)")
    axes[0, 0].set_ylabel(r"Spatial mean [$\mu$m]")

    for row, mode in enumerate(DISPLAY_MODES, start=1):
        axes[row, 0].plot(time_ms, pod_old[:, mode - 1], color="0.22", lw=0.75)
        axes[row, 1].plot(time_ms, pod_new[:, mode - 1], color="#4c72b0", lw=0.75)
        axes[row, 0].set_ylabel(rf"POD {mode} [$\mu$m]")

    for row in range(rows):
        for column in range(2):
            axes[row, column].grid(True, alpha=0.22, lw=0.55)
            axes[row, column].tick_params(labelsize=8)
        axes[row, 1].tick_params(labelleft=True)
    axes[-1, 0].set_xlabel("Time from window start [ms]")
    axes[-1, 1].set_xlabel("Time from window start [ms]")

    fig.suptitle(
        f"Representative 0.20 Vpp trajectories: {REPETITION}, frames "
        f"{START:,}--{START + LENGTH - 1:,}\n"
        "Independent bases: compare regularity, not mode identity or amplitude"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
