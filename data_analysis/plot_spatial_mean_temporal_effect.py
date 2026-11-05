"""Decompose spatial and temporal cleaning effects on the 0.20 Vpp mean.

The spatial-only trace comes from the full-record audit and is checked against
the production temporal offsets and final ``cleaned_hp50hz`` signal. All PSD
inputs receive the same zero-phase 50 Hz high-pass treatment.
"""

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


FINAL_CACHE = Path("runs/data_cleaning_production/center_signals_11272025_c_0.2vpp.npz")
SPATIAL_CACHE = Path(
    "runs/temporal_alignment_validation/step_events_0p20/"
    "full_recording_traces.npz"
)
CLEANED_H5 = Path(
    "/home/jonas/ucsd_thesis/"
    "11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5"
)
OUTPUT = Path("thesis_notes/figures/spatial_mean_temporal_effect.png")
CENTER_OUTPUT = Path("thesis_notes/figures/center_psd_cleaning_stage_effect.png")
WINDOW = slice(16_500, 20_000)
HIGHPASS_HZ = 50.0
NPERSEG = 8192
Q_MICRONS = 1.9969696


def relative(values: np.ndarray) -> np.ndarray:
    """Express a plotted window relative to its first sample."""
    return values - values[0]


def spectrum(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    return welch(
        values,
        fs=fs,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NPERSEG // 2,
        detrend="constant",
    )


def plot_center_psd_decomposition(
    frequency: np.ndarray,
    psd_raw: np.ndarray,
    psd_spatial: np.ndarray,
    psd_final: np.ndarray,
) -> None:
    """Plot the same three comparisons for the center-pixel observable."""
    comparisons = (
        (psd_raw, psd_spatial, "#c44e52", "#dd8452", "Raw (unfiltered)", "Spatially cleaned (unfiltered)", "(a) Spatial correction only"),
        (psd_spatial, psd_final, "#dd8452", "#4c72b0", "Spatially cleaned (unfiltered)", "Temporal + 50 Hz HP", "(b) Temporal stage + high-pass"),
        (psd_raw, psd_final, "#c44e52", "#4c72b0", "Raw (unfiltered)", "Fully cleaned (incl. 50 Hz HP)", "(c) Full pipeline"),
    )
    valid = frequency >= 30.0
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharex=True, sharey=True, constrained_layout=True)
    for axis, (first, second, c1, c2, label1, label2, title) in zip(axes, comparisons):
        axis.loglog(frequency[valid], first[valid], color=c1, lw=1.1, label=label1)
        axis.loglog(frequency[valid], second[valid], color=c2, lw=1.1, label=label2)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Frequency [Hz]")
        axis.set_xlim(30.0, 5.0e4)
        axis.set_ylim(4.0e-7, 3.0e1)
        axis.grid(True, which="both", alpha=0.22, lw=0.55)
        axis.legend(frameon=False, fontsize=8.0, loc="upper right")
        axis.axvline(HIGHPASS_HZ, color="0.45", ls=":", lw=0.8)
    axes[0].set_ylabel(r"Center-point PSD [$\mu$m$^2$/Hz]")
    fig.suptitle(
        "Center-point effect of each cleaning stage (0.20 Vpp)\n"
        "50 Hz high-pass is introduced only with the temporal stage"
    )
    CENTER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CENTER_OUTPUT, dpi=220)
    plt.close(fig)


def main() -> None:
    final_cache = np.load(FINAL_CACHE)
    spatial_cache = np.load(SPATIAL_CACHE)
    raw_mean = np.asarray(final_cache["raw_mean"], dtype=np.float64)
    cleaned_mean = np.asarray(final_cache["cln_mean"], dtype=np.float64)
    spatial_mean = np.asarray(spatial_cache["spatial"][:, 0], dtype=np.float64)
    raw_center = np.asarray(final_cache["raw_center"], dtype=np.float64)
    cleaned_center = np.asarray(final_cache["cln_center"], dtype=np.float64)
    spatial_center = np.asarray(spatial_cache["spatial"][:, 1], dtype=np.float64)
    fs = float(final_cache["fs"])
    time = np.arange(raw_mean.size, dtype=np.float64) / fs

    with h5py.File(CLEANED_H5, "r") as handle:
        temporal_offsets = np.asarray(
            handle["meta/temporal_offset_cycles"], dtype=np.float64
        )

    sos = butter(4, HIGHPASS_HZ, btype="highpass", fs=fs, output="sos")
    raw_hp = sosfiltfilt(sos, raw_mean)
    spatial_hp = sosfiltfilt(sos, spatial_mean)
    reconstructed_final = sosfiltfilt(
        sos, spatial_mean - Q_MICRONS * temporal_offsets
    )
    reconstruction_error = float(np.max(np.abs(reconstructed_final - cleaned_mean)))
    if reconstruction_error > 1e-3:
        raise RuntimeError(
            "Spatial cache and production output are inconsistent: "
            f"maximum mean reconstruction error is {reconstruction_error:.3g} um"
        )

    reconstructed_center = sosfiltfilt(
        sos, spatial_center - Q_MICRONS * temporal_offsets
    )
    center_reconstruction_error = float(
        np.max(np.abs(reconstructed_center - cleaned_center))
    )
    if center_reconstruction_error > 1e-3:
        raise RuntimeError(
            "Spatial cache and production output are inconsistent: "
            f"maximum center reconstruction error is {center_reconstruction_error:.3g} um"
        )

    frequency, psd_raw = spectrum(raw_hp, fs)
    _, psd_spatial = spectrum(spatial_hp, fs)
    _, psd_final = spectrum(cleaned_mean, fs)

    _, psd_raw_center = spectrum(raw_center, fs)
    _, psd_spatial_center = spectrum(spatial_center, fs)
    _, psd_final_center = spectrum(cleaned_center, fs)

    t_ms = 1e3 * (time[WINDOW] - time[WINDOW.start])
    trajectory_data = (
        (relative(raw_mean[WINDOW]), "#c44e52", "(a) Raw exported field"),
        (
            relative(spatial_mean[WINDOW]),
            "#dd8452",
            "(c) After spatial integer restoration",
        ),
        (
            relative(cleaned_mean[WINDOW]),
            "#4c72b0",
            "(e) After full production cleaning",
        ),
    )
    psd_data = (
        (
            psd_raw,
            psd_spatial,
            "#c44e52",
            "#dd8452",
            "Raw",
            "Spatially cleaned",
            "(b) Spatial correction only",
        ),
        (
            psd_spatial,
            psd_final,
            "#dd8452",
            "#4c72b0",
            "Spatially cleaned",
            "Spatial + temporal",
            "(d) Temporal correction only",
        ),
        (
            psd_raw,
            psd_final,
            "#c44e52",
            "#4c72b0",
            "Raw",
            "Fully cleaned",
            "(f) Full pipeline",
        ),
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11.0, 8.0),
        sharex="col",
        gridspec_kw={"width_ratios": (1.12, 1.0)},
        constrained_layout=True,
    )

    for row, (values, color, title) in enumerate(trajectory_data):
        axis = axes[row, 0]
        axis.plot(t_ms, values, color=color, lw=0.95)
        axis.set_title(title, loc="left")
        axis.set_ylabel(r"Relative $\bar h$ [$\mu$m]")
        axis.grid(True, alpha=0.22, lw=0.55)
    axes[-1, 0].set_xlabel("Time [ms]")

    valid = frequency >= HIGHPASS_HZ
    for row, (first, second, c1, c2, label1, label2, title) in enumerate(psd_data):
        axis = axes[row, 1]
        axis.loglog(frequency[valid], first[valid], color=c1, lw=1.1, label=label1)
        axis.loglog(frequency[valid], second[valid], color=c2, lw=1.1, label=label2)
        axis.set_title(title, loc="left")
        axis.set_ylabel(r"PSD [$\mu$m$^2$/Hz]")
        axis.set_xlim(HIGHPASS_HZ, 5.0e4)
        axis.set_ylim(4.0e-7, 3.0e1)
        axis.grid(True, which="both", alpha=0.22, lw=0.55)
        axis.legend(frameon=False, fontsize=8.0, loc="upper right")
    axes[-1, 1].set_xlabel("Frequency [Hz]")

    fig.suptitle("Spatial-mean effect of each cleaning stage (0.20 Vpp)")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    plot_center_psd_decomposition(
        frequency, psd_raw_center, psd_spatial_center, psd_final_center
    )
    print(f"Maximum production reconstruction error: {reconstruction_error:.3g} um")
    print(f"Maximum center reconstruction error: {center_reconstruction_error:.3g} um")
    print(f"Saved {OUTPUT}")
    print(f"Saved {CENTER_OUTPUT}")


if __name__ == "__main__":
    main()
