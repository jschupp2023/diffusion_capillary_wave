"""Compute and compare Center-Point PSD for Raw vs Cleaned 0.20 Vpp recording (11272025_c).

Follows standard PSD conventions:
- Center probe (y=ny//2, x=nx//2) across full 100,001 timesteps
- Welch method with nperseg=8192, 50% Hann overlap, detrend='constant'
- Matched 50 Hz zero-phase high-pass filter
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

RAW_PATH = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
CLN_PATH = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5")
CACHE_NPZ = Path("runs/data_cleaning_production/center_signals_11272025_c_0.2vpp.npz")
OUT_FIG_THESIS = Path("thesis_notes/figures/psd_0p20_raw_vs_cleaned.png")
OUT_FIG_ARTIFACT = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/psd_0p20_raw_vs_cleaned.png")


def extract_signals(raw_path: Path, cln_path: Path, cache_path: Path | None = None) -> dict[str, np.ndarray]:
    if cache_path and cache_path.exists():
        print(f"Loading cached signals from {cache_path}...")
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    print(f"Extracting signals from raw: {raw_path.name} and cleaned: {cln_path.name}...")
    started = time.perf_counter()

    with h5py.File(raw_path, "r") as f_raw, h5py.File(cln_path, "r") as f_cln:
        t_raw = np.asarray(f_raw["meta"]["t"], dtype=np.float64)
        n_frames = len(t_raw)
        dt = float(np.mean(np.diff(t_raw)))
        fs = 1.0 / dt

        main_raw = f_raw["main"]
        main_cln = f_cln["main"]

        first_key = "00000"
        sample_shape = main_raw[first_key].shape
        ny, nx = sample_shape
        cy, cx = ny // 2, nx // 2
        print(f"Frames: {n_frames:,} | Grid: {ny}x{nx} | Center probe at ({cy}, {cx}) | fs: {fs:.1f} Hz")

        raw_center = np.empty(n_frames, dtype=np.float32)
        cln_center = np.empty(n_frames, dtype=np.float32)
        raw_mean = np.empty(n_frames, dtype=np.float32)
        cln_mean = np.empty(n_frames, dtype=np.float32)

        report_every = n_frames // 5
        for i in range(n_frames):
            key = f"{i:05d}"
            r_frame = np.asarray(main_raw[key], dtype=np.float32)
            c_frame = np.asarray(main_cln[key], dtype=np.float32)

            raw_center[i] = r_frame[cy, cx]
            cln_center[i] = c_frame[cy, cx]
            raw_mean[i] = float(np.mean(r_frame))
            cln_mean[i] = float(np.mean(c_frame))

            if (i + 1) % report_every == 0 or (i + 1) == n_frames:
                pct = (i + 1) / n_frames * 100.0
                elapsed = time.perf_counter() - started
                print(f"  Extracted {i+1:,}/{n_frames:,} frames ({pct:.0f}%) in {elapsed:.1f} s")

    res = {
        "raw_center": raw_center,
        "cln_center": cln_center,
        "raw_mean": raw_mean,
        "cln_mean": cln_mean,
        "fs": np.array(fs),
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **res)
        print(f"Saved signals cache to {cache_path}")

    return res


def compute_and_plot_psd(
    raw_center: np.ndarray,
    cln_center: np.ndarray,
    fs: float,
    nperseg: int = 8192,
    highpass_cutoff: float = 50.0,
    out_paths: list[Path] | None = None,
) -> dict[str, float]:
    print(f"\nComputing Welch PSD (nperseg={nperseg}, 50% overlap, fs={fs:.1f} Hz)...")
    noverlap = nperseg // 2

    # High-pass filter raw to match cleaned (cleaned was filtered with 50 Hz 4th order Butterworth zero-phase)
    sos = butter(4, highpass_cutoff, btype="highpass", fs=fs, output="sos")
    raw_center_hp = sosfiltfilt(sos, raw_center.astype(np.float64)).astype(np.float32)

    f_raw, p_raw = welch(
        raw_center_hp,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )
    f_cln, p_cln = welch(
        cln_center,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )

    # Metrics
    mask_cascade = (f_raw >= 70) & (f_raw <= 1500)
    power_raw_cascade = np.trapezoid(p_raw[mask_cascade], f_raw[mask_cascade])
    power_cln_cascade = np.trapezoid(p_cln[mask_cascade], f_cln[mask_cascade])
    retained_cascade_pct = 100.0 * power_cln_cascade / power_raw_cascade

    idx_20k = np.argmin(np.abs(f_raw - 20000))
    noise_reduction_db_20k = 10.0 * np.log10(p_raw[idx_20k] / p_cln[idx_20k])

    print(f"  Retained power (70-1500 Hz): {retained_cascade_pct:.2f}%")
    print(f"  PSD at 20 kHz: Raw={p_raw[idx_20k]:.2e}, Cln={p_cln[idx_20k]:.2e} -> Noise reduction: {noise_reduction_db_20k:.2f} dB")

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    mask = (f_raw >= 30) & (f_raw <= 45000)

    ax.loglog(f_raw[mask], p_raw[mask], color="#d62728", lw=1.5, alpha=0.85, label="Raw DHM Surface (50 Hz HP)")
    ax.loglog(f_cln[mask], p_cln[mask], color="#1f77b4", lw=1.5, alpha=0.95, label="Cleaned Surface (Integer-Restored, 50 Hz HP)")

    # Reference slope Zakharov-Filonenko f^(-17/6) ~ f^(-2.83)
    f_ref = np.linspace(300, 2500, 100)
    p_ref = p_cln[np.argmin(np.abs(f_cln - 500))] * (f_ref / 500.0) ** (-17.0 / 6.0)
    ax.loglog(f_ref, p_ref, "k--", lw=1.2, alpha=0.7, label=r"Zakharov-Filonenko $\sim f^{-17/6}$")

    ax.set_xlabel("Frequency [Hz]", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Center Elevation PSD $S_{hh}(f)$ [$\mu\mathrm{m}^2/\mathrm{Hz}$]", fontsize=12, fontweight="bold")
    ax.set_title(
        r"$\Gamma = 0.20\,\mathrm{g}$ Center Point PSD: Raw vs. Cleaned (100k Timesteps, $N_{\mathrm{perseg}}=8192$)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(30, 45000)
    ax.set_ylim(1e-5, 1e2)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)

    # Key annotations
    ax.annotate(
        f"Plateau eliminated\n(>{noise_reduction_db_20k:.1f} dB at 20 kHz)",
        xy=(20000, p_cln[idx_20k]),
        xytext=(6000, 3e-5),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1f77b4", alpha=0.9),
    )

    fig.tight_layout()

    if out_paths:
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300)
            print(f"Saved plot to: {p}")

    plt.close(fig)

    metrics = {
        "retained_cascade_pct": float(retained_cascade_pct),
        "noise_reduction_db_20k": float(noise_reduction_db_20k),
        "raw_psd_20k": float(p_raw[idx_20k]),
        "cln_psd_20k": float(p_cln[idx_20k]),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--cln", type=Path, default=CLN_PATH)
    parser.add_argument("--nperseg", type=int, default=8192)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    cache_path = None if args.force_extract else CACHE_NPZ
    signals = extract_signals(args.raw, args.cln, cache_path=cache_path)

    metrics = compute_and_plot_psd(
        raw_center=signals["raw_center"],
        cln_center=signals["cln_center"],
        fs=float(signals["fs"]),
        nperseg=args.nperseg,
        out_paths=[OUT_FIG_THESIS, OUT_FIG_ARTIFACT],
    )

    metrics_path = Path("runs/data_cleaning_production/psd_0p20_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

