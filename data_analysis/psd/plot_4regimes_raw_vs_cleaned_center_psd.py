"""Compute and plot Center-Point PSD (Raw vs Cleaned) across all 4 datasets in one single plot.

Datasets:
  1. 0.04 Vpp
  2. 0.08 Vpp
  3. 0.20 Vpp
  4. 0.35 Vpp

Starting frequency: 50 Hz.
"""

from __future__ import annotations

import concurrent.futures
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

from data_analysis.pod.pod_2d import _frame_key, inspect_input

CACHE_FILE = Path("runs/psd_4regimes_comparison/center_signals_4regimes.npz")
OUTPUT_REPO_FIG = Path("runs/psd_4regimes_comparison/center_psd_4regimes_raw_vs_cleaned.png")
OUTPUT_ARTIFACT_FIG = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/center_psd_4regimes_raw_vs_cleaned.png")


def extract_single_center_trace(h5_path_str: str) -> np.ndarray:
    path = Path(h5_path_str)
    print(f"  Worker extracting center from {path.name}...")
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        main_grp = f["main"]
        first = main_grp[_frame_key(0)]
        cy, cx = first.shape[0] // 2, first.shape[1] // 2
        n_frames = len(f["meta/t"]) if "meta/t" in f else len(main_grp)
        center = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            center[i] = main_grp[_frame_key(i)][cy, cx]
    print(f"  Done extracting {path.name} ({n_frames:,} frames, {time.perf_counter()-t0:.1f} s)")
    return center


def load_or_extract_all_signals() -> dict[str, dict[str, np.ndarray]]:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        print(f"Loading cached center signals from {CACHE_FILE}...")
        d = np.load(CACHE_FILE)
        return {
            "0.04 Vpp": {"raw": d["c04_raw"], "cln": d["c04_cln"], "fs": float(d["fs"])},
            "0.08 Vpp": {"raw": d["c08_raw"], "cln": d["c08_cln"], "fs": float(d["fs"])},
            "0.20 Vpp": {"raw": d["c20_raw"], "cln": d["c20_cln"], "fs": float(d["fs"])},
            "0.35 Vpp": {"raw": d["c35_raw"], "cln": d["c35_cln"], "fs": float(d["fs"])},
        }

    print("Extracting signals across all 4 regimes (parallel workers)...")
    t0 = time.perf_counter()

    # 1. 0.08 Vpp: load existing cache if available
    cache_08 = Path("runs/validation_0p04_0p08_cleaning/center_0p08_raw_cache.npz")
    if cache_08.exists():
        d08 = np.load(cache_08)
        c08_raw = d08["center_raw"]
        fs = float(d08["fs"])
    else:
        c08_raw = extract_single_center_trace("/home/jonas/ucsd_thesis/11222025_h_0.08vpp_data_roi-none_cal-true.hdf5")
        fs = 115200.0

    # 0.08 cleaned is bit-for-bit identical before 50 Hz filter
    c08_cln = c08_raw.copy()

    # 2. 0.20 Vpp: load from runs/comparison_0p20_integer_cleaned/comparison_metrics.npz
    m20 = np.load("runs/comparison_0p20_integer_cleaned/comparison_metrics.npz")
    c20_raw = m20["raw_center"]
    c20_cln = m20["cln_center"]

    # 3. 0.04 raw, 0.35 raw, 0.35 cleaned extracted via ProcessPoolExecutor
    to_extract = {
        "c04_raw": "/home/jonas/ucsd_thesis/11222025_j_0.04vpp_data_roi-none_cal-true.hdf5",
        "c35_raw": "/home/jonas/ucsd_thesis/11252025_e_0.35vpp_data_roi-none_cal-true.hdf5",
        "c35_cln": "/home/jonas/ucsd_thesis/11252025_e_0.35vpp_data_roi-none_cal-true_cleaned_tau4.hdf5",
    }

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        fut_to_key = {executor.submit(extract_single_center_trace, path): key for key, path in to_extract.items()}
        for fut in concurrent.futures.as_completed(fut_to_key):
            key = fut_to_key[fut]
            results[key] = fut.result()

    c04_raw = results["c04_raw"]
    c04_cln = c04_raw.copy()  # 0.04 has 0 defects altered
    c35_raw = results["c35_raw"]
    c35_cln = results["c35_cln"]

    # Save master cache
    np.savez_compressed(
        CACHE_FILE,
        c04_raw=c04_raw, c04_cln=c04_cln,
        c08_raw=c08_raw, c08_cln=c08_cln,
        c20_raw=c20_raw, c20_cln=c20_cln,
        c35_raw=c35_raw, c35_cln=c35_cln,
        fs=fs,
    )
    print(f"Master center signals cached to {CACHE_FILE} in {time.perf_counter()-t0:.1f} s.")

    return {
        "0.04 Vpp": {"raw": c04_raw, "cln": c04_cln, "fs": fs},
        "0.08 Vpp": {"raw": c08_raw, "cln": c08_cln, "fs": fs},
        "0.20 Vpp": {"raw": c20_raw, "cln": c20_cln, "fs": fs},
        "0.35 Vpp": {"raw": c35_raw, "cln": c35_cln, "fs": fs},
    }


def main():
    data = load_or_extract_all_signals()
    fs = data["0.04 Vpp"]["fs"]
    cutoff_hz = 50.0
    sos = butter(4, cutoff_hz, btype="highpass", fs=fs, output="sos")

    nperseg = 16384
    noverlap = nperseg // 2

    # Regimes setup
    regimes = [
        {"name": "0.04 Vpp", "color": "#1f77b4", "power": "0.04"},
        {"name": "0.08 Vpp", "color": "#2ca02c", "power": "0.08"},
        {"name": "0.20 Vpp", "color": "#ff7f0e", "power": "0.20"},
        {"name": "0.35 Vpp", "color": "#d62728", "power": "0.35"},
    ]

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)

    print("\nComputing Welch PSDs and plotting...")
    f_master = None
    for r in regimes:
        name = r["name"]
        color = r["color"]
        sig_raw = data[name]["raw"]
        sig_cln = data[name]["cln"]

        # Apply 50 Hz HP filter to cleaned dataset
        sig_cln_hp = sosfiltfilt(sos, sig_cln.astype(np.float64)).astype(np.float32)

        # Welch PSD
        f, psd_raw = welch(sig_raw, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
        _, psd_cln = welch(sig_cln_hp, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
        if f_master is None:
            f_master = f

        # Mask f >= 50 Hz as requested
        mask = (f >= 50.0) & (f <= 4000.0)

        # Plot Raw (dashed) and Cleaned (solid)
        ax.plot(f[mask], psd_raw[mask], color=color, ls="--", lw=1.6, alpha=0.65, label=f"{name} — Raw")
        ax.plot(f[mask], psd_cln[mask], color=color, ls="-", lw=2.2, alpha=0.95, label=f"{name} — Cleaned (50 Hz HP)")

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(50.0, 4000.0)
    ax.set_ylim(1e-8, 2e2)

    # Key annotations
    ax.axvline(140.0, color="navy", ls=":", lw=1.3, alpha=0.75, label="140 Hz Bulk Heave / Acoustic Resonance")
    ax.axvline(280.0, color="purple", ls=":", lw=1.0, alpha=0.55)
    ax.axvline(420.0, color="purple", ls=":", lw=1.0, alpha=0.55)

    # Theoretical cascade line f^(-17/6)
    f_ref = np.linspace(300, 1500, 100)
    # anchor near 0.20 Vpp at 300 Hz
    s_ref = 0.08 * (f_ref / 300.0) ** (-17.0 / 6.0)
    ax.plot(f_ref, s_ref, "k-.", lw=1.6, alpha=0.8, label=r"Theory: $f^{-17/6} \approx f^{-2.83}$ (Zakharov-Filonenko)")

    ax.set_xlabel("Frequency $f$ [Hz] (Log scale, starting at 50 Hz)", fontsize=12, fontweight="medium")
    ax.set_ylabel("Power Spectral Density $S(f)$ [$\\mu$m$^2$/Hz]", fontsize=12, fontweight="medium")
    ax.set_title("Center-Point Power Spectral Density (PSD): Raw vs Cleaned Across 4 Forcing Regimes", fontsize=13.5, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.35)

    # Custom legend with two columns
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.92, ncol=2)

    OUTPUT_REPO_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_REPO_FIG, dpi=220)
    fig.savefig(OUTPUT_ARTIFACT_FIG, dpi=220)
    plt.close(fig)

    print(f"\nFigure generated successfully:")
    print(f"  1. {OUTPUT_REPO_FIG}")
    print(f"  2. {OUTPUT_ARTIFACT_FIG}")


if __name__ == "__main__":
    main()
