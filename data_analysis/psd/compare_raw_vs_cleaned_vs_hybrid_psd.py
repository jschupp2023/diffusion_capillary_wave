"""Compute and plot 3-way Center-Point PSD comparison: Raw vs Current Cleaned vs New Hybrid Cleaned.

Dataset: 11272025_c_0.2vpp (100,001 frames)
- Raw (matched 50 Hz zero-phase HP filter)
- Current Cleaned (unweighted Poisson integer unwrapped, 50 Hz HP)
- New Hybrid Cleaned (conservative hybrid mask + boundary growth harmonic unwrap, 50 Hz HP)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

CACHE_CUR = Path("runs/data_cleaning_production/center_signals_11272025_c_0.2vpp.npz")
CACHE_HYB = Path("runs/data_cleaning_production/hybrid_center_signals_11272025_c_0.2vpp.npz")

OUT_FIG_THESIS = Path("thesis_notes/figures/psd_0p20_raw_vs_cleaned.png")
OUT_FIG_ARTIFACT = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/psd_0p20_raw_vs_cleaned.png")
METRICS_JSON = Path("runs/data_cleaning_production/psd_0p20_hybrid_metrics.json")


def compute_3way_psd(nperseg: int = 8192, highpass_cutoff: float = 50.0):
    c1 = np.load(CACHE_CUR)
    c2 = np.load(CACHE_HYB)

    raw_center = c1["raw_center"]
    cln_center = c1["cln_center"]
    hyb_center = c2["hybrid_center"]
    fs = float(c1["fs"])

    noverlap = nperseg // 2

    # High-pass filter raw to match cleaned datasets
    sos = butter(4, highpass_cutoff, btype="highpass", fs=fs, output="sos")
    raw_center_hp = sosfiltfilt(sos, raw_center.astype(np.float64)).astype(np.float32)

    # Welch PSD
    f_raw, p_raw = welch(raw_center_hp, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_cln, p_cln = welch(cln_center, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_hyb, p_hyb = welch(hyb_center, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")

    # Metrics
    mask_cascade = (f_raw >= 70.0) & (f_raw <= 1500.0)
    power_raw_cascade = float(np.trapezoid(p_raw[mask_cascade], f_raw[mask_cascade]))
    power_cln_cascade = float(np.trapezoid(p_cln[mask_cascade], f_cln[mask_cascade]))
    power_hyb_cascade = float(np.trapezoid(p_hyb[mask_cascade], f_hyb[mask_cascade]))

    retained_cln_pct = 100.0 * power_cln_cascade / power_raw_cascade
    retained_hyb_pct = 100.0 * power_hyb_cascade / power_raw_cascade

    idx_20k = int(np.argmin(np.abs(f_raw - 20000.0)))
    noise_red_cln_db = float(10.0 * np.log10(p_raw[idx_20k] / p_cln[idx_20k]))
    noise_red_hyb_db = float(10.0 * np.log10(p_raw[idx_20k] / p_hyb[idx_20k]))

    # Scaling exponent beta (200 - 1200 Hz)
    fit_mask = (f_raw >= 200.0) & (f_raw <= 1200.0)
    slope_raw, _ = np.polyfit(np.log10(f_raw[fit_mask]), np.log10(p_raw[fit_mask]), 1)
    slope_cln, _ = np.polyfit(np.log10(f_cln[fit_mask]), np.log10(p_cln[fit_mask]), 1)
    slope_hyb, _ = np.polyfit(np.log10(f_hyb[fit_mask]), np.log10(p_hyb[fit_mask]), 1)

    print(f"Turbulent Power (70-1500 Hz):")
    print(f"  Raw:     {power_raw_cascade:.2f} um^2")
    print(f"  Current: {power_cln_cascade:.2f} um^2 ({retained_cln_pct:.2f}%)")
    print(f"  Hybrid:  {power_hyb_cascade:.2f} um^2 ({retained_hyb_pct:.2f}%)")
    print(f"Noise reduction at 20 kHz:")
    print(f"  Current: {noise_red_cln_db:.2f} dB")
    print(f"  Hybrid:  {noise_red_hyb_db:.2f} dB")

    # Plot
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)
    plot_mask = (f_raw >= 30.0) & (f_raw <= 45000.0)

    ax.loglog(f_raw[plot_mask], p_raw[plot_mask], color="#d62728", lw=1.4, alpha=0.75, label="Raw DHM Surface (50 Hz HP)")
    ax.loglog(f_cln[plot_mask], p_cln[plot_mask], color="#1f77b4", lw=1.5, alpha=0.85, label=f"Current Cleaned (Unweighted Poisson, {retained_cln_pct:.1f}% power)")
    ax.loglog(f_hyb[plot_mask], p_hyb[plot_mask], color="#2ca02c", lw=1.6, alpha=0.95, label=f"New Hybrid Cleaned (Boundary Growth Mask, {retained_hyb_pct:.1f}% power)")

    # Reference slope Zakharov-Filonenko f^(-17/6)
    f_ref = np.linspace(300, 2500, 100)
    p_ref = p_hyb[np.argmin(np.abs(f_hyb - 500))] * (f_ref / 500.0) ** (-17.0 / 6.0)
    ax.loglog(f_ref, p_ref, "k--", lw=1.2, alpha=0.7, label=r"Zakharov-Filonenko $\sim f^{-17/6}$")

    ax.set_xlabel("Frequency [Hz]", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Center Elevation PSD $S_{hh}(f)$ [$\mu\mathrm{m}^2/\mathrm{Hz}$]", fontsize=12, fontweight="bold")
    ax.set_title(
        r"$\Gamma = 0.20\,\mathrm{g}$ Center Point PSD: Raw vs. Current vs. Hybrid Cleaned" "\n"
        r"(100k Timesteps, $N_{\mathrm{perseg}}=8192$, Welch 50% Overlap)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(30, 45000)
    ax.set_ylim(1e-5, 1e2)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

    # Annotate noise floor drop
    ax.annotate(
        f"Step Noise Eliminated\n(-{noise_red_hyb_db:.1f} dB at 20 kHz)",
        xy=(20000, p_hyb[idx_20k]),
        xytext=(5500, 2.5e-5),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2ca02c", alpha=0.9),
    )

    # Annotate cascade retention
    ax.annotate(
        f"+{retained_hyb_pct - retained_cln_pct:.1f}% More Power Retained\n(No Crest Attenuation)",
        xy=(300, p_hyb[np.argmin(np.abs(f_hyb - 300))]),
        xytext=(70, 8e-1),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2ca02c", alpha=0.9),
    )

    fig.tight_layout()

    for out_p in [OUT_FIG_THESIS, OUT_FIG_ARTIFACT]:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)
        print(f"Saved PSD figure to {out_p}")

    plt.close(fig)

    metrics = {
        "fs": fs,
        "nperseg": nperseg,
        "power_raw_cascade_70_1500": power_raw_cascade,
        "power_cln_cascade_70_1500": power_cln_cascade,
        "power_hyb_cascade_70_1500": power_hyb_cascade,
        "retained_cln_pct": retained_cln_pct,
        "retained_hyb_pct": retained_hyb_pct,
        "noise_reduction_cln_db_20k": noise_red_cln_db,
        "noise_reduction_hyb_db_20k": noise_red_hyb_db,
        "slope_raw_200_1200": float(slope_raw),
        "slope_cln_200_1200": float(slope_cln),
        "slope_hyb_200_1200": float(slope_hyb),
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {METRICS_JSON}")
    return metrics


if __name__ == "__main__":
    compute_3way_psd()

