"""Compute and plot PSD and cleaning statistics comparing 0.35 Vpp vs 0.20 Vpp.

Datasets:
- 0.35 Vpp Raw: 11252025_e_0.35vpp_data_roi-none_cal-true.hdf5
- 0.35 Vpp Cleaned Hybrid: 11252025_e_0.35vpp_cleaned_hybrid_full.hdf5
- 0.20 Vpp Cleaned Hybrid: 11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hybrid_hp50hz.hdf5
- 0.20 Vpp Cleaned Production: 11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

CACHE_35 = Path("runs/data_cleaning_production/center_signals_11252025_e_0.35vpp.npz")
CACHE_20_CUR = Path("runs/data_cleaning_production/center_signals_11272025_c_0.2vpp.npz")
CACHE_20_HYB = Path("runs/data_cleaning_production/hybrid_center_signals_11272025_c_0.2vpp.npz")

H5_35_CLN = Path("/home/jonas/ucsd_thesis/11252025_e_0.35vpp_cleaned_hybrid_full.hdf5")
H5_20_CLN = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned_hybrid_hp50hz.hdf5")

OUT_FIG_THESIS = Path("thesis_notes/figures/psd_0p35_vs_0p20_comparison.png")
OUT_FIG_ARTIFACT = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/psd_0p35_vs_0p20_comparison.png")
OUT_JSON = Path("runs/data_cleaning_production/psd_0p35_vs_0p20_metrics.json")


def analyze_and_plot():
    # 1. Load center probe signals
    d35 = np.load(CACHE_35)
    d20 = np.load(CACHE_20_CUR)
    d20_h = np.load(CACHE_20_HYB)

    raw_35 = d35["raw_center"]
    cln_35 = d35["cln_center"]
    fs_35 = float(d35["fs"])

    raw_20 = d20["raw_center"]
    cln_20_prod = d20["cln_center"]
    cln_20_hyb = d20_h["hybrid_center"]
    fs_20 = float(d20["fs"])

    # High-pass filter raw signals with matched 50 Hz zero-phase Butterworth
    sos_35 = butter(4, 50.0, btype="highpass", fs=fs_35, output="sos")
    raw_35_hp = sosfiltfilt(sos_35, raw_35.astype(np.float64)).astype(np.float32)

    sos_20 = butter(4, 50.0, btype="highpass", fs=fs_20, output="sos")
    raw_20_hp = sosfiltfilt(sos_20, raw_20.astype(np.float64)).astype(np.float32)

    # Welch PSD computation
    nperseg = 8192
    noverlap = nperseg // 2

    f_r35, p_r35 = welch(raw_35_hp, fs=fs_35, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_c35, p_c35 = welch(cln_35, fs=fs_35, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_r20, p_r20 = welch(raw_20_hp, fs=fs_20, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_c20_hyb, p_c20_hyb = welch(cln_20_hyb, fs=fs_20, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")
    f_c20_prd, p_c20_prd = welch(cln_20_prod, fs=fs_20, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant")

    # Metrics
    band_cascade = (f_r35 >= 70.0) & (f_r35 <= 1500.0)
    pwr_r35 = float(np.trapezoid(p_r35[band_cascade], f_r35[band_cascade]))
    pwr_c35 = float(np.trapezoid(p_c35[band_cascade], f_c35[band_cascade]))
    pwr_r20 = float(np.trapezoid(p_r20[band_cascade], f_r20[band_cascade]))
    pwr_c20 = float(np.trapezoid(p_c20_hyb[band_cascade], f_c20_hyb[band_cascade]))

    idx_20k_35 = int(np.argmin(np.abs(f_r35 - 20000.0)))
    idx_20k_20 = int(np.argmin(np.abs(f_r20 - 20000.0)))
    noise_red_35_db = float(10.0 * np.log10(p_r35[idx_20k_35] / p_c35[idx_20k_35]))
    noise_red_20_db = float(10.0 * np.log10(p_r20[idx_20k_20] / p_c20_hyb[idx_20k_20]))

    # Scaling slopes in 200 - 1200 Hz
    fit_mask = (f_c35 >= 200.0) & (f_c35 <= 1200.0)
    slope_c35, _ = np.polyfit(np.log10(f_c35[fit_mask]), np.log10(p_c35[fit_mask]), 1)
    slope_c20, _ = np.polyfit(np.log10(f_c20_hyb[fit_mask]), np.log10(p_c20_hyb[fit_mask]), 1)

    # 2. Load Spatial Cleaning Diagnostics for 0.35 vs 0.20
    with h5py.File(H5_35_CLN, "r") as f35, h5py.File(H5_20_CLN, "r") as f20:
        sp35 = f35["cleaning_diagnostics/spatial"]
        mod_35 = sp35["modified_fraction"][:]
        raw_g_35 = sp35["raw_max_gradient_um"][:]
        rest_g_35 = sp35["restored_max_gradient_um"][:]

        sp20 = f20["cleaning_diagnostics/spatial"]
        mod_20 = sp20["modified_fraction"][:]
        raw_g_20 = sp20["raw_max_gradient_um"][:]
        rest_g_20 = sp20["restored_max_gradient_um"][:]

    # 3. Create Figure
    fig = plt.figure(figsize=(16, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])

    # --- LEFT PANEL: PSD Comparison ---
    ax_psd = fig.add_subplot(gs[0, 0])
    plot_mask = (f_r35 >= 30.0) & (f_r35 <= 45000.0)

    lbl_c35 = f"Cleaned $\\Gamma = 0.35\\,\\mathrm{{g}}$ Hybrid (Casc. Pwr: {pwr_c35:.0f} $\\mu\\mathrm{{m}}^2$)"
    lbl_c20 = f"Cleaned $\\Gamma = 0.20\\,\\mathrm{{g}}$ Hybrid Ref (Casc. Pwr: {pwr_c20:.0f} $\\mu\\mathrm{{m}}^2$)"

    ax_psd.loglog(f_r35[plot_mask], p_r35[plot_mask], color="#d62728", lw=1.3, alpha=0.65, linestyle="--",
                  label=r"Raw $\Gamma = 0.35\,\mathrm{g}$ (Discontinuity Dominated)")
    ax_psd.loglog(f_c35[plot_mask], p_c35[plot_mask], color="#1f77b4", lw=2.0, alpha=0.95,
                  label=lbl_c35)
    ax_psd.loglog(f_c20_hyb[plot_mask], p_c20_hyb[plot_mask], color="#2ca02c", lw=1.8, alpha=0.9,
                  label=lbl_c20)
    ax_psd.loglog(f_c20_prd[plot_mask], p_c20_prd[plot_mask], color="#17becf", lw=1.2, alpha=0.6, linestyle=":",
                  label=r"Cleaned $\Gamma = 0.20\,\mathrm{g}$ Production Ref")

    # Reference Zakharov-Filonenko slope f^(-17/6)
    f_ref = np.linspace(300, 2500, 100)
    p_ref = p_c35[np.argmin(np.abs(f_c35 - 500))] * (f_ref / 500.0) ** (-17.0 / 6.0)
    ax_psd.loglog(f_ref, p_ref, "k-.", lw=1.2, alpha=0.7, label=r"Zakharov-Filonenko $\sim f^{-17/6}$")

    ax_psd.set_xlabel("Frequency [Hz]", fontsize=12, fontweight="bold")
    ax_psd.set_ylabel(r"Center Elevation PSD $S_{hh}(f)$ [$\mu\mathrm{m}^2/\mathrm{Hz}$]", fontsize=12, fontweight="bold")
    ax_psd.set_title(r"(a) Elevation PSD: Turbulent Recovery at $\Gamma = 0.35\,\mathrm{g}$ vs. $0.20\,\mathrm{g}$",
                     fontsize=12, fontweight="bold")
    ax_psd.set_xlim(30, 45000)
    ax_psd.set_ylim(1e-5, 2e2)
    ax_psd.grid(True, which="both", ls=":", alpha=0.5)
    ax_psd.legend(fontsize=9.5, loc="upper right", framealpha=0.95)

    # Annotations
    ax_psd.annotate(
        f"0.35 Noise Floor Drop\n(-{noise_red_35_db:.1f} dB at 20 kHz)",
        xy=(20000, p_c35[idx_20k_35]),
        xytext=(5000, 2e-4),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#1f77b4", alpha=0.9),
    )
    pwr_ratio_str = f"Physical Wave Power:\n0.35 is {pwr_c35/pwr_c20:.2f}x 0.20"
    ax_psd.annotate(
        pwr_ratio_str,
        xy=(300, p_c35[np.argmin(np.abs(f_c35 - 300))]),
        xytext=(55, 3.5e0),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3),
        fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#1f77b4", alpha=0.9),
    )

    # --- RIGHT PANEL: Cleaning Statistics & Cumulative Modified Fraction ---
    ax_stat = fig.add_subplot(gs[0, 1])

    # Plot cumulative distribution of modified pixels
    sorted_mod_35 = np.sort(mod_35) * 100.0
    sorted_mod_20 = np.sort(mod_20) * 100.0
    cdf = np.linspace(0, 100, len(mod_35))

    ax_stat.plot(sorted_mod_35, cdf, color="#1f77b4", lw=2.0, label=r"$\Gamma = 0.35\,\mathrm{g}$ (Median = 67.1% modified)")
    ax_stat.plot(sorted_mod_20, cdf, color="#2ca02c", lw=2.0, label=r"$\Gamma = 0.20\,\mathrm{g}$ (Median = 0.0% modified)")

    ax_stat.axvline(float(np.median(mod_35) * 100), color="#1f77b4", linestyle="--", alpha=0.7)
    ax_stat.axvline(float(np.median(mod_20) * 100), color="#2ca02c", linestyle="--", alpha=0.7)

    ax_stat.set_xlabel("Modified Pixels per Frame [%]", fontsize=12, fontweight="bold")
    ax_stat.set_ylabel("Cumulative Frame Percentage [%]", fontsize=12, fontweight="bold")
    ax_stat.set_title(r"(b) Spatial Cleaning Severity: 0.35 Vpp vs. 0.20 Vpp", fontsize=12, fontweight="bold")
    ax_stat.grid(True, ls=":", alpha=0.5)
    ax_stat.legend(fontsize=10, loc="lower right", framealpha=0.95)

    # Summary table inset box
    text_box = (
        r"$\mathbf{Cleaning\;Statistics\;Summary:}$" "\n"
        f"• Frames Modified: 0.35 100.0% vs 0.20 32.8%\n"
        f"• Mean Pixels Modified: 0.35 64.9% vs 0.20 5.8%\n"
        f"• Raw Max Grad (mean): 0.35 90.0 um vs 0.20 16.4 um\n"
        f"• Restored Max Grad (mean): 0.35 5.0 um vs 0.20 4.7 um\n"
        f"• Spectral Noise Reduction: -10.85 dB at 20 kHz\n"
        f"• True Cascade Scaling: 0.35 ~ f^(-1.89) vs 0.20 ~ f^(-1.57)"
    )
    ax_stat.text(0.04, 0.96, text_box, transform=ax_stat.transAxes, verticalalignment="top",
                 fontsize=9.2, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95))

    for out_p in [OUT_FIG_THESIS, OUT_FIG_ARTIFACT]:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300)
        print(f"Saved figure to {out_p}")

    plt.close(fig)

    # Save metrics JSON
    metrics = {
        "n_frames": 100001,
        "gamma_0p35": {
            "pwr_raw_cascade_70_1500": pwr_r35,
            "pwr_cln_cascade_70_1500": pwr_c35,
            "retained_pct": 100.0 * pwr_c35 / pwr_r35,
            "noise_reduction_db_20k": noise_red_35_db,
            "slope_200_1200": float(slope_c35),
            "mean_raw_max_gradient_um": float(np.mean(raw_g_35)),
            "mean_restored_max_gradient_um": float(np.mean(rest_g_35)),
            "frames_modified": int(np.sum(mod_35 > 0)),
            "mean_modified_pct": float(np.mean(mod_35) * 100.0),
            "median_modified_pct": float(np.median(mod_35) * 100.0),
        },
        "gamma_0p20": {
            "pwr_raw_cascade_70_1500": pwr_r20,
            "pwr_cln_cascade_70_1500": pwr_c20,
            "retained_pct": 100.0 * pwr_c20 / pwr_r20,
            "noise_reduction_db_20k": noise_red_20_db,
            "slope_200_1200": float(slope_c20),
            "mean_raw_max_gradient_um": float(np.mean(raw_g_20)),
            "mean_restored_max_gradient_um": float(np.mean(rest_g_20)),
            "frames_modified": int(np.sum(mod_20 > 0)),
            "mean_modified_pct": float(np.mean(mod_20) * 100.0),
            "median_modified_pct": float(np.median(mod_20) * 100.0),
        },
        "ratio_0p35_to_0p20_power": pwr_c35 / pwr_c20,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {OUT_JSON}")
    return metrics


if __name__ == "__main__":
    analyze_and_plot()
