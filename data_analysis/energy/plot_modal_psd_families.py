"""Analyze and compare modal PSD spectra and mode families between cleaned 0p20 and 0p08."""

import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

OUT_FIG = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/modal_psd_families_0p20_vs_0p08.png")
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# 0p20 Cleaned POD
CLN_20_POD = Path("runs/cleaned_qc/cleaned_pod_0p20_r100.h5")
# 0p08 POD
POD_08 = Path("/home/jonas/ucsd_thesis/reduced_data/0p08/Ca_ac_0p000664_rep8/pod_2d_r1000.h5")

def main():
    print(f"Loading 0p20 POD: {CLN_20_POD}")
    print(f"Loading 0p08 POD: {POD_08}")

    with h5py.File(CLN_20_POD, "r") as f_20, h5py.File(POD_08, "r") as f_08:
        # Time & sampling
        t_20 = f_20["grid/time"][:]
        fs_20 = 1.0 / np.mean(np.diff(t_20))
        t_08 = f_08["grid/time"][:]
        fs_08 = 1.0 / np.mean(np.diff(t_08))

        # POD coordinates (scaled to physical um: a_k / sqrt(N) where sqrt(N) = 200)
        # In pod_2d: a_k = sum_p h_p phi_p, phi_p has norm 1, so physical elevation contribution is a_k / 200
        n_space_sqrt = 200.0
        
        # Load 5000 frames for 0p20 and 10000 frames for 0p08
        n_frames_20 = len(f_20["reduced/coefficients"])
        n_frames_08 = len(f_08["reduced/coefficients"])
        
        a_20 = f_20["reduced/coefficients"][:min(n_frames_20, 10000), :8] / n_space_sqrt
        mean_20 = f_20["preprocessing/frame_spatial_mean"][:min(n_frames_20, 10000)]
        
        a_08 = f_08["reduced/coefficients"][:min(n_frames_08, 10000), :8] / n_space_sqrt
        mean_08 = f_08["preprocessing/frame_spatial_mean"][:min(n_frames_08, 10000)]

        # Compute Welch PSDs
        nperseg = 2048
        f_20_psd, p_mean_20 = welch(mean_20, fs=fs_20, nperseg=nperseg, noverlap=nperseg//2)
        f_08_psd, p_mean_08 = welch(mean_08, fs=fs_08, nperseg=nperseg, noverlap=nperseg//2)

        p_modes_20 = [welch(a_20[:, k], fs=fs_20, nperseg=nperseg, noverlap=nperseg//2)[1] for k in range(8)]
        p_modes_08 = [welch(a_08[:, k], fs=fs_08, nperseg=nperseg, noverlap=nperseg//2)[1] for k in range(8)]

        # 4-Panel Figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

        # Panel A: 0p20 Cleaned Modal PSDs (Rich resonance spectrum across mode families)
        ax1 = axs[0, 0]
        mask_f = (f_20_psd >= 40) & (f_20_psd <= 2500)
        ax1.semilogy(f_20_psd[mask_f], p_mean_20[mask_f], "k-", lw=2.2, label=r"Spatial Mean $\bar{h}(t)$")
        for k in range(5):
            ax1.semilogy(f_20_psd[mask_f], p_modes_20[k][mask_f], color=colors[k], lw=1.5, label=f"Mode {k+1}")
        ax1.axvline(140, color="gray", ls="--", lw=1.2, label="140 Hz Bulk Heave")
        ax1.axvline(197, color="darkcyan", ls=":", lw=1.2, label="197 Hz Mode Family")
        ax1.axvline(365, color="magenta", ls=":", lw=1.2, label="365 Hz Mode Family")
        ax1.set_xlim(40, 2000)
        ax1.set_ylim(1e-4, 1e2)
        ax1.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
        ax1.set_ylabel(r"PSD [$\mu$m$^2$/Hz]", fontsize=11)
        ax1.set_title("A. Cleaned 0p20 Modal PSD: Rich Multi-Frequency Resonances", fontweight="bold", fontsize=12)
        ax1.grid(True, which="both", ls="--", alpha=0.35)
        ax1.legend(loc="upper right", fontsize=9.0, ncol=2)

        # Panel B: 0p08 Modal PSDs (Strong dominance of 140 Hz and narrow subharmonic)
        ax2 = axs[0, 1]
        ax2.semilogy(f_08_psd[mask_f], p_mean_08[mask_f], "k-", lw=2.2, label=r"Spatial Mean $\bar{h}(t)$")
        for k in range(5):
            ax2.semilogy(f_08_psd[mask_f], p_modes_08[k][mask_f], color=colors[k], lw=1.5, label=f"Mode {k+1}")
        ax2.axvline(140, color="gray", ls="--", lw=1.2, label="140 Hz Bulk Heave")
        ax2.set_xlim(40, 2000)
        ax2.set_ylim(1e-5, 1e2)
        ax2.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
        ax2.set_ylabel(r"PSD [$\mu$m$^2$/Hz]", fontsize=11)
        ax2.set_title("B. 0p08 Modal PSD: Weak Cascade, Dominated by Container Heave", fontweight="bold", fontsize=12)
        ax2.grid(True, which="both", ls="--", alpha=0.35)
        ax2.legend(loc="upper right", fontsize=9.0, ncol=2)

        # Panel C: Mode Family Attractor in Cleaned 0p20 (Mode 1 vs Mode 2 Traveling Wave Pair)
        ax3 = axs[1, 0]
        # Plot 10 ms trajectory (1152 points)
        pts = slice(1000, 2200)
        time_ms = (t_20[pts] - t_20[pts.start]) * 1000.0
        sc = ax3.scatter(a_20[pts, 0], a_20[pts, 1], c=time_ms, cmap="viridis", s=10, alpha=0.7)
        ax3.plot(a_20[pts, 0], a_20[pts, 1], color="navy", lw=1.0, alpha=0.5)
        ax3.set_xlabel(r"Mode 1 Elevation $\phi_1 \cdot a_1 / \sqrt{N}$ [$\mu$m]", fontsize=11)
        ax3.set_ylabel(r"Mode 2 Elevation $\phi_2 \cdot a_2 / \sqrt{N}$ [$\mu$m]", fontsize=11)
        ax3.set_title(r"C. Cleaned 0p20: Primary Traveling Wave Pair $(a_1, a_2)$ Limit Cycle", fontweight="bold", fontsize=12)
        ax3.axis("equal")
        ax3.grid(True, ls="--", alpha=0.35)
        cb = plt.colorbar(sc, ax=ax3, pad=0.02)
        cb.set_label("Time [ms]", fontsize=10)

        # Panel D: Variance Ratio: Spatial Mean vs Higher Modes
        ax4 = axs[1, 1]
        var_20 = [np.var(mean_20)] + [np.var(a_20[:, k]) for k in range(5)]
        var_08 = [np.var(mean_08)] + [np.var(a_08[:, k]) for k in range(5)]
        labels = [r"$\bar{h}$", "$M_1$", "$M_2$", "$M_3$", "$M_4$", "$M_5$"]
        
        x_idx = np.arange(len(labels))
        w = 0.35
        # Normalize each by total variance of (mean + 5 modes)
        tot_20 = sum(var_20)
        tot_08 = sum(var_08)
        pct_20 = [v / tot_20 * 100 for v in var_20]
        pct_08 = [v / tot_08 * 100 for v in var_08]

        ax4.bar(x_idx - w/2, pct_20, width=w, label=f"0p20 Cleaned (Total: {tot_20:.1f} $\\mu$m$^2$)", color="#1f77b4", edgecolor="black")
        ax4.bar(x_idx + w/2, pct_08, width=w, label=f"0p08 (Total: {tot_08:.1f} $\\mu$m$^2$)", color="#ff7f0e", edgecolor="black")
        ax4.set_xticks(x_idx)
        ax4.set_xticklabels(labels, fontsize=11)
        ax4.set_ylabel("Variance Contribution [%]", fontsize=11)
        ax4.set_title("D. Variance Distribution: Spatial Mean vs Wave Modes", fontweight="bold", fontsize=12)
        ax4.grid(True, axis="y", ls="--", alpha=0.35)
        ax4.legend(loc="upper right", fontsize=10)
        for i in range(len(labels)):
            ax4.text(x_idx[i] - w/2, pct_20[i] + 1, f"{pct_20[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
            ax4.text(x_idx[i] + w/2, pct_08[i] + 1, f"{pct_08[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        ax4.set_ylim(0, 70)

        plt.savefig(OUT_FIG, dpi=180)
        print(f"Saved: {OUT_FIG}")

if __name__ == "__main__":
    main()

