"""Comprehensive Diagnostic Suite for 0p20 Integer-Cycle Phase Restoration.

Generates:
1. Multi-Frame Diagnostic Suite (Clean Frame 2300, Cliff Frames 1553, 1009, 23000):
   - 2D surface maps (Raw, Old Poisson, New Integer)
   - Difference maps showing exactly where modifications occur
   - Cross-sectional profile plots through cliffs and wave crests
2. POD Trajectory Diagnostic Suite:
   - 3-Way trajectory overlays (Raw vs Old Poisson vs New Integer)
   - Spike episode (Frame 1553) vs Clean episode (Frame 2300)
   - Lag-1 increments da_k(t)
   - Mahalanobis anomaly distance metrics
3. Wavenumber Spectral Analysis E(k):
   - 2D isotropic radial wavenumber spectrum comparing Raw, Old Poisson, New Integer
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

from data_analysis.pod.pod_2d import _frame_key
from data_analysis.pod.phase_unwrap_gradient_poisson import (
    DEFAULT_Q,
    DEFAULT_TAU,
    restore_frame_poisson,
)

RAW_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5")
CLEAN_H5 = Path("/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true_cleaned.hdf5")
SHARED_POD_H5 = Path("/home/jonas/ucsd_thesis/reduced_data/0p20/shared_pod/shared_pod_r1000.h5")
REP13_POD_H5 = Path("/home/jonas/ucsd_thesis/reduced_data/0p20/Ca_ac_0p001660_rep13/pod_2d_r1000.h5")
ARTIFACT_DIR = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321")
OUTPUT_DIR = Path("runs/comparison_0p20_integer_cleaned")
TAU_CLEAN = 4.0


def generate_frame_diagnostics(output_dir: Path) -> dict[str, any]:
    """Render detailed 4-panel diagnostic figures for key test frames."""
    print("Generating frame-level diagnostics...", flush=True)
    frames_to_test = [
        {"fn": 2300, "row": 97, "label": "Clean Frame with Steep Physical Wave Crest"},
        {"fn": 1553, "row": 53, "label": "Catastrophic 75 um Phase Cliff"},
        {"fn": 1009, "row": 100, "label": "Catastrophic 25 um Phase Cliff"},
        {"fn": 23000, "row": 80, "label": "Catastrophic 20 um Phase Cliff"},
    ]
    
    results = {}
    with h5py.File(RAW_H5, "r") as h5:
        main = h5["main"]
        x_um = np.asarray(h5["meta"]["x"], dtype=np.float64)
        y_um = np.asarray(h5["meta"]["y"], dtype=np.float64)
        
        for item in frames_to_test:
            fn = item["fn"]
            row = item["row"]
            label = item["label"]
            raw_frame = np.asarray(main[_frame_key(fn)], dtype=np.float32)
            
            # 1. Old Poisson tau=1.8
            res_poiss = restore_frame_poisson(raw_frame, q=DEFAULT_Q, tau=1.8, method="poisson")
            # 2. New Integer tau=4.0
            res_int = restore_frame_poisson(raw_frame, q=DEFAULT_Q, tau=TAU_CLEAN, method="integer")
            
            diff_poiss = np.abs(res_poiss.restored - raw_frame)
            diff_int = np.abs(res_int.restored - raw_frame)
            
            results[str(fn)] = {
                "fn": fn,
                "label": label,
                "raw_max_grad": float(max(np.max(np.abs(np.diff(raw_frame, axis=1))), np.max(np.abs(np.diff(raw_frame, axis=0))))),
                "poiss_was_mod": bool(res_poiss.was_modified),
                "poiss_max_diff": float(np.max(diff_poiss)),
                "poiss_rms_diff": float(np.sqrt(np.mean(diff_poiss**2))),
                "int_was_mod": bool(res_int.was_modified),
                "int_max_diff": float(np.max(diff_int)),
                "int_rms_diff": float(np.sqrt(np.mean(diff_int**2))),
                "int_pixels_mod": int(res_int.n_pixels_modified),
                "int_pct_pixels_mod": float(res_int.pct_pixels_modified),
            }
            
            # Render 2x3 comprehensive diagnostic figure
            fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
            
            # Row 0: Surfaces
            vmin = min(raw_frame.min(), res_int.restored.min())
            vmax = max(raw_frame.max(), res_int.restored.max())
            
            im0 = axes[0, 0].imshow(raw_frame, origin="lower", cmap="viridis", extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f"Raw Surface (Frame {fn})\nmax |grad| = {results[str(fn)]['raw_max_grad']:.1f} um", fontweight="bold")
            axes[0, 0].set_ylabel("y [um]")
            fig.colorbar(im0, ax=axes[0, 0], label="Height [um]")
            axes[0, 0].axhline(y_um[row], color="red", ls="--", alpha=0.7, lw=1.2, label=f"Row {row}")
            axes[0, 0].legend(loc="upper right", fontsize=9)
            
            im1 = axes[0, 1].imshow(res_poiss.restored, origin="lower", cmap="viridis", extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f"Old Poisson (tau=1.8, continuous)\nmax |grad| = {res_poiss.rest_max_dx:.1f} um", fontweight="bold")
            fig.colorbar(im1, ax=axes[0, 1], label="Height [um]")
            
            im2 = axes[0, 2].imshow(res_int.restored, origin="lower", cmap="viridis", extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], vmin=vmin, vmax=vmax)
            axes[0, 2].set_title(f"New Integer-Cycle (tau={TAU_CLEAN:.1f})\nmax |grad| = {res_int.rest_max_dx:.1f} um", fontweight="bold")
            fig.colorbar(im2, ax=axes[0, 2], label="Height [um]")
            
            # Row 1: Profile and Differences
            axes[1, 0].plot(x_um, raw_frame[row], "r-", label="Raw", lw=1.5, alpha=0.8)
            axes[1, 0].plot(x_um, res_poiss.restored[row], "b--", label="Old Poisson (tau=1.8)", lw=1.5)
            axes[1, 0].plot(x_um, res_int.restored[row], "g:", label=f"New Integer (tau={TAU_CLEAN:.1f})", lw=2.0)
            axes[1, 0].set_title(f"Height Cross-Section at Row {row}", fontweight="bold")
            axes[1, 0].set_xlabel("x [um]")
            axes[1, 0].set_ylabel("Height [um]")
            axes[1, 0].legend(fontsize=9)
            axes[1, 0].grid(True, alpha=0.3)
            
            im3 = axes[1, 1].imshow(diff_poiss, origin="lower", cmap="hot", extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], vmin=0, vmax=max(2.0, float(np.percentile(diff_poiss, 99))))
            axes[1, 1].set_title(f"Old Poisson |Difference|\nMax = {results[str(fn)]['poiss_max_diff']:.2f} um (Global leak!)", fontweight="bold")
            axes[1, 1].set_xlabel("x [um]")
            axes[1, 1].set_ylabel("y [um]")
            fig.colorbar(im3, ax=axes[1, 1], label="|Difference| [um]")
            
            im4 = axes[1, 2].imshow(diff_int, origin="lower", cmap="hot", extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]], vmin=0, vmax=max(2.0, float(np.percentile(diff_int, 99))))
            axes[1, 2].set_title(f"New Integer |Difference|\nModified: {res_int.pct_pixels_modified:.1f}% pixels", fontweight="bold")
            axes[1, 2].set_xlabel("x [um]")
            fig.colorbar(im4, ax=axes[1, 2], label="|Difference| [um]")
            
            fig_path = output_dir / f"frame_comparison_{fn}.png"
            fig.savefig(fig_path, dpi=160)
            plt.close(fig)
            print(f"Saved {fig_path.name}")
            
    with (output_dir / "frame_diagnostics_summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    return results


def generate_trajectory_diagnostics(output_dir: Path) -> None:
    """Generate 3-way POD trajectory comparisons around Frame 1553 and Frame 2300."""
    print("Generating POD trajectory comparisons...", flush=True)
    with h5py.File(SHARED_POD_H5, "r") as f_pod:
        modes = f_pod["pod"]["modes"][:21] # (21, 200, 200)
        modes_flat = modes.reshape(21, -1) # (21, 40000)
        
    with h5py.File(REP13_POD_H5, "r") as f_rep:
        temporal_mean = f_rep["preprocessing"]["temporal_mean_field"][:]
        temporal_mean_flat = temporal_mean.ravel()
        
    scaler_path = Path("runs/comparison_0p20_cleaned/standardization_scaler_r20.json")
    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    mean_scale = np.asarray(scaler["mean"], dtype=np.float64)
    scale_scale = np.asarray(scaler["scale"], dtype=np.float64)
    
    episodes = [
        {"center": 1553, "window": 100, "title": "Episode Around Frame 1553 (75 um Cliff)"},
        {"center": 2300, "window": 100, "title": "Episode Around Frame 2300 (Clean Frame Verification)"},
    ]
    
    with h5py.File(RAW_H5, "r") as h5, h5py.File(CLEAN_H5, "r") as h5_clean:
        main = h5["main"]
        main_clean = h5_clean["main"]
        time_arr = np.asarray(h5["meta"]["t"])
        
        for ep in episodes:
            c = ep["center"]
            w = ep["window"]
            start = max(0, c - w)
            end = min(len(time_arr), c + w)
            n_steps = end - start
            
            time_sub = time_arr[start:end]
            coords_raw = np.empty((n_steps, 21), dtype=np.float64)
            coords_poiss = np.empty((n_steps, 21), dtype=np.float64)
            coords_int = np.empty((n_steps, 21), dtype=np.float64)
            
            for i in range(n_steps):
                fn = start + i
                raw_f = np.asarray(main[_frame_key(fn)], dtype=np.float32)
                cln_f = np.asarray(main_clean[_frame_key(fn)], dtype=np.float32)
                
                # Projections
                fl_raw = (raw_f - temporal_mean).ravel()
                coords_raw[i, 0] = np.mean(raw_f) * 200.0
                coords_raw[i, 1:] = modes_flat[1:] @ fl_raw
                
                # Old Poisson tau=1.8
                h_p = restore_frame_poisson(raw_f, q=DEFAULT_Q, tau=1.8, method="poisson").restored
                fl_p = (h_p - temporal_mean).ravel()
                coords_poiss[i, 0] = np.mean(h_p) * 200.0
                coords_poiss[i, 1:] = modes_flat[1:] @ fl_p
                
                # New Integer tau=4.0 + Temporal DC Aligned
                fl_int = (cln_f - temporal_mean).ravel()
                coords_int[i, 0] = np.mean(cln_f) * 200.0
                coords_int[i, 1:] = modes_flat[1:] @ fl_int
                
            # Standardize coordinates
            norm_raw = (coords_raw - mean_scale[:21]) / scale_scale[:21]
            norm_poiss = (coords_poiss - mean_scale[:21]) / scale_scale[:21]
            norm_int = (coords_int - mean_scale[:21]) / scale_scale[:21]
            
            # 1. Coordinate Overlays Plot (Modes 0, 1, 2, 3)
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
            plot_modes = [0, 1, 2, 3]
            for ax_idx, m in enumerate(plot_modes):
                axes[ax_idx].plot(time_sub * 1e3, norm_raw[:, m], "r-", lw=1.2, alpha=0.7, label="Original Raw")
                axes[ax_idx].plot(time_sub * 1e3, norm_poiss[:, m], "b--", lw=1.2, label="Old Poisson (tau=1.8)")
                axes[ax_idx].plot(time_sub * 1e3, norm_int[:, m], "g-", lw=1.5, label=f"Integer tau={TAU_CLEAN:.1f} + Temporal DC Locked")
                axes[ax_idx].axvline(time_arr[c] * 1e3, color="black", ls=":", lw=1.2)
                mode_label = f"$a_{{{m}}}(t) / \\sigma$" if m > 0 else r"$a_0(t) / \sigma$ (Spatial Mean)"
                axes[ax_idx].set_ylabel(mode_label, fontsize=11)
                axes[ax_idx].grid(True, alpha=0.3)
                if ax_idx == 0:
                    axes[ax_idx].legend(loc="upper right", fontsize=10, ncol=3)
                    axes[ax_idx].set_title(f"Shared POD Coordinate Trajectories: {ep['title']}", fontsize=12, fontweight="bold")
                    
            axes[-1].set_xlabel("Time [ms]", fontsize=11)
            fig_path = output_dir / f"pod_trajectory_overlay_{c}.png"
            fig.savefig(fig_path, dpi=160)
            plt.close(fig)
            print(f"Saved {fig_path.name}")
            
            # 2. Increments Delta a_k(t)
            da_raw = np.diff(norm_raw, axis=0)
            da_poiss = np.diff(norm_poiss, axis=0)
            da_int = np.diff(norm_int, axis=0)
            t_inc = time_sub[1:] * 1e3
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
            for ax_idx, m in enumerate(plot_modes):
                axes[ax_idx].plot(t_inc, da_raw[:, m], "r-", lw=1.2, alpha=0.7, label="Original Raw")
                axes[ax_idx].plot(t_inc, da_poiss[:, m], "b--", lw=1.2, label="Old Poisson (tau=1.8)")
                axes[ax_idx].plot(t_inc, da_int[:, m], "g-", lw=1.5, label=f"Integer tau={TAU_CLEAN:.1f} + Temporal DC Locked")
                axes[ax_idx].axvline(time_arr[c] * 1e3, color="black", ls=":", lw=1.2)
                inc_label = f"$\\Delta a_{{{m}}}(t) / \\sigma$" if m > 0 else r"$\Delta a_0(t) / \sigma$ (Spatial Mean Incr.)"
                axes[ax_idx].set_ylabel(inc_label, fontsize=11)
                axes[ax_idx].grid(True, alpha=0.3)
                if ax_idx == 0:
                    axes[ax_idx].legend(loc="upper right", fontsize=10, ncol=3)
                    axes[ax_idx].set_title(f"POD Coordinate Increments $\\Delta a(t)$: {ep['title']}", fontsize=12, fontweight="bold")
                    axes[ax_idx].legend(loc="upper right", fontsize=10, ncol=3)
                    axes[ax_idx].set_title(f"Lag-1 Coordinate Increments: {ep['title']}", fontsize=12, fontweight="bold")
                    
            axes[-1].set_xlabel("Time [ms]", fontsize=11)
            fig_path = output_dir / f"pod_increments_{c}.png"
            fig.savefig(fig_path, dpi=160)
            plt.close(fig)
            print(f"Saved {fig_path.name}")


def generate_radial_psd_comparison(output_dir: Path) -> None:
    """Compute and plot the 2D radial wavenumber spectrum E(k) across raw, old Poisson, and new integer."""
    print("Computing isotropic radial wavenumber spectrum E(k)...", flush=True)
    # Average radial spectrum across 200 sample frames
    indices = np.linspace(0, 100000, 200, dtype=int)
    
    with h5py.File(RAW_H5, "r") as h5, h5py.File(CLEAN_H5, "r") as h5_clean:
        main = h5["main"]
        clean_main = h5_clean["main"]
        dx_um = float(h5["meta"]["x"][1] - h5["meta"]["x"][0]) # 1.5684 um
        
        # Grid frequencies in rad/um
        N = 200
        kx = fft.fftfreq(N, d=dx_um) * 2 * np.pi
        ky = fft.fftfreq(N, d=dx_um) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        KR = np.sqrt(KX**2 + KY**2)
        
        k_bins = np.linspace(0, np.pi / dx_um, 100)
        k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
        
        Ek_raw_sum = np.zeros(len(k_centers))
        Ek_poiss_sum = np.zeros(len(k_centers))
        Ek_int_sum = np.zeros(len(k_centers))
        
        for idx in indices:
            raw_f = np.asarray(main[_frame_key(idx)], dtype=np.float32)
            # Remove mean
            f_raw_0 = raw_f - np.mean(raw_f)
            h_p_0 = restore_frame_poisson(raw_f, q=DEFAULT_Q, tau=1.8, method="poisson").restored
            h_p_0 -= np.mean(h_p_0)
            h_int_0 = np.asarray(clean_main[_frame_key(idx)], dtype=np.float32)
            h_int_0 -= np.mean(h_int_0)
            
            # 2D FFT
            P_raw = np.abs(fft.fft2(f_raw_0))**2 / (N**2)
            P_p = np.abs(fft.fft2(h_p_0))**2 / (N**2)
            P_int = np.abs(fft.fft2(h_int_0))**2 / (N**2)
            
            # Radially integrate
            for b in range(len(k_centers)):
                mask = (KR >= k_bins[b]) & (KR < k_bins[b+1])
                if np.any(mask):
                    Ek_raw_sum[b] += np.sum(P_raw[mask])
                    Ek_poiss_sum[b] += np.sum(P_p[mask])
                    Ek_int_sum[b] += np.sum(P_int[mask])
                    
        Ek_raw = Ek_raw_sum / len(indices)
        Ek_poiss = Ek_poiss_sum / len(indices)
        Ek_int = Ek_int_sum / len(indices)
        
        # Plot E(k)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.loglog(k_centers, Ek_raw, "r-", lw=1.8, label="Original Raw", alpha=0.8)
        ax.loglog(k_centers, Ek_poiss, "b--", lw=1.8, label="Old Poisson (tau=1.8)")
        ax.loglog(k_centers, Ek_int, "g-", lw=2.0, label=f"New Integer (tau={TAU_CLEAN:.1f} + Temporal DC Locked)")
        
        # Add theoretical power-law slope reference k^(-7/4) (Zakharov-Filonenko capillary turbulence spectrum)
        k_ref = k_centers[(k_centers > 0.1) & (k_centers < 1.0)]
        E_ref = 1e4 * (k_ref ** (-7.0 / 4.0))
        ax.loglog(k_ref, E_ref, "k:", lw=1.5, label=r"Capillary Turbulence $k^{-7/4}$")
        
        ax.set_title("Spatial Wavenumber Energy Spectrum $E(k)$ on $0.20\\,\\mathrm{V}_{\\mathrm{pp}}$", fontsize=13, fontweight="bold")
        ax.set_xlabel("Wavenumber $k$ [$\\mathrm{rad}/\\mu\\mathrm{m}$]", fontsize=11)
        ax.set_ylabel("Spectral Energy $E(k)$ [$\\mu\\mathrm{m}^3$]", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=11)
        
        fig_path = output_dir / "spatial_wavenumber_spectrum_Ek.png"
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        print(f"Saved {fig_path.name}")


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_frame_diagnostics(output_dir)
    generate_trajectory_diagnostics(output_dir)
    generate_radial_psd_comparison(output_dir)
    print("\nAll diagnostics completed successfully!")


if __name__ == "__main__":
    main()
