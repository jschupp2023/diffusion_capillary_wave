"""Forensic investigation of raw vs. POD capillary energy discrepancy.

Case: 0p08, Ca_ac_0p000664_rep8.
Raw file: /home/jonas/ucsd_thesis/11222025_h_0.08vpp_data_roi-none_cal-true.hdf5
POD file: /home/jonas/ucsd_thesis/reduced_data/0p08/Ca_ac_0p000664_rep8/pod_2d_r1000.h5
"""

import sys
from pathlib import Path
import json
import h5py
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.energy.capillary_energy import (
    length_scale,
    quadrature_weights,
    field_energies,
)

RAW_PATH = Path("/home/jonas/ucsd_thesis/11222025_h_0.08vpp_data_roi-none_cal-true.hdf5")
POD_PATH = Path("/home/jonas/ucsd_thesis/reduced_data/0p08/Ca_ac_0p000664_rep8/pod_2d_r1000.h5")
GAMMA = 0.0728  # N/m

def main():
    print(f"Loading raw file: {RAW_PATH}")
    print(f"Loading POD file: {POD_PATH}")

    with h5py.File(RAW_PATH, "r") as f_raw, h5py.File(POD_PATH, "r") as f_pod:
        # 1. Grids & units
        x_raw = f_raw["meta/x"][:]
        y_raw = f_raw["meta/y"][:]
        x_units = f_raw["meta"].attrs.get("x_units", "microns")
        y_units = f_raw["meta"].attrs.get("y_units", "microns")
        z_units = f_raw["meta"].attrs.get("z_units", "microns")
        
        x_m = x_raw * length_scale(str(x_units))
        y_m = y_raw * length_scale(str(y_units))
        z_scale = length_scale(str(z_units))

        # Check grid match
        np.testing.assert_allclose(x_raw, f_pod["grid/x"][:])
        np.testing.assert_allclose(y_raw, f_pod["grid/y"][:])

        # 2. POD items
        temporal_mean_field = f_pod["preprocessing/temporal_mean_field"][:] * z_scale  # (200, 200) in meters
        spatial_means = f_pod["preprocessing/frame_spatial_mean"][:] * z_scale        # (100001,) in meters
        modes = f_pod["pod/modes"][:]                                                  # (1000, 200, 200) float32
        coeffs = f_pod["reduced/coefficients"]                                         # (100001, 1000) float32 (units z)

        # 3. Sample frames (evaluate 500 frames uniformly spaced to get stable statistics)
        n_frames = len(f_pod["grid/time"])
        frame_indices = np.linspace(0, n_frames - 1, 500, dtype=int)

        print(f"Evaluating {len(frame_indices)} frames...")

        # Pre-calculate energy of temporal mean field alone (static surface)
        tmean_3d = temporal_mean_field[None, :, :]
        quad_tmean, exact_tmean = field_energies(tmean_3d, x_m, y_m)
        e_tmean_quad = GAMMA * quad_tmean[0]
        e_tmean_exact = GAMMA * exact_tmean[0]
        print(f"\n--- Static Temporal Mean Field Energy ---")
        print(f"  Exact: {e_tmean_exact:.4e} J  ({e_tmean_exact*1e12:.2f} pJ)")
        print(f"  Quad:  {e_tmean_quad:.4e} J   ({e_tmean_quad*1e12:.2f} pJ)")

        # Initialize energy accumulators
        ranks_to_test = [5, 10, 20, 50, 100, 200, 500, 1000]
        
        e_raw_exact = []
        e_raw_quad = []
        
        e_raw_detrend_exact = []  # raw minus temporal mean
        e_raw_detrend_quad = []
        
        e_pod_exact = {r: [] for r in ranks_to_test}
        e_pod_quad = {r: [] for r in ranks_to_test}
        
        e_pod_with_tmean_exact = {r: [] for r in ranks_to_test}
        e_pod_with_tmean_quad = {r: [] for r in ranks_to_test}

        e_residual_exact = {r: [] for r in ranks_to_test}
        e_residual_quad = {r: [] for r in ranks_to_test}

        batch_size = 50
        for b_start in range(0, len(frame_indices), batch_size):
            b_indices = frame_indices[b_start:b_start + batch_size]
            b_count = len(b_indices)
            
            # Load raw frames
            raw_batch = np.empty((b_count, 200, 200), dtype=np.float64)
            for i, idx in enumerate(b_indices):
                key = f"{idx:05d}"
                raw_batch[i] = f_raw["main"][key][:] * z_scale
                
            # 1. Raw energy
            q_raw, ex_raw = field_energies(raw_batch, x_m, y_m)
            e_raw_quad.extend(GAMMA * q_raw)
            e_raw_exact.extend(GAMMA * ex_raw)

            # 2. Raw minus temporal mean
            detrend_batch = raw_batch - temporal_mean_field[None, :, :]
            q_det, ex_det = field_energies(detrend_batch, x_m, y_m)
            e_raw_detrend_quad.extend(GAMMA * q_det)
            e_raw_detrend_exact.extend(GAMMA * ex_det)

            # 3. POD reconstructions for various ranks
            # Coeffs for this batch: shape (b_count, 1000)
            a_batch = (coeffs[b_indices, :] * z_scale).astype(np.float64)

            for r in ranks_to_test:
                # Reconstruct fluctuating field: sum_k a_k * phi_k
                rec_fluct = (a_batch[:, :r] @ modes[:r].reshape(r, -1).astype(np.float64)).reshape(b_count, 200, 200)
                
                # POD alone (fluctuating)
                q_p, ex_p = field_energies(rec_fluct, x_m, y_m)
                e_pod_quad[r].extend(GAMMA * q_p)
                e_pod_exact[r].extend(GAMMA * ex_p)

                # POD + temporal mean restored
                rec_with_tmean = rec_fluct + temporal_mean_field[None, :, :]
                q_pt, ex_pt = field_energies(rec_with_tmean, x_m, y_m)
                e_pod_with_tmean_quad[r].extend(GAMMA * q_pt)
                e_pod_with_tmean_exact[r].extend(GAMMA * ex_pt)

                # Residual: (detrend_batch - rec_fluct)
                residual = detrend_batch - rec_fluct
                q_res, ex_res = field_energies(residual, x_m, y_m)
                e_residual_quad[r].extend(GAMMA * q_res)
                e_residual_exact[r].extend(GAMMA * ex_res)

        print("\n================ ENERGY COMPARISON (Time-Averaged) ================")
        mean_raw_ex = np.mean(e_raw_exact) * 1e12
        mean_raw_q = np.mean(e_raw_quad) * 1e12
        print(f"Raw (as stored):")
        print(f"  Exact Energy:     {mean_raw_ex:.2f} pJ")
        print(f"  Quadratic Energy: {mean_raw_q:.2f} pJ")
        print(f"  Ratio Exact/Quad: {mean_raw_ex/mean_raw_q*100:.2f}%")

        mean_det_ex = np.mean(e_raw_detrend_exact) * 1e12
        mean_det_q = np.mean(e_raw_detrend_quad) * 1e12
        print(f"\nRaw minus Temporal Mean Field (True Fluctuations + Noise):")
        print(f"  Exact Energy:     {mean_det_ex:.2f} pJ  ({mean_det_ex/mean_raw_ex*100:.2f}% of Raw)")
        print(f"  Quadratic Energy: {mean_det_q:.2f} pJ  ({mean_det_q/mean_raw_q*100:.2f}% of Raw)")
        print(f"  Energy in static mean & cross terms: {mean_raw_ex - mean_det_ex:.2f} pJ (Static mean alone: {e_tmean_exact*1e12:.2f} pJ)")

        print(f"\nPOD Reconstructions (Fluctuations only):")
        print(f"{'Rank':>6} | {'POD Exact (pJ)':>15} | {'% of Raw':>10} | {'% of Detrended':>15} | {'Residual (pJ)':>14} | {'POD+TMean (pJ)':>15} | {'POD+TMean % Raw':>16}")
        print("-" * 105)
        for r in ranks_to_test:
            p_ex = np.mean(e_pod_exact[r]) * 1e12
            res_ex = np.mean(e_residual_exact[r]) * 1e12
            pt_ex = np.mean(e_pod_with_tmean_exact[r]) * 1e12
            pct_raw = p_ex / mean_raw_ex * 100
            pct_det = p_ex / mean_det_ex * 100
            pct_pt_raw = pt_ex / mean_raw_ex * 100
            print(f"{r:6d} | {p_ex:15.2f} | {pct_raw:9.1f}% | {pct_det:14.1f}% | {res_ex:14.2f} | {pt_ex:15.2f} | {pct_pt_raw:15.1f}%")

        # Check Parseval / Orthogonality of gradients:
        print("\n--- Gradient Orthogonality Check: E(detrended) vs E(POD) + E(residual) ---")
        for r in [10, 100, 1000]:
            p_q = np.mean(e_pod_quad[r]) * 1e12
            res_q = np.mean(e_residual_quad[r]) * 1e12
            sum_q = p_q + res_q
            print(f"Rank {r:4d}: POD_quad={p_q:.2f} pJ + Res_quad={res_q:.2f} pJ = {sum_q:.2f} pJ (Detrended={mean_det_q:.2f} pJ, discrepancy={sum_q - mean_det_q:.2f} pJ)")

        # Analyze spatial wavelength of residual vs POD
        print("\n--- Investigating Spatial Derivative Properties ---")
        # Check single frame
        raw_0 = f_raw["main"]["00000"][:] * z_scale
        det_0 = raw_0 - temporal_mean_field
        a_0 = (coeffs[0, :] * z_scale).astype(np.float64)
        rec_1000 = (a_0 @ modes.reshape(1000, -1).astype(np.float64)).reshape(200, 200)
        res_0 = det_0 - rec_1000

        print(f"Frame 0 Height amplitudes (Standard Deviations):")
        print(f"  Detrended Raw Field STD:  {det_0.std()*1e6:.4f} um")
        print(f"  POD Rank-1000 Rec STD:    {rec_1000.std()*1e6:.4f} um  ({(rec_1000.std()/det_0.std())**2*100:.2f}% of height variance)")
        print(f"  Residual Field STD:       {res_0.std()*1e6:.4f} um  ({(res_0.std()/det_0.std())**2*100:.2f}% of height variance)")
        
        gx_pod, gy_pod = np.gradient(rec_1000, x_m, y_m)
        gx_res, gy_res = np.gradient(res_0, x_m, y_m)
        gx_raw, gy_raw = np.gradient(det_0, x_m, y_m)

        slope_sq_pod = np.mean(gx_pod**2 + gy_pod**2)
        slope_sq_res = np.mean(gx_res**2 + gy_res**2)
        slope_sq_raw = np.mean(gx_raw**2 + gy_raw**2)

        print(f"\nFrame 0 Mean Squared Spatial Slopes <|grad h|^2>:")
        print(f"  Detrended Raw Slope^2: {slope_sq_raw:.6e} rad^2")
        print(f"  POD Rank-1000 Slope^2: {slope_sq_pod:.6e} rad^2  ({slope_sq_pod/slope_sq_raw*100:.2f}% of total slope variance!)")
        print(f"  Residual Slope^2:      {slope_sq_res:.6e} rad^2  ({slope_sq_res/slope_sq_raw*100:.2f}% of total slope variance!)")

if __name__ == "__main__":
    main()

