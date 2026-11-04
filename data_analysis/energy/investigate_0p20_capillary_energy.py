"""Forensic investigation of capillary energy in the cleaned 0p20 dataset.

Cleaned file: /home/jonas/ucsd_thesis/11232025_a_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5
Raw file: /home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5
POD file: /home/jonas/ucsd_thesis/reduced_data/0p20/Ca_ac_0p001660_rep1/pod_2d_r1000.h5
"""

import sys
from pathlib import Path
import h5py
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.energy.capillary_energy import (
    length_scale,
    field_energies,
)

CLEANED_PATH = Path("/home/jonas/ucsd_thesis/11232025_a_0.2vpp_data_roi-none_cal-true_cleaned_hp50hz.hdf5")
POD_PATH = Path("/home/jonas/ucsd_thesis/reduced_data/0p20/Ca_ac_0p001660_rep1/pod_2d_r1000.h5")
GAMMA = 0.0728  # N/m

def main():
    print(f"Loading cleaned file: {CLEANED_PATH}")
    print(f"Loading POD file: {POD_PATH}")

    with h5py.File(CLEANED_PATH, "r") as f_cln, h5py.File(POD_PATH, "r") as f_pod:
        # Grids & units
        x_raw = f_cln["meta/x"][:]
        y_raw = f_cln["meta/y"][:]
        x_units = f_cln["meta"].attrs.get("x_units", "microns")
        y_units = f_cln["meta"].attrs.get("y_units", "microns")
        z_units = f_cln["meta"].attrs.get("z_units", "microns")
        
        x_m = x_raw * length_scale(str(x_units))
        y_m = y_raw * length_scale(str(y_units))
        z_scale = length_scale(str(z_units))

        # Check grid match
        np.testing.assert_allclose(x_raw, f_pod["grid/x"][:])
        np.testing.assert_allclose(y_raw, f_pod["grid/y"][:])

        # POD items
        modes = f_pod["pod/modes"][:]       # (1000, 200, 200) float32
        coeffs = f_pod["reduced/coefficients"] # (100001, 1000) float32

        # Evaluate 500 frames uniformly spaced
        n_frames = len(f_pod["grid/time"])
        frame_indices = np.linspace(0, n_frames - 1, 500, dtype=int)

        ranks_to_test = [5, 10, 20, 50, 100, 200, 500, 1000]

        e_cln_exact = []
        e_cln_quad = []

        e_pod_exact = {r: [] for r in ranks_to_test}
        e_pod_quad = {r: [] for r in ranks_to_test}

        e_res_exact = {r: [] for r in ranks_to_test}
        e_res_quad = {r: [] for r in ranks_to_test}

        batch_size = 50
        print(f"Evaluating {len(frame_indices)} frames...")

        for b_start in range(0, len(frame_indices), batch_size):
            b_indices = frame_indices[b_start:b_start + batch_size]
            b_count = len(b_indices)

            cln_batch = np.empty((b_count, 200, 200), dtype=np.float64)
            for i, idx in enumerate(b_indices):
                key = f"{idx:05d}"
                cln_batch[i] = f_cln["main"][key][:] * z_scale

            # 1. Cleaned field energies
            q_cln, ex_cln = field_energies(cln_batch, x_m, y_m)
            e_cln_quad.extend(GAMMA * q_cln)
            e_cln_exact.extend(GAMMA * ex_cln)

            # 2. POD reconstructions
            a_batch = (coeffs[b_indices, :] * z_scale).astype(np.float64)

            for r in ranks_to_test:
                rec = (a_batch[:, :r] @ modes[:r].reshape(r, -1).astype(np.float64)).reshape(b_count, 200, 200)
                q_p, ex_p = field_energies(rec, x_m, y_m)
                e_pod_quad[r].extend(GAMMA * q_p)
                e_pod_exact[r].extend(GAMMA * ex_p)

                res = cln_batch - rec
                q_r, ex_r = field_energies(res, x_m, y_m)
                e_res_quad[r].extend(GAMMA * q_r)
                e_res_exact[r].extend(GAMMA * ex_r)

        mean_cln_ex = np.mean(e_cln_exact) * 1e12
        mean_cln_q = np.mean(e_cln_quad) * 1e12

        print("\n================ 0p20 CLEANED CAPILLARY ENERGY ================")
        print(f"Cleaned Dataset (Rep 1, 50 Hz High-Passed):")
        print(f"  Exact Geometric Energy:  {mean_cln_ex:.2f} pJ")
        print(f"  Quadratic Approximation: {mean_cln_q:.2f} pJ")
        print(f"  Exact / Quadratic Ratio: {mean_cln_ex/mean_cln_q*100:.2f}% (Nonlinear reduction: {100*(1-mean_cln_ex/mean_cln_q):.2f}%)")

        print(f"\nPOD Energy Capture by Rank (Own Basis):")
        print(f"{'Rank':>6} | {'POD Exact (pJ)':>15} | {'% of Cleaned':>14} | {'Residual (pJ)':>14} | {'POD Quad (pJ)':>15} | {'Quad %':>10}")
        print("-" * 88)
        for r in ranks_to_test:
            p_ex = np.mean(e_pod_exact[r]) * 1e12
            res_ex = np.mean(e_res_exact[r]) * 1e12
            pct_ex = p_ex / mean_cln_ex * 100
            p_q = np.mean(e_pod_quad[r]) * 1e12
            pct_q = p_q / mean_cln_q * 100
            print(f"{r:6d} | {p_ex:15.2f} | {pct_ex:13.1f}% | {res_ex:14.2f} | {p_q:15.2f} | {pct_q:9.1f}%")

        # Frame 0 detailed inspection:
        cln_0 = f_cln["main"]["00000"][:] * z_scale
        a_0 = (coeffs[0, :] * z_scale).astype(np.float64)
        rec_1000 = (a_0 @ modes.reshape(1000, -1).astype(np.float64)).reshape(200, 200)
        res_0 = cln_0 - rec_1000

        print(f"\nFrame 0 Height Amplitudes (Standard Deviations):")
        print(f"  Cleaned Field STD:    {cln_0.std()*1e6:.4f} um")
        print(f"  POD Rank-1000 STD:    {rec_1000.std()*1e6:.4f} um  ({(rec_1000.std()/cln_0.std())**2*100:.2f}% of height variance)")
        print(f"  Residual STD:         {res_0.std()*1e6:.4f} um  ({(res_0.std()/cln_0.std())**2*100:.2f}% of height variance)")

        gx_cln, gy_cln = np.gradient(cln_0, x_m, y_m)
        gx_pod, gy_pod = np.gradient(rec_1000, x_m, y_m)
        gx_res, gy_res = np.gradient(res_0, x_m, y_m)

        slope_sq_cln = np.mean(gx_cln**2 + gy_cln**2)
        slope_sq_pod = np.mean(gx_pod**2 + gy_pod**2)
        slope_sq_res = np.mean(gx_res**2 + gy_res**2)

        print(f"\nFrame 0 Mean Squared Spatial Slopes <|grad h|^2>:")
        print(f"  Cleaned Slope^2:   {slope_sq_cln:.6e} rad^2 (RMS slope: {np.sqrt(slope_sq_cln):.4f} rad)")
        print(f"  POD Rank-1000:     {slope_sq_pod:.6e} rad^2 ({slope_sq_pod/slope_sq_cln*100:.2f}%)")
        print(f"  Residual (Noise):  {slope_sq_res:.6e} rad^2 ({slope_sq_res/slope_sq_cln*100:.2f}%)")

if __name__ == "__main__":
    main()

