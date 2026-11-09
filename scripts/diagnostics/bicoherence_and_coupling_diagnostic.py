"""Bicoherence and Peak-Background Activity Coupling Diagnostic.

Generates a compact, publication-quality 4x3 diagnostic figure for the four
high-frequency resonant peaks (651, 757, 867, and 982 Hz):

Row 1: Peak near 651 Hz (f_7 = 650.84 Hz)
Row 2: Peak near 757 Hz (f_8 = 756.97 Hz)
Row 3: Peak near 867 Hz (f_9 = 867.08 Hz)
Row 4: Peak near 982 Hz (f_10 = 982.15 Hz)

Panels per row:
1. Fixed-input bicoherence: b^2(f_k, f) vs f for triads (f_k, f, f_k + f).
2. Fixed-output bicoherence: b^2(f, f_k - f) vs f for triads (f, f_k - f, f_k) over 0 < f <= f_k / 2.
3. Background activity correlation: C_k(f) = Corr(|z_k(t)|^2, P_f(t)) for background bands centered at f,
   comparing observed signal (P_full) against reconstructed remainder (P_rem), with the target peak band masked.

Usage:
    python scripts/diagnostics/bicoherence_and_coupling_diagnostic.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt, hilbert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.frequency_hybrid.local_excess_remainder import decompose, load_bank, bank_filters

PYTHON_BIN = "/home/jonas/miniforge3/envs/diff-model-mit/bin/python"
DATA_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data/clean_red_data")
OUT_DIR = Path("runs/diagnostics/bicoherence_and_coupling")

# 11 Clean production resonant peak frequencies and linewidths
PEAK_FREQS = [
    76.01, 132.42, 206.91, 282.52, 364.22,
    455.71, 547.00, 650.84, 756.97, 867.08, 982.15
]
PEAK_LINEWIDTHS = [
    9.89, 14.58, 24.08, 26.48, 35.25,
    28.06, 41.58, 41.55, 87.28, 55.76, 75.90
]
BAND_EDGES = [
    [50.00, 100.19], [100.19, 163.47], [163.47, 237.31], [237.31, 318.17],
    [318.17, 406.06], [406.06, 493.94], [493.94, 587.11], [587.11, 689.07],
    [689.07, 799.80], [799.80, 919.34], [919.34, 1042.38]
]

TARGET_PEAKS = [
    {"index": 7, "f_k": 650.84, "gamma": 41.55, "label": "651 Hz", "band": [587.11, 689.07]},
    {"index": 8, "f_k": 756.97, "gamma": 87.28, "label": "757 Hz", "band": [689.07, 799.80]},
    {"index": 9, "f_k": 867.08, "gamma": 55.76, "label": "867 Hz", "band": [799.80, 919.34]},
    {"index": 10, "f_k": 982.15, "gamma": 75.90, "label": "982 Hz", "band": [919.34, 1042.38]},
]


def coarse(x: np.ndarray, factor: int) -> np.ndarray:
    n = len(x) // factor
    return np.mean(x[:n * factor].reshape(n, factor, -1), axis=1).squeeze()


def bootstrap_ci(curves: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap confidence interval across repetitions (axis 0)."""
    rng = np.random.default_rng(seed)
    n_reps = len(curves)
    boot_means = np.zeros((n_boot, curves.shape[1]))
    for b in range(n_boot):
        idx = rng.choice(n_reps, size=n_reps, replace=True)
        boot_means[b] = np.mean(curves[idx], axis=0)
    alpha = (1.0 - ci) / 2.0
    lower = np.percentile(boot_means, 100.0 * alpha, axis=0)
    upper = np.percentile(boot_means, 100.0 * (1.0 - alpha), axis=0)
    return lower, upper


def compute_repetition_bicoherence_and_correlation(
    eta_full: np.ndarray,
    eta_rem: np.ndarray,
    pow_k_dict: dict[int, np.ndarray],
    fs_native: float,
    downsample_factor: int = 16,
    nperseg: int = 1024,
    overlap_fraction: float = 0.75,
    w_strip: int = 1,
    f_scan: np.ndarray | None = None,
    w_band: float = 30.0,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(42)

    # Downsample central surface elevation for bicoherence analysis
    # Decimate by 16: 115200 Hz -> 7200 Hz (Nyquist 3600 Hz)
    sig_full_7200 = signal.decimate(eta_full, downsample_factor, ftype="fir", zero_phase=True)
    fs = fs_native / downsample_factor  # 7200 Hz
    df = fs / nperseg                  # 7.03125 Hz

    # Segment and window for Fourier transform
    step = int(nperseg * (1.0 - overlap_fraction))
    starts = range(0, len(sig_full_7200) - nperseg + 1, step)
    segs = np.stack([sig_full_7200[i:i + nperseg] for i in starts])
    segs = segs - segs.mean(axis=1, keepdims=True)
    w = signal.windows.hann(nperseg)
    Z = np.fft.rfft(segs * w[None, :], axis=1)  # [n_win, nperseg//2 + 1]
    n_win, n_bins = Z.shape

    # Pre-generate surrogate phase-randomized / circularly shifted Z_third
    shift_offsets = rng.integers(1, n_win, size=20)

    rep_results = {}

    for target in TARGET_PEAKS:
        p_idx = target["index"]
        f_k = target["f_k"]
        k0 = int(round(f_k / df))
        # 3-bin strip [k0 - w_strip, ..., k0 + w_strip]
        strip_bins = [k0 + offset for offset in range(-w_strip, w_strip + 1) if 0 <= k0 + offset < n_bins]

        # -------------------------------------------------------------
        # 1. Fixed-Input Bicoherence: b^2(f_k, f) vs f for triads (f_k, f, f_k + f)
        # -------------------------------------------------------------
        max_f = min(1500.0, (fs / 2.0) - f_k - 20.0)
        f1_bins = np.arange(2, int(round(max_f / df)))
        f1_freqs = f1_bins * df

        b2_in_actual = []
        b2_in_null = []

        for j in f1_bins:
            # Average over the narrow strip around f_k
            strip_b2 = []
            strip_null = []
            for kb in strip_bins:
                km = kb + j
                if km < n_bins:
                    prod = Z[:, kb] * Z[:, j]
                    third = Z[:, km]
                    num = np.abs(np.mean(prod * third.conj())) ** 2
                    den = np.mean(np.abs(prod) ** 2) * np.mean(np.abs(third) ** 2)
                    strip_b2.append(num / max(den, 1e-20))

                    # Surrogate: circular shift across windows for third frequency
                    null_draws = []
                    for s in shift_offsets:
                        th_roll = np.roll(third, s, axis=0)
                        num_null = np.abs(np.mean(prod * th_roll.conj())) ** 2
                        null_draws.append(num_null / max(den, 1e-20))
                    strip_null.append(np.mean(null_draws))

            b2_in_actual.append(np.mean(strip_b2) if strip_b2 else 0.0)
            b2_in_null.append(np.mean(strip_null) if strip_null else 0.0)

        # -------------------------------------------------------------
        # 2. Fixed-Output Bicoherence: b^2(f, f_k - f) vs f for 0 < f <= f_k / 2
        # -------------------------------------------------------------
        half_k0 = k0 // 2
        f2_bins = np.arange(1, half_k0 + 1)
        f2_freqs = f2_bins * df

        b2_out_actual = []
        b2_out_null = []

        for j in f2_bins:
            strip_b2 = []
            strip_null = []
            for kb in strip_bins:
                j2 = kb - j
                if 0 <= j2 < n_bins:
                    prod = Z[:, j] * Z[:, j2]
                    third = Z[:, kb]
                    num = np.abs(np.mean(prod * third.conj())) ** 2
                    den = np.mean(np.abs(prod) ** 2) * np.mean(np.abs(third) ** 2)
                    strip_b2.append(num / max(den, 1e-20))

                    null_draws = []
                    for s in shift_offsets:
                        th_roll = np.roll(third, s, axis=0)
                        num_null = np.abs(np.mean(prod * th_roll.conj())) ** 2
                        null_draws.append(num_null / max(den, 1e-20))
                    strip_null.append(np.mean(null_draws))

            b2_out_actual.append(np.mean(strip_b2) if strip_b2 else 0.0)
            b2_out_null.append(np.mean(strip_null) if strip_null else 0.0)

        # -------------------------------------------------------------
        # 3. Background Activity Correlation: C_k(f) = Corr(|z_k|^2, P_f)
        # -------------------------------------------------------------
        pow_k = pow_k_dict[p_idx]
        corr_full = []
        corr_rem = []
        corr_null = []

        # Circular shift shifts for envelope null
        env_shifts = rng.integers(90, len(pow_k) - 90, size=20)  # > 100 ms at fs_env=900

        for f_c in f_scan:
            low = max(10.0, f_c - w_band / 2.0)
            high = min(fs_native / 2.0 - 10.0, f_c + w_band / 2.0)
            sos = butter(2, [low, high], btype="bandpass", fs=fs_native, output="sos")

            filt_full = sosfiltfilt(sos, eta_full)
            filt_rem = sosfiltfilt(sos, eta_rem)

            p_full_native = np.abs(hilbert(filt_full)) ** 2
            p_rem_native = np.abs(hilbert(filt_rem)) ** 2

            p_full_down = coarse(p_full_native, 128)[:len(pow_k)]
            p_rem_down = coarse(p_rem_native, 128)[:len(pow_k)]

            std_pow = np.std(pow_k)
            std_full = np.std(p_full_down)
            std_rem = np.std(p_rem_down)

            c_f = float(np.corrcoef(pow_k, p_full_down)[0, 1]) if (std_pow > 1e-12 and std_full > 1e-12) else 0.0
            c_r = float(np.corrcoef(pow_k, p_rem_down)[0, 1]) if (std_pow > 1e-12 and std_rem > 1e-12) else 0.0

            corr_full.append(c_f)
            corr_rem.append(c_r)

            # Null correlation via circular time shifts
            null_c = [float(np.corrcoef(pow_k, np.roll(p_rem_down, s))[0, 1]) for s in env_shifts]
            corr_null.append(np.mean(null_c))

        rep_results[p_idx] = {
            "f1_freqs": f1_freqs,
            "b2_in_actual": np.asarray(b2_in_actual),
            "b2_in_null": np.asarray(b2_in_null),
            "f2_freqs": f2_freqs,
            "b2_out_actual": np.asarray(b2_out_actual),
            "b2_out_null": np.asarray(b2_out_null),
            "corr_full": np.asarray(corr_full),
            "corr_rem": np.asarray(corr_rem),
            "corr_null": np.asarray(corr_null),
        }

    return rep_results, Z, df, fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="0p10")
    parser.add_argument("--out-dir", type=str, default="runs/diagnostics/bicoherence_and_coupling")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Bicoherence and Peak-Background Activity Coupling Diagnostic")
    print(f"Condition: {args.condition} (All 16 Clean Repetitions)")
    print("=" * 70)

    # 1. Load Data Loader and Shared Basis
    shared_basis_path = DATA_ROOT / f"{args.condition}/shared_pod/shared_pod_r200.h5"
    with h5py.File(shared_basis_path, "r") as h5:
        pod_modes = h5["pod"]["modes"][:10]
    cy, cx = pod_modes.shape[1] // 2, pod_modes.shape[2] // 2
    center_weights = pod_modes[:, cy, cx]
    print(f"Loaded shared POD basis (rank 10). Center probe weights: {center_weights[:3]}...")

    loader = SharedPODTrainingData(
        experiment=args.condition, rank=10, data_root=str(DATA_ROOT),
        shared_basis=str(shared_basis_path),
        validation_reps=[13, 14], test_reps=[15, 16],
        spatial_mean_representation="embedded_in_pod", output_dtype="float64"
    )
    all_rep_names = [f"Ca_ac_0p000830_rep{r}" for r in range(1, 17)]
    fs_native = 1.0 / loader.native_dt
    edge_samples = 13152

    # Load oscillator bank models
    osc_dir = Path(f"runs/clean_production/{args.condition}/oscillators")
    artifacts, osc_models = load_bank(osc_dir)
    filters = bank_filters(osc_models, fs_native)
    print(f"Loaded 11-band oscillator bank from {osc_dir}.")

    # Background scan frequencies
    f_scan = np.arange(50.0, 1500.0, 15.0)

    all_rep_data = []
    t0 = time.time()

    # Process all 16 repetitions
    for r_idx, rep_name in enumerate(all_rep_names):
        print(f"Processing Repetition {r_idx + 1:02d}/16 ({rep_name})...", end="", flush=True)
        # Check if precomputed full and remainder exist in clean_production
        rem_dir = Path(f"runs/clean_production/{args.condition}/remainder_data/data")
        full_p = rem_dir / f"{rep_name}_full.npy"
        rem_p = rem_dir / f"{rep_name}_remainder.npy"

        if full_p.exists() and rem_p.exists():
            full_coords = np.load(full_p)
            rem_coords = np.load(rem_p)
        else:
            t_arr, raw_coords = loader.read_coordinates(rep_name, 0, 100001)
            full_coords, rem_coords, _ = decompose(raw_coords, osc_models, filters, edge_samples)

        eta_full = full_coords @ center_weights
        eta_rem = rem_coords @ center_weights

        # Load cached mode power envelopes
        cache_p = Path(f"runs/diagnostics/comprehensive_peak_coupling_v3_audited/cache/clean_{rep_name}.npz")
        with np.load(cache_p) as z:
            power_all = z["power"]  # [551, 11]

        pow_k_dict = {t["index"]: power_all[:, t["index"]] for t in TARGET_PEAKS}

        rng_rep = np.random.default_rng(1000 + r_idx)
        rep_res, Z_rep, df_fft, fs_down = compute_repetition_bicoherence_and_correlation(
            eta_full, eta_rem, pow_k_dict, fs_native,
            downsample_factor=16, nperseg=1024, overlap_fraction=0.75,
            w_strip=1, f_scan=f_scan, w_band=30.0, rng=rng_rep
        )
        all_rep_data.append(rep_res)
        print(f" done ({time.time() - t0:.1f}s)")

    # -------------------------------------------------------------
    # Compute Cross-Repetition Statistics (Mean, Bootstrap CI, Null Envelopes)
    # -------------------------------------------------------------
    print("\nComputing cross-repetition statistics and bootstrap confidence bands...")
    results_summary = {}

    for target in TARGET_PEAKS:
        p_idx = target["index"]
        f1_freqs = all_rep_data[0][p_idx]["f1_freqs"]
        f2_freqs = all_rep_data[0][p_idx]["f2_freqs"]

        # Stack across repetitions [16, N_f]
        b2_in_all = np.stack([d[p_idx]["b2_in_actual"] for d in all_rep_data])
        b2_in_null_all = np.stack([d[p_idx]["b2_in_null"] for d in all_rep_data])

        b2_out_all = np.stack([d[p_idx]["b2_out_actual"] for d in all_rep_data])
        b2_out_null_all = np.stack([d[p_idx]["b2_out_null"] for d in all_rep_data])

        corr_full_all = np.stack([d[p_idx]["corr_full"] for d in all_rep_data])
        corr_rem_all = np.stack([d[p_idx]["corr_rem"] for d in all_rep_data])
        corr_null_all = np.stack([d[p_idx]["corr_null"] for d in all_rep_data])

        # Means and bootstrap CIs
        b2_in_mean = np.mean(b2_in_all, axis=0)
        b2_in_ci_lo, b2_in_ci_hi = bootstrap_ci(b2_in_all)
        b2_in_null_mean = np.mean(b2_in_null_all, axis=0)
        b2_in_null_p95 = np.percentile(b2_in_null_all, 95, axis=0)

        b2_out_mean = np.mean(b2_out_all, axis=0)
        b2_out_ci_lo, b2_out_ci_hi = bootstrap_ci(b2_out_all)
        b2_out_null_mean = np.mean(b2_out_null_all, axis=0)
        b2_out_null_p95 = np.percentile(b2_out_null_all, 95, axis=0)

        corr_full_mean = np.mean(corr_full_all, axis=0)
        corr_full_ci_lo, corr_full_ci_hi = bootstrap_ci(corr_full_all)

        corr_rem_mean = np.mean(corr_rem_all, axis=0)
        corr_rem_ci_lo, corr_rem_ci_hi = bootstrap_ci(corr_rem_all)

        corr_null_mean = np.mean(corr_null_all, axis=0)
        corr_null_p95 = np.percentile(np.abs(corr_null_all), 95, axis=0)

        results_summary[p_idx] = {
            "target": target,
            "f1_freqs": f1_freqs,
            "b2_in_all": b2_in_all,
            "b2_in_mean": b2_in_mean,
            "b2_in_ci": (b2_in_ci_lo, b2_in_ci_hi),
            "b2_in_null_mean": b2_in_null_mean,
            "b2_in_null_p95": b2_in_null_p95,
            "f2_freqs": f2_freqs,
            "b2_out_all": b2_out_all,
            "b2_out_mean": b2_out_mean,
            "b2_out_ci": (b2_out_ci_lo, b2_out_ci_hi),
            "b2_out_null_mean": b2_out_null_mean,
            "b2_out_null_p95": b2_out_null_p95,
            "f_scan": f_scan,
            "corr_full_all": corr_full_all,
            "corr_full_mean": corr_full_mean,
            "corr_full_ci": (corr_full_ci_lo, corr_full_ci_hi),
            "corr_rem_all": corr_rem_all,
            "corr_rem_mean": corr_rem_mean,
            "corr_rem_ci": (corr_rem_ci_lo, corr_rem_ci_hi),
            "corr_null_mean": corr_null_mean,
            "corr_null_p95": corr_null_p95,
        }

    # Also compute reference PSD from rep 1
    f_psd, p_psd = signal.welch(signal.decimate(full_coords @ center_weights, 16, ftype="fir", zero_phase=True),
                                fs=7200.0, nperseg=2048, noverlap=1024)

    # -------------------------------------------------------------
    # Plot Compact 4x3 Diagnostic Figure
    # -------------------------------------------------------------
    print("Generating compact 4x3 publication diagnostic figure...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), layout="constrained")

    row_titles = [
        f"Mode 7: {TARGET_PEAKS[0]['f_k']:.1f} Hz (LineWidth {TARGET_PEAKS[0]['gamma']:.1f} Hz)",
        f"Mode 8: {TARGET_PEAKS[1]['f_k']:.1f} Hz (LineWidth {TARGET_PEAKS[1]['gamma']:.1f} Hz)",
        f"Mode 9: {TARGET_PEAKS[2]['f_k']:.1f} Hz (LineWidth {TARGET_PEAKS[2]['gamma']:.1f} Hz)",
        f"Mode 10: {TARGET_PEAKS[3]['f_k']:.1f} Hz (LineWidth {TARGET_PEAKS[3]['gamma']:.1f} Hz)",
    ]

    for row_idx, target in enumerate(TARGET_PEAKS):
        p_idx = target["index"]
        res = results_summary[p_idx]
        f_k = target["f_k"]
        band = target["band"]

        # =========================================================
        # Panel 1: Fixed-Input Bicoherence b^2(f_k, f)
        # =========================================================
        ax1 = axes[row_idx, 0]
        f1 = res["f1_freqs"]

        # Faint individual curves
        for r_curve in res["b2_in_all"]:
            ax1.plot(f1, r_curve, color="steelblue", alpha=0.15, lw=0.8)

        # Bold mean and bootstrap CI
        ax1.plot(f1, res["b2_in_mean"], color="navy", lw=2.0, label="Observed Mean b²")
        ax1.fill_between(f1, res["b2_in_ci"][0], res["b2_in_ci"][1], color="steelblue", alpha=0.25, label="95% Bootstrap CI")

        # Matched surrogate null envelope
        ax1.plot(f1, res["b2_in_null_mean"], "k--", lw=1.2, alpha=0.7, label="Surrogate Null Mean")
        ax1.plot(f1, res["b2_in_null_p95"], ":", color="crimson", lw=1.4, alpha=0.85, label="Simultaneous 95% Null")

        # Resonant peak markers & bandwidth shading
        for pf, gamma in zip(PEAK_FREQS, PEAK_LINEWIDTHS):
            ax1.axvline(pf, color="gray", ls=":", alpha=0.4, lw=0.9)
            ax1.axvspan(pf - gamma / 2.0, pf + gamma / 2.0, color="gray", alpha=0.08)

        # Mark target peak f_k
        ax1.axvline(f_k, color="darkorange", ls="--", lw=1.5, alpha=0.9)

        ax1.set_xlim(10, 1500)
        ax1.set_ylim(0, 0.25)
        ax1.set_ylabel(f"b²({f_k:.0f}, f)\n[{target['label']}]", fontsize=10, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        if row_idx == 0:
            ax1.set_title("Panel 1: Fixed-Input Bicoherence\nTriad (f_k, f, f_k + f)", fontsize=11, fontweight="bold")
        if row_idx == 3:
            ax1.set_xlabel("Probing Frequency f [Hz]", fontsize=10)
        if row_idx == 0:
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

        # Small inset PSD reference in row 0 panel 1
        if row_idx == 0:
            ins = ax1.inset_axes([0.48, 0.45, 0.48, 0.42])
            ins.loglog(f_psd, p_psd, "k-", lw=1.0)
            for pf in PEAK_FREQS:
                ins.axvline(pf, color="red", ls=":", alpha=0.5, lw=0.8)
            ins.set_xlim(50, 1500)
            ins.set_ylim(1e-5, 1e2)
            ins.set_xticks([100, 500, 1000])
            ins.set_xticklabels(["100", "500", "1k"], fontsize=7)
            ins.set_yticks([])
            ins.set_title("Surface PSD Ref", fontsize=7, pad=2)
            ins.grid(True, which="both", alpha=0.2)

        # =========================================================
        # Panel 2: Fixed-Output Bicoherence b^2(f, f_k - f)
        # =========================================================
        ax2 = axes[row_idx, 1]
        f2 = res["f2_freqs"]

        # Faint individual curves
        for r_curve in res["b2_out_all"]:
            ax2.plot(f2, r_curve, color="darkseagreen", alpha=0.2, lw=0.8)

        # Bold mean and bootstrap CI
        ax2.plot(f2, res["b2_out_mean"], color="forestgreen", lw=2.0, label="Observed Mean b²")
        ax2.fill_between(f2, res["b2_out_ci"][0], res["b2_out_ci"][1], color="darkseagreen", alpha=0.25, label="95% Bootstrap CI")

        # Matched surrogate null envelope
        ax2.plot(f2, res["b2_out_null_mean"], "k--", lw=1.2, alpha=0.7, label="Surrogate Null Mean")
        ax2.plot(f2, res["b2_out_null_p95"], ":", color="crimson", lw=1.4, alpha=0.85, label="Simultaneous 95% Null")

        # Mark candidate difference frequencies (f_k - f_lower) in (0, f_k / 2]
        for lower_f in PEAK_FREQS:
            if lower_f < f_k:
                diff_f = f_k - lower_f
                if diff_f <= (f_k / 2.0) + 1e-3:
                    ax2.axvline(diff_f, color="purple", ls=":", alpha=0.7, lw=1.2)
                    ax2.text(diff_f, 0.20, f"Δ{diff_f:.0f}", rotation=90, fontsize=7, color="purple", ha="right", va="top")

        ax2.set_xlim(0, f_k / 2.0)
        ax2.set_ylim(0, 0.25)
        ax2.set_ylabel(f"b²(f, {f_k:.0f}-f)\n[{target['label']}]", fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        if row_idx == 0:
            ax2.set_title("Panel 2: Fixed-Output Bicoherence\nTriad (f, f_k - f, f_k) for 0 < f ≤ f_k/2", fontsize=11, fontweight="bold")
        if row_idx == 3:
            ax2.set_xlabel("Sub-harmonic Frequency f [Hz]", fontsize=10)
        if row_idx == 0:
            ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)

        # =========================================================
        # Panel 3: Background Activity Correlation C_k(f)
        # =========================================================
        ax3 = axes[row_idx, 2]
        f_b = res["f_scan"]

        # Faint individual curves for remainder
        for r_curve in res["corr_rem_all"]:
            ax3.plot(f_b, r_curve, color="orchid", alpha=0.15, lw=0.8)

        # Observed full signal mean (dashed gray)
        ax3.plot(f_b, res["corr_full_mean"], color="dimgray", ls="--", lw=1.6, alpha=0.85, label="Observed Full Signal C_k(f)")

        # Reconstructed remainder mean (solid magenta) + bootstrap CI
        ax3.plot(f_b, res["corr_rem_mean"], color="darkmagenta", lw=2.0, label="Reconstructed Remainder C_k(f)")
        ax3.fill_between(f_b, res["corr_rem_ci"][0], res["corr_rem_ci"][1], color="orchid", alpha=0.25, label="95% Bootstrap CI")

        # Null envelope from circular shift (mean and 95%)
        ax3.plot(f_b, res["corr_null_mean"], "k--", lw=1.0, alpha=0.6, label="Shift Null Mean")
        ax3.fill_between(f_b, -res["corr_null_p95"], res["corr_null_p95"], color="lightgray", alpha=0.3, label="±95% Shift Noise Envelope")

        # Mask target peak band and adjacent overlap
        mask_low, mask_high = band[0], band[1]
        ax3.axvspan(mask_low, mask_high, color="black", alpha=0.15, hatch="//", label="Target Band (Masked)")

        # Resonant peak vertical lines
        for pf in PEAK_FREQS:
            ax3.axvline(pf, color="gray", ls=":", alpha=0.35, lw=0.9)

        ax3.set_xlim(50, 1500)
        ax3.set_ylim(-0.15, 0.40)
        ax3.axhline(0.0, color="k", ls="-", lw=0.8, alpha=0.4)
        ax3.set_ylabel(f"Corr(|z_{target['label']}|², P_f)\n[Envelope Coupling]", fontsize=10, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        if row_idx == 0:
            ax3.set_title("Panel 3: Background Activity Correlation\nC_k(f) = Corr(|z_k|², P_f(t))", fontsize=11, fontweight="bold")
        if row_idx == 3:
            ax3.set_xlabel("Background Center Frequency f [Hz]", fontsize=10)
        if row_idx == 0:
            ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Bicoherence Slices and Peak-Background Activity Coupling Diagnostic ({args.condition}, 16 Repetitions)\n"
        r"Scalar Channel: Central Surface Elevation $\eta_{\mathrm{center}}(t)$ [fs = 7200 Hz, N_{win} = 1024 (df = 7.03 Hz), Strip Avg: 3 bins]",
        fontsize=13, fontweight="bold"
    )

    fig_path = out_dir / "bicoherence_and_coupling_diagnostic.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved publication diagnostic figure to {fig_path}")

    # Copy to artifact directory
    artifact_path = Path("/home/jonas/.gemini/antigravity/brain/53e417c6-a5d6-4796-99de-dd55a8bf28ee/bicoherence_and_coupling_diagnostic.png")
    import shutil
    shutil.copy(fig_path, artifact_path)
    print(f"Copied figure to artifact directory: {artifact_path}")

    # Save numerical curves
    npz_data = {}
    json_summary = {"condition": args.condition, "parameters": {
        "fs_native": fs_native, "fs_down": 7200.0, "nperseg": 1024, "df": 7.03125,
        "overlap_fraction": 0.75, "w_strip_bins": 3, "w_strip_hz": 21.09375,
        "background_band_width_hz": 30.0, "n_repetitions": len(all_rep_data)
    }, "peaks": {}}

    for target in TARGET_PEAKS:
        p_idx = target["index"]
        res = results_summary[p_idx]
        prefix = f"mode_{p_idx}"
        npz_data[f"{prefix}_f1"] = res["f1_freqs"]
        npz_data[f"{prefix}_b2_in_mean"] = res["b2_in_mean"]
        npz_data[f"{prefix}_b2_in_null_p95"] = res["b2_in_null_p95"]
        npz_data[f"{prefix}_f2"] = res["f2_freqs"]
        npz_data[f"{prefix}_b2_out_mean"] = res["b2_out_mean"]
        npz_data[f"{prefix}_b2_out_null_p95"] = res["b2_out_null_p95"]
        npz_data[f"{prefix}_f_scan"] = res["f_scan"]
        npz_data[f"{prefix}_corr_full_mean"] = res["corr_full_mean"]
        npz_data[f"{prefix}_corr_rem_mean"] = res["corr_rem_mean"]
        npz_data[f"{prefix}_corr_null_p95"] = res["corr_null_p95"]

        # Peak summary stats
        json_summary["peaks"][f"{target['f_k']:.1f}Hz"] = {
            "max_b2_in": float(np.max(res["b2_in_mean"])),
            "mean_b2_in_null_p95": float(np.mean(res["b2_in_null_p95"])),
            "b2_in_exceeds_null": bool(np.any(res["b2_in_mean"] > res["b2_in_null_p95"])),
            "max_b2_out": float(np.max(res["b2_out_mean"])),
            "mean_b2_out_null_p95": float(np.mean(res["b2_out_null_p95"])),
            "b2_out_exceeds_null": bool(np.any(res["b2_out_mean"] > res["b2_out_null_p95"])),
            "max_corr_rem_unmasked": float(np.max(np.where(
                (res["f_scan"] < target["band"][0]) | (res["f_scan"] > target["band"][1]),
                res["corr_rem_mean"], -1.0
            ))),
            "mean_corr_null_p95": float(np.mean(res["corr_null_p95"]))
        }

    npz_path = out_dir / "bicoherence_and_coupling_diagnostic.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved numerical curves to {npz_path}")

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved summary JSON to {json_path}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY TABLE:")
    print("=" * 70)
    print(f"Peak Frequency | Max b²_in (Triad f_k+f) | Max b²_out (Subharm) | Max Unmasked Rem Corr C_k(f) | Null Baseline")
    print("-" * 70)
    for target in TARGET_PEAKS:
        pf = f"{target['f_k']:.1f}Hz"
        info = json_summary["peaks"][pf]
        print(f"{pf:14s} | {info['max_b2_in']:12.4f} (Null: {info['mean_b2_in_null_p95']:.3f}) | {info['max_b2_out']:12.4f} (Null: {info['mean_b2_out_null_p95']:.3f}) | {info['max_corr_rem_unmasked']:18.4f} | Null: {info['mean_corr_null_p95']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
