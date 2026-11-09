"""Universal end-to-end training and evaluation pipeline for clean reduced data.

Executes for a specified forcing condition (e.g. 0p10, 0p15, 0p20):
1. Fit 11-band local-excess complex linear oscillator bank with automatic power normalization.
2. Prepare remainder data (offline modal decomposition).
3. Train compact TCN-15 Neural SDE with strictly dissipative SPSD operator & velocity Energy Score.
4. Fine-tune diagonal diffusion on multi-step displacement variance with 100% frozen drift.
5. Autonomous 50,000-step rollout evaluation (s_vec = 1.0, bank_scales = 1.0).
6. Generate standard multi-panel diagnostic suite (PSD, peaks, energy, modal variances, trajectories).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale
from modelling.data_preparation.frequency_components import settling_samples
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.frequency_hybrid.evaluate import describe
from modelling.frequency_hybrid.local_excess import (
    complex_covariance, fit_filtered_band, local_excess_modes,
    projected_band, simulate_complex_band,
)
from modelling.frequency_hybrid.local_excess_remainder import decompose, load_bank, prepare_data
from modelling.frequency_hybrid.study import center_weights, matrices, projected, spectral_comparison, write_json
from modelling.neural_sde.train import load_model

PYTHON_BIN = "/home/jonas/miniforge3/envs/diff-model-mit/bin/python"
DATA_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data/clean_red_data")

# 11 consistent resonant peak centers across regimes
CENTERS_GRID = [
    73.83, 126.56, 200.39, 274.22, 362.11,
    450.00, 537.89, 636.33, 741.80, 857.81, 980.86
]

# Benchmark capillary energy [10th, 50th, 90th] percentiles in pJ from ground truth full dataset
BENCHMARK_EXACT_PJ = {
    "0p07": [5.95, 7.23, 24.10],
    "0p08": [8.65, 15.18, 38.01],
    "0p10": [42.34, 53.59, 65.56],
    "0p15": [186.47, 213.36, 232.83],
    "0p20": [337.47, 381.23, 390.27],
    "0p25": [410.87, 448.71, 500.85],
}


def make_bands(centers: list[float], low_edge: float = 50.0) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    middle = (c[:-1] + c[1:]) / 2.0
    upper = c[-1] + (c[-1] - middle[-1])
    edges = np.r_[low_edge, middle, upper]
    bands = np.column_stack((edges[:-1], edges[1:]))
    return bands


def step1_fit_oscillators(exp: str, output_dir: Path, rank: int = 10) -> Path:
    print(f"\n==========================================")
    print(f"[{exp}] STEP 1: Fitting 11-Band Oscillator Bank")
    print(f"==========================================")
    osc_dir = output_dir / "oscillators"
    if (osc_dir / "summary.json").exists():
        print(f"Oscillators already fit: {osc_dir}")
        return osc_dir
    if osc_dir.exists():
        shutil.rmtree(osc_dir)
    osc_dir.mkdir(parents=True, exist_ok=True)

    shared_basis = DATA_ROOT / exp / "shared_pod/shared_pod_r200.h5"
    data = SharedPODTrainingData(
        exp, rank, data_root=str(DATA_ROOT), shared_basis=str(shared_basis),
        validation_reps=[13, 14], test_reps=[15, 16], output_dtype="float64"
    )
    fs = 1.0 / data.native_dt
    centers = np.asarray(CENTERS_GRID, dtype=float)
    bands = make_bands(centers, low_edge=50.0)
    orders = np.r_[2, np.full(len(centers) - 1, 4, dtype=int)]

    filters = [
        butter(int(order), band, btype="bandpass", fs=fs, output="sos")
        for order, band in zip(orders, bands)
    ]
    edge = max(int(np.ceil(0.1 * fs)), *(settling_samples(s, fs) for s in filters))

    names = data.train_repetitions + data.validation_repetitions
    raw = {name: data.read_coordinates(name, 0, data.frame_counts[name]) for name in names}
    train_records = [raw[name][1] for name in data.train_repetitions]
    frequency, train_spectrum = matrices(train_records, fs, 32768)
    weights = center_weights(data)

    band_rows = []
    for idx, (center, band, order, sos) in enumerate(zip(centers, bands, orders, filters)):
        excess = local_excess_modes(frequency, train_spectrum, weights, center, band, energy_fraction=0.95)
        if excess["rank"] > 1:
            excess["rank"] = 1
            excess["modes"] = excess["modes"][:, :1]

        train_components = []
        train_coefficients = []
        for name in data.train_repetitions:
            comp, coeffs = projected_band(raw[name][1], sos, excess["modes"], edge)
            train_components.append(comp)
            train_coefficients.append(coeffs)

        band_freq, band_spec = matrices(train_components, fs, 32768)
        temporal = fit_filtered_band(train_components, band_freq, band_spec, fs, band, sos)
        coeff_cov, coeff_factor = complex_covariance(train_coefficients)

        # Baseline filter-passed variance compensation
        passed_frac = max(float(temporal["filter_passed_variance_fraction"]), 1e-4)
        coeff_factor = coeff_factor / np.sqrt(passed_frac)

        # Empirical simulation variance check & closed-form power calibration
        temp_model = dict(
            modes=excess["modes"],
            coefficient_covariance=coeff_cov,
            coefficient_factor=coeff_factor,
            frequency_hz=temporal["frequency_hz"],
            gamma_per_second=temporal["gamma_per_second"],
            edges_hz=band,
            filter_order=order,
            fs=fs,
        )
        sim_raw = simulate_complex_band(temp_model, 40000, paths=2, seed=100 + idx)
        sim_filt = sosfiltfilt(sos, sim_raw, axis=1, padlen=edge)[:, edge:-edge, :]
        sim_var = float(np.mean(np.var(sim_filt, axis=1)))
        meas_var = float(np.mean([np.var(c) for c in train_components]))

        power_correction = np.sqrt(max(meas_var, 1e-12) / max(sim_var, 1e-12))
        coeff_factor = coeff_factor * power_correction
        coeff_cov = coeff_cov * (power_correction ** 2)

        model_save = dict(
            modes=excess["modes"],
            coefficient_covariance=coeff_cov,
            coefficient_factor=coeff_factor,
            frequency_hz=temporal["frequency_hz"],
            gamma_per_second=temporal["gamma_per_second"],
            edges_hz=band,
            filter_order=order,
            fs=fs,
            nominal_center_hz=center,
            filter_passed_variance_fraction=passed_frac,
            power_correction=float(power_correction),
        )
        np.savez_compressed(osc_dir / f"band_{idx:02d}.npz", **model_save)
        band_rows.append(dict(
            index=idx, nominal_center_hz=float(center),
            fitted_frequency_hz=float(temporal["frequency_hz"]),
            gamma_per_second=float(temporal["gamma_per_second"]),
            power_correction=float(power_correction)
        ))
        print(f"  Band {idx:02d} ({center:.1f} Hz): fit={temporal['frequency_hz']:.1f} Hz, "
              f"gamma={temporal['gamma_per_second']:.1f}/s, power_corr={power_correction:.2f}")

    write_json(osc_dir / "summary.json", {"experiment": exp, "bands": band_rows})
    print(f"Saved oscillator bank to {osc_dir}")
    return osc_dir


def step2_prepare_remainder(exp: str, output_dir: Path, osc_dir: Path, rank: int = 10) -> Path:
    print(f"\n==========================================")
    print(f"[{exp}] STEP 2: Preparing Remainder Data")
    print(f"==========================================")
    rem_output = output_dir / "remainder_data"
    manifest_path = rem_output / "data/manifest.json"
    if manifest_path.exists():
        print(f"Manifest already exists: {manifest_path}")
        return manifest_path

    shared_basis = DATA_ROOT / exp / "shared_pod/shared_pod_r200.h5"
    manifest = prepare_data(
        rem_output, osc_dir, experiment=exp, rank=rank,
        data_root=DATA_ROOT, shared_basis=shared_basis, include_test=False
    )
    print(f"Prepared remainder manifest: {manifest}")
    return Path(manifest)


def step3_train_neural_sde(exp: str, output_dir: Path, manifest_path: Path, rank: int = 10, epochs: int = 5) -> Path:
    print(f"\n==========================================")
    print(f"[{exp}] STEP 3: Training Compact TCN-15 Neural SDE ({epochs} epochs)")
    print(f"==========================================")
    ckpt_dir = output_dir / "checkpoint"
    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists():
        print(f"Checkpoint already exists: {best_pt}")
        return best_pt

    shared_basis = DATA_ROOT / exp / "shared_pod/shared_pod_r200.h5"
    cmd = [
        PYTHON_BIN, "-m", "modelling.neural_sde", "train",
        "--experiment", exp,
        "--rank", str(rank),
        "--data-root", str(DATA_ROOT),
        "--shared-basis", str(shared_basis),
        "--temporal-component-manifest", str(manifest_path),
        "--temporal-component", "remainder",
        "--validation-reps", "13", "14",
        "--test-reps", "15", "16",
        "--state-variable", "velocity",
        "--lag", "1",
        "--history", "15",
        "--drift-type", "damped_residual",
        "--damped-operator", "spsd",
        "--residual-architecture", "tcn",
        "--tcn-context-frames", "15",
        "--tcn-channels", "32",
        "--tcn-kernel-size", "3",
        "--tcn-dilations", "1", "2", "4",
        "--diffusion-type", "diagonal",
        "--train-mode", "joint",
        "--train-trim-percent", "0.0",
        "--diffusion-trim-percent", "0.0",
        "--multistep-objective", "energy_score",
        "--multistep-target", "velocity",
        "--multistep-weight", "0.03",
        "--multistep-horizons", "2", "5", "10", "20",
        "--multistep-particle-counts", "4", "4", "4", "4",
        "--multistep-window-fraction", "0.005",
        "--multistep-window-sampling", "batches",
        "--multistep-gradient-groups", "drift", "diffusion",
        "--tcn-rollout-cache",
        "--learning-rate", "0.0002",
        "--batch-size", "256",
        "--epochs", str(epochs),
        "--device", "cuda",
        "--output", str(ckpt_dir)
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"Neural SDE training finished in {time.time() - t0:.1f} s")
    return best_pt


def step4_finetune_diffusion(ckpt_path: Path, data_dir: Path, iterations: int = 100) -> Path:
    print(f"\n==========================================")
    print(f"STEP 4: Fine-Tuning Diffusion Variance (Frozen Drift, {iterations} steps)")
    print(f"==========================================")
    output_ckpt = ckpt_path.parent / "best_intrinsic.pt"
    if output_ckpt.exists():
        print(f"Intrinsic checkpoint already exists: {output_ckpt}")
        return output_ckpt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(str(ckpt_path), device=device)
    model.eval()

    # Freeze drift 100%
    for p in model.drift_model.parameters():
        p.requires_grad = False
    model.diffusion_model.raw_diagonal.requires_grad = True
    trainable_params = [model.diffusion_model.raw_diagonal]

    # Load remainder data
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    train_seqs = []
    for r in range(1, 13):
        matching_key = None
        for name in manifest["records"].keys():
            if re.search(r"(?:_rep|^rep)" + str(r) + r"$", name):
                matching_key = name
                break
        if matching_key:
            fpath = data_dir / manifest["records"][matching_key]["files"]["remainder"]["path"]
            train_seqs.append(torch.tensor(np.load(fpath), dtype=torch.float32))
        else:
            matches = sorted(list(data_dir.glob(f"*rep{r}_remainder.npy")))
            if matches:
                train_seqs.append(torch.tensor(np.load(matches[0]), dtype=torch.float32))

    n_coords = train_seqs[0].shape[-1]
    horizons = [20, 50, 100, 200]
    target_disp_vars = {}
    for H in horizons:
        diffs = torch.cat([seq[H:] - seq[:-H] for seq in train_seqs], dim=0)
        target_disp_vars[H] = diffs.var(dim=0).to(device)

    optimizer = optim.Adam(trainable_params, lr=0.01)
    h_steps = model.config.history_steps
    step_size = model.config.lag_steps * (1. if model.config.state_variable == "velocity" else model.config.dt)
    max_h = max(horizons)
    h_set = set(horizons)

    best_loss = float("inf")
    best_weights = None

    for it in range(1, iterations + 1):
        batch_z0, batch_hist = [], []
        for _ in range(64):
            seq = train_seqs[np.random.randint(len(train_seqs))]
            t_start = np.random.randint(h_steps, len(seq) - max_h - 1)
            batch_z0.append(seq[t_start])
            batch_hist.append(seq[t_start - h_steps:t_start].flip(0))

        z0 = torch.stack(batch_z0).unsqueeze(1).to(device)
        hist = torch.stack(batch_hist).unsqueeze(1).to(device)
        velocity = model._frame_velocity(z0, hist)

        state, history = z0, hist
        tcn_cache = model.new_tcn_cache()
        horizon_losses = []

        for step in range(1, max_h + 1):
            base_mean, noise_factor = model.velocity_base_step_distribution(state, history, velocity, tcn_cache=tcn_cache)
            noise = torch.randn(state.shape, device=device, dtype=state.dtype)
            velocity = base_mean + torch.matmul(noise, noise_factor.T)
            state = state + step_size * velocity
            history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)

            if step in h_set:
                gen_disp_var = (state - z0).squeeze(1).var(dim=0)
                target_var_h = target_disp_vars[step]
                log_gen = torch.log(gen_disp_var + 1e-8)
                log_target = torch.log(target_var_h + 1e-8)
                coord_weights = torch.ones(n_coords, device=device)
                if n_coords >= 5:
                    coord_weights[4] = 2.0  # mode 5 weight
                h_loss = (coord_weights * ((log_gen - log_target) ** 2)).mean()
                horizon_losses.append(h_loss)

        loss = torch.stack(horizon_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if it % 20 == 0 or it == iterations:
            print(f"  Diffusion step {it:03d}/{iterations}: loss = {loss.item():.4f}")

    model.load_state_dict(best_weights)
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_data["model_state"] = model.state_dict()
    if "training" not in ckpt_data:
        ckpt_data["training"] = {}
    ckpt_data["training"]["diffusion_variance_finetuned"] = True
    ckpt_data["training"]["diffusion_tuning_loss"] = best_loss
    torch.save(ckpt_data, output_ckpt)
    print(f"Saved fine-tuned intrinsic checkpoint: {output_ckpt}")
    return output_ckpt


def step5_evaluate_rollout(exp: str, output_dir: Path, ckpt_path: Path, osc_dir: Path, manifest_path: Path, horizon: int = 50000) -> dict:
    print(f"\n==========================================")
    print(f"[{exp}] STEP 5: Autonomous Rollout Evaluation ({horizon} steps)")
    print(f"==========================================")
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(str(ckpt_path), device=device)
    model.eval()

    with open(manifest_path) as f:
        manifest = json.load(f)
    edge = int(manifest["edge_samples"])
    fs = float(manifest["fs"])
    scored_len = horizon - 2 * edge

    data_dir = manifest_path.parent
    # Validation reps 13 and 14
    val_keys = []
    for r in [13, 14]:
        for name in manifest["records"].keys():
            if re.search(r"(?:_rep|^rep)" + str(r) + r"$", name):
                val_keys.append(name)
                break
    if len(val_keys) == 2:
        rem_files = [data_dir / manifest["records"][k]["files"]["remainder"]["path"] for k in val_keys]
        full_files = [data_dir / manifest["records"][k]["files"]["full"]["path"] for k in val_keys]
    else:
        rem_files = sorted(list(data_dir.glob("*rep13_remainder.npy")) + list(data_dir.glob("*rep14_remainder.npy")))
        full_files = sorted(list(data_dir.glob("*rep13_full.npy")) + list(data_dir.glob("*rep14_full.npy")))
    rem_data = [np.load(f) for f in rem_files]
    full_data = [np.load(f) for f in full_files]
    ref_full = np.stack(full_data)

    h_steps = model.config.history_steps
    inits = torch.tensor(np.stack([r[h_steps] for r in rem_data]), dtype=torch.float32, device=device)
    hists = torch.tensor(np.stack([r[h_steps - 1::-1].copy() for r in rem_data]), dtype=torch.float32, device=device)

    # 1. Rollout Neural SDE Remainder (Strictly unscaled: s_vec = 1.0)
    print(f"Generating autonomous Neural SDE rollout ({horizon} steps, s_vec=1.0)...")
    generator = torch.Generator(device=device).manual_seed(41)
    state = inits[:, None, :].expand(-1, 1, -1).clone()
    history = hists[:, None, :h_steps].expand(-1, 1, -1, -1).clone()
    velocity = model._frame_velocity(state, history)
    step_size = model.config.lag_steps * (1. if model.config.state_variable == "velocity" else model.config.dt)
    tcn_cache = model.new_tcn_cache()

    states = [state]
    with torch.no_grad():
        for _ in range(horizon):
            noise = torch.randn(state.shape, device=device, dtype=state.dtype, generator=generator)
            base_mean, noise_factor = model.velocity_base_step_distribution(state, history, velocity, tcn_cache=tcn_cache)
            velocity = base_mean + torch.matmul(noise, noise_factor.T)
            state = state + step_size * velocity
            history = torch.cat((state.unsqueeze(-2), history[..., :-1, :]), dim=-2)
            states.append(state)
    gen_rem = torch.stack(states, dim=2).cpu().numpy()[:, :, edge:edge + scored_len, :]

    # 2. Simulate Linear Oscillator Bank (Strictly unscaled: bank_scales = 1.0)
    print(f"Simulating 11-band oscillator bank ({scored_len} steps, bank_scales=1.0)...")
    _, models_bank = load_bank(osc_dir)
    n_coords = inits.shape[-1]
    bank_draw = np.zeros((2, 1, scored_len, n_coords), dtype=np.float64)

    overlap = 0.0578
    for idx, m in enumerate(models_bank):
        low, high = np.asarray(m["edges_hz"], dtype=float)
        width = high - low
        sos = butter(int(m.get("filter_order", 4)), [max(10.0, low - overlap * width), high + overlap * width],
                     btype="bandpass", fs=fs, output="sos")
        raw = simulate_complex_band(m, scored_len + 2 * edge, paths=2, seed=1041 + idx)
        filt = sosfiltfilt(sos, raw, axis=1, padlen=edge)[:, edge:-edge, :]
        bank_draw += filt[:, None, :, :]

    hybrid = gen_rem + bank_draw
    ref_crop = ref_full[:, edge:edge + scored_len]

    # 3. Highpass > 50 Hz for physical wave energy & variances
    sos50 = butter(4, 50.0, btype="highpass", fs=fs, output="sos")
    ref_hp = sosfiltfilt(sos50, ref_crop, axis=-2)
    hyb_hp = sosfiltfilt(sos50, hybrid[:, 0], axis=-2)

    # Load shared basis modes & stiffness
    shared_basis = DATA_ROOT / exp / "shared_pod/shared_pod_r200.h5"
    with h5py.File(shared_basis, "r") as h5:
        pod_modes = h5["pod"]["modes"][:n_coords]
        x_m = h5["grid"]["x"][:] * length_scale("mm")
        y_m = h5["grid"]["y"][:] * length_scale("mm")
    cy, cx = pod_modes.shape[1] // 2, pod_modes.shape[2] // 2
    center_weights = pod_modes[:, cy, cx]
    stiffness, _ = capillary_stiffness(pod_modes, x_m, y_m, gamma=0.0728)

    # Save rollout.npz and metrics.json for official data_analysis.plot_rollout
    val_names = val_keys if len(val_keys) == 2 else ["rep13", "rep14"]
    np.savez_compressed(
        eval_dir / "rollout.npz",
        trajectories=hybrid,
        reference=ref_crop,
        reference_time=np.stack([np.arange(scored_len) / fs, np.arange(scored_len) / fs]),
        repetition=np.array(val_names),
        start_index=np.array([edge, edge]),
    )
    metrics_data = {
        "rank": n_coords,
        "include_spatial_mean": False,
        "spatial_mean_representation": "embedded_in_pod",
        "source_shared_basis": str(shared_basis),
        "time_units": "s",
        "signal_units": "microns",
        "fs": fs,
        "physical_lag": 1.0 / fs,
        "lag_steps": 1,
        "spatial_mean_highpass_hz": None,
        "checkpoint": str(ckpt_path),
    }
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Run official repository plot_rollout
    cmd = [PYTHON_BIN, "-m", "data_analysis.plot_rollout", str(eval_dir), "--nperseg", "8192", "--modal-psd", "--overwrite"]
    print(f"Running official repository plot_rollout: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    opts = dict(exact=False, batch_size=256, stiffness=stiffness, include_spatial_mean=False)
    x_dummy, y_dummy = np.zeros(200), np.zeros(200)

    E_ref_raw = float(np.mean(compute_energies(ref_crop, pod_modes, x_dummy, y_dummy, length_scale("microns"), 0.0728, **opts)["quadratic"]))
    E_gen_raw = float(np.mean(compute_energies(hybrid[:, 0], pod_modes, x_dummy, y_dummy, length_scale("microns"), 0.0728, **opts)["quadratic"]))
    E_ref_hp = float(np.mean(compute_energies(ref_hp, pod_modes, x_dummy, y_dummy, length_scale("microns"), 0.0728, **opts)["quadratic"]))
    E_gen_hp = float(np.mean(compute_energies(hyb_hp, pod_modes, x_dummy, y_dummy, length_scale("microns"), 0.0728, **opts)["quadratic"]))
    energy_hp_ratio = E_gen_hp / E_ref_hp

    var_ref = np.var(ref_hp.reshape(-1, n_coords), axis=0)
    var_gen = np.var(hyb_hp.reshape(-1, n_coords), axis=0)
    var_ratios = var_gen / var_ref

    # Center probe PSD
    ref_center = np.einsum("cri,i->cr", ref_crop, center_weights)
    gen_center = np.einsum("cri,i->cr", hybrid[:, 0], center_weights)
    f_w = signal.welch(ref_center[0], fs=fs, nperseg=8192, noverlap=4096)[0]
    p_ref_c = np.mean([signal.welch(ref_center[c], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_gen_c = np.mean([signal.welch(gen_center[c], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)

    # POD mode sum PSD
    p_ref_pod = sum(np.mean([signal.welch(ref_crop[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0) for m in range(n_coords))
    p_gen_pod = sum(np.mean([signal.welch(hybrid[c, 0, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0) for m in range(n_coords))

    peak_ratios = {}
    peak_errors_db = []
    for pf in CENTERS_GRID:
        k = np.argmin(np.abs(f_w - pf))
        r = float(p_gen_c[k] / p_ref_c[k])
        peak_ratios[f"{pf:.1f}"] = r
        peak_errors_db.append(abs(10.0 * np.log10(max(r, 1e-6))))
    mean_peak_mae_db = float(np.mean(peak_errors_db))

    benchmark = BENCHMARK_EXACT_PJ.get(exp, [np.nan, np.nan, np.nan])
    gen_exact_pJ = E_gen_hp * 1e12

    summary = dict(
        experiment=exp,
        raw_energy_ref_J=E_ref_raw,
        raw_energy_gen_J=E_gen_raw,
        raw_energy_ratio=E_gen_raw / E_ref_raw,
        physical_energy_hp50_ref_J=E_ref_hp,
        physical_energy_hp50_gen_J=E_gen_hp,
        physical_energy_hp50_ratio=energy_hp_ratio,
        gen_exact_energy_pJ=gen_exact_pJ,
        benchmark_target_q10_q50_q90_pJ=benchmark,
        within_10_90_percentile=bool(benchmark[0] <= gen_exact_pJ <= benchmark[2]),
        modal_variance_ratios=var_ratios.tolist(),
        mean_modal_variance_ratio=float(np.mean(var_ratios)),
        peak_ratios=peak_ratios,
        mean_peak_contrast_error_db=mean_peak_mae_db
    )
    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 1. Full PSD Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    m_plot = (f_w >= 50) & (f_w <= 20000)
    axes[0].loglog(f_w[m_plot], p_ref_c[m_plot], color="black", label=f"{exp} Clean Ground Truth Ref", lw=1.5)
    axes[0].loglog(f_w[m_plot], p_gen_c[m_plot], color="tab:blue", label=f"{exp} Clean Unified Hybrid", lw=1.2, alpha=0.9)
    axes[0].set(ylabel="Center Probe PSD [µm²/Hz]", title=f"{exp} Clean Unified Hybrid: Center Spectrum")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].loglog(f_w[m_plot], p_ref_pod[m_plot], color="black", label="Reference (Sum POD 1-10)", lw=1.5)
    axes[1].loglog(f_w[m_plot], p_gen_pod[m_plot], color="tab:blue", label="Unified Hybrid (Sum POD 1-10)", lw=1.2, alpha=0.9)
    axes[1].set(xlabel="Frequency [Hz]", ylabel="POD Sum PSD [µm²/Hz]", title=f"{exp} Clean Unified Hybrid: POD Spectrum")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.savefig(eval_dir / "psd.png", dpi=200)
    plt.close(fig)

    # 2. Peaks Zoom
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    m_peak = (f_w >= 50) & (f_w <= 1200)
    ax.semilogy(f_w[m_peak], p_ref_c[m_peak], color="black", label=f"{exp} Reference (Val Center)", lw=1.5)
    ax.semilogy(f_w[m_peak], p_gen_c[m_peak], color="tab:blue", label=f"{exp} Hybrid Center", lw=1.2)
    for pf in CENTERS_GRID:
        ax.axvline(pf, color="gray", linestyle=":", alpha=0.5)
    ax.set(xlabel="Frequency [Hz]", ylabel="PSD [µm²/Hz]", title=f"{exp} Clean Resonant Carrier Peaks (50 - 1200 Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(eval_dir / "psd_peaks.png", dpi=200)
    plt.close(fig)

    # 3. Energy Comparison Chart
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    q10, med, q90 = benchmark
    ax.axhspan(q10, q90, color="green", alpha=0.18, label=f"Benchmark 10th-90th [{q10:.1f}, {q90:.1f}] pJ")
    ax.axhline(med, color="green", linestyle="--", lw=1.5, label=f"Benchmark Median ({med:.1f} pJ)")
    bars = ax.bar(["Ref Val HP50", "Generated Hybrid HP50"], [E_ref_hp * 1e12, gen_exact_pJ],
                  color=["black", "tab:blue"], alpha=0.85, width=0.45)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.01 * med, f"{h:.1f} pJ", ha="center", va="bottom", fontweight="bold")
    ax.set(ylabel="Capillary Wave Energy [pJ]", title=f"{exp} Physical Energy Scorecard vs Benchmark")
    ax.legend()
    fig.savefig(eval_dir / "energy.png", dpi=200)
    plt.close(fig)

    # 4. Modal Variances
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    coords = [f"POD {i+1}" for i in range(n_coords)]
    bars = ax.bar(coords, var_ratios, color="tab:blue", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="Target 1.0x")
    ax.axhspan(0.90, 1.10, color="green", alpha=0.15, label="Target Band [0.90, 1.10]")
    ax.axhspan(0.80, 1.20, color="gray", alpha=0.15, label="Acceptable Band [0.80, 1.20]")
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="Variance Ratio (Gen / Ref)", title=f"{exp} Clean Modal Variances (>50 Hz)")
    ax.legend()
    fig.savefig(eval_dir / "modal_variances.png", dpi=200)
    plt.close(fig)

    # 5. Trajectory Snippets
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout="constrained")
    dt = 1.0 / fs
    t_snip = np.arange(2000) * dt * 1000
    for idx, mode_idx in enumerate([0, 1, 2]):
        axes[idx].plot(t_snip, ref_full[0, :2000, mode_idx], color="black", label=f"Ref POD {mode_idx+1}", lw=1.2)
        axes[idx].plot(t_snip, hybrid[0, 0, :2000, mode_idx], color="tab:blue", label=f"Gen POD {mode_idx+1}", lw=1.0, alpha=0.8)
        axes[idx].set(ylabel=f"POD {mode_idx+1} [µm]")
        axes[idx].grid(alpha=0.3)
        axes[idx].legend()
    axes[2].set(xlabel="Time [ms]")
    fig.savefig(eval_dir / "trajectories.png", dpi=200)
    plt.close(fig)

    print(f"\n[{exp}] Rollout Evaluation Results:")
    print(f"  Physical Energy (>50 Hz): {gen_exact_pJ:.2f} pJ (Target: [{q10:.1f}, {q90:.1f}] pJ, Ratio vs Val: {energy_hp_ratio:.3f}x)")
    print(f"  Within [10th, 90th] Percentile: {benchmark[0] <= gen_exact_pJ <= benchmark[2]}")
    print(f"  Mean Modal Variance Ratio: {np.mean(var_ratios):.3f}x (M1: {var_ratios[0]:.2f}, M2: {var_ratios[1]:.2f}, M3: {var_ratios[2]:.2f})")
    print(f"  Mean Peak Contrast Error: {mean_peak_mae_db:.2f} dB")
    return summary


def run_pipeline(exp: str, output_base: Path, epochs: int = 5, diffusion_iters: int = 100) -> dict:
    t_start = time.time()
    out_dir = output_base / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    osc_dir = step1_fit_oscillators(exp, out_dir)
    manifest = step2_prepare_remainder(exp, out_dir, osc_dir)
    raw_ckpt = step3_train_neural_sde(exp, out_dir, manifest, epochs=epochs)
    intrinsic_ckpt = step4_finetune_diffusion(raw_ckpt, manifest.parent, iterations=diffusion_iters)
    results = step5_evaluate_rollout(exp, out_dir, intrinsic_ckpt, osc_dir, manifest)

    total_time = time.time() - t_start
    print(f"\n>>> [{exp}] Pipeline completed in {total_time/60.0:.2f} min <<<\n")
    results["elapsed_minutes"] = total_time / 60.0
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=["0p10", "0p15", "0p20"])
    parser.add_argument("--output-base", type=Path, default=Path("runs/clean_production"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--diffusion-iters", type=int, default=100)
    args = parser.parse_args()

    overall_results = {}
    for exp in args.experiments:
        res = run_pipeline(exp, args.output_base, epochs=args.epochs, diffusion_iters=args.diffusion_iters)
        overall_results[exp] = res

    summary_file = args.output_base / "all_conditions_summary.json"
    with open(summary_file, "w") as f:
        json.dump(overall_results, f, indent=2)
    print(f"\n==========================================")
    print(f"All conditions completed! Summary written to: {summary_file}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
