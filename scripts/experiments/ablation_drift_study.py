"""Ablation study on drift parameterization for the clean reduced dataset:

Compares on identical data (0p10 and 0p20):
1. 'spsd': Full SPSD matrix restoring + damping (L_K L_K^T, L_D L_D^T) + TCN residual
2. 'diagonal': Diagonal softplus restoring + damping (diag(k_i), diag(d_i)) + TCN residual
3. 'unrestricted_tcn': Pure neural drift with NO linear restoring or damping (drift_type=mlp, residual_architecture=tcn)

Evaluates:
- Training stability & convergence
- Long autonomous rollout stability (50,000 steps)
- Physical deformation energy (>50 Hz) vs benchmark [10th, 90th]
- Modal variance ratios (Modes 1 to 10)
- Resonant peak contrast MAE [dB]
- Mean relative coordinate bias
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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
from modelling.frequency_hybrid.local_excess import simulate_complex_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank
from modelling.neural_sde.train import load_model

PYTHON_BIN = "/home/jonas/miniforge3/envs/diff-model-mit/bin/python"
DATA_ROOT = Path("/home/jonas/ucsd_thesis/reduced_data/clean_red_data")

CENTERS_GRID = [
    73.83, 126.56, 200.39, 274.22, 362.11,
    450.00, 537.89, 636.33, 741.80, 857.81, 980.86
]

BENCHMARK_EXACT_PJ = {
    "0p07": [5.95, 7.23, 24.10],
    "0p08": [8.65, 15.18, 38.01],
    "0p10": [42.34, 53.59, 65.56],
    "0p15": [186.47, 213.36, 232.83],
    "0p20": [337.47, 381.23, 390.27],
    "0p25": [410.87, 448.71, 500.85],
}


def train_variant(exp: str, variant_name: str, out_dir: Path, manifest_path: Path, epochs: int = 5) -> Path:
    print(f"\n==========================================")
    print(f"[{exp} | {variant_name}] Training Neural SDE ({epochs} epochs)")
    print(f"==========================================")
    ckpt_dir = out_dir / "checkpoint"
    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists():
        print(f"Checkpoint already exists: {best_pt}")
        return best_pt

    shared_basis = DATA_ROOT / exp / "shared_pod/shared_pod_r200.h5"
    cmd = [
        PYTHON_BIN, "-m", "modelling.neural_sde", "train",
        "--experiment", exp,
        "--rank", "10",
        "--data-root", str(DATA_ROOT),
        "--shared-basis", str(shared_basis),
        "--temporal-component-manifest", str(manifest_path),
        "--temporal-component", "remainder",
        "--validation-reps", "13", "14",
        "--test-reps", "15", "16",
        "--state-variable", "velocity",
        "--lag", "1",
        "--history", "15",
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

    if variant_name == "spsd":
        cmd.extend(["--drift-type", "damped_residual", "--damped-operator", "spsd"])
    elif variant_name == "diagonal":
        cmd.extend(["--drift-type", "damped_residual", "--damped-operator", "diagonal"])
    elif variant_name == "unrestricted_tcn":
        cmd.extend(["--drift-type", "mlp"])
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"Neural SDE training finished in {time.time() - t0:.1f} s")
    return best_pt


def finetune_diffusion(ckpt_path: Path, manifest_path: Path, iterations: int = 100) -> Path:
    print(f"\n==========================================")
    print(f"Fine-Tuning Diffusion Variance (Frozen Drift, {iterations} steps)")
    print(f"==========================================")
    output_ckpt = ckpt_path.parent / "best_intrinsic.pt"
    if output_ckpt.exists():
        print(f"Intrinsic checkpoint already exists: {output_ckpt}")
        return output_ckpt

    data_dir = manifest_path.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(str(ckpt_path), device=device)
    model.eval()

    # Freeze drift 100%
    for p in model.drift_model.parameters():
        p.requires_grad = False
    model.diffusion_model.raw_diagonal.requires_grad = True
    trainable_params = [model.diffusion_model.raw_diagonal]

    # Load remainder data
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    train_seqs = []
    import re
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

        particles = 4
        s_sim = z0.expand(-1, particles, -1).clone()
        h_sim = hist.expand(-1, particles, -1, -1).clone()
        v_sim = velocity.expand(-1, particles, -1).clone()
        tcn_cache = model.new_tcn_cache()

        displacements = {H: None for H in horizons}
        for h_step in range(1, max_h + 1):
            noise = torch.randn(s_sim.shape, device=device, dtype=s_sim.dtype)
            base_mean, noise_factor = model.velocity_base_step_distribution(s_sim, h_sim, v_sim, tcn_cache=tcn_cache)
            v_sim = base_mean + torch.matmul(noise, noise_factor.T)
            s_sim = s_sim + step_size * v_sim
            h_sim = torch.cat((s_sim.unsqueeze(-2), h_sim[..., :-1, :]), dim=-2)
            if h_step in h_set:
                displacements[h_step] = s_sim - z0.expand(-1, particles, -1)

        loss = 0.0
        for H in horizons:
            disp = displacements[H].reshape(-1, n_coords)
            sim_var = disp.var(dim=0)
            tgt_var = target_disp_vars[H]
            rel_err = (sim_var - tgt_var) / (tgt_var + 1e-6)
            loss = loss + torch.mean(rel_err**2)

        optimizer.zero_grad()
        loss.backward()
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


def evaluate_variant(exp: str, variant_name: str, out_dir: Path, ckpt_path: Path, osc_dir: Path, manifest_path: Path, horizon: int = 50000) -> dict:
    print(f"\n==========================================")
    print(f"[{exp} | {variant_name}] Autonomous Rollout Evaluation ({horizon} steps)")
    print(f"==========================================")
    eval_dir = out_dir / "evaluation"
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
    import re
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

    # 1. Rollout Neural SDE Remainder
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

    # 2. Simulate Linear Oscillator Bank
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

    # Check finite
    if not np.isfinite(hybrid).all():
        print(f"WARNING: Hybrid trajectory contains non-finite values (diverged/exploded)!")
        return {"exploded": True, "variant": variant_name}

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

    peak_errors_db = []
    for pf in CENTERS_GRID:
        k = np.argmin(np.abs(f_w - pf))
        r = float(p_gen_c[k] / p_ref_c[k])
        peak_errors_db.append(abs(10.0 * np.log10(max(r, 1e-6))))
    mean_peak_mae_db = float(np.mean(peak_errors_db))

    benchmark = BENCHMARK_EXACT_PJ.get(exp, [np.nan, np.nan, np.nan])
    gen_exact_pJ = E_gen_hp * 1e12

    # Load quantitative metrics
    with open(eval_dir / "quantitative/summary.json") as f:
        q_summary = json.load(f)
    cov_tr = q_summary["coefficient_statistics"]["untrimmed"]["covariance_trace_ratio"]
    bias = q_summary["coefficient_statistics"]["untrimmed"]["mean_bias_relative_to_reference_rms"]

    with open(eval_dir / "psd_peaks/summary.json") as f:
        p_summary = json.load(f)["metrics"]
    official_peak_mae_db = p_summary.get("peak_contrast_error_db", mean_peak_mae_db)
    official_n_peaks = p_summary.get("reference_peak_count", len(CENTERS_GRID))

    summary = dict(
        experiment=exp,
        variant=variant_name,
        exploded=False,
        gen_exact_energy_pJ=gen_exact_pJ,
        ref_val_energy_pJ=E_ref_hp * 1e12,
        energy_ratio=energy_hp_ratio,
        benchmark_target_pJ=benchmark,
        within_10_90=bool(benchmark[0] <= gen_exact_pJ <= benchmark[2]),
        covariance_trace_ratio=cov_tr,
        mean_relative_bias=bias,
        modal_variance_ratios=var_ratios.tolist(),
        mean_modal_variance_ratio=float(np.mean(var_ratios)),
        peak_contrast_error_db=official_peak_mae_db,
        detected_peaks=official_n_peaks
    )
    with open(eval_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{exp} | {variant_name}] Results:")
    print(f"  Physical Energy: {gen_exact_pJ:.2f} pJ (Target: [{benchmark[0]:.1f}, {benchmark[2]:.1f}] pJ, Ratio vs Val: {energy_hp_ratio:.3f}x)")
    print(f"  Within [10, 90]: {benchmark[0] <= gen_exact_pJ <= benchmark[2]}")
    print(f"  Covariance Trace Ratio: {cov_tr:.3f}x")
    print(f"  Mean Modal Variance Ratio: {np.mean(var_ratios):.3f}x (M1: {var_ratios[0]:.2f}, M2: {var_ratios[1]:.2f}, M3: {var_ratios[2]:.2f})")
    print(f"  Mean Relative Bias: {bias:.4f}")
    print(f"  Peak Contrast Error: {official_peak_mae_db:.2f} dB ({official_n_peaks} peaks)")
    return summary


def run_ablation(exp: str, base_dir: Path, variants: list[str]):
    manifest_path = Path(f"runs/clean_production/{exp}/remainder_data/data/manifest.json")
    osc_dir = Path(f"runs/clean_production/{exp}/oscillators")

    results = {}
    for var in variants:
        v_dir = base_dir / f"{exp}_{var}"
        v_dir.mkdir(parents=True, exist_ok=True)
        raw_ckpt = train_variant(exp, var, v_dir, manifest_path, epochs=5)
        int_ckpt = finetune_diffusion(raw_ckpt, manifest_path, iterations=100)
        res = evaluate_variant(exp, var, v_dir, int_ckpt, osc_dir, manifest_path, horizon=50000)
        results[var] = res

    # Save summary table
    with open(base_dir / f"{exp}_ablation_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==========================================")
    print(f"[{exp}] ABLATION COMPARISON SUMMARY")
    print(f"==========================================")
    print(f"{'Variant':<18} | {'E_gen (pJ)':<10} | {'E_ratio':<8} | {'Cov_tr':<8} | {'Bias':<6} | {'Peak_Err':<12} | {'M1':<6} | {'M2':<6} | {'M3':<6} | {'M_mean':<6}")
    print("-" * 105)
    for var in variants:
        r = results[var]
        if r.get("exploded"):
            print(f"{var:<18} | EXPLODED / DIVERGED")
            continue
        v_r = r["modal_variance_ratios"]
        print(f"{var:<18} | {r['gen_exact_energy_pJ']:<10.2f} | {r['energy_ratio']:<8.3f} | {r['covariance_trace_ratio']:<8.3f} | {r['mean_relative_bias']:<6.3f} | {r['peak_contrast_error_db']:<5.2f}dB ({r['detected_peaks']:02d}) | {v_r[0]:<6.3f} | {v_r[1]:<6.3f} | {v_r[2]:<6.3f} | {r['mean_modal_variance_ratio']:<6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["0p10"])
    parser.add_argument("--variants", nargs="+", default=["diagonal", "unrestricted_tcn"])
    parser.add_argument("--output-base", type=Path, default=Path("runs/ablation_drift"))
    args = parser.parse_args()

    for exp in args.experiments:
        run_ablation(exp, args.output_base, args.variants)


if __name__ == "__main__":
    main()
