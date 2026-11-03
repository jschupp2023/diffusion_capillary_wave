"""Bounded fixed queue for the cached oscillator readout, via adaptive_controller.

Reuses the existing fitter, evaluator, plotting, health gates and atomic ledger.
No advisor may expand this queue or restart the cancelled TCN campaign.
"""
import hashlib
import fcntl
import os
from pathlib import Path
import subprocess
import signal
import time

import numpy as np
import torch

from .adaptive_controller import (read_json, write_json, run_logged, rollout_health,
                                  screen_gate_failures, extract_metrics)
from ..train import load_model
from modelling.acdm.conditional_edm.checkpoint import save_checkpoint


def fit_command(study, arm, root):
    return [study["python"], "-m", "modelling.neural_sde.fit_fixed_memory",
        "--parent", study["parent"], "--oscillators", study["oscillators"],
        "--output", str(root / "fits" / arm["id"]),
        "--train-windows", str(arm["train_windows"]),
        "--validation-windows", str(study["validation_windows"]),
        "--window", str(study["window"]), "--seed", str(arm["seed"]),
        "--ridge", *map(str, study["ridge"]), "--max-seconds", str(study["fit_timeout_seconds"]-10)]


@torch.no_grad()
def materialize_variant(fit_dir, destination, *, zero=False):
    """Keep official best.pt intact; explicitly label a rejected nonzero probe."""
    model, payload = load_model(fit_dir / "best.pt")
    report = read_json(fit_dir / "report.json")
    variant = "selected_correction"
    if zero:
        model.fixed_memory.readout.weight.zero_()
        variant = "matched_zero_readout_parent"
    elif report["selected_penalty"] is None:
        candidate = min(report["scores"][1:], key=lambda row: row["validation_error"])
        with np.load(fit_dir / "regression.npz") as arrays:
            x = arrays["train_features"].astype(np.float64)
            y = arrays["train_residuals"].astype(np.float64)
        scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-8)
        x /= scale
        weight = np.linalg.solve(x.T@x/len(x) + candidate["penalty"]*np.eye(x.shape[1]), x.T@y/len(x))
        factor = (model.diffusion_matrix()/model.target_std[:, None]).numpy().astype(np.float64)
        model.fixed_memory.feature_scale.copy_(torch.from_numpy(scale))
        model.fixed_memory.readout.weight.copy_(torch.from_numpy(factor@weight.T))
        variant = "exploratory_nonzero_rejected_by_one_step_selection"
    else:
        candidate = next(row for row in report["scores"] if row["penalty"] == report["selected_penalty"])
    details = dict(variant=variant, source_fit=str(fit_dir),
                   selected_penalty=report["selected_penalty"],
                   screened_penalty=None if zero else candidate["penalty"],
                   parent_validation_error=report["scores"][0]["validation_error"],
                   screened_validation_error=report["scores"][0]["validation_error"] if zero else candidate["validation_error"])
    payload["model_state"] = model.state_dict()
    payload["training"] = {**payload["training"], "controller_variant": details}
    destination.mkdir(parents=True, exist_ok=True)
    save_checkpoint(destination / "best.pt", payload)
    write_json(destination / "config.json", dict(data=payload["data_config"],
               model=payload["model_config"], training=payload["training"]))
    write_json(destination / "variant.json", details)
    return details


def assert_matched(reference_path, candidate_path):
    with np.load(reference_path, allow_pickle=False) as a, np.load(candidate_path, allow_pickle=False) as b:
        for key in ("reference", "reference_time", "repetition", "start_index"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"Unmatched reference {key}; comparisons are invalid")
        return dict(repetitions=b["repetition"].tolist(), start_indices=b["start_index"].tolist())


def run_study(path, *, dry_run=False):
    study = read_json(path)
    repo = Path(study["repository"])
    root = repo / study["output_root"]
    if dry_run:
        for arm in study["arms"]:
            print(fit_command(study, arm, root))
        print("Matched parent + one corrected screen per arm:", study["screen"])
        return
    torch.set_num_threads(2)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("a controller is already running this study") from None
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    state_path = root / "state.json"
    state = read_json(state_path) if state_path.exists() else dict(
        status="active", study_sha256=digest, started_at_unix=time.time(), fits={}, screens={})
    if state["study_sha256"] != digest:
        raise ValueError("Study changed since launch; use a new output root")
    if state["status"] != "active":
        print(f"Study is {state['status']}; refusing to restart automatically")
        return
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               MPLCONFIGDIR=str(root / ".mplconfig"), PYTHONUNBUFFERED="1")
    deadline = state["started_at_unix"] + study["total_timeout_seconds"]
    state["controller_pid"] = os.getpid()

    def save():
        state["elapsed_seconds"] = time.time()-state["started_at_unix"]
        write_json(state_path, state)

    def job(command, label, timeout):
        remaining = deadline-time.time()
        if remaining <= 0:
            raise TimeoutError("whole-study time budget exhausted")
        state["current_job"] = label
        save()
        run_logged(command, root / "logs" / f"{label}.log", repo, env,
                   timeout_seconds=min(timeout, remaining))

    def stop(signum, frame):
        raise KeyboardInterrupt

    previous_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGHUP)}
    save()
    try:
        for arm in study["arms"]:
            fit_dir = root / "fits" / arm["id"]
            record = state["fits"].setdefault(arm["id"], dict(status="planned", settings=arm))
            if record["status"] == "failed":
                continue
            try:
                if not (fit_dir / "best.pt").exists():
                    record["status"] = "fitting"
                    job(fit_command(study, arm, root), f"fit_{arm['id']}", study["fit_timeout_seconds"])
                record.update(status="completed", report=read_json(fit_dir / "report.json"))
            except subprocess.CalledProcessError as error:
                record.update(status="failed", returncode=error.returncode)
            save()
        completed = [arm for arm in study["arms"] if state["fits"][arm["id"]]["status"] == "completed"]
        if not completed:
            raise RuntimeError("all bounded fits failed; inspect logs")
        first = root / "fits" / completed[0]["id"]
        candidates = [("parent", first, True)] + [(arm["id"], root/"fits"/arm["id"], False) for arm in completed]
        settings = study["screen"]
        for name, fit_dir, zero in candidates:
            run_dir = root / "screens" / name
            record = state["screens"].setdefault(name, dict(status="planned"))
            if record["status"] in {"completed", "rejected", "failed"}:
                continue
            try:
                if not (run_dir / "best.pt").exists():
                    record["variant"] = materialize_variant(fit_dir, run_dir, zero=zero)
                else:
                    record["variant"] = read_json(run_dir / "variant.json")
                rollout = run_dir / "rollout"
                if not (rollout / "rollout.npz").exists():
                    command = [study["python"], "-m", "modelling.neural_sde", "evaluate",
                        "--checkpoint", str(run_dir/"best.pt"), "--output", str(rollout),
                        "--split", "validation", "--device", "cpu", "--no-metrics",
                        "--num-conditions", str(settings["conditions"]),
                        "--ensemble-size", str(settings["ensemble_size"]),
                        "--horizon", str(settings["horizon"]), "--seed", str(settings["seed"])]
                    job(command, f"rollout_{name}", study["rollout_timeout_seconds"])
                record["origins"] = assert_matched(root/"screens"/"parent"/"rollout"/"rollout.npz", rollout/"rollout.npz")
                record["health"] = rollout_health(rollout/"rollout.npz", run_dir/"best.pt")
                failures = screen_gate_failures(study["screen_gate"], record["health"])
                if failures:
                    record.update(status="rejected", rejection_reasons=failures)
                    save()
                    continue
                if not (rollout/"quantitative"/"summary.json").exists():
                    command = [study["python"], "-m", "data_analysis.plot_rollout", str(rollout),
                        "--overwrite", "--nperseg", str(settings["nperseg"]),
                        "--peak-nperseg", str(settings["nperseg"]), "--peak-band", "100", "1500",
                        "--discard", str(settings["discard"]), "--modal-psd", "--modal-psd-modes", "1", "2", "3", "4", "5", "10", "20"]
                    job(command, f"analysis_{name}", study["analysis_timeout_seconds"])
                record["metrics"] = extract_metrics(run_dir, rollout)
                failures = screen_gate_failures(study["screen_gate"], record["health"], record["metrics"])
                record.update(status="rejected" if failures else "completed", rejection_reasons=failures)
            except subprocess.CalledProcessError as error:
                record.update(status="failed", returncode=error.returncode)
            save()
        state["status"] = ("budget_exhausted" if time.time() >= deadline else
            "completed" if not any(x["status"] == "failed" for x in [*state["fits"].values(), *state["screens"].values()])
            else "completed_with_failures")
    except TimeoutError as error:
        state.update(status="budget_exhausted", error=str(error))
    except KeyboardInterrupt:
        state["status"] = "stopped"
        raise
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        state["current_job"] = None
        save()
        write_json(root/"summary.json", state)
        lines = ["# Fixed oscillator-memory study", "", f"Status: {state['status']}", "",
                 "See state.json and summary.json for fit scores, exact reference origins, health gates and PSD/moment metrics.",
                 "A rejected one-step correction is labelled exploratory; passing amplitude gates is not evidence of peak recovery.",
                 "Validation repetitions only. Shared starts are paired comparisons, not independent experimental repetitions.", "",
                 "| Arm | Status | Variant |", "|---|---|---|"]
        for name, row in state["screens"].items():
            lines.append(f"| {name} | {row['status']} | {row.get('variant', {}).get('variant', '')} |")
        (root/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(state_path)
