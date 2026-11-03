"""Resumable fixed architecture/calibration search for the ten-peak remainder."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .adaptive_controller import read_json, write_json, run_logged


def _ratio_penalty(value):
    return abs(math.log(max(float(value), 1e-12)))


def extract_candidate(output):
    """Read the final filtered hybrid, which is the declared model output."""
    report = read_json(output / "summary.json")["reports"]["hybrid_filtered"]
    quantitative = read_json(Path(report["common_metrics"]))
    energy = quantitative["energy"]["untrimmed"]["generated_over_reference"]
    psd = quantitative["psd"]["signals"]["center"][
        "mean_absolute_log10_band_power_error_decades"]
    pod_psd = quantitative["psd"]["signals"]["center_without_spatial_mean"][
        "mean_absolute_log10_band_power_error_decades"]
    peak_error = report["peaks"]["peak_contrast_error_db"]
    metrics = {
        "energy_ratio": energy,
        "overall_center_psd_error_decades": psd,
        "pod_only_center_psd_error_decades": pod_psd,
        "peak_contrast_error_db": peak_error,
        "peak_band_power_ratio": report["center_band_power_ratio"],
        "acf_rmse": report["acf_rmse_first_4000_lags"],
        "increment_variance_ratio": report["increment_variance_ratio"],
        "normalized_csd_rmse": report["normalized_csd_rmse"],
        "coherence_complex_rmse": report["coherence_complex_rmse"],
    }
    metrics["primary_score"] = (3*_ratio_penalty(energy) + 3*float(psd)
        + float(peak_error)/4 + _ratio_penalty(metrics["peak_band_power_ratio"]))
    metrics["priority_pass"] = bool(0.75 <= energy <= 1.3 and psd <= .35
        and peak_error <= 4 and .75 <= metrics["peak_band_power_ratio"] <= 1.25)
    return metrics


def command_for_arm(study, arm, root, repo):
    command = [study["python"], "-m", "modelling.neural_sde", "train"]
    if arm["mode"] == "fresh":
        command += study["fresh_base"] + [
            "--temporal-component-manifest", str(repo/study["component_manifest"]),
            "--temporal-component", "remainder"]
    else:
        parent = (repo/study["stable_parent"] if arm["mode"] == "finetune_external" else
                  root/"trials"/arm["parent"]/"best.pt")
        command += ["--checkpoint", str(parent), "--train-mode", "joint",
                    "--batch-size", "256", "--indexed-training-batches", "--device", "cuda"]
    return command + arm["options"] + ["--output", str(root/"trials"/arm["id"])]


def evaluation_command(study, checkpoint, output, settings, root, repo):
    return [study["python"], "-m", "modelling.frequency_hybrid.ten_peak_remainder", "evaluate",
        "--checkpoint", str(checkpoint), "--output", str(output), "--study",
        str((repo/study["component_manifest"]).parent.parent), "--oscillators",
        str(repo/study["oscillators"]), "--horizon", str(settings["horizon"]),
        "--conditions", str(settings["conditions"]), "--ensemble", str(settings["ensemble"]),
        "--seed", str(settings["seed"]), "--device", settings["device"],
        "--diffusion-scale", "1.0"]


def run_study(path, dry_run=False):
    path = Path(path).resolve(); study = read_json(path); repo = Path(study["repository"])
    root = repo/study["output_root"]
    commands = {arm["id"]: command_for_arm(study, arm, root, repo) for arm in study["arms"]}
    if dry_run:
        for arm in study["arms"]:
            print("TRAIN", arm["id"], commands[arm["id"]])
            if arm["evaluate"]:
                print("SCREEN", arm["id"], evaluation_command(study,
                    root/"trials"/arm["id"]/"best.pt", root/"trials"/arm["id"]/"screen",
                    study["screen"], root, repo))
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/"controller.lock").open("a")
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); raise RuntimeError("A controller is already running this study") from None
    digest = hashlib.sha256(path.read_bytes()).hexdigest(); state_path = root/"state.json"
    state = read_json(state_path) if state_path.exists() else {
        "version": 1, "status": "active", "study_sha256": digest,
        "started_at_unix": time.time(), "trials": {}, "confirmations": {}}
    if state["study_sha256"] != digest:
        raise ValueError("Study changed after launch; use a new output root")
    if state["status"] not in ("active", "stopped"):
        print(f"Study is {state['status']}; refusing to restart"); return
    state["status"] = "active"; state["controller_pid"] = os.getpid()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR=str(root/".mplconfig"), PYTHONUNBUFFERED="1")
    bindir = str(Path(study["python"]).parent); env["PATH"] = bindir+os.pathsep+env.get("PATH", "")
    ptxas = Path(bindir)/"ptxas"
    if ptxas.exists(): env["TRITON_PTXAS_PATH"] = str(ptxas)
    deadline = state["started_at_unix"] + study["total_timeout_seconds"]
    def save():
        state["elapsed_seconds"] = time.time()-state["started_at_unix"]
        write_json(state_path, state)
    def job(command, label, timeout):
        remaining = deadline-time.time()
        if remaining <= 0: raise TimeoutError("Whole-study time budget exhausted")
        state["current_job"] = label; save()
        run_logged(command, root/"logs"/f"{label}.log", repo, env,
                   timeout_seconds=min(timeout, remaining))
    def stop(signum, frame): raise KeyboardInterrupt
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGHUP)}
    save()
    try:
        state["baseline"] = extract_candidate(repo/study["baseline_evaluation"])
        save()
        for arm in study["arms"]:
            row = state["trials"].setdefault(arm["id"], {
                "status": "planned", "description": arm["description"], "evaluate": arm["evaluate"]})
            if row["status"] in ("completed", "failed", "cancelled_dependency"): continue
            if arm["mode"] == "finetune" and state["trials"].get(arm["parent"], {}).get("status") != "completed":
                row["status"] = "cancelled_dependency"; save(); continue
            run = root/"trials"/arm["id"]
            try:
                if not (run/"best.pt").exists():
                    row["status"] = "training"; save()
                    job(commands[arm["id"]], f"train_{arm['id']}", study["training_timeout_seconds"])
                row["status"] = "trained"; save()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                row.update(status="failed", stage="training", error=repr(error)); save(); continue
            if arm["evaluate"]:
                output = run/"screen"
                try:
                    if not (output/"summary.json").exists():
                        row["status"] = "screening"; save()
                        job(evaluation_command(study, run/"best.pt", output, study["screen"], root, repo),
                            f"screen_{arm['id']}", study["evaluation_timeout_seconds"])
                    row["metrics"] = extract_candidate(output)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError,
                        KeyError, FileNotFoundError) as error:
                    row.update(status="failed", stage="screen", error=repr(error)); save(); continue
            row["status"] = "completed"; save()
        candidates = sorted(((row["metrics"]["primary_score"], name) for name,row in
            state["trials"].items() if row.get("status") == "completed" and "metrics" in row))
        for _, name in candidates[:study["confirmation"]["count"]]:
            row = state["confirmations"].setdefault(name, {"status": "planned"})
            if row["status"] in ("completed", "failed"): continue
            checkpoint = root/"trials"/name/"best.pt"; output = root/"confirmations"/name
            try:
                if not (output/"summary.json").exists():
                    row["status"] = "running"; save()
                    job(evaluation_command(study, checkpoint, output, study["confirmation"], root, repo),
                        f"confirm_{name}", study["evaluation_timeout_seconds"])
                row.update(status="completed", metrics=extract_candidate(output)); save()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError,
                    KeyError, FileNotFoundError) as error:
                row.update(status="failed", error=repr(error)); save()
        state["ranking"] = [name for _,name in candidates]
        state["confirmation_ranking"] = [name for _,name in sorted(
            (row["metrics"]["primary_score"], name) for name,row in
            state["confirmations"].items() if row.get("status") == "completed")]
        failures = any(row["status"] in ("failed", "cancelled_dependency")
                       for row in state["trials"].values()) or any(
            row["status"] == "failed" for row in state["confirmations"].values())
        state["status"] = ("completed_with_failures" if candidates and failures else
                           "completed" if candidates else "completed_with_failures")
    except TimeoutError as error: state.update(status="budget_exhausted", error=str(error))
    except KeyboardInterrupt: state["status"] = "stopped"; raise
    except Exception as error: state.update(status="failed", error=repr(error)); raise
    finally:
        state["current_job"] = None; save(); write_json(root/"results.json", state)
        for sig,handler in handlers.items(): signal.signal(sig, handler)
        lines = ["# Ten-peak remainder architecture study", "", f"Status: {state['status']}", "",
            "All candidate screens use validation data and learned diffusion scale 1.0.", "",
            "| Candidate | Status | Energy | PSD decades | Peak error dB | Band power | Score |",
            "|---|---|---:|---:|---:|---:|---:|"]
        baseline = state.get("baseline", {})
        if baseline:
            lines.append(f"| previous TCN63 baseline | reference | {baseline['energy_ratio']:.4g} | "
                f"{baseline['overall_center_psd_error_decades']:.4g} | "
                f"{baseline['peak_contrast_error_db']:.4g} | "
                f"{baseline['peak_band_power_ratio']:.4g} | {baseline['primary_score']:.4g} |")
        for name,row in state["trials"].items():
            m = row.get("metrics", {})
            def show(key): return "" if key not in m else f"{m[key]:.4g}"
            lines.append(f"| {name} | {row['status']} | {show('energy_ratio')} | "
                f"{show('overall_center_psd_error_decades')} | {show('peak_contrast_error_db')} | "
                f"{show('peak_band_power_ratio')} | {show('primary_score')} |")
        (root/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(state_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("study", type=Path)
    parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    run_study(args.study, args.dry_run)


if __name__ == "__main__": main()
