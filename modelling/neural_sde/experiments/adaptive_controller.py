"""Run an event-driven Neural-SDE experiment campaign.

The controller itself contains no optimizer.  It runs the existing training,
evaluation, and plotting CLIs, writes a compact evidence record, and invokes a
read-only advisor only after a job has finished.  The durable JSON state makes
the campaign resumable without keeping an interactive agent alive while the
GPU is busy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import threading
from typing import Any


NEW_MODEL_ONLY = {
    "--experiment", "--rank", "--data-root", "--shared-basis",
    "--validation-reps", "--test-reps", "--no-spatial-mean",
    "--spatial-mean-highpass-hz", "--state-variable", "--lag",
    "--downsample-factor", "--downsample-edge-trim",
    "--lag-steps", "--history", "--history-offsets", "--hidden-layers",
    "--activation", "--drift-type", "--stiffness-init", "--damping-init",
    "--residual-architecture", "--tcn-context-frames", "--tcn-channels",
    "--tcn-kernel-size", "--tcn-dilations",
    "--damped-operator", "--surface-tension", "--diffusion-init",
    "--diffusion-type", "--diffusion-log-range", "--dt",
    "--velocity-memory-offsets", "--velocity-memory-rank",
    "--velocity-memory-norm-cap",
    "--velocity-oscillator-periods", "--velocity-oscillator-burnin",
    "--velocity-oscillator-radius-init", "--velocity-oscillator-norm-cap",
    "--velocity-innovation-offsets", "--velocity-innovation-rank",
    "--velocity-innovation-norm-cap",
}
RESERVED_FLAGS = {"--output", "--checkpoint", "--device"}
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")


class ManualDecisionRequired(RuntimeError):
    """The durable review packet is ready for a human-supplied decision."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnostic_history_valid(summary_path: Path, requested_history: int) -> bool:
    """Require the current auditable eligibility schema, not a stale report."""
    if not summary_path.is_file():
        return False
    try:
        report = read_json(summary_path)
        effective = int(report["eligibility_history_steps"])
        requested = int(report["requested_eligibility_history_steps"])
        conditioning = int(report["model_conditioning_history_steps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    identifiers = report.get("selected_window_identifiers_by_batch")
    return (requested == requested_history
            and effective >= max(requested_history, conditioning)
            and isinstance(identifiers, dict) and bool(identifiers)
            and all(record.get("sha256") for record in identifiers.values()))


def resolve_executable(command: str) -> str:
    """Resolve an executable, including Codex bundled with VS Code remote."""
    expanded = Path(command).expanduser()
    if expanded.parent != Path("."):
        if expanded.is_file() and os.access(expanded, os.X_OK):
            return str(expanded.resolve())
        raise FileNotFoundError(f"Configured executable is missing or not executable: {expanded}")
    if command == "codex":
        # Prefer the bundled client: distro packages can lag the model schema
        # and fail before an advisor request is submitted.
        architecture = {"x86_64": "linux-x86_64", "aarch64": "linux-aarch64"}.get(
            platform.machine(), f"linux-{platform.machine()}")
        candidates = []
        for root in (Path.home() / ".vscode-server/extensions",
                     Path.home() / ".vscode/extensions"):
            candidates.extend(path for path in root.glob(
                f"openai.chatgpt-*/bin/{architecture}/codex")
                if path.is_file() and os.access(path, os.X_OK))
        if candidates:
            return str(sorted(candidates)[-1].resolve())
    found = shutil.which(command)
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find {command!r} on PATH or in a VS Code OpenAI extension. "
        "Set advisor.command in the study JSON to an absolute executable path.")


def render_options(options: dict[str, list[str]]) -> list[str]:
    result: list[str] = []
    for flag, tokens in options.items():
        if not flag.startswith("--") or flag in RESERVED_FLAGS:
            raise ValueError(f"Invalid or controller-owned training flag: {flag}")
        result.append(flag)
        result.extend(str(token) for token in tokens)
    return result


def normalized_options(raw: dict[str, Any]) -> dict[str, list[str]]:
    result = {}
    for flag, value in raw.items():
        if value is None or value is False:
            continue
        if value is True:
            result[flag] = []
        elif isinstance(value, list):
            result[flag] = [str(item) for item in value]
        else:
            result[flag] = [str(value)]
    return result


def apply_overrides(base: dict[str, list[str]], overrides: list[dict[str, Any]],
                    allowed: dict[str, list[Any]], max_changes: int) -> dict[str, list[str]]:
    result = {key: list(value) for key, value in base.items()}
    seen = set()
    change_count = 0
    for item in overrides:
        flag = item["flag"]
        if flag in seen or flag not in allowed or flag in RESERVED_FLAGS:
            raise ValueError(f"Override is duplicated, reserved, or outside the study space: {flag}")
        seen.add(flag)
        proposed = None if item["remove"] else tuple(str(v) for v in item["tokens"])
        alternatives = {
            None if value is None else tuple(str(v) for v in value)
            for value in allowed[flag]
        }
        if proposed not in alternatives:
            raise ValueError(f"Value {proposed!r} is outside the allowed alternatives for {flag}.")
        current = tuple(result[flag]) if flag in result else None
        if proposed != current:
            change_count += 1
            if change_count > max_changes:
                raise ValueError(
                    f"Advisor requested {change_count} effective changes; "
                    f"limit is {max_changes}.")
        if proposed is None:
            result.pop(flag, None)
        else:
            result[flag] = list(proposed)
    return result


def validate_semantics(options: dict[str, list[str]]) -> None:
    diffusion = options.get("--diffusion-type", ["diagonal"])
    if diffusion != ["bounded_state_diagonal"] and "--diffusion-log-range" in options:
        raise ValueError("--diffusion-log-range is valid only for bounded_state_diagonal.")
    horizons = options.get("--multistep-horizons", [])
    for flag in ("--multistep-mode-counts", "--multistep-particle-counts",
                 "--multistep-horizon-weights"):
        if flag in options and len(options[flag]) != len(horizons):
            raise ValueError(f"{flag} needs one value per multistep horizon.")
    if "--multistep-particle-counts" in options:
        values = [int(value) for value in options["--multistep-particle-counts"]]
        if any(right > left for left, right in zip(values, values[1:])):
            raise ValueError("Multistep particle counts must be nonincreasing.")
    if "--velocity-memory-rank" in options and "--velocity-memory-offsets" not in options:
        raise ValueError("--velocity-memory-rank requires velocity-memory offsets.")
    if "--velocity-memory-norm-cap" in options and "--velocity-memory-rank" not in options:
        raise ValueError("--velocity-memory-norm-cap requires a velocity-memory rank.")
    oscillator = "--velocity-oscillator-periods" in options
    finite_memory = "--velocity-memory-offsets" in options
    innovation = "--velocity-innovation-offsets" in options
    if sum((oscillator, finite_memory, innovation)) > 1:
        raise ValueError("Velocity-memory families are mutually exclusive.")
    for flag in ("--velocity-oscillator-burnin", "--velocity-oscillator-radius-init",
                 "--velocity-oscillator-norm-cap"):
        if flag in options and not oscillator:
            raise ValueError(f"{flag} requires velocity-oscillator periods.")
    if "--velocity-innovation-rank" in options and not innovation:
        raise ValueError("--velocity-innovation-rank requires innovation offsets.")
    if innovation and "--velocity-innovation-rank" not in options:
        raise ValueError("Innovation offsets require --velocity-innovation-rank.")
    if "--velocity-innovation-norm-cap" in options and not innovation:
        raise ValueError("--velocity-innovation-norm-cap requires innovation offsets.")


def run_logged(command: list[str], log_path: Path, cwd: Path,
               env: dict[str, str] | None = None,
               timeout_seconds: float | None = None) -> None:
    if (timeout_seconds is not None
            and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0)):
        raise ValueError("timeout_seconds must be positive and finite")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
            start_new_session=timeout_seconds is not None)
        timed_out = threading.Event()
        timer = None
        kill_timer = None
        if timeout_seconds is not None:
            def terminate_on_timeout():
                nonlocal kill_timer
                if process.poll() is None:
                    timed_out.set()
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    def force_kill():
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    kill_timer = threading.Timer(5., force_kill)
                    kill_timer.daemon = True
                    kill_timer.start()
            timer = threading.Timer(timeout_seconds, terminate_on_timeout)
            timer.daemon = True
            timer.start()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            code = process.wait()
        except BaseException:
            if timeout_seconds is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise
        finally:
            if timer is not None:
                timer.cancel()
            if kill_timer is not None:
                kill_timer.cancel()
        if timed_out.is_set():
            code = 124
            log.write(f"TIMEOUT {timeout_seconds:g} seconds\n")
        log.write(f"EXIT {code}\n")
    if code:
        raise subprocess.CalledProcessError(code, command)


def get_path(value: Any, *keys: str, default=None):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def training_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {}
    best = min(rows, key=lambda row: row.get(
        "validation_total_loss", row.get("validation_nll", float("inf"))))
    result = {
        "epochs_completed": len(rows),
        "last_train_nll": rows[-1].get("train_nll"),
        "last_validation_nll": rows[-1].get("validation_nll"),
        "best_validation_total_epoch": best.get("epoch"),
        "best_validation_total_loss": best.get("validation_total_loss", best.get("validation_nll")),
    }
    result.update({key: value for key, value in best.items()
                   if key.startswith(("validation_stationary_lagged_covariance_",
                                      "validation_stationary_mean_",
                                      "validation_stationary_variance_",
                                      "validation_one_step", "validation_multistep",
                                      "validation_restoring_orthogonality_",
                                      "validation_restoring_residual_projection_",
                                      "validation_damping_orthogonality_",
                                      "validation_diagonal_linear_orthogonality_",
                                      "validation_diagonal_linear_residual_"))
                   and not key.endswith("_normalized_error")})
    history_prefixes = (
        "train_one_step", "validation_one_step",
        "train_multistep_loss_", "validation_multistep_loss_",
        "train_multistep_weighted_", "validation_multistep_weighted_",
        "train_stationary_mean_loss", "validation_stationary_mean_loss",
        "train_stationary_mean_weighted_", "validation_stationary_mean_weighted_",
        "train_stationary_mean_normalized_error_rmse",
        "validation_stationary_mean_normalized_error_rmse",
        "train_stationary_variance_loss", "validation_stationary_variance_loss",
        "train_stationary_variance_weighted_",
        "validation_stationary_variance_weighted_",
        "train_stationary_variance_normalized_error_rmse",
        "validation_stationary_variance_normalized_error_rmse",
    )
    result["history"] = [{
        key: value for key, value in row.items()
        if key in {"epoch", "train_nll", "validation_nll", "train_total_loss",
                   "validation_total_loss"}
        or (key.startswith(history_prefixes)
            and not key.endswith("_normalized_error"))
    } for row in rows]
    return result


def configuration_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    config = read_json(path)
    model = config.get("model", {})
    training = config.get("training", {})
    return {
        "data": {key: config.get("data", {}).get(key) for key in (
            "downsample_factor", "downsample_edge_trim", "spatial_mean_highpass_hz")},
        "model": {key: model.get(key) for key in (
            "state_variable", "dt", "lag_steps", "history_steps", "history_offsets",
            "velocity_memory_offsets", "velocity_memory_rank",
            "velocity_memory_norm_cap", "velocity_oscillator_periods",
            "velocity_oscillator_burnin", "velocity_oscillator_radius_init",
            "velocity_oscillator_norm_cap", "velocity_innovation_offsets",
            "velocity_innovation_rank", "velocity_innovation_norm_cap",
            "hidden_layers", "drift_type",
            "residual_architecture", "tcn_context_frames", "tcn_channels",
            "tcn_kernel_size", "tcn_dilations",
            "damped_operator", "structured_damping", "diffusion_type", "diffusion_log_range")},
        "training": {key: training.get(key) for key in (
            "train_mode", "train_trim_percent", "restoring_orthogonality_weight",
            "restoring_orthogonality_adjust_velocity", "damping_orthogonality_weight",
            "diagonal_linear_orthogonality_weight", "multistep_weight", "multistep_objective",
            "multistep_horizons", "multistep_horizon_weights",
            "multistep_position_weight", "multistep_velocity_weight",
            "multistep_mode_counts", "multistep_particle_counts",
            "multistep_target", "multistep_gradient_groups", "multistep_window_sampling",
            "multistep_window_fraction", "stationary_mean_weight",
            "stationary_variance_weight", "stationary_lagged_covariance_weight",
            "stationary_variance_pod_mode_count",
            "stationary_lagged_covariance_mode_count",
            "stationary_lagged_covariance_epsilon", "stationary_mean_horizon",
            "stationary_mean_horizons", "stationary_mean_trajectories",
            "stationary_mean_batch_fraction", "learning_rate", "epochs", "seed")},
    }


def _moment_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    stats = summary.get("coefficient_statistics", {})
    pod = stats.get("pod_fluctuation_coordinates", {})
    trimmed = pod.get("generated_untrimmed_vs_reference_training_trimmed")
    population = trimmed or pod.get("untrimmed", {})
    spatial = stats.get("spatial_mean_coordinate", {})
    spatial_population = (spatial.get("generated_untrimmed_vs_reference_training_trimmed")
                          or spatial.get("untrimmed", {}))
    return {
        "reference_population": "training_trimmed" if trimmed else "untrimmed",
        "pod_mean_standardized_rmse": population.get("mean_bias_reference_standardized_rmse"),
        "pod_covariance_trace_ratio": population.get("covariance_trace_ratio"),
        "pod_covariance_relative_error": population.get("covariance_relative_frobenius_error"),
        "spatial_mean_bias_standardized": spatial_population.get(
            "mean_bias_reference_standardized_rmse"),
        "spatial_mean_variance_ratio": spatial_population.get("covariance_trace_ratio"),
    }


def extract_metrics(run_dir: Path, rollout_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "configuration": configuration_summary(run_dir),
        "training": training_metrics(run_dir),
    }
    quantitative_path = rollout_dir / "quantitative" / "summary.json"
    if quantitative_path.is_file():
        quantitative = read_json(quantitative_path)
        if quantitative.get("spectral_peaks"):
            result["spectral_peaks"] = quantitative["spectral_peaks"]
        energy = quantitative.get("energy", {})
        trimmed = energy.get("generated_untrimmed_vs_reference_training_trimmed")
        comparison = trimmed or energy.get("untrimmed", {})
        result["rollout"] = {
            "horizon": quantitative.get("samples", 1) - 1,
            "conditions": quantitative.get("conditions"),
            "ensemble_size": quantitative.get("ensemble_size"),
            "energy_ratio": comparison.get("generated_over_reference"),
            **_moment_metrics(quantitative),
        }
        signals = get_path(quantitative, "psd", "signals", default={})
        result["psd"] = {
            name: {
                "mean_abs_log10_band_error": values.get(
                    "mean_absolute_log10_band_power_error_decades"),
                "bands": [{
                    "lower_hz": band.get("lower_hz"),
                    "upper_hz": band.get("upper_hz"),
                    "generated_over_reference": band.get("ratio"),
                    "absolute_log10_error_decades": band.get(
                        "absolute_log10_error_decades"),
                } for band in values.get("bands", [])],
            }
            for name, values in signals.items()
        }
    metrics_path = rollout_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = read_json(metrics_path)
        one_step = metrics.get("one_step", {})
        dynamics = metrics.get("rollout", {})
        result["one_step"] = {
            key: one_step.get(key) for key in (
                "conditional_mean_rmse", "conditional_mean_skill",
                "conditional_covariance_relative_error", "energy_score",
                "central_90_percent_coverage")
        }
        result["dynamics"] = {
            key: dynamics.get(key) for key in (
                "mean_rmse_average", "mean_relative_l2_error_global",
                "normalized_covariance_relative_error_average",
                "normalized_energy_relative_error_average",
                "consecutive_increment_correlation_rmse")
        }
    modal_path = rollout_dir / "modal" / "summary.json"
    if modal_path.is_file():
        modal = read_json(modal_path)
        population_name = modal.get("comparison_used_for_plot_and_primary_fields", "untrimmed")
        population = modal.get(population_name, modal)
        result["leading_modes"] = [
            {key: row.get(key) for key in (
                "mode", "rms_amplitude_ratio", "variance_ratio", "generated_mean",
                "reference_mean", "excess_fraction")}
            for row in population.get("modes", [])[:5]
        ]
    modal_psd_path = rollout_dir / "modal_psd" / "summary.json"
    if modal_psd_path.is_file():
        modal_psd = read_json(modal_psd_path)
        result["modal_psd"] = [{
            "mode": row.get("mode"),
            "mean_abs_log10_band_error": get_path(
                row, "band_power_comparison",
                "mean_absolute_log10_band_power_error_decades"),
            "power_ratio": row.get("generated_over_reference_power"),
        } for row in modal_psd.get("modes", [])]
    return result


def rollout_health(rollout_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Cheap per-path amplitude/nonfinite audit before expensive plotting.

    Uses the checkpoint's training position scales, not a small rollout's
    estimated variance. No clipping, trimming, or modification of trajectories.
    """
    import numpy as np
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    buffers = checkpoint["model_state"]
    mean = buffers["state_mean"].numpy().astype(float)
    scale = buffers["state_std"].numpy().astype(float)
    with np.load(rollout_path, allow_pickle=False) as saved:
        trajectories = saved["trajectories"].astype(float)
    if (trajectories.ndim != 4 or trajectories.shape[-1] != len(scale)
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid rollout shape or checkpoint normalization in screen audit")
    finite = np.isfinite(trajectories).all(-1)
    flat_finite = finite.reshape(-1, finite.shape[-1])
    first_bad = [int(np.flatnonzero(~path)[0]) if not path.all() else None
                 for path in flat_finite]
    # Spatial-mean presence is encoded by the data source, not reduced_dim.
    with rollout_path.with_name("metrics.json").open() as handle:
        metadata = json.load(handle)
    offset = int(metadata.get("include_spatial_mean", True))
    with np.errstate(over="ignore", invalid="ignore"):
        standardized = (trajectories - mean) / scale
        rms = np.sqrt(np.mean(standardized[..., offset:] ** 2, axis=-1))
        maximum_coordinate = np.max(np.abs(standardized), axis=-1)
    def finite_max(values):
        valid = values[np.isfinite(values)]
        return float(valid.max()) if valid.size else None
    return {
        "all_finite": bool(finite.all()),
        "nonfinite_paths": sum(value is not None for value in first_bad),
        "first_nonfinite_step_per_path": first_bad,
        "conditions": trajectories.shape[0], "ensemble_size": trajectories.shape[1],
        "horizon": trajectories.shape[2] - 1,
        "peak_pod_standardized_rms": finite_max(rms),
        "peak_absolute_standardized_coordinate": finite_max(maximum_coordinate),
        "amplitude_statistics_finite": bool(np.isfinite(rms).all()
                                             and np.isfinite(maximum_coordinate).all()),
    }


def screen_gate_failures(limits: dict[str, Any], health: dict[str, Any],
                         metrics: dict[str, Any] | None = None) -> list[str]:
    """Broad promotion safeguards, not scientific acceptance criteria."""
    failures = []
    if not health.get("all_finite") or not health.get("amplitude_statistics_finite"):
        failures.append("nonfinite trajectory or amplitude statistic")
    for key in ("peak_pod_standardized_rms", "peak_absolute_standardized_coordinate"):
        maximum = limits.get("max_" + key)
        value = health.get(key)
        if maximum is not None and (value is None or not math.isfinite(value) or value > maximum):
            failures.append(f"{key}={value} exceeds {maximum}")
    if metrics is not None:
        for key in ("energy_ratio", "pod_covariance_trace_ratio"):
            bounds = limits.get(key + "_range")
            if bounds is None:
                continue
            value = get_path(metrics, "rollout", key)
            if value is None or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                failures.append(f"{key}={value} outside {bounds}")
    return failures


def extract_innovation_metrics(summary_path: Path) -> dict[str, Any]:
    """Return the compact innovation evidence used by the advisor."""
    if not summary_path.is_file():
        return {}
    summary = read_json(summary_path)
    calibration = summary.get("whitened_calibration", {})
    acf = summary.get("acf", {}).get("whitened", {})
    psd = summary.get("psd_over_unit_white", {}).get("whitened", [])
    decorrelation = summary.get("decorrelation", {}).get("whitened", [])
    return {
        "split": summary.get("split", "train"),
        "retained_transitions": sum(
            row.get("retained_transitions", 0) for row in summary.get("retention", [])),
        "calibration": {
            key: calibration.get(key) for key in (
                "mean_rmse", "std_median", "std_min", "std_max",
                "covariance_relative_error_to_identity")
        },
        "acf": {
            lag: {key: values.get(key) for key in (
                "median_absolute_across_coordinates",
                "maximum_absolute_across_coordinates")}
            for lag, values in acf.items()
        },
        "psd_over_unit_white": [
            {key: band.get(key) for key in (
                "lower_hz", "upper_hz", "median_power_over_unit_white",
                "p10", "p90")}
            for band in psd
        ],
        "first_five_persistent_decorrelation_frames": [
            row.get("persistent_frames") for row in decorrelation[:5]
        ],
    }


def extract_objective_balance_metrics(summary_path: Path) -> dict[str, Any]:
    """Keep the gradient evidence needed by the advisor without stored vectors."""
    if not summary_path.is_file():
        return {}
    report = read_json(summary_path)
    return {
        key: report.get(key) for key in (
            "split", "window_count", "moment_trajectory_count",
            "multistep_horizons", "moment_horizons", "seeds",
            "parameter_counts", "multistep_gradient_groups", "objectives",
            "expected_gradient_cosines", "unweighted_routed_gradient_cosines",
            "window_seeds", "population_note", "noise_note",
            "requested_eligibility_history_steps", "eligibility_history_steps",
            "model_conditioning_history_steps", "batch_reports",
            "selected_window_identifiers_sha256",
            "selected_moment_window_identifiers_sha256",
            "combined_expected_gradient_norms", "multistep_horizon_gradients",
            "multistep_horizon_gradient_cosines", "multistep_diagnostic_overrides")
    }


class Controller:
    def __init__(self, study_path: Path, *, dry_run=False, manual=False):
        self.study_path = study_path.resolve()
        self.study = read_json(self.study_path)
        repository = self.study.get("repository")
        self.repo = Path(repository if repository is not None else
                         self.study_path.parents[3]).resolve()
        output = Path(self.study["output_root"])
        self.root = output if output.is_absolute() else self.repo / output
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path) if self.state_path.is_file() else {
            "version": 1, "status": "active", "trials": [], "decisions": []}
        # An explicit restart retries only the failed advisor boundary; all
        # completed training and evaluation stages remain durable.
        if self.state.get("status") == "advisor_failed":
            self.state["status"] = "active"
        self.python = self.study.get("python", sys.executable)
        self.dry_run, self.manual = dry_run, manual
        self.base = normalized_options(self.study["base_train_options"])
        self.allowed = self.study["allowed_overrides"]
        self._import_seeds()
        self._import_initial_trials()

    def save(self) -> None:
        write_json(self.state_path, self.state)

    @staticmethod
    def archive_diagnostic_output(output: Path) -> None:
        """Move a stale or partial diagnostic aside without deleting evidence."""
        if not output.is_dir() or not any(output.iterdir()):
            return
        index = 1
        while True:
            archived = output.with_name(f"{output.name}_invalid_{index}")
            if not archived.exists():
                output.rename(archived)
                return
            index += 1

    def objective_balance_current(self, trial: dict[str, Any]) -> bool:
        settings = self.study.get("objective_balance_diagnostic")
        if not settings:
            return True
        output = Path(trial["run_dir"]) / settings.get(
            "output_name", "objective_balance_diagnostic")
        return diagnostic_history_valid(
            output / "summary.json", int(settings.get("min_history", 0)))

    def score_sensitivity_current(self, trial: dict[str, Any]) -> bool:
        settings = self.study.get("energy_score_sensitivity")
        if not settings:
            return True
        output = Path(trial["run_dir"]) / settings.get(
            "output_name", "energy_score_sensitivity")
        return diagnostic_history_valid(
            output / "summary.json", int(settings.get("min_history", 0)))

    def _import_seeds(self) -> None:
        known = {trial["id"]: trial for trial in self.state["trials"]}
        for seed in self.study.get("seed_runs", []):
            run_dir = (self.repo / seed["run_dir"]).resolve()
            rollout_value = seed.get("rollout_dir")
            rollout_dir = ((self.repo / rollout_value).resolve()
                           if rollout_value else None)
            record = {
                "id": seed["id"], "kind": "seed", "status": "screened",
                "hypothesis": seed["hypothesis"], "change_family": seed["change_family"],
                "run_dir": str(run_dir), "rollouts": {}}
            if rollout_dir is not None:
                record["rollouts"]["seed"] = {
                    "path": str(rollout_dir),
                    "metrics": extract_metrics(run_dir, rollout_dir),
                }
            if seed["id"] in known:
                if known[seed["id"]].get("kind") != "seed":
                    raise ValueError(f"Seed id collides with a trained trial: {seed['id']}")
                known[seed["id"]].update(record)
            else:
                self.state["trials"].append(record)
        self.save()

    def _import_initial_trials(self) -> None:
        """Queue declared matched controls once, before consulting the advisor."""
        known = {trial["id"] for trial in self.state["trials"]}
        deferred = self.study.get("defer_initial_trial_dependencies", False)
        if deferred:
            # Dependencies must point backwards in the declaration, preventing
            # typos/cycles from silently dropping a required experimental arm.
            declared = {seed["id"] for seed in self.study.get("seed_runs", [])}
            for spec in self.study.get("initial_trials", []):
                if spec["id"] in declared:
                    raise ValueError(f"Duplicate initial trial id: {spec['id']}")
                if (spec["training_mode"] in {"finetune", "memory"}
                        and spec.get("parent_trial") not in declared):
                    raise ValueError("Deferred parent must be a seed or earlier initial trial")
                declared.add(spec["id"])
        for specification in self.study.get("initial_trials", []):
            trial_id = specification["id"]
            if trial_id in known:
                continue
            if deferred and specification["training_mode"] in {"finetune", "memory"}:
                parent_id = specification["parent_trial"]
                parent = next((t for t in self.state["trials"] if t["id"] == parent_id), None)
                if parent is None or parent["status"] in {"planned", "training"}:
                    continue
                if (parent["status"] == "failed"
                        or parent["status"].startswith("cancelled")
                        or not (Path(parent["run_dir"]) / "best.pt").is_file()):
                    self.state["trials"].append({
                        **specification, "kind": "trained", "status": "cancelled_dependency",
                        "origin": "initial_trial", "options": {}, "rollouts": {},
                        "run_dir": str(self.root / "trials" / trial_id),
                        "failure": {"reason": f"Required parent {parent_id} failed or has no checkpoint"},
                    })
                    known.add(trial_id)
                    continue
            decision = {
                "trial_name": trial_id,
                "training_mode": specification["training_mode"],
                "parent_trial": specification.get("parent_trial", ""),
                "hypothesis": specification["hypothesis"],
                "change_family": specification["change_family"],
                "expected_observation": specification["expected_observation"],
                "overrides": specification.get("overrides", []),
            }
            trial = self.add_trial(decision)
            trial["origin"] = "initial_trial"
            known.add(trial_id)
        self.save()

    def reconcile_interrupted_trial(self) -> None:
        """Never append a fresh optimization to an ambiguous partial run."""
        for trial in self.state["trials"]:
            if trial.get("status") != "training":
                continue
            checkpoint = Path(trial["run_dir"]) / "best.pt"
            if checkpoint.is_file():
                trial["status"] = "trained"
            else:
                trial["status"] = "failed"
                trial["failure"] = (
                    "Controller was interrupted during training before best.pt was written; "
                    "the partial directory is retained and will not be appended to.")
        self.save()

    def trial(self, trial_id: str) -> dict[str, Any]:
        matches = [trial for trial in self.state["trials"] if trial["id"] == trial_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous trial: {trial_id}")
        return matches[0]

    def promotion_trial(self, decision: dict[str, Any]) -> dict[str, Any]:
        """Resolve a promotion target, preferring an exact existing trial_name."""
        trial_ids = {trial["id"] for trial in self.state["trials"]}
        named = decision.get("trial_name", "")
        parent = decision.get("parent_trial", "")
        target = named if named in trial_ids else parent
        trial = self.trial(target)
        if target != parent:
            decision["promotion_target_correction"] = {
                "supplied_parent_trial": parent,
                "resolved_trial": target,
            }
        return trial

    def command_env(self) -> dict[str, str]:
        env = dict(os.environ)
        environment_bin = Path(self.python).expanduser().resolve().parent
        env["PATH"] = os.pathsep.join(
            (str(environment_bin), env.get("PATH", ""))).rstrip(os.pathsep)
        ptxas = environment_bin / "ptxas"
        if ptxas.is_file() and os.access(ptxas, os.X_OK):
            # Triton otherwise may select an older system CUDA compiler even
            # though PyTorch itself comes from the requested environment.
            env["TRITON_PTXAS_PATH"] = str(ptxas)
        env.setdefault("MPLCONFIGDIR", str(self.root / ".mplconfig"))
        return env

    def evaluate(self, trial: dict[str, Any], tier: str) -> None:
        screen_first = self.study.get("screen_before_innovations", False)
        limits = self.study.get("screen_gate")
        if tier == "confirmation" and limits and not get_path(
                trial, "rollouts", "screen", "gate", "passed", default=False):
            raise ValueError("Confirmation requires a passing screen gate")
        if not screen_first:
            self.evaluate_innovations(trial)
        settings = self.study[tier]
        rollout_dir = Path(trial["run_dir"]) / f"rollout_{settings['horizon']}_{tier}"
        record = trial.setdefault("rollouts", {}).setdefault(tier, {"path": str(rollout_dir)})
        checkpoint = Path(trial["run_dir"]) / "best.pt"
        if not (rollout_dir / "rollout.npz").is_file():
            command = [self.python, "-m", "modelling.neural_sde", "evaluate",
                       "--checkpoint", str(checkpoint), "--split", "validation",
                       "--num-conditions", str(settings["conditions"]),
                       "--ensemble-size", str(settings["ensemble_size"]),
                       "--horizon", str(settings["horizon"]), "--seed", str(settings["seed"]),
                       "--output", str(rollout_dir)]
            for key, flag in (("device", "--device"), ("batch_size", "--batch-size")):
                if key in settings:
                    command += [flag, str(settings[key])]
            if self.dry_run:
                print("DRY RUN", json.dumps(command))
                return
            run_logged(command, Path(trial["run_dir"]) / "controller_logs" / f"{tier}_evaluate.log",
                       self.repo, self.command_env())
        if tier == "screen" and limits and not self.dry_run:
            record["health"] = rollout_health(rollout_dir / "rollout.npz", checkpoint)
            failures = screen_gate_failures(limits, record["health"])
            if failures:
                record["gate"] = {"passed": False, "reasons": failures, "limits": limits}
                # Failed trajectories can produce NaN/Inf dynamics metrics.
                # Keep the rejection serializable without importing those values.
                record["metrics"] = {
                    "configuration": configuration_summary(Path(trial["run_dir"])),
                    "training": training_metrics(Path(trial["run_dir"])),
                }
                trial["status"] = "screen_rejected"
                self.save()
                return
        if not (rollout_dir / "quantitative" / "summary.json").is_file():
            plot = [self.python, "-m", "data_analysis.plot_rollout", str(rollout_dir),
                    "--overwrite", "--nperseg", str(settings.get("nperseg", 1024)),
                    "--modal-psd", "--modal-psd-modes",
                    *[str(value) for value in settings.get("modal_psd_modes", [1, 2, 3, 4, 5])]]
            if "peak_nperseg" in settings:
                plot += ["--peak-nperseg", str(settings["peak_nperseg"])]
            if "peak_band" in settings:
                plot += ["--peak-band", *map(str, settings["peak_band"])]
            if "discard" in settings:
                plot += ["--discard", str(settings["discard"])]
            if self.dry_run:
                print("DRY RUN", json.dumps(plot))
                return
            run_logged(plot, Path(trial["run_dir"]) / "controller_logs" / f"{tier}_plot.log",
                       self.repo, self.command_env())
        record["metrics"] = extract_metrics(Path(trial["run_dir"]), rollout_dir)
        audit_settings = self.study.get("restoring_audit")
        if audit_settings:
            audit_path = rollout_dir / "restoring_audit.json"
            if not audit_path.is_file():
                command = [self.python, "-m",
                           "modelling.neural_sde.experiments.restoring_audit",
                           "--rollout", str(rollout_dir), "--checkpoint", str(checkpoint),
                           "--output", str(audit_path),
                           "--samples", str(audit_settings.get("samples", 512)),
                           "--seed", str(audit_settings.get("seed", 0))]
                if self.dry_run:
                    print("DRY RUN", json.dumps(command))
                    return
                run_logged(command, Path(trial["run_dir"]) / "controller_logs" / f"{tier}_restoring_audit.log",
                           self.repo, self.command_env())
            record["metrics"]["restoring_audit"] = read_json(audit_path)
        if tier == "screen" and limits:
            failures = screen_gate_failures(limits, record["health"], record["metrics"])
            record["gate"] = {"passed": not failures, "reasons": failures, "limits": limits}
            if failures:
                trial["status"] = "screen_rejected"
                self.save()
                return
        if screen_first:
            self.evaluate_innovations(trial)
        self.evaluate_objective_balance(trial)
        self.evaluate_score_sensitivity(trial)
        trial["status"] = "confirmed" if tier == "confirmation" else "screened"
        self.save()

    def evaluate_innovations(self, trial: dict[str, Any]) -> None:
        """Run the held-out one-step innovation diagnostic once per checkpoint."""
        settings = self.study.get("innovation")
        if not settings or trial.get("innovation", {}).get("metrics"):
            return
        run_dir = Path(trial["run_dir"])
        output = run_dir / settings.get("output_name", "validation_innovation_diagnostic")
        summary = output / "summary.json"
        if not summary.is_file():
            if not self.dry_run and output.is_dir() and any(output.iterdir()):
                index = 1
                while True:
                    archived = output.with_name(f"{output.name}_incomplete_{index}")
                    if not archived.exists():
                        output.rename(archived)
                        break
                    index += 1
            command = [
                self.python, "-m",
                "data_analysis.correlations.training_innovation_diagnostic",
                "--checkpoint", str(run_dir / "best.pt"),
                "--output", str(output), "--split", settings.get("split", "validation"),
                "--device", settings.get("device", "cuda"),
                "--batch-size", str(settings.get("batch_size", 8192)),
                "--max-lag", str(settings.get("max_lag", 512)),
                "--threshold", str(settings.get("threshold", 0.1)),
                "--consecutive", str(settings.get("consecutive", 100)),
                "--min-pairs", str(settings.get("min_pairs", 1000)),
                "--nperseg", str(settings.get("nperseg", 1024)),
            ]
            if self.dry_run:
                print("DRY RUN", json.dumps(command))
                return
            run_logged(command, run_dir / "controller_logs" / "innovation.log",
                       self.repo, self.command_env())
        trial["innovation"] = {
            "path": str(output), "metrics": extract_innovation_metrics(summary)}
        self.save()

    def evaluate_objective_balance(self, trial: dict[str, Any]) -> None:
        """Compare the checkpoint's one-step, multistep, and moment gradients."""
        settings = self.study.get("objective_balance_diagnostic")
        if not settings:
            return
        run_dir = Path(trial["run_dir"])
        output = run_dir / settings.get("output_name", "objective_balance_diagnostic")
        summary = output / "summary.json"
        requested_history = int(settings.get("min_history", 0))
        if self.objective_balance_current(trial):
            trial["objective_balance"] = {
                "path": str(output),
                "metrics": extract_objective_balance_metrics(summary),
            }
            self.save()
            return
        trial.pop("objective_balance", None)
        if summary.is_file() or (output.is_dir() and any(output.iterdir())):
            if not self.dry_run:
                self.archive_diagnostic_output(output)
                self.save()
        if not summary.is_file() or self.dry_run:
            command = [
                self.python, "-m", "modelling.neural_sde.objective_balance_diagnostic",
                "--checkpoint", str(run_dir / "best.pt"), "--output", str(output),
                "--split", settings.get("split", "validation"),
                "--device", settings.get("device", "cuda"),
                "--windows", str(settings.get("windows", 64)),
                "--moment-trajectories", str(settings.get("moment_trajectories", 32)),
                "--window-seed", str(settings.get("window_seed", 0)),
                "--seeds", *[str(value) for value in settings.get("seeds", [0, 1])],
            ]
            if settings.get("window_seeds"):
                command += ["--window-seeds", *map(str, settings["window_seeds"])]
            command += ["--min-history", str(settings.get("min_history", 0))]
            if settings.get("moment_horizons"):
                command += ["--moment-horizons",
                            *[str(value) for value in settings["moment_horizons"]]]
            for key in ("multistep_horizons", "multistep_horizon_weights",
                        "multistep_particle_counts", "multistep_gradient_groups"):
                if settings.get(key) is not None:
                    command += ["--" + key.replace("_", "-"),
                                *[str(value) for value in settings[key]]]
            for key in ("multistep_objective", "multistep_target", "multistep_position_weight",
                        "multistep_velocity_weight", "multistep_weight",
                        "multistep_fraction", "mean_weight", "variance_weight", "moment_fraction"):
                if settings.get(key) is not None:
                    command += ["--" + key.replace("_", "-"), str(settings[key])]
            if self.dry_run:
                print("DRY RUN", json.dumps(command))
                return
            run_logged(command, run_dir / "controller_logs" / "objective_balance.log",
                       self.repo, self.command_env(),
                       timeout_seconds=settings.get("timeout_seconds"))
        trial["objective_balance"] = {
            "path": str(output),
            "metrics": extract_objective_balance_metrics(summary),
        }
        if not diagnostic_history_valid(summary, requested_history):
            raise RuntimeError("Regenerated objective-balance report failed history eligibility")
        self.save()

    def evaluate_score_sensitivity(self, trial):
        settings = self.study.get("energy_score_sensitivity")
        if not settings:
            return
        output = Path(trial["run_dir"]) / settings.get("output_name", "energy_score_sensitivity")
        summary = output / "summary.json"
        requested_history = int(settings.get("min_history", 0))
        if self.score_sensitivity_current(trial):
            report = read_json(summary)
            trial["score_sensitivity"] = {"path": str(output), "metrics": {
                key: value for key, value in report.items()
                if key not in {"draws", "selected_window_identifiers_by_batch"}}}
            trial["score_sensitivity"]["metrics"]["selected_window_sha256_by_batch"] = {
                seed: record["sha256"]
                for seed, record in report["selected_window_identifiers_by_batch"].items()}
            self.save()
            return
        trial.pop("score_sensitivity", None)
        if summary.is_file() or (output.is_dir() and any(output.iterdir())):
            if not self.dry_run:
                self.archive_diagnostic_output(output)
                self.save()
        if not summary.is_file() or self.dry_run:
            command = [self.python, "-m", "modelling.neural_sde.energy_score_sensitivity",
                       "--checkpoint", str(Path(trial["run_dir"]) / "best.pt"),
                       "--output", str(output)]
            for key in ("device", "windows", "particles", "min_history"):
                if key in settings:
                    command += ["--" + key.replace("_", "-"), str(settings[key])]
            for key in ("window_seeds", "seeds", "horizons"):
                if key in settings:
                    command += ["--" + key.replace("_", "-"), *map(str, settings[key])]
            if self.dry_run:
                print("DRY RUN", json.dumps(command))
                return
            run_logged(command, Path(trial["run_dir"]) / "controller_logs" / "score_sensitivity.log",
                       self.repo, self.command_env(), timeout_seconds=settings.get("timeout_seconds"))
        report = read_json(summary)
        trial["score_sensitivity"] = {"path": str(output), "metrics": {
            key: value for key, value in report.items()
            if key not in {"draws", "selected_window_identifiers_by_batch"}}}
        trial["score_sensitivity"]["metrics"]["selected_window_sha256_by_batch"] = {
            seed: record["sha256"]
            for seed, record in report["selected_window_identifiers_by_batch"].items()}
        if not diagnostic_history_valid(summary, requested_history):
            raise RuntimeError("Regenerated score-sensitivity report failed history eligibility")
        self.save()

    def validate_calibration_stage(self, decision, options):
        """Opt-in campaign guard: paired continuations, stable parents, no fresh moments."""
        if not self.study.get("staged_tcn_calibration"):
            return
        role = decision["change_family"]
        existing = [t for t in self.state["trials"] if t["kind"] == "trained"]
        if role == "short_history_baseline":
            if existing or decision["training_mode"] != "fresh":
                raise ValueError("Only the first baseline can be fresh")
            if any(float(options[key][0]) != 0 for key in
                   ("--stationary-mean-weight", "--stationary-variance-weight")):
                raise ValueError("Fresh baseline cannot use moments")
            return
        if decision["training_mode"] != "finetune":
            raise ValueError("Calibration stages must fine-tune")
        if any(t.get("change_family") == role for t in existing):
            raise ValueError("Each calibration stage is allowed once")
        parent = self.trial(decision["parent_trial"])
        if parent.get("status") not in {"screened", "confirmed"} or parent.get("kind") != "trained":
            raise ValueError("Calibration requires a successfully screened short-history parent")
        if not parent.get("objective_balance", {}).get("metrics") or not parent.get("score_sensitivity", {}).get("metrics"):
            raise ValueError("Parent needs gradient and score-sensitivity evidence")
        if options["--learning-rate"] != ["1e-05"] or options["--epochs"] != ["1"]:
            raise ValueError("Continuations require one epoch at lr 1e-5")
        by_role = {t.get("change_family"): t for t in existing}
        if role in {"es_control", "es_calibration"}:
            if parent.get("change_family") != "short_history_baseline":
                raise ValueError("ES pair must start from the same fresh baseline")
            if any(float(options[key][0]) != 0 for key in
                   ("--stationary-mean-weight", "--stationary-variance-weight")):
                raise ValueError("ES calibration cannot introduce moments")
            if role == "es_control":
                if options["--multistep-weight"] != ["0.03"] or options["--multistep-position-weight"] != ["2"]:
                    raise ValueError("ES control must retain baseline score")
            else:
                control = by_role.get("es_control")
                if control is None or control["status"] not in {"screened", "confirmed"}:
                    raise ValueError("Complete ES control before the alternative")
                keys = ("--multistep-weight", "--multistep-position-weight")
                if sum(options[k] != control["options"][k] for k in keys) != 1:
                    raise ValueError("ES alternative must change exactly one score coefficient")
        elif role in {"moment_control", "moment_refinement"}:
            if parent.get("change_family") not in {"short_history_baseline", "es_control", "es_calibration"}:
                raise ValueError("Moment pair needs a stable pre-moment parent")
            for key in ("--multistep-weight", "--multistep-position-weight"):
                if options[key] != parent["options"][key]:
                    raise ValueError("Moment stage must retain selected parent's score")
            if role == "moment_control":
                if any(float(options[key][0]) != 0 for key in
                       ("--stationary-mean-weight", "--stationary-variance-weight")):
                    raise ValueError("Moment continuation control must have zero weights")
            else:
                control = by_role.get("moment_control")
                if control is None or control["status"] not in {"screened", "confirmed"} or control["parent_trial"] != parent["id"]:
                    raise ValueError("Moment refinement needs completed same-parent continuation control")
                if any(options[key] != ["0.001"] for key in
                       ("--stationary-mean-weight", "--stationary-variance-weight")):
                    raise ValueError("First moment refinement starts at 0.001 for both terms")
        else:
            raise ValueError("Unknown staged calibration family")

    def preflight_moments(self, trial):
        if not self.study.get("staged_tcn_calibration") or trial.get("moment_preflight"):
            return
        options = trial["options"]
        if not any(float(options[key][0]) > 0 for key in
                   ("--stationary-mean-weight", "--stationary-variance-weight")):
            return
        parent = self.trial(trial["parent_trial"])
        output = Path(trial["run_dir"]) / "moment_preflight"
        command = [self.python, "-m", "modelling.neural_sde.objective_balance_diagnostic",
                   "--checkpoint", str(Path(parent["run_dir"]) / "best.pt"),
                   "--output", str(output), "--split", "train", "--device", "cuda",
                   "--windows", "256", "--moment-trajectories", "64",
                   "--window-seeds", "0", "1", "2", "--seeds", "0", "1", "2",
                   "--min-history", "15", "--moment-horizons",
                   *options["--stationary-mean-horizons"],
                   "--mean-weight", *options["--stationary-mean-weight"],
                   "--variance-weight", *options["--stationary-variance-weight"],
                   "--moment-fraction", *options["--stationary-mean-batch-fraction"]]
        if "--stationary-variance-pod-mode-count" in options:
            command += ["--variance-pod-mode-count",
                        *options["--stationary-variance-pod-mode-count"]]
        if self.dry_run:
            print("DRY RUN", json.dumps(command))
            return
        run_logged(command, Path(trial["run_dir"]) / "controller_logs" / "moment_preflight.log",
                   self.repo, self.command_env(), timeout_seconds=1800)
        trial["moment_preflight"] = {"path": str(output),
            "metrics": extract_objective_balance_metrics(output / "summary.json")}
        self.save()

    def add_trial(self, decision: dict[str, Any]) -> dict[str, Any]:
        trial_id = decision["trial_name"]
        if not SLUG.fullmatch(trial_id) or any(t["id"] == trial_id for t in self.state["trials"]):
            raise ValueError("Advisor trial_name must be a new lowercase slug.")
        max_changes = int(self.study.get("max_overrides_per_trial", 4))
        options = apply_overrides(self.base, decision["overrides"], self.allowed, max_changes)
        if options.get("--multistep-weight") == ["0"]:
            for flag in (
                    "--multistep-objective", "--multistep-target", "--multistep-horizons",
                    "--multistep-mode-counts", "--multistep-particle-counts",
                    "--multistep-horizon-weights", "--multistep-position-weight",
                    "--multistep-velocity-weight",
                    "--multistep-window-fraction", "--multistep-window-sampling",
                    "--multistep-validation-seed", "--multistep-gradient-groups"):
                options.pop(flag, None)
        validate_semantics(options)
        self.validate_calibration_stage(decision, options)
        mode = decision["training_mode"]
        allowed_modes = self.study.get("allowed_training_modes")
        if allowed_modes is not None and mode not in allowed_modes:
            raise ValueError(f"Study permits only training modes {allowed_modes}")
        parent_id = decision["parent_trial"]
        comparison_parent = ""
        if mode == "finetune":
            parent = self.trial(parent_id)
            if not (Path(parent["run_dir"]) / "best.pt").is_file():
                raise ValueError("Fine-tuning parent has no best.pt checkpoint.")
            changed = {item["flag"] for item in decision["overrides"]}
            forbidden = changed & NEW_MODEL_ONLY
            if forbidden:
                raise ValueError(f"Fine-tuning cannot change architecture/data flags: {sorted(forbidden)}")
            options = {flag: value for flag, value in options.items() if flag not in NEW_MODEL_ONLY}
        elif mode == "memory":
            parent = self.trial(parent_id)
            parent_checkpoint = Path(parent["run_dir"]) / "best.pt"
            if not parent_checkpoint.is_file():
                raise ValueError("Memory parent has no best.pt checkpoint.")
            parent_config = configuration_summary(Path(parent["run_dir"]))
            if (get_path(parent_config, "model", "velocity_memory_offsets")
                    or get_path(parent_config, "model", "velocity_oscillator_periods")
                    or get_path(parent_config, "model", "velocity_innovation_offsets")):
                raise ValueError("Memory training requires a memory-free parent checkpoint.")
            changed = {item["flag"] for item in decision["overrides"]}
            selectors = {"--velocity-memory-offsets", "--velocity-oscillator-periods",
                         "--velocity-innovation-offsets"}
            if len(changed & selectors) != 1:
                raise ValueError(
                    "Memory training requires exactly one memory-family selector.")
            memory_flags = {"--velocity-memory-offsets", "--velocity-memory-rank",
                            "--velocity-memory-norm-cap", "--velocity-oscillator-periods",
                            "--velocity-oscillator-burnin",
                            "--velocity-oscillator-radius-init",
                            "--velocity-oscillator-norm-cap",
                            "--velocity-innovation-offsets",
                            "--velocity-innovation-rank",
                            "--velocity-innovation-norm-cap"}
            forbidden = (changed & NEW_MODEL_ONLY) - memory_flags
            if forbidden:
                raise ValueError(
                    "Memory training cannot also change parent architecture/data flags: "
                    f"{sorted(forbidden)}")
            options = {
                flag: value for flag, value in options.items()
                if flag not in NEW_MODEL_ONLY or flag in memory_flags
            }
            options["--train-mode"] = ["memory"]
        elif mode == "fresh":
            # Advisors often name the matched control in parent_trial. Keep it
            # as provenance, while ensuring a fresh run loads no checkpoint.
            comparison_parent, parent_id = parent_id, ""
            if any(flag in options for flag in (
                    "--velocity-memory-offsets", "--velocity-memory-rank",
                    "--velocity-memory-norm-cap", "--velocity-oscillator-periods",
                    "--velocity-oscillator-burnin",
                    "--velocity-oscillator-radius-init",
                    "--velocity-oscillator-norm-cap",
                    "--velocity-innovation-offsets",
                    "--velocity-innovation-rank",
                    "--velocity-innovation-norm-cap")):
                raise ValueError("Velocity-memory options require memory training mode.")
        else:
            raise ValueError("training_mode must be fresh, finetune, or memory.")
        run_dir = self.root / "trials" / trial_id
        trial = {
            "id": trial_id, "kind": "trained", "status": "planned",
            "hypothesis": decision["hypothesis"], "change_family": decision["change_family"],
            "expected_observation": decision["expected_observation"],
            "training_mode": mode, "parent_trial": parent_id,
            "comparison_parent": comparison_parent,
            "options": options, "run_dir": str(run_dir), "rollouts": {},
        }
        self.state["trials"].append(trial)
        decision["controller_status"] = "applied"
        self.save()
        return trial

    def train(self, trial: dict[str, Any]) -> None:
        self.preflight_moments(trial)
        run_dir = Path(trial["run_dir"])
        command = [self.python, "-m", "modelling.neural_sde", "train"]
        if trial["training_mode"] in {"finetune", "memory"}:
            parent = self.trial(trial["parent_trial"])
            command += ["--checkpoint", str(Path(parent["run_dir"]) / "best.pt")]
        command += render_options(trial["options"])
        command += ["--output", str(run_dir)]
        trial["train_command"] = command
        if self.dry_run:
            print("DRY RUN", json.dumps(command))
            return
        trial["status"] = "training"
        self.save()
        run_logged(command, run_dir / "controller_logs" / "train.log",
                   self.repo, self.command_env(),
                   timeout_seconds=self.study.get("training_timeout_seconds"))
        trial["status"] = "trained"
        self.save()

    def compact_context(self) -> dict[str, Any]:
        return {
            "scientific_goal": self.study["scientific_goal"],
            "constraints": self.study["scientific_constraints"],
            "budget": self.study["budget"],
            "screen_gate": self.study.get("screen_gate"),
            "screen_protocol": self.study.get("screen"),
            "confirmation_protocol": self.study.get("confirmation"),
            "allowed_overrides": self.allowed,
            "trials": [{key: trial.get(key) for key in (
                "id", "kind", "status", "hypothesis", "change_family",
                "expected_observation", "training_mode", "parent_trial",
                "comparison_parent", "options", "rollouts", "innovation",
                "objective_balance", "particle_noise_probe", "baseline_failure",
                "score_sensitivity", "moment_preflight")}
                for trial in self.state["trials"]],
        }

    def advisor(self) -> dict[str, Any]:
        # Scientific interpretation can be refined while a long trial is running.
        # Refresh only advisor-facing text; execution settings stay fixed for the
        # lifetime of this controller process.
        latest_study = read_json(self.study_path)
        for key in ("scientific_goal", "scientific_constraints", "advisor_guidance"):
            if key in latest_study:
                self.study[key] = latest_study[key]
        context_path = self.root / "advisor_context.json"
        write_json(context_path, self.compact_context())
        if self.manual or self.study.get("advisor", {}).get("mode") == "manual":
            request = self.root / "decision_request.md"
            request.write_text(self.advisor_prompt(context_path))
            raise ManualDecisionRequired(f"Manual decision requested; review {request}")
        schema = self.repo / "modelling/neural_sde/experiments/adaptive_advisor_schema.json"
        decision_path = self.root / "advisor_decision.json"
        prompt = self.advisor_prompt(context_path)
        advisor_executable = resolve_executable(
            self.study.get("advisor", {}).get("command", "codex"))
        command = [advisor_executable,
                   "--ask-for-approval", "never", "exec",
                   "--ephemeral", "--sandbox", "read-only",
                   "-C", str(self.repo), "--output-schema", str(schema),
                   "--output-last-message", str(decision_path)]
        model = self.study.get("advisor", {}).get("model")
        if model:
            command += ["--model", model]
        images = self.latest_images()
        for image in images:
            command += ["--image", str(image)]
        command += ["-"]
        if self.dry_run:
            print("DRY RUN", json.dumps(command))
            print(prompt)
            return {"action": "stop", "reason": "dry run", "trial_name": "",
                    "parent_trial": "", "training_mode": "fresh", "hypothesis": "",
                    "change_family": "dry_run", "overrides": [],
                    "expected_observation": "", "stop_reason": "dry run",
                    "implementation_request": ""}
        completed = subprocess.run(command, cwd=self.repo, input=prompt, text=True,
                                   capture_output=True, env=self.command_env())
        (self.root / "advisor.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        decision = read_json(decision_path)
        decision["controller_status"] = "pending"
        self.state["decisions"].append(decision)
        self.save()
        return decision

    def latest_images(self) -> list[Path]:
        images: list[Path] = []
        for trial in reversed(self.state["trials"]):
            for tier in ("confirmation", "screen", "seed"):
                record = trial.get("rollouts", {}).get(tier)
                if not record:
                    continue
                root = Path(record["path"])
                for relative in ("psd_peaks/psd_peaks.png", "psd/psd.png", "modal/modal_diagnostics.png",
                                 "modal_psd/modal_psd.png", "coordinates/trajectories.png"):
                    path = root / relative
                    if path.is_file():
                        images.append(path)
                return images[:4]
        return images

    def advisor_prompt(self, context_path: Path) -> str:
        extra = self.study.get("advisor_guidance", "")
        experiment = self.study.get("base_train_options", {}).get(
            "--experiment", "the declared")
        return f"""You are the scientific advisor for a bounded Neural-SDE study of the {experiment}
capillary-wave case. Read {context_path}. The controller calls you only after a completed
run, so choose exactly one informative next action. Tune on validation only; test data are
locked. Compare energy/amplitude, POD mean and covariance, PSD shape (especially the
mid-frequency region), stability, one-step NLL, and held-out innovation calibration and ACF
separately. A low one-step NLL is not
evidence of a good rollout. Multistep losses can be particle-noisy or overdamp long horizons;
endpoint moment losses directly constrain marginal amplitude but do not identify conditional
dynamics or spectral shape. Follow the declared scientific constraints when deciding whether
either objective family is allowed in this study.

Use a matched ablation and state a falsifiable hypothesis. Prefer one conceptual change
family per trial. If the evidence indicates structural mismatch, use `request_implementation`
to write a precise, testable code-change request instead of pretending an unsupported flag exists.
`run` trains and screens a candidate, `promote` runs the expensive confirmation rollout for
an existing trial, and `stop` ends the study. `memory` training adds only the configured
finite-memory, stable oscillator-memory, or autoregressive innovation-memory parameters to
parent_trial and freezes the parent checkpoint. Innovation memory predicts the next standardized
parent residual from sampled past residuals with stable low-rank kernels, so it represents colored
stochastic forcing rather than another raw-state drift input.
Every override must exactly match an allowed alternative in the context. Fresh trials use
the fixed base configuration plus overrides; parent_trial may name their matched comparison
or be empty, but no checkpoint will be loaded. Fine-tuning may change training-objective
options but cannot change data or architecture. For `promote`, set both trial_name and
parent_trial to the exact id of the existing screened trial to confirm. Return only JSON
matching the schema. Set implementation_request to an empty string unless the action is
request_implementation.

Study-specific guidance:
{extra}"""

    def counts(self) -> tuple[int, int]:
        trained = sum(trial["kind"] == "trained"
                      and not trial.get("status", "").startswith("cancelled")
                      for trial in self.state["trials"])
        confirmations = sum("confirmation" in trial.get("rollouts", {})
                            for trial in self.state["trials"])
        return trained, confirmations

    @staticmethod
    def failure_tail(trial: dict[str, Any], limit=4000) -> str:
        root = Path(trial["run_dir"]) / "controller_logs"
        logs = sorted(root.glob("*.log"), key=lambda path: path.stat().st_mtime)
        if not logs:
            return ""
        text = logs[-1].read_text(errors="replace")
        return text[-limit:]

    def run(self) -> None:
        self.reconcile_interrupted_trial()
        while self.state["status"] == "active":
            if self.study.get("defer_initial_trial_dependencies", False):
                self._import_initial_trials()
            # Opt-in matched baselines on this study's grid/seeds before training.
            seed = next((t for t in self.state["trials"]
                         if t["kind"] == "seed" and "screen" not in t.get("rollouts", {})
                         and not t.get("baseline_failure")), None) if self.study.get("screen_seed_runs") else None
            if seed:
                try:
                    self.evaluate(seed, "screen")
                except subprocess.CalledProcessError as error:
                    seed["baseline_failure"] = {"returncode": error.returncode,
                                                "log_tail": self.failure_tail(seed)}
                    self.save()
                if self.dry_run:
                    return
                continue
            probe = self.study.get("objective_balance_particle_probe")
            if probe:
                trial = self.trial(probe["trial"])
                if not trial.get("particle_noise_probe"):
                    original = self.study["objective_balance_diagnostic"]
                    balance = trial.get("objective_balance")
                    self.study["objective_balance_diagnostic"] = {
                        **original, "output_name": probe["output_name"],
                        "multistep_particle_counts": probe["particle_counts"]}
                    try:
                        self.evaluate_objective_balance(trial)
                        trial["particle_noise_probe"] = trial.get("objective_balance")
                    except subprocess.CalledProcessError as error:
                        trial["particle_noise_probe"] = {"failure": {
                            "returncode": error.returncode, "log_tail": self.failure_tail(trial)}}
                    finally:
                        self.study["objective_balance_diagnostic"] = original
                        if balance is None:
                            trial.pop("objective_balance", None)
                        else:
                            trial["objective_balance"] = balance
                        self.save()
                    if self.dry_run:
                        return
                    continue
            pending = next((trial for trial in self.state["trials"]
                            if trial["kind"] == "trained" and trial["status"] in {
                                "planned", "training", "trained"}), None)
            if pending:
                try:
                    if pending["status"] in {"planned", "training"}:
                        self.train(pending)
                        if self.dry_run:
                            return
                    self.evaluate(pending, "screen")
                except subprocess.CalledProcessError as error:
                    pending["status"] = "failed"
                    pending["failure"] = {
                        "returncode": error.returncode,
                        "command": error.cmd,
                        "log_tail": self.failure_tail(pending),
                    }
                    self.save()
                if self.dry_run:
                    return
                continue
            missing_innovation = (next((
                trial for trial in self.state["trials"]
                if trial.get("status") in {"screened", "confirmed"}
                and not trial.get("innovation")
                and (Path(trial.get("run_dir", "")) / "best.pt").is_file()
            ), None) if self.study.get("innovation") else None)
            if missing_innovation:
                try:
                    self.evaluate_innovations(missing_innovation)
                except subprocess.CalledProcessError as error:
                    missing_innovation["innovation"] = {
                        "failure": {
                            "returncode": error.returncode,
                            "command": error.cmd,
                            "log_tail": self.failure_tail(missing_innovation),
                        }
                    }
                    self.save()
                if self.dry_run:
                    return
                continue
            missing_balance = (next((
                trial for trial in self.state["trials"]
                if trial.get("status") in {"screened", "confirmed"}
                and not self.objective_balance_current(trial)
                and not get_path(trial, "objective_balance", "failure")
                and (Path(trial.get("run_dir", "")) / "best.pt").is_file()
            ), None) if self.study.get("objective_balance_diagnostic") else None)
            if missing_balance:
                try:
                    self.evaluate_objective_balance(missing_balance)
                except subprocess.CalledProcessError as error:
                    missing_balance["objective_balance"] = {
                        "failure": {
                            "returncode": error.returncode,
                            "command": error.cmd,
                            "log_tail": self.failure_tail(missing_balance),
                        }
                    }
                    self.save()
                if self.dry_run:
                    return
                continue
            missing_sensitivity = (next((t for t in self.state["trials"]
                if t.get("status") in {"screened", "confirmed"}
                and not self.score_sensitivity_current(t)
                and (Path(t.get("run_dir", "")) / "best.pt").is_file()), None)
                if self.study.get("energy_score_sensitivity") else None)
            if missing_sensitivity:
                try:
                    self.evaluate_score_sensitivity(missing_sensitivity)
                except subprocess.CalledProcessError as error:
                    missing_sensitivity["score_sensitivity"] = {"failure": {
                        "returncode": error.returncode, "log_tail": self.failure_tail(missing_sensitivity)}}
                    self.save()
                if self.dry_run:
                    return
                continue
            trained, confirmations = self.counts()
            decision = next((item for item in self.state["decisions"]
                             if item.get("controller_status", "pending") == "pending"), None)
            if decision is None:
                try:
                    decision = self.advisor()
                except subprocess.CalledProcessError as error:
                    self.state.update(
                        status="advisor_failed",
                        advisor_failure={
                            "returncode": error.returncode,
                            "command": error.cmd,
                            "log": str(self.root / "advisor.log"),
                        })
                    self.save()
                    print(f"Advisor failed; see {self.root / 'advisor.log'}", flush=True)
                    return
            if self.dry_run:
                return
            action = decision["action"]
            if action == "stop":
                decision["controller_status"] = "applied"
                self.state.update(status="stopped", stop_reason=decision["stop_reason"])
                self.save()
                return
            if action == "request_implementation":
                request = {
                    key: decision.get(key) for key in (
                        "trial_name", "parent_trial", "hypothesis", "change_family",
                        "expected_observation", "reason", "implementation_request")
                }
                write_json(self.root / "implementation_request.json", request)
                (self.root / "implementation_request.md").write_text(
                    "# Controller implementation request\n\n"
                    + decision["implementation_request"].strip() + "\n")
                decision["controller_status"] = "applied"
                self.state.update(
                    status="implementation_required",
                    stop_reason="The advisor requested a new reviewed model implementation.")
                self.save()
                return
            if action == "promote":
                if confirmations >= self.study["budget"]["max_confirmations"]:
                    raise ValueError("Advisor requested promotion after confirmation budget was exhausted.")
                trial = self.promotion_trial(decision)
                if trial.get("status") not in {"screened", "confirmed"}:
                    raise ValueError("Only a successfully screened trial can be promoted.")
                if "confirmation" in trial.get("rollouts", {}):
                    raise ValueError("Trial is already confirmed.")
                try:
                    self.evaluate(trial, "confirmation")
                except subprocess.CalledProcessError as error:
                    trial["confirmation_failure"] = {
                        "returncode": error.returncode,
                        "command": error.cmd,
                        "log_tail": self.failure_tail(trial),
                    }
                    self.save()
                decision["controller_status"] = "applied"
                self.save()
            elif action == "run":
                if trained >= self.study["budget"]["max_training_trials"]:
                    raise ValueError(
                        "Advisor requested training after the training budget was exhausted; "
                        "promote an existing trial or stop.")
                self.add_trial(decision)
            else:
                raise ValueError(f"Unknown advisor action: {action}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the next command without running it.")
    parser.add_argument("--manual", action="store_true",
                        help="Write the advisor packet and stop instead of invoking Codex.")
    args = parser.parse_args(argv)
    if read_json(args.study).get("kind") == "fixed_memory":
        from .fixed_memory_study import run_study
        run_study(args.study, dry_run=args.dry_run)
        return
    controller = Controller(args.study, dry_run=args.dry_run, manual=args.manual)
    try:
        controller.run()
    except ManualDecisionRequired as error:
        print(error)
    print(controller.state_path)


if __name__ == "__main__":
    main()
