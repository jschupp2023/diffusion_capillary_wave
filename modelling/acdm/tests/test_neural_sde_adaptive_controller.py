import json
import subprocess
import sys
import pytest

from modelling.neural_sde.experiments.adaptive_controller import (
    Controller,
    apply_overrides,
    extract_metrics,
    extract_innovation_metrics,
    normalized_options,
    render_options,
    resolve_executable,
    validate_semantics,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_options_are_rendered_without_shell_interpolation():
    options = normalized_options({
        "--experiment": "0p10", "--history-offsets": [1, 2],
        "--no-spatial-mean": True, "--unused": None,
    })
    assert render_options(options) == [
        "--experiment", "0p10", "--history-offsets", "1", "2",
        "--no-spatial-mean",
    ]


def test_matched_seed_screen_precedes_planned_training(tmp_path, monkeypatch):
    study = tmp_path / "study.json"
    _write(study, {"repository": str(tmp_path), "output_root": "out",
                   "base_train_options": {}, "allowed_overrides": {},
                   "screen_seed_runs": True, "seed_runs": []})
    controller = Controller(study)
    controller.state["trials"] = [
        {"id": "parent", "kind": "seed", "status": "screened", "rollouts": {}},
        {"id": "child", "kind": "trained", "status": "planned"}]
    seen = []
    def evaluate(trial, tier):
        seen.append((trial["id"], tier))
        controller.state["status"] = "stopped"
    monkeypatch.setattr(controller, "evaluate", evaluate)
    controller.run()
    assert seen == [("parent", "screen")]
    assert controller.state["trials"][1]["status"] == "planned"


def test_particle_probe_preserves_primary_audit_and_uses_matched_settings(tmp_path, monkeypatch):
    study = tmp_path / "study.json"
    fixed = {"output_name": "m8", "windows": 16, "window_seeds": [1, 2],
             "multistep_particle_counts": [8, 8, 8]}
    _write(study, {"repository": str(tmp_path), "output_root": "out",
                   "base_train_options": {}, "allowed_overrides": {}, "seed_runs": [],
                   "objective_balance_diagnostic": fixed,
                   "objective_balance_particle_probe": {
                       "trial": "parent", "output_name": "m16", "particle_counts": [16, 16, 16]}})
    controller = Controller(study)
    trial = {"id": "parent", "kind": "seed", "status": "screened",
             "objective_balance": {"path": "m8", "metrics": {"value": 1}}}
    controller.state["trials"] = [trial]
    def diagnose(record):
        settings = controller.study["objective_balance_diagnostic"]
        assert settings == {**fixed, "output_name": "m16", "multistep_particle_counts": [16, 16, 16]}
        record["objective_balance"] = {"path": "m16", "metrics": {"value": 2}}
        controller.state["status"] = "stopped"
    monkeypatch.setattr(controller, "evaluate_objective_balance", diagnose)
    controller.run()
    assert trial["objective_balance"]["path"] == "m8"
    assert trial["particle_noise_probe"]["path"] == "m16"
    assert controller.study["objective_balance_diagnostic"] == fixed


def test_peak_metrics_reach_advisor_context(tmp_path):
    _write(tmp_path / "quantitative" / "summary.json", {
        "spectral_peaks": {"metrics": {"peak_contrast_error_db": 3.8}}})
    metrics = extract_metrics(tmp_path, tmp_path)
    assert metrics["spectral_peaks"]["metrics"]["peak_contrast_error_db"] == 3.8


def test_resolve_explicit_executable(tmp_path):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(executable.stat().st_mode | 0o100)
    assert resolve_executable(str(executable)) == str(executable.resolve())
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_executable(str(tmp_path / "missing"))


def test_advisor_overrides_are_bounded_by_declared_space():
    base = {"--train-trim-percent": ["5"], "--history-offsets": ["1"]}
    allowed = {"--train-trim-percent": [[0], [5]], "--history-offsets": [[1], [1, 2]]}
    changed = apply_overrides(base, [{
        "flag": "--history-offsets", "tokens": ["1", "2"], "remove": False,
    }], allowed, 1)
    assert changed["--history-offsets"] == ["1", "2"]
    with pytest.raises(ValueError, match="outside the allowed"):
        apply_overrides(base, [{
            "flag": "--train-trim-percent", "tokens": ["2.5"], "remove": False,
        }], allowed, 1)


def test_override_limit_counts_effective_changes_not_repeated_base_values():
    base = {
        "--multistep-weight": ["0.1"],
        "--learning-rate": ["0.0001"],
        "--epochs": ["5"],
    }
    allowed = {
        "--multistep-weight": [[0], [0.1]],
        "--learning-rate": [[0.0001], [0.0003]],
        "--epochs": [[3], [5]],
    }
    result = apply_overrides(base, [
        {"flag": "--multistep-weight", "tokens": ["0"], "remove": False},
        {"flag": "--learning-rate", "tokens": ["0.0001"], "remove": False},
        {"flag": "--epochs", "tokens": ["5"], "remove": False},
    ], allowed, 1)
    assert result["--multistep-weight"] == ["0"]

    with pytest.raises(ValueError, match="effective changes"):
        apply_overrides(base, [
            {"flag": "--multistep-weight", "tokens": ["0"], "remove": False},
            {"flag": "--learning-rate", "tokens": ["0.0003"], "remove": False},
        ], allowed, 1)


def test_semantic_validation_catches_coupled_schedules():
    with pytest.raises(ValueError, match="one value per"):
        validate_semantics({
            "--multistep-horizons": ["2", "5"],
            "--multistep-mode-counts": ["20"],
        })
    validate_semantics({
        "--velocity-innovation-offsets": ["1", "2", "5"],
        "--velocity-innovation-rank": ["4"],
        "--velocity-innovation-norm-cap": ["0.95"],
    })
    with pytest.raises(ValueError, match="require --velocity-innovation-rank"):
        validate_semantics({"--velocity-innovation-offsets": ["1", "2", "5"]})


def test_fresh_trial_keeps_advisor_parent_as_comparison_only(tmp_path):
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path),
        "output_root": "output",
        "base_train_options": {"--history-offsets": [1]},
        "allowed_overrides": {"--history-offsets": [[1], [1, 2]]},
        "seed_runs": [],
    })
    controller = Controller(study)
    decision = {
        "trial_name": "history_ablation", "training_mode": "fresh",
        "parent_trial": "matched_control", "hypothesis": "history helps",
        "change_family": "history", "expected_observation": "better PSD",
        "overrides": [{"flag": "--history-offsets", "tokens": ["1", "2"],
                       "remove": False}],
    }
    controller.state["decisions"].append(decision)
    trial = controller.add_trial(decision)
    assert trial["parent_trial"] == ""
    assert trial["comparison_parent"] == "matched_control"
    assert decision["controller_status"] == "applied"


def test_controller_environment_uses_selected_python_toolchain(tmp_path):
    environment = tmp_path / "env" / "bin"
    environment.mkdir(parents=True)
    python = environment / "python"
    ptxas = environment / "ptxas"
    python.write_text("")
    ptxas.write_text("")
    ptxas.chmod(ptxas.stat().st_mode | 0o100)
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "python": str(python),
        "output_root": "output", "base_train_options": {},
        "allowed_overrides": {}, "seed_runs": [],
    })
    env = Controller(study).command_env()
    assert env["PATH"].split(":")[0] == str(environment.resolve())
    assert env["TRITON_PTXAS_PATH"] == str(ptxas.resolve())
    with pytest.raises(ValueError, match="bounded_state_diagonal"):
        validate_semantics({
            "--diffusion-type": ["full"], "--diffusion-log-range": ["0.5"],
        })


def test_logged_command_enforces_timeout_without_leaving_child(tmp_path):
    from modelling.neural_sde.experiments.adaptive_controller import run_logged
    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_logged(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            log, tmp_path, timeout_seconds=.05)
    assert error.value.returncode == 124
    assert "TIMEOUT 0.05 seconds" in log.read_text()


def test_promotion_prefers_existing_trial_name_over_stale_parent(tmp_path):
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "output_root": "output",
        "base_train_options": {}, "allowed_overrides": {}, "seed_runs": [],
    })
    controller = Controller(study)
    controller.state["trials"] = [
        {"id": "old_parent", "status": "confirmed"},
        {"id": "new_candidate", "status": "screened"},
    ]
    decision = {
        "action": "promote", "trial_name": "new_candidate",
        "parent_trial": "old_parent",
    }
    assert controller.promotion_trial(decision)["id"] == "new_candidate"
    assert decision["promotion_target_correction"] == {
        "supplied_parent_trial": "old_parent",
        "resolved_trial": "new_candidate",
    }


def test_extract_metrics_builds_small_scientific_packet(tmp_path):
    run = tmp_path / "run"
    rollout = run / "rollout"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "validation_nll": 4., "validation_total_loss": 5.}) + "\n")
    _write(rollout / "metrics.json", {
        "one_step": {"conditional_mean_skill": .8},
        "rollout": {"consecutive_increment_correlation_rmse": .2},
    })
    _write(rollout / "quantitative" / "summary.json", {
        "samples": 101, "conditions": 2, "ensemble_size": 3,
        "energy": {"untrimmed": {"generated_over_reference": 1.2}},
        "coefficient_statistics": {
            "pod_fluctuation_coordinates": {"untrimmed": {
                "mean_bias_reference_standardized_rmse": .1,
                "covariance_trace_ratio": .9,
                "covariance_relative_frobenius_error": .2,
            }},
            "spatial_mean_coordinate": {"untrimmed": {
                "mean_bias_reference_standardized_rmse": .3,
                "covariance_trace_ratio": 1.1,
            }},
        },
        "psd": {"signals": {"center": {
            "mean_absolute_log10_band_power_error_decades": .25,
            "bands": [{"lower_hz": 100., "upper_hz": 1000., "ratio": .8,
                       "absolute_log10_error_decades": .1}],
        }}},
    })
    packet = extract_metrics(run, rollout)
    assert packet["rollout"]["horizon"] == 100
    assert packet["rollout"]["energy_ratio"] == 1.2
    assert packet["rollout"]["pod_covariance_trace_ratio"] == .9
    assert packet["psd"]["center"]["mean_abs_log10_band_error"] == .25
    assert packet["psd"]["center"]["bands"][0]["generated_over_reference"] == .8
    assert packet["one_step"]["conditional_mean_skill"] == .8
    assert packet["dynamics"]["consecutive_increment_correlation_rmse"] == .2


def test_extract_innovation_metrics_builds_small_packet(tmp_path):
    summary = tmp_path / "summary.json"
    _write(summary, {
        "split": "validation",
        "retention": [{"retained_transitions": 12}],
        "whitened_calibration": {
            "mean_rmse": .1, "std_median": .9, "std_min": .8, "std_max": 1.1,
            "covariance_relative_error_to_identity": .2,
        },
        "acf": {"whitened": {"1": {
            "median_absolute_across_coordinates": .05,
            "maximum_absolute_across_coordinates": .2,
        }}},
        "psd_over_unit_white": {"whitened": [{
            "lower_hz": 0, "upper_hz": 100, "median_power_over_unit_white": 1.2,
            "p10": .8, "p90": 1.5,
        }]},
        "decorrelation": {"whitened": [{"persistent_frames": 17}]},
    })
    packet = extract_innovation_metrics(summary)
    assert packet["split"] == "validation"
    assert packet["retained_transitions"] == 12
    assert packet["acf"]["1"]["maximum_absolute_across_coordinates"] == .2
    assert packet["first_five_persistent_decorrelation_frames"] == [17]


def test_memory_trial_keeps_only_checkpoint_compatible_options(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "best.pt").write_text("checkpoint")
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "output_root": "output",
        "base_train_options": {
            "--experiment": "0p10", "--rank": 20,
            "--hidden-layers": [64, 64], "--learning-rate": .001,
        },
        "allowed_overrides": {
            "--velocity-memory-offsets": [[1, 2, 5]],
            "--learning-rate": [[.0003], [.001]],
        },
        "seed_runs": [{
            "id": "parent", "run_dir": "parent", "hypothesis": "control",
            "change_family": "seed",
        }],
    })
    controller = Controller(study)
    trial = controller.add_trial({
        "trial_name": "memory125", "training_mode": "memory",
        "parent_trial": "parent", "hypothesis": "memory whitens",
        "change_family": "velocity_memory",
        "expected_observation": "lower ACF",
        "overrides": [
            {"flag": "--velocity-memory-offsets", "tokens": ["1", "2", "5"],
             "remove": False},
            {"flag": "--learning-rate", "tokens": ["0.0003"], "remove": False},
        ],
    })
    assert trial["parent_trial"] == "parent"
    assert trial["options"]["--train-mode"] == ["memory"]
    assert trial["options"]["--velocity-memory-offsets"] == ["1", "2", "5"]
    assert "--experiment" not in trial["options"]
    assert "--hidden-layers" not in trial["options"]


def test_zero_multistep_weight_removes_dependent_base_options(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "best.pt").write_text("checkpoint")
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "output_root": "output",
        "base_train_options": {
            "--multistep-weight": .1, "--multistep-horizons": [2, 5],
            "--multistep-mode-counts": [3, 3],
            "--multistep-window-fraction": .1,
        },
        "allowed_overrides": {
            "--multistep-weight": [[0], [.1]],
            "--velocity-memory-offsets": [[1]],
        },
        "seed_runs": [{
            "id": "parent", "run_dir": "parent", "hypothesis": "control",
            "change_family": "seed",
        }],
    })
    controller = Controller(study)
    trial = controller.add_trial({
        "trial_name": "one_step_memory", "training_mode": "memory",
        "parent_trial": "parent", "hypothesis": "isolate one step",
        "change_family": "memory", "expected_observation": "lower ACF",
        "overrides": [
            {"flag": "--velocity-memory-offsets", "tokens": ["1"], "remove": False},
            {"flag": "--multistep-weight", "tokens": ["0"], "remove": False},
        ],
    })
    assert trial["options"]["--multistep-weight"] == ["0"]
    assert "--multistep-horizons" not in trial["options"]
    assert "--multistep-mode-counts" not in trial["options"]
    assert "--multistep-window-fraction" not in trial["options"]


def test_initial_trials_are_queued_once_in_declared_order(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "best.pt").write_text("checkpoint")
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "output_root": "output",
        "base_train_options": {
            "--multistep-weight": .1, "--multistep-horizons": [2, 5],
            "--multistep-particle-counts": [64, 64],
            "--stationary-mean-weight": 0,
        },
        "allowed_overrides": {
            "--multistep-weight": [[0], [.1]],
            "--stationary-mean-weight": [[0], [.01]],
        },
        "seed_runs": [{
            "id": "parent", "run_dir": "parent", "hypothesis": "control",
            "change_family": "seed",
        }],
        "initial_trials": [{
            "id": "one_step", "training_mode": "finetune",
            "parent_trial": "parent", "hypothesis": "continued training control",
            "change_family": "one_step", "expected_observation": "little change",
            "overrides": [{
                "flag": "--multistep-weight", "tokens": ["0"], "remove": False,
            }],
        }, {
            "id": "multistep", "training_mode": "finetune",
            "parent_trial": "parent", "hypothesis": "closed-loop exposure helps",
            "change_family": "multistep", "expected_observation": "lower energy",
            "overrides": [],
        }],
    })
    controller = Controller(study)
    assert [trial["id"] for trial in controller.state["trials"]] == [
        "parent", "one_step", "multistep"]
    assert controller.trial("one_step")["origin"] == "initial_trial"
    assert "--multistep-horizons" not in controller.trial("one_step")["options"]
    assert controller.trial("multistep")["options"]["--multistep-weight"] == ["0.1"]

    resumed = Controller(study)
    assert [trial["id"] for trial in resumed.state["trials"]] == [
        "parent", "one_step", "multistep"]

    dry_run = Controller(study, dry_run=True)
    dry_run.train(dry_run.trial("one_step"))
    assert dry_run.trial("one_step")["status"] == "planned"


def _deferred_study(tmp_path):
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "output_root": "output",
        "base_train_options": {}, "allowed_overrides": {},
        "seed_runs": [], "defer_initial_trial_dependencies": True,
        "initial_trials": [
            {"id": "pretrain", "training_mode": "fresh", "parent_trial": "",
             "hypothesis": "parent", "change_family": "pretrain",
             "expected_observation": "checkpoint", "overrides": []},
            {"id": "arm", "training_mode": "finetune", "parent_trial": "pretrain",
             "hypothesis": "child", "change_family": "horizon",
             "expected_observation": "spectrum", "overrides": []},
        ],
    })
    return study


def test_deferred_initial_child_waits_for_completed_parent_and_survives_restart(tmp_path):
    study = _deferred_study(tmp_path)
    controller = Controller(study)
    assert [t['id'] for t in controller.state['trials']] == ['pretrain']
    parent = controller.trial('pretrain')
    checkpoint = tmp_path / 'output/trials/pretrain/best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('checkpoint')
    parent['status'] = 'training'
    controller._import_initial_trials()
    assert len(controller.state['trials']) == 1  # best.pt during training is not sufficient
    parent['status'] = 'screened'
    controller._import_initial_trials()
    assert controller.trial('arm')['parent_trial'] == 'pretrain'
    resumed = Controller(study)
    assert [t['id'] for t in resumed.state['trials']] == ['pretrain', 'arm']


def test_deferred_children_run_before_advisor_without_restart(tmp_path, monkeypatch):
    controller = Controller(_deferred_study(tmp_path))
    seen = []
    def train(trial):
        checkpoint = tmp_path / 'output/trials' / trial['id'] / 'best.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('checkpoint')
        trial['status'] = 'trained'
        seen.append(trial['id'])
    def evaluate(trial, tier):
        trial['status'] = 'screened'
        if trial['id'] == 'arm':
            controller.state['status'] = 'stopped'
    monkeypatch.setattr(controller, 'train', train)
    monkeypatch.setattr(controller, 'evaluate', evaluate)
    monkeypatch.setattr(controller, 'advisor', lambda: pytest.fail('Advisor called before child'))
    controller.run()
    assert seen == ['pretrain', 'arm']


def test_failed_deferred_parent_never_supplies_partial_checkpoint(tmp_path):
    controller = Controller(_deferred_study(tmp_path))
    parent = controller.trial('pretrain')
    checkpoint = tmp_path / 'output/trials/pretrain/best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('partial checkpoint')
    parent['status'] = 'failed'
    controller._import_initial_trials()
    assert controller.trial('arm')['status'] == 'cancelled_dependency'


def test_deferred_initial_dependencies_reject_forward_references(tmp_path):
    study = _deferred_study(tmp_path)
    settings = json.loads(study.read_text())
    settings['initial_trials'].reverse()
    _write(study, settings)
    with pytest.raises(ValueError, match='earlier initial trial'):
        Controller(study)


def test_screen_plot_forwards_common_discard(tmp_path, capsys):
    study = tmp_path / 'study.json'
    _write(study, {
        'repository': str(tmp_path), 'output_root': 'output',
        'base_train_options': {}, 'allowed_overrides': {}, 'seed_runs': [],
        'screen': {'horizon': 1000, 'discard': 100},
    })
    controller = Controller(study, dry_run=True)
    run = tmp_path / 'run'
    rollout = run / 'rollout_1000_screen'
    rollout.mkdir(parents=True)
    (rollout / 'rollout.npz').write_bytes(b'placeholder')
    controller.evaluate({'run_dir': str(run)}, 'screen')
    output = capsys.readouterr().out
    assert '"--discard", "100"' in output


def test_study_without_innovation_reaches_advisor(tmp_path):
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "output_root": "output",
        "scientific_goal": "reach the advisor", "scientific_constraints": [],
        "budget": {"max_training_trials": 0, "max_confirmations": 0},
        "base_train_options": {}, "allowed_overrides": {}, "seed_runs": [],
    })
    controller = Controller(study)
    controller.advisor = lambda: {
        "action": "stop", "stop_reason": "comparison complete",
        "controller_status": "pending",
    }
    controller.run()
    assert controller.state["status"] == "stopped"
    assert controller.state["stop_reason"] == "comparison complete"


def test_controller_accepts_oscillator_memory_as_isolated_warm_start(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "best.pt").write_text("checkpoint")
    _write(parent / "config.json", {"model": {
        "velocity_memory_offsets": None, "velocity_oscillator_periods": None,
    }})
    study = tmp_path / "study.json"
    _write(study, {
        "repository": str(tmp_path), "output_root": "output",
        "base_train_options": {
            "--experiment": "0p10", "--rank": 20, "--hidden-layers": [64, 64],
        },
        "allowed_overrides": {
            "--velocity-oscillator-periods": [[8, 32, 128]],
            "--velocity-oscillator-burnin": [[256]],
        },
        "seed_runs": [{
            "id": "parent", "run_dir": "parent", "hypothesis": "control",
            "change_family": "seed",
        }],
    })
    controller = Controller(study)
    trial = controller.add_trial({
        "trial_name": "oscillator_memory", "training_mode": "memory",
        "parent_trial": "parent", "hypothesis": "stable recurrent memory whitens",
        "change_family": "oscillator_memory", "expected_observation": "lower ACF",
        "overrides": [
            {"flag": "--velocity-oscillator-periods",
             "tokens": ["8", "32", "128"], "remove": False},
            {"flag": "--velocity-oscillator-burnin", "tokens": ["256"],
             "remove": False},
        ],
    })
    assert trial["options"]["--train-mode"] == ["memory"]
    assert trial["options"]["--velocity-oscillator-periods"] == ["8", "32", "128"]
    assert "--experiment" not in trial["options"]
    assert "--hidden-layers" not in trial["options"]
