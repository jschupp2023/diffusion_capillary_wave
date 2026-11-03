"""Cheap, opt-in TCN campaign screening and promotion safeguards."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from modelling.neural_sde.experiments import adaptive_controller as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_health_reports_first_nonfinite_and_uses_saved_scale(tmp_path):
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": {"state_mean": torch.tensor([10., 20.]),
                               "state_std": torch.tensor([2., 4.])}}, checkpoint)
    path = tmp_path / "rollout.npz"
    write(tmp_path / "metrics.json", {"include_spatial_mean": True})
    values = np.broadcast_to([10., 20.], (1, 2, 5, 2)).copy()
    values[0, 0, 2, 1] = 32.  # three training stds, independent of reference variance
    np.savez(path, trajectories=values)
    health = module.rollout_health(path, checkpoint)
    assert health["all_finite"]
    assert health["peak_pod_standardized_rms"] == 3.
    assert health["first_nonfinite_step_per_path"] == [None, None]
    values[0, 1, 3:, 1] = np.nan
    np.savez(path, trajectories=values)
    health = module.rollout_health(path, checkpoint)
    assert not health["all_finite"]
    assert health["nonfinite_paths"] == 1
    assert health["first_nonfinite_step_per_path"] == [None, 3]
    assert module.screen_gate_failures({}, health)


def health(**overrides):
    return {"all_finite": True, "amplitude_statistics_finite": True,
            "peak_pod_standardized_rms": 2.,
            "peak_absolute_standardized_coordinate": 4., **overrides}


LIMITS = {"max_peak_pod_standardized_rms": 10,
          "max_peak_absolute_standardized_coordinate": 50,
          "energy_ratio_range": [.25, 4],
          "pod_covariance_trace_ratio_range": [.25, 4]}


@pytest.mark.parametrize("energy,covariance", [(0., 1.), (5., 1.), (1., .1), (1., 5.),
                                               (None, 1.), (float("nan"), 1.)])
def test_moment_gate_rejects_collapse_explosion_or_missing_evidence(energy, covariance):
    assert module.screen_gate_failures(LIMITS, health(), {
        "rollout": {"energy_ratio": energy, "pod_covariance_trace_ratio": covariance}})


def controller(tmp_path):
    study = tmp_path / "study.json"
    write(study, {"repository": str(tmp_path), "output_root": "output",
                  "base_train_options": {}, "allowed_overrides": {},
                  "screen_before_innovations": True, "screen_gate": LIMITS,
                  "screen": {"horizon": 1000, "conditions": 8,
                             "ensemble_size": 4, "seed": 814},
                  "confirmation": {"horizon": 50000}})
    result = module.Controller(study)
    trial = {"id": "candidate", "kind": "trained", "status": "trained",
             "run_dir": str(tmp_path / "candidate"), "rollouts": {}}
    result.state["trials"].append(trial)
    rollout_dir = Path(trial["run_dir"]) / "rollout_1000_screen"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "rollout.npz").touch()
    return result, trial, rollout_dir


def test_extreme_screen_skips_plotting_and_innovations(tmp_path, monkeypatch):
    ctrl, trial, directory = controller(tmp_path)
    monkeypatch.setattr(module, "rollout_health", lambda *args: health(peak_pod_standardized_rms=1.e8))
    monkeypatch.setattr(module, "run_logged", lambda *args: pytest.fail("should not plot"))
    monkeypatch.setattr(module, "extract_metrics", lambda *args: pytest.fail("must not import nonfinite dynamics"))
    monkeypatch.setattr(ctrl, "evaluate_innovations", lambda *args: pytest.fail("should not diagnose"))
    ctrl.evaluate(trial, "screen")
    assert trial["status"] == "screen_rejected"
    assert not trial["rollouts"]["screen"]["gate"]["passed"]
    assert json.loads(ctrl.state_path.read_text())["trials"][0]["status"] == "screen_rejected"
    with pytest.raises(ValueError, match="passing screen gate"):
        ctrl.evaluate(trial, "confirmation")


@pytest.mark.parametrize("energy,passed", [(1.5, True), (20., False)])
def test_innovations_only_after_both_health_and_moment_gates(tmp_path, monkeypatch, energy, passed):
    ctrl, trial, directory = controller(tmp_path)
    events = []
    monkeypatch.setattr(module, "rollout_health", lambda *args: health())
    def plot(*args):
        events.append("plot")
        write(directory / "quantitative" / "summary.json", {})
    monkeypatch.setattr(module, "run_logged", plot)
    monkeypatch.setattr(module, "extract_metrics", lambda *args: {
        "rollout": {"energy_ratio": energy, "pod_covariance_trace_ratio": 1.5}})
    monkeypatch.setattr(ctrl, "evaluate_innovations", lambda *args: events.append("innovation"))
    ctrl.evaluate(trial, "screen")
    assert events == (["plot", "innovation"] if passed else ["plot"])
    assert trial["rollouts"]["screen"]["gate"]["passed"] is passed
    assert trial["status"] == ("screened" if passed else "screen_rejected")


def test_training_metrics_select_legacy_best_and_retain_one_step_details(tmp_path):
    rows = [{"epoch": 1, "validation_nll": 20.},
            {"epoch": 2, "validation_nll": 19., "validation_one_step_nll": 18.}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = module.training_metrics(tmp_path)
    assert result["best_validation_total_epoch"] == 2
    assert result["best_validation_total_loss"] == 19.
    assert result["validation_one_step_nll"] == 18.


def test_training_metrics_retains_scalar_objective_history(tmp_path):
    rows = [{
        "epoch": 1, "train_nll": 20., "validation_nll": 19.,
        "validation_total_loss": 19.5,
        "validation_stationary_mean_weighted_contribution": .02,
        "validation_stationary_mean_normalized_error_rmse": .3,
        "validation_stationary_mean_normalized_error": [1., 2.],
        "train_stationary_variance_weighted_contribution": .01,
    }]
    (tmp_path / "metrics.jsonl").write_text(json.dumps(rows[0]) + "\n")
    result = module.training_metrics(tmp_path)
    assert result["validation_stationary_mean_weighted_contribution"] == .02
    assert result["history"][0][
        "validation_stationary_mean_normalized_error_rmse"] == .3
    assert result["history"][0][
        "train_stationary_variance_weighted_contribution"] == .01
    assert "validation_stationary_mean_normalized_error" not in result
    assert "validation_stationary_mean_normalized_error" not in result["history"][0]


def test_tcn_study_queues_matched_bounded_finetunes(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    study = json.loads((repo / "modelling/neural_sde/experiments/0p10_tcn63_rollout_study.json").read_text())
    study["repository"] = str(tmp_path)
    seed = tmp_path / study["seed_runs"][0]["run_dir"]
    seed.mkdir(parents=True)
    (seed / "best.pt").touch()
    study_path = tmp_path / "study.json"
    write(study_path, study)
    ctrl = module.Controller(study_path, dry_run=True)
    trials = [trial for trial in ctrl.state["trials"] if trial["kind"] == "trained"]
    assert len(trials) == 4
    for trial in trials:
        assert trial["training_mode"] == "finetune"
        assert trial["parent_trial"] == "tcn63_parent"
        assert trial["options"]["--train-trim-percent"] == ["5"]
        assert "--experiment" not in trial["options"]
        if trial["options"]["--multistep-weight"] != ["0"]:
            assert trial["options"]["--multistep-gradient-groups"] == ["drift"]
            assert trial["options"]["--multistep-window-sampling"] == ["examples"]
            assert "--multistep-mode-counts" not in trial["options"]
    decision = {"trial_name": "invalid_fresh", "training_mode": "fresh",
                "parent_trial": "tcn63_parent", "overrides": []}
    with pytest.raises(ValueError, match="permits only"):
        ctrl.add_trial(decision)


def test_tcn_followup_preserves_recipe_and_bounds_continuation(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    study = json.loads((repo / "modelling/neural_sde/experiments/0p10_tcn63_rollout_followup_study.json").read_text())
    study['repository'] = str(tmp_path)
    for seed in study['seed_runs']:
        directory = tmp_path / seed['run_dir']
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'best.pt').touch()
    path = tmp_path / 'followup.json'
    write(path, study)
    ctrl = module.Controller(path, dry_run=True)
    trials = [t for t in ctrl.state['trials'] if t['kind'] == 'trained']
    assert len(trials) == study['budget']['max_training_trials'] == 2
    assert trials[0]['parent_trial'] == 'tcn63_epoch2'
    assert trials[0]['options']['--epochs'] == ['1']
    assert trials[0]['options']['--multistep-weight'] == ['0.03']
    assert trials[1]['options']['--multistep-weight'] == ['0.1']
    assert study['allowed_overrides']['--multistep-target'] == [['velocity']]
    for t in trials:
        assert t['options']['--multistep-target'] == ['velocity']
        assert t['options']['--multistep-particle-counts'] == ['32']*4
        assert t['options']['--multistep-horizons'] == ['2','5','10','20']
        assert t['options']['--multistep-window-fraction'] == ['0.025']
        assert t['options']['--history-noise-weight'] == ['0']
    for t in trials[1:]:
        assert t['parent_trial'] == 'tcn63_parent'
        assert t['options']['--epochs'] == ['3']


def test_economic_stability_study_is_bounded_and_queues_declared_arms(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    study = json.loads((repo / "modelling/neural_sde/experiments/"
                        "0p10_tcn63_economic_stability_study.json").read_text())
    study["repository"] = str(tmp_path)
    for seed in study["seed_runs"]:
        directory = tmp_path / seed["run_dir"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "best.pt").touch()
    path = tmp_path / "economic.json"
    write(path, study)
    ctrl = module.Controller(path, dry_run=True)
    trials = [trial for trial in ctrl.state["trials"] if trial["kind"] == "trained"]
    assert len(trials) == 8
    assert study["budget"]["max_training_trials"] == 9
    assert study["training_timeout_seconds"] == 45 * 60
    assert study["screen"]["horizon"] == 25000
    assert {trial["parent_trial"] for trial in trials} == {
        "stable_unstructured_parent", "structured_damping_parent"}
    assert sum(trial["parent_trial"] == "structured_damping_parent"
               for trial in trials) == 2
    moment_only = ctrl.trial("moments_dense_w0p03")
    assert moment_only["options"]["--multistep-weight"] == ["0"]
    assert "--multistep-horizons" not in moment_only["options"]
    assert moment_only["options"]["--stationary-mean-horizons"] == [
        "2", "4", "5", "10", "20", "50", "100"]
    assert ctrl.trial("combined_long50_pruned")["options"][
        "--multistep-particle-counts"] == ["16", "16", "8", "8", "4"]
