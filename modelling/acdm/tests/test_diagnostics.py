from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from modelling.acdm.conditional_edm.config import EDMConfig, TrainConfig
from modelling.acdm.conditional_edm.data import fit_normalization, transition_batches
from modelling.acdm.conditional_edm.diagnostics import (
    TrainingDiagnostics,
    energy_report,
    one_step_statistics,
)
from modelling.acdm.conditional_edm.model import ConditionalEDM
from modelling.acdm.conditional_edm.checkpoint import load_checkpoint
from modelling.acdm.conditional_edm.train import train, main


@pytest.mark.parametrize("target_mode", ["increment", "future_state"])
def test_fixed_probes_repeat_and_preserve_training_rng(shared_data, target_mode):
    cfg = EDMConfig(reduced_dim=3, native_dt=.01, hidden_dim=8, num_blocks=1,
                    target_mode=target_mode, history_conditioning=True, history_steps=2,
                    dropout=.2, num_sampling_steps=3)
    model = ConditionalEDM(cfg)
    model.set_normalization(fit_normalization(shared_data, cfg))
    diagnostic = TrainingDiagnostics(shared_data, cfg, TrainConfig(fixed_mse_batch_size=7, rollout_horizon=3))
    state = torch.get_rng_state().clone()
    first = diagnostic.fixed_mse(model)
    assert first == diagnostic.fixed_mse(model)
    assert set(first) == {f"fixed_MSE_{s}_sigma" for s in ("low", "mid", "high")}
    assert all(np.isfinite(value) and value >= 0 for value in first.values())
    rollout = diagnostic.rollout(model)
    assert rollout == diagnostic.rollout(model)
    assert rollout["train"]["repetition"] in shared_data.splits["train"]
    assert rollout["validation"]["repetition"] in shared_data.splits["validation"]
    assert model.training
    assert torch.equal(state, torch.get_rng_state())
    json.dumps(rollout, allow_nan=False)


def test_ddpm_fixed_probes_use_kohl_steps(shared_data):
    cfg = EDMConfig(reduced_dim=3, diffusion_formulation="ddpm", hidden_dim=8,
                    num_blocks=1, history_conditioning=True, history_steps=2)
    model = ConditionalEDM(cfg)
    model.set_normalization(fit_normalization(shared_data, cfg))
    diagnostic = TrainingDiagnostics(
        shared_data, cfg, TrainConfig(fixed_mse_batch_size=7, rollout_every=0))
    result = diagnostic.fixed_mse(model)
    assert set(result) == {"fixed_MSE_r1", "fixed_MSE_r10", "fixed_MSE_r20"}
    assert all(np.isfinite(value) and value >= 0 for value in result.values())


def test_energy_flags_growth_nonfinite_and_independent_mean_drift():
    reference = np.ones((1, 4, 3))
    generated = np.ones((1, 2, 4, 3))
    generated[0, 0, 2:, 1:] = 10
    generated[0, 1, 3, 1] = np.nan
    generated[..., -1, 0] += 4
    result = energy_report(generated, reference, np.eye(2), 1., 4, 10.)
    assert result["first_failed_step"] == [2, 3]
    assert result["fraction_steps_failed"] == 3 / 8
    assert result["finite_fraction"] == 7 / 8
    assert result["mean_energy_J"] is None
    assert result["max_spatial_mean_drift_m"] == 2
    json.dumps(result, allow_nan=False)


def test_one_step_statistics_separates_mean_pod_spread_and_amplitude():
    current = torch.tensor([[1., 2., 0.], [2., 0., 1.]])
    target = torch.tensor([[2., 1., 1.], [1., 2., 0.]])
    mean = torch.tensor([[1.5, 1., 0.], [1.5, 1., 1.]])
    offset = torch.tensor([.5, 1., 2.])
    samples = torch.stack((mean - offset, mean + offset), dim=1)
    result = one_step_statistics(samples, current, target)
    assert result["conditional_mean_rmse"]["spatial_mean"] == pytest.approx(.5)
    assert result["spread_ratio"]["spatial_mean"] == pytest.approx(1.)
    assert result["spread_ratio"]["pod"] == pytest.approx(10 / 3)
    assert result["pod_amplitude"]["stochastic_contribution"] == pytest.approx(5.)
    assert result["spatial_mean_amplitude"]["bias"] == pytest.approx(0.)
    json.dumps(result, allow_nan=False)


def test_ram_cache_preserves_transitions_statistics_and_never_reads_again(shared_data, monkeypatch):
    cfg = EDMConfig(reduced_dim=3, history_conditioning=True, history_steps=2, lag_steps=2)
    expected = list(transition_batches(shared_data, cfg, "train", 3, seed=7))
    stats = fit_normalization(shared_data, cfg)
    assert not shared_data.cache_coordinates(0)["enabled"]
    assert shared_data.cache_coordinates(.01)["enabled"]
    assert set(shared_data._coordinate_cache) == set(shared_data.train_repetitions + shared_data.validation_repetitions)
    def fail(*args, **kwargs):
        raise AssertionError("Cached coordinates should not open HDF5")
    monkeypatch.setattr("modelling.data_preparation.prepare_shared_pod_training.h5py.File", fail)
    actual = list(transition_batches(shared_data, cfg, "train", 3, seed=7))
    for a, b in zip(expected, actual):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for key, value in fit_normalization(shared_data, cfg).to_dict().items():
        torch.testing.assert_close(value, stats.to_dict()[key], rtol=0, atol=0)


def test_training_diagnostic_cadence_and_resume_overrides(tmp_path, shared_data):
    cfg = EDMConfig(reduced_dim=3, hidden_dim=8, num_blocks=1, num_sampling_steps=3)
    tc = TrainConfig(epochs=2, fixed_mse_batch_size=4, rollout_every=2, rollout_horizon=3,
                     one_step_conditions=4, one_step_ensemble_size=3,
                     batch_size=4, max_train_batches=1, max_validation_batches=1)
    path = train(data=shared_data, model_config=cfg, train_config=tc, output_dir=tmp_path / "run",
                 device=torch.device("cpu"))
    checkpoint = load_checkpoint(path)
    records = checkpoint["history"]
    assert "rollout" not in records[0] and "rollout" in records[1]
    assert "one_step" not in records[0] and records[1]["one_step"]["conditions"] == 4
    assert (tmp_path / "run" / "evaluation.jsonl").exists()
    assert (tmp_path / "run" / "diagnostics" / "epoch_0002.png").exists()
    assert records[0]["train"]["elapsed_seconds"] >= records[0]["train"]["data_seconds"] >= 0
    assert records[1]["diagnostic_seconds"] > 0
    main(["--resume", str(path), "--epochs", "3", "--cache-gib", "0", "--rollout-every", "0",
          "--fixed-sigmas", ".01", ".1", "1", "--device", "cpu"])
    saved = load_checkpoint(path)
    assert saved["train_config"]["cache_gib"] == 0
    assert saved["train_config"]["fixed_sigmas"] == (.01, .1, 1.)
    assert "rollout" not in saved["history"][-1]
    # Monitoring consumes no training random draws and does not affect optimization.
    baseline = train(data=shared_data, model_config=cfg,
                     train_config=replace(tc, fixed_mse_batch_size=0, rollout_every=0, cache_gib=0),
                     output_dir=tmp_path / "baseline", device=torch.device("cpu"))
    for key, value in load_checkpoint(baseline)["model_state"].items():
        torch.testing.assert_close(value, checkpoint["model_state"][key], rtol=0, atol=0)
