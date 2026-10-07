from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from modelling.acdm.conditional_edm.checkpoint import load_checkpoint, load_model, save_checkpoint
from modelling.acdm.conditional_edm.config import EDMConfig, TrainConfig
from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.acdm.conditional_edm.evaluate import evaluate_model
from modelling.acdm.conditional_edm.train import train


def test_training_resume_evaluation_and_source_identity(tmp_path, shared_data):
    config = EDMConfig(reduced_dim=3, hidden_dim=16, num_blocks=1, noise_embedding_dim=8,
                       sigma_min=.01, sigma_max=2, num_sampling_steps=4,
                       lag_steps=2, history_conditioning=True, history_steps=2)
    training = TrainConfig(batch_size=4, epochs=1, max_train_batches=2, max_validation_batches=1)
    output = tmp_path / "run"
    def run(config_train, **kw):
        return train(data=shared_data, output_dir=output, model_config=config,
                     train_config=config_train, device=torch.device("cpu"), **kw)
    path = run(training)
    checkpoint = load_checkpoint(path)
    assert checkpoint["format_version"] == 2
    assert checkpoint["global_step"] == 2
    assert checkpoint["model_config"]["native_dt"] == pytest.approx(.01)
    assert checkpoint["split_labels"] == {k: list(v) for k, v in shared_data.splits.items()}
    reopened = checkpoint_data(checkpoint)
    assert reopened.splits == shared_data.splits
    model, _ = load_model(path)
    result, report = evaluate_model(model, reopened, num_conditions=3, ensemble_size=2,
                                    horizon=3, batch_size=2, sampling_steps=4)
    assert result["trajectories"].shape == (3, 2, 4, 3)
    assert np.isfinite(result["trajectories"]).all()
    assert set(result["repetition"]) == set(shared_data.splits["test"])
    assert report["physical_lag"] == pytest.approx(.02)
    resumed = replace(training, epochs=2)
    run(resumed, resume=path)
    resumed_checkpoint = load_checkpoint(path)
    assert resumed_checkpoint["epoch"] == 1
    # Same schedule, batches, and RNG as an uninterrupted constant-LR run.
    uninterrupted = train(data=shared_data, output_dir=tmp_path / "continuous", model_config=config,
                          train_config=resumed, device=torch.device("cpu"))
    for key, value in load_checkpoint(uninterrupted)["model_state"].items():
        torch.testing.assert_close(value, resumed_checkpoint["model_state"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="saved training configuration"):
        run(replace(resumed, epochs=3, learning_rate=.2), resume=path)
    name = shared_data.splits["train"][0]
    with h5py.File(shared_data.source_paths[name], "r+") as f:
        f["reduced/coefficients"][0, 0] += 1
    with pytest.raises(ValueError, match="Checkpoint data"):
        checkpoint_data(resumed_checkpoint)


def test_public_train_and_evaluate_commands(tmp_path, shared_data):
    from modelling.acdm.conditional_edm.train import main as train_main
    from modelling.acdm.conditional_edm.evaluate import main as evaluate_main
    output = tmp_path / "cli"
    train_main(["--experiment", "0p20", "--rank", "2", "--data-root", shared_data.source_config["data_root"],
                "--output", str(output), "--device", "cpu", "--epochs", "1", "--hidden-dim", "8",
                "--num-blocks", "1", "--batch-size", "4", "--max-train-batches", "1",
                "--max-validation-batches", "1"])
    evaluate_main(["--checkpoint", str(output / "best.pt"), "--output", str(output / "evaluation"),
                   "--device", "cpu", "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "2",
                   "--sampling-steps", "3"])
    with np.load(output / "evaluation/rollout.npz") as result:
        assert result["trajectories"].shape == (2, 2, 3, 3)
        assert set(result["repetition"]) == set(shared_data.splits["test"])


def test_pod_only_train_and_save_rollout_without_aggregate_metrics(tmp_path, shared_data):
    from modelling.acdm.conditional_edm.train import main as train_main
    from modelling.acdm.conditional_edm.evaluate import main as evaluate_main

    output = tmp_path / "pod_only"
    train_main(["--experiment", "0p20", "--rank", "2", "--no-spatial-mean",
                "--data-root", shared_data.source_config["data_root"],
                "--output", str(output), "--device", "cpu", "--epochs", "1",
                "--hidden-dim", "8", "--num-blocks", "1", "--batch-size", "4",
                "--max-train-batches", "1", "--max-validation-batches", "1", "--cache-gib", "0"])
    checkpoint = load_checkpoint(output / "best.pt")
    assert checkpoint["model_config"]["reduced_dim"] == 2
    assert checkpoint["data_config"]["include_spatial_mean"] is False
    assert checkpoint_data(checkpoint).n_coordinates == 2
    rollout = output / "rollout"
    evaluate_main(["--checkpoint", str(output / "best.pt"), "--output", str(rollout),
                   "--device", "cpu", "--split", "validation", "--num-conditions", "1",
                   "--ensemble-size", "1", "--horizon", "4", "--sampling-steps", "3",
                   "--no-metrics"])
    with np.load(rollout / "rollout.npz") as result:
        assert result["trajectories"].shape == (1, 1, 5, 2)
        assert np.isfinite(result["trajectories"]).all()


def test_pod_only_normalization_reuses_matching_checkpoint(tmp_path, shared_data):
    from modelling.acdm.conditional_edm.data import fit_normalization
    from modelling.acdm.conditional_edm.train import _pod_only_normalization
    from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData

    full_config = EDMConfig(reduced_dim=3, target_mode="increment", lag_steps=2)
    full_stats = fit_normalization(shared_data, full_config)
    path = tmp_path / "source.pt"
    save_checkpoint(path, dict(model_config=full_config.to_dict(),
                               data_signature=shared_data.signature(),
                               normalization=full_stats.to_dict()))
    pod_only = SharedPODTrainingData(**shared_data.source_config, include_spatial_mean=False)
    config = EDMConfig(reduced_dim=2, target_mode="increment", lag_steps=2)
    stats = _pod_only_normalization(path, pod_only, config, pod_only.signature())
    torch.testing.assert_close(stats.state_mean, full_stats.state_mean[1:])
    torch.testing.assert_close(stats.target_std, full_stats.target_std[1:])
    with pytest.raises(ValueError, match="transition settings"):
        _pod_only_normalization(path, pod_only, replace(config, lag_steps=1), pod_only.signature())
