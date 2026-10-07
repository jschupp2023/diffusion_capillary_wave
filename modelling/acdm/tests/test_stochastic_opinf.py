from __future__ import annotations

import json
import math

import h5py
import numpy as np
import torch

from data_analysis.rollout import Rollout
from modelling.stochastic_opinf.config import OpInfConfig
from modelling.stochastic_opinf.model import StochasticOpInfModel, input_signal
from modelling.stochastic_opinf.train import (
    load_model,
    main,
    regularization_candidates,
    regularization_vector,
    segmented_states,
)


def test_original_regularization_and_input_defaults() -> None:
    config = OpInfConfig(reduced_dim=3, dt=0.25)
    assert config.input_amplitude == 1.0
    assert config.input_frequency == 7.0e6
    np.testing.assert_allclose(input_signal(config, 3), [1.0, 1.0, 1.0])
    candidates, operator = regularization_candidates("A", 3)
    np.testing.assert_allclose(candidates, [1.0e-1, 1.0e2, 1.0e5])
    assert operator == "A"
    candidates, operator = regularization_candidates("AN", 3)
    np.testing.assert_allclose(candidates, [1.0, 1.0e5, 1.0e10])
    assert operator == "N"
    np.testing.assert_array_equal(regularization_vector("ABN", 4.0), [0.0, 0.0, 4.0])
    candidates, operator = regularization_candidates("ABN", 3, 1.0e2, 1.0e6)
    np.testing.assert_allclose(candidates, [1.0e2, 1.0e4, 1.0e6])
    assert operator == "N"
    np.testing.assert_array_equal(
        regularization_vector("ABN", 4.0, 1.0e3, 2.0),
        [1.0e3, 2.0, 4.0],
    )


def test_segmentation_never_crosses_repetitions(shared_data) -> None:
    mean = np.zeros(shared_data.n_coordinates)
    scale = np.ones(shared_data.n_coordinates)
    segments, counts = segmented_states(shared_data, "train", 8, mean, scale)
    assert counts == {
        shared_data.train_repetitions[0]: 2,
        shared_data.train_repetitions[1]: 2,
    }
    assert segments.shape == (4, 8, shared_data.n_coordinates)
    _, first = shared_data.read_coordinates(shared_data.train_repetitions[0], 0, 8)
    _, second_rep = shared_data.read_coordinates(shared_data.train_repetitions[1], 0, 8)
    np.testing.assert_allclose(segments[0], first)
    np.testing.assert_allclose(segments[2], second_rep)


def test_rollout_returns_physical_states_and_includes_initial_condition() -> None:
    config = OpInfConfig(reduced_dim=2, dt=0.1, sigma=0.0)
    model = StochasticOpInfModel(
        config,
        state_mean=np.array([10.0, -2.0]),
        state_std=np.array([2.0, 4.0]),
    )
    initial = torch.tensor([[12.0, 6.0]])
    generated = model.rollout(initial, horizon=3, num_trajectories=2, seed=7)
    assert generated.shape == (1, 2, 4, 2)
    torch.testing.assert_close(generated, initial[:, None, None].expand_as(generated))


def test_train_and_evaluate_reuse_shared_split_and_rollout_schema(shared_data, tmp_path) -> None:
    with h5py.File(shared_data.basis_path, "r+") as basis:
        for name in basis["repetitions"]:
            basis[f"repetitions/{name}"].attrs["signal_units"] = "microns"
    output = tmp_path / "opinf"
    main([
        "train",
        "--experiment", shared_data.experiment,
        "--rank", str(shared_data.rank),
        "--data-root", shared_data.source_config["data_root"],
        "--segment-length", "8",
        "--regularization-count", "2",
        "--output", str(output),
    ])
    checkpoint_path = output / "best.pt"
    assert checkpoint_path.is_file()
    model, checkpoint = load_model(checkpoint_path)
    assert checkpoint["data_signature"] == shared_data.signature()
    assert checkpoint["training"]["total_segments"] == 4
    assert checkpoint["training"]["selection_metric"].startswith("training segmented")
    assert model.config.reduced_dim == shared_data.rank + 1
    assert math.isfinite(checkpoint["training"]["selected_mean_error"])
    config = json.loads((output / "config.json").read_text())
    assert config["split_labels"] == {
        key: list(value) for key, value in shared_data.splits.items()
    }

    rollout_dir = output / "evaluation"
    main([
        "evaluate",
        "--checkpoint", str(checkpoint_path),
        "--num-conditions", "3",
        "--ensemble-size", "2",
        "--horizon", "4",
        "--device", "cpu",
        "--output", str(rollout_dir),
    ])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as saved:
        assert saved["trajectories"].shape == (3, 2, 5, shared_data.n_coordinates)
        np.testing.assert_allclose(
            saved["trajectories"][:, :, 0],
            np.repeat(saved["reference"][:, None, 0], 2, axis=1),
        )
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["model_type"] == "stochastic_opinf"
    assert report["state_variable"] == "state"
    assert report["input_amplitude"] == 1.0
    assert report["input_frequency"] == 7.0e6
    assert "one_step" in report and "rollout" in report
    assert Rollout(rollout_dir).generated.shape == (
        3, 2, 5, shared_data.n_coordinates
    )


def test_custom_abn_regularization_and_highpass_are_checkpointed(
        highpass_shared_data, tmp_path) -> None:
    data = highpass_shared_data
    output = tmp_path / "opinf_abn_highpass"
    main([
        "train",
        "--experiment", data.experiment,
        "--rank", str(data.rank),
        "--data-root", data.source_config["data_root"],
        "--model-form", "ABN",
        "--spatial-mean-highpass-hz", "20",
        "--segment-length", "8",
        "--regularization-count", "2",
        "--a-regularization", "1000",
        "--b-regularization", "2",
        "--n-regularization-min", "10",
        "--n-regularization-max", "100",
        "--h-regularization", "10000",
        "--output", str(output),
    ])
    _, checkpoint = load_model(output / "best.pt")
    training = checkpoint["training"]
    assert checkpoint["data_config"]["spatial_mean_highpass_hz"] == 20.0
    assert training["a_regularization"] == 1000.0
    assert training["b_regularization"] == 2.0
    assert training["n_regularization_min"] == 10.0
    assert training["n_regularization_max"] == 100.0
    assert training["h_regularization"] == 10000.0
    assert training["selected_regularization_vector"][:2] == [1000.0, 2.0]
    assert training["selected_regularization_vector"][2] in (10.0, 100.0)

    rollout_dir = output / "evaluation"
    main([
        "evaluate",
        "--checkpoint", str(output / "best.pt"),
        "--split", "test",
        "--num-conditions", "1",
        "--ensemble-size", "1",
        "--horizon", "2",
        "--no-metrics",
        "--device", "cpu",
        "--output", str(rollout_dir),
    ])
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["spatial_mean_highpass_hz"] == 20.0
