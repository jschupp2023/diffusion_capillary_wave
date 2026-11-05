import json

import numpy as np

from data_analysis.psd.rollout_psd import analyze as analyze_psd
from data_analysis.rollout import Rollout
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import (
    _configure_capillary_energy, load_model as load_sde, main as neural_sde_main,
)


def test_rollout_reconstructs_and_reports_embedded_spatial_mean(
        embedded_shared_data, tmp_path):
    rng = np.random.default_rng(77)
    samples = 64
    reference = rng.normal(size=(1, samples, embedded_shared_data.rank))
    generated = np.repeat(reference[:, None], 2, axis=1)
    time = np.arange(samples, dtype=float)[None] * embedded_shared_data.native_dt
    rollout_path = tmp_path / "evaluation" / "rollout.npz"
    rollout_path.parent.mkdir()
    np.savez_compressed(
        rollout_path, trajectories=generated, reference=reference,
        reference_time=time,
        repetition=np.asarray([embedded_shared_data.train_repetitions[0]]),
        start_index=np.asarray([0]),
    )
    metadata = {
        "rank": embedded_shared_data.rank,
        "include_spatial_mean": False,
        "spatial_mean_representation": "embedded_in_pod",
        "source_shared_basis": str(embedded_shared_data.basis_path),
        "time_units": "seconds",
        "physical_lag": embedded_shared_data.native_dt,
        "lag_steps": 1,
    }
    (rollout_path.parent / "metrics.json").write_text(json.dumps(metadata))

    rollout = Rollout(rollout_path)
    assert rollout.spatial_mean_available
    assert not rollout.include_spatial_mean
    reconstructed = rollout.reconstruct(rollout.reference)
    np.testing.assert_allclose(
        rollout.spatial_mean(rollout.reference), reconstructed.mean((-2, -1)))

    output = analyze_psd(
        rollout, nperseg=32, output=tmp_path / "psd", overwrite=True)
    report = json.loads((output / "summary.json").read_text())
    assert report["spatial_mean_representation"] == "embedded_in_pod"
    assert report["spatial_mean_available"]
    assert set(report["signals"]) == {
        "center", "center_without_spatial_mean", "spatial_mean"}


def test_neural_sde_train_and_evaluate_embedded_coordinates(
        embedded_shared_data, tmp_path):
    output = tmp_path / "embedded_sde"
    neural_sde_main([
        "train", "--experiment", embedded_shared_data.experiment,
        "--rank", str(embedded_shared_data.rank),
        "--data-root", embedded_shared_data.source_config["data_root"],
        "--hidden-layers", "8", "--epochs", "1", "--batch-size", "7",
        "--device", "cpu", "--output", str(output),
    ])
    model, checkpoint = load_sde(output / "best.pt")
    assert model.config.reduced_dim == embedded_shared_data.rank
    assert checkpoint["data_config"]["spatial_mean_representation"] == "embedded_in_pod"
    evaluation = output / "evaluation"
    neural_sde_main([
        "evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
        "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
        "--device", "cpu", "--output", str(evaluation),
    ])
    rollout = Rollout(evaluation)
    assert rollout.spatial_mean_representation == "embedded_in_pod"
    assert rollout.generated.shape == (2, 2, 4, embedded_shared_data.rank)


def test_capillary_energy_uses_every_embedded_pod_coordinate(embedded_shared_data):
    config = SDEConfig(
        embedded_shared_data.n_coordinates, embedded_shared_data.native_dt,
        state_variable="velocity", history_steps=1, drift_type="capillary_energy",
        capillary_mean_mode=False, hidden_layers=(4,))
    model = NeuralSDE(config)
    report = _configure_capillary_energy(model, embedded_shared_data)
    assert model.drift_model.energy_matrix.shape == (
        embedded_shared_data.rank, embedded_shared_data.rank)
    assert report["spatial_mean_coordinate"] is None
    assert not report["spatial_mean_Q_row_and_column_zero"]
