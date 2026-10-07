"""Physical screening must cover held-out repetitions and penalize failed/collapsed samples."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modelling.acdm.experiments.rollout_benchmark import RolloutBenchmark
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData


@pytest.fixture
def benchmark(shared_data):
    data = SharedPODTrainingData("0p20", rank=2, data_root=shared_data.source_config["data_root"],
                                 validation_reps=[3, 4], test_reps=[])
    return RolloutBenchmark(data, horizon=8, conditions=4, ensemble_size=2,
                            max_history_steps=2, nperseg=8)


def test_benchmark_balances_repetitions_and_preserves_real_history(benchmark):
    config = SimpleNamespace(lag_steps=1, history_conditioning=True, history_steps=2)
    reference, history, times, names, starts = benchmark.reference_windows(config)
    assert set(names) == set(benchmark.data.validation_repetitions)
    assert all((names == name).sum() == 2 for name in set(names))
    for i, (name, start) in enumerate(zip(names, starts)):
        t, values = benchmark.data.read_coordinates(name, int(start - 2), int(start + 9))
        np.testing.assert_array_equal(reference[i], values[2:])
        np.testing.assert_array_equal(history[i], values[[1, 0]])
        np.testing.assert_array_equal(times[i], t[2:])
    config.history_steps = 1
    other = benchmark.reference_windows(config)
    np.testing.assert_array_equal(other[0], reference)
    np.testing.assert_array_equal(other[-1], starts)
    config.history_steps = 3
    with pytest.raises(ValueError, match="reservation"):
        benchmark.reference_windows(config)


def test_benchmark_penalizes_both_excess_and_missing_energy_and_nonfinite(benchmark):
    rng = np.random.default_rng(12)
    reference = rng.normal(size=(4, 9, 3))
    generated = np.repeat(reference[:, None], 2, axis=1)
    exact = benchmark._metrics(generated, reference, .01)
    assert exact["score"] == pytest.approx(0, abs=1e-10)
    assert exact["energy"]["mean_ratio"] == pytest.approx(1)
    assert exact["energy"]["mean_absolute_log10_error"] == pytest.approx(0, abs=1e-10)
    for scale in (.5, 2):
        report = benchmark._metrics(generated * scale, reference, .01)
        assert report["energy"]["mean_ratio"] == pytest.approx(scale**2)
        assert report["psd"]["center"]["power_ratio"] == pytest.approx(scale**2)
        assert report["score"] > 1
    collapsed = benchmark._metrics(np.zeros_like(generated), reference, .01)
    assert collapsed["score"] > 10
    generated[0, 0, 3, 0] = np.nan
    failed = benchmark._metrics(generated, reference, .01)
    assert failed["nonfinite_fraction"] == .125
    assert failed["explosive_fraction"] >= .125
    assert failed["score"] > 15
    json.dumps(failed, allow_nan=False)


def test_benchmark_evaluate_saves_compatible_arrays_and_restores_mode(benchmark, tmp_path):
    config = SimpleNamespace(lag_steps=1, history_conditioning=True, history_steps=2)
    reference, history, _, _, _ = benchmark.reference_windows(config)

    class Replay(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.register_buffer("state_mean", torch.zeros(3))

        def rollout(self, initial_state, *, history_states, horizon, num_trajectories, num_steps, seed):
            assert not self.training
            np.testing.assert_allclose(initial_state, reference[:, 0], rtol=1e-6)
            np.testing.assert_allclose(history_states, history, rtol=1e-6)
            assert horizon == 8 and num_steps == 32 and seed == benchmark.seed
            return torch.as_tensor(np.repeat(reference[:, None], num_trajectories, axis=1))

    model = Replay()
    report = benchmark.evaluate(model, save_dir=tmp_path)
    assert model.training
    assert report["score"] == pytest.approx(0, abs=1e-10)
    assert report["split"] == "validation"
    assert report["physical_lag"] == pytest.approx(.01)
    with np.load(tmp_path / "rollout.npz") as result:
        assert result["trajectories"].shape == (4, 2, 9, 3)
        np.testing.assert_array_equal(result["reference"], reference)
    assert json.loads((tmp_path / "metrics.json").read_text())["score"] == report["score"]
