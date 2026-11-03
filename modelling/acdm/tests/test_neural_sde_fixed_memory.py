"""Causality, exact finite history, parent preservation and cached regression."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.fit_fixed_memory import ridge_candidates, eligible_indices, validate_source
from modelling.neural_sde.train import load_model


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def pair(tcn=False):
    torch.manual_seed(1)
    c = SDEConfig(3, .001, hidden_layers=(5,), history_steps=1, state_variable="velocity",
                  residual_architecture="tcn" if tcn else "mlp", tcn_context_frames=7,
                  tcn_channels=5, tcn_dilations=(1, 2))
    parent = NeuralSDE(c).eval()
    c = replace(c, history_offsets=c.effective_history_offsets, fixed_memory_frequencies=(30., 80.),
                fixed_memory_damping=(10., 20.), fixed_memory_window=20)
    model = NeuralSDE(c).eval()
    model.load_state_dict(parent.state_dict(), strict=False)
    with torch.no_grad():
        model.fixed_memory.projection.copy_(torch.randn(3, 2))
    return parent, model


def test_cached_direct_recursive_features_and_no_future_leakage():
    _, model = pair()
    m = model.fixed_memory
    values = torch.randn(160, 3).cumsum(0)
    cache = m.cache(values, model.target_mean, model.target_std)
    changed = values.clone()
    changed[100:] += 99
    np.testing.assert_array_equal(cache[:100], m.cache(changed, model.target_mean, model.target_std)[:100])
    with torch.no_grad():
        h, forcing = m.initialize(values[20], values[:20].flip(0), model.target_mean, model.target_std)
        for t in range(20, 159):
            direct, _ = m.initialize(values[t], values[t-20:t].flip(0), model.target_mean, model.target_std)
            torch.testing.assert_close(h, direct, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(h, torch.from_numpy(cache[t]), rtol=2e-5, atol=2e-5)
            new = m.forcing(values[t+1]-values[t], model.target_mean, model.target_std)
            old = m.forcing(values[t-19]-values[t-20], model.target_mean, model.target_std)
            h = m.advance(h, new, old)
    assert torch.all(torch.linalg.vector_norm(m.pole, dim=-1) < 1)
    np.testing.assert_array_equal(m.cache(values[:80], model.target_mean, model.target_std), cache[:80])


@pytest.mark.parametrize("tcn", [False, True])
def test_parent_equivalence_checkpoint_and_recursive_rollout(tmp_path, tcn):
    parent, model = pair(tcn)
    state, history = torch.randn(2, 3), torch.randn(2, 20, 3)
    reference = parent.rollout(state, history_states=history[:, :parent.config.history_steps],
                               horizon=30, num_trajectories=2, seed=3)
    torch.testing.assert_close(model.rollout(state, history_states=history,
                                horizon=30, num_trajectories=2, seed=3), reference, rtol=0, atol=0)
    with torch.no_grad():
        model.fixed_memory.readout.weight.normal_(0, .002)
    # Compare efficient carried recurrence with recomputation on the complete
    # sliding history at every generated step, including nonzero diffusion.
    rng = torch.Generator().manual_seed(8)
    x, hist, states = state.clone(), history.clone(), [state.clone()]
    for _ in range(35):
        mean, factor = model.velocity_step_distribution(x, hist, x-hist[:, 0])
        noise = torch.randn(x.shape, generator=rng)
        new = x + mean + noise @ factor.T
        hist = torch.cat((x[:, None], hist[:, :-1]), 1)
        x = new
        states.append(x)
    expected = torch.stack(states, 1)
    model.tcn_rollout_cache = True
    actual = model.rollout(state, history_states=history, horizon=35, num_trajectories=1, seed=8)[:, 0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    path = tmp_path / "model.pt"
    torch.save(dict(model_config=model.config.to_dict(), model_state=model.state_dict()), path)
    loaded, _ = load_model(path)
    torch.testing.assert_close(loaded.rollout(state, history_states=history,
                               horizon=35, num_trajectories=1, seed=8)[:, 0], expected,
                               rtol=1e-4, atol=1e-4)
    assert all(not p.requires_grad for n, p in loaded.named_parameters() if not n.startswith("fixed_memory.readout"))
    torch.testing.assert_close(loaded.diffusion_matrix(), parent.diffusion_matrix(), rtol=0, atol=0)


def test_ridge_recovers_planted_readout_and_train_only_scaling():
    rng = np.random.default_rng(7)
    x, xv, w = rng.normal(size=(300, 4)), rng.normal(size=(60, 4)), rng.normal(size=(4, 3))
    fitted, scale, selected, scores = ridge_candidates(x, x@w, xv, xv@w, [1e-8, .1])
    np.testing.assert_allclose(fitted/scale[:, None], w, atol=1e-6)
    _, scale2, _, _ = ridge_candidates(x, x@w, 100*xv, xv@w, [.1])
    np.testing.assert_array_equal(scale, scale2)
    assert selected == 1e-8 and scores[1]["validation_error"] < scores[0]["validation_error"]
    fitted, _, selected, _ = ridge_candidates(x, x@w, xv, -xv@w, [.001, .1])
    assert selected is None and np.count_nonzero(fitted) == 0


def test_trim_includes_complete_memory_and_target():
    values = np.zeros((30, 2))
    values[10:, 0] = 100
    indices = eligible_indices(values, 5, 1., [1., 1.])
    assert 9 not in indices and 14 not in indices
    assert 8 in indices and 15 in indices


def test_config_serialization_and_restrictions():
    _, model = pair()
    assert SDEConfig.from_dict(model.config.to_dict()) == model.config
    with pytest.raises(ValueError, match="native-lag"):
        replace(model.config, lag_steps=2)
    with pytest.raises(ValueError, match="sub-Nyquist"):
        replace(model.config, fixed_memory_frequencies=(600., 80.))


def test_source_basis_and_split_are_checked():
    from types import SimpleNamespace
    reference = dict(shared_basis="/tmp/basis.h5", rank=2, spatial_mean_highpass_hz=50.)
    source = dict(source_provenance=dict(single_rollout=reference,
                  training_repetitions=[1], test_repetitions_used=[]))
    checkpoint = dict(data_config=reference.copy())
    data = SimpleNamespace(splits={"train": ["example_rep1", "example_rep2"]})
    validate_source(source, checkpoint, data)
    source["source_provenance"]["training_repetitions"] = [3]
    with pytest.raises(ValueError, match="training split"):
        validate_source(source, checkpoint, data)
    source["source_provenance"]["training_repetitions"] = [1]
    checkpoint["data_config"]["shared_basis"] = "/tmp/other.h5"
    with pytest.raises(ValueError, match="same POD basis"):
        validate_source(source, checkpoint, data)
