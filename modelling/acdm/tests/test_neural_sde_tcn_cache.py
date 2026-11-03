"""Reference-equivalent streaming, including BPTT through cache buffers."""
import copy

import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.losses import (MonteCarloMultistepNLL,
                                         MultistepVelocityEnergyScore,
                                         StationaryMeanLoss)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model_pair(**kwargs):
    torch.manual_seed(19)
    options = dict(reduced_dim=3, dt=.01, state_variable="velocity", history_steps=1,
                   residual_architecture="tcn", tcn_context_frames=7, tcn_channels=5,
                   tcn_kernel_size=3, tcn_dilations=(1, 2), hidden_layers=(5,))
    options.update(kwargs)
    reference = NeuralSDE(SDEConfig(**options)).double()
    with torch.no_grad():
        # Structured residuals normally start at zero: exercise all layers.
        for p in reference.drift_model.parameters():
            p.uniform_(-.08, .08)
        reference.state_mean.copy_(torch.randn(3))
        reference.state_std.copy_(torch.rand(3) + .5)
        reference.target_mean.copy_(torch.randn(3) * .1)
        reference.target_std.copy_(torch.rand(3) + .5)
    cached = copy.deepcopy(reference)
    cached.tcn_rollout_cache = True
    return reference, cached


@pytest.mark.parametrize("drift,diffusion", [
    ("mlp", "diagonal"), ("mlp", "full"),
    ("damped_residual", "state_diagonal"),
    ("damped_residual", "bounded_state_diagonal"),
])
@pytest.mark.parametrize("objective", [
    "velocity", "displacement", "energy_score", "moments", "lagged_covariance"])
def test_losses_and_all_gradients(drift, diffusion, objective):
    reference, cached = model_pair(drift_type=drift, diffusion_type=diffusion)
    current = torch.randn(4, 3, dtype=torch.double)
    history = torch.randn(4, 7, 3, dtype=torch.double)
    targets = torch.randn(4, 3, 3, dtype=torch.double)
    results = []
    for model in (reference, cached):
        x, h = current.clone().requires_grad_(), history.clone().requires_grad_()
        rng = torch.Generator().manual_seed(29)
        if objective == "lagged_covariance":
            losses = StationaryMeanLoss((1, 4, 12), lagged_covariance_mode_count=2).per_horizon_statistics_with_lagged_covariance(
                model, x, targets, history_states=h, generator=rng)
            value = sum(v[0] + .1*v[2] + .01*v[4] for v in losses.values())
        elif objective == "moments":
            loss = StationaryMeanLoss((1, 4, 12)).statistics(
                model, x, targets, history_states=h, generator=rng)
            value = loss[0] + .1 * loss[2]
        elif objective == "energy_score":
            loss = MultistepVelocityEnergyScore(
                (1, 4, 12), particle_counts=(5, 4, 2))
            value = loss(model, x, targets, history_states=h, generator=rng)
        else:
            loss = MonteCarloMultistepNLL((1, 4, 12), target=objective,
                                         particle_counts=(5, 4, 2))
            value = loss(model, x, targets, history_states=h, generator=rng)
        grads = torch.autograd.grad(value, (x, h, *model.parameters()), allow_unused=True)
        results.append((value, grads))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-9, atol=1e-10)
    for expected, actual in zip(results[0][1], results[1][1]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("context,kernel,dilations", [(7, 3, (1, 2)), (3, 3, (1, 2)), (1, 1, (1,))])
def test_rollout_and_short_context_fallback(context, kernel, dilations):
    reference, cached = model_pair(tcn_context_frames=context, tcn_kernel_size=kernel,
                                   tcn_dilations=dilations)
    x = torch.randn(2, 3, dtype=torch.double)
    h = torch.randn(2, context, 3, dtype=torch.double)
    a = reference.rollout(x, horizon=30, num_trajectories=3, history_states=h, seed=12)
    b = cached.rollout(x, horizon=30, num_trajectories=3, history_states=h, seed=12)
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    if context == 3:
        assert cached.new_tcn_cache() is None


def test_cache_reconstructs_old_velocity_and_remains_call_local():
    reference, cached = model_pair()
    x = torch.randn(2, 3, dtype=torch.double)
    h = torch.randn(2, 7, 3, dtype=torch.double)
    cache = cached.new_tcn_cache()
    for step in range(15):
        # Deliberately inconsistent explicit velocity: caching the previous
        # explicit token permanently would fail this test, even in float64.
        v = torch.randn_like(x)
        expected = reference.drift(x, h, v)
        actual = cached.cached_tcn_drift(x, h, v, cache)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        h, x = torch.cat((x[:, None], h[:, :-1]), 1), torch.randn_like(x)
    assert cached.new_tcn_cache() == {"buffers": None}


@pytest.mark.parametrize("seed", [None, 0, 17])
@pytest.mark.parametrize("horizons", [None, (1, 2, 5)])
@pytest.mark.parametrize("cached_coordinates", [False, True])
@pytest.mark.parametrize("trim", [False, True])
def test_indexed_batches_match_reference(shared_data, seed, horizons, cached_coordinates, trim):
    from modelling.neural_sde.batches import indexed_velocity_batches
    from modelling.neural_sde.train import _sde_transition_batches, _trim_keep
    config = SDEConfig(reduced_dim=shared_data.n_coordinates, dt=shared_data.native_dt,
                       state_variable="velocity", history_steps=3)
    model = NeuralSDE(config)
    if cached_coordinates:
        shared_data.cache_coordinates(1.)
    scale = torch.ones(config.reduced_dim) * 10
    cutoff = .08 if trim else None
    expected = []
    for batch in _sde_transition_batches(shared_data, config, "train", 7, seed, horizons):
        if cutoff is not None:
            keep = _trim_keep(batch, model, cutoff, scale)
            if not keep.any():
                continue
            batch = {k: v[keep] for k, v in batch.items()}
        expected.append(batch)
    actual = list(indexed_velocity_batches(shared_data, config, "train", 7, seed,
                                           horizons, cutoff, scale))
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_execution_flags_checkpoint_roundtrip(shared_data, tmp_path):
    from modelling.neural_sde.train import main, load_model
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"], "--state-variable", "velocity",
          "--residual-architecture", "tcn", "--tcn-context-frames", "7",
          "--tcn-channels", "4", "--tcn-dilations", "1", "2", "--hidden-layers", "4",
          "--train-trim-percent", "5", "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--multistep-weight", ".03", "--multistep-horizons", "2", "4",
          "--multistep-particles", "3", "--multistep-window-fraction", ".5",
          "--tcn-rollout-cache", "--indexed-training-batches", "--output", str(tmp_path / "run")])
    model, checkpoint = load_model(tmp_path / "run" / "best.pt")
    assert model.tcn_rollout_cache and model.indexed_training_batches
    assert checkpoint["training"]["tcn_rollout_cache"]
    assert checkpoint["training"]["indexed_training_batches"]


@pytest.mark.parametrize("fraction", [.025, .5, 1.])
def test_selection_before_materialization_preserves_rng(shared_data, fraction):
    import math
    from modelling.neural_sde.batches import indexed_velocity_batches
    from modelling.neural_sde.train import _sde_transition_batches, _trim_keep
    shared_data.cache_coordinates(1.)
    config = SDEConfig(reduced_dim=shared_data.n_coordinates, dt=shared_data.native_dt,
                       state_variable="velocity", history_steps=3)
    model = NeuralSDE(config)
    scale = torch.ones(config.reduced_dim)
    generators = [torch.Generator().manual_seed(42) for _ in range(2)]
    expected = []
    for b in _sde_transition_batches(shared_data, config, "train", 7, 18, (2, 4)):
        keep = _trim_keep(b, model, 100., scale)
        if not keep.any():
            continue
        b = {k: v[keep] for k, v in b.items()}
        n = len(b["current_state"])
        take = max(1, math.ceil(fraction*n))
        if take != n:
            ids = torch.randperm(n, generator=generators[0])[:take].sort().values
            b = {k: v[ids] for k, v in b.items()}
        expected.append(b)
    actual = list(indexed_velocity_batches(shared_data, config, "train", 7, 18,
                                           (2, 4), 100., scale, fraction, generators[1]))
    assert len(actual) == len(expected) > 0
    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    for a, b in zip(actual, expected):
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


@pytest.mark.parametrize("sampling", ["examples", "batches"])
def test_epoch_updates_preserve_population_and_objective(shared_data, sampling):
    from modelling.neural_sde.train import _epoch
    from modelling.neural_sde.losses import EulerMaruyamaNLL
    shared_data.cache_coordinates(1.)
    reference, cached = model_pair(reduced_dim=shared_data.n_coordinates)
    # Training batches are float32. Use modest normalization for fixture data.
    reference = reference.float()
    with torch.no_grad():
        reference.state_std.fill_(100.)
    cached = copy.deepcopy(reference)
    cached.tcn_rollout_cache = cached.indexed_training_batches = True
    results = []
    for model in (reference, cached):
        metrics = {}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss = _epoch(model, shared_data, "train", 5, "cpu", EulerMaruyamaNLL(),
                      optimizer, seed=4, train_trim_cutoff=10.,
                      multistep_objective=MonteCarloMultistepNLL((2, 4), particles=3),
                      multistep_weight=.03, multistep_noise_seed=18,
                      multistep_window_fraction=.5, multistep_window_sampling=sampling,
                      metric_output=metrics)
        results.append((loss, metrics))
    assert results[0][1]["multistep_window_count"] == results[1][1]["multistep_window_count"]
    assert results[0][0] == pytest.approx(results[1][0], rel=2e-5)
    for a, b in zip(reference.parameters(), cached.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6)


def test_trim_prefix_boundaries_nan_and_lag():
    import numpy as np
    from types import SimpleNamespace
    from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
    from modelling.neural_sde.batches import indexed_velocity_batches
    from modelling.neural_sde.train import _sde_transition_batches, _trim_keep
    values = np.arange(50, dtype=np.float64)[:, None]
    values[15:25] += 5
    values[36] = np.nan
    data = SimpleNamespace(splits={"train": ["a"]}, frame_counts={"a": 50}, batch_size=8,
                           _coordinate_cache={"a": (np.arange(50), values)})
    data.read_coordinates = lambda name, start, stop: (np.arange(50)[start:stop], values[start:stop])
    data.iter_transitions = SharedPODTrainingData.iter_transitions.__get__(data)
    config = SDEConfig(1, .01, state_variable="velocity", history_steps=1, lag_steps=2)
    model = NeuralSDE(config)
    expected = []
    for b in _sde_transition_batches(data, config, "train", 8, 4, (2, 3)):
        keep = _trim_keep(b, model, 1., torch.ones(1))
        if keep.any():
            expected.append({k: v[keep] for k, v in b.items()})
    actual = list(indexed_velocity_batches(data, config, "train", 8, 4,
                                           (2, 3), 1., torch.ones(1)))
    assert len(actual) == len(expected) > 0
    assert 0 < sum(len(b["current_state"]) for b in actual) < 43
    for a, b in zip(actual, expected):
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def test_compiled_cache_gradients_and_state_dict(monkeypatch):
    import torch._functorch.config as functorch_config
    compile_function = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw:
                        compile_function(fn, backend="aot_eager", **kw))
    monkeypatch.setattr(functorch_config, "donated_buffer", True)
    reference, cached = model_pair()
    cached.tcn_rollout_compile = True
    names = tuple(cached.state_dict())
    current = torch.randn(4, 3, dtype=torch.double)
    history = torch.randn(4, 7, 3, dtype=torch.double)
    target = torch.randn(4, 2, 3, dtype=torch.double)
    results = []
    for model in (reference, cached):
        loss = MonteCarloMultistepNLL((2, 8), particle_counts=(3, 2))(
            model, current, target, history_states=history,
            generator=torch.Generator().manual_seed(4))
        results.append((loss, torch.autograd.grad(loss, tuple(model.parameters()))))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-9, atol=1e-10)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)
    assert not functorch_config.donated_buffer
    assert tuple(cached.state_dict()) == names


def test_compiled_cache_supports_routed_moment_gradients(monkeypatch):
    """Mean/variance routing reuses one compiled rollout graph."""
    import torch._functorch.config as functorch_config
    compile_function = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw:
                        compile_function(fn, backend="aot_eager", **kw))
    monkeypatch.setattr(functorch_config, "donated_buffer", True)
    _, model = model_pair()
    model.tcn_rollout_compile = True
    current = torch.randn(4, 3, dtype=torch.double)
    history = torch.randn(4, 7, 3, dtype=torch.double)
    target = torch.randn(4, 1, 3, dtype=torch.double)
    mean_loss, _, variance_loss, _ = StationaryMeanLoss(4).statistics(
        model, current, target, history_states=history,
        generator=torch.Generator().manual_seed(7))
    drift_parameters = tuple(model.drift_model.parameters())
    diffusion_parameters = tuple(model.diffusion_model.parameters())
    drift_gradients = torch.autograd.grad(
        mean_loss + .1 * variance_loss, drift_parameters, retain_graph=True)
    diffusion_gradients = torch.autograd.grad(variance_loss, diffusion_parameters)
    assert all(torch.isfinite(value).all() for value in
               (*drift_gradients, *diffusion_gradients))
    assert not functorch_config.donated_buffer


def test_moment_sampler_keeps_global_windows(shared_data):
    from modelling.neural_sde.train import _globally_mixed_retained_velocity_batches
    shared_data.cache_coordinates(1.)
    model = NeuralSDE(SDEConfig(shared_data.n_coordinates, shared_data.native_dt,
                                state_variable="velocity", history_steps=3))
    reference = list(_globally_mixed_retained_velocity_batches(
        shared_data, model, "train", 5, 2, 8, 100., torch.ones(shared_data.n_coordinates), (2, 4)))
    model.indexed_training_batches = True
    optimized = list(_globally_mixed_retained_velocity_batches(
        shared_data, model, "train", 5, 2, 8, 100., torch.ones(shared_data.n_coordinates), (2, 4)))
    assert len(reference) == len(optimized) > 0
    for a, b in zip(reference, optimized):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
