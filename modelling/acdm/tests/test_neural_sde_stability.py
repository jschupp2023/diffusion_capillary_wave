"""Analytic discrete moments, fixed metrics, and opt-in training integration."""
import json

import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.stability import (DiscreteLyapunovRegularizer, StabilityConfig,
                                           discrete_linear_core, fit_stability_regularizer)
from modelling.neural_sde.train import _epoch, load_model, main, train
from modelling.neural_sde.losses import EulerMaruyamaNLL


def velocity_model(lag=1, diffusion="full", history=(1,)):
    model = NeuralSDE(SDEConfig(
        reduced_dim=2, dt=.003, lag_steps=lag, history_offsets=history,
        state_variable="velocity", hidden_layers=(5,), drift_type="damped_residual",
        damped_operator="spsd", stiffness_init=.1, damping_init=.2,
        diffusion_type=diffusion, diffusion_init=.15)).double()
    model.state_mean.copy_(torch.tensor([2., -3.]))
    model.target_mean.copy_(torch.tensor([.1, -.2]))
    model.state_std.copy_(torch.tensor([3., 5.]))
    model.target_std.copy_(torch.tensor([.4, .7]))
    with torch.no_grad():
        model.drift_model.stiffness_factor.entries[1] = .03
        model.drift_model.damping_factor.entries[1] = -.02
        if diffusion == "full":
            model.diffusion_model.lower_entries.fill_(.08)
        elif diffusion in {"state_diagonal", "bounded_state_diagonal"}:
            model.diffusion_model.network[-1].weight.fill_(.04)
    return model


def arbitrary_metric(model, **kwargs):
    # Cross terms expose an accidentally omitted z-v covariance.
    root = torch.tensor([[1., .2, .3, 0.], [0., 1., .1, .4],
                         [0., 0., 1., .2], [0., 0., 0., 1.]], dtype=torch.float64)
    return DiscreteLyapunovRegularizer(
        StabilityConfig(weight=.3, a=0., b=.9, **kwargs), root.T @ root,
        torch.cat((model.state_std, model.target_std)), {})


@pytest.mark.parametrize("lag,diffusion,history", [
    (1, "full", (1,)), (3, "full", (1,)),
    (1, "bounded_state_diagonal", (1, 3)), (3, "bounded_state_diagonal", (1,))])
def test_analytic_augmented_expectation_matches_actual_sampled_rollout(lag, diffusion, history):
    torch.manual_seed(42)
    model = velocity_model(lag, diffusion, history)
    state = torch.tensor([[1.2, -.6]], dtype=torch.float64)
    previous = state[:, None] - torch.arange(1, model.config.history_steps + 1)[None, :, None] * .1
    velocity = model._frame_velocity(state, previous)
    mean, factor = model.augmented_step_distribution(state, previous, velocity)
    covariance = factor @ factor.transpose(-1, -2)
    _, velocity_factor = model.velocity_step_distribution(state, previous, velocity)
    c = velocity_factor @ velocity_factor.transpose(-1, -2)
    torch.testing.assert_close(covariance[..., :2, :2], lag**2 * c)
    torch.testing.assert_close(covariance[..., :2, 2:], lag * c)
    regularizer = arbitrary_metric(model)
    expected = regularizer.value(mean) + torch.einsum("ij,...ji->...", regularizer.P, covariance)
    with torch.no_grad():
        samples = model.rollout(state, history_states=previous, horizon=1,
                                num_trajectories=60000, seed=901)[0, :, 1]
        augmented = torch.cat((samples, (samples - state) / lag), -1)
        energies = regularizer.value(augmented)
    assert abs(float(energies.mean() - expected.detach().squeeze())) < 5 * float(energies.std() / len(energies)**.5)
    torch.testing.assert_close(augmented.mean(0), mean.squeeze(0), atol=.008, rtol=.01)
    torch.testing.assert_close(torch.cov(augmented.T), covariance.squeeze(0), atol=.002, rtol=.04)
    metrics = regularizer.diagnostics(model, state, previous)
    torch.testing.assert_close(metrics["mean_delta_v"],
                               (expected - regularizer.value(torch.cat((state, velocity), -1))).mean())


@pytest.mark.parametrize("diffusion", ["diagonal", "full", "state_diagonal", "bounded_state_diagonal"])
def test_penalty_gradients_reach_drift_diffusion_but_not_metric(diffusion):
    model = velocity_model(diffusion=diffusion)
    state = torch.tensor([[4., 2.], [2., 3.]], dtype=torch.float64)
    history = (state - 1.)[:, None]
    regularizer = arbitrary_metric(model)
    metrics = regularizer(model, state, history_states=history)
    assert metrics["penalty"] > 0
    metrics["weighted_penalty"].backward()
    for component in (model.drift_model, model.diffusion_model):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 0
    assert not list(regularizer.parameters())
    assert regularizer.metric.grad is None
    assert regularizer.scale.grad is None


def test_core_matches_actual_update_jacobian_and_fixed_lyapunov_equation():
    model = velocity_model(lag=3)
    scale = torch.cat((model.state_std, model.target_std))
    state = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    history = (state - .1)[:, None]

    def update(scaled):
        z, v = (scaled * scale).chunk(2)
        mean, _ = model.augmented_step_distribution(z[None], (z-v)[None, None], v[None])
        return mean[0] / scale

    jacobian = torch.autograd.functional.jacobian(update, torch.ones(4, dtype=torch.float64))
    core = discrete_linear_core(model)
    torch.testing.assert_close(jacobian, core)
    reg = fit_stability_regularizer(model, StabilityConfig(weight=1., metric="lyapunov"), [(state, history)])
    assert reg.reference["spectral_radius"] < 1
    assert torch.linalg.eigvalsh(reg.P).min() > 0
    q = torch.diag(torch.tensor(reg.reference["Q_scaled_diagonal"], dtype=torch.float64))
    torch.testing.assert_close(core.T @ reg.metric @ core - reg.metric, -q)
    torch.testing.assert_close(reg.value(torch.cat((state, state-history[:, 0]), -1)).mean(),
                               torch.tensor(1., dtype=torch.float64))
    before = reg.P.clone()
    with torch.no_grad():
        model.drift_model.stiffness_factor.entries.mul_(10)
    torch.testing.assert_close(reg.P, before, rtol=0, atol=0)


def test_positive_operators_can_be_discretely_unstable_and_fallback_is_explicit():
    model = velocity_model(lag=100)
    states = torch.ones(2, 2, dtype=torch.float64)
    batches = [(states, torch.zeros(2, 1, 2, dtype=torch.float64))]
    assert torch.linalg.eigvalsh(model.drift_model.stiffness_matrix()).min() > 0
    assert torch.linalg.eigvalsh(model.drift_model.damping_matrix()).min() > 0
    with pytest.raises(ValueError, match="not Schur-stable"):
        fit_stability_regularizer(model, StabilityConfig(weight=1., metric="lyapunov"), batches)
    with pytest.warns(UserWarning, match="empirical data-scaled"):
        reg = fit_stability_regularizer(model, StabilityConfig(weight=1.), batches)
    assert reg.reference["empirical"]
    assert reg.reference["spectral_radius"] > 1
    assert "not Schur-stable" in reg.reference["fallback_reason"]


def test_perturbed_and_generated_inputs_preserve_sparse_history(monkeypatch):
    model = velocity_model(history=(1, 3))
    state = torch.tensor([[2., 3.]], dtype=torch.float64)
    history = state[:, None] - torch.arange(1., 4.)[None, :, None] * .2
    reg = arbitrary_metric(model, perturb_scale=.4, generated_steps=2)
    observed = []
    original = reg.diagnostics

    def capture(model, z, past, velocity=None, **kwargs):
        observed.append((z.clone(), past.clone(), velocity.clone()))
        return original(model, z, past, velocity, **kwargs)

    monkeypatch.setattr(reg, "diagnostics", capture)
    metrics = reg(model, state, history_states=history,
                  generator=torch.Generator().manual_seed(7))
    assert len(observed) == 4
    z, past, velocity = observed[1]
    dz, dv = z-state, velocity-.2
    torch.testing.assert_close(past, history + dz[:, None] - torch.arange(1., 4.)[None, :, None] * dv[:, None])
    for z, past, velocity in observed:
        torch.testing.assert_close(z-past[:, 0], velocity)
    torch.testing.assert_close(observed[2][1][:, 0], state)
    torch.testing.assert_close(observed[3][1][:, 0], observed[2][0])
    assert all(torch.isfinite(value) for value in metrics.values())


def test_generated_lagged_velocity_is_carried_not_reconstructed_from_history(monkeypatch):
    model = velocity_model(lag=3)
    state = torch.tensor([[2., 3.]], dtype=torch.float64)
    history = (state-.2)[:, None]
    reg = arbitrary_metric(model, generated_steps=2)
    velocities = []
    original = reg.diagnostics

    def capture(model, z, past, velocity=None, **kwargs):
        velocities.append((z-past[:, 0], velocity.clone()))
        return original(model, z, past, velocity, **kwargs)

    monkeypatch.setattr(reg, "diagnostics", capture)
    reg(model, state, history_states=history, generator=torch.Generator().manual_seed(7))
    for displacement, velocity in velocities[1:]:
        torch.testing.assert_close(displacement, 3 * velocity)


@pytest.mark.parametrize("multistep_weight", [0., .2])
def test_cli_trains_with_fixed_metric_and_separate_validation_diagnostics(shared_data, tmp_path, multistep_weight):
    output = tmp_path / "stability"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"], "--state-variable", "velocity",
          "--history-offsets", "1", "3", "--drift-type", "damped_residual",
          "--damped-operator", "spsd", "--hidden-layers", "5", "--epochs", "2",
          "--batch-size", "7", "--device", "cpu", "--train-trim-percent", "5",
          "--stability-weight", ".3", "--stability-a", ".02", "--stability-b", ".03",
          "--stability-metric", "lyapunov", "--stability-perturb-scale", ".1",
          "--stability-generated-steps", "2", "--stability-validation-seed", "18",
          "--multistep-weight", str(multistep_weight), "--multistep-particles", "3",
          "--multistep-horizons", "2", "3", "--output", str(output)])
    model, checkpoint = load_model(output / "latest.pt")
    saved = json.loads((output / "config.json").read_text())["training"]["stability"]
    assert checkpoint["training"]["stability"] == saved
    assert saved["reference"]["kind"] == "lyapunov"
    assert saved["config"]["a"] == .02 and saved["config"]["b"] == .03
    assert saved["reference"]["training_state_count"] > 0
    for record in map(json.loads, (output / "metrics.jsonl").read_text().splitlines()):
        for split in ("train", "validation"):
            raw = record[f"{split}_stability_penalty"]
            weighted = record[f"{split}_stability_weighted_penalty"]
            assert weighted == pytest.approx(.3 * raw)
            assert 0 <= record[f"{split}_stability_violation_fraction"] <= 1
            assert record[f"{split}_stability_observed_mean_v"] > 0
            assert record[f"{split}_stability_generated_mean_v"] > 0
            expected = record[f"{split}_nll"] + weighted
            if multistep_weight:
                expected += multistep_weight * (record[f"{split}_multistep_nll_h2"] + record[f"{split}_multistep_nll_h3"]) / 2
            assert record[f"{split}_total_loss"] == pytest.approx(expected)
    # Reload the frozen reference without refitting to validation or to learned operators.
    reg = DiscreteLyapunovRegularizer(StabilityConfig(**saved["config"]),
        torch.tensor(saved["metric"], dtype=torch.float64),
        torch.tensor(saved["scale"], dtype=torch.float64), saved["reference"])
    results = []
    for _ in range(2):
        metrics = {}
        _epoch(model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL(),
               stability_objective=reg, stability_noise_seed=18, metric_output=metrics)
        results.append(metrics)
    assert results[0] == results[1]


@pytest.mark.parametrize("multistep_weight", [0., .2])
def test_disabled_preserves_training_rng_parameters_and_metrics(shared_data, tmp_path, monkeypatch, multistep_weight):
    import importlib
    module = importlib.import_module("modelling.neural_sde.train")

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled regularizer must not fit or sample")

    monkeypatch.setattr(module, "fit_stability_regularizer", forbidden)
    config = SDEConfig(reduced_dim=shared_data.n_coordinates, dt=shared_data.native_dt,
                       state_variable="velocity", history_steps=1, hidden_layers=(4,))
    results = []
    for index, stability in enumerate((None, StabilityConfig(weight=0., metric="lyapunov",
                                        perturb_scale=100., generated_steps=100))):
        path = train(shared_data, tmp_path / str(index), config, epochs=1, batch_size=7,
                     seed=22, stability=stability, multistep_weight=multistep_weight,
                     multistep_horizons=(2,), multistep_particles=3)
        _, checkpoint = load_model(path)
        results.append((checkpoint, torch.random.get_rng_state(),
                        (path.parent / "metrics.jsonl").read_text()))
    assert "stability" not in results[0][0]["training"]
    assert results[0][2] == results[1][2]
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    for name, value in results[0][0]["model_state"].items():
        torch.testing.assert_close(value, results[1][0]["model_state"][name], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"a": -1.}, {"b": 0.}, {"b": 1.},
    {"b": float("nan")}, {"weight": -1.}, {"generated_steps": -1},
    {"perturb_scale": float("inf")}, {"metric": "learned"}])
def test_invalid_stability_config_rejected(kwargs):
    with pytest.raises(ValueError):
        StabilityConfig(**kwargs)
