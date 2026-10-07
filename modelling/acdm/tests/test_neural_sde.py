"""Focused Neural SDE checks using the existing shared-POD test source."""
import json
import math

import h5py
import numpy as np
import pytest
import torch

from data_analysis.energy.capillary_energy import capillary_stiffness, length_scale
from data_analysis.rollout import Rollout
from modelling.acdm.conditional_edm.evaluate import reference_windows
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.losses import (EulerMaruyamaNLL,
                                         MonteCarloMultistepNLL,
                                         MonteCarloMultistepVelocityNLL,
                                         StationaryMeanLoss)
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import (compute_diffusion_validation_metric,
                                       compute_training_trim_cutoff, _capped_multistep_batches,
                                       _configure_capillary_energy,
                                       _epoch, _fit_sde_normalization,
                                       _fit_trimmed_sde_normalization,
                                       _globally_mixed_retained_velocity_batches,
                                       _sde_transition_batches,
                                       _trim_keep,
                                       compute_trimmed_increment_covariance,
                                       load_model, main, set_trainable_components, train)


def test_drift_diffusion_likelihood_and_rollout_shapes():
    model = NeuralSDE(SDEConfig(reduced_dim=3, dt=.01, hidden_layers=(8,), diffusion_init=.2))
    current = torch.randn(5, 3)
    following = current + .01 * torch.randn(5, 3)
    assert model.drift(current).shape == current.shape
    assert model.diffusion_diagonal().shape == (3,)
    assert torch.all(model.diffusion_diagonal() > 0)
    nll = EulerMaruyamaNLL()(model, current, following)
    assert torch.isfinite(nll)
    nll.backward()
    assert model.diffusion_model.raw_diagonal.grad is not None
    generated = model.rollout(current, horizon=4, num_trajectories=6, seed=7)
    assert generated.shape == (5, 6, 5, 3)
    torch.testing.assert_close(generated[:, :, 0], current[:, None].expand(-1, 6, -1))
    torch.testing.assert_close(generated, model.rollout(current, horizon=4, num_trajectories=6, seed=7))

    unscaled = model.rollout(current, horizon=1, num_trajectories=2, seed=11)
    scaled = model.rollout(current, horizon=1, num_trajectories=2, seed=11,
                           diffusion_scale=.4)
    deterministic = current[:, None] + model.config.dt * model.drift(current)[:, None]
    torch.testing.assert_close(scaled[:, :, 1] - deterministic,
                               .4 * (unscaled[:, :, 1] - deterministic))
    zero = model.rollout(current, horizon=1, num_trajectories=2, seed=11,
                         diffusion_scale=0.)
    torch.testing.assert_close(zero[:, :, 1], deterministic.expand(-1, 2, -1))
    with pytest.raises(ValueError, match="diffusion_scale"):
        model.rollout(current, horizon=1, num_trajectories=2, diffusion_scale=-.1)


def test_nll_matches_physical_euler_gaussian():
    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.02, hidden_layers=(4,), diffusion_init=.3))
    model.state_mean.copy_(torch.tensor([2., -1.]))
    model.state_std.copy_(torch.tensor([3., 5.]))
    current = torch.tensor([[1., 2.], [3., 4.]])
    following = torch.tensor([[1.5, 2.1], [2.8, 4.3]])
    mean = current + model.config.dt * model.drift(current)
    std = math.sqrt(model.config.dt) * model.diffusion_diagonal()
    expected = (.5 * ((following - mean) / std).square()
                + torch.log(std) + .5 * math.log(2 * math.pi)).sum(-1).mean()
    torch.testing.assert_close(EulerMaruyamaNLL()(model, current, following), expected)


def test_full_diffusion_nll_and_rollout_use_correlated_noise():
    config = SDEConfig(reduced_dim=2, dt=.02, hidden_layers=(4,), diffusion_init=.3,
                       diffusion_type="full")
    model = NeuralSDE(config)
    model.state_mean.copy_(torch.tensor([2., -1.]))
    model.state_std.copy_(torch.tensor([3., 5.]))
    with torch.no_grad():
        model.diffusion_model.lower_entries.copy_(torch.tensor([.2]))
    current = torch.tensor([[1., 2.], [3., 4.]])
    following = torch.tensor([[1.5, 2.1], [2.8, 4.3]])
    mean = current + config.dt * model.drift(current)
    scale_tril = math.sqrt(config.dt) * model.diffusion_matrix()
    expected = -torch.distributions.MultivariateNormal(
        mean, scale_tril=scale_tril).log_prob(following).mean()
    nll = EulerMaruyamaNLL()(model, current, following)
    torch.testing.assert_close(nll, expected)
    nll.backward()
    assert model.diffusion_model.raw_diagonal.grad is not None
    assert model.diffusion_model.lower_entries.grad is not None

    noise = torch.randn((2, 1, 2), generator=torch.Generator().manual_seed(7))
    generated = model.rollout(current, horizon=1, num_trajectories=1, seed=7)
    torch.testing.assert_close(
        generated[:, :, 1],
        current[:, None] + config.dt * model.drift(current)[:, None]
        + torch.matmul(noise, scale_tril.T),
    )


def test_state_diagonal_diffusion_uses_drift_architecture_and_conditioning():
    config = SDEConfig(reduced_dim=2, dt=.02, hidden_layers=(4, 5),
                       diffusion_init=.3, diffusion_type="state_diagonal")
    model = NeuralSDE(config)
    drift_shapes = [(layer.in_features, layer.out_features)
                    for layer in model.drift_model.network if isinstance(layer, torch.nn.Linear)]
    diffusion_shapes = [(layer.in_features, layer.out_features)
                        for layer in model.diffusion_model.network
                        if isinstance(layer, torch.nn.Linear)]
    assert diffusion_shapes == drift_shapes

    current = torch.tensor([[1., 2.], [3., 4.]])
    following = torch.tensor([[1.5, 2.1], [2.8, 4.3]])
    initial = model.normalized_diffusion_matrix(current)
    assert initial.shape == (2, 2, 2)
    torch.testing.assert_close(torch.diagonal(initial, dim1=-2, dim2=-1),
                               torch.full((2, 2), .300001), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(initial, torch.diag_embed(
        torch.diagonal(initial, dim1=-2, dim2=-1)))

    with torch.no_grad():
        model.diffusion_model.network[-1].weight[0, 0] = 1.
    factor = math.sqrt(config.dt) * model.diffusion_matrix(current)
    assert not torch.allclose(factor[0], factor[1])
    mean = current + config.dt * model.drift(current)
    expected = -torch.distributions.MultivariateNormal(
        mean, scale_tril=factor).log_prob(following).mean()
    nll = EulerMaruyamaNLL()(model, current, following)
    torch.testing.assert_close(nll, expected)
    nll.backward()
    assert all(parameter.grad is not None for parameter in model.diffusion_model.parameters())

    noise = torch.randn((2, 1, 2), generator=torch.Generator().manual_seed(7))
    generated = model.rollout(current, horizon=1, num_trajectories=1, seed=7)
    rollout_factor = math.sqrt(config.dt) * model.diffusion_matrix(current[:, None])
    torch.testing.assert_close(
        generated[:, :, 1],
        current[:, None] + config.dt * model.drift(current)[:, None]
        + torch.matmul(rollout_factor, noise.unsqueeze(-1)).squeeze(-1),
    )


def test_bounded_state_diagonal_diffusion_has_smooth_finite_range():
    config = SDEConfig(reduced_dim=2, dt=.02, hidden_layers=(4, 5),
                       diffusion_init=.3, diffusion_type="bounded_state_diagonal",
                       diffusion_log_range=1.5)
    model = NeuralSDE(config)
    drift_shapes = [(layer.in_features, layer.out_features)
                    for layer in model.drift_model.network if isinstance(layer, torch.nn.Linear)]
    diffusion_shapes = [(layer.in_features, layer.out_features)
                        for layer in model.diffusion_model.network
                        if isinstance(layer, torch.nn.Linear)]
    assert diffusion_shapes == drift_shapes

    current = torch.tensor([[1., 2.], [3., 4.]])
    initial = model.diffusion_model(model._drift_input(current))
    torch.testing.assert_close(initial, torch.full((2, 2), .300001), atol=1e-6, rtol=1e-6)
    EulerMaruyamaNLL()(model, current, current + .1).backward()
    assert all(parameter.grad is not None for parameter in model.diffusion_model.parameters())

    with torch.no_grad():
        model.diffusion_model.network[-1].weight.fill_(1.e6)
        model.diffusion_model.network[-1].bias.copy_(torch.tensor([1.e6, -1.e6]))
    sigma = model.diffusion_model(torch.randn(100, 2) * 1.e6)
    baseline = model.diffusion_model.baseline()
    lower = baseline * math.exp(-config.diffusion_log_range)
    upper = baseline * math.exp(config.diffusion_log_range)
    assert torch.all(sigma >= lower)
    assert torch.all(sigma <= upper)
    assert torch.isfinite(sigma).all()


def test_legacy_config_defaults_to_diagonal_diffusion():
    values = SDEConfig(reduced_dim=2, dt=.02).to_dict()
    values.pop("diffusion_type")
    values.pop("diffusion_log_range")
    values.pop("drift_type")
    values.pop("stiffness_init")
    values.pop("damping_init")
    values.pop("damped_operator")
    values.pop("surface_tension")
    values.pop("capillary_mean_mode")
    restored = SDEConfig.from_dict(values)
    assert restored.diffusion_type == "diagonal"
    assert restored.diffusion_log_range == 2.
    assert restored.drift_type == "mlp"
    assert restored.damped_operator == "diagonal"
    assert restored.surface_tension == .0728
    assert restored.capillary_mean_mode


def test_damped_residual_drift_is_positive_diagonal_plus_zero_initialized_residual():
    config = SDEConfig(reduced_dim=2, dt=.02, hidden_layers=(4,),
                       state_variable="velocity", history_steps=1,
                       drift_type="damped_residual", stiffness_init=.2,
                       damping_init=.3, diffusion_init=.1)
    model = NeuralSDE(config)
    normalized = torch.tensor([[1., -2., 3., -4.]])
    expected = torch.tensor([[-.2 * 1. - .3 * 3., -.2 * -2. - .3 * -4.]])
    torch.testing.assert_close(model.drift_model(normalized), expected)
    torch.testing.assert_close(model.drift_model.stiffness_diagonal(), torch.full((2,), .2))
    torch.testing.assert_close(model.drift_model.damping_diagonal(), torch.full((2,), .3))
    assert torch.all(model.drift_model.stiffness_diagonal() > 0)
    assert torch.all(model.drift_model.damping_diagonal() > 0)
    assert torch.count_nonzero(model.drift_model.network[-1].weight) == 0
    assert torch.count_nonzero(model.drift_model.network[-1].bias) == 0

    current = torch.tensor([[1., 2.], [3., 4.]])
    history = current[:, None] - torch.tensor([[[.1, -.2]], [[-.3, .4]]])
    following = current + torch.tensor([[.2, -.1], [-.1, .3]])
    loss = EulerMaruyamaNLL()(model, current, following, history_states=history,
                              following_history_states=current[:, None])
    loss.backward()
    assert model.drift_model.raw_stiffness.grad is not None
    assert model.drift_model.raw_damping.grad is not None
    assert model.drift_model.network[-1].weight.grad is not None

    with pytest.raises(ValueError, match="requires state_variable velocity"):
        SDEConfig(reduced_dim=2, dt=.02, drift_type="damped_residual")


def test_damped_residual_spsd_operators_are_symmetric_semidefinite_and_differentiable():
    config = SDEConfig(reduced_dim=2, dt=.02, hidden_layers=(4,),
                       state_variable="velocity", history_steps=1,
                       drift_type="damped_residual", damped_operator="spsd",
                       stiffness_init=.2, damping_init=.3)
    model = NeuralSDE(config)
    torch.testing.assert_close(model.drift_model.stiffness_matrix(), .2 * torch.eye(2))
    torch.testing.assert_close(model.drift_model.damping_matrix(), .3 * torch.eye(2))
    with torch.no_grad():
        model.drift_model.stiffness_factor.entries.copy_(torch.tensor([1., 2., 3.]))
        model.drift_model.damping_factor.entries.copy_(torch.tensor([2., -1., 1.]))
    stiffness = torch.tensor([[1., 2.], [2., 13.]])
    damping = torch.tensor([[4., -2.], [-2., 2.]])
    torch.testing.assert_close(model.drift_model.stiffness_matrix(), stiffness)
    torch.testing.assert_close(model.drift_model.damping_matrix(), damping)
    torch.testing.assert_close(stiffness, stiffness.T)
    torch.testing.assert_close(damping, damping.T)
    assert torch.linalg.eigvalsh(stiffness).min() >= -1e-6
    assert torch.linalg.eigvalsh(damping).min() >= -1e-6

    normalized = torch.tensor([[1., -2., 3., -4.]])
    expected = -normalized[:, :2] @ stiffness - normalized[:, 2:] @ damping
    torch.testing.assert_close(model.drift_model(normalized), expected)
    model.drift_model(normalized).square().sum().backward()
    assert model.drift_model.stiffness_factor.entries.grad is not None
    assert model.drift_model.damping_factor.entries.grad is not None

    with pytest.raises(ValueError, match="non-diagonal damped_operator"):
        SDEConfig(reduced_dim=2, dt=.02, damped_operator="spsd")


def test_capillary_energy_drift_uses_fixed_energy_positive_mass_and_mean_stiffness():
    config = SDEConfig(reduced_dim=3, dt=.02, hidden_layers=(4,),
                       state_variable="velocity", history_steps=1,
                       drift_type="capillary_energy", stiffness_init=.2,
                       damping_init=.3)
    model = NeuralSDE(config)
    energy = torch.tensor([[0., 0., 0.], [0., 2., .5], [0., .5, 3.]])
    model.drift_model.set_energy_matrix(energy)

    assert not model.drift_model.energy_matrix.requires_grad
    torch.testing.assert_close(model.drift_model.energy_matrix, energy)
    assert torch.all(model.drift_model.mass_diagonal() > 0)
    damping = model.drift_model.damping_matrix()
    torch.testing.assert_close(damping, damping.T)
    assert torch.linalg.eigvalsh(damping).min() >= -1e-6
    torch.testing.assert_close(model.drift_model.mean_stiffness(), torch.tensor(.2))

    normalized = torch.tensor([[1., -2., .5, 3., -4., 2.]])
    position, velocity = normalized.split(3, dim=-1)
    force = position @ energy + velocity @ damping
    expected = -torch.linalg.solve(torch.diag(model.drift_model.mass_diagonal()),
                                   force.unsqueeze(-1)).squeeze(-1)
    expected[:, 0] -= .2 * position[:, 0]
    torch.testing.assert_close(model.drift_model(normalized), expected)

    model.drift_model(normalized).square().sum().backward()
    assert model.drift_model.raw_mass_multiplier.grad is not None
    assert model.drift_model.damping_factor.entries.grad is not None
    assert model.drift_model.raw_mean_stiffness.grad is not None
    assert model.drift_model.network[-1].weight.grad is not None

    with pytest.raises(ValueError, match="requires state_variable velocity"):
        SDEConfig(reduced_dim=2, dt=.02, drift_type="capillary_energy")


def test_capillary_energy_matrix_is_built_in_standardized_coordinates(shared_data):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       hidden_layers=(4,), state_variable="velocity", history_steps=1,
                       drift_type="capillary_energy")
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    report = _configure_capillary_energy(model, shared_data)
    energy = model.drift_model.energy_matrix.detach().numpy()

    assert energy.shape == (shared_data.n_coordinates,) * 2
    assert np.all(energy[0] == 0.)
    assert np.all(energy[:, 0] == 0.)
    assert np.linalg.eigvalsh(energy).min() >= -1e-12
    with h5py.File(shared_data.basis_path, "r") as basis:
        modes = np.asarray(basis["pod/modes"][:shared_data.rank])
        x_m = np.asarray(basis["grid/x"]) * length_scale(basis["grid/x"].attrs["units"])
        y_m = np.asarray(basis["grid/y"]) * length_scale(basis["grid/y"].attrs["units"])
    physical, _ = capillary_stiffness(modes, x_m, y_m, gamma=config.surface_tension)
    scale_m = model.state_std.detach().numpy()[1:] * length_scale(shared_data.signal_units)
    expected = scale_m[:, None] * physical * scale_m[None, :]
    np.testing.assert_allclose(energy[1:, 1:], expected, rtol=2e-6, atol=0.)
    assert report["coordinate_space"] == "training-standardized shared-POD position"
    assert report["energy_units"] == "J"
    assert report["spatial_mean_Q_row_and_column_zero"]


def test_lagged_nll_and_zero_drift_match_physical_gaussian():
    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.02, lag_steps=4,
                                hidden_layers=(4,), diffusion_init=.3))
    model.state_mean.copy_(torch.tensor([2., -1.]))
    model.state_std.copy_(torch.tensor([3., 5.]))
    current = torch.tensor([[1., 2.], [3., 4.]])
    following = torch.tensor([[1.5, 2.1], [2.8, 4.3]])
    h = model.config.lag_steps * model.config.dt
    std = math.sqrt(h) * model.diffusion_diagonal()
    for zero_drift in (False, True):
        mean = current if zero_drift else current + h * model.drift(current)
        expected = (.5 * ((following - mean) / std).square()
                    + torch.log(std) + .5 * math.log(2 * math.pi)).sum(-1).mean()
        torch.testing.assert_close(EulerMaruyamaNLL()(model, current, following,
                                                       zero_drift=zero_drift), expected)
    noise = torch.randn((2, 1, 2), generator=torch.Generator().manual_seed(7))
    generated = model.rollout(current, horizon=1, num_trajectories=1, seed=7)
    torch.testing.assert_close(generated[:, :, 1],
                               current[:, None] + h * model.drift(current)[:, None]
                               + std * noise)


def test_history_conditioning_uses_lagged_states_and_shifts_generated_history():
    config = SDEConfig(reduced_dim=2, dt=.01, lag_steps=2, history_steps=2,
                       hidden_layers=(4,), diffusion_init=.3)
    model = NeuralSDE(config)
    model.state_mean.copy_(torch.tensor([2., -1.]))
    model.state_std.copy_(torch.tensor([3., 5.]))
    current = torch.tensor([[1., 2.]])
    following = torch.tensor([[1.5, 2.1]])
    history = torch.tensor([[[.8, 1.8], [.6, 1.6]]])
    expected_input = torch.cat(((current - model.state_mean) / model.state_std,
                                ((history - model.state_mean) / model.state_std).flatten(1)), dim=1)
    torch.testing.assert_close(model._drift_input(current, history), expected_input)
    expected_residual = ((following - current) / model.state_std
                         - config.lag_steps * model.drift_model(expected_input))
    std = math.sqrt(config.lag_steps) * model.diffusion_model()
    expected_nll = (.5 * (expected_residual / std).square() + torch.log(std)
                    + .5 * math.log(2 * math.pi)).sum(-1).mean()
    expected_nll += torch.log(model.state_std).sum()
    torch.testing.assert_close(EulerMaruyamaNLL()(model, current, following,
                                                   history_states=history), expected_nll)
    with pytest.raises(ValueError, match="history_states"):
        EulerMaruyamaNLL()(model, current, following)

    observed = []
    def record_history(state, previous):
        observed.append(previous.clone())
        return torch.zeros_like(state)
    model.drift = record_history
    generated = model.rollout(current, history_states=history, horizon=3,
                              num_trajectories=2, seed=7)
    assert generated.shape == (1, 2, 4, 2)
    torch.testing.assert_close(observed[0], history[:, None].expand(-1, 2, -1, -1))
    torch.testing.assert_close(observed[1][..., 0, :], current[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(observed[1][..., 1, :], history[:, None, 0].expand(-1, 2, -1))
    torch.testing.assert_close(observed[2][..., 0, :], generated[:, :, 1])
    torch.testing.assert_close(observed[2][..., 1, :], current[:, None].expand(-1, 2, -1))


def test_sparse_history_offsets_select_taps_and_retain_complete_rollout_buffer():
    config = SDEConfig(reduced_dim=1, dt=.01, history_offsets=(1, 2, 3, 5, 10),
                       state_variable="increment", hidden_layers=(4,), diffusion_init=.1)
    model = NeuralSDE(config)
    assert config.history_steps == 10
    assert config.effective_history_offsets == (1, 2, 3, 5, 10)
    assert model.drift_model.network[0].in_features == 6
    restored = SDEConfig.from_dict(config.to_dict())
    assert restored == config

    current = torch.tensor([[10.]])
    history = torch.arange(9., -1., -1.).reshape(1, 10, 1)
    expected_input = torch.tensor([[10., 1., 8., 7., 5., 0.]])
    torch.testing.assert_close(model._drift_input(current, history), expected_input)

    observed = []
    def record_history(state, previous):
        observed.append(previous.clone())
        return torch.zeros_like(state)
    model.drift = record_history
    model.rollout(current, history_states=history, horizon=2, num_trajectories=2, seed=7)
    expanded = history[:, None].expand(-1, 2, -1, -1)
    torch.testing.assert_close(observed[0], expanded)
    torch.testing.assert_close(observed[1][..., 0, :], current[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(observed[1][..., 1:, :], expanded[..., :-1, :])

    with pytest.raises(ValueError, match="begin with 1"):
        SDEConfig(reduced_dim=1, dt=.01, history_offsets=(2, 5),
                  state_variable="increment")
    with pytest.raises(ValueError, match="strictly increasing"):
        SDEConfig(reduced_dim=1, dt=.01, history_offsets=(1, 3, 3))


def test_increment_state_likelihood_and_rollout_evolve_consecutive_increments():
    config = SDEConfig(reduced_dim=2, dt=.02, lag_steps=3, history_steps=1,
                       state_variable="increment", hidden_layers=(4,), diffusion_init=.3)
    model = NeuralSDE(config)
    model.state_mean.copy_(torch.tensor([2., -1.]))
    model.state_std.copy_(torch.tensor([3., 5.]))
    model.target_mean.copy_(torch.tensor([.1, -.2]))
    model.target_std.copy_(torch.tensor([.4, .8]))
    current = torch.tensor([[1., 2.]])
    previous = torch.tensor([[[.8, 1.7]]])
    following = torch.tensor([[1.5, 2.1]])
    expected_input = torch.cat(((current - model.state_mean) / model.state_std,
                                (current - previous[:, 0] - model.target_mean) / model.target_std), dim=1)
    torch.testing.assert_close(model._drift_input(current, previous), expected_input)
    h = config.dt * config.lag_steps
    next_increment = following - current
    current_increment = current - previous[:, 0]
    mean_increment = current_increment + h * model.drift(current, previous)
    std = math.sqrt(h) * model.diffusion_diagonal()
    expected_nll = (.5 * ((next_increment - mean_increment) / std).square()
                    + torch.log(std) + .5 * math.log(2 * math.pi)).sum(-1).mean()
    torch.testing.assert_close(EulerMaruyamaNLL()(model, current, following,
                                                   history_states=previous), expected_nll)
    with pytest.raises(ValueError, match="history_states"):
        EulerMaruyamaNLL()(model, current, following)

    generated = model.rollout(current, history_states=previous, horizon=2,
                              num_trajectories=1, seed=7)
    generator = torch.Generator().manual_seed(7)
    noise1 = torch.randn((1, 1, 2), generator=generator)
    noise2 = torch.randn((1, 1, 2), generator=generator)
    first = current[:, None] + current_increment[:, None] + h * model.drift(current, previous)[:, None] + std * noise1
    second_history = current[:, None, None]
    second = first + (first - current[:, None]) + h * model.drift(first, second_history) + std * noise2
    torch.testing.assert_close(generated[:, :, 1], first)
    torch.testing.assert_close(generated[:, :, 2], second)


def test_increment_state_cli_trains_and_evaluates_at_its_lag(shared_data, tmp_path):
    output = tmp_path / "increment_sde"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "increment", "--lag", "2", "--hidden-layers", "8",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu", "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    assert model.config.state_variable == "increment"
    assert model.config.history_steps == 1
    assert checkpoint["model_config"]["state_variable"] == "increment"
    rollout_dir = output / "evaluation"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
          "--no-metrics", "--device", "cpu", "--output", str(rollout_dir)])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        assert result["trajectories"].shape == (2, 2, 4, shared_data.n_coordinates)
        assert np.all(result["start_index"] >= 2)
    assert json.loads((rollout_dir / "metrics.json").read_text())["state_variable"] == "increment"
    refined = tmp_path / "increment_refined"
    main(["train", "--checkpoint", str(output / "best.pt"), "--lag", "2",
          "--state-variable", "increment", "--epochs", "1", "--batch-size", "7",
          "--device", "cpu", "--output", str(refined)])
    assert load_model(refined / "best.pt")[0].config.state_variable == "increment"
    with pytest.raises(ValueError, match="training lag"):
        main(["evaluate", "--checkpoint", str(output / "best.pt"), "--rollout-lag", "1",
              "--no-metrics", "--output", str(tmp_path / "invalid")])


def test_increment_state_diffusion_covariance_uses_second_differences():
    previous = np.array([[0., 0.], [0., 0.], [0., 0.]])
    current = np.array([[1., 0.], [0., 1.], [1., 1.]])
    following = np.array([[3., 0.], [0., 3.], [2., 3.]])

    class Data:
        n_coordinates = 2
        native_dt = .1

        def iter_transitions(self, split, *, lag_steps, history_steps, batch_size):
            assert split == "validation" and lag_steps == 2 and history_steps == 1
            yield "rep", dict(current_state=current, next_state=following,
                              history_states=previous[:, None, :])

    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.1, lag_steps=2,
                                history_steps=1, state_variable="increment"))
    model.target_std.copy_(torch.tensor([2., 4.]))
    covariance, _ = compute_trimmed_increment_covariance(Data(), model, 0, 8)
    changes = (following - 2 * current + previous) / [2., 4.]
    np.testing.assert_allclose(covariance, changes.T @ changes / (len(changes) * .2))


def test_velocity_state_uses_native_increment_likelihood_and_second_order_rollout():
    config = SDEConfig(reduced_dim=2, dt=.02, lag_steps=3, history_steps=1,
                       state_variable="velocity", hidden_layers=(4,), diffusion_init=.3)
    model = NeuralSDE(config)
    model.state_mean.copy_(torch.tensor([2., -1.]))
    model.state_std.copy_(torch.tensor([3., 5.]))
    model.target_mean.copy_(torch.tensor([1., -2.]))
    model.target_std.copy_(torch.tensor([4., 8.]))
    current = torch.tensor([[1., 2.]])
    history = torch.tensor([[[.8, 1.7]]])
    following = torch.tensor([[1.5, 2.1]])
    following_history = torch.tensor([[[1.3, 2.0]]])
    velocity = current - history[:, 0]
    next_velocity = following - following_history[:, 0]
    expected_input = torch.cat(((current - model.state_mean) / model.state_std,
                                (velocity - model.target_mean) / model.target_std), dim=1)
    torch.testing.assert_close(model._drift_input(current, history), expected_input)
    h = config.lag_steps
    std = math.sqrt(h) * model.diffusion_diagonal()
    mean_velocity = velocity + h * model.drift(current, velocity=velocity)
    expected_nll = (.5 * ((next_velocity - mean_velocity) / std).square()
                    + torch.log(std) + .5 * math.log(2 * math.pi)).sum(-1).mean()
    torch.testing.assert_close(EulerMaruyamaNLL()(model, current, following,
                                                   history_states=history,
                                                   following_history_states=following_history), expected_nll)

    generated = model.rollout(current, history_states=history, horizon=2,
                              num_trajectories=1, seed=7)
    generator = torch.Generator().manual_seed(7)
    noise1 = torch.randn((1, 1, 2), generator=generator)
    noise2 = torch.randn((1, 1, 2), generator=generator)
    velocity1 = velocity[:, None] + h * model.drift(
        current, velocity=velocity)[:, None] + std * noise1
    state1 = current[:, None] + h * velocity1
    velocity2 = velocity1 + h * model.drift(state1, velocity=velocity1) + std * noise2
    state2 = state1 + h * velocity2
    torch.testing.assert_close(generated[:, :, 1], state1)
    torch.testing.assert_close(generated[:, :, 2], state2)


def test_native_lag_velocity_is_exactly_equivalent_to_increment_mode():
    options = dict(reduced_dim=2, dt=.02, lag_steps=1, history_steps=1,
                   hidden_layers=(4,), diffusion_init=.3)
    increment = NeuralSDE(SDEConfig(**options, state_variable="increment"))
    velocity = NeuralSDE(SDEConfig(**options, state_variable="velocity"))
    velocity.load_state_dict(increment.state_dict())
    current = torch.tensor([[1., 2.], [3., 4.]])
    history = torch.tensor([[[.8, 1.7]], [[2.9, 3.6]]])
    following = torch.tensor([[1.5, 2.1], [3.2, 4.1]])
    increment_nll = EulerMaruyamaNLL()(increment, current, following,
                                       history_states=history)
    velocity_nll = EulerMaruyamaNLL()(velocity, current, following,
                                      history_states=history,
                                      following_history_states=current[:, None])
    torch.testing.assert_close(velocity_nll, increment_nll)
    torch.testing.assert_close(
        velocity.rollout(current, history_states=history, horizon=3,
                         num_trajectories=2, seed=7),
        increment.rollout(current, history_states=history, horizon=3,
                          num_trajectories=2, seed=7))


def test_multistep_horizon_one_reproduces_velocity_nll():
    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.02, lag_steps=2,
                                history_steps=1, state_variable="velocity",
                                hidden_layers=(4,), diffusion_init=.3,
                                diffusion_type="full"))
    with torch.no_grad():
        model.diffusion_model.lower_entries.copy_(torch.tensor([.12]))
    current = torch.tensor([[1., 2.], [3., 4.]])
    history = torch.tensor([[[.8, 1.7]], [[2.9, 3.6]]])
    following = torch.tensor([[1.5, 2.1], [3.2, 4.1]])
    following_history = torch.tensor([[[1.3, 2.0]], [[3.1, 3.9]]])
    target_velocity = following - following_history[:, 0]
    expected = EulerMaruyamaNLL()(
        model, current, following, history_states=history,
        following_history_states=following_history)
    actual = MonteCarloMultistepVelocityNLL((1,), particles=17)(
        model, current, target_velocity[:, None], history_states=history,
        generator=torch.Generator().manual_seed(91))
    torch.testing.assert_close(actual, expected)


def test_multistep_mode_count_scores_correct_full_covariance_pod_marginal():
    model = NeuralSDE(SDEConfig(reduced_dim=3, dt=.02, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,),
                                diffusion_init=.3, diffusion_type="full"))
    model.drift_model.zero_output()
    with torch.no_grad():
        model.diffusion_model.lower_entries.copy_(torch.tensor([.18, -.07, .11]))
    current = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    history = current[:, None] - torch.tensor([.2, -.1, .3])
    targets = torch.tensor([[[8., .4, -9.]], [[-7., -.2, 10.]]])
    objective = MonteCarloMultistepNLL(
        (1,), particles=17, target="velocity", mode_counts=(1,),
        pod_coordinate_offset=1)
    actual = objective(model, current, targets, history_states=history)

    velocity = model._frame_velocity(current, history)
    mean, factor = model.velocity_step_distribution(current, history, velocity)
    marginal_scale = torch.linalg.vector_norm(factor[1])
    expected = -torch.distributions.Normal(
        mean[:, 1], marginal_scale).log_prob(targets[:, 0, 1]).mean()
    torch.testing.assert_close(actual, expected)

    changed_excluded = targets.clone()
    changed_excluded[..., 0] += 1000
    changed_excluded[..., 2] -= 1000
    torch.testing.assert_close(
        objective(model, current, changed_excluded, history_states=history), actual)


def test_multistep_mode_count_validation():
    with pytest.raises(ValueError, match="one positive POD-mode count"):
        MonteCarloMultistepNLL((2, 4), mode_counts=(3,))
    with pytest.raises(ValueError, match="velocity horizon"):
        MonteCarloMultistepNLL((2,), target="displacement", mode_counts=(1,))

    model = NeuralSDE(SDEConfig(3, .01, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,)))
    current = torch.zeros(1, 3)
    history = torch.zeros(1, 1, 3)
    target = torch.zeros(1, 1, 3)
    with pytest.raises(ValueError, match="only 2 are available"):
        MonteCarloMultistepNLL(
            (1,), mode_counts=(3,), pod_coordinate_offset=1)(
                model, current, target, history_states=history)


def test_multistep_particle_schedule_prunes_hot_loop_after_horizons():
    class RecordingDrift(torch.nn.Module):
        def __init__(self, dimension):
            super().__init__()
            self.dimension = dimension
            self.leading_shapes = []

        def forward(self, features):
            self.leading_shapes.append(tuple(features.shape[:-1]))
            return torch.zeros(features.shape[:-1] + (self.dimension,),
                               device=features.device, dtype=features.dtype)

    model = NeuralSDE(SDEConfig(2, .01, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,)))
    recording = RecordingDrift(2)
    model.drift_model = recording
    current = torch.zeros(2, 2)
    history = torch.zeros(2, 1, 2)
    targets = torch.zeros(2, 3, 2)
    loss = MonteCarloMultistepNLL(
        (2, 4, 6), particles=5, particle_counts=(5, 3, 1))(
            model, current, targets, history_states=history,
            generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(loss)
    assert recording.leading_shapes == [
        (2, 5), (2, 5), (2, 3), (2, 3), (2, 1), (2, 1)]

    with pytest.raises(ValueError, match="nonincreasing"):
        MonteCarloMultistepNLL((2, 4), particle_counts=(3, 4))


def test_repeated_particle_schedule_preserves_scalar_behavior():
    model = NeuralSDE(SDEConfig(2, .01, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,)))
    current = torch.randn(3, 2)
    history = torch.randn(3, 1, 2)
    targets = torch.randn(3, 2, 2)
    options = dict(history_states=history)
    scalar = MonteCarloMultistepNLL((2, 4), particles=5)(
        model, current, targets, generator=torch.Generator().manual_seed(8), **options)
    scheduled = MonteCarloMultistepNLL(
        (2, 4), particles=5, particle_counts=(5, 5))(
            model, current, targets, generator=torch.Generator().manual_seed(8), **options)
    torch.testing.assert_close(scheduled, scalar)


@pytest.mark.parametrize("target", ["velocity", "displacement"])
def test_multistep_gradients_pass_through_intermediate_sampled_state(target):
    class FirstStepDrift(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_step = torch.nn.Parameter(torch.tensor(.2))
            self.calls = 0

        def forward(self, normalized_input):
            self.calls += 1
            if self.calls == 1:
                return self.first_step.expand(normalized_input.shape[:-1] + (1,))
            # The final conditional mean depends on the generated position.
            return normalized_input[..., :1]

    class FirstStepDiffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_step = torch.nn.Parameter(torch.tensor(.35))
            self.calls = 0

        def forward(self):
            self.calls += 1
            if self.calls == 1:
                return self.first_step[None]
            # Numerically constant final covariance with zero direct derivative.
            return (self.first_step * 0 + .3)[None]

    model = NeuralSDE(SDEConfig(reduced_dim=1, dt=.01, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,)))
    drift, diffusion = FirstStepDrift(), FirstStepDiffusion()
    model.drift_model, model.diffusion_model = drift, diffusion
    current = torch.tensor([[.4], [.7]])
    history = torch.tensor([[[.1]], [[.2]]])
    targets = torch.tensor([[[1.1]], [[-.3]]])
    loss = MonteCarloMultistepNLL((2,), particles=64, target=target)(
        model, current, targets, history_states=history,
        generator=torch.Generator().manual_seed(12))
    loss.backward()
    assert drift.calls == 2 and diffusion.calls == 2
    assert drift.first_step.grad is not None and drift.first_step.grad.abs() > 0
    assert diffusion.first_step.grad is not None and diffusion.first_step.grad.abs() > 0


def test_multistep_ou_mixture_converges_to_discrete_analytic_likelihood():
    theta, sigma, horizon = .18, .32, 4

    class OUDrift(torch.nn.Module):
        def forward(self, normalized_input):
            return -theta * normalized_input[..., 1:]

    model = NeuralSDE(SDEConfig(reduced_dim=1, dt=.01, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,),
                                diffusion_init=sigma))
    model.drift_model = OUDrift()
    velocities = torch.tensor([-.7, -.2, .1, .5, .9])[:, None]
    current = torch.zeros_like(velocities)
    history = -velocities[:, None, :]
    coefficient = 1 - theta
    analytic_mean = coefficient**horizon * velocities
    analytic_variance = sigma**2 * sum(coefficient**(2 * power)
                                       for power in range(horizon))
    targets = analytic_mean + torch.tensor([-.2, .1, 0., .15, -.1])[:, None]
    analytic = (-torch.distributions.Normal(
        analytic_mean, math.sqrt(analytic_variance)).log_prob(targets).mean())

    estimates = []
    for particles in (64, 20_000):
        estimates.append(MonteCarloMultistepVelocityNLL(
            (horizon,), particles=particles)(
                model, current, targets[:, None], history_states=history,
                generator=torch.Generator().manual_seed(73)))
    assert abs(estimates[1] - analytic) < .02
    assert abs(estimates[1] - analytic) < abs(estimates[0] - analytic)


def test_multistep_batches_use_complete_within_repetition_windows(shared_data):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,))
    horizons = (2, 4)
    batches = list(_sde_transition_batches(
        shared_data, config, "train", 1000, multistep_horizons=horizons))
    expected_count = sum(shared_data.frame_counts[name] - 1 - max(horizons)
                         for name in shared_data.train_repetitions)
    assert sum(len(batch["current_state"]) for batch in batches) == expected_count
    for batch in batches:
        path = batch["state_path"]
        torch.testing.assert_close(batch["next_state"], path[:, 2])
        expected_targets = torch.stack(
            tuple(path[:, 1 + horizon] - path[:, horizon] for horizon in horizons),
            dim=1)
        # Targets are differenced in float64 before conversion, whereas this
        # expected value differences the already converted float32 path.
        torch.testing.assert_close(batch["target_velocities"], expected_targets,
                                   rtol=5e-4, atol=5e-5)
        expected_displacements = torch.stack(
            tuple(path[:, 1 + horizon] - path[:, 1] for horizon in horizons), dim=1)
        torch.testing.assert_close(batch["target_displacements"], expected_displacements,
                                   rtol=5e-4, atol=5e-5)


@pytest.mark.parametrize("target", ["velocity", "displacement"])
def test_multistep_validation_noise_is_reproducible(shared_data, target):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,))
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    multistep = MonteCarloMultistepNLL((2, 4), particles=7, target=target)
    first, second = {}, {}
    for metrics in (first, second):
        _epoch(model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL(),
               multistep_objective=multistep, multistep_weight=.25,
               multistep_noise_seed=123, metric_output=metrics)
    assert first == second


def test_stationary_mean_loss_crosses_independent_endpoint_means():
    config = SDEConfig(2, dt=.01, history_steps=1, state_variable="velocity",
                       hidden_layers=(4,))
    model = NeuralSDE(config)
    model.state_std.copy_(torch.tensor([2., 4.]))
    with torch.no_grad():
        for parameter in model.drift_model.parameters():
            parameter.zero_()
        model.drift_model.network[-1].bias.copy_(torch.tensor([.5, -1.]))
        model.diffusion_model.raw_diagonal.copy_(torch.tensor([-.2, .3]))
    current = torch.tensor([[2., 4.], [4., -4.]])
    history = current.unsqueeze(1).clone()
    data_endpoint = torch.tensor([[2.2, 3.5], [4.1, -4.2]])
    objective = StationaryMeanLoss(horizon=1)
    seed = 31
    loss, mean_error, variance_loss, variance_error = objective.statistics(
        model, current, data_endpoint, history_states=history,
        generator=torch.Generator().manual_seed(seed))
    state = current[:, None, :].expand(-1, 2, -1)
    expanded_history = history[:, None].expand(-1, 2, -1, -1)
    velocity = model._frame_velocity(state, expanded_history)
    mean, factor = model.velocity_step_distribution(state, expanded_history, velocity)
    noise = torch.randn(velocity.shape, generator=torch.Generator().manual_seed(seed))
    endpoint = state + mean + torch.matmul(noise, factor.T)
    errors = (endpoint.mean(0) - data_endpoint.mean(0)) / model.state_std
    expected_loss = errors[0].mul(errors[1]).mean()
    normalized_endpoint = (endpoint - model.state_mean) / model.state_std
    normalized_data = (data_endpoint - model.state_mean) / model.state_std
    variance_errors = (normalized_endpoint.var(dim=0, correction=1)
                       - normalized_data.var(dim=0, correction=1))
    torch.testing.assert_close(mean_error, errors.mean(0))
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(variance_error, variance_errors.mean(0))
    torch.testing.assert_close(
        variance_loss, variance_errors[0].mul(variance_errors[1]).mean())
    loss.backward()
    assert any(parameter.grad is not None
               for parameter in model.drift_model.parameters())


def test_stationary_mean_sampler_mixes_source_blocks(monkeypatch):
    blocks = [
        {"current_state": torch.full((4, 2), float(index)),
         "history_states": torch.full((4, 1, 2), float(index)),
         "target_displacements": torch.full((4, 1, 2), 10.)}
        for index in range(5)
    ]

    def transition_batches(*args, **kwargs):
        yield from blocks

    monkeypatch.setattr(
        "modelling.neural_sde.train._sde_transition_batches",
        transition_batches)
    stub_model = type("StubModel", (), {"config": object()})()
    ensembles = list(_globally_mixed_retained_velocity_batches(
        None, stub_model, "train", 4, 3, 7))
    torch.testing.assert_close(
        ensembles[0]["current_state"][:, 0], torch.tensor([0., 1., 2.]))
    torch.testing.assert_close(
        ensembles[0]["data_endpoint"][:, 0], torch.tensor([10., 11., 12.]))
    assert len(ensembles) == 1


def test_stationary_mean_epoch_sampling_is_reproducible(shared_data):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,))
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    objective = StationaryMeanLoss(horizon=3)
    first, second = {}, {}
    for metrics in (first, second):
        _epoch(
            model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL(),
            stationary_mean_objective=objective, stationary_mean_weight=.2,
            stationary_mean_trajectories=3, stationary_mean_batch_fraction=.5,
            stationary_mean_noise_seed=19, stationary_mean_window_seed=23,
            metric_output=metrics)
    assert first == second
    assert first["stationary_mean_trajectory_count"] > 0
    assert first["stationary_mean_batch_count"] > 0
    assert len(first["stationary_mean_normalized_error"]) == shared_data.n_coordinates


def test_stationary_mean_epoch_adds_gradients_only_to_drift(shared_data):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,),
                       diffusion_type="bounded_state_diagonal", diffusion_log_range=.5)
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diffusion_before = {
        name: value.detach().clone() for name, value in model.diffusion_model.state_dict().items()
    }
    drift_before = {
        name: value.detach().clone() for name, value in model.drift_model.state_dict().items()
    }

    class ZeroDriftObjective(torch.nn.Module):
        def forward(self, model, current, following, **options):
            return sum(parameter.sum() * 0 for parameter in model.drift_model.parameters())

    _epoch(
        model, shared_data, "train", 7, "cpu", ZeroDriftObjective(), optimizer,
        stationary_mean_objective=StationaryMeanLoss(horizon=2),
        stationary_mean_weight=.2, stationary_mean_trajectories=3,
        stationary_mean_batch_fraction=1, stationary_mean_noise_seed=11,
        stationary_mean_window_seed=13)
    assert any(not torch.equal(value, drift_before[name])
               for name, value in model.drift_model.state_dict().items())
    for name, value in model.diffusion_model.state_dict().items():
        torch.testing.assert_close(value, diffusion_before[name], rtol=0, atol=0)


def test_stationary_variance_epoch_adds_diffusion_gradients(shared_data):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,),
                       diffusion_type="bounded_state_diagonal", diffusion_log_range=.5)
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diffusion_before = {
        name: value.detach().clone() for name, value in model.diffusion_model.state_dict().items()
    }

    class ZeroObjective(torch.nn.Module):
        def forward(self, model, current, following, **options):
            return sum(parameter.sum() * 0 for parameter in model.parameters())

    metrics = {}
    _epoch(
        model, shared_data, "train", 7, "cpu", ZeroObjective(), optimizer,
        stationary_mean_objective=StationaryMeanLoss(horizon=2),
        stationary_variance_weight=.2, stationary_mean_trajectories=3,
        stationary_mean_batch_fraction=1, stationary_mean_noise_seed=11,
        stationary_mean_window_seed=13, metric_output=metrics)
    assert any(not torch.equal(value, diffusion_before[name])
               for name, value in model.diffusion_model.state_dict().items())
    assert "stationary_variance_loss" in metrics
    assert len(metrics["stationary_variance_normalized_error"]) == shared_data.n_coordinates


def test_multistep_window_fraction_subsamples_only_multistep_examples(shared_data):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,))
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    selections = []

    class RecordingNLL(MonteCarloMultistepNLL):
        def per_horizon(self, model, current, targets, **kwargs):
            selections.append(current.detach().clone())
            return super().per_horizon(model, current, targets, **kwargs)

    metrics = {}
    _epoch(model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL(),
           multistep_objective=RecordingNLL((2,), particles=2),
           multistep_weight=.1, multistep_noise_seed=13,
           multistep_window_fraction=.4, multistep_window_seed=17,
           metric_output=metrics)
    assert selections
    assert max(map(len, selections)) <= math.ceil(.4 * 7)
    assert metrics["multistep_window_count"] == sum(map(len, selections))
    assert 0 < metrics["multistep_window_count"] < metrics["window_count"]

    selections.clear()
    batch_metrics = {}
    _epoch(model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL(),
           multistep_objective=RecordingNLL((2,), particles=2),
           multistep_weight=.1, multistep_noise_seed=13,
           multistep_window_fraction=.4, multistep_window_sampling="batches",
           multistep_window_seed=17, metric_output=batch_metrics)
    assert selections
    # The selected batch is scored whole rather than reduced to ceil(.4*7)=3.
    assert max(map(len, selections)) > math.ceil(.4 * 7)
    assert batch_metrics["multistep_window_count"] == sum(map(len, selections))


def test_capped_multistep_windows_do_not_depend_on_scoring_batch_size(shared_data):
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,))
    model = NeuralSDE(config)
    selections = []
    for batch_size in (2, 7):
        batches = _capped_multistep_batches(
            shared_data, model, "validation", batch_size, (2, 4), 5, 29)
        selections.append(torch.cat([batch["current_state"] for batch in batches]))
    assert len(selections[0]) == 5
    torch.testing.assert_close(selections[0], selections[1])


@pytest.mark.parametrize("target", ["velocity", "displacement"])
def test_multistep_cli_logs_separate_horizons_and_checkpoint_options(shared_data, tmp_path, target):
    output = tmp_path / "velocity_multistep"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "velocity", "--lag", "1", "--hidden-layers", "8",
          "--multistep-weight", ".2", "--multistep-horizons", "2", "4",
          "--multistep-particles", "5", "--multistep-validation-seed", "19",
          "--multistep-target", target, "--train-trim-percent", "5",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(output)])
    _, checkpoint = load_model(output / "best.pt")
    training = checkpoint["training"]
    assert training["multistep_weight"] == .2
    assert training["multistep_target"] == target
    assert training["multistep_horizons"] == [2, 4]
    assert training["multistep_particles"] == 5
    assert training["multistep_validation_seed"] == 19
    record = json.loads((output / "metrics.jsonl").read_text())
    assert record["multistep_target"] == target
    assert record["validation_multistep_windows_scored"] > 0
    for key in ("train_nll", "validation_nll", "train_total_loss",
                "validation_total_loss", "train_multistep_nll_h2",
                "train_multistep_nll_h4", "validation_multistep_nll_h2",
                "validation_multistep_nll_h4"):
        assert math.isfinite(record[key])

    evaluation = output / "likelihood_m3"
    main(["evaluate", "--checkpoint", str(output / "best.pt"),
          "--num-conditions", "1", "--ensemble-size", "1", "--horizon", "1",
          "--no-metrics", "--multistep-horizons", "2", "4",
          "--multistep-particles", "3", "--multistep-seed", "23",
          "--multistep-max-windows", "5", "--multistep-window-seed", "29",
          "--device", "cpu", "--output", str(evaluation)])
    likelihood = json.loads((evaluation / "metrics.json").read_text())[
        "multistep_likelihood"]
    assert likelihood["particles"] == 3
    assert likelihood["target"] == target
    assert likelihood["training_trim_applied"] is True
    assert likelihood["normalized_increment_norm_cutoff"] == training["train_trim_norm_cutoff"]
    assert likelihood["seed"] == 23
    assert likelihood["max_windows"] == 5
    assert likelihood["window_seed"] == 29
    assert likelihood["windows_scored"] == 5
    assert likelihood["horizons"] == [2, 4]
    assert set(likelihood["per_horizon_nll"]) == {"2", "4"}

    overridden_target = "displacement" if target == "velocity" else "velocity"
    overridden = output / "likelihood_override"
    main(["evaluate", "--checkpoint", str(output / "best.pt"),
          "--num-conditions", "1", "--ensemble-size", "1", "--horizon", "1",
          "--no-metrics", "--multistep-horizons", "2", "4",
          "--multistep-target", overridden_target, "--multistep-particles", "3",
          "--multistep-max-windows", "5", "--device", "cpu", "--output", str(overridden)])
    assert json.loads((overridden / "metrics.json").read_text())[
        "multistep_likelihood"]["target"] == overridden_target
    if target == "velocity":
        checkpoint["training"].pop("multistep_target")
        legacy_path = output / "legacy.pt"
        torch.save(checkpoint, legacy_path)
        legacy_output = output / "legacy_likelihood"
        main(["evaluate", "--checkpoint", str(legacy_path),
              "--num-conditions", "1", "--ensemble-size", "1", "--horizon", "1",
              "--no-metrics", "--multistep-horizons", "2", "--multistep-particles", "3",
              "--multistep-max-windows", "5", "--device", "cpu", "--output", str(legacy_output)])
        assert json.loads((legacy_output / "metrics.json").read_text())[
            "multistep_likelihood"]["target"] == "velocity"


def test_multistep_mode_counts_cli_are_saved_and_inherited(shared_data, tmp_path):
    output = tmp_path / "velocity_selected_modes"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "velocity", "--lag", "1", "--hidden-layers", "4",
          "--multistep-weight", ".1", "--multistep-horizons", "2", "4",
          "--multistep-mode-counts", "2", "1", "--multistep-particles", "3",
          "--multistep-particle-counts", "3", "2",
          "--multistep-window-fraction", ".5",
          "--multistep-window-sampling", "batches",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(output)])
    _, checkpoint = load_model(output / "best.pt")
    assert checkpoint["training"]["multistep_mode_counts"] == [2, 1]
    assert checkpoint["training"]["multistep_particle_counts"] == [3, 2]
    assert checkpoint["training"]["multistep_window_fraction"] == .5
    assert checkpoint["training"]["multistep_window_sampling"] == "batches"
    record = json.loads((output / "metrics.jsonl").read_text())
    assert record["multistep_mode_counts"] == [2, 1]
    assert record["multistep_particle_counts"] == [3, 2]
    assert record["multistep_window_fraction"] == .5
    assert record["multistep_window_sampling"] == "batches"
    assert (record["train_multistep_windows_scored"]
            < record["train_multistep_windows_available"])

    evaluation = output / "likelihood"
    main(["evaluate", "--checkpoint", str(output / "best.pt"),
          "--num-conditions", "1", "--ensemble-size", "1", "--horizon", "1",
          "--no-metrics", "--multistep-horizons", "2", "4",
          "--multistep-particles", "3", "--multistep-max-windows", "3",
          "--multistep-particle-counts", "3", "2",
          "--device", "cpu", "--output", str(evaluation)])
    likelihood = json.loads((evaluation / "metrics.json").read_text())[
        "multistep_likelihood"]
    assert likelihood["mode_counts"] == [2, 1]
    assert likelihood["particle_counts"] == [3, 2]
    assert likelihood["coordinate_selection"].startswith("first N shared-POD modes")


def test_stationary_mean_cli_is_saved_and_logged(shared_data, tmp_path):
    output = tmp_path / "velocity_stationary_mean"
    main(["train", "--experiment", shared_data.experiment,
          "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "velocity", "--lag", "1",
          "--hidden-layers", "4", "--train-trim-percent", "5",
          "--stationary-mean-weight", ".03",
          "--stationary-variance-weight", ".04",
          "--stationary-mean-horizon", "2",
          "--stationary-mean-trajectories", "2",
          "--stationary-mean-batch-fraction", "1",
          "--stationary-mean-validation-seed", "17",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(output)])
    _, checkpoint = load_model(output / "best.pt")
    training = checkpoint["training"]
    assert training["stationary_mean_weight"] == .03
    assert training["stationary_variance_weight"] == .04
    assert training["stationary_mean_horizon"] == 2
    assert training["stationary_mean_trajectories"] == 2
    assert training["stationary_mean_batch_fraction"] == 1
    assert training["stationary_mean_validation_seed"] == 17
    assert training["stationary_mean_gradient_parameters"].startswith("mean: drift only")
    assert training["stationary_mean_generated_paths_per_start"] == 2
    assert "generated-minus-measured endpoint mean" in training[
        "stationary_mean_definition"]
    record = json.loads((output / "metrics.jsonl").read_text())
    assert record["stationary_mean_weight"] == .03
    assert record["stationary_variance_weight"] == .04
    assert record["stationary_mean_horizon"] == 2
    assert record["train_stationary_mean_trajectory_count"] > 0
    assert record["validation_stationary_mean_trajectory_count"] > 0
    assert len(record["validation_stationary_mean_normalized_error"]) == shared_data.n_coordinates
    assert "validation_stationary_mean_weighted_contribution" in record
    assert "validation_stationary_variance_weighted_contribution" in record


def test_lag_one_velocity_sparse_history_conditions_residual_and_rollout():
    config = SDEConfig(reduced_dim=1, dt=.02, lag_steps=1,
                       history_offsets=(1, 2, 3, 5, 10),
                       state_variable="velocity", hidden_layers=(4,),
                       drift_type="damped_residual", damped_operator="spsd",
                       stiffness_init=.2, damping_init=.3,
                       diffusion_type="bounded_state_diagonal")
    model = NeuralSDE(config)
    assert config.history_steps == 10
    assert model.drift_model.network[0].in_features == 6
    assert model.diffusion_model.network[0].in_features == 6

    current = torch.tensor([[10.]])
    history = torch.arange(9., -1., -1.).reshape(1, 10, 1)
    expected_input = torch.tensor([[10., 1., 8., 7., 5., 0.]])
    torch.testing.assert_close(model._drift_input(current, history), expected_input)
    torch.testing.assert_close(model.drift_model(expected_input), torch.tensor([[-2.3]]))

    observed = []

    def record_history(state, previous, velocity=None):
        observed.append(previous.clone())
        return torch.zeros_like(state)

    model.drift = record_history
    model.rollout(current, history_states=history, horizon=2,
                  num_trajectories=2, seed=7, diffusion_scale=0.)
    expanded = history[:, None].expand(-1, 2, -1, -1)
    torch.testing.assert_close(observed[0], expanded)
    torch.testing.assert_close(observed[1][..., 0, :], current[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(observed[1][..., 1:, :], expanded[..., :-1, :])

    with pytest.raises(ValueError, match="begin with 1"):
        SDEConfig(reduced_dim=1, dt=.02, lag_steps=1, history_offsets=(2, 5),
                  state_variable="velocity")
    with pytest.raises(ValueError, match="only at lag_steps 1"):
        SDEConfig(reduced_dim=1, dt=.02, lag_steps=2, history_offsets=(1, 2),
                  state_variable="velocity")


def test_velocity_trim_checks_the_complete_native_path():
    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.01, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,)))
    batch = {
        "current_state": torch.zeros(2, 2),
        "next_state": torch.ones(2, 2),
        "history_states": torch.tensor([[[0., 0.]], [[10., 0.]]]),
        "state_path": torch.tensor([[[0., 0.], [0., 0.], [1., 1.]],
                                    [[10., 0.], [0., 0.], [1., 1.]]]),
    }
    assert _trim_keep(batch, model, 2.).tolist() == [True, False]


def test_velocity_cli_trains_and_evaluates_at_different_lags(shared_data, tmp_path):
    output = tmp_path / "velocity_sde"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "velocity", "--lag", "2", "--hidden-layers", "8",
          "--drift-type", "damped_residual", "--stiffness-init", ".02",
          "--damping-init", ".03", "--damped-operator", "spsd",
          "--train-trim-percent", "5", "--epochs", "1", "--batch-size", "7",
          "--device", "cpu", "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    assert model.config.state_variable == "velocity"
    assert model.config.drift_type == "damped_residual"
    assert checkpoint["model_config"]["drift_type"] == "damped_residual"
    assert checkpoint["model_config"]["damped_operator"] == "spsd"
    assert model.config.history_steps == 1
    assert torch.linalg.eigvalsh(model.drift_model.stiffness_matrix()).min() >= -1e-6
    assert torch.linalg.eigvalsh(model.drift_model.damping_matrix()).min() >= -1e-6
    assert model.drift_model.network[0].in_features == 2 * shared_data.n_coordinates
    assert checkpoint["training"]["train_trim_removed_transition_count"] >= 1

    rollout_dir = output / "evaluation_lag3"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
          "--rollout-lag", "3", "--no-metrics", "--device", "cpu",
          "--output", str(rollout_dir)])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        assert result["trajectories"].shape == (2, 2, 4, shared_data.n_coordinates)
        assert np.all(result["start_index"] >= 1)
        np.testing.assert_allclose(np.diff(result["reference_time"], axis=1),
                                   3 * shared_data.native_dt)
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["training_lag_steps"] == 2
    assert report["lag_steps"] == 3
    assert report["state_variable"] == "velocity"
    assert report["drift_type"] == "damped_residual"
    assert report["damped_operator"] == "spsd"
    assert len(report["stiffness_diagonal"]) == shared_data.n_coordinates
    assert len(report["damping_diagonal"]) == shared_data.n_coordinates
    assert np.asarray(report["stiffness_matrix"]).shape == (shared_data.n_coordinates,) * 2
    assert np.asarray(report["damping_matrix"]).shape == (shared_data.n_coordinates,) * 2
    assert min(report["stiffness_eigenvalues"]) >= -1e-6
    assert min(report["damping_eigenvalues"]) >= -1e-6


def test_capillary_energy_cli_persists_structure_and_uses_standard_rollout(shared_data, tmp_path):
    output = tmp_path / "capillary_energy_sde"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "velocity", "--lag", "1", "--hidden-layers", "8",
          "--drift-type", "capillary_energy", "--surface-tension", ".07",
          "--stiffness-init", ".02", "--damping-init", ".03",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    energy = model.drift_model.energy_matrix.detach()
    assert model.config.drift_type == "capillary_energy"
    assert model.config.surface_tension == .07
    assert torch.count_nonzero(energy[0]) == 0
    assert torch.count_nonzero(energy[:, 0]) == 0
    assert torch.linalg.eigvalsh(energy).min() >= -1e-12
    assert torch.all(model.drift_model.mass_diagonal() > 0)
    assert torch.linalg.eigvalsh(model.drift_model.damping_matrix()).min() >= -1e-6
    training_energy = checkpoint["training"]["capillary_energy"]
    assert training_energy["definition"].startswith("0.5 * q_standardized")
    assert training_energy["surface_tension_N_per_m"] == .07

    rollout_dir = output / "evaluation"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
          "--no-metrics", "--device", "cpu", "--output", str(rollout_dir)])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        assert result["trajectories"].shape == (2, 2, 4, shared_data.n_coordinates)
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["drift_type"] == "capillary_energy"
    assert report["structured_drift_time_unit"] == "one native frame"
    assert np.asarray(report["capillary_energy_matrix"]).shape == (
        shared_data.n_coordinates, shared_data.n_coordinates)
    assert len(report["effective_mass_diagonal"]) == shared_data.n_coordinates
    assert min(report["damping_eigenvalues"]) >= -1e-6
    assert report["mean_mode_stiffness"] > 0


def test_velocity_sparse_history_cli_with_structured_drift(shared_data, tmp_path):
    output = tmp_path / "velocity_sparse_history"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "velocity", "--lag", "1",
          "--history-offsets", "1", "2", "3", "5", "10",
          "--drift-type", "damped_residual", "--damped-operator", "spsd",
          "--diffusion-type", "bounded_state_diagonal", "--diffusion-log-range", ".5",
          "--hidden-layers", "8", "--epochs", "1", "--batch-size", "7",
          "--device", "cpu", "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    assert model.config.history_offsets == (1, 2, 3, 5, 10)
    assert model.drift_model.network[0].in_features == 6 * shared_data.n_coordinates
    assert model.diffusion_model.network[0].in_features == 6 * shared_data.n_coordinates
    assert checkpoint["model_config"]["history_offsets"] == (1, 2, 3, 5, 10)

    rollout_dir = output / "evaluation"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
          "--no-metrics", "--device", "cpu", "--output", str(rollout_dir)])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        assert result["trajectories"].shape == (2, 2, 4, shared_data.n_coordinates)
        assert np.all(result["start_index"] >= 10)
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["history_offsets"] == [1, 2, 3, 5, 10]


def test_bounded_state_diagonal_cli_checkpoint_and_rollout(shared_data, tmp_path):
    output = tmp_path / "bounded_state_diagonal_sde"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--diffusion-type", "bounded_state_diagonal", "--diffusion-log-range", "1.25",
          "--hidden-layers", "8",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    assert model.config.diffusion_type == "bounded_state_diagonal"
    assert model.config.diffusion_log_range == 1.25
    assert checkpoint["model_config"]["diffusion_type"] == "bounded_state_diagonal"
    assert checkpoint["model_config"]["diffusion_log_range"] == 1.25
    assert model.diffusion_model.network[0].in_features == shared_data.n_coordinates

    rollout = output / "rollout"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
          "--diffusion-scale", ".9", "--no-metrics", "--device", "cpu",
          "--output", str(rollout)])
    with np.load(rollout / "rollout.npz") as saved:
        assert saved["trajectories"].shape == (2, 2, 4, shared_data.n_coordinates)
    report = json.loads((rollout / "metrics.json").read_text())
    assert report["diffusion_type"] == "bounded_state_diagonal"
    assert report["diffusion_log_range"] == 1.25
    assert report["diffusion_scale"] == .9


def test_component_freezing_and_optimizer_membership():
    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.01, lag_steps=2,
                                hidden_layers=(4,), diffusion_init=.3))
    drift = list(model.drift_model.parameters())
    diffusion = list(model.diffusion_model.parameters())
    trainable = set_trainable_components(model, "diffusion")
    assert all(not parameter.requires_grad for parameter in drift)
    assert all(parameter.requires_grad for parameter in diffusion)
    assert {id(parameter) for parameter in trainable} == {id(parameter) for parameter in diffusion}
    current = torch.randn(6, 2)
    following = current + .1 * torch.randn(6, 2)
    optimizer = torch.optim.Adam(trainable, lr=.01)
    optimizer.zero_grad()
    EulerMaruyamaNLL()(model, current, following, zero_drift=True).backward()
    assert all(parameter.grad is None for parameter in drift)
    assert all(parameter.grad is not None for parameter in diffusion)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    frozen_diffusion = [parameter.detach().clone() for parameter in diffusion]
    trainable = set_trainable_components(model, "drift")
    assert {id(parameter) for parameter in trainable} == {id(parameter) for parameter in drift}
    optimizer = torch.optim.Adam(trainable, lr=.01)
    optimizer.zero_grad()
    EulerMaruyamaNLL()(model, current, following).backward()
    assert all(parameter.grad is None for parameter in diffusion)
    optimizer.step()
    for before, after in zip(frozen_diffusion, diffusion):
        torch.testing.assert_close(before, after)


def test_diffusion_validation_metric_uses_trimmed_full_covariance():
    rng = np.random.default_rng(6)
    increments = rng.normal(size=(100, 2))
    increments[5] = [20., -15.]

    class Data:
        n_coordinates = 2
        native_dt = .01

        def iter_transitions(self, split, *, lag_steps, history_steps, batch_size):
            assert split == "validation" and lag_steps == 2 and history_steps == 0
            for start in range(0, len(increments), batch_size):
                delta = increments[start:start + batch_size]
                yield "rep", dict(current_state=np.zeros_like(delta), next_state=delta)

    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.01, lag_steps=2,
                                hidden_layers=(4,), diffusion_init=.3))
    model.state_std.copy_(torch.tensor([2., 4.]))
    data = Data()
    q, counts = compute_trimmed_increment_covariance(data, model, 5, 17)
    normalized = increments / [2., 4.]
    untrimmed, untrimmed_counts = compute_trimmed_increment_covariance(data, model, 0, 17)
    np.testing.assert_allclose(untrimmed, normalized.T @ normalized / (len(normalized) * .02))
    assert untrimmed_counts["diffusion_trim_removed_count"] == 0
    assert untrimmed_counts["diffusion_trim_removed_l2_energy_fraction"] == 0
    norms = np.linalg.norm(normalized, axis=1)
    keep = norms <= np.percentile(norms, 95)
    expected = normalized[keep].T @ normalized[keep] / (keep.sum() * .02)
    np.testing.assert_allclose(q, expected)
    assert counts["diffusion_trim_removed_count"] == (~keep).sum()
    assert counts["diffusion_trim_removed_l2_energy_fraction"] == pytest.approx(
        np.sum(norms[~keep] ** 2) / np.sum(norms ** 2))
    report = compute_diffusion_validation_metric(model, data, 5, 17)
    sigma = model.diffusion_model().detach().numpy()
    a_model = np.diag(sigma**2 / .01)
    assert report["diffusion_covariance_relative_error"] == pytest.approx(
        np.linalg.norm(a_model - expected) / np.linalg.norm(expected))
    assert report["diffusion_model_trace"] == pytest.approx(np.trace(a_model))
    assert report["diffusion_empirical_trace"] == pytest.approx(np.trace(expected))

    state_dependent = NeuralSDE(SDEConfig(
        reduced_dim=2, dt=.01, lag_steps=2, hidden_layers=(4,),
        diffusion_init=.3, diffusion_type="state_diagonal"))
    state_dependent.state_std.copy_(model.state_std)
    conditional_report = compute_diffusion_validation_metric(
        state_dependent, data, 5, 17)
    factor = state_dependent.normalized_diffusion_matrix(torch.zeros(1, 2))[0].detach().numpy()
    assert conditional_report["diffusion_model_trace"] == pytest.approx(
        np.trace(factor @ factor.T / .01))
    assert conditional_report["diffusion_empirical_trace"] == pytest.approx(np.trace(expected))
    physical_norms = np.linalg.norm(increments, axis=1)
    physical_keep = physical_norms <= np.percentile(physical_norms, 95)
    physical_q = (increments[physical_keep].T @ increments[physical_keep]
                  / (physical_keep.sum() * .02))
    physical_a = np.diag((model.state_std.detach().numpy() * sigma)**2 / .01)
    physical_report = compute_diffusion_validation_metric(model, data, 5, 17, "physical")
    assert physical_report["diffusion_covariance_relative_error"] == pytest.approx(
        np.linalg.norm(physical_a - physical_q) / np.linalg.norm(physical_q))


def test_training_trim_excludes_large_training_and_validation_pairs():
    increments = np.ones((100, 2), dtype=np.float64)
    increments[-1] = [100., 100.]

    class Data:
        n_coordinates = 2

        def iter_transitions(self, split, *, batch_size, **_):
            assert split in {"train", "validation"}
            for start in range(0, len(increments), batch_size):
                delta = increments[start:start + batch_size]
                yield "rep", dict(current_state=np.zeros_like(delta), next_state=delta)

    class MeanIncrement(torch.nn.Module):
        def forward(self, model, current, following, *, zero_drift=False):
            return (following[:, 0] - current[:, 0]).mean()

    data = Data()
    model = NeuralSDE(SDEConfig(reduced_dim=2, dt=.01, hidden_layers=(4,)))
    cutoff, report = compute_training_trim_cutoff(data, model, 1., 17)
    assert report["train_trim_removed_count"] == 1
    assert report["train_trim_increment_count"] == 100
    trimmed = _epoch(model, data, "train", 17, "cpu", MeanIncrement(),
                     train_trim_cutoff=cutoff)
    validation = _epoch(model, data, "validation", 17, "cpu", MeanIncrement(),
                        train_trim_cutoff=cutoff)
    assert trimmed == pytest.approx(1.)
    assert validation == pytest.approx(1.)


def test_training_trim_refits_normalization_with_frozen_selection_scale():
    current = np.arange(100, dtype=np.float64)[:, None]
    increments = np.linspace(1., 2., 100)[:, None]
    increments[-1] = 100.

    class Data:
        n_coordinates = 1

        def iter_transitions(self, split, *, batch_size, **_):
            assert split in {"train", "validation"}
            for start in range(0, len(current), batch_size):
                stop = start + batch_size
                yield "rep", dict(current_state=current[start:stop],
                                  next_state=current[start:stop] + increments[start:stop])

    data = Data()
    model = NeuralSDE(SDEConfig(reduced_dim=1, dt=.01, hidden_layers=(4,)))
    selection_scale = model.state_std.detach().clone()
    cutoff, report = compute_training_trim_cutoff(data, model, 1., 17)
    stats, retained = _fit_trimmed_sde_normalization(
        data, model, cutoff, selection_scale, 17)
    assert report["train_trim_removed_count"] == 1
    assert retained == 99
    torch.testing.assert_close(stats.state_mean, torch.tensor(current[:-1].mean(0)).float())
    torch.testing.assert_close(stats.state_std, torch.tensor(current[:-1].std(0)).float())
    torch.testing.assert_close(stats.target_mean, torch.tensor(increments[:-1].mean(0)).float())
    torch.testing.assert_close(stats.target_std, torch.tensor(increments[:-1].std(0)).float())

    # Even after the model scale changes, the frozen selection scale excludes
    # exactly the pair that defined the original percentile mask.
    model.state_std.fill_(100.)

    class MeanIncrement(torch.nn.Module):
        def forward(self, model, current_state, following, *, zero_drift=False):
            return (following - current_state).mean()

    value = _epoch(model, data, "train", 17, "cpu", MeanIncrement(),
                   train_trim_cutoff=cutoff,
                   train_trim_state_std=selection_scale)
    assert value == pytest.approx(float(increments[:-1].mean()))


def test_training_cutoff_rejects_whole_reference_windows_with_large_increments():
    values = np.array([[0.], [0.], [0.], [100.], [100.], [100.], [100.], [100.],
                       [100.], [100.]])

    class Data:
        n_coordinates = 1
        splits = {"test": ["rep"]}
        frame_counts = {"rep": len(values)}

        @staticmethod
        def read_coordinates(name, start, stop):
            assert name == "rep"
            return np.arange(start, stop, dtype=float), values[start:stop]

    report = {}
    reference, _, _, _, starts = reference_windows(
        Data(), SDEConfig(1, dt=.01), "test", 50, horizon=2, seed=4,
        increment_norm_cutoff=1., state_std=np.ones(1), filter_report=report)
    assert not np.isin(starts, [1, 2]).any()
    assert np.all(np.abs(np.diff(reference.numpy(), axis=1)) <= 1)
    assert report["candidate_windows_before_trim"] == 8
    assert report["candidate_windows_after_trim"] == 6
    assert report["generated_trajectories_trimmed"] is False


def test_train_and_evaluate_reuse_shared_data_and_rollout_format(shared_data, tmp_path):
    # The compact EDM fixture omits metadata needed only by downstream plotters.
    with h5py.File(shared_data.basis_path, "r+") as basis:
        for name in basis["repetitions"]:
            basis[f"repetitions/{name}"].attrs["signal_units"] = "microns"
    output = tmp_path / "sde"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--hidden-layers", "8", "--diffusion-init", ".3", "--diffusion-type", "full",
          "--epochs", "1",
          "--train-trim-percent", "1",
          "--batch-size", "7", "--device", "cpu", "--output", str(output)])
    checkpoint_path = output / "best.pt"
    assert checkpoint_path.is_file()
    model, checkpoint = load_model(checkpoint_path)
    assert checkpoint["data_config"] == shared_data.source_config
    assert checkpoint["data_signature"] == shared_data.signature()
    assert checkpoint["training"]["train_trim_percent"] == 1
    assert checkpoint["training"]["train_trim_removed_count"] > 0
    assert checkpoint["training"]["validation_trim_increment_count"] > 0
    assert checkpoint["training"]["normalization_fit"] == "retained_training_transitions"
    assert checkpoint["training"]["normalization_retained_transition_count"] > 0
    assert len(checkpoint["training"]["train_trim_state_std"]) == shared_data.n_coordinates
    assert checkpoint["model_config"]["diffusion_type"] == "full"
    assert model.config.diffusion_type == "full"
    assert model.diffusion_matrix().shape == (shared_data.n_coordinates,
                                               shared_data.n_coordinates)
    assert model.state_std.shape == (shared_data.n_coordinates,)
    assert math.isfinite(json.loads((output / "metrics.jsonl").read_text())["validation_nll"])

    rollout_dir = output / "evaluation"
    main(["evaluate", "--checkpoint", str(checkpoint_path), "--split", "test",
          "--num-conditions", "3", "--ensemble-size", "2", "--horizon", "4",
          "--enforce-train-trim-on-reference-windows",
          "--device", "cpu", "--output", str(rollout_dir)])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        assert result["trajectories"].shape == (3, 2, 5, shared_data.n_coordinates)
        assert result["reference"].shape == (3, 5, shared_data.n_coordinates)
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["model_type"] == "neural_sde"
    assert "one_step" in report and "rollout" in report
    assert report["rank"] == shared_data.rank
    assert report["source_shared_basis"] == str(shared_data.basis_path)
    assert report["reference_window_filter"]["generated_trajectories_trimmed"] is False
    assert report["reference_window_filter"]["candidate_windows_after_trim"] > 0
    assert report["reference_window_trim_policy"]["enforced"] is True
    assert report["reference_window_trim_policy"]["normalization_scale_source"] == "saved_pretrim_state_std"
    saved_rollout = Rollout(rollout_dir)
    assert saved_rollout.generated.shape == (3, 2, 5, shared_data.n_coordinates)
    assert saved_rollout.provenance()["reference_window_filter"] == report["reference_window_filter"]

    unfiltered_rollout_dir = output / "evaluation_unfiltered"
    main(["evaluate", "--checkpoint", str(checkpoint_path), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "1", "--horizon", "4",
          "--no-metrics", "--device", "cpu", "--output", str(unfiltered_rollout_dir)])
    unfiltered_report = json.loads((unfiltered_rollout_dir / "metrics.json").read_text())
    assert unfiltered_report["reference_window_trim_policy"]["enforced"] is False
    assert unfiltered_report["reference_window_trim_policy"]["caution"] is not None
    assert "reference_window_filter" not in unfiltered_report


def test_highpass_training_argument_is_saved_and_reused_for_evaluation(
        highpass_shared_data, tmp_path):
    data = highpass_shared_data
    output = tmp_path / "sde_highpass"
    main(["train", "--experiment", data.experiment, "--rank", str(data.rank),
          "--data-root", data.source_config["data_root"],
          "--spatial-mean-highpass-hz", "20",
          "--hidden-layers", "8", "--epochs", "1", "--batch-size", "7",
          "--device", "cpu", "--output", str(output)])
    _, checkpoint = load_model(output / "best.pt")
    assert checkpoint["data_config"]["spatial_mean_highpass_hz"] == 20.0
    assert checkpoint["data_signature"]["spatial_mean_dataset"].endswith(
        "cutoff_20_hz"
    )

    rollout_dir = output / "evaluation"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "1", "--horizon", "3",
          "--no-metrics", "--device", "cpu", "--output", str(rollout_dir)])
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["spatial_mean_highpass_hz"] == 20.0
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        for condition, (name, start) in enumerate(
                zip(result["repetition"].astype(str), result["start_index"])):
            with h5py.File(data.source_paths[name]) as source:
                expected = np.sqrt(data.n_space) * source[data.spatial_mean_dataset][start]
            assert result["reference"][condition, 0, 0] == pytest.approx(expected)


def test_history_training_and_evaluation_keep_physical_rollout_shape(shared_data, tmp_path):
    output = tmp_path / "history2"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--hidden-layers", "8", "--lag", "2", "--history", "2",
          "--train-trim-percent", "1", "--epochs", "1", "--batch-size", "7",
          "--device", "cpu", "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    assert model.config.history_steps == 2
    assert model.config.history_conditioning
    assert model.drift_model.network[0].in_features == 3 * shared_data.n_coordinates
    assert checkpoint["training"]["train_trim_increment_count"] == 30
    legacy = dict(checkpoint["model_config"], history_conditioning=False)
    legacy.pop("history_steps")
    assert SDEConfig.from_dict(legacy).history_steps == 0

    rollout_dir = output / "evaluation"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "3", "--horizon", "4",
          "--no-metrics", "--device", "cpu", "--output", str(rollout_dir)])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        assert result["trajectories"].shape == (2, 3, 5, shared_data.n_coordinates)
        assert np.all(result["start_index"] >= 4)
        np.testing.assert_allclose(result["trajectories"][:, :, 0],
                                   np.broadcast_to(result["reference"][:, None, 0],
                                                   (2, 3, shared_data.n_coordinates)))
    assert json.loads((rollout_dir / "metrics.json").read_text())["history_steps"] == 2


def test_sparse_history_cli_checkpoint_and_evaluation(shared_data, tmp_path):
    output = tmp_path / "sparse_history"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--state-variable", "increment", "--history-offsets", "1", "2", "3", "5", "10",
          "--hidden-layers", "8", "--epochs", "1", "--batch-size", "7",
          "--device", "cpu", "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    assert model.config.history_steps == 10
    assert model.config.history_offsets == (1, 2, 3, 5, 10)
    assert model.drift_model.network[0].in_features == 6 * shared_data.n_coordinates
    assert checkpoint["model_config"]["history_offsets"] == (1, 2, 3, 5, 10)

    rollout_dir = output / "evaluation"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
          "--no-metrics", "--device", "cpu", "--output", str(rollout_dir)])
    with np.load(rollout_dir / "rollout.npz", allow_pickle=False) as result:
        assert result["trajectories"].shape == (2, 2, 4, shared_data.n_coordinates)
        assert np.all(result["start_index"] >= 10)
    report = json.loads((rollout_dir / "metrics.json").read_text())
    assert report["history_steps"] == 10
    assert report["history_offsets"] == [1, 2, 3, 5, 10]


def test_two_stage_training_preserves_diffusion_and_uses_larger_lag(shared_data, tmp_path):
    stage1, stage2 = tmp_path / "diffusion", tmp_path / "drift"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"], "--hidden-layers", "8",
          "--train-mode", "diffusion", "--lag", "1", "--diffusion-trim-percent", "5",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu", "--output", str(stage1)])
    diffusion_model, diffusion_checkpoint = load_model(stage1 / "best.pt")
    torch.testing.assert_close(diffusion_model.drift(torch.zeros(1, shared_data.n_coordinates)),
                               torch.zeros(1, shared_data.n_coordinates))
    first_log = json.loads((stage1 / "metrics.jsonl").read_text())
    for key in ("diffusion_covariance_relative_error", "diffusion_model_trace",
                "diffusion_empirical_trace", "diffusion_trim_removed_count",
                "diffusion_trim_removed_percent", "diffusion_trim_removed_l2_energy_fraction"):
        assert key in first_log
    assert diffusion_checkpoint["training"]["train_mode"] == "diffusion"

    main(["train", "--train-mode", "drift", "--lag", "2", "--checkpoint", str(stage1 / "best.pt"),
          "--epochs", "1", "--batch-size", "7", "--device", "cpu", "--output", str(stage2)])
    drift_model, drift_checkpoint = load_model(stage2 / "best.pt")
    assert drift_model.config.lag_steps == 2
    torch.testing.assert_close(drift_model.diffusion_model.raw_diagonal,
                               diffusion_model.diffusion_model.raw_diagonal)
    torch.testing.assert_close(drift_model.state_std, diffusion_model.state_std)
    assert not torch.allclose(drift_model.drift_model.network[-1].weight,
                              diffusion_model.drift_model.network[-1].weight)
    assert drift_checkpoint["training"]["train_mode"] == "drift"
    assert "diffusion_covariance_relative_error" not in json.loads((stage2 / "metrics.jsonl").read_text())

    rollout = stage2 / "rollout"
    main(["evaluate", "--checkpoint", str(stage2 / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "4",
          "--no-metrics", "--device", "cpu", "--output", str(rollout)])
    report = json.loads((rollout / "metrics.json").read_text())
    assert report["lag_steps"] == 2
    assert report["training_lag_steps"] == 2
    assert report["physical_lag"] == pytest.approx(2 * shared_data.native_dt)

    fine_rollout = stage2 / "rollout_lag1"
    main(["evaluate", "--checkpoint", str(stage2 / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "4",
          "--rollout-lag", "1", "--no-metrics", "--device", "cpu",
          "--output", str(fine_rollout)])
    fine_report = json.loads((fine_rollout / "metrics.json").read_text())
    assert fine_report["training_lag_steps"] == 2
    assert fine_report["lag_steps"] == 1
    assert fine_report["physical_lag"] == pytest.approx(shared_data.native_dt)
    with np.load(fine_rollout / "rollout.npz") as saved:
        np.testing.assert_allclose(np.diff(saved["reference_time"], axis=1), shared_data.native_dt)
    scored_rollout = stage2 / "scored_lag1"
    main(["evaluate", "--checkpoint", str(stage2 / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "4",
          "--rollout-lag", "1", "--device", "cpu", "--output", str(scored_rollout)])
    assert "one_step" in json.loads((scored_rollout / "metrics.json").read_text())
    assert load_model(stage2 / "best.pt")[0].config.lag_steps == 2


def test_drift_only_training_starts_without_checkpoint(shared_data, tmp_path):
    output = tmp_path / "drift_from_scratch"
    config = SDEConfig(shared_data.n_coordinates, dt=shared_data.native_dt,
                       hidden_layers=(8,), diffusion_init=.3, lag_steps=3)
    torch.manual_seed(0)
    untrained = NeuralSDE(config)
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"],
          "--train-mode", "drift", "--lag", "3", "--hidden-layers", "8",
          "--diffusion-init", ".3", "--train-trim-percent", "1",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(output)])
    trained, checkpoint = load_model(output / "best.pt")
    assert checkpoint["training"]["train_mode"] == "drift"
    assert checkpoint["training"]["initial_checkpoint"] is None
    assert checkpoint["training"]["train_trim_removed_count"] > 0
    torch.testing.assert_close(trained.diffusion_model.raw_diagonal,
                               untrained.diffusion_model.raw_diagonal)
    assert not torch.allclose(trained.drift_model.network[-1].weight,
                              untrained.drift_model.network[-1].weight)


def test_joint_checkpoint_warm_starts_both_components_at_larger_lag(shared_data, tmp_path):
    first, same, second = tmp_path / "lag2", tmp_path / "lag2_finetune", tmp_path / "lag3"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"], "--hidden-layers", "8",
          "--lag", "2", "--train-trim-percent", "1",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(first)])
    before, first_checkpoint = load_model(first / "best.pt")
    main(["train", "--checkpoint", str(first), "--lag", "2",
          "--train-trim-percent", "1",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(same)])
    same_lag, same_checkpoint = load_model(same / "best.pt")
    assert same_lag.config.lag_steps == 2
    assert same_checkpoint["training"]["initial_checkpoint"] == str(
        (first / "best.pt").resolve())
    assert same_checkpoint["training"]["train_trim_rule_source"] == "checkpoint"
    assert (same_checkpoint["training"]["train_trim_norm_cutoff"]
            == first_checkpoint["training"]["train_trim_norm_cutoff"])
    assert (same_checkpoint["training"]["train_trim_state_std"]
            == first_checkpoint["training"]["train_trim_state_std"])
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--checkpoint", str(first), "--lag", "3", "--train-trim-percent", "1",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu",
          "--output", str(second)])
    after, checkpoint = load_model(second / "best.pt")
    assert after.config.lag_steps == 3
    assert checkpoint["training"]["train_mode"] == "joint"
    assert checkpoint["training"]["initial_checkpoint"] == str((first / "best.pt").resolve())
    assert checkpoint["training"]["train_trim_removed_count"] > 0
    assert checkpoint["data_signature"] == shared_data.signature()
    assert checkpoint["training"]["train_trim_rule_source"] == "checkpoint"
    assert (checkpoint["training"]["train_trim_norm_cutoff"]
            == first_checkpoint["training"]["train_trim_norm_cutoff"])
    assert (checkpoint["training"]["train_trim_state_std"]
            == first_checkpoint["training"]["train_trim_state_std"])
    assert checkpoint["training"]["normalization_fit"] == "retained_training_transitions"
    assert not torch.allclose(after.drift_model.network[-1].weight,
                              before.drift_model.network[-1].weight)
    assert not torch.allclose(after.diffusion_model.raw_diagonal,
                              before.diffusion_model.raw_diagonal)


def test_mismatched_dt_is_rejected(shared_data, tmp_path):
    config = SDEConfig(shared_data.n_coordinates, dt=2 * shared_data.native_dt)
    with pytest.raises(ValueError, match="dt must match"):
        train(shared_data, tmp_path / "bad", config, epochs=1)
