"""Displacement mixture mathematics and complete-window trim integration."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.losses import (EulerMaruyamaNLL, MonteCarloMultistepNLL,
                                         MonteCarloMultistepVelocityNLL)
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import _epoch, _sde_transition_batches


@pytest.mark.parametrize("lag", [1, 3])
def test_displacement_h1_density_jacobian_and_no_particle_draw(lag):
    model = NeuralSDE(SDEConfig(2, .01, lag_steps=lag, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,),
                                diffusion_type="full", diffusion_init=.3))
    with torch.no_grad():
        model.target_std.copy_(torch.tensor([2., .5]))
        model.diffusion_model.lower_entries.fill_(.13)
    current = torch.tensor([[1., 2.], [3., 4.]])
    history = current[:, None]-torch.tensor([.2, -.1])
    velocity_target = torch.tensor([[[.4, -.3]], [[.1, .2]]])
    generator = torch.Generator().manual_seed(11)
    before = generator.get_state().clone()
    vloss = MonteCarloMultistepVelocityNLL((1,), particles=17)(
        model, current, target_velocities=velocity_target, history_states=history)
    dloss = MonteCarloMultistepNLL((1,), particles=17, target="displacement")(
        model, current, lag*velocity_target, history_states=history, generator=generator)
    torch.testing.assert_close(dloss, vloss + 2*math.log(lag))
    torch.testing.assert_close(generator.get_state(), before)
    if lag == 1:
        expected = EulerMaruyamaNLL()(model, current, current+velocity_target[:, 0],
                                      history_states=history)
        torch.testing.assert_close(dloss, expected)


@pytest.mark.parametrize("lag", [1, 3])
def test_displacement_joint_gmm_matches_independent_two_step_calculation(lag):
    model = NeuralSDE(SDEConfig(2, .01, lag_steps=lag, history_steps=1,
                                state_variable="velocity", hidden_layers=(4,),
                                diffusion_type="full", diffusion_init=.3)).double()
    model.drift_model.zero_output()
    with torch.no_grad():
        model.diffusion_model.lower_entries.fill_(.2)
        model.target_std.copy_(torch.tensor([2., .7]))
    current = torch.tensor([[100., -70.], [50., 80.]], dtype=torch.float64)
    incoming = torch.tensor([[.4, -.2], [-.3, .5]], dtype=torch.float64)
    history = (current-incoming)[:, None]
    target = torch.tensor([[2., -.8], [-1.5, 2.1]], dtype=torch.float64)
    particles, seed = 13, 72
    actual = MonteCarloMultistepNLL((2,), particles, target="displacement")(
        model, current, target[:, None], history_states=history,
        generator=torch.Generator().manual_seed(seed))
    # Zero drift: v1 = v0+e1, delta2 = 2*h*v1+h*e2.
    factor = math.sqrt(lag)*model.diffusion_matrix()
    epsilon = torch.randn(2, particles, 2, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(seed)) @ factor.T
    component_mean = 2*lag*(incoming[:, None]+epsilon)
    normal = torch.distributions.MultivariateNormal(component_mean, scale_tril=lag*factor)
    logp = normal.log_prob(target[:, None])
    expected = -(torch.logsumexp(logp, dim=1)-math.log(particles)).mean()
    torch.testing.assert_close(actual, expected)
    # This is a joint mixture, not a mean of individual-path losses.
    assert not torch.isclose(actual, -logp.mean())


def test_displacement_integrated_ou_matches_analytic_marginal():
    theta, sigma, horizon = .18, .32, 4

    class OUDrift(torch.nn.Module):
        def forward(self, features):
            return -theta*features[..., 1:]

    model = NeuralSDE(SDEConfig(1, .01, history_steps=1, state_variable="velocity",
                                hidden_layers=(4,), diffusion_init=sigma))
    model.drift_model = OUDrift()
    incoming = torch.tensor([-.7, -.2, .1, .5, .9])[:, None]
    current = torch.zeros_like(incoming)
    history = -incoming[:, None]
    coefficient = 1-theta
    mean = sum(coefficient**j for j in range(1, horizon+1))*incoming
    variance = sigma**2*sum(sum(coefficient**j for j in range(horizon-i+1))**2
                           for i in range(1, horizon+1))
    target = mean+torch.tensor([-.2, .1, 0., .15, -.1])[:, None]
    analytic = -torch.distributions.Normal(mean, math.sqrt(variance)).log_prob(target).mean()
    estimate = MonteCarloMultistepNLL((horizon,), 50000, target="displacement")(
        model, current, target[:, None], history_states=history,
        generator=torch.Generator().manual_seed(73))
    assert abs(estimate-analytic) < .02


@pytest.mark.parametrize("target", ["velocity", "displacement"])
@pytest.mark.parametrize("cap", [None, 2])
def test_multistep_trim_rejects_interior_history_and_endpoint_jumps(target, cap):
    # Samples 1, 2, 3 contain an interior, prehistory, or final jump.
    # The interior-jump sample returns to its original path before the endpoint.
    path = np.broadcast_to(np.arange(6)[None, :, None]*.1, (5, 6, 2)).copy()
    path += np.arange(5)[:, None, None]*10
    path[1, 3, 0] += 100
    path[2, 0, 1] -= 100
    path[3, -1, 0] += 100

    def transitions(*args, **kwargs):
        yield "synthetic", dict(current_state=path[:, 1].copy(), next_state=path[:, -1].copy(),
                                history_states=path[:, :1].copy(),
                                next_history_states=path[:, -2:-1].copy(), state_path=path.copy())

    data = SimpleNamespace(batch_size=5, n_coordinates=2, iter_transitions=transitions)
    model = NeuralSDE(SDEConfig(2, .01, history_steps=1, state_variable="velocity", hidden_layers=(4,)))
    seen = []

    class CheckedNLL(MonteCarloMultistepNLL):
        def per_horizon(self, model, current, targets, **kwargs):
            seen.append(current.clone())
            expected = torch.full_like(targets, .4 if self.target == "displacement" else .1)
            torch.testing.assert_close(targets, expected)
            return super().per_horizon(model, current, targets, **kwargs)

    # Different M and batch sizes must still score the identical retained states.
    selections = []
    for particles, batch_size in ((3, 2), (11, 7)):
        seen.clear()
        metrics = {}
        _epoch(model, data, "validation", batch_size, "cpu", EulerMaruyamaNLL(),
               train_trim_cutoff=1., train_trim_state_std=torch.ones(2),
               multistep_objective=CheckedNLL((4,), particles, target=target),
               multistep_weight=.1, multistep_noise_seed=3, metric_output=metrics,
               multistep_max_windows=cap, multistep_window_seed=5)
        assert metrics["window_count"] == 2
        selections.append(torch.cat(seen))
    expected = torch.tensor(path[[0, 4], 1], dtype=torch.float32)
    torch.testing.assert_close(selections[0], expected)
    torch.testing.assert_close(selections[1], expected)


def test_displacement_target_uses_true_lagged_position_change(shared_data):
    config = SDEConfig(shared_data.n_coordinates, shared_data.native_dt, lag_steps=3,
                       history_steps=1, state_variable="velocity", hidden_layers=(4,))
    batch = next(_sde_transition_batches(shared_data, config, "train", 100,
                                         multistep_horizons=(1, 2)))
    path = batch['state_path']
    expected = path[:, [4, 7]]-path[:, 1, None]
    torch.testing.assert_close(batch['target_displacements'], expected, rtol=5e-4, atol=5e-5)
    assert not torch.allclose(batch['target_displacements'], 3*batch['target_velocities'])


def test_zero_multistep_weight_ignores_target_and_preserves_random_state(shared_data):
    config = SDEConfig(shared_data.n_coordinates, shared_data.native_dt, history_steps=1,
                       state_variable="velocity", hidden_layers=(4,))
    model = NeuralSDE(config)
    before = torch.get_rng_state().clone()
    baseline = _epoch(model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL())
    actual = _epoch(model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL(),
                    multistep_weight=0.,
                    multistep_objective=MonteCarloMultistepNLL((10000,), target="displacement"))
    assert actual == baseline
    torch.testing.assert_close(torch.get_rng_state(), before)
