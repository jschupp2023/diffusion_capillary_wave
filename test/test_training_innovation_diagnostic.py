import numpy as np
import torch

from data_analysis.correlations.training_innovation_diagnostic import (
    eligible_velocity_transitions, transition_innovations)
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE


def test_velocity_transition_trim_requires_complete_stencil():
    edge_good = np.array([True, True, False, True, True, True])
    np.testing.assert_array_equal(
        eligible_velocity_transitions(edge_good, history_steps=1),
        [True, False, False, True, True])
    np.testing.assert_array_equal(
        eligible_velocity_transitions(edge_good, history_steps=2),
        [False, False, False, True])


def test_joint_whitening_matches_known_velocity_innovations():
    model = NeuralSDE(SDEConfig(
        reduced_dim=2, dt=1., state_variable="velocity", lag_steps=1,
        history_steps=1, hidden_layers=(4,), diffusion_type="full",
        diffusion_init=.5))
    model.drift_model.zero_output()
    velocities = np.array([
        [0., 0.], [.5, -1.], [1., -.5], [.5, 0.], [0., .5]], dtype=np.float32)
    states = np.vstack((np.zeros((1, 2), dtype=np.float32), np.cumsum(velocities, axis=0)))
    marginal, whitened, eligible = transition_innovations(
        model, states, np.ones(len(states) - 1, dtype=bool),
        batch_size=8, device=torch.device("cpu"))
    expected = np.diff(velocities, axis=0) / .5
    np.testing.assert_array_equal(eligible, np.ones(len(expected), dtype=bool))
    np.testing.assert_allclose(marginal[0], expected, atol=2e-5)
    np.testing.assert_allclose(whitened[0], expected, atol=2e-5)
