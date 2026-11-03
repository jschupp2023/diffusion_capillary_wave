"""Optional temporal drift replacement: indexing, objectives, and saved workflow."""
import json

import numpy as np
import pytest
import torch

from data_analysis.correlations.training_innovation_diagnostic import transition_innovations
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.experiments.adaptive_controller import NEW_MODEL_ONLY, configuration_summary
from modelling.neural_sde.losses import EulerMaruyamaNLL, MonteCarloMultistepNLL, StationaryMeanLoss
from modelling.neural_sde.model import MLPDrift, NeuralSDE, TCNDrift
from modelling.neural_sde.stability import DiscreteLyapunovRegularizer, StabilityConfig
from modelling.neural_sde.train import (
    _damping_residual_orthogonality, _diagonal_linear_residual_orthogonality,
    _restoring_residual_orthogonality, _sde_transition_batches, _trim_keep,
    load_model, main,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**kwargs):
    options = dict(reduced_dim=2, dt=.01, state_variable="velocity", history_steps=1,
                   residual_architecture="tcn", tcn_context_frames=7,
                   tcn_channels=4, tcn_kernel_size=3, tcn_dilations=(1, 2),
                   hidden_layers=(4,))
    options.update(kwargs)
    return SDEConfig(**options)


def test_config_roundtrip_and_legacy_default():
    cfg = config()
    assert cfg.history_steps == cfg.tcn_receptive_field == 7
    assert cfg.effective_history_offsets == (1,)
    assert SDEConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    sparse = config(history_offsets=(1, 2, 10), history_steps=10)
    assert sparse.history_steps == 10
    assert sparse.effective_history_offsets == (1, 2, 10)
    assert SDEConfig.from_dict(sparse.to_dict()) == sparse
    legacy = dict(reduced_dim=2, dt=.01, hidden_layers=[4], state_variable="velocity",
                  history_steps=1)
    old = NeuralSDE(SDEConfig.from_dict(legacy))
    assert isinstance(old.drift_model, MLPDrift)
    copy = NeuralSDE(SDEConfig.from_dict(legacy))
    copy.load_state_dict(old.state_dict(), strict=True)
    assert old.config.residual_architecture == "mlp"


@pytest.mark.parametrize("options,match", [
    ({"state_variable": "state"}, "native-lag"),
    ({"lag_steps": 2}, "native-lag"),
    ({"drift_type": "capillary_energy"}, "supports"),
    ({"velocity_memory_offsets": (1,)}, "additive memory"),
    ({"tcn_context_frames": 8}, "receptive field"),
    ({"tcn_channels": 0}, "tcn_channels"),
    ({"tcn_dilations": ()}, "tcn_dilations"),
    ({"tcn_dilations": (True,)}, "tcn_dilations"),
    ({"tcn_kernel_size": 0}, "tcn_kernel_size"),
])
def test_invalid_tcn_configuration(options, match):
    with pytest.raises(ValueError, match=match):
        config(**options)


def test_chronological_positions_and_native_velocities():
    model = NeuralSDE(config(reduced_dim=1, tcn_context_frames=3))
    model.state_mean.fill_(2.)
    model.state_std.fill_(2.)
    model.target_mean.fill_(1.)
    model.target_std.fill_(3.)
    # Chronological positions: 1, 3, 6, 10; history is supplied newest first.
    state = torch.tensor([[10.]])
    history = torch.tensor([[[6.], [3.], [1.]]])
    expected = torch.tensor([[[.5, 1/3], [2., 2/3], [4., 1.]]])
    torch.testing.assert_close(model._tcn_input(state, history), expected)
    overridden = model._tcn_input(state, history, torch.tensor([[7.]]))
    torch.testing.assert_close(overridden[..., :-1, :], expected[..., :-1, :])
    assert overridden[0, -1, 1] == 2.
    with pytest.raises(ValueError, match="preceding states"):
        model.drift(state, history[:, :2])
    expanded = model._tcn_input(state[:, None].expand(-1, 5, -1),
                                history[:, None].expand(-1, 5, -1, -1))
    torch.testing.assert_close(expanded, expected[:, None].expand(-1, 5, -1, -1))


def test_tcn_is_causal_and_reaches_oldest_token():
    network = TCNDrift(2, 4, 3, (1, 2), "tanh")
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.fill_(.05)
    sequence = torch.zeros(2, 7, 4, requires_grad=True)
    modified = sequence.detach().clone()
    modified[:, 4:] = 2.
    torch.testing.assert_close(network.sequence_output(sequence)[:, :4],
                               network.sequence_output(modified)[:, :4])
    network(sequence).sum().backward()
    assert torch.all(sequence.grad.abs().sum(-1) > 0)
    particle_sequence = sequence.detach()[:, None].expand(-1, 3, -1, -1)
    torch.testing.assert_close(network(particle_sequence),
                               network(sequence.detach())[:, None].expand(-1, 3, -1))
    assert network(sequence[0]).shape == (2,)


@pytest.mark.parametrize("drift_type,structured_damping", [
    ("mlp", True), ("damped_residual", True), ("damped_residual", False),
])
def test_replacement_structure_gradients_and_objectives(drift_type, structured_damping):
    torch.manual_seed(12)
    model = NeuralSDE(config(drift_type=drift_type, structured_damping=structured_damping,
                             diffusion_type="full"))
    assert not any(isinstance(module, MLPDrift) for module in model.drift_model.modules())
    network = (model.drift_model if drift_type == "mlp" else model.drift_model.residual)
    assert isinstance(network, TCNDrift)
    state, history, following = torch.randn(5, 2), torch.randn(5, 7, 2), torch.randn(5, 2)
    velocity = state - history[:, 0]
    if drift_type == "damped_residual":
        features = model._drift_input(state, history)
        linear = (-features[:, :2] @ model.drift_model.stiffness_matrix()
                  - features[:, 2:4] @ model.drift_model.damping_matrix())
        # Structured initialization keeps the neural term zero, like the MLP.
        torch.testing.assert_close(model.drift(state, history), linear)
        with torch.no_grad():
            network.network[-1].weight.fill_(.1)
        torch.testing.assert_close(model.drift(state, history),
                                   linear + model.nonlinear_drift(state, history))
        for regularizer in (_restoring_residual_orthogonality,
                            *([_damping_residual_orthogonality,
                               _diagonal_linear_residual_orthogonality]
                              if structured_damping else [])):
            assert torch.isfinite(regularizer(model, state, history, 1.e-6)[0])
    else:
        torch.testing.assert_close(model.drift(state, history),
                                   network(model._tcn_input(state, history)))
    mean, factor = model.velocity_step_distribution(state, history, velocity)
    expected = -torch.distributions.MultivariateNormal(mean, scale_tril=factor).log_prob(
        following - state).mean()
    one = EulerMaruyamaNLL()(model, state, following, history_states=history)
    torch.testing.assert_close(one, expected)
    multi = MonteCarloMultistepNLL(horizons=(1, 2), particles=3)
    targets = torch.stack((following - state, torch.randn_like(state)), dim=1)
    horizons = multi.per_horizon(model, state, targets, history_states=history)
    torch.testing.assert_close(horizons[1], one)
    (one + horizons[2]).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.drift_model.parameters())
    assert network.network[0].weight.grad.abs().sum() > 0


def test_diffusion_inputs_are_not_expanded_by_tcn_context():
    legacy = NeuralSDE(config(residual_architecture="mlp", diffusion_type="state_diagonal"))
    temporal = NeuralSDE(config(diffusion_type="state_diagonal"))
    temporal.diffusion_model.load_state_dict(legacy.diffusion_model.state_dict())
    state, history = torch.randn(5, 2), torch.randn(5, 7, 2)
    torch.testing.assert_close(legacy.diffusion_matrix(state, history[:, :1]),
                               temporal.diffusion_matrix(state, history))
    assert temporal.diffusion_model.network[0].in_features == 4


def test_rollout_shifts_dense_history_and_matches_transition():
    torch.manual_seed(18)
    model = NeuralSDE(config())
    current, history = torch.randn(2, 2), torch.randn(2, 7, 2)
    generated = model.rollout(current, history_states=history, horizon=3,
                              num_trajectories=2, diffusion_scale=0., seed=9)
    state, past = current, history
    for step in range(1, 4):
        mean, _ = model.velocity_step_distribution(state, past, state - past[:, 0])
        following = state + mean
        torch.testing.assert_close(generated[:, :, step], following[:, None].expand(-1, 2, -1))
        past = torch.cat((state[:, None], past[:, :-1]), dim=1)
        state = following
    model.drift_model.zero_output()
    torch.testing.assert_close(model.drift(current, history), torch.zeros_like(current))


@pytest.mark.parametrize("drift_type,operator", [
    ("mlp", "diagonal"), ("damped_residual", "diagonal"), ("damped_residual", "spsd"),
])
def test_stationary_and_stability_objectives(drift_type, operator):
    torch.manual_seed(14)
    model = NeuralSDE(config(drift_type=drift_type, damped_operator=operator,
                             diffusion_type="bounded_state_diagonal"))
    state, history, endpoint = torch.randn(4, 2), torch.randn(4, 7, 2), torch.randn(4, 2)
    loss = StationaryMeanLoss(horizon=2)(model, state, endpoint, history_states=history)
    regularizer = DiscreteLyapunovRegularizer(
        StabilityConfig(weight=.1, a=0., b=.9, generated_steps=2, perturb_scale=.1),
        torch.eye(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64), {})
    metrics = regularizer(model, state, history_states=history)
    (loss + metrics["weighted_penalty"]).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_default_full_context_and_controller_metadata(tmp_path):
    cfg = SDEConfig(2, .01, state_variable="velocity", residual_architecture="tcn")
    assert cfg.history_steps == cfg.tcn_context_frames == cfg.tcn_receptive_field == 63
    model = NeuralSDE(cfg)
    assert model.drift(torch.zeros(2, 2), torch.zeros(2, 63, 2)).shape == (2, 2)
    fields = ("residual_architecture", "tcn_context_frames", "tcn_channels",
              "tcn_kernel_size", "tcn_dilations")
    assert all("--" + name.replace("_", "-") in NEW_MODEL_ONLY for name in fields)
    (tmp_path / "config.json").write_text(json.dumps({"model": cfg.to_dict()}))
    summary = configuration_summary(tmp_path)["model"]
    assert summary["tcn_context_frames"] == 63
    assert summary["residual_architecture"] == "tcn"


def test_dense_history_used_by_batches_trim_and_innovations(shared_data):
    model = NeuralSDE(config(reduced_dim=shared_data.n_coordinates))
    batch = next(_sde_transition_batches(shared_data, model.config, "train", 5))
    assert batch["history_states"].shape == (5, 7, shared_data.n_coordinates)
    assert batch["state_path"].shape[1] == 9  # seven past, current, next
    synthetic = {"current_state": torch.zeros(2, shared_data.n_coordinates),
                 "state_path": torch.zeros(2, 9, shared_data.n_coordinates)}
    synthetic["state_path"][0, 0, 0] = 100.  # oldest velocity, not current pair
    assert _trim_keep(synthetic, model, 1.).tolist() == [False, True]
    values = np.random.default_rng(9).normal(size=(40, shared_data.n_coordinates)).cumsum(0)
    edges = np.ones(39, dtype=bool)
    edges[15] = False
    marginal, whitened, eligible = transition_innovations(
        model, values, edges, batch_size=4, device=torch.device("cpu"))
    assert len(eligible) == 32
    assert sum(len(segment) for segment in whitened) == eligible.sum()
    assert all(np.isfinite(segment).all() for segment in marginal + whitened)


@pytest.mark.parametrize("drift_type", ["mlp", "damped_residual"])
def test_cli_training_checkpoint_evaluation_and_warm_start(shared_data, tmp_path, drift_type):
    output = tmp_path / "tcn"
    main(["train", "--experiment", shared_data.experiment, "--rank", str(shared_data.rank),
          "--data-root", shared_data.source_config["data_root"], "--state-variable", "velocity",
          "--drift-type", drift_type, "--residual-architecture", "tcn",
          "--tcn-context-frames", "7", "--tcn-channels", "4", "--tcn-dilations", "1", "2",
          "--hidden-layers", "4", "--train-trim-percent", "0",
          "--epochs", "1", "--batch-size", "7", "--device", "cpu", "--output", str(output)])
    model, checkpoint = load_model(output / "best.pt")
    assert model.config.residual_architecture == "tcn"
    assert model.config.history_steps == 7
    assert checkpoint["model_config"]["tcn_dilations"] == (1, 2)
    evaluation = tmp_path / "evaluation"
    main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "test",
          "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "3",
          "--device", "cpu", "--output", str(evaluation)])
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert metrics["residual_architecture"] == "tcn"
    assert metrics["tcn_receptive_field"] == 7
    warm = ["train", "--checkpoint", str(output), "--epochs", "1", "--batch-size", "7",
            "--device", "cpu", "--train-trim-percent", "0", "--output", str(tmp_path / "warm")]
    with pytest.raises(SystemExit):
        main(warm + ["--residual-architecture", "mlp"])
    for option in (["--tcn-context-frames", "3"], ["--tcn-channels", "8"],
                   ["--tcn-kernel-size", "2"], ["--tcn-dilations", "1", "3"]):
        with pytest.raises(SystemExit):
            main(warm + option)
    main(warm + ["--multistep-weight", ".1", "--multistep-horizons", "1", "2",
                 "--multistep-particles", "3"])
    restored, _ = load_model(tmp_path / "warm" / "best.pt")
    assert restored.config == model.config
