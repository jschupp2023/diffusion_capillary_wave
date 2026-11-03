"""History perturbation geometry, opt-in behavior, and drift-only gradients."""
import copy
import json

import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.history_noise import (
    perturb_history, noisy_history_drift_loss, validate_history_noise)
from modelling.neural_sde.losses import EulerMaruyamaNLL
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import _epoch, _fit_sde_normalization, load_model, main


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cfg(dimension=3, **kwargs):
    return SDEConfig(dimension, dt=.01, state_variable="velocity",
                     residual_architecture="tcn", tcn_context_frames=3,
                     tcn_channels=4, tcn_dilations=(1,), hidden_layers=(4,), **kwargs)


def test_integrated_noise_is_anchored_consistent_and_reproducible():
    history = torch.randn(5, 3, 2)
    original = history.clone()
    scale = torch.tensor([2., 5.])
    seed = lambda: torch.Generator().manual_seed(7)
    eps = torch.randn(history.shape, generator=seed()) * (.1 * scale)
    noisy = perturb_history(history, scale, .1, generator=seed())
    delta = noisy - history
    torch.testing.assert_close(delta, -eps.cumsum(-2))
    positions = torch.cat((torch.zeros(5, 1, 2), delta), dim=1)
    torch.testing.assert_close(positions[:, :-1] - positions[:, 1:], eps)
    torch.testing.assert_close(history, original, rtol=0, atol=0)
    torch.testing.assert_close(noisy, perturb_history(history, scale, .1, generator=seed()))
    generator = seed(); before = generator.get_state()
    assert perturb_history(history, scale, 0., generator=generator) is history
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize("drift_type", ["mlp", "damped_residual"])
@pytest.mark.parametrize("diffusion_type", ["diagonal", "full", "bounded_state_diagonal"])
def test_auxiliary_never_trains_diffusion_or_changes_target(drift_type, diffusion_type):
    model = NeuralSDE(cfg(drift_type=drift_type, diffusion_type=diffusion_type))
    current, following, history = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3, 3)
    copies = [x.clone() for x in (current, following, history)]
    loss = noisy_history_drift_loss(model, current, following, history, .1,
                                   generator=torch.Generator().manual_seed(3))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.drift_model.parameters())
    assert all(p.grad is None for p in model.diffusion_model.parameters())
    for x, y in zip((current, following, history), copies):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    zero = noisy_history_drift_loss(model, current, following, history, 0., generator=None)
    prediction, _ = model.velocity_step_distribution(current, history, current-history[:, 0])
    expected = ((prediction-(following-current))/model.target_std).square().mean()
    torch.testing.assert_close(zero, expected)


def test_epoch_disabled_equivalence_and_validation_seed(shared_data):
    model = NeuralSDE(cfg())
    model.set_normalization(_fit_sde_normalization(shared_data, model.config))
    first, second = copy.deepcopy(model), copy.deepcopy(model)
    a, b = {}, {}
    for m, report, options in [(first, a, {}), (second, b, {"history_noise_weight":0., "history_noise_std":10.})]:
        torch.manual_seed(99)
        _epoch(m, shared_data, "train", 7, "cpu", EulerMaruyamaNLL(),
               torch.optim.SGD(m.parameters(), lr=1e-4), seed=3,
               metric_output=report, **options)
    assert a == b
    for k, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[k], rtol=0, atol=0)
    reports = []
    for noise_seed in (11, 11, 12):
        report = {}
        _epoch(model, shared_data, "validation", 7, "cpu", EulerMaruyamaNLL(),
               history_noise_weight=.2, history_noise_std=.1,
               history_noise_seed=noise_seed, metric_output=report)
        reports.append(report)
    assert reports[0] == reports[1]
    assert reports[0]['one_step_nll'] == reports[2]['one_step_nll']
    assert reports[0]['history_noise_loss'] != reports[2]['history_noise_loss']
    assert reports[0]['total_loss'] == pytest.approx(reports[0]['one_step_nll'] + .2*reports[0]['history_noise_loss'])


def test_epoch_routes_only_auxiliary_to_drift(shared_data):
    model = NeuralSDE(cfg(diffusion_type="bounded_state_diagonal"))
    model.set_normalization(_fit_sde_normalization(shared_data, model.config))
    before = copy.deepcopy(model.state_dict())
    class ZeroObjective(torch.nn.Module):
        def forward(self, model, *args, **kwargs):
            return sum(p.sum()*0 for p in model.drift_model.parameters())
    _epoch(model, shared_data, "train", 7, "cpu", ZeroObjective(),
           torch.optim.SGD(model.parameters(), lr=1e-4), history_noise_weight=.2)
    assert any(not torch.equal(v, before[k]) for k,v in model.state_dict().items() if k.startswith('drift_model'))
    for k,v in model.state_dict().items():
        if k.startswith('diffusion_model'):
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)


@pytest.mark.parametrize("weight,std,mode", [(-1., .1, 'joint'), (float('nan'), .1, 'joint'),
                                           (.1, -1., 'joint'), (.1, float('inf'), 'joint'),
                                           (.1, .1, 'diffusion'), (.1, .1, 'memory')])
def test_invalid_options(weight, std, mode):
    with pytest.raises(ValueError):
        validate_history_noise(cfg(), weight, std, 0, mode)


def test_cli_checkpoint_roundtrip_and_continuation(shared_data, tmp_path):
    output = tmp_path/'noise'
    main(['train','--experiment',shared_data.experiment,'--rank',str(shared_data.rank),
          '--data-root',shared_data.source_config['data_root'],'--state-variable','velocity',
          '--residual-architecture','tcn','--tcn-context-frames','3','--tcn-channels','4',
          '--tcn-dilations','1','--epochs','1','--batch-size','7','--device','cpu',
          '--history-noise-weight','.2','--history-noise-std','.03',
          '--history-noise-validation-seed','9','--output',str(output)])
    model, saved = load_model(output/'best.pt')
    assert saved['training']['history_noise_weight'] == .2
    assert saved['training']['history_noise_std'] == .03
    assert saved['training']['history_noise_validation_seed'] == 9
    record = json.loads((output/'metrics.jsonl').read_text())
    assert record['validation_total_loss'] == pytest.approx(record['validation_nll'] + .2*record['validation_history_noise_loss'])
    warm = tmp_path/'warm'
    main(['train','--checkpoint',str(output/'best.pt'),'--epochs','1','--batch-size','7',
          '--device','cpu','--output',str(warm)])
    restored, ckpt = load_model(warm/'best.pt')
    assert ckpt['training']['history_noise_weight'] == 0.
    assert restored.config == model.config
