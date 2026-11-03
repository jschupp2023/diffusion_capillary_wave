"""Score probes and staged calibration must preserve the training experiment."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.losses import MultistepVelocityEnergyScore
from modelling.neural_sde.energy_score_sensitivity import endpoint_probes
from modelling.neural_sde.objective_balance_diagnostic import _objective_summary, _gradient_vector
from modelling.neural_sde.train import _capped_multistep_batches
from modelling.neural_sde.experiments.adaptive_controller import (
    Controller, diagnostic_history_valid, normalized_options)


def model():
    return NeuralSDE(SDEConfig(reduced_dim=4, dt=.01, state_variable='velocity',
        residual_architecture='tcn', tcn_context_frames=7, tcn_dilations=(1,1,1),
        tcn_channels=4, tcn_kernel_size=3, hidden_layers=(4,)))


def test_probe_matches_training_score_with_nontrivial_scales_and_start_positions():
    torch.manual_seed(17)
    m = model()
    m.state_mean.copy_(torch.tensor([1., -2., 3., 4.]))
    m.state_std.copy_(torch.tensor([.5, 2., 3., 4.]))
    m.target_std.copy_(torch.tensor([2., 3., 4., .2]))
    samples = torch.randn(8, 6, 8)
    target = torch.randn(8, 8)
    current = torch.randn(8, 4) * 3
    objective = MultistepVelocityEnergyScore((20,), 6, target='joint', position_weight=2)
    expected = float(objective._score_samples(m, 20, samples, target))
    result = endpoint_probes(samples, target, current, m, 2, 1, 1)
    assert result['joint_score'] == pytest.approx(expected, rel=2e-6)
    assert result['joint_score'] == pytest.approx(
        result['observation_distance'] - result['subtracted_pair_distance'])
    shifted = endpoint_probes(samples, target, current + 20, m, 2, 1, 1)
    assert shifted['joint_score'] == pytest.approx(expected, rel=2e-6)


def test_probe_detects_known_position_bias_and_excess_spread():
    m = model()
    samples = torch.zeros(4, 4, 8)
    target = torch.zeros(4, 8)
    current = torch.zeros(4, 4)
    samples[..., :4] = 2
    biased = endpoint_probes(samples, target, current, m, 2, 1, 1)
    assert biased['all_positions_mean_correction_0.5_score_delta'] < 0
    samples[:, :2, :4] = -2
    spread = endpoint_probes(samples, target, current, m, 2, 1, 1)
    assert spread['all_positions_amplitude_0.9_score_delta'] < 0
    assert spread['all_positions_amplitude_1.1_score_delta'] > 0


def test_zero_weight_preserves_raw_gradient_direction_and_sparse_coefficient():
    groups = {'all': np.arange(2), 'drift': np.array([0]), 'diffusion': np.array([1])}
    kwargs = dict(losses=[1., 2.], vectors=[np.array([2., 3.]), np.array([4., 5.])],
        fraction=.005, route_indices=np.array([0]), groups=groups)
    zero = _objective_summary(weight=0., **kwargs)
    np.testing.assert_array_equal(zero['mean_routed_gradient'], [3., 0.])
    assert zero['groups']['drift']['expected_weighted_gradient_norm'] == 0
    active = _objective_summary(weight=.03, **kwargs)
    assert active['selected_update_coefficient'] == pytest.approx(6.)
    assert active['groups']['drift']['selected_update_weighted_gradient_norm'] == pytest.approx(18.)
    parameter = torch.tensor([0.], requires_grad=True)
    with pytest.raises(FloatingPointError):
        _gradient_vector(parameter.sqrt().sum(), [('parameter', parameter)])
    with pytest.raises(FloatingPointError, match="objective loss"):
        _gradient_vector((parameter + float('inf')).sum(), [('parameter', parameter)])


def test_short_history_retains_parameter_count_and_diagnostic_stencil_does_not_change_drift():
    torch.manual_seed(14)
    short = model()
    long = NeuralSDE(replace(short.config, history_steps=15, tcn_context_frames=15,
                            tcn_dilations=(1,2,4)))
    assert sum(p.numel() for p in short.parameters()) == sum(p.numel() for p in long.parameters())
    current, history = torch.randn(3, 4), torch.randn(3, 15, 4)
    velocity = current - history[:, 0]
    before = short.velocity_base_step_distribution(current, history[:, :7], velocity)
    short.config = replace(short.config, history_steps=15)
    after = short.velocity_base_step_distribution(current, history, velocity)
    for expected, actual in zip(before, after):
        torch.testing.assert_close(actual, expected)


class EligibilityData:
    """One trajectory with a jump visible at history 15 but not history 7."""
    batch_size = 64
    splits = {'validation': ('rep13',)}

    def __init__(self):
        self.values = np.zeros((40, 4), dtype=np.float32)
        self.values[9:] = 10

    def iter_transitions(self, split, *, lag_steps, history_steps,
                         history_spacing_steps, include_next_history,
                         include_state_path, batch_size, seed):
        assert split == 'validation' and history_spacing_steps == 1
        starts = np.arange(history_steps, len(self.values) - lag_steps)
        offsets = np.arange(1, history_steps + 1)
        path_offsets = np.arange(-history_steps, lag_steps + 1)
        batch = {
            'current_state': self.values[starts],
            'next_state': self.values[starts + lag_steps],
            'history_states': self.values[starts[:, None] - offsets],
            'next_history_states': self.values[starts[:, None] + lag_steps - offsets],
            'state_path': self.values[starts[:, None] + path_offsets],
            'start_index': starts,
        }
        yield 'rep13', batch


def _selected_by_start(data, checkpoint_model, history, cutoff=None):
    batches = list(_capped_multistep_batches(
        data, checkpoint_model, 'validation', 64, (2,), 64, None,
        cutoff, np.ones(4), eligibility_history_steps=history,
        include_window_ids=True))
    merged = {key: torch.cat([part[key] for part in batches]) for key in batches[0]}
    return {int(start): index for index, start in enumerate(
        merged['window_start_index'].long().tolist())}, merged


def test_diagnostic_history_uses_extra_frames_only_for_eligibility():
    checkpoint_model = model()
    data = EligibilityData()
    short_ids, short = _selected_by_start(data, checkpoint_model, 7, cutoff=1)
    long_ids, long = _selected_by_start(data, checkpoint_model, 15, cutoff=1)
    assert 20 in short_ids
    assert 20 not in long_ids  # the frame-9 jump lies only in the extra stencil
    assert long['history_states'].shape[1] == checkpoint_model.config.history_steps == 7

    _, long_untrimmed = _selected_by_start(data, checkpoint_model, 15)
    short_index, long_index = short_ids[20], int(
        (long_untrimmed['window_start_index'].long() == 20).nonzero()[0])
    velocity = short['current_state'][short_index] - short['history_states'][short_index, 0]
    before = checkpoint_model.velocity_base_step_distribution(
        short['current_state'][short_index:short_index + 1],
        short['history_states'][short_index:short_index + 1], velocity[None])
    after = checkpoint_model.velocity_base_step_distribution(
        long_untrimmed['current_state'][long_index:long_index + 1],
        long_untrimmed['history_states'][long_index:long_index + 1], velocity[None])
    for expected, actual in zip(before, after):
        torch.testing.assert_close(actual, expected)


def test_cached_diagnostic_requires_requested_eligibility_and_audit(tmp_path):
    summary = tmp_path / 'summary.json'
    summary.write_text('{"requested_eligibility_history_steps": 15, '
                       '"eligibility_history_steps": 15, '
                       '"model_conditioning_history_steps": 7, '
                       '"selected_window_identifiers_by_batch": '
                       '{"41019": {"sha256": "abc"}}}')
    assert diagnostic_history_valid(summary, 15)
    assert not diagnostic_history_valid(summary, 16)
    summary.write_text('{"eligibility_history_steps": 15}')
    assert not diagnostic_history_valid(summary, 15)


def staged():
    controller = object.__new__(Controller)
    controller.study = {'staged_tcn_calibration': True}
    options = normalized_options({'--learning-rate':1e-5, '--epochs':1,
        '--stationary-mean-weight':0, '--stationary-variance-weight':0,
        '--multistep-weight':.03, '--multistep-position-weight':2})
    parent = dict(id='base',kind='trained',status='screened',change_family='short_history_baseline',
        options=deepcopy(options),objective_balance={'metrics':{'ok':True}},
        score_sensitivity={'metrics':{'ok':True}})
    controller.state = {'trials':[parent]}
    return controller, options


def decision(role, parent='base'):
    return dict(change_family=role,training_mode='finetune',parent_trial=parent)


def test_calibration_requires_diagnostics_and_same_parent_control():
    c, options = staged()
    with pytest.raises(ValueError, match='control before'):
        c.validate_calibration_stage(decision('es_calibration'), options)
    c.validate_calibration_stage(decision('es_control'), options)
    c.state['trials'].append(dict(id='control',kind='trained',status='screened',
        change_family='es_control',options=deepcopy(options),parent_trial='base'))
    options['--multistep-weight'] = ['0.1']
    c.validate_calibration_stage(decision('es_calibration'), options)
    options['--multistep-position-weight'] = ['4']
    with pytest.raises(ValueError, match='exactly one'):
        c.validate_calibration_stage(decision('es_calibration'), options)
    c.state['trials'][0]['objective_balance'] = {}
    with pytest.raises(ValueError, match='gradient and score'):
        c.validate_calibration_stage(decision('moment_control'), options)


def test_moments_require_low_learning_rate_and_completed_same_parent_control():
    c, options = staged()
    bad = dict(options, **{'--learning-rate':['0.001']})
    with pytest.raises(ValueError, match='lr 1e-5'):
        c.validate_calibration_stage(decision('moment_control'), bad)
    c.validate_calibration_stage(decision('moment_control'), options)
    options['--stationary-mean-weight'] = options['--stationary-variance-weight'] = ['0.001']
    with pytest.raises(ValueError, match='same-parent'):
        c.validate_calibration_stage(decision('moment_refinement'), options)
    control = dict(id='control',kind='trained',status='screened',change_family='moment_control',parent_trial='wrong')
    c.state['trials'].append(control)
    with pytest.raises(ValueError, match='same-parent'):
        c.validate_calibration_stage(decision('moment_refinement'), options)
    control['parent_trial'] = 'base'
    c.validate_calibration_stage(decision('moment_refinement'), options)
