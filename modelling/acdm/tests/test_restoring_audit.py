import json

import numpy as np
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.experiments.restoring_audit import sample_histories, coherent_response
from modelling.neural_sde.experiments.adaptive_controller import training_metrics


def test_history_selection_never_crosses_bad_increment():
    paths = np.arange(20, dtype=float).reshape(1, 20, 1)
    good = np.ones((1, 19), dtype=bool)
    good[0, 8] = False
    current, history, count = sample_histories(paths, 3, 100, 0, good)
    times = current[:, 0].numpy().astype(int)
    assert count == 12
    assert not np.any((times >= 8) & (times <= 11))
    np.testing.assert_array_equal(history[:, :, 0], times[:, None] - np.arange(1, 4))


def test_coherent_response_recovers_explicit_stiffness_with_zero_residual():
    cfg = SDEConfig(reduced_dim=2, dt=.01, state_variable="velocity", history_steps=1,
                    drift_type="damped_residual", structured_damping=False,
                    stiffness_init=.2, hidden_layers=(4,))
    model = NeuralSDE(cfg)
    current = torch.randn(16, 2)
    history = torch.randn(16, 1, 2)
    result = coherent_response(model, current, history, [0, 1], .1)
    expected = (model.drift_model.stiffness_diagonal() * model.target_std / model.state_std).detach()
    np.testing.assert_allclose(result['mean'], expected, rtol=1e-5)
    assert result['anti_restoring_fraction'] == [0., 0.]


def test_advisor_sees_orthogonality_at_selected_epoch(tmp_path):
    rows = [dict(epoch=1, validation_nll=5., validation_total_loss=6.,
                 validation_restoring_orthogonality_loss=.1,
                 validation_restoring_residual_projection_coefficients=[.2]),
            dict(epoch=2, validation_nll=4., validation_total_loss=7.,
                 validation_restoring_orthogonality_loss=.3,
                 validation_restoring_residual_projection_coefficients=[.4])]
    (tmp_path/'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    metrics = training_metrics(tmp_path)
    assert metrics['best_validation_total_epoch'] == 1
    assert metrics['validation_restoring_orthogonality_loss'] == .1
    assert metrics['validation_restoring_residual_projection_coefficients'] == [.2]
