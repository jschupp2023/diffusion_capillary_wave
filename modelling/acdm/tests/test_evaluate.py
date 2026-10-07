from __future__ import annotations

import pytest
import torch

from modelling.acdm.conditional_edm.evaluate import stochastic_rollout_metrics


def test_identical_generated_and_reference_rollouts_have_zero_moment_errors() -> None:
    reference = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 2.0], [3.0, 3.0], [4.0, 7.0]],
            [[1.0, -1.0], [3.0, 0.0], [4.0, 3.0], [8.0, 5.0]],
            [[-1.0, 2.0], [0.0, 5.0], [2.0, 4.0], [5.0, 8.0]],
            [[2.0, 1.0], [5.0, 3.0], [7.0, 7.0], [8.0, 6.0]],
        ]
    )
    trajectories = reference[:, None].repeat(1, 3, 1, 1)
    metrics = stochastic_rollout_metrics(
        trajectories,
        reference,
        state_mean=torch.zeros(2),
        state_std=torch.ones(2),
        target_mean=torch.zeros(2),
        target_std=torch.ones(2),
    )

    assert metrics["one_step"]["conditional_mean_rmse"] == pytest.approx(0.0)
    assert metrics["one_step"]["conditional_covariance_relative_error"] == pytest.approx(0.0)
    assert metrics["one_step"]["marginal_covariance_relative_error"] == pytest.approx(0.0)
    assert metrics["rollout"]["mean_rmse_average"] == pytest.approx(0.0)
    assert metrics["rollout"]["mean_relative_l2_error_average"] == pytest.approx(0.0)
    assert metrics["rollout"]["mean_relative_l2_error_global"] == pytest.approx(0.0)
    assert metrics["rollout"]["covariance_relative_error_average"] == pytest.approx(0.0)
    assert metrics["rollout"]["consecutive_increment_correlation_rmse"] == pytest.approx(0.0)

    scaled_metrics = stochastic_rollout_metrics(
        2 * trajectories,
        reference,
        state_mean=torch.tensor([10.0, -5.0]),
        state_std=torch.tensor([2.0, 4.0]),
        target_mean=torch.zeros(2),
        target_std=torch.ones(2),
    )
    assert scaled_metrics["rollout"]["mean_relative_l2_error_over_time"] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )
    assert scaled_metrics["rollout"]["mean_relative_l2_error_global"] == pytest.approx(1.0)
    assert scaled_metrics["rollout"]["covariance_relative_error_over_time"] == pytest.approx(
        [3.0, 3.0, 3.0, 3.0]
    )
    assert scaled_metrics["rollout"]["covariance_relative_error_average"] == pytest.approx(3.0)
    assert torch.allclose(
        torch.tensor(scaled_metrics["rollout"]["raw_generated_mean_over_time"]),
        2 * reference.mean(dim=0),
    )
    assert torch.allclose(
        torch.tensor(scaled_metrics["rollout"]["raw_reference_mean_over_time"]),
        reference.mean(dim=0),
    )
