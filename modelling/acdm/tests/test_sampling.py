from __future__ import annotations

import pytest
import torch

from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.data import NormalizationStats
from modelling.acdm.conditional_edm.evaluate import (
    rollout_statistics,
    segmented_ensemble_metrics,
    segmented_reference_ensemble,
)
from modelling.acdm.conditional_edm.model import ConditionalEDM
from modelling.acdm.conditional_edm.sampling import edm_schedule, heun_integrate


def _model() -> ConditionalEDM:
    config = EDMConfig(
        reduced_dim=2,
        hidden_dim=16,
        num_blocks=1,
        noise_embedding_dim=8,
        sigma_min=0.01,
        sigma_max=2.0,
        num_sampling_steps=5,
    )
    model = ConditionalEDM(config)
    model.set_normalization(
        NormalizationStats(
            state_mean=torch.zeros(2),
            state_std=torch.ones(2),
            target_mean=torch.zeros(2),
            target_std=torch.ones(2),
        )
    )
    return model


def test_schedule_endpoints_and_monotonicity() -> None:
    schedule = edm_schedule(sigma_min=0.01, sigma_max=4.0, rho=7.0, num_steps=8)
    assert schedule.shape == (9,)
    assert schedule[0] == 4.0
    assert schedule[-2] == pytest.approx(0.01)
    assert schedule[-1] == 0.0
    assert torch.all(schedule[1:] <= schedule[:-1])
    with pytest.raises(ValueError):
        edm_schedule(sigma_min=0.01, sigma_max=4.0, rho=7.0, num_steps=1)


def test_heun_correction_and_final_euler_step() -> None:
    sigmas = torch.tensor([2.0, 1.0, 0.0])
    initial = torch.tensor([[3.0]])
    result = heun_integrate(lambda x, sigma: sigma * x, initial, sigmas)
    assert torch.allclose(result, torch.tensor([[3.75]]))


def test_constant_denoiser_reaches_clean_target() -> None:
    sigmas = edm_schedule(
        sigma_min=0.01, sigma_max=5.0, rho=3.0, num_steps=12, dtype=torch.float64
    )
    clean = torch.tensor([[1.5, -0.25], [-2.0, 3.0]], dtype=torch.float64)
    initial = 5.0 * torch.randn_like(clean)
    result = heun_integrate(lambda x, sigma: clean, initial, sigmas)
    assert torch.allclose(result, clean, atol=1.0e-10, rtol=1.0e-10)


def test_sampling_and_rollout_are_reproducible_and_batched() -> None:
    torch.manual_seed(4)
    model = _model()
    current = torch.randn(3, 2)
    first = model.sample_next(current, num_samples=4, seed=12)
    second = model.sample_next(current, num_samples=4, seed=12)
    assert first.shape == (3, 4, 2)
    assert torch.equal(first, second)
    assert not first.requires_grad

    trajectory = model.rollout(current, horizon=3, num_trajectories=4, seed=8)
    assert trajectory.shape == (3, 4, 4, 2)
    assert torch.allclose(trajectory[:, :, 0], current[:, None, :].expand(-1, 4, -1))
    assert not trajectory.requires_grad


def test_each_ensemble_member_uses_its_seeded_gaussian_initialization() -> None:
    class IdentityDenoiser(ConditionalEDM):
        def denoise(self, x, sigma, current_state, params=None):
            return x

    model = IdentityDenoiser(
        EDMConfig(
            reduced_dim=2,
            hidden_dim=8,
            num_blocks=0,
            noise_embedding_dim=4,
            conditioning_mode="clean",
            sigma_min=0.01,
            sigma_max=3.0,
            num_sampling_steps=4,
        )
    )
    model.set_normalization(
        NormalizationStats(
            state_mean=torch.zeros(2),
            state_std=torch.ones(2),
            target_mean=torch.zeros(2),
            target_std=torch.ones(2),
        )
    )
    current = torch.zeros(2, 2)
    samples = model.sample_next(current, num_samples=3, seed=19)
    generator = torch.Generator().manual_seed(19)
    expected = 3.0 * torch.randn(6, 2, generator=generator).reshape(2, 3, 2)
    assert torch.equal(samples, expected)
    assert torch.unique(samples.reshape(-1, 2), dim=0).shape[0] == 6


def test_rollout_reuses_generated_state_and_denormalizes_residual() -> None:
    class UnitIncrementDenoiser(ConditionalEDM):
        def denoise(self, x, sigma, current_state, params=None):
            return torch.full_like(x, 0.25)

    model = UnitIncrementDenoiser(
        EDMConfig(
            reduced_dim=2,
            hidden_dim=8,
            num_blocks=0,
            noise_embedding_dim=4,
            conditioning_mode="clean",
            sigma_min=0.01,
            sigma_max=2.0,
            num_sampling_steps=4,
        )
    )
    model.set_normalization(
        NormalizationStats(
            state_mean=torch.tensor([10.0, -5.0]),
            state_std=torch.tensor([2.0, 4.0]),
            target_mean=torch.full((2,), 0.5),
            target_std=torch.full((2,), 2.0),
        )
    )
    initial = torch.tensor([[2.0, -3.0]])
    trajectory = model.rollout(initial, horizon=3, num_trajectories=2, seed=4)
    expected = torch.stack([initial + step for step in range(4)], dim=1)
    expected = expected[:, None].expand(-1, 2, -1, -1)
    assert torch.allclose(trajectory, expected)
    statistics = rollout_statistics(trajectory)
    assert statistics["mean"].shape == (1, 4, 2)
    assert statistics["variance"].shape == (1, 4, 2)
    assert statistics["covariance"].shape == (1, 4, 2, 2)


def test_segmented_reference_ensemble_uses_complete_nonoverlapping_windows() -> None:
    states = torch.arange(2 * 11, dtype=torch.float32).reshape(2, 11, 1)
    segments = segmented_reference_ensemble(
        states,
        [0, 1],
        segment_length=4,
        lag_steps=2,
    )

    assert torch.equal(segments["run_ids"], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(segments["start_indices"], torch.tensor([0, 4, 0, 4]))
    assert torch.equal(segments["relative_steps"], torch.tensor([0, 2]))
    assert torch.equal(
        segments["reference"][:, :, 0],
        torch.tensor([[0.0, 2.0], [4.0, 6.0], [11.0, 13.0], [15.0, 17.0]]),
    )


def test_segmented_reference_ensemble_omits_windows_without_history() -> None:
    states = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    segments = segmented_reference_ensemble(
        states,
        [0],
        segment_length=4,
        lag_steps=1,
        history_steps=2,
    )

    assert torch.equal(segments["start_indices"], torch.tensor([4, 8]))
    assert torch.equal(
        segments["history_states"][:, :, 0],
        torch.tensor([[3.0, 2.0], [7.0, 6.0]]),
    )
    assert segments["skipped_missing_history"] == 1


def test_segmented_ensemble_metrics_compare_mean_and_covariance_relatively() -> None:
    reference = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 3.0]],
            [[2.0, 4.0], [3.0, 5.0]],
            [[4.0, 8.0], [5.0, 9.0]],
        ]
    )
    identical = segmented_ensemble_metrics(reference, reference)
    assert identical["mean_relative_l2_error"] == pytest.approx(0.0)
    assert identical["covariance_relative_frobenius_error"] == pytest.approx(0.0)

    scaled = segmented_ensemble_metrics(2 * reference, reference)
    assert scaled["mean_relative_l2_error"] == pytest.approx(1.0)
    assert scaled["covariance_relative_frobenius_error"] == pytest.approx(3.0)


def test_history_conditioned_sampling_requires_and_accepts_previous_state() -> None:
    model = ConditionalEDM(
        EDMConfig(
            reduced_dim=2,
            history_conditioning=True,
            hidden_dim=8,
            num_blocks=0,
            noise_embedding_dim=4,
            sigma_min=0.01,
            sigma_max=2.0,
            num_sampling_steps=4,
        )
    )
    model.set_normalization(
        NormalizationStats(
            state_mean=torch.zeros(2),
            state_std=torch.ones(2),
            target_mean=torch.zeros(2),
            target_std=torch.ones(2),
        )
    )
    current = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="history"):
        model.sample_next(current)
    trajectory = model.rollout(
        current,
        previous_state=torch.full_like(current, -1.0),
        horizon=2,
        num_trajectories=3,
        seed=5,
    )
    assert trajectory.shape == (1, 3, 3, 2)
