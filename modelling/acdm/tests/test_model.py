from __future__ import annotations

import pytest
import torch

from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.data import NormalizationStats
from modelling.acdm.conditional_edm.model import ConditionalEDM, c_in, c_noise, c_out, c_skip, loss_weight


def _stats(dimension: int) -> NormalizationStats:
    return NormalizationStats(
        state_mean=torch.zeros(dimension),
        state_std=torch.ones(dimension),
        target_mean=torch.zeros(dimension),
        target_std=torch.ones(dimension),
    )


def test_edm_coefficients_are_finite_and_consistent() -> None:
    sigma = torch.tensor([2.0e-3, 0.1, 1.0, 80.0])
    sigma_data = 0.7
    denominator = sigma.square() + sigma_data**2
    coefficients = [
        c_skip(sigma, sigma_data),
        c_out(sigma, sigma_data),
        c_in(sigma, sigma_data),
        c_noise(sigma),
    ]
    assert all(torch.isfinite(value).all() for value in coefficients)
    assert torch.allclose(coefficients[0], sigma_data**2 / denominator)
    assert torch.allclose(coefficients[1], sigma * sigma_data / denominator.sqrt())
    assert torch.allclose(coefficients[2], denominator.rsqrt())
    assert torch.allclose(coefficients[3], 0.25 * sigma.log())
    weight = loss_weight(sigma, sigma_data)
    assert torch.isfinite(weight).all()
    assert torch.allclose(weight, denominator / (sigma * sigma_data).square())
    assert torch.allclose(weight * c_out(sigma, sigma_data).square(), torch.ones_like(sigma))
    with pytest.raises(ValueError):
        c_noise(torch.tensor(0.0))
    with pytest.raises(ValueError, match="finite"):
        EDMConfig(reduced_dim=2, p_mean=float("nan"))


def test_denoiser_shapes_sigma_broadcasting_and_conditioning() -> None:
    torch.manual_seed(3)
    config = EDMConfig(
        reduced_dim=4,
        param_dim=2,
        hidden_dim=24,
        num_blocks=1,
        noise_embedding_dim=8,
        conditioning_mode="clean",
        sigma_max=3.0,
        num_sampling_steps=4,
    )
    model = ConditionalEDM(config)
    model.set_normalization(_stats(4), param_mean=torch.zeros(2), param_std=torch.ones(2))
    x = torch.randn(5, 4)
    current = torch.randn(5, 4)
    params = torch.randn(5, 2)

    scalar = model.denoise(x, 0.5, current, params)
    batched = model.denoise(x, torch.full((5, 1), 0.5), current, params)
    changed_condition = model.denoise(x, 0.5, current + 1.0, params)
    assert scalar.shape == (5, 4)
    assert torch.allclose(scalar, batched)
    assert not torch.allclose(scalar, changed_condition)


def test_weighted_loss_is_finite_and_differentiable() -> None:
    config = EDMConfig(
        reduced_dim=3,
        hidden_dim=16,
        num_blocks=1,
        noise_embedding_dim=8,
        sigma_max=2.0,
        num_sampling_steps=4,
    )
    model = ConditionalEDM(config)
    model.set_normalization(_stats(3))
    batch = {"current_state": torch.randn(7, 3), "next_state": torch.randn(7, 3)}
    loss, diagnostics = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(diagnostics) == {
        "loss",
        "mean_sigma",
        "mean_unweighted_mse",
        "denoised_norm",
        "target_norm",
        "target_loss",
        "conditioning_loss",
    }
    torch.testing.assert_close(loss, diagnostics["target_loss"] + diagnostics["conditioning_loss"])
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_loss_uses_the_residual_target_before_normalization() -> None:
    class CapturingEDM(ConditionalEDM):
        captured_target: torch.Tensor | None = None

        def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
            self.captured_target = target.detach().clone()
            return super().normalize_target(target)

    model = CapturingEDM(
        EDMConfig(
            reduced_dim=2,
            hidden_dim=8,
            num_blocks=0,
            noise_embedding_dim=4,
            p_std=0.0,
            sigma_max=2.0,
            num_sampling_steps=4,
        )
    )
    model.set_normalization(_stats(2))
    current = torch.tensor([[2.0, -1.0], [4.0, 3.0]])
    next_state = torch.tensor([[5.0, 2.0], [3.0, 8.0]])
    model.loss({"current_state": current, "next_state": next_state})
    assert torch.equal(model.captured_target, next_state - current)


def test_history_conditioning_requires_previous_state() -> None:
    model = ConditionalEDM(
        EDMConfig(
            reduced_dim=2,
            history_conditioning=True,
            hidden_dim=8,
            num_blocks=0,
            noise_embedding_dim=4,
            sigma_max=2.0,
            num_sampling_steps=4,
        )
    )
    model.set_normalization(_stats(2))
    batch = {"current_state": torch.randn(3, 2), "next_state": torch.randn(3, 2)}
    with pytest.raises(ValueError, match="history_states"):
        model.loss(batch)
    batch["previous_state"] = torch.randn(3, 2)
    assert torch.isfinite(model.loss(batch)[0])


def test_two_history_increments_are_normalized_and_flattened() -> None:
    model = ConditionalEDM(
        EDMConfig(
            reduced_dim=1,
            history_conditioning=True,
            history_steps=2,
            hidden_dim=8,
            num_blocks=0,
            noise_embedding_dim=4,
            sigma_max=2.0,
            num_sampling_steps=4,
        )
    )
    model.set_normalization(_stats(1))
    current = torch.tensor([[5.0]])
    history = torch.tensor([[[3.0], [2.0]]])
    assert torch.equal(model.normalize_history(current, history), torch.tensor([[2.0, 1.0]]))


def test_future_state_history_does_not_subtract_state_mean_from_increments():
    model = ConditionalEDM(EDMConfig(reduced_dim=1, target_mode="future_state",
                                     history_conditioning=True, history_steps=2))
    model.set_normalization(NormalizationStats(
        torch.tensor([10.]), torch.tensor([2.]), torch.tensor([10.]), torch.tensor([2.])))
    actual = model.normalize_history(torch.tensor([[4.]]), torch.tensor([[[2.], [0.]]]))
    torch.testing.assert_close(actual, torch.tensor([[1., 1.]]))
