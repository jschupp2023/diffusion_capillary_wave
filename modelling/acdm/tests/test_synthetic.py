from __future__ import annotations

import torch

from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.data import NormalizationStats
from modelling.acdm.conditional_edm.model import ConditionalEDM
from modelling.acdm.conditional_edm.sampling import edm_schedule, heun_integrate


def test_probability_flow_with_deterministic_linear_oracle() -> None:
    torch.manual_seed(0)
    current = torch.randn(32, 2, dtype=torch.float64)
    matrix = torch.tensor([[0.8, -0.2], [0.1, 0.7]], dtype=torch.float64)
    clean = current @ matrix.T
    sigmas = edm_schedule(
        sigma_min=0.002, sigma_max=10.0, rho=7.0, num_steps=32, dtype=torch.float64
    )
    initial = sigmas[0] * torch.randn_like(clean)
    samples = heun_integrate(lambda x, sigma: clean, initial, sigmas)
    assert torch.allclose(samples, clean, atol=1.0e-10, rtol=1.0e-10)


def test_probability_flow_with_linear_gaussian_oracle() -> None:
    torch.manual_seed(1)
    num_samples = 20_000
    mean = torch.tensor([0.7, -0.4], dtype=torch.float64)
    variance = torch.tensor([0.3, 0.8], dtype=torch.float64)
    sigmas = edm_schedule(
        sigma_min=0.001, sigma_max=12.0, rho=7.0, num_steps=80, dtype=torch.float64
    )
    initial = mean + torch.sqrt(variance + sigmas[0].square()) * torch.randn(
        num_samples, 2, dtype=torch.float64
    )

    def posterior_mean(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        gain = variance / (variance + sigma.square())
        return mean + gain * (x - mean)

    samples = heun_integrate(posterior_mean, initial, sigmas)
    empirical_mean = samples.mean(dim=0)
    empirical_variance = samples.var(dim=0, correction=0)
    assert torch.allclose(empirical_mean, mean, atol=0.025, rtol=0.0)
    assert torch.allclose(empirical_variance, variance, atol=0.03, rtol=0.04)


def test_production_initialization_matches_finite_sigma_variance() -> None:
    class GaussianOracle(ConditionalEDM):
        def denoise(self, x, sigma, current_state, params=None):
            sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
            return x / (1.0 + sigma.square())

    sigma_max = 4.0
    model = GaussianOracle(
        EDMConfig(
            reduced_dim=1,
            target_mode="future_state",
            hidden_dim=8,
            num_blocks=0,
            noise_embedding_dim=4,
            conditioning_mode="clean",
            sigma_min=0.001,
            sigma_max=sigma_max,
            num_sampling_steps=80,
        )
    ).double()
    model.set_normalization(
        NormalizationStats(
            state_mean=torch.zeros(1, dtype=torch.float64),
            state_std=torch.ones(1, dtype=torch.float64),
            target_mean=torch.zeros(1, dtype=torch.float64),
            target_std=torch.ones(1, dtype=torch.float64),
        )
    )
    samples = model.sample_next(
        torch.zeros(20_000, 1, dtype=torch.float64), num_samples=1, seed=5
    )[:, 0, 0]
    expected_variance = sigma_max**2 / (1.0 + sigma_max**2)
    assert abs(float(samples.mean())) < 0.02
    assert abs(float(samples.var(correction=0)) - expected_variance) < 0.025
