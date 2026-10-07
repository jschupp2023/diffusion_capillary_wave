from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .config import EDMConfig

if TYPE_CHECKING:
    from .data import NormalizationStats

DDPM_STEPS = 20


def _positive_sigma(sigma: Tensor | float) -> Tensor:
    value = torch.as_tensor(sigma)
    if torch.any(value <= 0):
        raise ValueError("EDM coefficients require sigma > 0")
    return value


def c_skip(sigma: Tensor | float, sigma_data: float) -> Tensor:
    sigma = _positive_sigma(sigma)
    return sigma_data**2 / (sigma.square() + sigma_data**2)


def c_out(sigma: Tensor | float, sigma_data: float) -> Tensor:
    sigma = _positive_sigma(sigma)
    return sigma * sigma_data / torch.sqrt(sigma.square() + sigma_data**2)


def c_in(sigma: Tensor | float, sigma_data: float) -> Tensor:
    sigma = _positive_sigma(sigma)
    return torch.rsqrt(sigma.square() + sigma_data**2)


def c_noise(sigma: Tensor | float) -> Tensor:
    return 0.25 * torch.log(_positive_sigma(sigma))


def loss_weight(sigma: Tensor | float, sigma_data: float) -> Tensor:
    sigma = _positive_sigma(sigma)
    return (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()


def _preconditioning(sigma: Tensor, sigma_data: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute all denoiser coefficients after one positivity check."""

    sigma = _positive_sigma(sigma)
    denominator = sigma.square() + sigma_data**2
    return (
        sigma_data**2 / denominator,
        sigma * sigma_data / denominator.sqrt(),
        denominator.rsqrt(),
        0.25 * sigma.log(),
    )


def _activation(name: str) -> nn.Module:
    choices: dict[str, type[nn.Module]] = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }
    try:
        return choices[name]()
    except KeyError as error:
        raise ValueError(f"unknown activation {name!r}") from error


class FourierNoiseEmbedding(nn.Module):
    def __init__(self, dimension: int, max_frequency: float = 1_000.0) -> None:
        super().__init__()
        half = dimension // 2
        frequencies = 2 * math.pi * torch.exp(torch.linspace(0, math.log(max_frequency), half))
        self.register_buffer("frequencies", frequencies)
        self.dimension = dimension

    def forward(self, noise: Tensor) -> Tensor:
        angles = noise[:, None] * self.frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dimension % 2:
            embedding = torch.cat((embedding, noise[:, None]), dim=-1)
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layers(inputs)


class ConditionalMLP(nn.Module):
    """The unchanged MLP backbone, with mode-dependent input/output widths."""

    def __init__(self, config: EDMConfig) -> None:
        super().__init__()
        self.noise_embedding = FourierNoiseEmbedding(config.noise_embedding_dim)
        input_dim = (
            (2 + (config.history_steps if config.history_conditioning else 0))
            * config.reduced_dim
            + config.noise_embedding_dim
            + config.param_dim
        )
        self.input = nn.Linear(input_dim, config.hidden_dim)
        self.activation = _activation(config.activation)
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim, config.activation, config.dropout)
            for _ in range(config.num_blocks)
        )
        output_dim = config.joint_dim if config.conditioning_mode == "joint_noised" else config.reduced_dim
        self.output = nn.Linear(config.hidden_dim, output_dim)
        self.config = config

    def forward(
        self,
        noisy_state: Tensor,
        noise: Tensor,
        current_state: Tensor | None = None,
        params: Tensor | None = None,
        history_increments: Tensor | None = None,
    ) -> Tensor:
        embedding = self.noise_embedding(noise)
        if self.config.conditioning_mode == "joint_noised":
            if any(value is not None for value in (current_state, params, history_increments)):
                raise ValueError("joint_noised mode expects one concatenated noisy state")
            if noisy_state.shape[-1] != self.config.joint_dim:
                raise ValueError("joint_noised backbone input has the wrong width")
            if self.config.diffusion_formulation == "ddpm":
                parts = [noisy_state, embedding]
            else:
                condition, target = noisy_state.split(
                    (self.config.conditioning_dim, self.config.reduced_dim), dim=-1
                )
                # Preserve the EDM feature ordering for checkpoint compatibility.
                parts = [target, embedding, condition]
        elif current_state is None:
            raise ValueError("clean mode requires current_state conditioning")
        else:
            parts = [noisy_state, embedding, current_state]
        if self.config.history_conditioning:
            expected = (noisy_state.shape[0], self.config.history_steps * self.config.reduced_dim)
            if self.config.conditioning_mode == "clean" and (
                history_increments is None or history_increments.shape != expected
            ):
                raise ValueError("history_increments are missing or have the wrong shape")
            if history_increments is not None:
                parts.append(history_increments)
        elif history_increments is not None:
            raise ValueError("history was supplied to a model without history conditioning")
        if self.config.param_dim:
            if self.config.conditioning_mode == "clean" and (
                params is None or params.shape != (noisy_state.shape[0], self.config.param_dim)
            ):
                raise ValueError("params have the wrong shape or are missing")
            if params is not None:
                parts.append(params)
        elif params is not None:
            raise ValueError("params were supplied to a model with param_dim=0")
        hidden = self.activation(self.input(torch.cat(parts, dim=-1)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.activation(hidden))


def _batch_sigma(sigma: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        return value.expand(reference.shape[0])
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError("sigma must be scalar, [batch], or [batch, 1]")
    if value.shape[0] == 1:
        return value.expand(reference.shape[0])
    if value.shape[0] != reference.shape[0]:
        raise ValueError("sigma batch dimension does not match x")
    return value


def _batch_step(step: Tensor | int, reference: Tensor) -> Tensor:
    value = torch.as_tensor(step, device=reference.device)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError("diffusion step must be scalar or [batch]")
    value = value.long()
    if torch.any((value < 1) | (value > DDPM_STEPS)):
        raise ValueError(f"diffusion step must lie in [1, {DDPM_STEPS}]")
    return value


class ConditionalEDM(nn.Module):
    """EDM denoiser for a normalized next-state or residual transition target."""

    def __init__(self, config: EDMConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = ConditionalMLP(config)
        self.register_buffer("state_mean", torch.zeros(config.reduced_dim))
        self.register_buffer("state_std", torch.ones(config.reduced_dim))
        self.register_buffer("target_mean", torch.zeros(config.reduced_dim))
        self.register_buffer("target_std", torch.ones(config.reduced_dim))
        self.register_buffer("param_mean", torch.zeros(config.param_dim))
        self.register_buffer("param_std", torch.ones(config.param_dim))
        if config.diffusion_formulation == "ddpm":
            betas = torch.linspace(1.e-4 * (500 / DDPM_STEPS),
                                   .02 * (500 / DDPM_STEPS), DDPM_STEPS)
            alphas = 1 - betas
            alpha_bars = alphas.cumprod(0)
            alpha_bars_prev = torch.cat((torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]))
            self.register_buffer("ddpm_betas", betas, persistent=False)
            self.register_buffer("ddpm_alphas", alphas, persistent=False)
            self.register_buffer("ddpm_alpha_bars", alpha_bars, persistent=False)
            self.register_buffer("ddpm_posterior_variance",
                                 betas * (1 - alpha_bars_prev) / (1 - alpha_bars), persistent=False)

    def set_normalization(
        self,
        stats: NormalizationStats,
        *,
        param_mean: Tensor | None = None,
        param_std: Tensor | None = None,
    ) -> None:
        values = {
            "state_mean": stats.state_mean,
            "state_std": stats.state_std,
            "target_mean": stats.target_mean,
            "target_std": stats.target_std,
        }
        if self.config.param_dim:
            if param_mean is None or param_std is None:
                raise ValueError("parameter normalization is required when param_dim > 0")
            values.update(param_mean=param_mean, param_std=param_std)
        for name, value in values.items():
            buffer = getattr(self, name)
            value = torch.as_tensor(value, device=buffer.device, dtype=buffer.dtype)
            if value.shape != buffer.shape:
                raise ValueError(f"normalization shape mismatch for {name}")
            if name.endswith("std") and torch.any(value <= 0):
                raise ValueError(f"{name} must be positive")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            buffer.copy_(value)

    def normalization_state(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, name).detach().cpu().clone()
            for name in ("state_mean", "state_std", "target_mean", "target_std", "param_mean", "param_std")
        }

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_params(self, params: Tensor | None) -> Tensor | None:
        if self.config.param_dim == 0:
            if params is not None:
                raise ValueError("params were supplied to a model with param_dim=0")
            return None
        if params is None or params.shape[-1] != self.config.param_dim:
            raise ValueError("params have the wrong shape or are missing")
        return (params - self.param_mean) / self.param_std

    def normalize_target(self, target: Tensor) -> Tensor:
        return (target - self.target_mean) / self.target_std

    def denormalize_target(self, target: Tensor) -> Tensor:
        return target * self.target_std + self.target_mean

    def normalize_history(self, current_state: Tensor, history_states: Tensor) -> Tensor:
        expected = (current_state.shape[0], self.config.history_steps, self.config.reduced_dim)
        if history_states.shape != expected:
            raise ValueError(f"history_states must have shape {expected}")
        sequence = torch.cat((current_state[:, None], history_states), dim=1)
        increments = sequence[:, :-1] - sequence[:, 1:]
        # Future-state normalization has a nonzero state mean, which must not
        # be subtracted from a difference between two states.
        normalized = (self.normalize_target(increments) if self.config.target_mode == "increment"
                      else increments / self.state_std)
        return normalized.flatten(start_dim=1)

    @staticmethod
    def conditioning_vector(current_state, params=None, history_increments=None):
        """Normalized features: current state, history increments, parameters."""
        return torch.cat([v for v in (current_state, history_increments, params) if v is not None], dim=-1)

    def noise_conditioning(self, sigma, current_state, params=None, history_increments=None,
                           *, generator=None, noise=None):
        """Forward-noise clean conditioning at exactly the target EDM level."""
        if self.config.conditioning_mode == "clean":
            return current_state, params, history_increments
        clean = self.conditioning_vector(current_state, params, history_increments)
        if clean.ndim != 2 or clean.shape[-1] != self.config.conditioning_dim:
            raise ValueError("concatenated conditioning has the wrong shape")
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        if noise.shape != clean.shape:
            raise ValueError("Conditioning noise must match the concatenated conditioning features")
        noisy = clean + _batch_sigma(sigma, clean)[:, None] * noise
        history_dim = 0 if history_increments is None else history_increments.shape[-1]
        state, history, parameters = noisy.split((current_state.shape[-1], history_dim, self.config.param_dim), dim=-1)
        return state, parameters if params is not None else None, history if history_increments is not None else None

    def split_joint(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the condition and target slices of ``x=(c,d)`` safely."""
        if x.ndim != 2 or x.shape[-1] != self.config.joint_dim:
            raise ValueError(f"joint state must have shape [batch, {self.config.joint_dim}]")
        condition, target = x.split((self.config.conditioning_dim, self.config.reduced_dim), dim=-1)
        return condition, target

    def denoise(
        self,
        x: Tensor,
        sigma: Tensor | float,
        current_state: Tensor | None = None,
        params: Tensor | None = None,
        history_increments: Tensor | None = None,
    ) -> Tensor:
        """Apply EDM preconditioning to a target (clean mode) or joint state."""

        if self.config.diffusion_formulation != "edm":
            raise ValueError("denoise is only available for EDM models")
        expected_dim = (self.config.joint_dim if self.config.conditioning_mode == "joint_noised"
                        else self.config.reduced_dim)
        if x.ndim != 2 or x.shape[-1] != expected_dim:
            raise ValueError(f"x must have shape [batch, {expected_dim}]")
        sigma_batch = _batch_sigma(sigma, x)
        skip, output_scale, input_scale, noise = _preconditioning(
            sigma_batch, self.config.sigma_data
        )
        if self.config.conditioning_mode == "joint_noised":
            if any(value is not None for value in (current_state, params, history_increments)):
                raise ValueError("joint_noised mode takes conditioning inside x")
            backbone_output = self.backbone(input_scale[:, None] * x, noise)
            target_output, condition_output = backbone_output.split(
                (self.config.reduced_dim, self.config.conditioning_dim), dim=-1
            )
            backbone_output = torch.cat((condition_output, target_output), dim=-1)
        else:
            if current_state is None or current_state.shape != x.shape:
                raise ValueError("clean current_state must have the same shape as target x")
            backbone_output = self.backbone(
                input_scale[:, None] * x, noise, current_state, params, history_increments
            )
        if backbone_output.shape != x.shape:
            raise ValueError("denoiser output shape does not match its diffused state")
        return skip[:, None] * x + output_scale[:, None] * backbone_output

    def predict_noise(self, x: Tensor, step: Tensor | int) -> Tensor:
        """Joint DDPM epsilon predictor; no EDM preconditioning is applied."""
        if self.config.diffusion_formulation != "ddpm":
            raise ValueError("predict_noise is only available for DDPM models")
        self.split_joint(x)
        steps = _batch_step(step, x)
        output = self.backbone(x, (steps - 1).to(x.dtype))
        self.split_joint(output)
        return output

    def sample_training_sigma(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        log_sigma = self.config.p_mean + self.config.p_std * torch.randn(
            batch_size, device=device, dtype=dtype, generator=generator
        )
        sigma = log_sigma.exp()
        if self.config.train_sigma_min is not None:
            sigma = sigma.clamp_min(self.config.train_sigma_min)
        if self.config.train_sigma_max is not None:
            sigma = sigma.clamp_max(self.config.train_sigma_max)
        return sigma

    def loss(
        self,
        batch: dict[str, Tensor],
        *,
        generator: torch.Generator | None = None,
        sigma: float | None = None,
        diffusion_step: Tensor | int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        current_raw = batch["current_state"]
        next_raw = batch["next_state"]
        if (current_raw.ndim != 2 or current_raw.shape != next_raw.shape
                or current_raw.shape[-1] != self.config.reduced_dim):
            raise ValueError("current_state and next_state have incompatible shapes")

        condition = self.normalize_state(current_raw)
        history = None
        if self.config.history_conditioning:
            history_states = batch.get("history_states")
            if history_states is None and self.config.history_steps == 1:
                previous_raw = batch.get("previous_state")
                history_states = None if previous_raw is None else previous_raw[:, None]
            if history_states is None:
                raise ValueError("history_states are required for history conditioning")
            history = self.normalize_history(current_raw, history_states)
        target_raw = next_raw - current_raw if self.config.target_mode == "increment" else next_raw
        target = self.normalize_target(target_raw)
        params = self.normalize_params(batch.get("params"))
        if self.config.diffusion_formulation == "ddpm":
            if sigma is not None:
                raise ValueError("sigma is not used by DDPM; pass diffusion_step")
            clean_condition = self.conditioning_vector(condition, params, history)
            x0 = torch.cat((clean_condition, target), dim=-1)
            if x0.shape != (target.shape[0], self.config.joint_dim):
                raise ValueError("joint clean state has an unexpected shape")
            steps = (_batch_step(diffusion_step, x0) if diffusion_step is not None else
                     torch.randint(1, DDPM_STEPS + 1, (len(x0),), device=x0.device,
                                   generator=generator))
            indices = steps - 1
            noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
            sqrt_alpha_bar = self.ddpm_alpha_bars[indices].sqrt()[:, None]
            sqrt_one_minus = (1 - self.ddpm_alpha_bars[indices]).sqrt()[:, None]
            predicted_noise = self.predict_noise(sqrt_alpha_bar * x0 + sqrt_one_minus * noise, steps)
            predicted_condition, predicted_target = self.split_joint(predicted_noise)
            condition_noise, target_noise = self.split_joint(noise)
            squared_error = (predicted_target - target_noise).square()
            condition_error = (predicted_condition - condition_noise).square()
            objective = torch.cat((condition_error, squared_error), dim=-1).mean()
            scale = 1 / self.config.joint_dim
            return objective, {
                "loss": objective.detach(),
                "mean_diffusion_step": steps.float().mean().detach(),
                "mean_unweighted_mse": squared_error.mean().detach(),
                "predicted_noise_norm": predicted_target.norm(dim=-1).mean().detach(),
                "noise_norm": target_noise.norm(dim=-1).mean().detach(),
                "target_loss": (scale * squared_error.sum(-1).mean()).detach(),
                "conditioning_loss": (scale * condition_error.sum(-1).mean()).detach(),
            }

        if diffusion_step is not None:
            raise ValueError("diffusion_step is only used by DDPM")
        sigma = self.sample_training_sigma(
            target.shape[0], device=target.device, dtype=target.dtype, generator=generator
        ) if sigma is None else _batch_sigma(sigma, target)
        if self.config.conditioning_mode == "joint_noised":
            # Training diffuses x0=(condition,target) jointly with one sigma.
            clean_condition = self.conditioning_vector(condition, params, history)
            x0 = torch.cat((clean_condition, target), dim=-1)
            if x0.shape != (target.shape[0], self.config.joint_dim):
                raise ValueError("joint clean state has an unexpected shape")
            noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
            denoised_full = self.denoise(x0 + sigma[:, None] * noise, sigma)
            denoised_condition, denoised = self.split_joint(denoised_full)
            squared_error_full = (denoised_full - x0).square()
        else:
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
            denoised = self.denoise(target + sigma[:, None] * noise, sigma, condition, params, history)
            squared_error_full = (denoised - target).square()

        squared_error = (denoised - target).square()
        per_sample = squared_error_full.sum(dim=-1)
        objective = (loss_weight(sigma, self.config.sigma_data) * per_sample).mean()
        diagnostics = {
            "loss": objective.detach(),
            "mean_sigma": sigma.mean().detach(),
            "mean_unweighted_mse": squared_error.mean().detach(),
            "denoised_norm": denoised.norm(dim=-1).mean().detach(),
            "target_norm": target.norm(dim=-1).mean().detach(),
        }
        if self.config.conditioning_mode == "joint_noised":
            weight = loss_weight(sigma, self.config.sigma_data)
            target_loss = (weight * squared_error.sum(-1)).mean()
            condition_loss = (weight * (denoised_condition - clean_condition).square().sum(-1)).mean()
            diagnostics["target_loss"] = target_loss.detach()
            diagnostics["conditioning_loss"] = condition_loss.detach()
        return objective, diagnostics

    def sample_next(
        self,
        current_state: Tensor,
        params: Tensor | None = None,
        *,
        previous_state: Tensor | None = None,
        history_states: Tensor | None = None,
        num_steps: int | None = None,
        num_samples: int = 1,
        seed: int | None = None,
    ) -> Tensor:
        from .sampling import sample_next

        return sample_next(
            self,
            current_state,
            params,
            previous_state=previous_state,
            history_states=history_states,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
        )

    def rollout(
        self,
        initial_state: Tensor,
        params: Tensor | None = None,
        *,
        previous_state: Tensor | None = None,
        history_states: Tensor | None = None,
        horizon: int,
        num_trajectories: int = 1,
        num_steps: int | None = None,
        seed: int | None = None,
    ) -> Tensor:
        from .sampling import rollout

        return rollout(
            self,
            initial_state,
            params,
            previous_state=previous_state,
            history_states=history_states,
            horizon=horizon,
            num_trajectories=num_trajectories,
            num_steps=num_steps,
            seed=seed,
        )
