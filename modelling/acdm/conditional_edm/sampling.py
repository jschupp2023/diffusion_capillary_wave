from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .model import ConditionalEDM


def edm_schedule(
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_steps: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """EDM power-law schedule with ``num_steps`` positive levels and a final zero."""

    if sigma_min <= 0 or sigma_max <= sigma_min:
        raise ValueError("require 0 < sigma_min < sigma_max")
    if rho <= 0 or num_steps < 2:
        raise ValueError("rho must be positive and num_steps at least 2")
    ramp = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
    levels = (
        sigma_max ** (1 / rho)
        + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ).pow(rho)
    levels[0] = sigma_max
    levels[-1] = sigma_min
    return torch.cat((levels, levels.new_zeros(1)))


@torch.no_grad()
def heun_integrate(
    denoise: Callable[[Tensor, Tensor], Tensor],
    initial: Tensor,
    sigmas: Tensor,
) -> Tensor:
    """Integrate dx/dsigma=(x-D(x,sigma))/sigma with second-order Heun steps."""

    if initial.ndim != 2:
        raise ValueError("initial must have shape [batch, dimension]")
    if sigmas.ndim != 1 or sigmas.numel() < 3:
        raise ValueError("sigmas must contain at least two positive levels and a final zero")
    if sigmas.device != initial.device:
        raise ValueError("sigmas and initial must be on the same device")
    if not bool(torch.all(sigmas[:-1] > 0)) or sigmas[-1].item() != 0:
        raise ValueError("all schedule levels except the last must be positive; the last must be zero")
    if not bool(torch.all(sigmas[1:] <= sigmas[:-1])):
        raise ValueError("sigmas must be nonincreasing")

    x = initial
    final_step = sigmas.numel() - 2
    for index in range(sigmas.numel() - 1):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        derivative = (x - denoise(x, sigma)) / sigma
        step = sigma_next - sigma
        euler = x + step * derivative
        if index == final_step:
            x = euler
        else:
            derivative_next = (euler - denoise(euler, sigma_next)) / sigma_next
            x = x + step * 0.5 * (derivative + derivative_next)
    return x


def _seeded_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _check_sampling_inputs(
    model: ConditionalEDM,
    current_state: Tensor,
    history_states: Tensor | None,
    params: Tensor | None,
) -> None:
    if current_state.ndim != 2 or current_state.shape[-1] != model.config.reduced_dim:
        raise ValueError("current_state must have shape [batch, reduced_dim]")
    if current_state.device != model.state_mean.device:
        raise ValueError("current_state and model must be on the same device")
    if not current_state.is_floating_point():
        raise ValueError("current_state must be floating point")
    if current_state.dtype != model.state_mean.dtype:
        raise ValueError("current_state and model must have the same floating-point dtype")
    if model.config.history_conditioning:
        expected = (current_state.shape[0], model.config.history_steps, model.config.reduced_dim)
        if history_states is None or history_states.shape != expected:
            raise ValueError(f"history_states are required with shape {expected}")
        if history_states.device != current_state.device or history_states.dtype != current_state.dtype:
            raise ValueError("history_states must match current_state device and dtype")
    elif history_states is not None:
        raise ValueError("history was supplied to a model without history conditioning")
    if model.config.param_dim:
        expected = (current_state.shape[0], model.config.param_dim)
        if params is None or params.shape != expected:
            raise ValueError(f"params must have shape {expected}")
        if params.device != current_state.device or params.dtype != current_state.dtype:
            raise ValueError("params must match current_state device and dtype")
    elif params is not None:
        raise ValueError("params were supplied to a model with param_dim=0")


def _prepare_history(
    model: ConditionalEDM,
    current_state: Tensor,
    previous_state: Tensor | None,
    history_states: Tensor | None,
) -> Tensor | None:
    if not model.config.history_conditioning:
        if previous_state is not None or history_states is not None:
            raise ValueError("history was supplied to a model without history conditioning")
        return None
    if history_states is not None and previous_state is not None:
        raise ValueError("pass history_states or previous_state, not both")
    if history_states is None:
        if model.config.history_steps != 1 or previous_state is None:
            raise ValueError("history_states are required for this model")
        history_states = previous_state[:, None]
    return history_states


def _sample_next(
    model: ConditionalEDM,
    current_state: Tensor,
    history_states: Tensor | None,
    params: Tensor | None,
    *,
    num_steps: int | None,
    num_samples: int,
    generator: torch.Generator | None,
) -> Tensor:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    _check_sampling_inputs(model, current_state, history_states, params)
    batch_size, reduced_dim = current_state.shape
    flat_current = current_state.repeat_interleave(num_samples, dim=0)
    flat_history = (
        None if history_states is None else history_states.repeat_interleave(num_samples, dim=0)
    )
    flat_params = None if params is None else params.repeat_interleave(num_samples, dim=0)
    condition = model.normalize_state(flat_current)
    history = None if flat_history is None else model.normalize_history(flat_current, flat_history)
    normalized_params = model.normalize_params(flat_params)

    config = model.config
    clean_condition = model.conditioning_vector(condition, normalized_params, history)
    if clean_condition.shape != (len(flat_current), config.conditioning_dim):
        raise ValueError("concatenated sampling condition has an unexpected shape")

    if config.diffusion_formulation == "ddpm":
        steps = len(model.ddpm_betas)
        if num_steps is not None and num_steps != steps:
            raise ValueError(f"DDPM uses exactly {steps} sampling steps")
        target_state = torch.randn(flat_current.shape, device=current_state.device,
                                   dtype=current_state.dtype, generator=generator)
        conditioning_noise = torch.randn(clean_condition.shape, device=current_state.device,
                                         dtype=current_state.dtype, generator=generator)
        for step in range(steps, 0, -1):
            index = step - 1
            # Kohl: reuse epsilon_c and reconstruct the known c_r marginal each step.
            c_r = (model.ddpm_alpha_bars[index].sqrt() * clean_condition
                   + (1 - model.ddpm_alpha_bars[index]).sqrt() * conditioning_noise)
            joint = torch.cat((c_r, target_state), dim=-1)
            _, target_noise = model.split_joint(model.predict_noise(joint, step))
            target_state = model.ddpm_alphas[index].rsqrt() * (
                target_state
                - model.ddpm_betas[index] * target_noise
                / (1 - model.ddpm_alpha_bars[index]).sqrt()
            )
            if step > 1:
                target_state += model.ddpm_posterior_variance[index].sqrt() * torch.randn(
                    target_state.shape, device=target_state.device, dtype=target_state.dtype,
                    generator=generator)
        normalized_target = target_state
    else:
        sigmas = edm_schedule(
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            rho=config.rho,
            num_steps=config.num_sampling_steps if num_steps is None else num_steps,
            device=current_state.device,
            dtype=current_state.dtype,
        )
        initial = sigmas[0] * torch.randn(
            flat_current.shape,
            device=current_state.device,
            dtype=current_state.dtype,
            generator=generator,
        )
        conditioning_noise = None
        if config.conditioning_mode == "joint_noised":
            conditioning_noise = torch.randn(clean_condition.shape, device=current_state.device,
                                             dtype=current_state.dtype, generator=generator)

        def denoise(x, sigma):
            if x.shape != (len(flat_current), reduced_dim):
                raise ValueError("sampler target slice has an unexpected shape")
            if config.conditioning_mode == "joint_noised":
                # Rebuild c_sigma from known c0 on every solver evaluation; only d is integrated.
                c_sigma = clean_condition + sigma * conditioning_noise
                joint = torch.cat((c_sigma, x), dim=-1)
                _, denoised_target = model.split_joint(model.denoise(joint, sigma))
                return denoised_target
            if config.history_conditioning:
                return model.denoise(x, sigma, condition, normalized_params, history)
            return model.denoise(x, sigma, condition, normalized_params)
        normalized_target = heun_integrate(denoise, initial, sigmas)
    target = model.denormalize_target(normalized_target)
    next_state = flat_current + target if config.target_mode == "increment" else target
    return next_state.reshape(batch_size, num_samples, reduced_dim)


@torch.no_grad()
def sample_next(
    model: ConditionalEDM,
    current_state: Tensor,
    params: Tensor | None = None,
    *,
    previous_state: Tensor | None = None,
    history_states: Tensor | None = None,
    num_steps: int | None = None,
    num_samples: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw physical next states with shape [batch, num_samples, reduced_dim]."""

    if seed is not None and generator is not None:
        raise ValueError("pass either seed or generator, not both")
    if generator is None:
        generator = _seeded_generator(current_state.device, seed)
    history_states = _prepare_history(model, current_state, previous_state, history_states)
    was_training = model.training
    model.eval()
    try:
        return _sample_next(
            model,
            current_state,
            history_states,
            params,
            num_steps=num_steps,
            num_samples=num_samples,
            generator=generator,
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def rollout(
    model: ConditionalEDM,
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
    """Autoregressive physical rollout shaped [batch, ensemble, horizon+1, mode]."""

    if horizon < 0 or num_trajectories < 1:
        raise ValueError("horizon must be nonnegative and num_trajectories positive")
    history_states = _prepare_history(model, initial_state, previous_state, history_states)
    _check_sampling_inputs(model, initial_state, history_states, params)
    generator = _seeded_generator(initial_state.device, seed)
    batch_size, reduced_dim = initial_state.shape
    current = (
        initial_state[:, None, :]
        .expand(batch_size, num_trajectories, reduced_dim)
        .reshape(batch_size * num_trajectories, reduced_dim)
        .clone()
    )
    history = None
    if history_states is not None:
        history = (
            history_states[:, None]
            .expand(batch_size, num_trajectories, model.config.history_steps, reduced_dim)
            .reshape(batch_size * num_trajectories, model.config.history_steps, reduced_dim)
            .clone()
        )
    flat_params = None
    if params is not None:
        flat_params = (
            params[:, None, :]
            .expand(batch_size, num_trajectories, model.config.param_dim)
            .reshape(batch_size * num_trajectories, model.config.param_dim)
        )

    states = [current.reshape(batch_size, num_trajectories, reduced_dim)]
    was_training = model.training
    model.eval()
    try:
        for _physical_index in range(horizon):
            next_state = _sample_next(
                model,
                current,
                history,
                flat_params,
                num_steps=num_steps,
                num_samples=1,
                generator=generator,
            )[:, 0]
            if model.config.history_conditioning:
                history = torch.cat((current[:, None], history[:, :-1]), dim=1)
            current = next_state
            states.append(current.reshape(batch_size, num_trajectories, reduced_dim))
    finally:
        model.train(was_training)
    return torch.stack(states, dim=2)
