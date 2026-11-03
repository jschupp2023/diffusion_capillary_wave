"""Optional one-step drift robustness on consistently perturbed measured history."""
import math

import torch


def validate_history_noise(config, weight, std, seed, train_mode):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("history_noise_weight must be finite and nonnegative")
    if not math.isfinite(std) or std < 0:
        raise ValueError("history_noise_std must be finite and nonnegative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("history noise seed must be an integer")
    if weight == 0:
        return
    if (config.state_variable != "velocity" or config.lag_steps != 1
            or config.history_steps < 1):
        raise ValueError("history noise requires native-lag velocity with history")
    if train_mode not in {"joint", "drift"}:
        raise ValueError("history noise requires joint or drift training")
    if any((config.velocity_memory_offsets, config.velocity_oscillator_periods,
            config.velocity_innovation_offsets)):
        raise ValueError("history noise does not support additive memory models")


def perturb_history(history, velocity_std, std, *, generator):
    """Anchor a_t and integrate IID velocity perturbations backwards in time.

    History is newest first. If eps[j] perturbs v_(t-j), the position
    perturbation at age k is -sum(eps[:k]). Thus all input velocities derived
    from positions agree, while current position and the observed target stay
    unchanged. std is relative to the checkpoint's native-velocity target_std,
    not its position scale. No input tensor is mutated.
    """
    if std == 0:
        return history
    noise = torch.randn(history.shape, dtype=history.dtype, device=history.device,
                        generator=generator) * (std * velocity_std.detach())
    return history - noise.cumsum(dim=-2)


def noisy_history_drift_loss(model, current, following, history, std, *, generator):
    """Coordinate-mean normalized velocity MSE; diffusion is never evaluated.

    Predict the measured next velocity from the noisy past and unchanged
    current position. This auxiliary loss is not an augmented Gaussian NLL.
    """
    noisy = perturb_history(history, model.target_std, std, generator=generator)
    velocity = current - noisy[..., 0, :]
    prediction = velocity + model.drift(current, noisy, velocity)
    target = following - current
    return ((prediction - target) / model.target_std.detach()).square().mean()
