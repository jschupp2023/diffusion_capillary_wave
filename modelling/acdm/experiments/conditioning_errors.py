"""Training-only empirical conditioning errors for small rollout experiments.

Unlike independent Gaussian perturbations, each bank entry keeps the spatial
and temporal correlations of one model-generated current/history window.
Short windows matter: stochastic trajectory divergence is not all model error.
"""
from dataclasses import dataclass
import math

import torch

from modelling.acdm.conditional_edm.evaluate import reference_windows


def _draw_indices(count, size, device, generator):
    draw_device = device if generator is None else generator.device
    return torch.randint(count, (size,), device=draw_device, generator=generator).to(device)


@dataclass
class ErrorBank:
    errors: torch.Tensor  # [entry, current then newest-first history, coordinate]
    replay: dict[str, torch.Tensor]  # generated conditioning and recorded true next
    summary: dict

    def __len__(self):
        return len(self.errors)

    def to(self, device):
        return ErrorBank(self.errors.to(device),
                         {key: value.to(device) for key, value in self.replay.items()}, self.summary)

    def sample_replay_batch(self, batch_size, *, generator=None):
        """Sample model-generated training inputs paired with the recorded next state."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = _draw_indices(len(self), batch_size, self.errors.device, generator)
        return {key: value[indices] for key, value in self.replay.items()}


@torch.no_grad()
def build_error_bank(model, data, *, num_conditions=64, horizon=8, seed=0,
                     sampling_steps=None, max_error_std=2., error_scale="state", batch_size=32):
    """Collect short free-rollout errors from TRAIN repetitions only, on CPU.

    ``horizon`` is the number of generated conditioning states per start; each
    has a recorded next-state target. Histories initially include the real past.
    We reject nonfinite windows and windows whose maximum per-frame RMS error
    exceeds ``max_error_std`` in training state (or target) standard deviations.
    This filters unusable augmentation examples; generated outputs are not clipped.
    """
    if min(num_conditions, horizon, batch_size) < 1:
        raise ValueError("Condition count, horizon and batch size must be positive")
    if not math.isfinite(max_error_std) or max_error_std <= 0:
        raise ValueError("max_error_std must be positive and finite")
    if error_scale not in ("state", "target"):
        raise ValueError("error_scale must be 'state' or 'target'")
    if model.config.param_dim:
        raise ValueError("Empirical banks currently support single-experiment models without parameters")
    reference, history, _, names, starts = reference_windows(
        data, model.config, "train", num_conditions, horizon + 1, seed)
    device, dtype = model.state_mean.device, model.state_mean.dtype
    actual_steps = (len(model.ddpm_betas) if getattr(model.config, "diffusion_formulation", "edm") == "ddpm"
                    else sampling_steps)
    generated = []
    for start in range(0, num_conditions, batch_size):
        stop = start + batch_size
        generated.append(model.rollout(
            reference[start:stop, 0].to(device=device, dtype=dtype),
            history_states=None if history is None else history[start:stop].to(device=device, dtype=dtype),
            horizon=horizon, num_trajectories=1, num_steps=actual_steps, seed=seed + start,
        )[:, 0].cpu())
    generated = torch.cat(generated)
    h = 0 if history is None else history.shape[1]
    prefix = reference[:, :0] if history is None else history.flip(1)
    truth = torch.cat((prefix, reference[:, :horizon + 1]), dim=1)
    prediction = torch.cat((prefix, generated), dim=1)
    indices = h + torch.arange(1, horizon + 1)[:, None] - torch.arange(h + 1)[None]
    errors = (prediction[:, indices] - truth[:, indices]).flatten(0, 1)
    conditioning = prediction[:, indices].flatten(0, 1)
    next_states = reference[:, 2:horizon + 2].flatten(0, 1)
    scale = getattr(model, f"{error_scale}_std").detach().cpu()
    scores = (errors / scale).square().mean(-1).sqrt().amax(-1)
    keep = torch.isfinite(errors).all(dim=(1, 2)) & (scores <= max_error_std)
    summary = dict(split="train", num_conditions=num_conditions, horizon=horizon, seed=seed,
                   sampling_steps=actual_steps or model.config.num_sampling_steps,
                   candidates=len(errors), accepted=int(keep.sum()), rejected=int((~keep).sum()),
                   max_error_std=max_error_std, error_scale=error_scale,
                   source_repetitions=names.tolist(), source_start_indices=starts.tolist())
    if not keep.any():
        raise ValueError(f"No finite empirical conditioning windows below the error limit: {summary}")
    summary.update(accepted_rms_median=float(scores[keep].median()),
                   accepted_rms_max=float(scores[keep].max()))
    replay = dict(current_state=conditioning[keep, 0], next_state=next_states[keep])
    if h:
        replay["history_states"] = conditioning[keep, 1:]
    return ErrorBank(errors[keep], replay, summary)


def perturb_batch(batch, bank, *, strength=1., probability=.5, generator=None):
    """Replay a whole error window onto selected rows of a clean training batch.

    The physical ``next_state`` stays fixed. Consequently an increment model's
    target becomes true-next minus perturbed-current, teaching recovery rather
    than adding the same erroneous displacement to both inputs and targets.
    Inputs are never modified in place. Sampling uses the supplied RNG if any.
    """
    if not math.isfinite(strength) or strength < 0 or not 0 <= probability <= 1:
        raise ValueError("Require finite strength >= 0 and probability in [0, 1]")
    current = batch["current_state"]
    history = batch.get("history_states")
    expected = (1 if history is None else history.shape[1] + 1, current.shape[-1])
    if tuple(bank.errors.shape[1:]) != expected:
        raise ValueError("Error-bank history length and coordinates must match the training batch")
    result = dict(batch)
    if strength == 0 or probability == 0:
        return result
    indices = _draw_indices(len(bank), len(current), bank.errors.device, generator)
    error = bank.errors[indices].to(device=current.device, dtype=current.dtype) * strength
    if probability < 1:
        draw_device = current.device if generator is None else generator.device
        selected = torch.rand(len(current), device=draw_device, generator=generator) < probability
        error = error * selected.to(current.device)[:, None, None]
    result["current_state"] = current + error[:, 0]
    if history is not None:
        result["history_states"] = history + error[:, 1:]
    return result
