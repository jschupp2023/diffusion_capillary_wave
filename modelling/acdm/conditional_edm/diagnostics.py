"""Small reproducible probes; no changes to training RNG or prepared-data files."""
import h5py
import numpy as np
import torch

from data_analysis.energy.capillary_energy import capillary_stiffness, length_scale
from modelling.acdm.experiments.rollout_benchmark import RolloutBenchmark
from .evaluate import reference_windows


def energy_report(generated, reference, stiffness, z_scale, n_space, factor):
    """Quadratic fluctuation energy; nonfinite trajectories count as failures."""
    def energies(values):
        coefficients = values[..., 1:].astype(np.float64) * z_scale
        with np.errstate(over="ignore", invalid="ignore"):
            return .5 * np.einsum("...i,ij,...j->...", coefficients, stiffness, coefficients, optimize=True)

    g, r = energies(generated), energies(reference)
    threshold = max(float(r.max()) * factor, np.finfo(float).tiny)
    finite = np.isfinite(generated).all(-1) & np.isfinite(g)
    failed = ~finite | (g > threshold)
    first = [int(np.flatnonzero(row)[0]) if row.any() else None for row in failed.reshape(-1, failed.shape[-1])]
    drift = np.abs(generated[..., 0] - generated[..., :1, 0]) * z_scale / np.sqrt(n_space)

    def number(value):
        return float(value) if np.isfinite(value) else None

    mean, maximum, reference_mean = g.mean(), g.max(), float(r.mean())
    return dict(mean_energy_J=number(mean), max_energy_J=number(maximum),
                reference_mean_energy_J=reference_mean, reference_max_energy_J=float(r.max()),
                mean_energy_over_reference=number(mean / reference_mean) if reference_mean > 0 else None,
                max_energy_over_reference_max=number(maximum / r.max()) if r.max() > 0 else None,
                energy_threshold_J=threshold, threshold_factor=factor,
                fraction_steps_failed=float(failed.mean()), fraction_trajectories_failed=float(failed.any(-1).mean()),
                finite_fraction=float(finite.mean()), first_failed_step=first,
                max_spatial_mean_drift_m=number(drift.max()))


def one_step_statistics(samples, current, target):
    """Teacher-forced conditional statistics in physical reduced coordinates."""
    if samples.ndim != 3 or samples.shape[0] != len(current) or samples.shape[2] != current.shape[1]:
        raise ValueError("samples must have shape [condition, ensemble, coordinate]")
    if current.shape != target.shape or current.ndim != 2 or current.shape[1] < 2:
        raise ValueError("current and target must be matching reduced states with a POD block")
    mean = samples.mean(1)
    variance = samples.var(1, correction=0)
    error2 = (mean - target).square()
    tiny = torch.finfo(samples.dtype).tiny

    def scalar(value):
        return float(value.detach().cpu())

    def ratio(spread, error):
        return scalar(spread.sum() / error.sum().clamp_min(tiny))

    current_pod, target_pod = current[:, 1:], target[:, 1:]
    mean_pod, variance_pod = mean[:, 1:], variance[:, 1:]
    pod_mean_map = mean_pod.square().sum(-1) - current_pod.square().sum(-1)
    pod_stochastic = variance_pod.sum(-1)
    pod_data = target_pod.square().sum(-1) - current_pod.square().sum(-1)
    mean_map = mean[:, 0].square() - current[:, 0].square()
    stochastic = variance[:, 0]
    mean_data = target[:, 0].square() - current[:, 0].square()
    return {
        "conditions": len(current),
        "ensemble_size": samples.shape[1],
        "coordinate_order": "sqrt(N)*spatial_mean, shared POD coefficients",
        "variance_estimator": "ensemble population variance (correction=0)",
        "conditional_mean_rmse": {
            "spatial_mean": scalar(error2[:, 0].mean().sqrt()),
            "pod": scalar(error2[:, 1:].mean().sqrt()),
            "full": scalar(error2.mean().sqrt()),
        },
        "spread_ratio": {
            "spatial_mean": ratio(variance[:, 0], error2[:, 0]),
            "pod": ratio(variance[:, 1:], error2[:, 1:]),
        },
        "pod_amplitude": {
            "model_change": scalar((pod_mean_map + pod_stochastic).mean()),
            "data_change": scalar(pod_data.mean()),
            "bias": scalar((pod_mean_map + pod_stochastic - pod_data).mean()),
            "mean_map_contribution": scalar(pod_mean_map.mean()),
            "stochastic_contribution": scalar(pod_stochastic.mean()),
        },
        "spatial_mean_amplitude": {
            "model_change": scalar((mean_map + stochastic).mean()),
            "data_change": scalar(mean_data.mean()),
            "bias": scalar((mean_map + stochastic - mean_data).mean()),
            "mean_map_contribution": scalar(mean_map.mean()),
            "stochastic_contribution": scalar(stochastic.mean()),
        },
    }


class TrainingDiagnostics:
    def __init__(self, data, model_config, config):
        self.data, self.model_config, self.config = data, model_config, config
        self.seed = config.seed + 100_000
        self.batch = None
        self.one_step_batch = None
        self.windows = {}
        self.stiffness = None
        self.rollout_benchmark = None
        if config.rollout_every and config.epochs >= config.rollout_every:
            history = model_config.history_steps if model_config.history_conditioning else 0
            required = (config.rollout_horizon + history) * model_config.lag_steps
            for split in ("train", "validation"):
                if max(data.frame_counts[n] for n in data.splits[split]) <= required:
                    raise ValueError(f"No {split} reference window is long enough; reduce --rollout-horizon.")
            if config.rollout_horizon >= 4:
                self.rollout_benchmark = RolloutBenchmark(
                    data, horizon=config.rollout_horizon,
                    conditions=len(data.splits["validation"]),
                    ensemble_size=config.rollout_ensemble_size, seed=self.seed,
                    sampling_steps=model_config.num_sampling_steps,
                    max_history_steps=model_config.history_steps if model_config.history_conditioning else 0,
                    nperseg=min(256, config.rollout_horizon + 1),
                    energy_factor=config.rollout_energy_factor,
                    surface_tension=config.surface_tension)
            if config.one_step_conditions:
                reference, history, _, _, _ = reference_windows(
                    data, model_config, "validation", config.one_step_conditions, 1, self.seed)
                self.one_step_batch = dict(current_state=reference[:, 0], next_state=reference[:, 1])
                if history is not None:
                    self.one_step_batch["history_states"] = history
        if config.fixed_mse_batch_size:
            reference, history, _, _, _ = reference_windows(
                data, model_config, "validation", config.fixed_mse_batch_size, 1, self.seed)
            self.batch = dict(current_state=reference[:, 0], next_state=reference[:, 1])
            if history is not None:
                self.batch["history_states"] = history

    @torch.no_grad()
    def fixed_mse(self, model):
        if self.batch is None:
            return {}
        device = model.state_mean.device
        batch = {key: value.to(device) for key, value in self.batch.items()}
        was_training = model.training
        model.eval()
        try:
            result = {}
            if model.config.diffusion_formulation == "ddpm":
                for step in (1, 10, 20):
                    generator = torch.Generator(device=device).manual_seed(self.seed)
                    _, metrics = model.loss(batch, diffusion_step=step, generator=generator)
                    result[f"fixed_MSE_r{step}"] = float(metrics["mean_unweighted_mse"])
                return result
            for label, sigma in zip(("low", "mid", "high"), self.config.fixed_sigmas):
                generator = torch.Generator(device=device).manual_seed(self.seed)
                _, metrics = model.loss(batch, sigma=sigma, generator=generator)
                result[f"fixed_MSE_{label}_sigma"] = float(metrics["mean_unweighted_mse"])
            return result
        finally:
            model.train(was_training)

    @torch.no_grad()
    def one_step(self, model):
        if self.one_step_batch is None:
            return {}
        device = model.state_mean.device
        batch = {key: value.to(device) for key, value in self.one_step_batch.items()}
        samples = model.sample_next(
            batch["current_state"], history_states=batch.get("history_states"),
            num_samples=self.config.one_step_ensemble_size, seed=self.seed)
        return one_step_statistics(samples, batch["current_state"], batch["next_state"])

    def rollout(self, model):
        if self.stiffness is None:
            with h5py.File(self.data.basis_path, "r") as basis:
                modes = basis["pod/modes"][:self.data.rank]
                grids = [basis[f"grid/{a}"][:] * length_scale(str(basis[f"grid/{a}"].attrs["units"]))
                         for a in ("x", "y")]
            self.stiffness, _ = capillary_stiffness(modes, *grids, gamma=self.config.surface_tension)
        sampling_steps = (len(model.ddpm_betas) if model.config.diffusion_formulation == "ddpm"
                          else model.config.num_sampling_steps)
        result = dict(horizon=self.config.rollout_horizon, ensemble_size=self.config.rollout_ensemble_size,
                      sampling_steps=sampling_steps,
                      duration=self.config.rollout_horizon * model.config.physical_lag,
                      time_units=self.data.time_units, gamma_N_per_m=self.config.surface_tension,
                      energy="quadratic shared-rank fluctuation energy; temporal mean field omitted")
        for split in ("train", "validation"):
            if split not in self.windows:
                self.windows[split] = reference_windows(self.data, self.model_config, split, 1,
                                                        self.config.rollout_horizon, self.seed)
            reference, history, _, names, starts = self.windows[split]
            device = model.state_mean.device
            generated = model.rollout(reference[:, 0].to(device),
                                      history_states=None if history is None else history.to(device),
                                      horizon=self.config.rollout_horizon,
                                      num_trajectories=self.config.rollout_ensemble_size, seed=self.seed).cpu().numpy()
            result[split] = energy_report(generated, reference.numpy(), self.stiffness,
                                          length_scale(self.data.signal_units), self.data.n_space,
                                          self.config.rollout_energy_factor)
            result[split].update(repetition=str(names[0]), start_index=int(starts[0]))
            if split == "validation" and self.rollout_benchmark is not None:
                result["benchmark"] = self.rollout_benchmark._metrics(
                    generated, reference.numpy(), self.data.native_dt * model.config.lag_steps)
        return result
