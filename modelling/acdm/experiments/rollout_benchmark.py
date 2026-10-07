"""Reproducible validation-only screening of physical rollout statistics.

The score is a screening heuristic, not an estimate of forecasting skill. It
penalizes both excess and missing energy/power. Never use test repetitions for
model selection. Short windows have limited spectral resolution.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from data_analysis.energy.capillary_energy import capillary_stiffness, length_scale
from data_analysis.psd.pod_center_psd import compute_welch


class RolloutBenchmark:
    def __init__(self, data, horizon=512, conditions=4, ensemble_size=2, seed=814,
                 sampling_steps=32, max_history_steps=64, nperseg=256,
                 energy_factor=100., surface_tension=.0728):
        self.data, self.horizon, self.conditions = data, horizon, conditions
        self.ensemble_size, self.seed = ensemble_size, seed
        self.sampling_steps, self.max_history_steps = sampling_steps, max_history_steps
        self.nperseg, self.energy_factor = nperseg, energy_factor
        self.names = tuple(data.splits.get("validation", ()))
        if (horizon < 4 or conditions < len(self.names) or not self.names
                or min(ensemble_size, sampling_steps) < 1 or max_history_steps < 0
                or nperseg < 4 or energy_factor <= 1):
            raise ValueError("Need positive counts, horizon >=4, and enough conditions to cover validation repetitions.")
        self.z_scale = length_scale(data.signal_units)
        with h5py.File(data.basis_path, "r") as basis:
            modes = basis["pod/modes"][:data.rank].astype(np.float64)
            grids = [basis[f"grid/{a}"][:] * length_scale(str(basis[f"grid/{a}"].attrs["units"]))
                     for a in ("x", "y")]
        self.stiffness, _ = capillary_stiffness(modes, *grids, gamma=surface_tension)
        self.center_modes = modes[:, modes.shape[1] // 2, modes.shape[2] // 2]
        self.surface_tension = surface_tension
        self._windows = {}

    def reference_windows(self, config):
        """Stratified starts in every validation repetition; independent of history length."""
        lag = config.lag_steps
        history = config.history_steps if config.history_conditioning else 0
        if history > self.max_history_steps:
            raise ValueError("History exceeds benchmark reservation; construct with a larger max_history_steps.")
        key = (lag, history)
        if key in self._windows:
            return self._windows[key]
        rng = np.random.default_rng(self.seed)
        references, histories, times, names, starts = [], [], [], [], []
        for rep, name in enumerate(self.names):
            count = self.conditions // len(self.names) + (rep < self.conditions % len(self.names))
            first = self.max_history_steps * lag
            available = self.data.frame_counts[name] - self.horizon * lag - first
            if available < count:
                raise ValueError(f"Validation repetition {name} is too short for the reserved history and horizon.")
            # Spread starts across each repetition, with a deterministic jitter inside each stratum.
            fractions = (np.arange(count) + rng.uniform(.2, .8, count)) / count
            for start in first + np.floor(fractions * available).astype(int):
                t, values = self.data.read_coordinates(name, int(start - history * lag),
                                                      int(start + self.horizon * lag + 1))
                offset = history * lag
                indices = offset + lag * np.arange(self.horizon + 1)
                references.append(values[indices])
                histories.append(values[offset - lag * np.arange(1, history + 1)])
                times.append(t[indices])
                names.append(name)
                starts.append(int(start))
        result = (np.stack(references), np.stack(histories) if history else None,
                  np.stack(times), np.asarray(names), np.asarray(starts))
        self._windows[key] = result
        return result

    def _energies(self, coordinates):
        coefficients = coordinates[..., 1:].astype(np.float64) * self.z_scale
        with np.errstate(over="ignore", invalid="ignore"):
            values = .5 * np.einsum("...i,ij,...j->...", coefficients, self.stiffness,
                                    coefficients, optimize=True)
        return np.maximum(values, 0)

    @staticmethod
    def _ratio(numerator, denominator):
        # Finite clipping ensures failures cannot disappear through NaN aggregation.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratio = np.asarray(numerator) / np.maximum(denominator, np.finfo(float).tiny)
        return np.clip(np.nan_to_num(ratio, nan=1e12, posinf=1e12, neginf=1e12), 1e-12, 1e12)

    def _metrics(self, generated, reference, dt):
        g, r = self._energies(generated)[..., 1:], self._energies(reference)[..., 1:]
        finite = np.isfinite(generated).all((-1, -2)) & np.isfinite(g).all(-1)
        threshold = self.energy_factor * np.maximum(r.max(-1), np.finfo(float).tiny)
        explosive = ~finite | (g > threshold[:, None, None]).any(-1)
        # A failed trajectory is assigned a large energy penalty, never dropped.
        g = np.where(finite[..., None], g, 1e12 * np.maximum(r.mean(-1), 1e-100)[:, None, None])
        ratios = self._ratio(g.mean(-1), r.mean(-1)[:, None])
        energy_ratio = float(self._ratio(g.mean(), r.mean()))
        energy = dict(mean_ratio=energy_ratio, median_ratio=float(np.median(ratios)),
                      q90_ratio=float(np.quantile(ratios, .9)), q99_ratio=float(np.quantile(ratios, .99)),
                      trajectory_mean_ratios=ratios.tolist(),
                      trajectory_q95_ratios=self._ratio(np.quantile(g, .95, axis=-1),
                                                         np.quantile(r, .95, axis=-1)[:, None]).tolist(),
                      reference_mean_J=float(r.mean()), threshold_factor=self.energy_factor)
        signals = {}
        for values, label in ((generated, "generated"), (reference, "reference")):
            mean = values[..., 0].astype(np.float64) / np.sqrt(self.data.n_space)
            with np.errstate(over="ignore", invalid="ignore"):
                fluctuation = values[..., 1:].astype(np.float64) @ self.center_modes
            signals[label] = dict(center=mean + fluctuation, spatial_mean=mean,
                                  center_without_spatial_mean=fluctuation)
        spectra, spectral_errors = {}, []
        for name in signals["reference"]:
            reference_psd = [compute_welch(v, 1 / dt, self.nperseg, .5)[0]
                             for v in signals["reference"][name]]
            frequency = reference_psd[0].frequency
            r_psd = np.stack([v.density for v in reference_psd])
            floor = max(float(r_psd.max()) * 1e-10, np.finfo(float).tiny)
            generated_psd = []
            failed_psd = 0
            for signal in signals["generated"][name].reshape(-1, generated.shape[-2]):
                try:
                    if not np.isfinite(signal).all():
                        raise ValueError("Nonfinite trajectory")
                    generated_psd.append(compute_welch(signal, 1 / dt, self.nperseg, .5)[0].density)
                except ValueError:
                    generated_psd.append(1e12 * (r_psd.mean(0) + floor))
                    failed_psd += 1
            g_psd = np.stack(generated_psd).mean(0)
            r_psd = r_psd.mean(0)
            ratio = self._ratio(g_psd + floor, r_psd + floor)
            # Exclude DC; reference-relative floor limits influence of numerically empty bins.
            log_error = float(np.mean(np.abs(np.log10(ratio[1:]))))
            spectral_errors.append(log_error)
            bands = {}
            nyquist = .5 / dt
            for band, low, high in (("low", 0., .1), ("mid", .1, .5), ("high", .5, 1.)):
                mask = (frequency > low * nyquist) & (frequency <= high * nyquist)
                bands[band] = float(self._ratio(g_psd[mask].sum() + floor, r_psd[mask].sum() + floor)) if mask.any() else None
            spectra[name] = dict(power_ratio=float(self._ratio(g_psd.sum() + floor, r_psd.sum() + floor)),
                                 mean_absolute_log10_error=log_error, band_power_ratios=bands,
                                 failed_psd_fraction=failed_psd / (generated.shape[0] * generated.shape[1]))
        # Symmetric log terms penalize collapse as strongly as excess power.
        energy_error = float(np.mean(np.abs(np.log10(ratios))))
        energy["mean_absolute_log10_error"] = energy_error
        score = energy_error + float(np.mean(spectral_errors)) + 100 * (~finite).mean() + 20 * explosive.mean()
        return dict(score=float(score), energy=energy, psd=spectra,
                    nonfinite_fraction=float((~finite).mean()), explosive_fraction=float(explosive.mean()),
                    frequency_spacing_hz=float(frequency[1] - frequency[0]),
                    score_definition="mean abs log10 per-trajectory energy ratio + mean log10 PSD error + 100*nonfinite fraction + 20*explosive fraction")

    @torch.no_grad()
    def evaluate(self, model, *, save_dir=None):
        reference, history, times, names, starts = self.reference_windows(model.config)
        device = model.state_mean.device
        was_training = model.training
        model.eval()
        sampling_steps = (len(model.ddpm_betas)
                          if getattr(model.config, "diffusion_formulation", "edm") == "ddpm"
                          else self.sampling_steps)
        try:
            generated = model.rollout(torch.as_tensor(reference[:, 0], dtype=torch.float32, device=device),
                                      history_states=None if history is None else torch.as_tensor(history, dtype=torch.float32, device=device),
                                      horizon=self.horizon, num_trajectories=self.ensemble_size,
                                      num_steps=sampling_steps, seed=self.seed).cpu().numpy()
        finally:
            model.train(was_training)
        dt = self.data.native_dt * model.config.lag_steps
        report = self._metrics(generated, reference, dt)
        report.update(split="validation", seed=self.seed, horizon=self.horizon, conditions=self.conditions,
                      ensemble_size=self.ensemble_size, sampling_steps=sampling_steps,
                      repetition=names.tolist(), start_index=starts.tolist(),
                      history_steps=model.config.history_steps if model.config.history_conditioning else 0,
                      lag_steps=model.config.lag_steps, native_dt=self.data.native_dt, physical_lag=dt,
                      duration=self.horizon * dt, time_units=self.data.time_units,
                      source_shared_basis=str(self.data.basis_path), rank=self.data.rank,
                      gamma_N_per_m=self.surface_tension, energy_definition="quadratic; temporal mean field omitted",
                      psd_bands="(0,0.1], (0.1,0.5], (0.5,1] times Nyquist; DC omitted from spectral error",
                      coordinate_order="sqrt(N)*spatial_mean, shared POD coefficients")
        if save_dir is not None:
            directory = Path(save_dir)
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / "rollout.npz", trajectories=generated, reference=reference,
                                reference_time=times, repetition=names, start_index=starts)
            (directory / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
