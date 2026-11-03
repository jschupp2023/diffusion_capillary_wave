"""Diagnose finite-particle error in a trained multistep mixture likelihood."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.acdm.conditional_edm.train import resolve_device

from .losses import MonteCarloMultistepNLL
from .train import (_capped_multistep_batches, _validated_multistep_mode_counts,
                    _validated_multistep_particle_counts, load_model)


_DIAGNOSTIC_NAMES = (
    "log_likelihood",
    "effective_sample_size",
    "effective_sample_fraction",
    "coefficient_of_variation_squared",
    "relative_standard_error",
    "largest_component_fraction",
)


def _distribution_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    report = {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
    }
    if not finite.size:
        report.update({name: None for name in (
            "mean", "standard_deviation", "minimum", "p01", "p10",
            "median", "p90", "p99", "maximum")})
        return report
    quantiles = np.quantile(finite, (0, .01, .1, .5, .9, .99, 1))
    report.update(
        mean=float(finite.mean()),
        standard_deviation=float(finite.std()),
        minimum=float(quantiles[0]),
        p01=float(quantiles[1]),
        p10=float(quantiles[2]),
        median=float(quantiles[3]),
        p90=float(quantiles[4]),
        p99=float(quantiles[5]),
        maximum=float(quantiles[6]),
    )
    return report


def _resolve_checkpoint_path(path: Path) -> Path:
    path = Path(path)
    return path / "best.pt" if path.is_dir() else path


def _resolve_settings(checkpoint: dict, rank: int, *,
                      horizons: tuple[int, ...] | None,
                      target: str | None,
                      mode_counts: tuple[int, ...] | None,
                      particles: int | None,
                      particle_counts: tuple[int, ...] | None
                      ) -> tuple[tuple[int, ...], str, tuple[int, ...] | None,
                                 tuple[int, ...]]:
    training = checkpoint.get("training", {})
    saved_horizons = training.get("multistep_horizons")
    saved_horizons = None if saved_horizons is None else list(saved_horizons)
    if horizons is None:
        if not saved_horizons:
            raise ValueError(
                "Checkpoint has no saved multistep horizons; supply --horizons")
        horizons = tuple(saved_horizons)
    else:
        horizons = tuple(horizons)
    if (not horizons
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                   for value in horizons)
            or tuple(sorted(set(horizons))) != horizons):
        raise ValueError("Horizons must be strictly increasing positive integers")

    saved_target = training.get("multistep_target", "velocity")
    target = target or saved_target
    if target not in {"velocity", "displacement"}:
        raise ValueError("Target must be velocity or displacement")

    if (mode_counts is None and list(horizons) == saved_horizons
            and target == saved_target):
        saved_modes = training.get("multistep_mode_counts")
        mode_counts = None if saved_modes is None else tuple(saved_modes)
    mode_counts = _validated_multistep_mode_counts(
        mode_counts, horizons, rank, target)

    if particle_counts is None and particles is None and list(horizons) == saved_horizons:
        saved_counts = training.get("multistep_particle_counts")
        if saved_counts is not None:
            particle_counts = tuple(saved_counts)
    if particles is None:
        particles = int(training.get("multistep_particles", 100))
    particle_counts = _validated_multistep_particle_counts(
        particle_counts, horizons, particles)
    return horizons, target, mode_counts, particle_counts


def _empirical_seed_statistics(log_likelihood: np.ndarray,
                               predicted_rse: np.ndarray) -> dict:
    """Compare within-cloud RSE estimates with independent-seed variation."""
    seed_count, window_count = log_likelihood.shape
    seed_mean_nll = np.full(seed_count, np.nan, dtype=np.float64)
    for index, values in enumerate(log_likelihood):
        finite = values[np.isfinite(values)]
        if finite.size:
            seed_mean_nll[index] = -finite.mean()
    report = {"mean_nll_by_seed": seed_mean_nll.tolist(),
              "mean_nll_across_seeds": _distribution_summary(seed_mean_nll)}
    if seed_count < 2:
        report.update(
            empirical_log_likelihood_sd=None,
            empirical_likelihood_relative_sd=None,
            empirical_to_predicted_rse_ratio=None,
            note="At least two seeds are required for empirical seed dispersion.",
        )
        return report

    valid = np.isfinite(log_likelihood).all(axis=0)
    log_sd = np.full(window_count, np.nan, dtype=np.float64)
    likelihood_rsd = np.full(window_count, np.nan, dtype=np.float64)
    if valid.any():
        logs = log_likelihood[:, valid]
        log_sd[valid] = logs.std(axis=0, ddof=1)
        shifted = np.exp(logs - logs.max(axis=0, keepdims=True))
        likelihood_rsd[valid] = (
            shifted.std(axis=0, ddof=1) / shifted.mean(axis=0))
    predicted = np.nanmean(predicted_rse, axis=0)
    ratio = np.divide(
        likelihood_rsd, predicted,
        out=np.full_like(likelihood_rsd, np.nan),
        where=np.isfinite(likelihood_rsd) & np.isfinite(predicted) & (predicted > 0),
    )
    report.update(
        empirical_log_likelihood_sd=_distribution_summary(log_sd),
        empirical_likelihood_relative_sd=_distribution_summary(likelihood_rsd),
        empirical_to_predicted_rse_ratio=_distribution_summary(ratio),
        note=("Empirical dispersion uses independent particle seeds on identical windows; "
              "the ratio compares it with the mean within-cloud RSE estimate."),
    )
    return report


def diagnose(checkpoint_path: Path, output: Path, *, split: str = "validation",
             batch_size: int = 256, max_windows: int = 10_000,
             window_seed: int = 0, seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
             device: torch.device | str = "cpu", data_root: Path | None = None,
             shared_basis: Path | None = None,
             horizons: tuple[int, ...] | None = None,
             target: str | None = None,
             mode_counts: tuple[int, ...] | None = None,
             particles: int | None = None,
             particle_counts: tuple[int, ...] | None = None) -> Path:
    """Evaluate weight degeneracy on fixed retained windows and save JSON/NPZ."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation, or test")
    if batch_size < 1 or max_windows < 1:
        raise ValueError("Batch size and max windows must be positive")
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)):
        raise ValueError("Seeds must be a nonempty sequence of distinct integers")

    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    device = torch.device(device)
    model, checkpoint = load_model(checkpoint_path, device)
    if model.config.state_variable != "velocity":
        raise ValueError("Multistep particle diagnostics require a velocity checkpoint")
    data = checkpoint_data(
        checkpoint, data_root=data_root, shared_basis=shared_basis)
    horizons, target, mode_counts, particle_counts = _resolve_settings(
        checkpoint, data.rank, horizons=horizons, target=target,
        mode_counts=mode_counts, particles=particles,
        particle_counts=particle_counts)
    objective = MonteCarloMultistepNLL(
        horizons, particle_counts[0], target=target, mode_counts=mode_counts,
        pod_coordinate_offset=int(data.include_spatial_mean),
        particle_counts=particle_counts)

    training = checkpoint.get("training", {})
    trim_cutoff = training.get("train_trim_norm_cutoff")
    trim_state_std = training.get("train_trim_state_std")
    batches = []
    target_key = "target_displacements" if target == "displacement" else "target_velocities"
    for batch in _capped_multistep_batches(
            data, model, split, batch_size, horizons, max_windows, window_seed,
            trim_cutoff, trim_state_std):
        batches.append({name: batch[name] for name in (
            "current_state", "history_states", target_key)})
    if not batches:
        raise ValueError("No retained multistep windows were available")
    window_count = sum(len(batch["current_state"]) for batch in batches)

    stored: dict[str, np.ndarray] = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "horizons": np.asarray(horizons, dtype=np.int64),
        "mode_counts": np.asarray(
            mode_counts if mode_counts is not None else
            (data.n_coordinates,) * len(horizons), dtype=np.int64),
        "particle_counts": np.asarray(particle_counts, dtype=np.int64),
        "current_state": np.concatenate([
            batch["current_state"].numpy() for batch in batches]),
        "targets": np.concatenate([batch[target_key].numpy() for batch in batches]),
    }
    collected = {
        horizon: {name: [] for name in _DIAGNOSTIC_NAMES}
        for horizon in horizons
    }
    with torch.no_grad():
        for seed in seeds:
            per_seed = {
                horizon: {name: [] for name in _DIAGNOSTIC_NAMES}
                for horizon in horizons
            }
            generator = torch.Generator(device=device).manual_seed(seed)
            for batch in batches:
                diagnostics: dict[int, dict[str, torch.Tensor]] = {}
                objective.per_horizon(
                    model, batch["current_state"].to(device),
                    batch[target_key].to(device),
                    history_states=batch["history_states"].to(device),
                    generator=generator, particle_diagnostics=diagnostics)
                for horizon in horizons:
                    for name in _DIAGNOSTIC_NAMES:
                        per_seed[horizon][name].append(
                            diagnostics[horizon][name].detach().cpu().numpy())
            for horizon in horizons:
                for name in _DIAGNOSTIC_NAMES:
                    collected[horizon][name].append(
                        np.concatenate(per_seed[horizon][name]))

    per_horizon_report = {}
    for index, horizon in enumerate(horizons):
        arrays = {}
        for name in _DIAGNOSTIC_NAMES:
            values = np.stack(collected[horizon][name])
            arrays[name] = values
            stored[f"h{horizon}_{name}"] = values
        per_horizon_report[str(horizon)] = {
            "scored_dimension": (data.n_coordinates if mode_counts is None
                                 else mode_counts[index]),
            "particle_count": 1 if horizon == 1 else particle_counts[index],
            "distributions_over_seeds_and_windows": {
                name: _distribution_summary(values)
                for name, values in arrays.items()
            },
            "seed_stability": _empirical_seed_statistics(
                arrays["log_likelihood"], arrays["relative_standard_error"]),
        }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "per_window_diagnostics.npz", **stored)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "split": split,
        "window_count": window_count,
        "window_seed": window_seed,
        "particle_seeds": list(seeds),
        "target": target,
        "target_definition": (
            "a[n + H*lag_steps] - a[n]" if target == "displacement" else
            "a[n + H*lag_steps] - a[n + H*lag_steps - 1]"),
        "horizons": list(horizons),
        "mode_counts": None if mode_counts is None else list(mode_counts),
        "particle_counts": list(particle_counts),
        "coordinate_selection": (
            "all reduced coordinates" if mode_counts is None else
            "first N shared-POD modes at each horizon; spatial mean excluded"),
        "lag_steps": model.config.lag_steps,
        "training_trim_applied": trim_cutoff is not None,
        "normalized_increment_norm_cutoff": trim_cutoff,
        "trim_rule": (
            "every native increment from required prehistory through the farthest endpoint"
            if trim_cutoff is not None
            else None),
        "estimator_conventions": {
            "weight": "final marginal Gaussian density at the observed endpoint",
            "effective_sample_size": "1 / sum(alpha**2)",
            "coefficient_of_variation_squared": "M / ESS - 1 (variance divisor M)",
            "relative_standard_error": "sqrt(CV**2 / M)",
            "largest_component_fraction": "max(alpha)",
        },
        "per_horizon": per_horizon_report,
        "array_file": "per_window_diagnostics.npz",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint file or run directory containing best.pt.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=10_000)
    parser.add_argument("--window-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--target", choices=("velocity", "displacement"))
    parser.add_argument("--mode-counts", type=int, nargs="+")
    parser.add_argument(
        "--particles", type=int,
        help="Use this fixed particle count at every horizon instead of the saved schedule.")
    parser.add_argument(
        "--particle-counts", type=int, nargs="+",
        help="Explicit nonincreasing particle count at each horizon.")
    args = parser.parse_args(argv)
    output = diagnose(
        args.checkpoint, args.output, split=args.split,
        batch_size=args.batch_size, max_windows=args.max_windows,
        window_seed=args.window_seed, seeds=tuple(args.seeds),
        device=resolve_device(args.device), data_root=args.data_root,
        shared_basis=args.shared_basis,
        horizons=None if args.horizons is None else tuple(args.horizons),
        target=args.target,
        mode_counts=None if args.mode_counts is None else tuple(args.mode_counts),
        particles=args.particles,
        particle_counts=(None if args.particle_counts is None else
                         tuple(args.particle_counts)),
    )
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
