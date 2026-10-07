"""Quantitative energy, spectrum, and moment metrics for saved rollouts.

This complements the plots produced by :mod:`data_analysis.plot_rollout` with
one machine-readable summary. Energy and marginal coefficient moments compare
complete generated trajectories against both complete references and reference
frames retained by the checkpoint's training-increment rule. Generated paths
are never trimmed. PSDs always use complete, uniformly sampled trajectories.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data_analysis.energy.capillary_energy import capillary_stiffness, length_scale
from data_analysis.energy.rollout_energy import compute_energies
from data_analysis.psd.compare_psd_band_power import integrate_band, logarithmic_edges
from data_analysis.psd.rollout_psd import ensemble_psd, fluctuation_signals, point_signals
from data_analysis.psd.pod_center_psd import select_point
from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory


def training_trim_specification(rollout):
    """Load the exact saved training cutoff and frozen pre-trim state scale."""
    policy = rollout.metadata.get("reference_window_trim_policy", {})
    if not policy.get("training_trim_available", False):
        return None
    cutoff = policy.get("normalized_increment_norm_cutoff",
                        rollout.metadata.get("train_trim_norm_cutoff"))
    config_candidates = []
    checkpoint = rollout.metadata.get("checkpoint")
    if checkpoint:
        config_candidates.append(Path(checkpoint).expanduser().resolve().parent / "config.json")
    config_candidates.append(rollout.path.parent.parent / "config.json")
    config_path = next((path for path in config_candidates if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(
            "Training trim is recorded, but config.json with train_trim_state_std "
            "could not be found beside the checkpoint or rollout parent.")
    config = json.loads(config_path.read_text())
    training = config.get("training", {})
    scale = np.asarray(training.get("train_trim_state_std"), dtype=float)
    config_cutoff = training.get("train_trim_norm_cutoff")
    if (cutoff is None or config_cutoff is None or not np.isclose(cutoff, config_cutoff)
            or scale.shape != (rollout.generated.shape[-1],)
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Saved training-trim cutoff or pre-trim state scale is missing or inconsistent.")
    state_variable = str(rollout.metadata.get(
        "state_variable", config.get("model", {}).get("state_variable", "state")))
    training_lag = int(rollout.metadata.get(
        "training_lag_steps", config.get("model", {}).get("lag_steps", 1)))
    rollout_lag = int(rollout.metadata.get("lag_steps", training_lag))
    if rollout_lag != training_lag:
        raise ValueError(
            "Reference-frame training trim requires rollout lag to match the checkpoint training lag.")
    return dict(
        cutoff=float(cutoff),
        state_std=scale,
        train_trim_percent=float(training.get(
            "train_trim_percent", policy.get("train_trim_percent", 0.))),
        state_variable=state_variable,
        lag_steps=training_lag,
        config_path=str(config_path),
        data_config=config.get("data", {}),
    )


def velocity_training_trim_mask_from_native(native_paths, sample_count,
                                            specification):
    """Mask coarse velocity frames using every native increment in their stencil."""
    native_paths = np.asarray(native_paths, dtype=float)
    lag = specification["lag_steps"]
    expected_width = (sample_count - 1) * lag + 2
    if (native_paths.ndim != 3
            or native_paths.shape[1:] != (expected_width, len(specification["state_std"]))):
        raise ValueError("Native reference paths do not match rollout length, lag, or coordinates.")
    normalized_increment = np.diff(native_paths, axis=1) / specification["state_std"]
    good = np.linalg.norm(normalized_increment, axis=-1) <= specification["cutoff"]
    bad_prefix = np.concatenate(
        (np.zeros((len(good), 1), dtype=np.int64),
         np.cumsum(~good, axis=1, dtype=np.int64)), axis=1)
    starts = lag * np.arange(sample_count - 1)
    bad_counts = bad_prefix[:, starts + lag + 1] - bad_prefix[:, starts]
    keep = np.zeros((len(native_paths), sample_count), dtype=bool)
    keep[:, :-1] = bad_counts == 0
    return keep


def velocity_reference_training_trim_mask(rollout, specification):
    """Recover native source paths and apply the exact lag-k velocity trim."""
    if rollout.start_index is None:
        raise ValueError(
            "Exact coarse-lag velocity trimming requires start_index in rollout.npz.")
    from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData

    saved = specification["data_config"]
    required = ("experiment", "rank", "data_root", "validation_reps", "test_reps")
    if any(name not in saved for name in required):
        raise ValueError("Saved data configuration is incomplete for native reference recovery.")
    data = SharedPODTrainingData(
        saved["experiment"], int(saved["rank"]), data_root=saved["data_root"],
        shared_basis=rollout.basis_path,
        validation_reps=saved["validation_reps"], test_reps=saved["test_reps"],
        include_spatial_mean=rollout.include_spatial_mean,
        spatial_mean_highpass_hz=rollout.spatial_mean_highpass_hz,
    )
    if data.n_coordinates != rollout.reference.shape[-1]:
        raise ValueError("Recovered source coordinate count does not match rollout.")
    lag = specification["lag_steps"]
    sample_count = rollout.reference.shape[1]
    native_paths = []
    current_offsets = 1 + lag * np.arange(sample_count)
    for condition, (name, start) in enumerate(
            zip(rollout.repetitions, rollout.start_index, strict=True)):
        start = int(start)
        _, values = data.read_coordinates(
            str(name), start - 1, start + (sample_count - 1) * lag + 1)
        if not np.allclose(values[current_offsets], rollout.reference[condition],
                           rtol=1e-5, atol=1e-5):
            raise ValueError("Recovered native reference path disagrees with saved rollout samples.")
        native_paths.append(values)
    keep = velocity_training_trim_mask_from_native(
        np.stack(native_paths), sample_count, specification)
    report = _trim_report(
        keep, specification,
        "complete native velocity stencil from the previous frame through the lagged endpoint; all lag+1 increments pass",
        len(rollout.reference),
    )
    report.update(native_intermediate_frames_recovered=True,
                  source_shared_basis=str(rollout.basis_path))
    return keep, report


def _trim_report(keep, specification, rule, boundary_frames):
    return dict(
        applied_to="reference only; generated trajectories are never trimmed",
        state_variable=specification["state_variable"],
        lag_steps=specification["lag_steps"],
        train_trim_percent=specification["train_trim_percent"],
        normalized_increment_norm_cutoff=specification["cutoff"],
        normalization_scale_source="saved pre-trim training state standard deviation",
        config_path=specification["config_path"],
        frame_rule=rule,
        total_reference_frames=int(keep.size),
        retained_reference_frames=int(keep.sum()),
        removed_reference_frames=int((~keep).sum()),
        removed_reference_frame_percent=100 * float((~keep).mean()),
        incomplete_boundary_frames=int(boundary_frames),
    )


def reference_training_trim_mask(reference, specification):
    """Apply the checkpoint's increment cutoff to reference frames only.

    State/increment training retains a current frame when its outgoing modeled
    increment passes the cutoff. Velocity training uses a previous/current/next
    stencil, so an interior current frame is retained only when both adjacent
    native increments pass. Boundary frames without a complete stencil are not
    included in the trimmed reference statistic.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 3 or reference.shape[1] < 3:
        raise ValueError("Reference trim requires [condition, time, coordinate] with at least three times.")
    normalized_increment = np.diff(reference, axis=1) / specification["state_std"]
    norms = np.linalg.norm(normalized_increment, axis=-1)
    good = norms <= specification["cutoff"]
    keep = np.zeros(reference.shape[:-1], dtype=bool)
    if specification["state_variable"] == "velocity":
        keep[:, 1:-1] = good[:, :-1] & good[:, 1:]
        rule = "complete previous/current/next velocity stencil; both adjacent native increments pass"
        boundary_frames = 2 * reference.shape[0]
    else:
        keep[:, :-1] = good
        rule = "current frame whose outgoing modeled-lag increment passes"
        boundary_frames = reference.shape[0]
    if keep.sum() < 2:
        raise ValueError("Fewer than two reference frames remain under the saved training trim.")
    report = _trim_report(keep, specification, rule, boundary_frames)
    return keep, report


def reference_training_trim_population(rollout):
    """Return the exact saved training-trim mask/report for a rollout reference.

    The result is cached on the in-memory ``Rollout`` so the combined plotting
    command does not recover the same native velocity path separately for the
    modal and quantitative diagnostics.
    """
    cache_name = "_reference_training_trim_population"
    if hasattr(rollout, cache_name):
        return getattr(rollout, cache_name)
    specification = training_trim_specification(rollout)
    if specification is None:
        result = (None, None)
    elif specification["state_variable"] == "velocity":
        result = velocity_reference_training_trim_mask(rollout, specification)
    else:
        result = reference_training_trim_mask(rollout.reference, specification)
    setattr(rollout, cache_name, result)
    return result


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _energy_population(values, keep_mask=None):
    values = np.asarray(values, dtype=float)
    selected = values.reshape(-1) if keep_mask is None else values[keep_mask]
    return dict(mean_J=float(selected.mean()), frame_count=int(len(selected)))


def energy_comparison(generated_energy, reference_energy, reference_mask=None,
                      trim_report=None):
    generated = _energy_population(generated_energy)
    reference = _energy_population(reference_energy)
    report = dict(
        definition="quadratic capillary deformation energy; spatial mean omitted because its gradient is zero",
        units="J",
        untrimmed=dict(
            generated=generated,
            reference=reference,
            generated_over_reference=_ratio(generated["mean_J"], reference["mean_J"]),
        ),
        generated_untrimmed_vs_reference_training_trimmed=None,
    )
    if reference_mask is not None:
        reference_trimmed = _energy_population(reference_energy, reference_mask)
        report["generated_untrimmed_vs_reference_training_trimmed"] = dict(
            generated=generated,
            reference=reference_trimmed,
            generated_over_reference=_ratio(
                generated["mean_J"], reference_trimmed["mean_J"]),
            reference_trim=trim_report,
        )
    return report


def _population_moments(samples, keep_mask=None, chunk_size=65536):
    """Accumulate moments without copying a rollout-sized float64 array."""
    samples = np.asarray(samples)
    if samples.ndim < 2 or chunk_size < 1:
        raise ValueError("Moment samples must include sample and coordinate axes.")
    flat = samples.reshape(-1, samples.shape[-1])
    mask = None if keep_mask is None else np.asarray(keep_mask, dtype=bool).reshape(-1)
    if mask is not None and len(mask) != len(flat):
        raise ValueError("Moment mask must match all non-coordinate sample axes.")
    total = np.zeros(flat.shape[1], dtype=float)
    second = np.zeros((flat.shape[1], flat.shape[1]), dtype=float)
    count = 0
    for start in range(0, len(flat), chunk_size):
        batch = np.asarray(flat[start:start + chunk_size], dtype=float)
        if mask is not None:
            batch = batch[mask[start:start + chunk_size]]
        if not np.isfinite(batch).all():
            raise ValueError("Moment samples must be finite.")
        total += batch.sum(axis=0)
        second += batch.T @ batch
        count += len(batch)
    if count < 2:
        raise ValueError("At least two retained samples are needed for moments.")
    mean = total / count
    covariance = second / count - np.outer(mean, mean)
    covariance = .5 * (covariance + covariance.T)
    return mean, covariance, count


def moment_comparison(generated, reference, reference_mask=None):
    """Compare marginal reduced-coordinate distributions over selected frames."""
    generated_mean, generated_covariance, generated_count = _population_moments(
        generated)
    reference_mean, reference_covariance, reference_count = _population_moments(
        reference, reference_mask)
    mean_bias = generated_mean - reference_mean
    covariance_difference = generated_covariance - reference_covariance
    reference_second_moment_trace = float(
        np.trace(reference_covariance) + reference_mean @ reference_mean)
    reference_rms = float(np.sqrt(max(reference_second_moment_trace, 0)))
    reference_covariance_norm = float(np.linalg.norm(reference_covariance))
    reference_std = np.sqrt(np.maximum(np.diag(reference_covariance), 0))
    valid = reference_std > np.finfo(float).eps * max(float(reference_std.max()), 1.)
    standardized_mean_rmse = (
        float(np.sqrt(np.mean(np.square(mean_bias[valid] / reference_std[valid]))))
        if valid.any() else None
    )
    scale = np.outer(reference_std, reference_std)
    valid_covariance = scale > np.finfo(float).eps * max(float(scale.max()), 1.)
    standardized_covariance_rmse = (
        float(np.sqrt(np.mean(np.square(covariance_difference[valid_covariance]
                                      / scale[valid_covariance]))))
        if valid_covariance.any() else None
    )
    return dict(
        generated_sample_count=generated_count,
        reference_sample_count=reference_count,
        generated_mean=generated_mean.tolist(),
        reference_mean=reference_mean.tolist(),
        mean_bias=mean_bias.tolist(),
        mean_bias_l2=float(np.linalg.norm(mean_bias)),
        mean_bias_relative_to_reference_rms=(
            None if reference_rms == 0 else float(np.linalg.norm(mean_bias) / reference_rms)),
        mean_bias_reference_standardized_rmse=standardized_mean_rmse,
        generated_variance=np.diag(generated_covariance).tolist(),
        reference_variance=np.diag(reference_covariance).tolist(),
        generated_covariance_trace=float(np.trace(generated_covariance)),
        reference_covariance_trace=float(np.trace(reference_covariance)),
        covariance_trace_ratio=_ratio(
            float(np.trace(generated_covariance)), float(np.trace(reference_covariance))),
        covariance_frobenius_error=float(np.linalg.norm(covariance_difference)),
        covariance_relative_frobenius_error=(
            None if reference_covariance_norm == 0
            else float(np.linalg.norm(covariance_difference) / reference_covariance_norm)),
        covariance_reference_standardized_rmse=standardized_covariance_rmse,
        _generated_covariance=generated_covariance,
        _reference_covariance=reference_covariance,
    )


def _json_moments(values):
    return {key: value for key, value in values.items() if not key.startswith("_")}


def _moment_populations(generated, reference, reference_mask):
    untrimmed = moment_comparison(generated, reference)
    reference_trimmed = (None if reference_mask is None else
                         moment_comparison(generated, reference,
                                           reference_mask=reference_mask))
    return untrimmed, reference_trimmed


def coefficient_statistics(generated, reference, reference_mask,
                           include_spatial_mean, trim_report=None):
    untrimmed, reference_trimmed = _moment_populations(
        generated, reference, reference_mask)
    arrays = dict(
        generated_covariance=untrimmed["_generated_covariance"],
        reference_covariance=untrimmed["_reference_covariance"],
    )
    report = dict(
        definition="marginal physical reduced-coordinate distribution pooled over conditions, ensembles, and time",
        coordinate_units="same displacement units as rollout coefficients",
        coordinate_order="optional sqrt(N)*spatial mean followed by shared POD coefficients",
        untrimmed=_json_moments(untrimmed),
        generated_untrimmed_vs_reference_training_trimmed=(
            None if reference_trimmed is None else _json_moments(reference_trimmed)),
        reference_training_trim=trim_report,
    )
    if reference_trimmed is not None:
        arrays["reference_covariance_training_trimmed"] = (
            reference_trimmed["_reference_covariance"])
    if include_spatial_mean:
        for name, coordinate_slice in (
                ("spatial_mean_coordinate", slice(0, 1)),
                ("pod_fluctuation_coordinates", slice(1, None))):
            group_untrimmed, group_reference_trimmed = _moment_populations(
                generated[..., coordinate_slice], reference[..., coordinate_slice],
                reference_mask)
            report[name] = dict(
                untrimmed=_json_moments(group_untrimmed),
                generated_untrimmed_vs_reference_training_trimmed=(
                    None if group_reference_trimmed is None
                    else _json_moments(group_reference_trimmed)),
            )
            arrays.update({
                f"generated_covariance_{name}": group_untrimmed["_generated_covariance"],
                f"reference_covariance_{name}": group_untrimmed["_reference_covariance"],
            })
            if group_reference_trimmed is not None:
                arrays[f"reference_covariance_{name}_training_trimmed"] = (
                    group_reference_trimmed["_reference_covariance"])
    return report, arrays


def _band_metrics(frequency, generated_density, reference_density, lower_frequency,
                  bins_per_decade):
    positive = frequency > 0
    if not positive.any():
        raise ValueError("PSD has no positive frequencies.")
    effective_lower = max(float(lower_frequency), float(frequency[positive][0]))
    maximum = float(frequency[-1])
    if effective_lower >= maximum:
        raise ValueError("PSD lower frequency must lie below Nyquist.")
    # Anchor the logarithmic grid to the requested lower frequency (10 Hz by
    # default), as in the raw/POD band-power analysis. If Welch cannot resolve
    # that low, retain one partial first band beginning at its first positive
    # bin, then continue on the original whole/half-decade grid.
    requested_edges = logarithmic_edges(float(lower_frequency), maximum,
                                        bins_per_decade)
    interior = requested_edges[(requested_edges > effective_lower)
                               & (requested_edges < maximum)]
    edges = np.concatenate(([effective_lower], interior, [maximum]))
    generated_mean = generated_density.reshape(-1, generated_density.shape[-1]).mean(axis=0)
    reference_mean = reference_density.reshape(-1, reference_density.shape[-1]).mean(axis=0)
    density_floor = max(float(reference_mean.max()) * 1e-12, np.finfo(float).tiny)
    bands, log_errors = [], []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        generated_power = integrate_band(frequency, generated_mean, lower, upper)
        reference_power = integrate_band(frequency, reference_mean, lower, upper)
        floor_power = density_floor * (upper - lower)
        ratio = (generated_power + floor_power) / (reference_power + floor_power)
        log_error = abs(float(np.log10(ratio)))
        log_errors.append(log_error)
        bands.append(dict(lower_hz=float(lower), upper_hz=float(upper),
                          generated_power=float(generated_power),
                          reference_power=float(reference_power), ratio=float(ratio),
                          absolute_log10_error_decades=log_error))
    return dict(
        requested_lower_frequency_hz=float(lower_frequency),
        effective_lower_frequency_hz=effective_lower,
        upper_frequency_hz=maximum,
        bins_per_decade=int(bins_per_decade),
        comparison="equal-weight mean absolute log10(generated/reference) of integrated band powers",
        mean_absolute_log10_band_power_error_decades=float(np.mean(log_errors)),
        rms_log10_band_power_error_decades=float(np.sqrt(np.mean(np.square(log_errors)))),
        maximum_absolute_log10_band_power_error_decades=float(np.max(log_errors)),
        bands=bands,
    )


def psd_comparison(rollout, *, point=None, nperseg=1024, overlap=.5,
                   lower_frequency=10., bins_per_decade=2):
    if nperseg < 4 or not 0 <= overlap < 1 or lower_frequency <= 0:
        raise ValueError("PSD settings require nperseg >= 4, overlap in [0, 1), and positive lower frequency.")
    if bins_per_decade not in (1, 2):
        raise ValueError("Bins per decade must be 1 or 2.")
    selected = select_point(rollout.modes.shape[1:], rollout.x, rollout.y,
                            rollout.x_units, rollout.y_units, point, None)
    if rollout.include_spatial_mean:
        generated = point_signals(rollout.generated, rollout.modes, rollout.n_space, selected)
        reference = point_signals(rollout.reference, rollout.modes, rollout.n_space, selected)
    else:
        generated = fluctuation_signals(rollout.generated, rollout.modes, selected)
        reference = fluctuation_signals(rollout.reference, rollout.modes, selected)
    report = {}
    segment = n_overlap = None
    for name in generated:
        frequency, generated_density, segment, n_overlap = ensemble_psd(
            generated[name], rollout.fs, nperseg, overlap)
        _, reference_density, _, _ = ensemble_psd(reference[name], rollout.fs, nperseg, overlap)
        report[name] = _band_metrics(frequency, generated_density, reference_density,
                                     lower_frequency, bins_per_decade)
    return dict(
        estimator="Welch: Hann window, constant detrend, density scaling; trajectory PSDs averaged before band integration",
        point=dict(y_index=selected.y_index, x_index=selected.x_index,
                   x=selected.x_coordinate, y=selected.y_coordinate),
        nperseg=int(segment),
        noverlap=int(n_overlap),
        frequency_spacing_hz=float(frequency[1] - frequency[0]),
        trimming="none; PSD requires complete uniformly sampled trajectories",
        signals=report,
    )


def _number(value, precision=3):
    return "undefined" if value is None else f"{value:.{precision}g}"


def quick_summary(report):
    """Return a compact Markdown view of the central quantitative results."""
    energy = report["energy"]
    untrimmed = energy["untrimmed"]
    trimmed = energy["generated_untrimmed_vs_reference_training_trimmed"]
    ratio = untrimmed["generated_over_reference"]
    excess = None if ratio is None else 100 * (ratio - 1)
    statistics = report["coefficient_statistics"]
    pod = statistics.get("pod_fluctuation_coordinates", statistics)["untrimmed"]

    central = (f"**Central result:** generated quadratic energy is {_number(excess)}% above "
               "the untrimmed reference.")
    if trimmed is not None:
        trim_percent = trimmed["reference_trim"]["train_trim_percent"]
        trimmed_ratio = trimmed["generated_over_reference"]
        trimmed_excess = None if trimmed_ratio is None else 100 * (trimmed_ratio - 1)
        central += (f" Against the reference filtered by the checkpoint's {trim_percent:g}% "
                    f"training-increment rule, it is {_number(trimmed_excess)}% above; "
                    "generated trajectories remain untrimmed.")
    lines = [
        "# Quantitative rollout: quick summary",
        "",
        central,
        "",
        "| Quantity | Generated/reference result |",
        "|---|---:|",
        (f"| Mean quadratic energy, both untrimmed | "
         f"{_number(untrimmed['generated']['mean_J'])} / "
         f"{_number(untrimmed['reference']['mean_J'])} J = **{_number(ratio)}×** |"),
        (f"| POD mean bias / reference RMS | "
         f"**{_number(pod['mean_bias_relative_to_reference_rms'])}** |"),
        (f"| POD covariance trace | "
         f"**{_number(pod['covariance_trace_ratio'])}×** |"),
        (f"| POD covariance relative Frobenius error | "
         f"**{_number(pod['covariance_relative_frobenius_error'])}** |"),
    ]
    if "spatial_mean_coordinate" in statistics:
        spatial_mean = statistics["spatial_mean_coordinate"]["untrimmed"]
        lines.extend([
            (f"| Spatial-mean bias / reference RMS | "
             f"**{_number(spatial_mean['mean_bias_relative_to_reference_rms'])}** |"),
            (f"| Spatial-mean variance | "
             f"**{_number(spatial_mean['covariance_trace_ratio'])}×** |"),
        ])
    if trimmed is not None:
        lines.insert(7,
            (f"| Mean quadratic energy, generated untrimmed / reference training-trimmed | "
             f"{_number(trimmed['generated']['mean_J'])} / "
             f"{_number(trimmed['reference']['mean_J'])} J = "
             f"**{_number(trimmed_ratio)}×** |"))

    lines.extend(["", "## Half-decade PSD comparison", "",
                  "| Signal | Mean absolute log-power error | Approx. factor |",
                  "|---|---:|---:|"])
    labels = dict(center="Center", center_without_spatial_mean="Center, POD only",
                  spatial_mean="Spatial mean")
    for name, values in report["psd"]["signals"].items():
        error = values["mean_absolute_log10_band_power_error_decades"]
        lines.append(f"| {labels.get(name, name.replace('_', ' '))} | "
                     f"{_number(error)} decades | {_number(10**error)}× |")
    lines.extend([
        "",
        "Ratios are ideal at 1; bias/error measures are ideal at 0. PSD factor is "
        "`10^(mean absolute log10 band-power error)` and has no over/under direction.",
        "Reference means the saved measured trajectory in the model's shared, "
        "rank-resolved POD space, not the full raw camera field.",
    ])
    return "\n".join(lines) + "\n"


def analyze(rollout, *, gamma=.0728, point=None,
            nperseg=1024, overlap=.5, psd_lower_frequency=10., bins_per_decade=2,
            output=None, overwrite=False):
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Surface tension must be positive and finite.")
    x_m = rollout.x * length_scale(rollout.x_units)
    y_m = rollout.y * length_scale(rollout.y_units)
    stiffness, _ = capillary_stiffness(rollout.modes, x_m, y_m, gamma=gamma)
    options = dict(stiffness=stiffness, include_spatial_mean=rollout.include_spatial_mean)
    generated_energy = compute_energies(
        rollout.generated, rollout.modes, x_m, y_m, length_scale(rollout.units), gamma,
        batch_size=65536, **options)["quadratic"]
    reference_energy = compute_energies(
        rollout.reference, rollout.modes, x_m, y_m, length_scale(rollout.units), gamma,
        batch_size=65536, **options)["quadratic"]
    reference_mask, trim_report = reference_training_trim_population(rollout)
    energy = energy_comparison(
        generated_energy, reference_energy, reference_mask, trim_report)
    moments, moment_arrays = coefficient_statistics(
        rollout.generated, rollout.reference, reference_mask,
        rollout.include_spatial_mean, trim_report)
    report = rollout.provenance()
    report.update(
        purpose="descriptive rollout distribution and dynamics metrics; not a forecasting score",
        energy=energy,
        coefficient_statistics=moments,
        psd=psd_comparison(
            rollout, point=point, nperseg=nperseg, overlap=overlap,
            lower_frequency=psd_lower_frequency, bins_per_decade=bins_per_decade),
    )
    out = output_directory(rollout, "quantitative", output, overwrite)
    np.savez_compressed(
        out / "metrics.npz",
        **moment_arrays,
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "quick_summary.md").write_text(quick_summary(report))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--gamma", type=float, default=.0728)
    parser.add_argument("--point", nargs=2, type=int, metavar=("Y", "X"),
                        help="PSD pixel indices; default grid center.")
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=.5)
    parser.add_argument("--psd-lower-frequency", type=float, default=10.,
                        help="Requested lower PSD comparison frequency [Hz]; raised to the first positive Welch bin when needed.")
    parser.add_argument("--bins-per-decade", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    rollout = Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard)
    print(analyze(
        rollout, gamma=args.gamma, point=args.point, nperseg=args.nperseg, overlap=args.overlap,
        psd_lower_frequency=args.psd_lower_frequency, bins_per_decade=args.bins_per_decade,
        output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
