"""Train and evaluate stochastic OpInf on normalized shared-POD trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from modelling.acdm.conditional_edm.checkpoint import load_checkpoint, save_checkpoint
from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.data import checkpoint_data, fit_normalization
from modelling.acdm.conditional_edm.evaluate import evaluate_model
from modelling.acdm.conditional_edm.train import resolve_device
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT

from .config import OpInfConfig
from .model import (
    StochasticOpInfModel,
    infer_diffusion,
    infer_drift,
    input_signal,
    original_moment_errors,
    page_cov,
)


def regularization_candidates(
    model_form: str,
    count: int,
    n_regularization_min: float = 1.0,
    n_regularization_max: float = 1.0e10,
) -> tuple[np.ndarray, str]:
    """Original logarithmic grid: regularize N when present, otherwise A."""
    if count < 1:
        raise ValueError("regularization count must be positive")
    if "N" in model_form:
        if (not np.isfinite(n_regularization_min)
                or not np.isfinite(n_regularization_max)
                or n_regularization_min <= 0
                or n_regularization_min > n_regularization_max):
            raise ValueError("N regularization bounds must be positive and increasing")
        return np.geomspace(n_regularization_min, n_regularization_max, count), "N"
    return np.logspace(-1, 5, count), "A"


def regularization_vector(
    model_form: str,
    candidate: float,
    a_regularization: float = 0.0,
    b_regularization: float = 0.0,
) -> np.ndarray:
    if model_form == "ABN":
        return np.array([a_regularization, b_regularization, candidate])
    if model_form == "AN":
        return np.array([a_regularization, candidate])
    if model_form == "AB":
        return np.array([candidate, b_regularization])
    return np.array([candidate])


def segmented_states(
    data: SharedPODTrainingData,
    split: str,
    segment_length: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Normalize then split each repetition into non-overlapping windows."""
    if segment_length < 7:
        raise ValueError("segment_length must be at least 7 for derivative filters")
    segments, counts = [], {}
    for name in data.splits[split]:
        _, values = data.read_coordinates(name, 0, data.frame_counts[name])
        count = len(values) // segment_length
        if count:
            normalized = (values[:count * segment_length] - mean) / scale
            segments.append(normalized.reshape(count, segment_length, data.n_coordinates))
        counts[name] = count
    if not segments:
        raise ValueError(f"split {split!r} has no complete segments")
    return np.concatenate(segments), counts


def ensemble_statistics(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean [coordinate,time] and covariance [coordinate,coordinate,time]."""
    states = np.transpose(segments, (2, 0, 1))
    return states.mean(axis=1), page_cov(states)


def _save_regularization_plot(
    candidates: np.ndarray,
    mean_errors: np.ndarray,
    covariance_errors: np.ndarray,
    selected: int,
    operator: str,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.loglog(candidates, mean_errors, "o-", label="training mean")
    axis.loglog(candidates, covariance_errors, "s-", label="training covariance")
    axis.axvline(candidates[selected], color="0.3", linestyle="--", label="selected")
    axis.set(xlabel=f"{operator} regularization", ylabel="relative error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def train(
    data: SharedPODTrainingData,
    output: Path,
    config: OpInfConfig,
    *,
    segment_length: int = 5_760,
    regularization_count: int = 10,
    a_regularization: float = 0.0,
    b_regularization: float = 0.0,
    n_regularization_min: float = 1.0,
    n_regularization_max: float = 1.0e10,
    h_regularization: float = 1.0e5,
    seed: int = 42,
) -> Path:
    """Fit all original regularization candidates and save the best model."""
    if (config.reduced_dim != data.n_coordinates
            or not np.isclose(config.dt, data.native_dt, rtol=1.0e-4, atol=0.0)):
        raise ValueError("model dimension and dt must match the shared-POD data")
    if not np.isfinite(h_regularization) or h_regularization < 0:
        raise ValueError("h_regularization must be finite and nonnegative")
    if (not np.isfinite(a_regularization) or a_regularization < 0
            or not np.isfinite(b_regularization) or b_regularization < 0):
        raise ValueError("A and B regularization must be finite and nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    normalization = fit_normalization(
        data,
        EDMConfig(reduced_dim=data.n_coordinates, native_dt=data.native_dt,
                  target_mode="increment", history_conditioning=False),
    )
    state_mean = normalization.state_mean.double().numpy()
    state_std = normalization.state_std.double().numpy()
    segments, segment_counts = segmented_states(
        data, "train", segment_length, state_mean, state_std
    )
    reference_mean, reference_covariance = ensemble_statistics(segments)
    initial_conditions = segments[:, 0].T
    signal = input_signal(config, segment_length)
    candidates, regularized_operator = regularization_candidates(
        config.model_form,
        regularization_count,
        n_regularization_min,
        n_regularization_max,
    )
    mean_errors = np.full(len(candidates), np.nan)
    covariance_errors = np.full(len(candidates), np.nan)
    fitted: list[dict[str, np.ndarray] | None] = [None] * len(candidates)

    for index, candidate in enumerate(candidates):
        try:
            mass, drift, input_operator, bilinear = infer_drift(
                reference_mean,
                signal,
                config.dt,
                is_bilinear=config.is_bilinear,
                has_input_term=config.has_input_term,
                regularization=regularization_vector(
                    config.model_form,
                    float(candidate),
                    a_regularization,
                    b_regularization,
                ),
            )
            diffusion, diffusion_covariance = infer_diffusion(
                reference_covariance,
                signal,
                config.dt,
                drift,
                bilinear,
                h_regularization,
            )
            mean_errors[index], covariance_errors[index] = original_moment_errors(
                config=config,
                mass=mass,
                drift=drift,
                input_operator=input_operator,
                bilinear=bilinear,
                diffusion=diffusion,
                initial_conditions=initial_conditions,
                reference_mean=reference_mean,
                reference_covariance=reference_covariance,
                seed=seed,
            )
            fitted[index] = dict(
                mass=mass,
                drift=drift,
                input_operator=input_operator,
                bilinear=bilinear,
                diffusion=diffusion,
                diffusion_covariance=diffusion_covariance,
            )
            print(
                f"candidate={candidate:.6e} mean_error={mean_errors[index]:.6e} "
                f"covariance_error={covariance_errors[index]:.6e}",
                flush=True,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            print(f"candidate={candidate:.6e} failed: {error}", flush=True)

    finite = np.isfinite(mean_errors)
    if not finite.any():
        raise RuntimeError("every regularization candidate failed")
    selected = int(np.argmin(np.where(finite, mean_errors, np.inf)))
    operators = fitted[selected]
    assert operators is not None
    model = StochasticOpInfModel(
        config,
        mass=operators["mass"],
        drift=operators["drift"],
        input_operator=operators["input_operator"],
        bilinear=operators["bilinear"],
        diffusion=operators["diffusion"],
        state_mean=normalization.state_mean,
        state_std=normalization.state_std,
        target_mean=normalization.target_mean,
        target_std=normalization.target_std,
    )
    training = dict(
        segment_length=segment_length,
        segment_counts=segment_counts,
        total_segments=int(len(segments)),
        regularized_operator=regularized_operator,
        a_regularization=float(a_regularization),
        b_regularization=float(b_regularization),
        n_regularization_min=float(n_regularization_min),
        n_regularization_max=float(n_regularization_max),
        regularization_candidates=candidates.tolist(),
        regularization_mean_errors=mean_errors.tolist(),
        regularization_covariance_errors=covariance_errors.tolist(),
        selected_index=selected,
        selected_regularization=float(candidates[selected]),
        selected_regularization_vector=regularization_vector(
            config.model_form,
            float(candidates[selected]),
            a_regularization,
            b_regularization,
        ).tolist(),
        selected_mean_error=float(mean_errors[selected]),
        selected_covariance_error=float(covariance_errors[selected]),
        h_regularization=h_regularization,
        selection_metric="training segmented ensemble relative mean error",
        seed=seed,
    )
    payload = dict(
        model_type="stochastic_opinf",
        model_config=config.to_dict(),
        data_config=data.source_config,
        data_signature=data.signature(),
        model_state=model.state_dict(),
        training=training,
    )
    checkpoint = output / "best.pt"
    save_checkpoint(checkpoint, payload)
    metadata = dict(
        model_type="stochastic_opinf",
        experiment=data.experiment,
        rank=data.rank,
        include_spatial_mean=data.include_spatial_mean,
        spatial_mean_highpass_hz=data.spatial_mean_highpass_hz,
        coordinate_count=data.n_coordinates,
        split_labels={key: list(value) for key, value in data.splits.items()},
        input_signal=(
            f"{config.input_amplitude} * cos(2*pi*{config.input_frequency}*relative_time)"
        ),
        model=config.to_dict(),
        training=training,
    )
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        output / "regularization.npz",
        candidates=candidates,
        mean_errors=mean_errors,
        covariance_errors=covariance_errors,
        selected_regularization_vector=np.asarray(
            training["selected_regularization_vector"], dtype=np.float64
        ),
    )
    _save_regularization_plot(
        candidates,
        mean_errors,
        covariance_errors,
        selected,
        regularized_operator,
        output / "regularization.png",
    )
    print(
        f"selected {regularized_operator}_reg={candidates[selected]:.6e} "
        f"vector={training['selected_regularization_vector']} "
        f"from {len(segments)} training segments",
        flush=True,
    )
    return checkpoint


def load_model(path: Path, device: torch.device | str = "cpu"):
    checkpoint = load_checkpoint(path, device)
    if checkpoint.get("model_type") != "stochastic_opinf":
        raise ValueError("checkpoint is not a stochastic OpInf model")
    model = StochasticOpInfModel(
        OpInfConfig.from_dict(checkpoint["model_config"]),
        diffusion=checkpoint["model_state"]["diffusion"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def evaluate(
    checkpoint_path: Path,
    output: Path,
    *,
    split: str = "test",
    num_conditions: int = 32,
    ensemble_size: int = 16,
    horizon: int = 50,
    batch_size: int = 8,
    seed: int = 0,
    device: torch.device | str = "cpu",
    data_root=None,
    shared_basis=None,
    metrics: bool = True,
) -> Path:
    model, checkpoint = load_model(checkpoint_path, device)
    data = checkpoint_data(checkpoint, data_root=data_root, shared_basis=shared_basis)
    result, report = evaluate_model(
        model,
        data,
        split=split,
        num_conditions=num_conditions,
        ensemble_size=ensemble_size,
        horizon=horizon,
        batch_size=batch_size,
        seed=seed,
        metrics=metrics,
    )
    report.update(
        source_shared_basis=str(data.basis_path),
        rank=data.rank,
        checkpoint=str(Path(checkpoint_path).resolve()),
        model_type="stochastic_opinf",
        model_form=model.config.model_form,
        input_amplitude=model.config.input_amplitude,
        input_frequency=model.config.input_frequency,
        input_time_origin="reset to zero at each rollout window",
        spatial_mean_highpass_hz=data.spatial_mean_highpass_hz,
        state_variable="state",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "rollout.npz", **result)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m modelling.stochastic_opinf")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="fit a normalized stochastic OpInf model")
    training.add_argument("--experiment", required=True)
    training.add_argument("--rank", type=int, required=True)
    training.add_argument(
        "--model-form", choices=("A", "AB", "AN", "ABN"), default="A",
        help="Original drift terms; u(t) affects only forms containing B or N (default: A).",
    )
    training.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    training.add_argument("--shared-basis", type=Path)
    training.add_argument("--validation-reps", type=int, nargs="+")
    training.add_argument("--test-reps", type=int, nargs="*")
    training.add_argument(
        "--spatial-mean-highpass-hz", type=float,
        help="Use the matching stored high-pass spatial-mean trajectory.",
    )
    training.add_argument("--segment-length", type=int, default=5_760)
    training.add_argument("--regularization-count", type=int, default=10)
    training.add_argument(
        "--a-regularization", "--a-reg", type=float, default=0.0,
        help="Fixed A-operator regularization when N is grid-searched (default 0).",
    )
    training.add_argument(
        "--b-regularization", "--b-reg", type=float, default=0.0,
        help="Fixed B-operator regularization (default 0).",
    )
    training.add_argument(
        "--n-regularization-min", "--n-reg-min", type=float, default=1.0,
        help="Smallest N-operator grid value (default 1).",
    )
    training.add_argument(
        "--n-regularization-max", "--n-reg-max", type=float, default=1.0e10,
        help="Largest N-operator grid value (default 1e10).",
    )
    training.add_argument("--h-regularization", type=float, default=1.0e5)
    training.add_argument("--sigma", type=float, default=1.0)
    training.add_argument("--input-amplitude", type=float, default=1.0)
    training.add_argument("--input-frequency", type=float, default=7.0e6)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--output", type=Path, required=True)

    evaluation = commands.add_parser("evaluate", help="sample and score physical test rollouts")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data-root", type=Path)
    evaluation.add_argument("--shared-basis", type=Path)
    evaluation.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluation.add_argument("--num-conditions", type=int, default=32)
    evaluation.add_argument("--ensemble-size", type=int, default=16)
    evaluation.add_argument("--horizon", type=int, default=50)
    evaluation.add_argument("--batch-size", type=int, default=8)
    evaluation.add_argument("--seed", type=int, default=0)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--no-metrics", action="store_true")
    evaluation.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        data = SharedPODTrainingData(
            args.experiment,
            args.rank,
            data_root=args.data_root,
            shared_basis=args.shared_basis,
            validation_reps=args.validation_reps,
            test_reps=args.test_reps,
            spatial_mean_highpass_hz=args.spatial_mean_highpass_hz,
        )
        config = OpInfConfig(
            reduced_dim=data.n_coordinates,
            dt=data.native_dt,
            model_form=args.model_form,
            input_amplitude=args.input_amplitude,
            input_frequency=args.input_frequency,
            sigma=args.sigma,
        )
        path = train(
            data,
            args.output,
            config,
            segment_length=args.segment_length,
            regularization_count=args.regularization_count,
            a_regularization=args.a_regularization,
            b_regularization=args.b_regularization,
            n_regularization_min=args.n_regularization_min,
            n_regularization_max=args.n_regularization_max,
            h_regularization=args.h_regularization,
            seed=args.seed,
        )
    else:
        path = evaluate(
            args.checkpoint,
            args.output,
            split=args.split,
            num_conditions=args.num_conditions,
            ensemble_size=args.ensemble_size,
            horizon=args.horizon,
            batch_size=args.batch_size,
            seed=args.seed,
            device=resolve_device(args.device),
            data_root=args.data_root,
            shared_basis=args.shared_basis,
            metrics=not args.no_metrics,
        )
    print(path.resolve())
