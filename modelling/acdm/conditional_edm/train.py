from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from time import perf_counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .checkpoint import load_checkpoint, save_checkpoint
from .config import EDMConfig, TrainConfig, RUNTIME_OPTIONS
from .diagnostics import TrainingDiagnostics
from .data import NormalizationStats, checkpoint_data, fit_normalization, transition_batches
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT
from .model import ConditionalEDM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3g}"


def _progress_line(record: dict[str, Any]) -> str:
    fields = [f"epoch={record['epoch']}", f"loss={_format_number(record['train']['loss'])}"]
    one_step = record.get("one_step")
    if one_step:
        rmse, spread = one_step["conditional_mean_rmse"], one_step["spread_ratio"]
        fields.extend((
            f"rmse(mean/pod)={_format_number(rmse['spatial_mean'])}/{_format_number(rmse['pod'])}",
            f"R(mean/pod)={_format_number(spread['spatial_mean'])}/{_format_number(spread['pod'])}",
            f"dQ_bias={_format_number(one_step['pod_amplitude']['bias'])}",
            f"dm2_bias={_format_number(one_step['spatial_mean_amplitude']['bias'])}",
        ))
    benchmark = record.get("rollout", {}).get("benchmark")
    if benchmark:
        fields.extend((
            f"rollout_E={_format_number(benchmark['energy']['mean_absolute_log10_error'])}",
            f"PSD(mean/pod)={_format_number(benchmark['psd']['spatial_mean']['mean_absolute_log10_error'])}/"
            f"{_format_number(benchmark['psd']['center_without_spatial_mean']['mean_absolute_log10_error'])}",
        ))
    return " ".join(fields)


def _pod_only_normalization(checkpoint_path: Path, data: SharedPODTrainingData,
                            model_config: EDMConfig, data_signature: dict) -> NormalizationStats:
    """Slice existing training-only scales after verifying identical sources and transitions."""
    if data.include_spatial_mean:
        raise ValueError("Normalization reuse is intended only for POD-only training.")
    source = load_checkpoint(checkpoint_path)
    source_config = EDMConfig.from_dict(source["model_config"])
    if (source_config.reduced_dim != data.n_coordinates + 1
            or source_config.target_mode != model_config.target_mode
            or source_config.lag_steps != model_config.lag_steps
            or source_config.stride_steps != model_config.stride_steps):
        raise ValueError("Source checkpoint normalization has incompatible dimensions or transition settings.")
    original = dict(source["data_signature"])
    current = dict(data_signature)
    original.pop("coordinate_order", None)
    current.pop("coordinate_order", None)
    original.pop("include_spatial_mean", None)
    current.pop("include_spatial_mean", None)
    if original != current:
        raise ValueError("Source checkpoint uses different shared POD sources or repetition splits.")
    stats = source["normalization"]
    return NormalizationStats(*(stats[key][1:].clone() for key in
                                ("state_mean", "state_std", "target_mean", "target_std")))


def _save_diagnostic_plot(history: list[dict[str, Any]], output: Path, epoch: int) -> None:
    """Small checkpoint summary; full scalar history remains in JSONL/checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([row["epoch"] for row in history])
    figure, axes = plt.subplots(5, 1, figsize=(7, 11), sharex=True)
    axes[0].plot(epochs, [row["train"]["loss"] for row in history])
    axes[0].set_ylabel("train loss")
    axes[0].set_yscale("log")
    evaluated = [row for row in history if row.get("one_step")]
    eval_epochs = [row["epoch"] for row in evaluated]
    series = (
        ("rollout energy error", [row.get("rollout", {}).get("benchmark", {}).get("energy", {}).get(
            "mean_absolute_log10_error", np.nan) for row in evaluated]),
        ("POD dQ bias", [row["one_step"]["pod_amplitude"]["bias"] for row in evaluated]),
        ("mean dm2 bias", [row["one_step"]["spatial_mean_amplitude"]["bias"] for row in evaluated]),
        ("mean PSD error", [row.get("rollout", {}).get("benchmark", {}).get("psd", {}).get(
            "spatial_mean", {}).get("mean_absolute_log10_error", np.nan) for row in evaluated]),
    )
    for axis, (label, values) in zip(axes[1:], series):
        axis.plot(eval_epochs, values, marker="o")
        axis.set_ylabel(label)
        if "bias" in label:
            axis.axhline(0, color="0.5", linewidth=.8)
    axes[-1].set_xlabel("epoch")
    figure.tight_layout()
    directory = output / "diagnostics"
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / f"epoch_{epoch:04d}.png", dpi=150)
    figure.savefig(directory / "latest.png", dpi=150)
    plt.close(figure)


def run_epoch(
    model: ConditionalEDM,
    loader,
    *,
    device: torch.device,
    optimizer: AdamW | None = None,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, Tensor] = {}
    num_examples = 0
    num_batches = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    data_seconds = 0.
    iterator = iter(loader)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        while max_batches is None or num_batches < max_batches:
            before_read = perf_counter()
            try:
                raw_batch = next(iterator)
            except StopIteration:
                break
            data_seconds += perf_counter() - before_read
            num_batches += 1
            batch = _move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, diagnostics = model.loss(batch)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = batch["current_state"].shape[0]
            num_examples += batch_size
            for name, value in diagnostics.items():
                value = value.detach().double() * batch_size
                if name not in totals:
                    totals[name] = value
                else:
                    totals[name].add_(value)

    if num_examples == 0:
        raise RuntimeError("data loader produced no batches")
    # One transfer at epoch end, instead of five synchronizations per batch.
    values = (torch.stack(list(totals.values())) / num_examples).cpu().tolist()
    metrics = dict(zip(totals, values))
    metrics.update(elapsed_seconds=perf_counter() - started, data_seconds=data_seconds,
                   examples=num_examples)
    return metrics, num_batches


def train(
    *,
    data: SharedPODTrainingData,
    output_dir: str | Path,
    model_config: EDMConfig,
    train_config: TrainConfig,
    device: torch.device,
    resume: str | Path | None = None,
    normalization_from_checkpoint: str | Path | None = None,
) -> Path:
    seed_everything(train_config.seed)
    data_signature = data.signature()
    if model_config.reduced_dim != data.n_coordinates:
        raise ValueError("Model dimension must equal the prepared coordinate count.")
    if not data.include_spatial_mean and train_config.rollout_every:
        raise ValueError("POD-only training requires rollout_every=0; use a saved rollout for energy diagnostics.")
    if model_config.native_dt is None:
        model_config = replace(model_config, native_dt=data.native_dt)
    elif not np.isclose(model_config.native_dt, data.native_dt, rtol=1e-4, atol=0):
        raise ValueError("Model native_dt does not match the source trajectories.")
    if not data.splits["validation"]:
        raise ValueError("Preparation must provide at least one validation repetition.")
    required = model_config.lag_steps * (1 + (model_config.history_steps if model_config.history_conditioning else 0))
    if any(data.frame_counts[name] <= required for split in ("train", "validation") for name in data.splits[split]):
        raise ValueError("A training/validation repetition is too short for the selected history and lag.")
    if model_config.param_dim:
        raise ValueError("Single-experiment training requires param_dim=0.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] | None = None
    cache_start = perf_counter()
    cache_info = data.cache_coordinates(train_config.cache_gib)
    cache_info["seconds"] = perf_counter() - cache_start
    print(f"cache_enabled={cache_info['enabled']} cache_seconds={_format_number(cache_info['seconds'])}",
          flush=True)
    if resume is None:
        stats = (_pod_only_normalization(Path(normalization_from_checkpoint), data, model_config, data_signature)
                 if normalization_from_checkpoint is not None else fit_normalization(data, model_config))
        model = ConditionalEDM(model_config)
        model.set_normalization(stats)
        start_epoch = 0
        global_step = 0
        best_validation_loss = float("inf")
        history: list[dict[str, Any]] = []
    else:
        if normalization_from_checkpoint is not None:
            raise ValueError("Normalization is already stored in the resume checkpoint.")
        checkpoint = load_checkpoint(resume, device)
        saved_config = EDMConfig.from_dict(checkpoint["model_config"])
        if saved_config != model_config:
            raise ValueError("resume model configuration does not match the checkpoint")
        saved_train_config = TrainConfig.from_dict(checkpoint["train_config"])
        overrides = {name: getattr(train_config, name) for name in RUNTIME_OPTIONS}
        if replace(saved_train_config, epochs=train_config.epochs, **overrides) != train_config:
            raise ValueError("resume must reuse the saved training configuration except for epochs")
        if checkpoint["data_signature"] != data_signature:
            raise ValueError("Checkpoint data do not match the current shared-POD sources or splits.")
        model = ConditionalEDM(model_config)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        history = list(checkpoint["history"])

    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = None
    if train_config.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, train_config.epochs - start_epoch),
            eta_min=train_config.min_learning_rate,
        )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    if checkpoint is not None:
        torch.set_rng_state(checkpoint["rng_state"]["torch"].cpu())
        if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_state"]["cuda"]])
        old_best = Path(resume).resolve().parent / "best.pt"
        new_best = output.resolve() / "best.pt"
        if old_best != new_best and old_best.exists() and not new_best.exists():
            shutil.copy2(old_best, new_best)

    metadata = {
        "data": data.source_config,
        "model": model_config.to_dict(),
        "training": train_config.to_dict(),
        "split_labels": {k: list(v) for k, v in data.splits.items()},
        "device": str(device),
        "coordinate_cache": cache_info,
        "normalization_source": (str(Path(normalization_from_checkpoint).resolve())
                                 if normalization_from_checkpoint is not None else
                                 checkpoint.get("normalization_source") if checkpoint is not None else None),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"config={output / 'config.json'} device={device}", flush=True)

    if start_epoch >= train_config.epochs:
        raise ValueError(
            f"checkpoint already completed {start_epoch} epochs; request a larger --epochs value"
        )

    diagnostics = TrainingDiagnostics(data, model_config, train_config)
    for epoch in range(start_epoch, train_config.epochs):
        train_metrics, train_batches = run_epoch(
            model,
            transition_batches(data, model_config, "train", train_config.batch_size,
                               seed=train_config.seed + epoch),
            device=device,
            optimizer=optimizer,
            max_batches=train_config.max_train_batches,
        )
        validation_metrics, _ = run_epoch(
            model,
            transition_batches(data, model_config, "validation", train_config.batch_size),
            device=device,
            max_batches=train_config.max_validation_batches,
        )
        diagnostic_start = perf_counter()
        validation_metrics.update(diagnostics.fixed_mse(model))
        one_step_metrics = None
        rollout_metrics = None
        if train_config.rollout_every and (epoch + 1) % train_config.rollout_every == 0:
            one_step_metrics = diagnostics.one_step(model)
            rollout_metrics = diagnostics.rollout(model)
        diagnostic_seconds = perf_counter() - diagnostic_start
        if scheduler is not None:
            scheduler.step()
        global_step += train_batches
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "diagnostic_seconds": diagnostic_seconds,
        }
        if one_step_metrics:
            record["one_step"] = one_step_metrics
        if rollout_metrics is not None:
            record["rollout"] = rollout_metrics
        history.append(record)
        print(_progress_line(record), flush=True)
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        if one_step_metrics:
            with (output / "evaluation.jsonl").open("a") as log:
                log.write(json.dumps({key: record[key] for key in
                                      ("epoch", "train", "one_step", "rollout")}) + "\n")
            _save_diagnostic_plot(history, output, epoch + 1)

        validation_loss = validation_metrics.get("target_loss", validation_metrics["loss"])
        improved = validation_loss < best_validation_loss
        best_validation_loss = min(best_validation_loss, validation_loss)
        payload = {
            "format_version": 2,
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "normalization": model.normalization_state(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "history": history,
            "split_labels": {k: list(v) for k, v in data.splits.items()},
            "data_signature": data_signature,
            "data_config": data.source_config,
            "normalization_source": metadata["normalization_source"],
            "rng_state": {"torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        }
        save_checkpoint(output / "latest.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)

    return output / "latest.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modelling.acdm.conditional_edm train",
        description="Train a conditional EDM on reduced trajectories",
    )
    parser.add_argument("--experiment", help="Power folder, e.g. 0p20 (required for a new run).")
    parser.add_argument("--rank", type=int, help="Shared fluctuation rank; model dimension is rank+1 unless --no-spatial-mean.")
    parser.add_argument("--no-spatial-mean", action="store_true",
                        help="Train only on shared POD coefficients; omit coordinate zero from inputs and targets.")
    parser.add_argument("--normalization-from-checkpoint", type=Path,
                        help="For a POD-only run, reuse matching training-only scales from a mean-inclusive checkpoint.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--shared-basis", type=Path)
    parser.add_argument("--validation-reps", type=int, nargs="+", help="Passed to data preparation.")
    parser.add_argument("--test-reps", type=int, nargs="*", help="Passed to data preparation; empty disables test split.")
    parser.add_argument("--output", type=Path, default=None, help="run directory")
    parser.add_argument("--resume", type=Path, default=None, help="latest.pt to continue")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")

    parser.add_argument("--target-mode", choices=("increment", "future_state"), default="increment")
    parser.add_argument("--diffusion-formulation", choices=("edm", "ddpm"), default="edm",
                        help="EDM or the fixed 20-step Kohl DDPM baseline (default: edm).")
    parser.add_argument("--conditioning-mode", choices=("clean", "joint_noised"),
                        default="joint_noised",
                        help="Clean conditional EDM or ACDM-style joint diffusion (default: joint_noised).")
    parser.add_argument("--lag-steps", type=int, default=1)
    parser.add_argument(
        "--history-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="condition on previous lagged states (default: enabled)",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=2,
        help="number of previous lagged states (default: 2)",
    )
    parser.add_argument(
        "--stride-steps",
        dest="stride_steps",
        type=int,
        default=1,
        help="keep every nth transition while preserving the physical lag",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--noise-embedding-dim", type=int, default=64)
    parser.add_argument("--activation", choices=("silu", "gelu", "relu"), default="silu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--p-mean", type=float, default=-1.2)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--train-sigma-min", type=float, default=0.002)
    parser.add_argument("--train-sigma-max", type=float, default=80.0)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sampling-steps", type=int, default=32)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine"),
        default="cosine",
        help="learning-rate schedule (default: cosine)",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1.0e-5,
        help="final learning-rate floor for cosine scheduling",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=None, help="total epochs; default 20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--cache-gib", type=float, help="RAM cache cap, default 2 GiB; 0 streams. Also capped at 25%% of available RAM.")
    parser.add_argument("--fixed-mse-batch-size", type=int, help="Fixed validation transitions, default 256; 0 disables.")
    parser.add_argument("--fixed-sigmas", type=float, nargs=3, metavar=("LOW", "MID", "HIGH"), help="Default: 0.03 0.3 3.")
    parser.add_argument("--rollout-every", type=int, help="Diagnostic interval in epochs, default 10; 0 disables.")
    parser.add_argument("--one-step-conditions", type=int,
                        help="Fixed held-out conditions per diagnostic, default 32; 0 disables.")
    parser.add_argument("--one-step-ensemble-size", type=int,
                        help="Independent samples per held-out condition, default 32.")
    parser.add_argument("--rollout-horizon", type=int, help="Diagnostic physical steps, default 200.")
    parser.add_argument("--rollout-ensemble-size", type=int, help="Samples per fixed train/validation start, default 2.")
    parser.add_argument("--rollout-energy-factor", type=float, help="Alert above this multiple of reference maximum energy; default 10.")
    parser.add_argument("--surface-tension", type=float, help="Diagnostic surface tension [N/m], default 0.0728.")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(arguments)
    runtime = {name: getattr(args, name) for name in RUNTIME_OPTIONS if getattr(args, name) is not None}
    if "fixed_sigmas" in runtime:
        runtime["fixed_sigmas"] = tuple(runtime["fixed_sigmas"])
    device = resolve_device(args.device)
    output = args.output
    if output is None:
        output = args.resume.parent if args.resume is not None else Path("runs/conditional_edm")

    if args.resume is None:
        if args.experiment is None or args.rank is None:
            parser.error("A new run requires --experiment and --rank.")
        data = SharedPODTrainingData(args.experiment, args.rank,
                                    data_root=args.data_root or DEFAULT_DATA_ROOT,
                                    shared_basis=args.shared_basis,
                                    validation_reps=args.validation_reps, test_reps=args.test_reps,
                                    include_spatial_mean=not args.no_spatial_mean)
        if args.no_spatial_mean:
            runtime.setdefault("rollout_every", 0)
        model_config = EDMConfig(
            reduced_dim=data.n_coordinates,
            native_dt=data.native_dt,
            target_mode=args.target_mode,
            diffusion_formulation=args.diffusion_formulation,
            conditioning_mode=args.conditioning_mode,
            lag_steps=args.lag_steps,
            stride_steps=args.stride_steps,
            history_conditioning=args.history_conditioning,
            history_steps=args.history_steps,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            noise_embedding_dim=args.noise_embedding_dim,
            activation=args.activation,
            dropout=args.dropout,
            sigma_data=args.sigma_data,
            p_mean=args.p_mean,
            p_std=args.p_std,
            train_sigma_min=args.train_sigma_min,
            train_sigma_max=args.train_sigma_max,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            num_sampling_steps=args.sampling_steps,
        )
        train_config = TrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler=args.lr_scheduler,
            min_learning_rate=args.min_learning_rate,
            weight_decay=args.weight_decay,
            epochs=20 if args.epochs is None else args.epochs,
            seed=args.seed,
            max_train_batches=args.max_train_batches,
            max_validation_batches=args.max_validation_batches,
        )
    else:
        allowed_resume_options = {"--data-root", "--shared-basis", "--output", "--resume", "--device", "--epochs"}
        allowed_resume_options.update("--" + name.replace("_", "-") for name in RUNTIME_OPTIONS)
        specified_options = {
            token.split("=", 1)[0] for token in arguments if token.startswith("--")
        }
        locked_options = sorted(specified_options - allowed_resume_options)
        if locked_options:
            parser.error(
                "resume reuses checkpoint settings; only runtime diagnostics/cache, --epochs, --device, --output, and data location "
                f"may be specified (remove {', '.join(locked_options)})"
            )
        checkpoint = load_checkpoint(args.resume)
        data = checkpoint_data(checkpoint, data_root=args.data_root, shared_basis=args.shared_basis)
        model_config = EDMConfig.from_dict(checkpoint["model_config"])
        train_config = TrainConfig.from_dict(checkpoint["train_config"])
        if args.epochs is not None:
            train_config = replace(train_config, epochs=args.epochs)

    train_config = replace(train_config, **runtime)
    train(
        data=data,
        output_dir=output,
        model_config=model_config,
        train_config=train_config,
        device=device,
        resume=args.resume,
        normalization_from_checkpoint=args.normalization_from_checkpoint,
    )


if __name__ == "__main__":
    main()
