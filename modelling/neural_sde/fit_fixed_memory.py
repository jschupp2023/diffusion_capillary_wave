"""Bounded one-step Gaussian-likelihood fit of a frozen oscillator-memory readout.

Run with python -m modelling.neural_sde.fit_fixed_memory --help.
Only the linear readout is estimated. No stochastic rollout is differentiated.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import NeuralSDE
from .train import load_model
from modelling.acdm.conditional_edm.checkpoint import save_checkpoint
from modelling.acdm.conditional_edm.data import checkpoint_data


def ridge_candidates(features, residuals, validation_features, validation_residuals, penalties):
    """Train-only RMS scaling and ridge; validation selects against a zero readout.

    Residuals are whitened by the frozen parent diffusion. The reported score is
    mean squared whitened error, proportional to Gaussian NLL up to constants.
    Penalty is lambda * ||W||^2, with data loss averaged over training examples.
    """
    x, y, xv, yv = (np.asarray(a, dtype=np.float64) for a in
                     (features, residuals, validation_features, validation_residuals))
    if (not len(x) or not len(xv) or any(not np.isfinite(a).all() for a in (x, y, xv, yv))):
        raise ValueError("ridge requires finite nonempty training and validation data")
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-8)
    x, xv = x / scale, xv / scale
    gram, rhs = x.T @ x / len(x), x.T @ y / len(x)
    best = np.zeros_like(rhs)
    best_score = float(np.mean(yv*yv))
    scores = [dict(penalty=None, train_error=float(np.mean(y*y)), validation_error=best_score)]
    selected = None
    for penalty in penalties:
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("ridge penalties must be finite and positive")
        weight = np.linalg.solve(gram + penalty * np.eye(len(gram)), rhs)
        score = float(np.mean((yv - xv @ weight)**2))
        scores.append(dict(penalty=float(penalty), train_error=float(np.mean((y-x@weight)**2)),
                           validation_error=score))
        if score < best_score:
            best, best_score, selected = weight, score, float(penalty)
    return best, scale, selected, scores


def eligible_indices(values, history, cutoff=None, scale=None):
    indices = np.arange(history, len(values)-1)
    if cutoff is not None:
        if scale is None:
            raise ValueError("trimmed parent requires its saved trim normalization")
        v = torch.as_tensor(values, dtype=torch.float32)
        bad = ~(torch.linalg.vector_norm((v[1:]-v[:-1]) / torch.as_tensor(scale), dim=-1) <= cutoff)
        prefix = np.r_[0, np.cumsum(bad.numpy())]
        indices = indices[prefix[indices+1] == prefix[indices-history]]
    return indices


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_source(source, checkpoint, data):
    """Reject unrelated POD bases and peak filters fitted on held-out records."""
    provenance = source["source_provenance"]
    reference = provenance["single_rollout"]
    if (Path(reference["shared_basis"]).resolve() != Path(checkpoint["data_config"]["shared_basis"]).resolve()
            or reference["rank"] != checkpoint["data_config"]["rank"]
            or reference["spatial_mean_highpass_hz"] != checkpoint["data_config"].get("spatial_mean_highpass_hz")):
        raise ValueError("oscillator source and parent must use the same POD basis, rank and mean preprocessing")
    train_ids = {int(name.rsplit("rep", 1)[1]) for name in data.splits["train"]}
    if (not set(provenance["training_repetitions"]).issubset(train_ids)
            or provenance.get("test_repetitions_used")):
        raise ValueError("oscillator fitting records must belong to the parent's training split")


@torch.no_grad()
def fit(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    if (min(args.train_windows, args.validation_windows, args.batch_size, args.max_seconds, args.threads) <= 0
            or not np.isfinite(args.max_seconds) or args.window < 0):
        raise ValueError("window limits, batch size, and time budget must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; existing experiments are never overwritten")
    parent, checkpoint = load_model(Path(args.parent), "cpu")
    c = parent.config
    if (c.state_variable != "velocity" or c.lag_steps != 1
            or c.diffusion_type not in {"diagonal", "full"}
            or any(x is not None for x in (c.fixed_memory_frequencies, c.velocity_memory_offsets,
                       c.velocity_oscillator_periods, c.velocity_innovation_offsets))):
        raise ValueError("use a native-lag velocity parent with constant diffusion and no additive memory")
    with np.load(args.oscillators, allow_pickle=False) as source:
        frequencies = source["frequencies_hz"].copy()
        damping = source["damping_per_second"].copy()
        mixing, spatial_scale = source["mixing"].copy(), source["scale"].copy()
    if mixing.shape != (len(frequencies), c.reduced_dim, 1):
        raise ValueError("requires rank-one peak mixing in the same ordered POD coordinates as the parent")
    # Common spatial unit conversion cancels when each projection is normalized.
    projection = (mixing[:, :, 0] * spatial_scale[None, :]).T / parent.target_std.numpy()[:, None]
    norms = np.linalg.norm(projection, axis=0)
    if not np.isfinite(projection).all() or np.any(norms <= 0):
        raise ValueError("peak mixing must provide finite nonzero projection columns")
    projection /= norms
    window = args.window or int(np.ceil(5 / (min(damping) * c.dt)))
    config = replace(c, history_offsets=c.effective_history_offsets,
                     fixed_memory_frequencies=tuple(frequencies),
                     fixed_memory_damping=tuple(damping), fixed_memory_window=window)
    model = NeuralSDE(config).eval()
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    if unexpected or any(not name.startswith("fixed_memory.") for name in missing):
        raise RuntimeError("parent checkpoint failed to transfer exactly")
    model.fixed_memory.projection.copy_(torch.from_numpy(projection))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("fixed_memory.readout."))
    data = checkpoint_data(checkpoint)
    source_config_path = (Path(args.oscillator_config) if args.oscillator_config
                          else Path(args.oscillators).parent.parent / "config.json")
    source_config = json.loads(source_config_path.read_text())
    validate_source(source_config, checkpoint, data)
    data.cache_coordinates(2.)
    training = checkpoint.get("training", {})
    parent.to(args.device)
    normalized_factor = (parent.diffusion_matrix() / parent.target_std[:, None]).cpu().numpy().astype(np.float64)
    features, targets, cache, counts, ids = {}, {}, {}, {}, {}
    for split, limit in (("train", args.train_windows), ("validation", args.validation_windows)):
        # Validation windows must not change when training sample count/seed changes.
        rng = np.random.default_rng(args.seed if split == "train" else args.validation_seed)
        xs, ys, selected_ids = [], [], []
        names = data.splits[split]
        remaining = limit
        for rep_index, name in enumerate(names):
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("time budget reached; no training campaign was launched")
            values = (data._coordinate_cache[name][1] if name in data._coordinate_cache
                      else data.read_coordinates(name, 0, data.frame_counts[name])[1])
            cached = model.fixed_memory.cache(values, model.target_mean, model.target_std)
            cache[name] = cached
            eligible = eligible_indices(values, config.history_steps,
                                         training.get("train_trim_norm_cutoff"),
                                         training.get("train_trim_state_std"))
            count = min(len(eligible), int(np.ceil(remaining / (len(names)-rep_index))))
            indices = np.sort(rng.choice(eligible, count, replace=False))
            counts[f"{split}/{name}"] = dict(eligible=len(eligible), selected=count)
            remaining -= count
            for start in range(0, count, args.batch_size):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("parent inference exceeded the requested time budget")
                ix = indices[start:start+args.batch_size]
                state = torch.as_tensor(values[ix], dtype=torch.float32, device=args.device)
                history = torch.as_tensor(values[ix[:, None]-np.arange(1, c.history_steps+1)],
                                          dtype=torch.float32, device=args.device)
                following = torch.as_tensor(values[ix+1], dtype=torch.float32, device=args.device)
                mean, _ = parent.velocity_step_distribution(state, history, state-history[:, 0])
                residual = ((following-state-mean) / parent.target_std).cpu().numpy()
                whitened = np.linalg.solve(normalized_factor, residual.T).T
                xs.append(cached[ix].reshape(len(ix), -1))
                ys.append(whitened)
                selected_ids.extend((rep_index, int(i)) for i in ix)
        if not xs:
            raise ValueError(f"no eligible {split} windows; shorten the memory or inspect the saved trim")
        features[split], targets[split] = np.concatenate(xs), np.concatenate(ys)
        ids[split] = np.asarray(selected_ids)
        print(f"{split}: {len(features[split])} windows, {time.monotonic()-started:.2f}s elapsed", flush=True)
    weights, scale, selected, scores = ridge_candidates(features["train"], targets["train"],
        features["validation"], targets["validation"], args.ridge)
    model.fixed_memory.feature_scale.copy_(torch.from_numpy(scale))
    model.fixed_memory.readout.weight.copy_(torch.from_numpy(normalized_factor @ weights.T))
    # Frozen parameters and diffusion must be byte-for-byte unchanged.
    for name, value in parent.cpu().state_dict().items():
        if not torch.equal(model.state_dict()[name], value):
            raise RuntimeError(f"unexpected parent modification: {name}")
    report = dict(parent=str(Path(args.parent).resolve()), parent_sha256=sha256(args.parent),
        oscillators=str(Path(args.oscillators).resolve()), oscillator_sha256=sha256(args.oscillators),
        oscillator_config=str(source_config_path.resolve()), oscillator_config_sha256=sha256(source_config_path),
        seed=args.seed, validation_seed=args.validation_seed, window_frames=window, window_seconds=window*c.dt,
        max_omitted_decay=float(np.exp(-min(damping)*window*c.dt)),
        selected_penalty=selected, scores=scores, counts=counts,
        elapsed_seconds=time.monotonic()-started, trainable_parameters=weights.size,
        objective="one-step frozen-diffusion Gaussian NLL plus ridge on whitened readout",
        caution="Validation one-step improvement does not establish rollout stability or peak recovery.",
        projection="unit-normalized physical peak mixing divided by parent velocity std; same POD ordering required")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "features.npz", **cache)
    np.savez_compressed(output / "regression.npz", train_features=features["train"],
        train_residuals=targets["train"], validation_features=features["validation"],
        validation_residuals=targets["validation"], train_ids=ids["train"], validation_ids=ids["validation"])
    report["elapsed_seconds"] = time.monotonic() - started
    payload = {**checkpoint, "model_config": config.to_dict(), "model_state": model.state_dict(),
               "epoch": 0, "training": {**training, "fixed_memory_fit": report,
                "train_mode": "fixed_memory_readout", "tcn_rollout_cache": True}}
    save_checkpoint(output / "best.pt", payload)
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    (output / "config.json").write_text(json.dumps(dict(data=checkpoint["data_config"],
        model=config.to_dict(), training=payload["training"]), indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return model, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", required=True)
    p.add_argument("--oscillators", required=True, help="rank-one fitted model.npz in identical POD ordering")
    p.add_argument("--oscillator-config", help="oscillator fit provenance JSON; default: config.json above model directory")
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=0, help="finite memory in model frames; 0 uses five slowest decay times")
    p.add_argument("--train-windows", type=int, default=8192)
    p.add_argument("--validation-windows", type=int, default=2048)
    p.add_argument("--ridge", type=float, nargs="+", default=[.001, .01, .1, 1.])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validation-seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=180.)
    args = p.parse_args(argv)
    fit(args)


if __name__ == "__main__":
    main()
