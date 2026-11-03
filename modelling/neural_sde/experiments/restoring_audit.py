"""Read-only restoring/residual audit on measured and generated saved histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modelling.neural_sde.train import load_model


def sample_histories(paths, history, samples, seed, edge_good=None):
    """Sample full stencils without crossing paths or removed increments."""
    indices = np.arange(history, paths.shape[1] - 1)
    if edge_good is None:
        eligible = np.ones((len(paths), len(indices)), dtype=bool)
    else:
        prefix = np.pad(np.cumsum(~edge_good, axis=1), ((0, 0), (1, 0)))
        eligible = prefix[:, indices + 1] == prefix[:, indices - history]
    path, local = np.nonzero(eligible)
    if not len(path):
        raise ValueError("No complete eligible history stencils")
    selected = np.random.default_rng(seed).choice(len(path), min(samples, len(path)), replace=False)
    path, time = path[selected], indices[local[selected]]
    return (torch.tensor(paths[path, time], dtype=torch.float32),
            torch.tensor(paths[path[:, None], time[:, None] - np.arange(1, history + 1)],
                         dtype=torch.float32), int(eligible.sum()))


@torch.inference_mode()
def coherent_response(model, current, history, coordinates, shift_fraction):
    """Physical diagonal -df_i/dx_i under a common shift of the entire history."""
    velocity = current - history[:, 0]
    values = []
    for coordinate in coordinates:
        shift = torch.zeros_like(current)
        delta = shift_fraction * model.state_std[coordinate]
        shift[:, coordinate] = delta
        plus = model.drift(current + shift, history + shift[:, None], velocity)
        minus = model.drift(current - shift, history - shift[:, None], velocity)
        values.append((-(plus[:, coordinate] - minus[:, coordinate]) / (2 * delta)).numpy())
    values = np.stack(values, axis=1)
    return dict(mean=values.mean(0).tolist(), median=np.median(values, axis=0).tolist(),
                anti_restoring_fraction=(values < 0).mean(0).tolist())


@torch.inference_mode()
def population_audit(model, current, history, eligible):
    inputs = model._drift_input(current, history)
    residual = model.nonlinear_drift(current, history, normalized_input=inputs)
    dimension = model.config.reduced_dim
    position, velocity = inputs[:, :dimension], inputs[:, dimension:2 * dimension]
    stiffness = model.drift_model.stiffness_diagonal()
    linear = -position * stiffness
    z, v, r = (x.numpy().astype(float) for x in (position, velocity, residual))
    # Pooled per-coordinate least squares is descriptive, distinct from the
    # training loss's frozen all-velocity adjustment and minibatch averaging.
    coefficients = []
    for i in range(dimension):
        design = np.column_stack((np.ones(len(z)), z[:, i], v[:, i]))
        coefficients.append(np.linalg.lstsq(design, r[:, i], rcond=None)[0][1:].tolist())
    coordinates = list(range(1, min(6, dimension)))
    physical = stiffness * model.target_std / model.state_std
    return dict(samples=len(current), eligible_stencils=eligible,
                normalized_stiffness=stiffness.tolist(),
                explicit_physical_restoring_sensitivity=physical.tolist(),
                normalized_restoring_rms=linear.square().mean(0).sqrt().tolist(),
                normalized_residual_rms=residual.square().mean(0).sqrt().tolist(),
                residual_position_velocity_projection=coefficients,
                projection_note="Pooled residual ~ intercept + own position + own velocity; not the training penalty.",
                sensitivity_coordinates=coordinates,
                coherent_response={str(epsilon): coherent_response(
                    model, current, history, coordinates, epsilon) for epsilon in (.05, .1)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    torch.set_num_threads(2)
    model, _ = load_model(args.checkpoint, "cpu")
    model.eval()
    if (model.config.state_variable != "velocity" or model.config.lag_steps != 1
            or model.config.drift_type != "damped_residual"
            or model.config.damped_operator != "diagonal"):
        raise ValueError("Audit requires native-lag velocity with diagonal structured drift")
    config = json.loads(args.checkpoint.with_name("config.json").read_text())
    training = config["training"]
    metadata = json.loads((args.rollout / "metrics.json").read_text())
    if metadata["split"] != "validation":
        raise ValueError("This study audit only accepts validation rollouts")
    with np.load(args.rollout / "rollout.npz", allow_pickle=False) as arrays:
        reference = arrays["reference"]
        generated = arrays["trajectories"]
    generated = generated.reshape(-1, *generated.shape[-2:])
    good = np.linalg.norm(np.diff(reference.astype(float), axis=1) /
                          np.asarray(training["train_trim_state_std"]), axis=-1) <= training["train_trim_norm_cutoff"]
    report = dict(checkpoint=str(args.checkpoint.resolve()), split="validation", seed=args.seed,
                  reference_rule="All history-through-next increments pass the saved cutoff; generated histories untrimmed.",
                  caution="Local coherent-shift sensitivities and pooled projections are not stability guarantees.")
    for label, paths, edges in (("measured", reference, good), ("generated", generated, None)):
        current, history, eligible = sample_histories(
            paths, model.config.history_steps, args.samples, args.seed, edges)
        report[label] = population_audit(model, current, history, eligible)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
