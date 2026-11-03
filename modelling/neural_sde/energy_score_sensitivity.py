"""Frozen joint-score probes using the training rollout and exact score formula.

Endpoint perturbations are diagnostic counterfactuals, not dynamically consistent
trajectories. Oracle mean corrections use held-out endpoints only for diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.acdm.conditional_edm.train import resolve_device
from .losses import MultistepVelocityEnergyScore
from .train import _capped_multistep_batches, load_model
from .multistep_particle_diagnostic import _distribution_summary


def _window_identifiers(batch, data, split):
    sources = tuple(data.splits[split])
    source_indices = batch['window_source_index'].detach().cpu().long().tolist()
    starts = batch['window_start_index'].detach().cpu().long().tolist()
    return [{'source': sources[source], 'start_index': start}
            for source, start in zip(source_indices, starts, strict=True)]


def _identifier_digest(records):
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def score_terms(samples, target):
    """Standardized samples [start,particle,coordinate], targets [start,coordinate]."""
    observation = torch.linalg.vector_norm(samples - target[:, None], dim=-1).mean()
    m = samples.shape[1]
    i, j = torch.triu_indices(m, m, offset=1, device=samples.device)
    pair = torch.linalg.vector_norm(samples[:, i] - samples[:, j], dim=-1).sum(1).mean() / (m * (m - 1))
    return observation - pair, observation, pair


def endpoint_probes(samples, targets, current, model, position_weight, velocity_weight, offset):
    """Separate physical position mean and spread without changing velocities."""
    d = model.config.reduced_dim
    position = (current[:, None] + samples[..., :d] - model.state_mean) / model.state_std
    target_position = (current + targets[..., :d] - model.state_mean) / model.state_std
    velocity = samples[..., d:] / model.target_std
    target_velocity = targets[..., d:] / model.target_std
    def joint(p):
        return torch.cat((position_weight ** .5 * p, velocity_weight ** .5 * velocity), -1)
    target_joint = torch.cat((position_weight ** .5 * target_position,
                              velocity_weight ** .5 * target_velocity), -1)
    base, observation, pair = score_terms(joint(position), target_joint)
    pos_score = score_terms(position, target_position)[0]
    vel_score = score_terms(velocity, target_velocity)[0]
    pos_error = position_weight * (position - target_position[:, None]).square().sum(-1).mean()
    vel_error = velocity_weight * (velocity - target_velocity[:, None]).square().sum(-1).mean()
    generated_mean = position.mean((0, 1))
    reference_mean = target_position.mean(0)
    masks = {"all_positions": torch.ones(d, device=position.device),
             "leading_pod": torch.zeros(d, device=position.device)}
    masks['leading_pod'][offset:min(offset + 3, d)] = 1
    if offset:
        masks['spatial_mean'] = torch.zeros(d, device=position.device)
        masks['spatial_mean'][0] = 1
    result = dict(joint_score=float(base), observation_distance=float(observation),
                  subtracted_pair_distance=float(pair), position_marginal_score=float(pos_score),
                  velocity_marginal_score=float(vel_score),
                  weighted_position_squared_error=float(pos_error),
                  weighted_velocity_squared_error=float(vel_error),
                  position_squared_error_fraction=float(pos_error / (pos_error + vel_error).clamp_min(1e-30)))
    for name, mask in masks.items():
        for fraction in (.25, .5):
            changed = position - fraction * mask * (generated_mean - reference_mean)
            result[f'{name}_mean_correction_{fraction:g}_score_delta'] = float(score_terms(joint(changed), target_joint)[0] - base)
        for factor in (.8, .9, 1.1, 1.2):
            changed = position + (factor - 1) * mask * (position - generated_mean)
            result[f'{name}_amplitude_{factor:g}_score_delta'] = float(score_terms(joint(changed), target_joint)[0] - base)
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError('Nonfinite joint-score endpoint probe')
    return result


class ProbeScore(MultistepVelocityEnergyScore):
    """Observe endpoints inside the existing recurrence; change no sampling code."""
    def _score_samples(self, model, horizon, samples, target):
        self.probes[horizon] = endpoint_probes(samples, target, self.initial,
            model, self.position_weight, self.velocity_weight, self.offset)
        loss = super()._score_samples(model, horizon, samples, target)
        if not np.isclose(float(loss), self.probes[horizon]['joint_score'], rtol=2e-5, atol=2e-5):
            raise AssertionError('Probe score differs from training score')
        return loss


def diagnose(checkpoint, output, *, device='cpu', windows=128, window_seeds=(41019,41020,41021),
             seeds=(51019,51020,51021), horizons=(2,4,10,20,50,100), particles=16,
             min_history=15):
    if windows < 2 or particles < 2 or not seeds or not window_seeds:
        raise ValueError('Need at least two windows/particles and nonempty seeds')
    device = torch.device(device)
    model, saved = load_model(Path(checkpoint), device)
    model.eval()
    if not isinstance(min_history, int) or isinstance(min_history, bool) or min_history < 0:
        raise ValueError('min_history must be a nonnegative integer')
    model_history_steps = model.config.history_steps
    eligibility_history_steps = max(model_history_steps, min_history)
    train = saved['training']
    if train.get('multistep_target') != 'joint':
        raise ValueError('Joint checkpoint required')
    data = checkpoint_data(saved)
    objective = ProbeScore(tuple(horizons), particles, target='joint',
        position_weight=train['multistep_position_weight'],
        velocity_weight=train['multistep_velocity_weight'])
    objective.offset = int(data.include_spatial_mean)
    records = []
    selected_by_batch = {}
    with torch.no_grad():
        for window_seed in window_seeds:
            parts = list(_capped_multistep_batches(data, model, 'validation', windows,
                tuple(horizons), windows, window_seed, train.get('train_trim_norm_cutoff'),
                train.get('train_trim_state_std'),
                eligibility_history_steps=eligibility_history_steps,
                include_window_ids=True))
            if not parts:
                raise ValueError('No retained probe windows')
            batch = {key: torch.cat([b[key] for b in parts]).to(device) for key in parts[0]}
            identifiers = _window_identifiers(batch, data, 'validation')
            selected_by_batch[str(window_seed)] = {
                'identifiers': identifiers,
                'sha256': _identifier_digest(identifiers),
            }
            objective.initial = batch['current_state']
            targets = torch.cat((batch['target_displacements'], batch['target_velocities']), -1)
            for seed in seeds:
                objective.probes = {}
                objective.per_horizon(model, batch['current_state'], targets,
                    history_states=batch['history_states'],
                    generator=torch.Generator(device=device).manual_seed(seed))
                records.append(dict(window_seed=window_seed, particle_seed=seed,
                                    windows=len(objective.initial), horizons=objective.probes))
    by_horizon = {}
    for horizon in horizons:
        keys = records[0]['horizons'][horizon]
        by_horizon[str(horizon)] = {}
        for key in keys:
            values = np.asarray([r['horizons'][horizon][key] for r in records])
            item = _distribution_summary(values)
            if key.endswith('score_delta'):
                item['fraction_draws_lower_score'] = float((values < 0).mean())
            by_horizon[str(horizon)][key] = item
    report = dict(checkpoint=str(Path(checkpoint).resolve()), split='validation',
        requested_eligibility_history_steps=min_history,
        eligibility_history_steps=eligibility_history_steps,
        model_conditioning_history_steps=model_history_steps,
        selected_window_identifiers_by_batch=selected_by_batch,
        max_horizon=max(horizons),
        window_seeds=list(window_seeds), particle_seeds=list(seeds), particles=particles,
        windows_per_batch=windows, by_horizon=by_horizon, draws=records,
        interpretation='Negative score deltas favor the endpoint modification. Mean corrections use measured endpoint means (oracle diagnostic only). Amplitude changes preserve the pooled generated position mean and all velocities. These endpoints need not correspond to any realizable model change. Position and velocity marginal scores are NOT additive components of the joint score. All horizons here use common max-horizon-clean windows, unlike training each objective on its own eligible population. Draw fractions are descriptive and are not confidence levels; draws reuse batches and random seeds. Normalization remains checkpoint-specific.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines=['# Joint energy-score sensitivity','',report['interpretation'],'',
           '| horizon | joint ES | position squared-error fraction | leading POD amplitude 0.9 delta | leading POD half-mean-correction delta |',
           '|---:|---:|---:|---:|---:|']
    for h, row in by_horizon.items():
        lines.append(f"| {h} | {row['joint_score']['mean']:.5g} | {row['position_squared_error_fraction']['mean']:.3f} | {row['leading_pod_amplitude_0.9_score_delta']['mean']:.5g} | {row['leading_pod_mean_correction_0.5_score_delta']['mean']:.5g} |")
    (output/'quick_summary.md').write_text('\n'.join(lines)+'\n')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--windows',type=int,default=128)
    parser.add_argument('--window-seeds',type=int,nargs='+',default=[41019,41020,41021])
    parser.add_argument('--seeds',type=int,nargs='+',default=[51019,51020,51021])
    parser.add_argument('--horizons',type=int,nargs='+',default=[2,4,10,20,50,100])
    parser.add_argument('--particles',type=int,default=16)
    parser.add_argument('--min-history',type=int,default=15)
    args=vars(parser.parse_args()); args['device']=resolve_device(args['device'])
    print(diagnose(**args))


if __name__=='__main__':
    main()
