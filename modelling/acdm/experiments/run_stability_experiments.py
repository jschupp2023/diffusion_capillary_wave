"""Controlled validation experiments on shared-POD ACDM rollout stability.

This runner samples training transitions uniformly with replacement. A round
is a fixed number of optimizer updates, not a full pass through the data.
Existing checkpoints and source data are read-only; outputs use a new directory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from modelling.acdm.conditional_edm.checkpoint import load_checkpoint, load_model, save_checkpoint
from modelling.acdm.conditional_edm.config import EDMConfig, TrainConfig
from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.acdm.conditional_edm.train import seed_everything
from .conditioning_errors import build_error_bank, perturb_batch


class ResidentTransitions:
    """GPU coordinates with valid start indices that never cross repetitions."""
    def __init__(self, data, device):
        arrays, self.ranges = [], []
        offset = 0
        for name in data.train_repetitions:
            _, values = data.read_coordinates(name, 0, data.frame_counts[name])
            arrays.append(torch.as_tensor(values, dtype=torch.float32, device=device))
            self.ranges.append((offset, offset + len(values)))
            offset += len(values)
        self.values = torch.cat(arrays)
        self.indices = {}

    def sample(self, config, size, generator):
        lag = config.lag_steps
        history = config.history_steps if config.history_conditioning else 0
        key = (lag, history)
        if key not in self.indices:
            self.indices[key] = torch.cat([torch.arange(a + history * lag, b - lag, device=self.values.device)
                                           for a, b in self.ranges])
        valid = self.indices[key]
        ids = valid[torch.randint(len(valid), (size,), device=self.values.device, generator=generator)]
        batch = dict(current_state=self.values[ids], next_state=self.values[ids + lag])
        if history:
            past = torch.arange(1, history + 1, device=self.values.device) * lag
            batch['history_states'] = self.values[ids[:, None] - past]
        return batch


def warm_start(path, history_steps, device):
    old, checkpoint = load_model(path, device)
    cfg = replace(old.config, history_steps=history_steps)
    if cfg == old.config:
        return old, checkpoint
    if history_steps < old.config.history_steps or cfg.param_dim:
        raise ValueError('Warm start supports extending history without parameter channels.')
    model = type(old)(cfg).to(device)
    state = model.state_dict()
    for name, value in old.state_dict().items():
        if state[name].shape == value.shape:
            state[name] = value
        elif name == 'backbone.input.weight':
            state[name].zero_()
            if cfg.diffusion_formulation == 'ddpm':
                old_c, new_c = old.config.conditioning_dim, cfg.conditioning_dim
                state[name][:, :old_c] = value[:, :old_c]
                state[name][:, new_c:] = value[:, old_c:]
            else:
                state[name][:, :value.shape[1]] = value
        elif name in ('backbone.output.weight', 'backbone.output.bias') and cfg.conditioning_mode == 'joint_noised':
            if cfg.diffusion_formulation == 'ddpm':
                old_c, new_c = old.config.conditioning_dim, cfg.conditioning_dim
                state[name][:old_c] = value[:old_c]
                state[name][new_c:] = value[old_c:]
            else:
                state[name][:value.shape[0]] = value
        else:
            raise ValueError(f'Cannot transfer {name}: {value.shape} -> {state[name].shape}')
    model.load_state_dict(state)
    return model, checkpoint


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(args):
    from .rollout_benchmark import RolloutBenchmark
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    seed_everything(args.seed)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    json_write(directory/'plan.json', plan)
    source = checkpoint_data(load_checkpoint(args.baseline))
    print('Preparing shared coordinates', flush=True)
    cache = source.cache_coordinates(args.cache_gib)
    print(json.dumps({'cache': cache}), flush=True)
    resident = ResidentTransitions(source, device)
    benchmark = RolloutBenchmark(source, horizon=args.horizon, conditions=args.conditions,
                                 ensemble_size=args.ensemble, seed=args.validation_seed)
    signature = source.signature()
    outcomes = []
    for spec in plan:
        dest = directory/spec['name']
        if (dest/'result.json').exists():
            outcomes.append(json.loads((dest/'result.json').read_text()))
            continue
        try:
            dest.mkdir(exist_ok=True)
            seed_everything(args.seed)
            parent = Path(spec.get('checkpoint', str(args.baseline)))
            old_cfg = EDMConfig.from_dict(load_checkpoint(parent)['model_config'])
            model, parent_checkpoint = warm_start(parent, spec.get('history_steps', old_cfg.history_steps), device)
            if parent_checkpoint['data_signature'] != signature:
                raise ValueError('Parent checkpoint sources/splits differ from this experiment.')
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec.get('learning_rate', args.learning_rate), weight_decay=0.)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            augmentation_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
            loss_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
            bank = None
            history, best = [], float('inf')
            rounds = spec.get('rounds', args.rounds)
            print(json.dumps({'candidate': spec, 'rounds': rounds}), flush=True)
            start = perf_counter()
            for round_index in range(rounds + 1):
                record = {'round': round_index, 'updates': round_index * args.updates}
                if round_index:
                    augmentation = spec.get('augmentation', 'none')
                    if augmentation != 'none' and (bank is None or (round_index-1) % args.refresh_bank_every == 0):
                        bank = build_error_bank(model, source, num_conditions=args.bank_conditions,
                                                horizon=spec.get('error_horizon', args.error_horizon),
                                                seed=args.seed + 1000 + round_index, sampling_steps=args.sampling_steps,
                                                max_error_std=spec.get('max_error_std', 2.))
                        bank = bank.to(device)
                        record['error_bank'] = bank.summary
                    model.train()
                    loss_sum = torch.zeros((), device=device)
                    for _ in range(args.updates):
                        batch = resident.sample(model.config, args.batch_size, generator)
                        if augmentation == 'errors':
                            batch = perturb_batch(batch, bank, strength=spec.get('error_strength', .3),
                                                  probability=spec.get('probability', .5), generator=augmentation_generator)
                        elif augmentation == 'replay':
                            replay = bank.sample_replay_batch(args.batch_size, generator=augmentation_generator)
                            mask = torch.rand(args.batch_size, device=device, generator=augmentation_generator) < spec.get('probability', .25)
                            batch = {key: torch.where(mask.reshape(-1, *([1] * (value.ndim-1))), replay[key], value)
                                     for key, value in batch.items()}
                        optimizer.zero_grad(set_to_none=True)
                        loss, metrics = model.loss(batch, generator=loss_generator)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                        optimizer.step()
                        loss_sum += loss.detach()
                    record['training_loss'] = float(loss_sum / args.updates)
                    if not np.isfinite(record['training_loss']):
                        raise ValueError('Training loss became nonfinite.')
                if round_index == 0 or round_index % args.evaluate_every == 0 or round_index == rounds:
                    model.eval()
                    report = benchmark.evaluate(model)
                    record['validation'] = report
                    score = report['score']
                    if score < best:
                        best = score
                        train_cfg = TrainConfig(batch_size=args.batch_size, epochs=max(1, rounds),
                                                learning_rate=spec.get('learning_rate', args.learning_rate),
                                                max_train_batches=args.updates, rollout_every=0, seed=args.seed)
                        payload = dict(format_version=2, model_config=model.config.to_dict(), train_config=train_cfg.to_dict(),
                                       model_state=model.state_dict(), normalization=model.normalization_state(),
                                       optimizer_state=optimizer.state_dict(), scheduler_state=None,
                                       epoch=round_index-1, global_step=record['updates'], best_validation_loss=float('inf'),
                                       history=history+[record], split_labels={k:list(v) for k,v in source.splits.items()},
                                       data_config=source.source_config, data_signature=signature,
                                       rng_state={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},
                                       experiment={'spec':spec,'parent':str(parent.resolve()),'selection':'validation rollout score',
                                                   'protocol':'uniform training transitions with replacement',
                                                   'updates_per_round':args.updates,'validation':report})
                        save_checkpoint(dest/'best.pt', payload)
                        json_write(dest/'best_validation.json', report)
                record['elapsed_seconds'] = perf_counter() - start
                history.append(record)
                with (dest/'history.jsonl').open('a') as f:
                    f.write(json.dumps(record, allow_nan=False)+'\n')
                print(json.dumps({'candidate':spec['name'], **record}, allow_nan=False), flush=True)
            result = dict(name=spec['name'], best_score=best, checkpoint=str(dest/'best.pt'),
                          seconds=perf_counter()-start, spec=spec)
            json_write(dest/'result.json', result)
            outcomes.append(result)
            json_write(directory/'results.json', sorted(outcomes, key=lambda v:v.get('best_score', float('inf'))))
            del model, optimizer, bank
            torch.cuda.empty_cache()
        except Exception as error:
            result = dict(name=spec['name'], failed=True, error=str(error), spec=spec)
            json_write(dest/'failure.json', result)
            outcomes.append(result)
            print(json.dumps(result), flush=True)
            json_write(directory/'results.json', sorted(outcomes, key=lambda v:v.get('best_score', float('inf'))))
    return outcomes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('runs/0p20_r50_lag1/best.pt'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--validation-seed',type=int,default=814)
    p.add_argument('--cache-gib',type=float,default=2.)
    p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--updates',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--evaluate-every',type=int,default=4)
    p.add_argument('--horizon',type=int,default=256)
    p.add_argument('--conditions',type=int,default=4)
    p.add_argument('--ensemble',type=int,default=2)
    p.add_argument('--sampling-steps',type=int,default=32)
    p.add_argument('--bank-conditions',type=int,default=64)
    p.add_argument('--error-horizon',type=int,default=8)
    p.add_argument('--refresh-bank-every',type=int,default=4)
    run(p.parse_args())

if __name__=='__main__':
    main()
