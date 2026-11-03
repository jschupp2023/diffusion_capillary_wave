"""Bounded, read-only-checkpoint profile of the current TCN multistep recipe.

Uses measured, trim-eligible training windows and the production losses. Adam
has lr=0: it exercises the update without changing model parameters. Nothing
is written to the source checkpoint or its training directory.
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import time

import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.neural_sde.losses import EulerMaruyamaNLL, MonteCarloMultistepNLL
from modelling.neural_sde.batches import indexed_velocity_batches
from modelling.neural_sde.train import (
    load_model, _sde_transition_batches, _trim_keep, _retained_multistep_batches,
    _multistep_gradient_parameters,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--tcn-rollout-cache', action='store_true')
    parser.add_argument('--indexed-training-batches', action='store_true')
    parser.add_argument('--skip-trace', action='store_true')
    parser.add_argument('--compile-cache', action='store_true',
                        help='Benchmark compilation of the cached TCN kernel (requires cache).')
    args = parser.parse_args()
    if args.batches < 3 or args.output.exists():
        parser.error('Need >=3 batches and a new output directory')
    args.output.mkdir(parents=True)
    device = torch.device('cuda')
    model, checkpoint = load_model(args.checkpoint, device)
    model.tcn_rollout_cache = args.tcn_rollout_cache
    model.indexed_training_batches = args.indexed_training_batches
    if args.compile_cache:
        if not args.tcn_rollout_cache:
            parser.error('--compile-cache requires --tcn-rollout-cache')
        model.tcn_rollout_compile = True
    model.train()
    training = checkpoint['training']
    horizons = tuple(training['multistep_horizons'])
    particles = tuple(training['multistep_particle_counts'])
    fraction = training['multistep_window_fraction']
    weight = training['multistep_weight']
    assert weight > 0 and training['multistep_window_sampling'] == 'examples'
    assert training['multistep_target'] == 'velocity'
    assert training['multistep_gradient_groups'] == ['drift']
    batch_size = args.batch_size or training['batch_size']
    cutoff, scale = training['train_trim_norm_cutoff'], training['train_trim_state_std']
    start = time.perf_counter()
    data = checkpoint_data(checkpoint)
    coordinate_cache = data.cache_coordinates(2.)  # production train() does this too
    data_open_s = time.perf_counter()-start
    one_batches = iter(indexed_velocity_batches(
        data, model.config, 'train', batch_size, seed=0, cutoff=cutoff, state_std=scale)
        if args.indexed_training_batches else
        _sde_transition_batches(data, model.config, 'train', batch_size, seed=0))
    selection_rng = torch.Generator().manual_seed(0)
    multi_batches = iter(indexed_velocity_batches(
        data, model.config, 'train', batch_size, 0, horizons, cutoff, scale,
        fraction, selection_rng) if args.indexed_training_batches else
        _retained_multistep_batches(data, model, 'train', batch_size, horizons, 0, cutoff, scale))
    cpu_batches, fetch = [], []
    for index in range(args.batches + 3):
        start = time.perf_counter()
        while True:
            batch = next(one_batches)
            keep = (torch.ones(len(batch['current_state']), dtype=torch.bool)
                    if args.indexed_training_batches else _trim_keep(batch, model, cutoff, scale))
            if keep.any():
                if not args.indexed_training_batches:
                    batch = {k:v[keep] for k,v in batch.items()}
                break
        one_s = time.perf_counter()-start
        start = time.perf_counter()
        multi = next(multi_batches)
        if args.indexed_training_batches:
            take = len(multi['current_state'])
            selected = torch.arange(take)
        else:
            take = max(1, math.ceil(fraction*len(multi['current_state'])))
            selected = torch.randperm(len(multi['current_state']), generator=selection_rng)[:take].sort().values
        multi_s = time.perf_counter()-start
        cpu_batches.append((batch, multi, selected))
        fetch.append(dict(one_step_s=one_s, multistep_s=multi_s,
                          one_count=len(batch['current_state']), multi_candidates=len(multi['current_state']),
                          selected_count=take))
        print('Prepared batch', index, fetch[-1], flush=True)
    one_objective = EulerMaruyamaNLL()
    multi_objective = MonteCarloMultistepNLL(horizons=horizons, particle_counts=particles)
    parameters, _ = _multistep_gradient_parameters(model, ('drift',))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    records = []

    def step(inputs, timed=False, multistep=True):
        durations = {}
        @contextmanager
        def stage(name):
            if timed:
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.profiler.record_function(name):
                yield
            if timed:
                torch.cuda.synchronize()
            durations[name] = (time.perf_counter()-start)*1000
        batch, multi, selected = inputs
        with stage('prepare_and_transfer'):
            current, following = batch['current_state'].to(device), batch['next_state'].to(device)
            history = batch['history_states'].to(device)
            next_history = batch['next_history_states'].to(device)
        with stage('clean_one_step_forward'):
            one = one_objective(model, current, following, history_states=history,
                                following_history_states=next_history)
        if multistep:
            with stage('multistep_prepare_and_transfer'):
                mc = multi['current_state'].index_select(0, selected).to(device)
                mh = multi['history_states'].index_select(0, selected).to(device)
                targets = multi['target_velocities'].index_select(0, selected).to(device)
            with stage('multistep_forward'):
                losses = multi_objective.per_horizon(model, mc, targets, history_states=mh)
                auxiliary = weight * torch.stack(tuple(losses.values())).mean()
        with stage('finite_check_and_zero_grad'):
            assert torch.isfinite(one + auxiliary if multistep else one)
            optimizer.zero_grad(set_to_none=True)
        with stage('clean_one_step_backward'):
            one.backward()
        if multistep:
            with stage('multistep_backward'):
                gradients = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
            with stage('accumulate_auxiliary_gradients'):
                for parameter, gradient in zip(parameters, gradients):
                    if gradient is not None:
                        if parameter.grad is None:
                            parameter.grad = gradient
                        else:
                            parameter.grad.add_(gradient)
        with stage('adam_update'):
            optimizer.step()
            model.validate_velocity_memory()
        with stage('loss_metrics_to_cpu'):
            float(one.detach())
            if multistep:
                for value in losses.values():
                    float(value.detach())
        return durations

    for batch in cpu_batches[:3]:
        step(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for batch in cpu_batches[3:]:
        records.append(step(batch, timed=True))
    throughput = {}
    for enabled in [False, True]:
        start = time.perf_counter()
        for batch in cpu_batches[3:]:
            step(batch, multistep=enabled)
        torch.cuda.synchronize()
        throughput['multistep' if enabled else 'one_step_only'] = (time.perf_counter()-start)*1000/args.batches
    table = ''
    if not args.skip_trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as prof:
            step(cpu_batches[-1])
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output/'trace.json'))
        table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)
        (args.output/'operators.txt').write_text(table)
    for k,v in model.state_dict().items():
        assert torch.equal(v.detach().cpu(), initial[k]), k
    timing_peak_mib = torch.cuda.max_memory_allocated()/2**20
    # Compare the actual checkpoint's MC loss and all parameter gradients with
    # identical CUDA noise, independently of lr=0 optimizer timings above.
    verification = []
    _, multi, selected = cpu_batches[-1]
    mc = multi['current_state'].index_select(0, selected).to(device)
    mh = multi['history_states'].index_select(0, selected).to(device)
    targets = multi['target_velocities'].index_select(0, selected).to(device)
    convolution_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # cuDNN defaults to allowing TF32 whereas Linear defaults to IEEE
        # FP32. Test function/gradient equivalence with MATCHED precision,
        # rather than confusing that pre-existing backend difference with a
        # cache error. These settings do not affect the timings above.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        for cached in (False, args.tcn_rollout_cache):
            model.tcn_rollout_cache = cached
            value = multi_objective(model, mc, targets, history_states=mh,
                                    generator=torch.Generator(device=device).manual_seed(29))
            gradients = torch.autograd.grad(value, tuple(model.parameters()), allow_unused=True)
            vector = torch.cat([g.flatten() for g in gradients if g is not None]).detach()
            verification.append((value.detach(), vector))
    finally:
        torch.backends.cudnn.allow_tf32 = convolution_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    ref, fast = verification
    accuracy = dict(loss_relative_error=((fast[0]-ref[0]).abs()/ref[0].abs().clamp_min(1e-12)).item(),
                    gradient_relative_l2=((fast[1]-ref[1]).norm()/ref[1].norm().clamp_min(1e-12)).item())
    if accuracy['loss_relative_error'] > 1e-4 or accuracy['gradient_relative_l2'] > 1e-3:
        raise AssertionError(f'Cached/reference verification failed: {accuracy}')
    summary = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_epoch=checkpoint['epoch'],
                   device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                   cpu_threads=torch.get_num_threads(), batch_size=batch_size,
                   coordinate_cache=coordinate_cache,
                   tcn_rollout_cache=args.tcn_rollout_cache,
                   indexed_training_batches=args.indexed_training_batches,
                   compile_cache=args.compile_cache,
                   cuda_accuracy=accuracy,
                   accuracy_precision='IEEE FP32, both TF32 backends disabled',
                   timing_cudnn_tf32=convolution_tf32, timing_matmul_tf32=matmul_tf32,
                   context=model.config.tcn_context_frames, coordinates=model.config.reduced_dim,
                   particles=particles, horizons=horizons, fraction=fraction,
                   weight=weight, data_open_s=data_open_s, fetch=fetch,
                   isolated_stage_ms={k:statistics.median(r[k] for r in records) for k in records[0]},
                   full_step_mean_ms=throughput,
                   peak_allocated_MiB=timing_peak_mib,
                   limitations=['Bounded training-window sample, not a full-epoch timing.',
                                'Stage timings synchronize CUDA; full-step timing has no added stage synchronizations.',
                                'Source iterator startup is reported separately; repeated data processing later in an epoch is not measured.',
                                'Adam lr=0 exercises state/update overhead but leaves checkpoint weights unchanged.',
                                'No architecture, particle-count, tensor-dtype or objective changes; convolution and linear backend arithmetic may differ.',
                                'Coordinate caching matches production training; the original profile omitted this warmup.'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(table, flush=True)


if __name__ == '__main__':
    main()
