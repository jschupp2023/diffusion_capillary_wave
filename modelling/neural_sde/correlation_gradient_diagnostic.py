"""Frozen train/validation audit of correlation matching; never optimizes weights.

Reuses the real two-branch loss and objective-balance gradient accounting.
Records per-lag errors/gradients and separates start-batch from rollout-noise
variation. Parent GMM/energy-score and endpoint objectives remain at their
saved horizons. Long correlation horizons do not alter these anchors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .correlation import fit_correlation_reference
from .losses import (StationaryMeanLoss, EulerMaruyamaNLL, MonteCarloMultistepNLL,
                     MultistepVelocityEnergyScore)
from .train import load_model, _globally_mixed_retained_velocity_batches
from .objective_balance_diagnostic import (_gradient_vector, _parameter_groups,
    _group_slices, _objective_summary, _window_identifiers, _cosine)
from modelling.acdm.conditional_edm.data import checkpoint_data


def noise_components(vectors):
    """Input [start batch, noise seed, parameter]; no independence claim for starts."""
    x = np.asarray(vectors)
    mean = x.mean((0, 1))
    batch_means = x.mean(1)
    norm = float(np.linalg.norm(mean))
    within = float(np.sqrt(np.mean(np.sum((x-batch_means[:, None])**2, axis=-1))))
    between = float(np.sqrt(np.mean(np.sum((batch_means-mean)**2, axis=-1))))
    return dict(mean_norm=norm, within_batch_noise_rms=within if x.shape[1]>1 else None,
                between_batch_rms=between if x.shape[0]>1 else None,
                within_batch_noise_over_mean=within/norm if norm and x.shape[1]>1 else None,
                between_batch_over_mean=between/norm if norm and x.shape[0]>1 else None)


def diagnose(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    if args.starts < 2 or args.weight < 0 or not np.isfinite(args.weight) or not 0 < args.fraction <= 1:
        raise ValueError("requires >=2 starts, finite nonnegative weight, and fraction in (0,1]")
    if (len(set(args.window_seeds)) != len(args.window_seeds)
            or len(set(args.noise_seeds)) != len(args.noise_seeds)):
        raise ValueError("seeds must be distinct")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, checkpoint = load_model(Path(args.checkpoint), device)
    if model.fixed_memory is not None:
        raise ValueError("audit the original native neural SDE, not its fixed-memory extension")
    if model.config.state_variable != "velocity" or model.config.lag_steps != 1:
        raise ValueError("correlation gradient audit requires native-lag velocity")
    model.tcn_rollout_cache = True
    model.indexed_training_batches = True
    model.eval()
    fingerprint = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    training = checkpoint.get("training", {})
    data = checkpoint_data(checkpoint)
    data.cache_coordinates(2.)
    cutoff, scale = training.get("train_trim_norm_cutoff"), training.get("train_trim_state_std")
    horizons = tuple(args.horizons)
    mh = tuple(training.get("multistep_horizons") or (1,))
    eh = tuple(training.get("stationary_mean_horizons") or (training.get("stationary_mean_horizon",100),))
    union = tuple(sorted(set(horizons+mh+eh+(1,))))
    reference, reference_report = fit_correlation_reference(data, model, union, cutoff, scale)
    corr = StationaryMeanLoss(horizons, lagged_covariance_coordinates=tuple(args.coordinates),
        lagged_covariance_reference_variance=reference).to(device)
    target = training.get("multistep_target", "velocity")
    kwargs = dict(target=target, mode_counts=training.get("multistep_mode_counts"),
        pod_coordinate_offset=int(data.include_spatial_mean), particle_counts=training.get("multistep_particle_counts"),
        horizon_weights=training.get("multistep_horizon_weights"))
    cls = MonteCarloMultistepNLL
    if training.get("multistep_objective", "gmm_nll") == "energy_score":
        cls = MultistepVelocityEnergyScore
        kwargs.update(position_weight=training.get("multistep_position_weight",1.),
                      velocity_weight=training.get("multistep_velocity_weight",1.))
    multistep = cls(mh, training.get("multistep_particles",100), **kwargs)
    moments = StationaryMeanLoss(eh, pod_coordinate_offset=int(data.include_spatial_mean),
        variance_pod_mode_count=training.get("stationary_variance_pod_mode_count")).to(device)
    parameters, indices, counts = _parameter_groups(model)
    groups = _group_slices(parameters, indices)
    drift = groups["drift"]
    all_indices = groups["all"]
    routes = dict(one_step_nll=all_indices,
        multistep=np.unique(np.concatenate([groups[g] for g in training.get("multistep_gradient_groups",["drift"])])),
        stationary_mean=drift, stationary_variance=all_indices, correlation=drift)
    specs = dict(one_step_nll=(1.,1.),
        multistep=(training.get("multistep_weight",0.),training.get("multistep_window_fraction",1.)),
        stationary_mean=(training.get("stationary_mean_weight",0.),training.get("stationary_mean_batch_fraction",1.)),
        stationary_variance=(training.get("stationary_variance_weight",0.),training.get("stationary_mean_batch_fraction",1.)),
        correlation=(args.weight,args.fraction))
    samples = {key: [] for key in specs}
    losses = {key: [] for key in specs}
    lag_samples = {h: [] for h in horizons}
    lag_losses = {h: [] for h in horizons}
    errors, identifiers, timings = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for ws in args.window_seeds:
        batch = next(_globally_mixed_retained_velocity_batches(data, model, args.split,
            min(256,data.batch_size), args.starts, ws, cutoff, scale, union, include_window_ids=True), None)
        if batch is None:
            raise ValueError("no complete eligible start ensemble")
        identifiers.append(dict(window_seed=ws, windows=_window_identifiers(batch,data,args.split)))
        current, history, endpoints = [batch[k].to(device) for k in ("current_state","history_states","data_endpoints")]
        following = endpoints[:,union.index(1)]
        following_history = torch.cat((current[:,None],history[:,:-1]),1)
        # Every objective uses identical start conditions, with the full union stencil.
        step = EulerMaruyamaNLL()(model,current,following,history_states=history,following_history_states=following_history)
        nll_vector = _gradient_vector(step,parameters)
        nll_value = float(step.detach())
        targets_v = []
        # Read exact native endpoint velocities; endpoint displacement alone is insufficient.
        for ident in identifiers[-1]["windows"]:
            values = data._coordinate_cache[ident["source"]][1] if ident["source"] in data._coordinate_cache else data.read_coordinates(ident["source"],0,data.frame_counts[ident["source"]])[1]
            ends = ident["start_index"]+model.config.lag_steps*np.asarray(mh)
            targets_v.append(values[ends]-values[ends-1])
        velocities = torch.tensor(np.asarray(targets_v),dtype=current.dtype,device=device)
        displacements = endpoints[:,[union.index(h) for h in mh]]-current[:,None]
        targets = (velocities if target=="velocity" else displacements if target=="displacement" else
                   torch.cat((displacements,velocities),-1))
        for ns in args.noise_seeds:
            tick = time.monotonic()
            if device.type=="cuda": torch.cuda.synchronize(device)
            samples["one_step_nll"].append(nll_vector);losses["one_step_nll"].append(nll_value)
            loss = multistep(model,current,targets,history_states=history,
                generator=torch.Generator(device=device).manual_seed(ns))
            losses["multistep"].append(float(loss.detach()));samples["multistep"].append(_gradient_vector(loss,parameters))
            st = moments.per_horizon_statistics(model,current,endpoints[:,[union.index(h) for h in eh]],
                history_states=history,generator=torch.Generator(device=device).manual_seed(ns+10000))
            for key, column in (("stationary_mean",0),("stationary_variance",2)):
                loss = torch.stack([v[column] for v in st.values()]).mean()
                losses[key].append(float(loss.detach()));samples[key].append(_gradient_vector(loss,parameters,retain_graph=column==0))
            st = corr.per_horizon_statistics_with_lagged_covariance(model,current,
                endpoints[:,[union.index(h) for h in horizons]],history_states=history,
                generator=torch.Generator(device=device).manual_seed(ns+20000))
            errors.append({str(h):v[5].detach().cpu().tolist() for h,v in st.items()})
            for h,v in st.items():
                lag_losses[h].append(float(v[4].detach()))
                lag_samples[h].append(_gradient_vector(v[4],parameters,retain_graph=True))
            loss = torch.stack([v[4] for v in st.values()]).mean()
            losses["correlation"].append(float(loss.detach()));samples["correlation"].append(_gradient_vector(loss,parameters))
            if device.type=="cuda": torch.cuda.synchronize(device)
            timings.append(time.monotonic()-tick)
            print(f"window seed {ws}, noise seed {ns}: {timings[-1]:.2f}s; correlation {losses['correlation'][-1]:.6g}",flush=True)
            del st, loss
    reports, vectors = {}, {}
    for name in samples:
        w,f = specs[name]
        report = _objective_summary(losses[name],samples[name],weight=w,fraction=f,route_indices=routes[name],groups=groups)
        vectors[name] = report.pop("mean_routed_gradient")
        arr = np.stack(samples[name]).reshape(len(args.window_seeds),len(args.noise_seeds),-1)
        report["noise_components_drift"] = noise_components(arr[:,:,drift])
        reports[name] = report
    lag_reports = {}
    for h in horizons:
        report = _objective_summary(lag_losses[h],lag_samples[h],weight=args.weight/len(horizons),
            fraction=args.fraction,route_indices=drift,groups=groups)
        v = report.pop("mean_routed_gradient")
        cosine = _cosine(v[drift],vectors["one_step_nll"][drift])
        report.update(lag_ms=h*model.config.lag_steps*model.config.dt*1000,
                      cosine_with_nll_drift=float(cosine) if np.isfinite(cosine) else None)
        lag_reports[str(h)] = report
    anchor = sum(specs[k][0]*v for k,v in vectors.items() if k!="correlation")
    cosine = _cosine(anchor[drift],vectors["correlation"][drift])
    summary = dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=fingerprint,
        data_config=checkpoint.get("data_config"),data_signature=checkpoint.get("data_signature"),
        split=args.split,coordinates=args.coordinates,horizons=args.horizons,
        reference=reference_report,window_identifiers=identifiers,window_seeds=args.window_seeds,
        noise_seeds=args.noise_seeds,starts_per_batch=args.starts,objectives=reports,
        per_horizon=lag_reports,per_coordinate_errors=errors,
        parent_anchor_drift_norm=float(np.linalg.norm(anchor[drift])),
        correlation_anchor_cosine=float(cosine) if np.isfinite(cosine) else None,
        seconds_per_seed=timings,elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        parent_objective_horizons=dict(multistep=mh,endpoint=eh),
        caution="Frozen diagnostic; no optimizer updates. Two rollout branches per start. All objectives use common max-lag eligible starts. Noise estimates from few batches/seeds are preliminary; gradient norms are not Adam steps.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/"gradients.npz",**{k:np.stack(v) for k,v in samples.items()},
                        **{f'correlation_h{h}':np.stack(v) for h,v in lag_samples.items()})
    lines=['# Correlation gradient preflight','',summary['caution'],'',
        '| Objective | Expected weighted drift norm | Selected-update drift norm |','|---|---:|---:|']
    for name,r in reports.items():
        g=r['groups']['drift'];lines.append(f"| {name} | {g['expected_weighted_gradient_norm']:.6g} | {g['selected_update_weighted_gradient_norm']:.6g} |")
    lines += ['',f"Correlation vs existing anchor cosine: {summary['correlation_anchor_cosine']}",
        f"Total elapsed: {summary['elapsed_seconds']:.2f}s",'',
        'Per-lag gradients, per-coordinate errors, start/noise variability and complete provenance: summary.json.']
    (out/'quick_summary.md').write_text('\n'.join(lines)+'\n')
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()!=fingerprint:
        raise RuntimeError("source checkpoint changed during diagnostic")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['train','validation'],default='train')
    p.add_argument('--coordinates',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--horizons',type=int,nargs='+',default=[128,256,512,1024,2048,4096])
    p.add_argument('--starts',type=int,default=16)
    p.add_argument('--window-seeds',type=int,nargs='+',default=[0,1])
    p.add_argument('--noise-seeds',type=int,nargs='+',default=[0,1])
    p.add_argument('--weight',type=float,default=.01);p.add_argument('--fraction',type=float,default=.01)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args(argv);diagnose(args)


if __name__=='__main__':
    main()
