"""Displacement and conditional drift audit of saved native-lag velocity rollouts.

Run with ``python -m data_analysis.correlations.diagnose_velocity_drift ROLLOUT``.
Source data/checkpoints are read only. Trim reference stencils, never generated
paths; do not concatenate or compress gaps. Linear projections are descriptive
effective coefficients, not identified physical stiffness/damping operators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_analysis.rollout import Rollout, output_directory
from data_analysis.rollout_metrics import training_trim_specification, reference_training_trim_mask
from modelling.neural_sde.train import load_model


def window_mask(keep, horizon):
    """All training stencils t,...,t+h-1 must pass; never bridge a bad edge."""
    n = keep.shape[1] - horizon
    prefix = np.pad(np.cumsum(~keep, axis=1), ((0, 0), (1, 0)))
    return prefix[:, horizon:horizon+n] == prefix[:, :n]


def linear_projection(q, v, response):
    """Joint least-squares response = intercept + position matrix + velocity matrix."""
    design = np.column_stack((np.ones(len(q)), q, v))
    coef, _, rank, singular = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coef
    return coef, dict(rank=int(rank), condition_number=float(singular[0]/singular[-1]),
                     r2=(1 - np.mean((response-fitted)**2, axis=0)
                         / np.var(response, axis=0)).tolist())


def correlation(x, y):
    x, y = x-x.mean(0), y-y.mean(0)
    return np.sum(x*y, axis=0)/np.sqrt(np.sum(x*x, axis=0)*np.sum(y*y, axis=0))


def evaluate_drift(model, current, previous):
    drift, variance = [], []
    with torch.no_grad():
        for start in range(0, len(current), 4096):
            a = torch.tensor(current[start:start+4096], dtype=torch.float32)
            p = torch.tensor(previous[start:start+4096], dtype=torch.float32)[:, None]
            mean, factor = model.velocity_step_distribution(a, p, a-p[:, 0])
            drift.append((mean-(a-p[:, 0])).numpy())
            diagonal_variance = factor.square().sum(-1)
            if diagonal_variance.ndim == 1:
                diagonal_variance = diagonal_variance.expand(len(a), -1)
            variance.append(diagonal_variance.numpy())
    return np.concatenate(drift).astype(float), np.concatenate(variance).astype(float)


def structured_components(model, current, previous):
    """Exact normalized drift decomposition, including signed cancellation."""
    if model.config.drift_type != 'damped_residual':
        return None
    module = model.drift_model
    k = module.stiffness_matrix().detach().numpy().astype(float)
    d = module.damping_matrix().detach().numpy().astype(float)
    q = (current-model.state_mean.detach().numpy())/model.state_std.detach().numpy()
    v = (current-previous-model.target_mean.detach().numpy())/model.target_std.detach().numpy()
    restoring, damping = -q@k, -v@d
    total, _ = evaluate_drift(model, current, previous)
    total /= model.target_std.detach().numpy()
    linear = restoring+damping
    residual = total-linear
    result = dict(stiffness_matrix=k.tolist(), damping_matrix=d.tolist(),
                  stiffness_eigenvalues=np.linalg.eigvalsh(k).tolist(),
                  damping_eigenvalues=np.linalg.eigvalsh(d).tolist(),
                  stiffness_initial=model.config.stiffness_init,
                  damping_initial=model.config.damping_init)
    for name, sl in (('pod', slice(1,None)), ('spatial_mean', slice(0,1)), ('all', slice(None))):
        terms = {key:a[:,sl] for key,a in dict(restoring=restoring,damping=damping,
                    linear=linear,residual=residual,total=total).items()}
        rms = {key:float(np.sqrt(np.mean(a*a))) for key,a in terms.items()}
        lin,res,tot = terms['linear'],terms['residual'],terms['total']
        result[name] = dict(rms=rms, residual_over_linear_rms=rms['residual']/rms['linear'],
            residual_linear_cosine=float(np.sum(lin*res)/np.sqrt(np.sum(lin*lin)*np.sum(res*res))),
            cancellation_fraction=float(1-np.sum(tot*tot)/(np.sum(lin*lin)+np.sum(res*res))),
            residual_signed_total_projection=float(np.sum(res*tot)/np.sum(tot*tot)))
    result['rms_per_coordinate'] = {key:np.sqrt(np.mean(a*a,axis=0)).tolist()
        for key,a in dict(restoring=restoring,damping=damping,linear=linear,residual=residual,total=total).items()}
    physical_scale = model.target_std.detach().numpy()[1:]
    result['pod_physical_residual_over_linear_rms'] = float(
        np.linalg.norm(np.array(result['rms_per_coordinate']['residual'])[1:]*physical_scale)
        / np.linalg.norm(np.array(result['rms_per_coordinate']['linear'])[1:]*physical_scale))
    coef, _ = linear_projection(q, v, residual)
    result['residual_linear_projection'] = coef.tolist()
    return result


def jacobian_summary(model, current, previous, rng, samples):
    indices = rng.choice(len(current), min(samples, len(current)), replace=False)
    state = torch.tensor(current[indices], dtype=torch.float32)
    history = torch.tensor(previous[indices], dtype=torch.float32)[:, None]
    features = model._drift_input(state, history).detach().requires_grad_(True)
    drift = model.drift_model(features)
    d = current.shape[-1]
    jq, jv = [], []
    for mode in range(d):
        derivative = torch.autograd.grad(drift[:, mode].sum(), features, retain_graph=True)[0]
        jq.append(derivative[:, mode].detach().numpy())
        jv.append(derivative[:, d+mode].detach().numpy())
    jq, jv = np.stack(jq, axis=1), np.stack(jv, axis=1)
    scale = (model.target_std/model.state_std).detach().numpy()
    return dict(samples=len(indices),
                mean_effective_restoring=(-jq.mean(0)*scale).tolist(),
                mean_effective_damping=(-jv.mean(0)).tolist(),
                outward_position_derivative_fraction=(jq > 0).mean(0).tolist(),
                antidamping_derivative_fraction=(jv > 0).mean(0).tolist())


def displacement_analysis(g, r, keep, lags, fs):
    rows = []
    for h in lags:
        dg = g[:, :, h:].astype(float)-g[:, :, :-h]
        dr = r[:, h:].astype(float)-r[:, :-h]
        gm = np.mean(dg*dg, axis=(0, 1, 2))
        rm = np.mean(dr*dr, axis=(0, 1))
        valid = window_mask(keep, h)
        endpoint = keep[:, :-h] & keep[:, h:]
        row = dict(horizon=h, milliseconds=1000*h/fs,
                   full_window_count=int(valid.sum()), full_window_fraction=float(valid.mean()),
                   endpoint_count=int(endpoint.sum()), generated_ms=gm.tolist(),
                   raw_reference_ms=rm.tolist())
        for name, mask in (("full_window", valid), ("endpoint", endpoint)):
            if mask.sum() < 2:
                row[name+"_reference_ms"] = None
                row[name+"_pod_ratio"] = None
                continue
            ms = np.mean(dr[mask]**2, axis=0)
            row[name+"_reference_ms"] = ms.tolist()
            row[name+"_pod_ratio"] = float(gm[1:].sum()/ms[1:].sum())
            row[name+"_balanced_ratio"] = float(np.mean(gm[1:]/ms[1:]))
            row[name+"_reference_position_correlation"] = correlation(r[:, :-h][mask], r[:, h:][mask]).tolist()
        row["generated_position_correlation"] = correlation(
            g[:, :, :-h].reshape(-1, g.shape[-1]).astype(float),
            g[:, :, h:].reshape(-1, g.shape[-1]).astype(float)).tolist()
        rows.append(row)
    return rows


def drift_population(model, current, previous, following, rng, jac_samples):
    mu = model.state_mean.detach().numpy().astype(float)
    s = model.state_std.detach().numpy().astype(float)
    vm = model.target_mean.detach().numpy().astype(float)
    vs = model.target_std.detach().numpy().astype(float)
    q, v = (current-mu)/s, (current-previous-vm)/vs
    observed = (following-2*current+previous)/vs
    prediction, variance = evaluate_drift(model, current, previous)
    predicted = prediction/vs
    coefficients, reports = {}, {}
    for name, response in (("observed", observed), ("learned", predicted)):
        coef, details = linear_projection(q, v, response)
        coefficients[name] = coef
        d = len(s)
        details.update(effective_restoring=(-np.diag(coef[1:1+d])*vs/s).tolist(),
                       effective_damping=(-np.diag(coef[1+d:])).tolist(),
                       intercept=coef[0].tolist())
        reports[name] = details
    residual = observed-predicted
    reports.update(samples=len(current),
        conditional_mean_residual_rms=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        predicted_noise_variance=np.mean(variance/vs**2, axis=0).tolist(),
        innovation_mse=np.mean(residual**2, axis=0).tolist(),
        noise_to_innovation_ratio=(np.mean(variance/vs**2, axis=0)/np.mean(residual**2, axis=0)).tolist(),
        position_rms_in_training_std=np.sqrt(np.mean(q*q, axis=0)).tolist(),
        velocity_rms_in_training_std=np.sqrt(np.mean(v*v, axis=0)).tolist(),
        observed_increment_correlation=correlation(current-previous, following-current).tolist(),
        jacobian=jacobian_summary(model, current, previous, rng, jac_samples))
    reports['structured_components'] = structured_components(model, current, previous)
    # Identical two-dimensional position/velocity bins for measured and model
    # acceleration. Other coordinates retain their actual conditional mixture.
    edges = np.array([-np.inf, -2, -1, -.5, .5, 1, 2, np.inf])
    maps = {}
    for mode in (0, 1, 2, 3, 10, 20):
        if mode >= len(s):
            continue
        ix, iv = np.digitize(q[:, mode], edges)-1, np.digitize(v[:, mode], edges)-1
        cells = ix*7+iv
        count = np.bincount(cells, minlength=49).reshape(7, 7)
        entry = dict(count=count)
        for name, a in (("observed", observed), ("learned", predicted)):
            total = np.bincount(cells, weights=a[:, mode], minlength=49).reshape(7, 7)
            entry[name] = np.divide(total, count, out=np.full((7, 7), np.nan), where=count>=50)
        maps[mode] = entry
    return reports, coefficients, maps


def plot_results(out, rows, populations, maps):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    t = [row["milliseconds"] for row in rows]
    for key, label in (("full_window", "All intervening stencils retained"),
                       ("endpoint", "Retained endpoints only (sensitivity)")):
        axes[0].semilogx(t, [row[key+"_pod_ratio"] for row in rows], 'o-', label=label)
    axes[0].axhline(1, color='gray', lw=1)
    axes[0].set(xlabel="Horizon (ms)", ylabel="POD displacement mean-square ratio")
    axes[0].legend(fontsize=7)
    axes[1].semilogx(t, [100*row["full_window_fraction"] for row in rows], 'o-')
    axes[1].set(xlabel="Horizon (ms)", ylabel="Fully retained reference windows (%)")
    for mode in (1, 2):
        axes[2].semilogx(t, [row["generated_position_correlation"][mode] for row in rows], label=f'Generated POD {mode}')
        axes[2].semilogx(t, [row.get("endpoint_reference_position_correlation", [np.nan]*21)[mode] for row in rows], '--', label=f'Retained endpoints POD {mode}')
    axes[2].set(xlabel="Horizon (ms)", ylabel="Position pair correlation")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'displacement.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, color in (("reference", "black"), ("generated", "tab:orange")):
        for kind, style in (("observed", '--'), ("learned", '-')):
            for ax, quantity in zip(axes, ("effective_restoring", "effective_damping")):
                values = populations[name][kind][quantity]
                ax.plot(range(1, len(values)), values[1:], style, color=color, label=f'{name}: {kind}')
    for ax, label in zip(axes, ('Effective restoring coefficient / frame²', 'Effective damping coefficient / frame')):
        ax.set(xlabel='POD mode', ylabel=label); ax.axhline(0, color='gray', lw=.7); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Full multivariate linear projections; descriptive, not physical identification')
    fig.tight_layout(); fig.savefig(out/'drift_coefficients.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for row, mode in enumerate((1, 2, 3)):
        entry = maps['reference'][mode]
        arrays = [entry['observed'], entry['learned'], entry['learned']-entry['observed']]
        limit = max(np.nanmax(np.abs(a)) for a in arrays)
        for col, a in enumerate(arrays):
            im = axes[row, col].imshow(a, origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit)
            axes[row, col].set(title=f"POD {mode}: {['measured', 'learned', 'learned − measured'][col]}", xlabel='Incoming increment (training std)', ylabel='Position (training std)', xticks=[0,3,6], xticklabels=['<−2','≈0','>2'], yticks=[0,3,6], yticklabels=['<−2','≈0','>2'])
            fig.colorbar(im, ax=axes[row, col], shrink=.8)
    fig.suptitle('Mean change in increment / training increment std; bins with ≥50 samples')
    fig.tight_layout(); fig.savefig(out/'conditional_drift.png', dpi=160); plt.close(fig)


def plot_structured(out, populations):
    components = populations['reference']['structured_components']
    if components is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    k = np.diag(components['stiffness_matrix'])
    d = np.diag(components['damping_matrix'])
    modes = np.arange(1, len(k))
    axes[0].plot(modes, k[1:], 'o-', label='Stiffness diagonal')
    axes[0].plot(modes, d[1:], 'o-', label='Damping diagonal')
    axes[0].axhline(components['stiffness_initial'], color='gray', ls=':', label='Initialization (both)')
    axes[0].set(xlabel='POD mode', ylabel='Normalized operator diagonal')
    for pop, style in (('reference','-'),('generated','--')):
        c = populations[pop]['structured_components']['rms_per_coordinate']
        for term, color in (('linear','tab:blue'),('residual','tab:orange'),('total','black')):
            axes[1].plot(modes, np.array(c[term])[1:], style, color=color, label=f'{pop}: {term}')
        coef = np.array(populations[pop]['structured_components']['residual_linear_projection'])
        ndim = len(k)
        effective_k = k-np.diag(coef[1:1+ndim])
        effective_d = d-np.diag(coef[1+ndim:])
        axes[2].plot(modes, (effective_k/k)[1:], style, color='tab:blue', label=f'{pop}: restoring')
        axes[2].plot(modes, (effective_d/d)[1:], style, color='tab:orange', label=f'{pop}: damping')
    axes[1].set(xlabel='POD mode', ylabel='Drift RMS / training increment std')
    axes[2].set(xlabel='POD mode', ylabel='Total effective coefficient / explicit matrix')
    axes[2].axhline(1,color='gray',ls=':'); axes[2].set_yscale('symlog', linthresh=1)
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Explicit matrices and neural residual; total coefficients use joint linear projections')
    fig.tight_layout(); fig.savefig(out/'structured_decomposition.png',dpi=160); plt.close(fig)


def analyze(args):
    torch.set_num_threads(4)
    rollout = Rollout(args.rollout)
    model, checkpoint = load_model(Path(rollout.metadata['checkpoint']), 'cpu')
    if (model.config.state_variable != 'velocity' or model.config.lag_steps != 1
            or model.config.history_steps != 1 or rollout.lag_steps != 1
            or not rollout.include_spatial_mean):
        raise ValueError('This audit requires lag-1 velocity, one history frame, and spatial mean coordinate.')
    spec = training_trim_specification(rollout)
    if spec is None:
        raise ValueError('Saved training trim is required.')
    if not np.isclose(checkpoint['training']['train_trim_norm_cutoff'], spec['cutoff']):
        raise ValueError('Checkpoint and trim specification disagree.')
    out = output_directory(rollout, 'temporal_drift', args.output)
    r, g = rollout.reference, rollout.generated
    keep, trim_report = reference_training_trim_mask(r, spec)
    lags = sorted({h for h in args.lags if 1 <= h < r.shape[1]})
    print('Computing displacement curves...', flush=True)
    rows = displacement_analysis(g, r, keep, lags, rollout.fs)
    # Deduplicate overlapping measured windows for the conditional drift audit.
    condition, time = np.where(keep)
    if rollout.start_index is None:
        raise ValueError('start_index required to deduplicate reference states.')
    keys = np.array([f'{rollout.repetitions[c]}:{rollout.start_index[c]+t}' for c,t in zip(condition,time)])
    _, unique = np.unique(keys, return_index=True)
    condition, time = condition[unique], time[unique]
    rng = np.random.default_rng(args.seed)
    populations, coefficients, maps = {}, {}, {}
    print(f'Evaluating {len(time)} distinct retained measured states...', flush=True)
    populations['reference'], coefficients['reference'], maps['reference'] = drift_population(
        model, r[condition,time].astype(float), r[condition,time-1].astype(float),
        r[condition,time+1].astype(float), rng, args.jacobian_samples)
    # Per-repetition projections are a sensitivity check, not independent-window CIs.
    rep_reports = {}
    for rep in np.unique(rollout.repetitions):
        selected = rollout.repetitions[condition] == rep
        c, t = condition[selected], time[selected]
        rep_reports[str(rep)], _, _ = drift_population(
            model, r[c,t].astype(float), r[c,t-1].astype(float), r[c,t+1].astype(float), rng, min(256,args.jacobian_samples))
    paths = g.reshape(-1, g.shape[2], g.shape[3])
    flat = rng.choice(len(paths)*(paths.shape[1]-2), min(args.generated_samples, len(paths)*(paths.shape[1]-2)), replace=False)
    path, time = flat//(paths.shape[1]-2), flat%(paths.shape[1]-2)+1
    print(f'Evaluating {len(time)} generated states...', flush=True)
    populations['generated'], coefficients['generated'], maps['generated'] = drift_population(
        model, paths[path,time].astype(float), paths[path,time-1].astype(float),
        paths[path,time+1].astype(float), rng, args.jacobian_samples)
    intervals = []
    for mask in keep:
        edges = np.diff(np.r_[False, mask, False].astype(int))
        intervals.extend((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).tolist())
    report = dict(provenance=rollout.provenance(), checkpoint=rollout.metadata['checkpoint'],
        checkpoint_sha256=hashlib.sha256(Path(rollout.metadata['checkpoint']).read_bytes()).hexdigest(),
        checkpoint_epoch=checkpoint.get('epoch'), model_config=model.config.to_dict(),
        seed=args.seed, trim=trim_report, displacement=rows, drift=populations,
        reference_repetition_sensitivity=rep_reports,
        retained_run_lengths=dict(maximum=max(intervals), median=float(np.median(intervals))),
        conventions=dict(velocity='coordinate change per native frame, not per second',
            drift='E[v_next-v_current | current position, incoming increment]',
            regression='all 21 standardized positions and 21 increments jointly, with intercept',
            coefficients='diagonal of full linear projection, converted to native-frame units; positive denotes restoring/damping',
            generated='untrimmed; same checkpoint diffusion_scale as saved rollout',
            long_horizon='full-window selection becomes more restrictive with horizon; endpoint-only is a sensitivity control and can cross excluded increments',
            uncertainty='two validation repetitions; overlapping windows; descriptive results, no independent-sample confidence intervals'),
        position_within_path_variance_ratio=float(
            np.var(g[...,1:].astype(float),axis=2).mean((0,1)).sum()/np.mean([
                np.var(rr[kk,1:].astype(float),axis=0) for rr,kk in zip(r,keep)],axis=0).sum()))
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    arrays = {f'{pop}_{name}_linear_coefficients':coef for pop, entries in coefficients.items() for name, coef in entries.items()}
    arrays.update({f'{pop}_mode{mode}_{name}':a for pop, entries in maps.items() for mode, values in entries.items() for name,a in values.items()})
    np.savez_compressed(out/'drift_arrays.npz', **arrays)
    plot_results(out, rows, populations, maps)
    plot_structured(out, populations)
    print(out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--generated-samples', type=int, default=100000)
    parser.add_argument('--jacobian-samples', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lags', type=int, nargs='+', default=[1,2,5,10,20,50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000])
    args = parser.parse_args()
    if args.generated_samples < 2 or args.jacobian_samples < 1:
        parser.error('positive sample counts required (generated >= 2)')
    analyze(args)


if __name__ == '__main__':
    main()
