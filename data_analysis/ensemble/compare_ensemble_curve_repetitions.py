"""Subset stability and power discrimination of time-dependent ensemble moments.

Rank-k fields use each repetition's own POD basis; spatial and temporal mean
baselines remain removed. Comparisons are exact in space at sampled common times.
All 12-of-16 subsets are evaluated, plus seeded disjoint 8-vs-8 controls.
"""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from data_analysis.bispectrum.compare_bispectral_repetitions import discover


def weights(subsets, n):
    """Mean and unbiased sample covariance weights on a repetition ensemble."""
    mask = np.zeros((len(subsets), n))
    for row, subset in zip(mask, subsets):
        row[list(subset)] = 1
    count = mask.sum(axis=1)
    if np.any(count < 2):
        raise ValueError('Need at least two ensemble members.')
    w = mask/count[:, None]
    h = (np.eye(n)[None] * mask[:, :, None] -
         mask[:, :, None]*mask[:, None, :]/count[:, None, None])/(count[:, None, None]-1)
    return w, h.reshape(len(subsets), n*n)


def snapshot_gram(a, b, overlap, pixels):
    """Spatial dot products z_i(t).z_j(t)/pixels; a,b have shape time,rep,rank."""
    nt, na, rank = a.shape
    nb = b.shape[1]
    result = np.empty((nt, na, nb))
    for i in range(na):
        projected = (a[:, i] @ overlap[i].reshape(rank, nb*rank)).reshape(nt, nb, rank)
        result[:, i] = np.sum(projected*b, axis=2)/pixels
    return result


def moment_kernels(gram):
    """Integrated inner products for ensemble means and spatial covariances."""
    nt, na, nb = gram.shape
    flat = gram.reshape(nt, -1)
    fourth = (flat.T @ flat / nt).reshape(na, nb, na, nb).transpose(0, 2, 1, 3)
    return gram.mean(axis=0), fourth.reshape(na*na, nb*nb)


def inner_rows(left, kernel, right):
    return np.sum((left @ kernel)*right, axis=1)


def norms(descriptors, kernels):
    return [inner_rows(d, k, d) for d,k in zip(descriptors, kernels)]


def distances(left, right, aa, bb, ab):
    return [np.sqrt(np.maximum(inner_rows(l, ka, l)+inner_rows(r, kb, r)-2*inner_rows(l, cross, r), 0))
            for l,r,ka,kb,cross in zip(left, right, aa, bb, ab)]


def curve_sizes(gram, subset):
    """Display norms only; distance scoring uses full spatial statistics."""
    k = gram[:, subset][:, :, subset]
    mean_size = np.sqrt(np.maximum(k.mean(axis=(1, 2)), 0))
    centered = k-k.mean(axis=1, keepdims=True)-k.mean(axis=2, keepdims=True)+k.mean(axis=(1, 2))[:, None, None]
    covariance_size = np.sqrt(np.sum(centered**2, axis=(1, 2)))/(len(subset)-1)
    return mean_size, covariance_size


def write_csv(path, header, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_data(args, cases):
    cache_path = args.output_dir/'sampled_rank_data.h5'
    modes, coefficients, manifest = [], [], []
    grid = indices = None
    rng = np.random.default_rng(args.seed)
    with h5py.File(cache_path, 'a') as cache:
        for i,c in enumerate(cases):
            path = c['pod_file']
            stat = path.stat()
            config = dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                          rank=args.rank, time_samples=args.time_samples, seed=args.seed, schema=1)
            encoded = json.dumps(config, sort_keys=True)
            key = f"{c['power']}_rep{c['rep']}"
            with h5py.File(path, 'r') as f:
                if not all(f.attrs[v] for v in ('instantaneous_spatial_mean_removed', 'temporal_mean_field_removed')):
                    raise ValueError('Expected centered POD dynamics.')
                x,y,t = [f[f'grid/{v}'][:] for v in ('x','y','time')]
                units = str(f['reduced/coefficients'].attrs['units'])
                current = (x,y,t-t[0],units)
                if grid is None:
                    grid = current
                    count = min(args.time_samples, len(t))
                    # One random sample per equal-sized time stratum avoids fixed-stride aliasing.
                    edges = np.linspace(0, len(t), count+1, dtype=int)
                    indices = np.array([rng.integers(a,b) for a,b in zip(edges[:-1], edges[1:])])
                if units != grid[3] or any(a.shape != b.shape or not np.allclose(a,b,rtol=1e-6,atol=1e-8)
                                         for a,b in zip(current[:3],grid[:3])):
                    raise ValueError(f'Mismatched grids or units: {path}')
                if not all(np.allclose(np.diff(v), np.diff(v)[0], rtol=1e-4, atol=1e-8) for v in (x,y)):
                    raise ValueError('Expected uniform pixel grid.')
                if args.rank > f['pod/modes'].shape[0]:
                    raise ValueError('Insufficient stored modes.')
                if key not in cache or cache[key].attrs.get('config') != encoded:
                    u = np.asarray(f['pod/modes'][:args.rank],dtype=float).reshape(args.rank,-1)
                    # On disk, coefficient chunks span every mode; read once then sample.
                    a = np.asarray(f['reduced/coefficients'][:, :args.rank],dtype=float)[indices]
                    if not np.isfinite(a).all() or not np.isfinite(u).all():
                        raise ValueError('Nonfinite POD data.')
                    if key in cache:
                        del cache[key]
                    g = cache.create_group(key)
                    g.create_dataset('modes',data=u,compression='gzip')
                    g.create_dataset('coefficients',data=a,compression='gzip')
                    g.attrs['config'] = encoded
                    cache.flush()
                    action = 'loaded'
                else:
                    action = 'cached'
                modes.append(cache[key]['modes'][:])
                coefficients.append(cache[key]['coefficients'][:])
                manifest.append(dict(power=c['power'],rep=c['rep'],**config))
            if (i+1)%16 == 0 or i == 0:
                print(f'[{i+1}/{len(cases)}] {action} {key}',flush=True)
    return np.array(modes),np.array(coefficients),grid[2][indices],indices,manifest,units


def run(args):
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    cases = discover(args.data_root,args.powers,None)
    powers = list(dict.fromkeys(c['power'] for c in cases))
    groups = [np.array([i for i,c in enumerate(cases) if c['power']==p]) for p in powers]
    if any(len(g)!=16 for g in groups):
        raise ValueError('This analysis requires 16 repetitions per power.')
    u,a,time,indices,manifest,units = load_data(args,cases)
    pixels = u.shape[-1]
    print('Computing overlaps between all repetition-specific spatial bases...',flush=True)
    flat = u.reshape(len(cases)*args.rank,pixels)
    overlap = (flat @ flat.T).reshape(len(cases),args.rank,len(cases),args.rank)
    del flat,u
    rng = np.random.default_rng(args.seed+1)
    subsets = list(itertools.combinations(range(16),12))
    twelve = weights(subsets,16)
    full = weights([tuple(range(16))],16)
    # Fix member 0 in the first half to enumerate each complementary partition once.
    partitions = [(0,*s) for s in itertools.combinations(range(1,16),7)]
    selected = rng.choice(len(partitions),min(args.splits,len(partitions)),replace=False)
    halves = [partitions[i] for i in selected]
    complements = [tuple(i for i in range(16) if i not in h) for h in halves]
    left,right = weights(halves,16),weights(complements,16)
    plot_ids = rng.choice(len(subsets),64,replace=False)
    # One member from each neighboring time-stratum pair gives two interleaved coverage checks.
    parity = rng.integers(0,2,size=len(time)//2)
    time_a = 2*np.arange(len(parity))+parity
    time_b = 2*np.arange(len(parity))+(1-parity)
    kernels, grams, within, full_norms = [],[],[],[]
    stability_rows, sample_rows, sensitivity_rows, summary = [],[],[],{}
    within_split_rows, between_split_rows = [],[]
    metrics = ('mean','covariance')
    with PdfPages(out/'ensemble_curve_comparison.pdf') as pdf:
        for p,ix in zip(powers,groups):
            coeff = a[ix].transpose(1,0,2)
            gram = snapshot_gram(coeff,coeff,overlap[ix][:,:,ix,:],pixels)
            kk = moment_kernels(gram)
            kernels.append(kk)
            grams.append(gram)
            fn = norms(full,kk)
            full_norms.append(fn)
            deviations = [np.sqrt(np.maximum(inner_rows(d-f,k,d-f),0)) for d,f,k in zip(twelve,full,kk)]
            control = distances(left,right,kk,kk,kk)
            within.append(control)
            sensitivity = [moment_kernels(gram[idx]) for idx in (time_a,time_b)]
            summary[p] = {}
            for m,key in enumerate(metrics):
                scale = float(np.sqrt(max(fn[m][0],0)))
                relative = deviations[m]/scale if scale>0 else np.full(len(subsets),np.nan)
                med,p05,p95 = np.quantile(relative,[.5,.05,.95])
                summary[p][key] = dict(relative_subset_error_median=float(med),relative_subset_error_p05=float(p05),
                                       relative_subset_error_p95=float(p95),full_ensemble_norm=scale,
                                       disjoint_half_mean_distance=float(control[m].mean()))
                if key == 'mean':
                    state_rms = float(np.sqrt(np.trace(kk[0])/16))
                    summary[p][key].update(state_rms=state_rms, full_mean_over_state_rms=scale/state_rms,
                                           median_subset_error_over_state_rms=float(np.median(deviations[m]))/state_rms)
                stability_rows.append([p,key,scale,med,p05,p95,float(control[m].mean())])
                for subset,error,rel in zip(subsets,deviations[m],relative):
                    sample_rows.append([p,key,' '.join(str(cases[ix[i]]['rep']) for i in subset),error,rel])
                checks=[]
                for kcheck in sensitivity:
                    d=twelve[m]-full[m]
                    norm=float(np.sqrt(max(inner_rows(full[m],kcheck[m],full[m])[0],0)))
                    checks.append(float(np.median(np.sqrt(np.maximum(inner_rows(d,kcheck[m],d),0))/norm)))
                sensitivity_rows.append([p,key,*checks])
                for j,value in enumerate(control[m]):
                    within_split_rows.append([p,key,j,value])
            fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
            full_curves=curve_sizes(gram,list(range(16)))
            samples=np.array([curve_sizes(gram,list(subsets[j])) for j in plot_ids])
            for m,ax in enumerate(axes):
                lo,hi=np.quantile(samples[:,m], [.05,.95],axis=0)
                ax.fill_between(time,lo,hi,alpha=.25,label='5–95% across 64 displayed 12-repetition subsets')
                ax.plot(time,full_curves[m],lw=.6,label='All 16 repetitions')
                ax.set(xlabel='Time from common onset [s]',ylabel=f"Spatial RMS [{units if m==0 else units+'²'}]",
                       title='Ensemble mean-field norm' if m==0 else 'Ensemble covariance norm')
                ax.legend(fontsize=8)
            fig.suptitle(f'{p}: rank {args.rank}; both mean baselines removed\nNorm curves illustrate size; distances compare complete spatial statistics')
            fig.savefig(out/f'{p}_ensemble_curves.png',dpi=140)
            pdf.savefig(fig)
            plt.close(fig)
            np.savez_compressed(out/f'{p}_curve_sizes.npz',time=time,full_mean_rms=full_curves[0],
                                full_covariance_rms=full_curves[1],subset_norm_curves=samples,subset_indices=plot_ids)
            print(f"{p}: 12/16 median relative errors: mean {100*summary[p]['mean']['relative_subset_error_median']:.1f}%, covariance {100*summary[p]['covariance']['relative_subset_error_median']:.1f}%",flush=True)
        ratios=np.ones((2,len(powers),len(powers)))
        pair_rows=[]
        discrimination_checks=[]
        for i,j in itertools.combinations(range(len(powers)),2):
            ia,ib=groups[i],groups[j]
            gram=snapshot_gram(a[ia].transpose(1,0,2),a[ib].transpose(1,0,2),overlap[ia][:,:,ib,:],pixels)
            ab=moment_kernels(gram)
            # Use independently randomized balanced eight-member selections across powers.
            order=rng.permutation(len(halves))
            orientation_a=rng.integers(0,2,size=len(halves))[:,None]
            orientation_b=rng.integers(0,2,size=len(halves))[:,None]
            cross_left=tuple(np.where(orientation_a==0,l,r) for l,r in zip(left,right))
            cross_right=tuple(np.where(orientation_b==0,
                                       l[order],r[order]) for l,r in zip(left,right))
            cross=distances(cross_left,cross_right,kernels[i],kernels[j],ab)
            full_distance=distances(full,full,kernels[i],kernels[j],ab)
            for m,key in enumerate(metrics):
                baseline=(within[i][m].mean()+within[j][m].mean())/2
                ratio=float(cross[m].mean()/baseline)
                ratios[m,i,j]=ratios[m,j,i]=ratio
                pair_rows.append([powers[i],powers[j],key,float(full_distance[m][0]),float(cross[m].mean()),float(baseline),ratio])
                between_split_rows.extend([powers[i],powers[j],key,b,float(v)] for b,v in enumerate(cross[m]))
            for idx in (time_a,time_b):
                kaa,kbb,kab=moment_kernels(grams[i][idx]),moment_kernels(grams[j][idx]),moment_kernels(gram[idx])
                cd=distances(cross_left,cross_right,kaa,kbb,kab)
                wa,wb=distances(left,right,kaa,kaa,kaa),distances(left,right,kbb,kbb,kbb)
                discrimination_checks.append([powers[i],powers[j], 'A' if idx is time_a else 'B',
                                               *[float(cd[m].mean()/((wa[m].mean()+wb[m].mean())/2)) for m in range(2)]])
            print(f"Compared {powers[i]} vs {powers[j]}: mean ratio {ratios[0,i,j]:.2f}; covariance {ratios[1,i,j]:.2f}",flush=True)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for m,key in enumerate(metrics):
            med=np.array([summary[p][key]['relative_subset_error_median'] for p in powers])*100
            lo=np.array([summary[p][key]['relative_subset_error_p05'] for p in powers])*100
            hi=np.array([summary[p][key]['relative_subset_error_p95'] for p in powers])*100
            axes[m].errorbar(np.arange(len(powers)),med,yerr=[med-lo,hi-med],fmt='o',capsize=4)
            axes[m].set(xticks=np.arange(len(powers)),xticklabels=powers,ylabel='Difference from full 16-repetition ensemble [%]',
                        title=f'Ensemble {key}: median and 5–95% subset range')
            axes[m].tick_params(axis='x',rotation=45)
        fig.suptitle('Stability across all 1,820 twelve-repetition subsets\nFull spatial statistics compared at matched times; full ensemble is an overlapping reference')
        fig.savefig(out/'subset_stability.png',dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
        for m,key in enumerate(metrics):
            data=ratios[m].copy()
            np.fill_diagonal(data,np.nan)
            mesh=axes[m].imshow(data,vmin=1,vmax=max(1.01,float(np.nanmax(data))),cmap='viridis')
            axes[m].set(xticks=np.arange(len(powers)),xticklabels=powers,yticks=np.arange(len(powers)),yticklabels=powers,title=f'Ensemble {key}')
            axes[m].tick_params(axis='x',rotation=45)
            for i in range(len(powers)):
                for j in range(len(powers)):
                    if i!=j:
                        axes[m].text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if data[i,j]<(1+np.nanmax(data))/2 else 'black')
            fig.colorbar(mesh,ax=axes[m],label='Between-power / within-power distance')
        fig.suptitle(f'Power discrimination using eight-repetition ensembles\nWithin-power reference: {len(halves)} disjoint 8-vs-8 partitions; equal ensemble sizes')
        fig.savefig(out/'power_discrimination.png',dpi=180)
        pdf.savefig(fig)
        plt.close(fig)
    write_csv(out/'subset_stability.csv',['power','metric','full_norm','relative_median','relative_p05','relative_p95','mean_disjoint_8_distance'],stability_rows)
    write_csv(out/'all_subset_errors.csv',['power','metric','repetitions','absolute_error','relative_error'],sample_rows)
    write_csv(out/'power_pairs.csv',['power_a','power_b','metric','full16_distance','between8_mean','within8_baseline','between_within_ratio'],pair_rows)
    write_csv(out/'time_sampling_stability.csv',['power','metric','relative_median_time_A','relative_median_time_B'],sensitivity_rows)
    write_csv(out/'time_sampling_discrimination.csv',['power_a','power_b','time_subset','mean_ratio','covariance_ratio'],discrimination_checks)
    write_csv(out/'within8_distances.csv',['power','metric','partition','distance'],within_split_rows)
    write_csv(out/'between8_distances.csv',['power_a','power_b','metric','partition','distance'],between_split_rows)
    configuration=dict(rank=args.rank,time_samples=len(time),seed=args.seed,disjoint_partitions=len(halves),
                       time_alignment='Common onset confirmed by user; stored elapsed-time grids match',
                       preprocessing='Stored rank-k POD dynamics; instantaneous spatial mean and temporal mean field not restored',
                       time_indices=indices.tolist(),halves=halves,display_subset_indices=plot_ids.tolist(),sources=manifest)
    (out/'configuration.json').write_text(json.dumps(configuration,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Time-dependent ensemble mean and covariance', '',
           f'{len(cases)} repetitions, {len(powers)} powers, rank {args.rank}; {len(time)} matched elapsed times across the recordings. Common forcing onset confirmed by the user.',
           'At each sampled time, reconstruct each repetition into the original pixel grid: z_r(t)=U_r a_r(t). Both saved temporal and instantaneous spatial mean baselines stay removed. Repetition-specific POD coordinates are never directly averaged.',
           'For subset S with m members: mu_S(t)=sum(z_r(t))/m; C_S(t)=sum[(z_r(t)-mu_S(t))(z_r(t)-mu_S(t)).T]/(m-1). These are ensemble statistics at each time, not temporal averages of individual covariances.',
           'Mean distance: sqrt(mean_t ||mu_A(t)-mu_B(t)||_2² / pixels). Covariance distance: sqrt(mean_t ||C_A(t)-C_B(t)||_F² / pixels²). Covariance uses the unbiased m-1 normalization for every ensemble size.',
           'Spatial comparisons are exact for the stored rank-k reconstructions, via basis overlaps and snapshot Gram matrices. Time averages are estimates from one seeded random snapshot per time stratum; this avoids regular-stride aliasing. This is not a claim of temporal-resolution convergence. Independent interleaved half-sample checks are saved in CSV.',
           'All 1,820 twelve-member subsets are compared with all 16 members. Relative error divides by the full ensemble statistic norm over space and time. Subset ranges describe omission sensitivity; they are NOT confidence intervals. Shared members make this an optimistic test of independent-ensemble reproducibility.',
           f'Power discrimination instead compares eight-member ensembles: within-power distances use {len(halves)} seeded disjoint 8-vs-8 partitions. Between-power distances use independently selected eight-member ensembles. Ratios divide mean between-power distance by the average within-power distance of the two powers. A ratio near one means no extra average separation relative to independent within-power ensembles; above one does not guarantee disjoint distributions.',
           'Repeated subset distances are dependent. No p-values, classifier accuracy or acquisition-block adjustments are claimed. A large power-separation ratio can reflect magnitude differences and does not establish distinct normalized temporal shapes.',
           'Per-power curve plots show spatial norms of mean and covariance, with pointwise 5–95% bands from 64 displayed subsets. Those norm plots are summaries only: distance scores compare full spatial statistics, not just these scalar norm curves.', '',
           '| Power | Mean: median 12/16 error | Covariance: median 12/16 error |', '|---|---:|---:|']
    for p in powers:
        lines.append(f"| {p} | {100*summary[p]['mean']['relative_subset_error_median']:.1f}% | {100*summary[p]['covariance']['relative_subset_error_median']:.1f}% |")
    lines += ['', 'For a ROM, generate an ensemble with matching initial-condition sampling, onset alignment, duration, preprocessing and ensemble size. Compare its moment trajectories to the experimental ensemble with the subset/disjoint variability as context. No ROM data were analyzed.',
              '', 'Open subset_stability.png and power_discrimination.png for the compact results, or ensemble_curve_comparison.pdf for every plot.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Results: {out}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent.parent/'reduced_data'
    p.add_argument('--data-root',type=Path,default=root)
    p.add_argument('--powers',nargs='+',default=['all'])
    p.add_argument('--rank',type=int,default=10)
    p.add_argument('--time-samples',type=int,default=2048)
    p.add_argument('--splits',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    if args.rank<1 or args.time_samples<4 or args.splits<2:
        p.error('Need positive rank, at least four time samples and two partitions.')
    if args.output_dir is None:
        args.output_dir=root/f'ensemble_curve_repetition_comparisons/rank_{args.rank}_t{args.time_samples}'
    run(args)
