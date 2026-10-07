"""Compare repetition-specific rank-k POD fields in original spatial coordinates.

No instantaneous spatial mean is restored. Low-rank factors provide exact
covariance distances for the stored rank-k fields without huge spatial matrices.
"""
import argparse
import csv
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
from scipy.spatial.distance import pdist, squareform
from data_analysis.bispectrum.compare_bispectral_repetitions import discover, group_summary, plot_distances


def spatial_covariance_distances(factors):
    """C_i=F_i.T @ F_i; return spatial-entry RMS distances and inner products."""
    n, rank, pixels = factors.shape
    flat = factors.reshape(n*rank, pixels)
    gram = flat @ flat.T
    inner = np.sum(gram.reshape(n, rank, n, rank)**2, axis=(1, 3)) / pixels**2
    squared = inner.diagonal()[:, None] + inner.diagonal()[None, :] - 2*inner
    distance = np.sqrt(np.maximum(squared, 0))
    np.fill_diagonal(distance, 0)
    return distance, inner


def held_out_from_inner(inner, labels):
    """Distances to averages of the other repetitions, from a Gram matrix."""
    labels = np.asarray(labels)
    errors, sizes = np.empty(len(labels)), np.empty(len(labels))
    for i, power in enumerate(labels):
        others = np.flatnonzero((labels == power) & (np.arange(len(labels)) != i))
        if len(others) == 0:
            raise ValueError('Need at least two repetitions per power.')
        ref_squared = float(inner[np.ix_(others, others)].mean())
        errors[i] = np.sqrt(max(float(inner[i, i] + ref_squared - 2*inner[i, others].mean()), 0))
        sizes[i] = np.sqrt(max(ref_squared, 0))
    relative = np.divide(errors, sizes, out=np.full_like(errors, np.nan), where=sizes > 0)
    return errors, relative


def write_csv(path, header, rows):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_statistics(cases, rank, cache_path):
    """Cache rank-k sufficient statistics; never restore instantaneous spatial mean."""
    means, restored, factors, diagnostics, manifest = [], [], [], [], []
    reference_grid = None
    with h5py.File(cache_path, 'a') as cache:
        for i, c in enumerate(cases):
            source = c['pod_file']
            stat = source.stat()
            key = f"{c['power']}_rep{c['rep']}"
            fingerprint = dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns, rank=rank, schema=1)
            encoded = json.dumps(fingerprint, sort_keys=True)
            with h5py.File(source, 'r') as f:
                if not all(f.attrs[name] for name in ('instantaneous_spatial_mean_removed', 'temporal_mean_field_removed')):
                    raise ValueError(f'Unexpected POD preprocessing: {source}')
                if rank > f['pod/modes'].shape[0]:
                    raise ValueError('Requested rank exceeds stored rank.')
                x, y, t = [f[f'grid/{v}'][:] for v in ('x', 'y', 'time')]
                units = str(f['reduced/coefficients'].attrs['units'])
                grid = (x, y, t-t[0], units)
                if reference_grid is None:
                    reference_grid = grid
                if units != reference_grid[3] or any(a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-8)
                                                    for a, b in zip(grid[:3], reference_grid[:3])):
                    raise ValueError(f'Different grid, units or time interval: {source}')
                # Stored coordinates retain float32 rounding despite float64 storage.
                if not all(np.allclose(np.diff(v), np.diff(v)[0], rtol=1e-4, atol=1e-8) for v in (x, y)):
                    raise ValueError('Expected uniform spatial pixel grid.')
                if key not in cache or cache[key].attrs.get('fingerprint') != encoded:
                    modes = np.asarray(f['pod/modes'][:rank], dtype=float).reshape(rank, -1)
                    coefficients = np.asarray(f['reduced/coefficients'][:, :rank], dtype=float)
                    if not np.isfinite(coefficients).all() or not np.isfinite(modes).all():
                        raise ValueError(f'Nonfinite POD data: {source}')
                    mean_a = coefficients.mean(axis=0)
                    coefficients -= mean_a
                    cov_a = coefficients.T @ coefficients / len(coefficients)
                    eig, vectors = np.linalg.eigh(cov_a)
                    factor = (np.sqrt(np.maximum(eig, 0))[:, None] * vectors.T) @ modes
                    mean_field = mean_a @ modes
                    physical_mean = mean_field + np.asarray(f['preprocessing/temporal_mean_field'], dtype=float).ravel()
                    if key in cache:
                        del cache[key]
                    g = cache.create_group(key)
                    for name, value in dict(mean_field=mean_field, physical_mean=physical_mean,
                                            covariance_factor=factor, coefficient_mean=mean_a,
                                            coefficient_covariance=cov_a).items():
                        g.create_dataset(name, data=value, compression='gzip')
                    g.attrs.update(fluctuation_rms=float(np.sqrt(np.sum(factor**2)/modes.shape[1])),
                                   retained_energy_fraction=float(f['pod/cumulative_energy_fraction'][rank-1]),
                                   sample_count=len(t), units=units)
                    g.attrs['fingerprint'] = encoded
                    cache.flush()
                    action = 'computed'
                else:
                    action = 'cached'
                g = cache[key]
                means.append(g['mean_field'][:])
                restored.append(g['physical_mean'][:])
                factors.append(g['covariance_factor'][:])
                diagnostics.append([c['power'], c['rep'], g.attrs['fluctuation_rms'],
                                    np.sqrt(np.mean(means[-1]**2)), g.attrs['retained_energy_fraction']])
                manifest.append(dict(power=c['power'], repetition=c['rep'], **fingerprint))
            if (i+1) % 8 == 0 or i == 0:
                print(f'[{i+1}/{len(cases)}] {action} {key}', flush=True)
    return np.array(means), np.array(restored), np.array(factors), diagnostics, manifest, units


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cases = discover(args.data_root, args.powers, None)
    means, restored, factors, diagnostics, manifest, units = load_statistics(cases, args.rank, out/'statistics.h5')
    labels = [c['power'] for c in cases]
    names = [f"{c['power']}_rep{c['rep']}" for c in cases]
    pixels = means.shape[1]
    print('Computing physical-space covariance distances from low-rank factors...', flush=True)
    covariance_distance, covariance_inner = spatial_covariance_distances(factors)
    datasets = []
    for key, title, values in [('dynamic_mean', f'Mean of centered rank-{args.rank} dynamics', means),
                               ('restored_temporal_mean', 'Mean field with temporal baseline restored', restored)]:
        distance = squareform(pdist(values / np.sqrt(pixels)))
        datasets.append((key, title, units, distance, values @ values.T / pixels))
    datasets.append(('covariance', f'Rank-{args.rank} spatial covariance', units+'²', covariance_distance, covariance_inner))
    summaries, rows, pairs, overview = {}, [], [], []
    with PdfPages(out/'mean_covariance_comparison.pdf') as pdf:
        compact, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        for key, title, unit, distance, inner in datasets:
            s, block = group_summary(distance, labels)
            errors, relative = held_out_from_inner(inner, labels)
            powers = list(s['within'])
            for p in powers:
                ix = np.array(labels) == p
                rel = relative[ix]
                entry = dict(held_out_median=float(np.median(errors[ix])), held_out_p95=float(np.quantile(errors[ix], .95)),
                             relative_held_out_median=float(np.median(rel)) if np.isfinite(rel).all() else None)
                s['within'][p].update(entry)
                rows.append([key, p, s['within'][p]['mean'], *entry.values()])
            for entry in s['between']:
                pairs.append([key, entry['power_a'], entry['power_b'], entry['mean'], entry['between_to_within_ratio']])
            ratios = [r['between_to_within_ratio'] for r in s['between']]
            adjacent = [r['between_to_within_ratio'] for r in s['between']
                        if powers.index(r['power_b'])-powers.index(r['power_a']) == 1]
            overview.append([key, float(np.median(ratios)), float(np.median(adjacent)), float(np.nanmedian(relative))])
            summaries[key] = s
            write_csv(out/f'{key}_distances.csv', ['repetition', *names], ([n, *r] for n,r in zip(names, distance)))
            write_csv(out/f'{key}_held_out.csv', ['repetition', 'distance', 'relative_distance'], zip(names, errors, relative))
            fig = plot_distances(distance, cases, block, title+f'; spatial mean removed\nPhysical-space RMS distance [{unit}]')
            fig.savefig(out/f'{key}_comparison.png', dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            if key in ('dynamic_mean', 'covariance'):
                ax = axes[0 if key == 'dynamic_mean' else 1]
                if key == 'dynamic_mean':
                    displayed = 100 * errors / np.array([r[2] for r in diagnostics])
                    ylabel = 'Mean error / fluctuation RMS [%]'
                    title += '\nNear zero by temporal-mean subtraction'
                else:
                    displayed = 100 * relative
                    ylabel = 'Covariance error / held-out reference norm [%]'
                ax.boxplot([displayed[np.array(labels) == p] for p in powers], tick_labels=powers)
                ax.tick_params(axis='x', rotation=45)
                ax.set(title=title, ylabel=ylabel)
            print(key, 'median between/within', round(overview[-1][1], 3),
                  'adjacent', round(overview[-1][2], 3), flush=True)
        compact.suptitle(f'Rank-{args.rank} experimental reference variability; spatial mean removed')
        compact.savefig(out/'reference_variability.png', dpi=180)
        pdf.savefig(compact)
        plt.close(compact)
    write_csv(out/'within_power.csv', ['metric', 'power', 'mean_pair_distance', 'held_out_median', 'held_out_p95', 'relative_held_out_median'], rows)
    write_csv(out/'between_power.csv', ['metric', 'power_a', 'power_b', 'mean_distance', 'between_within_ratio'], pairs)
    write_csv(out/'overview.csv', ['metric', 'median_pair_ratio', 'median_adjacent_pair_ratio', 'median_relative_held_out_error'], overview)
    write_csv(out/'reconstruction_diagnostics.csv', ['power', 'repetition', 'fluctuation_rms', 'dynamic_mean_rms', 'retained_energy_fraction'], diagnostics)
    (out/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    lines = ['# Rank-k reconstructed mean and covariance repeatability', '',
        f'Rank: {args.rank}; {len(cases)} repetitions across {len(set(labels))} powers; {pixels} spatial pixels. Full saved recordings, identical grids/units/durations checked.',
        'Each repetition uses its OWN POD basis. All comparisons take place in original spatial coordinates. No instantaneous spatial mean is restored.',
        'Primary dynamics: z_r(t)=U_r a_r(t); mean: U_r mean_t(a_r). POD preprocessing removed temporal means; this statistic is near zero by construction. Ratios of these tiny residual mean differences should not be interpreted as physical discrimination.',
        'A separate sensitivity output restores the saved temporal mean field. It still excludes instantaneous spatial mean. That baseline is measured preprocessing information, not a mean generated by centered ROM dynamics.',
        'Covariance: C_r=U_r S_r U_r.T, where S_r=mean_t[(a_r-mean(a_r))(a_r-mean(a_r)).T]. Population normalization 1/N. This is exact for the stored rank-k reconstruction, and remains rank-truncated relative to the experiment.',
        'Write C_r=F_r.T F_r. Inner product <C_A,C_B>=||F_A F_B.T||_F². Distance is sqrt(||C_A||_F²+||C_B||_F²-2<C_A,C_B>)/number_of_pixels. No spatial covariance matrix is allocated; no shared-basis approximation is used.',
        'Mean distances are spatial RMS; covariance distances are RMS across all spatial matrix entries. No amplitude normalization is applied to pair distances. Rank-k retained energy fractions are in reconstruction_diagnostics.csv.',
        'Within-power means exclude self pairs; between-power means include every cross-power repetition pair. Between/within divides by the average of the two within-power means. Adjacent-power summaries avoid relying only on widely separated powers.',
        'Held-out error compares each repetition with the average statistic of all OTHER repetitions at that power. Relative error divides by the norm of that held-out average. Average individual covariances, not the covariance of an average trajectory. Near-zero dynamic-mean relative errors are not meaningful.',
        'For ROM validation, use identical preprocessing, units, rank definition and duration; reconstruct the ROM into the same spatial grid and compare its mean/covariance to the experimental average. statistics.h5 retains spatial covariance factors and mean fields for those references. No ROM trajectory was analyzed.',
        'Pair distances are dependent. These describe empirical acquisition variability, not confidence intervals or independent pair samples. No significance claims or acquisition-block corrections. Poor discrimination alone does not establish poor repeatability.', '',
        '| Statistic | Median between/within | Adjacent powers | Median relative held-out error |',
        '|---|---:|---:|---:|']
    lines += [f'| {k} | {r:.3f} | {a:.3f} | {e:.3f} |' for k,r,a,e in overview]
    max_mean_ratio = max(r[3]/r[2] for r in diagnostics)
    lines += ['', '## Observations', '',
              f'The largest reconstructed dynamic-mean RMS / fluctuation RMS is {max_mean_ratio:.3g}. The mean is a centering check, not an independently measured discriminator of the centered dynamics.',
              'Covariance repeatability depends on power. Median held-out relative discrepancies:', '',
              '| Power | Covariance discrepancy |', '|---|---:|']
    for power, row in summaries['covariance']['within'].items():
        lines.append(f"| {power} | {100*row['relative_held_out_median']:.1f}% |")
    lines += ['', 'These results do not support declaring covariance universally unsuitable. Use the power-specific repetition variability as the ROM reference; separation is weaker for some nearby powers. Neither a near-zero mean nor matched static covariance guarantees matched temporal or higher-order dynamics.',
              'The compact figure normalizes dynamic-mean errors by each repetition\'s fluctuation RMS and covariance errors by the held-out average covariance norm. All pairwise distance figures retain physical RMS units.']
    lines += ['', 'Open mean_covariance_comparison.pdf for all plots, or reference_variability.png for the compact view. CSVs retain per-power and per-pair results.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent / 'reduced_data'
    p.add_argument('--data-root', type=Path, default=root)
    p.add_argument('--powers', nargs='+', default=['all'])
    p.add_argument('--rank', type=int, default=10)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if args.rank < 1:
        p.error('Rank must be positive.')
    if args.output_dir is None:
        args.output_dir = args.data_root/f'mean_covariance_repetition_comparisons/rank_{args.rank}'
    run(args)
