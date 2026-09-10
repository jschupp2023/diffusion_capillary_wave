"""Sample capillary excess energy directly from calibrated raw height maps.

Reuse the POD analysis's derivative, quadrature, units and frame estimator.
An optional matching POD cache supplies preprocessing means, never modes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from pod_2d import inspect_input, _frame_key, _preprocess_block
from pod_analysis_report import _check_grid
from pod_capillary_energy import (field_energies, length_scale, select_energy_frames,
                                  quadrature_weights, fingerprint, write_csv, plt)


def compute(args):
    started = time.perf_counter()
    info = inspect_input(args.input.resolve(), 0, None)
    if info.units['time'] not in ('seconds', 'second', 's'):
        raise ValueError('Expected raw time in seconds.')
    indices, weights = select_energy_frames(info.time, args.frames, args.frame_seed)
    x_m, y_m = info.x*length_scale(info.units['x']), info.y*length_scale(info.units['y'])
    z_scale = length_scale(info.units['z'])
    variants = ['raw_as_stored']
    mean = spatial_mean = None
    if args.mean_from_pod:
        with h5py.File(args.mean_from_pod) as f:
            if Path(str(f.attrs.get('source_file', ''))).resolve() != info.path.resolve():
                raise ValueError('Mean-field cache must refer to this exact raw input.')
            if int(f.attrs.get('source_start_position', -1)) != 0 or int(f.attrs.get('source_stop_position_exclusive', -1)) != info.n_frames:
                raise ValueError('Mean-field cache must cover the complete raw recording.')
            _check_grid(info.x, info.y, f['grid/x'][:], f['grid/y'][:], str(args.mean_from_pod))
            if not np.array_equal(f['grid/time'][:], info.time):
                raise ValueError('Mean-field cache has a different time grid.')
            if not f.attrs.get('temporal_mean_field_removed', False):
                raise ValueError('Expected a POD cache with temporal mean removal.')
            ds = f['preprocessing/temporal_mean_field']
            if length_scale(str(ds.attrs['units'])) != z_scale:
                raise ValueError('Raw and mean-field displacement units differ.')
            mean = np.asarray(ds, dtype=float).reshape(-1)
            if mean.size != info.n_space or not np.isfinite(mean).all():
                raise ValueError('Invalid cached mean field.')
            remove_spatial_mean = bool(f.attrs['instantaneous_spatial_mean_removed'])
            spatial_mean = np.asarray(f['preprocessing/frame_spatial_mean'], dtype=float)
            if len(spatial_mean) != info.n_frames or not np.isfinite(spatial_mean).all():
                raise ValueError('Invalid cached frame means.')
        variants.append('raw_with_pod_mean_removal')
    energies = np.empty((len(variants), 2, len(indices)))
    with h5py.File(info.path) as f:
        for start in range(0, len(indices), args.batch_size):
            stop = min(start+args.batch_size, len(indices))
            block = np.stack([np.asarray(f['main'][_frame_key(info.frame_numbers[i])], dtype=np.float32) for i in indices[start:stop]])
            if block.shape[1:] != info.frame_shape or not np.isfinite(block).all():
                raise ValueError('Invalid raw height frame.')
            quad, exact = field_energies(block.astype(float)*z_scale, x_m, y_m)
            energies[0, :, start:stop] = args.gamma*np.stack([quad, exact])
            if mean is not None:
                flat = block.reshape(len(block), -1)
                centered = _preprocess_block(flat, spatial_mean[indices[start:stop]], mean,
                                            remove_spatial_mean, True).reshape(block.shape)
                quad, exact = field_energies(centered.astype(float)*z_scale, x_m, y_m)
                energies[1, :, start:stop] = args.gamma*np.stack([quad, exact])
            if stop % (args.batch_size*8) == 0 or stop == len(indices):
                print(f'Raw energy: {stop}/{len(indices)} selected frames ({time.perf_counter()-started:.1f} s)', flush=True)
    means = energies @ weights
    area = float(quadrature_weights(x_m).sum()*quadrature_weights(y_m).sum())
    rows = [dict(field=variant, Ebar_quad_J=float(means[i, 0]), Ebar_exact_J=float(means[i, 1]),
                 exact_reduction_percent=float(100*(1-means[i, 1]/means[i, 0])) if means[i, 0] else 0.,
                 rms_spatial_slope=float(np.sqrt(2*means[i, 0]/(args.gamma*area))),
                 frames_evaluated=len(indices), full_frame_count=info.n_frames)
            for i, variant in enumerate(variants)]
    settings = dict(source=fingerprint(info.path), mean_source=fingerprint(args.mean_from_pod) if args.mean_from_pod else None,
                    gamma_N_per_m=args.gamma, frame_seed=args.frame_seed, frames_evaluated=len(indices),
                    full_frame_count=info.n_frames, frame_shape=info.frame_shape, raw_units=info.units,
                    duration_seconds=float(info.time[-1]-info.time[0]), area_m2=area,
                    estimation='uniform without replacement; full trapezoidal weights / inclusion probability',
                    preprocessing=variants, elapsed_seconds=time.perf_counter()-started)
    return info, indices, weights, energies, variants, rows, settings


def run(args):
    info, indices, weights, energies, variants, rows, settings = compute(args)
    means = energies @ weights
    area = settings['area_m2']
    x_m, y_m = info.x*length_scale(info.units['x']), info.y*length_scale(info.units['y'])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/'configuration.json').write_text(json.dumps(settings, indent=2)+'\n')
    write_csv(out/'raw_energy_summary.csv', list(rows[0]), [list(r.values()) for r in rows])
    with h5py.File(out/'raw_capillary_energy.h5', 'w') as f:
        f.attrs['configuration'] = json.dumps(settings)
        f.create_dataset('frame_indices', data=indices)
        f.create_dataset('frame_numbers', data=info.frame_numbers[indices])
        f.create_dataset('time_seconds', data=info.time[indices])
        f.create_dataset('mean_weights', data=weights)
        f.create_dataset('x_m', data=x_m)
        f.create_dataset('y_m', data=y_m)
        for i, variant in enumerate(variants):
            g = f.create_group(variant)
            g.create_dataset('E_quad', data=energies[i, 0]).attrs['units'] = 'J'
            g.create_dataset('E_exact', data=energies[i, 1]).attrs['units'] = 'J'
            g.attrs['Ebar_quad_J'], g.attrs['Ebar_exact_J'] = means[i]
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    pos = np.arange(len(variants))
    ax.bar(pos-.18, means[:, 0]*1e9, width=.36, label='Quadratic')
    ax.bar(pos+.18, means[:, 1]*1e9, width=.36, label='Exact geometric')
    ax.set(xticks=pos, xticklabels=[v.replace('_', ' ') for v in variants],
           ylabel='Estimated time-averaged energy (nJ)', title=f'Raw surface energy: {len(indices):,} random frames\n{info.path.name}')
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(out/'raw_energy_report.pdf'); fig.savefig(out/'raw_energy_report.png', dpi=160); plt.close(fig)
    table='\n'.join(f"| {r['field']} | {r['Ebar_quad_J']*1e9:.6f} | {r['Ebar_exact_J']*1e9:.6f} | {r['exact_reduction_percent']:.2f}% | {r['rms_spatial_slope']:.3f} |" for r in rows)
    notes = f'''# Raw experimental capillary-energy estimate

Input: `{info.path}`. Gamma: {args.gamma} N/m. Selected {len(indices):,} of {info.n_frames:,} frames uniformly without replacement, seed {args.frame_seed}.
Duration: {settings['duration_seconds']:.6f} s. Full spatial grid: {info.frame_shape}; area: {area:.8g} m².
Calibrated x, y and height units are converted from microns to metres before differentiating.

| Field | Quadratic mean (nJ) | Exact mean (nJ) | Exact reduction | RMS slope |
|---|---:|---:|---:|---:|
{table}

Raw-as-stored energy includes any static mean surface. The optional mean-removed result uses the full-recording preprocessing means saved in `{args.mean_from_pod}`.
Its source path, spatial grid, full time coverage and units were checked against the raw input.
No POD modes or truncation are used in either calculation. Only the {len(indices):,} selected raw frames are read for energy evaluation; a mean surface is not estimated from this sample.
The mean-removed field follows the original POD preprocessing, including its float32 subtraction. Spatially constant offsets have zero gradient energy.

The observables are gamma/2 integral |grad eta|² dA and gamma integral (sqrt(1+|grad eta|²)-1) dA.
Finite differences and trapezoidal spatial quadrature match pod_capillary_energy.py. The nonlinear integrand is evaluated in its stable form s/(sqrt(1+s)+1).
Time averages are sampling estimates using full time-grid weights divided by frame inclusion probability; no trapezoidal interpolation across random time gaps is performed.
These are capillary deformation energies of measured fields, not total mechanical energy. Spatial differentiation also includes small-scale measurement noise; a raw-versus-POD difference need not all be physical signal.

Saved: raw_energy_summary.csv, raw_capillary_energy.h5, raw_energy_report.pdf/png and configuration.json.
'''
    (out/'README.md').write_text(notes)
    print(json.dumps(rows, indent=2), flush=True)
    print(f'Saved {out}', flush=True)
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--mean-from-pod', type=Path, help='Matching cache used only for full-recording preprocessing means.')
    p.add_argument('--frames', type=int, default=2000)
    p.add_argument('--frame-seed', type=int, default=12345)
    p.add_argument('--gamma', type=float, default=.0728)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if min(a.frames, a.batch_size) < 1 or a.frame_seed < 0 or not np.isfinite(a.gamma) or a.gamma <= 0:
        p.error('Counts and gamma must be positive; frame seed must be nonnegative.')
    return a


if __name__ == '__main__':
    run(parse_args())
