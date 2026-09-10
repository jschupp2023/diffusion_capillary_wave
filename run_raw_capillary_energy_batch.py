"""Collect compact raw-surface energy results from the lab experiment tree.

Defaults follow run_pod_batch.py: /disk/hyk049/DHM_new_experiment/0p*/Ca_ac*_rep*/.
All powers are included, including zero/low input. No POD files or mean removal
are needed. Run sequentially; completed recordings are checked and reused.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import traceback

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '4')

import h5py
import numpy as np

from run_pod_batch import DEFAULT_INPUT_ROOT, HDF5_SUFFIXES
from raw_capillary_energy import compute
from pod_capillary_energy import fingerprint, length_scale


SUMMARY_FIELDS = ['power', 'realization', 'repetition', 'source_relative', 'raw_filename', 'recording_key',
                  'Ebar_exact_J', 'Ebar_quad_J', 'exact_reduction_percent', 'rms_spatial_slope',
                  'frames_evaluated', 'full_frame_count', 'duration_seconds', 'area_m2',
                  'frame_seed', 'gamma_N_per_m', 'temporal_mean_removed', 'elapsed_seconds']


def atomic_csv(path, fields, rows):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def discover(raw_root, powers=None):
    """Match the existing lab layout; refuse to guess between multiple raw files."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f'Raw root does not exist: {raw_root}')
    jobs, issues = [], []
    directories = sorted((p for p in raw_root.iterdir() if p.is_dir() and re.fullmatch(r'0p\d+', p.name)),
                         key=lambda p: (float(p.name.replace('p', '.')), p.name))
    if powers:
        missing = sorted(set(powers)-{p.name for p in directories})
        issues += [dict(directory=str(raw_root/p), reason='requested power directory missing') for p in missing]
        directories = [p for p in directories if p.name in powers]
    for power in directories:
        repetitions = sorted((p for p in power.iterdir() if p.is_dir() and p.name.startswith('Ca_ac')),
                             key=lambda p: (int(re.search(r'_rep(\d+)$', p.name)[1]) if re.search(r'_rep(\d+)$', p.name) else -1, p.name))
        if not repetitions:
            issues.append(dict(directory=str(power), reason='no Ca_ac recording directories'))
        for rep in repetitions:
            candidates = sorted(p for p in rep.iterdir() if p.is_file() and p.suffix.lower() in HDF5_SUFFIXES)
            if len(candidates) != 1:
                issues.append(dict(directory=str(rep), reason=f'expected exactly one raw .h5/.hdf5 file; found {len(candidates)}'))
                continue
            match = re.search(r'_rep(\d+)$', rep.name)
            jobs.append(dict(power=power.name, realization=rep.name, repetition=int(match[1]) if match else '',
                             input=candidates[0].resolve(), source_relative=str(candidates[0].relative_to(raw_root)),
                             recording_key=f'recordings/{power.name}/{rep.name}'))
    return jobs, issues


def summarize(cache, out):
    rows = []
    if 'recordings' in cache:
        for power in cache['recordings'].values():
            for recording in power.values():
                if recording.attrs.get('complete', False):
                    rows.append(json.loads(recording.attrs['summary']))
    rows.sort(key=lambda r: (float(r['power'].replace('p', '.')), r['realization']))
    atomic_csv(out/'recording_summary.csv', SUMMARY_FIELDS, rows)
    power_rows = []
    for power in sorted({r['power'] for r in rows}, key=lambda p: float(p.replace('p', '.'))):
        selected = [r for r in rows if r['power']==power]
        entry = dict(power=power, recordings=len(selected))
        for label in ['exact', 'quad']:
            values = np.array([r[f'Ebar_{label}_J'] for r in selected])
            entry[f'mean_{label}_J'] = float(values.mean())
            entry[f'sd_{label}_J'] = float(values.std(ddof=1)) if len(values)>1 else None
            entry[f'median_{label}_J'] = float(np.median(values))
        power_rows.append(entry)
    fields = ['power', 'recordings', 'mean_exact_J', 'sd_exact_J', 'median_exact_J', 'mean_quad_J', 'sd_quad_J', 'median_quad_J']
    atomic_csv(out/'power_summary.csv', fields, power_rows)
    return len(rows)


def run(args):
    jobs, issues = discover(args.raw_root, args.powers)
    if args.max_files:
        jobs = jobs[:args.max_files]
    if args.dry_run:
        for job in jobs:
            print(f"{job['power']} / {job['realization']} -> {job['input']}")
        for issue in issues:
            print(f"SKIP {issue['directory']}: {issue['reason']}")
        print(f'{len(jobs)} recordings; {len(issues)} discovery issues. No computations performed.')
        return int(bool(issues) or not jobs)
    if not jobs:
        raise ValueError('No valid recording files discovered. Use --dry-run to inspect the layout.')
    config = dict(schema_version=1, frames=args.frames, frame_seed=args.frame_seed, gamma_N_per_m=args.gamma,
                  field='raw_as_stored', spatial_mean_removed=False, temporal_mean_removed=False,
                  observables=['exact_geometric_excess', 'quadratic'],
                  spatial_method='finite differences, second-order boundaries; trapezoidal quadrature',
                  temporal_method='uniform without replacement; full trapezoidal weights / inclusion probability')
    encoded = json.dumps(config, sort_keys=True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_name = 'batch_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.log'
    failed = 0
    with h5py.File(out/'raw_capillary_energy.h5', 'a') as cache, (out/log_name).open('w') as log:
        if cache.attrs.get('analysis_configuration', encoded) != encoded:
            raise ValueError('Output uses different analysis settings. Choose a new --output-dir.')
        cache.attrs['analysis_configuration'] = encoded
        def status(message):
            print(message, flush=True)
            log.write(message+'\n'); log.flush()
        settings = dict(**config, raw_root=str(args.raw_root.resolve()), batch_size=args.batch_size,
                        powers=args.powers or 'all', max_files=args.max_files, updated_utc=datetime.now(timezone.utc).isoformat())
        (out/'configuration.json').write_text(json.dumps(settings, indent=2)+'\n')
        manifest = [dict(power=j['power'], realization=j['realization'], source_relative=j['source_relative'],
                         source_file=str(j['input']), recording_key=j['recording_key']) for j in jobs]
        atomic_csv(out/'manifest.csv', list(manifest[0]), manifest)
        atomic_csv(out/'discovery_issues.csv', ['directory', 'reason'], issues)
        for issue in issues:
            status(f"DISCOVERY ISSUE: {issue['directory']}: {issue['reason']}")
        summarize(cache, out)
        for i, job in enumerate(jobs):
            key = job['recording_key']
            try:
                mark = json.dumps(fingerprint(job['input']), sort_keys=True)
                if key in cache and cache[key].attrs.get('complete', False) and cache[key].attrs.get('source_fingerprint') == mark and not args.overwrite:
                    status(f"[{i+1}/{len(jobs)}] cached {job['power']}/{job['realization']}")
                    continue
                if key in cache:
                    cache[key].attrs['complete'] = False
                    cache.flush()
                status(f"[{i+1}/{len(jobs)}] computing {job['power']}/{job['realization']}")
                single_args = argparse.Namespace(input=job['input'], mean_from_pod=None, frames=args.frames,
                                                 frame_seed=args.frame_seed, gamma=args.gamma, batch_size=args.batch_size)
                with contextlib.redirect_stdout(log):
                    info, indices, weights, energies, variants, values, metadata = compute(single_args)
                if fingerprint(job['input']) != json.loads(mark):
                    raise ValueError('Source file changed during computation; result was not saved.')
                values = values[0]
                summary = dict(power=job['power'], realization=job['realization'], repetition=job['repetition'],
                               source_relative=job['source_relative'], raw_filename=job['input'].name, recording_key=key,
                               **{k:v for k,v in values.items() if k!='field'}, duration_seconds=metadata['duration_seconds'],
                               area_m2=metadata['area_m2'], frame_seed=args.frame_seed, gamma_N_per_m=args.gamma,
                               temporal_mean_removed=False, elapsed_seconds=metadata['elapsed_seconds'])
                pending = '_pending/'+key
                if pending in cache:
                    del cache[pending]
                g = cache.create_group(pending)
                g.attrs['source_fingerprint'] = mark
                g.attrs['metadata'] = json.dumps(metadata)
                g.attrs['summary'] = json.dumps(summary)
                for name, array, unit in [('frame_indices', indices, 'index'), ('frame_numbers', info.frame_numbers[indices], 'index'),
                                          ('time_seconds', info.time[indices], 's'), ('mean_weights', weights, 'dimensionless'),
                                          ('x_m', info.x*length_scale(info.units['x']), 'm'), ('y_m', info.y*length_scale(info.units['y']), 'm'),
                                          ('E_exact', energies[0, 1], 'J'), ('E_quad', energies[0, 0], 'J')]:
                    g.create_dataset(name, data=array, compression='gzip', shuffle=True).attrs['units'] = unit
                g.attrs['complete'] = True
                cache.flush()
                if key in cache:
                    del cache[key]
                cache.require_group(str(Path(key).parent))
                cache.move(pending, key)
                cache.flush()
                status(f"  saved: exact={summary['Ebar_exact_J']*1e9:.6g} nJ; {summary['elapsed_seconds']:.1f} s")
            except Exception as error:
                failed += 1
                status(f"  FAILED {job['input']}: {error}")
                log.write(traceback.format_exc()+'\n'); log.flush()
            finally:
                summarize(cache, out)
        total = summarize(cache, out)
        cache.attrs['completed_recordings'] = total
        cache.attrs['last_run_failures'] = failed
        status(f'Finished: {total} completed recordings in archive; {failed} failed jobs; {len(issues)} discovery issues.')
    (out/'README.md').write_text(f'''# Raw capillary-energy collection

Transfer this entire folder to the laptop. The raw height-map files and POD files are not needed to read these outputs.

- `recording_summary.csv`: one row per completed recording, power/repetition identifiers, exact and quadratic mean energy in joules, source-relative path, sampled count, duration, area and RMS slope.
- `power_summary.csv`: mean, median and SD of recording-level means across repetitions. SD is not the sampling uncertainty of an individual time mean.
- `raw_capillary_energy.h5`: compact sampled per-frame energies, frame indices/numbers, selected times, original time-quadrature sampling weights, spatial coordinates, units and source metadata for every completed recording.
- `configuration.json`, `manifest.csv`, `discovery_issues.csv`, `batch_*.log`: settings, discovered inputs, skipped/ambiguous directories and computation failures.

Defaults: {args.frames} random frames without replacement, seed {args.frame_seed}, gamma={args.gamma} N/m, full spatial grid. The frame seed is reused independently for each recording (equal-length recordings therefore use the same frame indices).
No temporal or spatial mean subtraction; no POD truncation, smoothing or normalization. The exact geometric excess-area formula is evaluated at selected frames and the full-time mean is estimated with inverse-inclusion-probability trapezoidal weights.
Every available power is included unless --powers is supplied; low-power and zero-input folders are not excluded.
Power labels come from the experiment folders, as in the existing batch runners. Original filenames are also saved for auditing.

Only compact scalar time samples and metadata are stored, never raw height maps. At 2,000 samples the six sampled vectors require about 96 kB per recording before compression, plus grids and HDF5 metadata.
The raw measurements can contain static deformation and spatial measurement noise; the result is capillary deformation energy of the measured surface, not total mechanical energy.

Rerunning the same command skips completed results when the analysis settings and source path/size/modification time match. Interrupted or failed recordings are retried; a completed recording is committed only after all its arrays are written.
Changed frames, gamma or frame seed require a different output directory. --overwrite recomputes jobs under the same analysis configuration.
Run a single batch process per output directory. CSVs are refreshed after each job and include all completed archive entries, including prior partial runs.

Archive currently contains {total} completed recordings. Last run: {failed} computation failures and {len(issues)} discovery issues. Review the logs for coverage before comparing powers.
''')
    return int(bool(failed or issues))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--powers', nargs='+', help='Optional power folders; default all, including low/zero inputs.')
    p.add_argument('--frames', type=int, default=2000)
    p.add_argument('--frame-seed', type=int, default=12345)
    p.add_argument('--gamma', type=float, default=.0728)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-files', type=int)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args(argv)
    if min(a.frames, a.batch_size) < 1 or a.frame_seed < 0 or not np.isfinite(a.gamma) or a.gamma<=0 or (a.max_files is not None and a.max_files<1):
        p.error('Counts and gamma must be positive; seed must be nonnegative.')
    return a


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
