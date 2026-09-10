"""Capillary deformation energy of a shared-basis POD reconstruction.

    python pod_capillary_energy.py --rank 10 100 1000 --references 3 --seed 12345
    python pod_capillary_energy.py --basis-scope per-power --references 2
    python pod_capillary_energy.py --exact --basis-scope per-power --references 1 --powers 0p35 --max-repetitions 1 --rank 500
    python pod_capillary_energy.py --rank 10 --references 1

Default geometry integrates both slopes of the cached 2-D surface. The optional
horizontal centerline implements the 1-D formula (energy per unit transverse
width). Three reference recordings and ranks 10, 100, 1000 are compared by
default, excluding powers <= 0.04. Reports focus on time-averaged energy.
Use --basis-scope per-power for two reference recordings within each power.
Source PODs and time-weighted second moments are reused; no POD is recomputed.
The default evaluates quadratic means directly from coefficient second moments.
Add --exact for expensive geometric excess-energy validation on selected cases.
With --exact, energy trajectories are cached and fields reconstructed in batches.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import textwrap
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from pod_analysis_report import load_basis, _check_grid
from pod_lagged_covariance import discover_repetitions
from compare_pod_lagged_covariance import read_signature, write_csv

ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")


def project_gamma() -> tuple[float, str]:
    """Read the documented project constant without executing its module."""
    path = Path(__file__).with_name("capillary_wave_analysis.py")
    if path.is_file():
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "surf_ten" for t in node.targets):
                try:
                    value = float(ast.literal_eval(node.value))
                except (ValueError, TypeError):
                    continue
                if np.isfinite(value) and value > 0:
                    return value, f"{path.name}:{node.lineno}, surf_ten [N/m]"
    return 1., "fallback gamma=1; physical surface-tension prefactor unspecified"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, nargs="+", default=[10, 100, 1000], help="Target ranks (default: 10 100 1000).")
    parser.add_argument("--source-rank", type=int, help="Stored modes used for projection (default: max(100, largest target rank)).")
    parser.add_argument("--pod-rank", type=int, default=1000)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--powers", nargs="+", default=["all"], help="Compared powers; powers <= 0.04 are excluded.")
    parser.add_argument("--geometry", choices=["surface", "centerline"], default="surface")
    parser.add_argument("--gamma", type=float, help="Surface tension [N/m]; project constant if available, otherwise 1. "
                        "Use 1 for geometric energy up to the physical prefactor.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--basis-scope", choices=["global", "per-power"], default="global",
                        help="Share each basis globally or only within its own power.")
    parser.add_argument("--references", type=int, help="Distinct reference recordings globally (default: 3), or per power (default: 2).")
    parser.add_argument("--max-repetitions", type=int)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--exact", action="store_true", help="Also reconstruct every frame for exact geometric energy (default: fast quadratic means only).")
    parser.add_argument("--exact-frames", type=int, help="Estimate exact mean from this many uniformly sampled distinct frames; requires --exact.")
    parser.add_argument("--frame-seed", type=int, default=12345, help="Frame-sampling seed, independent of basis selection (default: 12345).")
    parser.add_argument("--field-batch-size", type=int, default=128,
                        help="Reconstructed frames in memory at once (default: 128); no time subsampling.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--moment-cache", type=Path, help="Reusable source second-moment cache; independent of reference basis and gamma.")
    args = parser.parse_args(argv)
    if args.frame_seed < 0 or (args.exact_frames is not None and (args.exact_frames < 1 or not args.exact)):
        parser.error('--exact-frames must be positive and requires --exact; --frame-seed must be nonnegative.')
    if args.references is None:
        args.references = 2 if args.basis_scope == 'per-power' else 3
    args.rank = sorted(set(args.rank))
    if args.source_rank is None:
        args.source_rank = max(100, max(args.rank))
    if min(args.rank) < 1 or args.source_rank < max(args.rank) or args.source_rank > args.pod_rank:
        parser.error("Require 1 <= target ranks <= source rank <= stored POD rank.")
    if min(args.batch_size, args.field_batch_size, args.references) < 1 or args.seed < 0 or (args.max_repetitions is not None and args.max_repetitions < 1):
        parser.error("Batch size, reference count and repetition limit must be positive; seed must be nonnegative.")
    default_gamma, source = project_gamma()
    args.gamma_source = source if args.gamma is None else "explicit --gamma"
    args.gamma = default_gamma if args.gamma is None else args.gamma
    if not np.isfinite(args.gamma) or args.gamma <= 0:
        parser.error("--gamma must be finite and positive.")
    if args.output_dir is None:
        args.output_dir = args.data_root / "capillary_energy" / (
            args.geometry + ('_per_power' if args.basis_scope == 'per-power' else '') +
            f"_bases_{args.references}_ranks_" + "_".join(map(str, args.rank)) + f"_seed_{args.seed}")
        if args.exact_frames is not None:
            args.output_dir = args.output_dir.with_name(args.output_dir.name+f'_frames_{args.exact_frames}_seed_{args.frame_seed}')
    if args.moment_cache is None:
        args.moment_cache = args.data_root / "capillary_energy" / f"source_moments_rank_{args.source_rank}.h5"
    return args


def length_scale(unit: str) -> float:
    unit = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    scales = {"m": 1., "meter": 1., "meters": 1., "metres": 1., "metre": 1.,
              "mm": 1e-3, "millimeters": 1e-3, "um": 1e-6, "micron": 1e-6,
              "microns": 1e-6, "micrometers": 1e-6, "micrometres": 1e-6,
              "nm": 1e-9, "nanometers": 1e-9}
    if unit not in scales:
        raise ValueError(f"Unsupported spatial/displacement unit {unit!r}; cannot compute physical energy.")
    return scales[unit]


def quadrature_weights(grid: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature, including half weights at the boundaries."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Spatial grids must be finite, strictly increasing and have >=3 points.")
    steps = np.diff(grid)
    return np.r_[steps[0]/2, (steps[:-1]+steps[1:])/2, steps[-1]/2]


def capillary_stiffness(modes: np.ndarray, x_m: np.ndarray, y_m: np.ndarray,
                        geometry: str = "surface", gamma: float = 1.) -> tuple[np.ndarray, dict]:
    """Build gamma times the discrete gradient Gram matrix of dimensionless modes.

    Central finite differences inside; second-order one-sided derivatives at
    boundaries; trapezoidal quadrature. No periodic boundary is assumed.
    """
    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 3 or modes.shape[1:] != (len(y_m), len(x_m)) or not np.isfinite(modes).all():
        raise ValueError("Expected finite modes with shape (rank, y, x).")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Surface tension must be positive and finite.")
    wx, wy = quadrature_weights(x_m), quadrature_weights(y_m)
    center = int(np.argmin(np.abs(y_m - .5*(y_m[0]+y_m[-1]))))
    if geometry == "centerline":
        gx = np.gradient(modes[:, center, :], x_m, axis=-1, edge_order=2)
        matrix = (gx*wx) @ gx.T
        measure = float(wx.sum())
    elif geometry == "surface":
        weight = np.outer(wy, wx).ravel()
        gx = np.gradient(modes, x_m, axis=-1, edge_order=2).reshape(len(modes), -1)
        matrix = (gx*weight) @ gx.T
        del gx
        gy = np.gradient(modes, y_m, axis=-2, edge_order=2).reshape(len(modes), -1)
        matrix += (gy*weight) @ gy.T
        measure = float(weight.sum())
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    matrix *= gamma
    symmetry_error = float(np.linalg.norm(matrix-matrix.T, ord="fro"))
    matrix = .5*(matrix+matrix.T)
    info = dict(measure=measure, centerline_y_index=center, centerline_y_m=float(y_m[center]),
                symmetry_error_before_symmetrizing=symmetry_error)
    return matrix, info


def matrix_diagnostics(matrix: np.ndarray) -> dict:
    eig = np.linalg.eigvalsh(.5*(matrix+matrix.T))
    tolerance = 1e-10*max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    if eig[0] < -tolerance:
        raise ValueError(f"Capillary stiffness is not positive semidefinite: minimum eigenvalue {eig[0]:.6g}")
    return dict(symmetry_error=float(np.linalg.norm(matrix-matrix.T, ord="fro")),
                minimum_eigenvalue=float(eig[0]), maximum_eigenvalue=float(eig[-1]),
                psd_tolerance=tolerance, positive_semidefinite=True)


def energy_series(coordinates_m: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = .5*np.einsum("ti,ij,tj->t", coordinates_m, matrix, coordinates_m, optimize=True)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite capillary energy.")
    tolerance = 1e-10*max(float(np.max(np.abs(result))), np.finfo(float).tiny)
    if np.min(result) < -tolerance:
        raise ValueError("Negative capillary energy beyond numerical tolerance.")
    return np.maximum(result, 0)


def energy_statistics(t: np.ndarray, energy: np.ndarray) -> dict:
    if len(t) != len(energy) or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Energy needs a matching, strictly increasing time grid.")
    return dict(time_averaged_energy=float(np.trapezoid(energy, t)/(t[-1]-t[0])),
                sample_mean=float(np.mean(energy)), median=float(np.median(energy)),
                std=float(np.std(energy, ddof=1)), minimum=float(np.min(energy)),
                q05=float(np.quantile(energy, .05)), q25=float(np.quantile(energy, .25)),
                q75=float(np.quantile(energy, .75)), q95=float(np.quantile(energy, .95)),
                maximum=float(np.max(energy)))


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)



def temporal_weights(t: np.ndarray) -> np.ndarray:
    """Normalized trapezoidal weights, valid also for nonuniform time grids."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Expected at least two finite, strictly increasing times.')
    dt = np.diff(t)
    return np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]/(t[-1]-t[0])


def coefficient_second_moment(coefficients, t, source_rank, scale, batch_size):
    """Integral of b(t)b(t)^T / duration in m², without additional centering.

    Read bounded batches; no full time-by-rank projection or diagonal-POD
    covariance assumption is needed. Retain all cross terms and endpoints.
    """
    weights = temporal_weights(t)
    if coefficients.ndim != 2 or coefficients.shape[0] != len(t) or coefficients.shape[1] < source_rank:
        raise ValueError('Coefficient shape is inconsistent with time or source rank.')
    moment = np.zeros((source_rank, source_rank))
    for start in range(0, len(t), batch_size):
        stop = min(start+batch_size, len(t))
        batch = np.array(coefficients[start:stop, :source_rank], dtype=float, copy=True)
        if not np.isfinite(batch).all():
            raise ValueError('Nonfinite POD coefficients.')
        batch *= scale*np.sqrt(weights[start:stop, None])
        moment += batch.T @ batch
    if not np.isfinite(moment).all():
        raise ValueError('Nonfinite source second moment.')
    return .5*(moment+moment.T)


def mean_energies(moment, overlap, geometric_matrix, ranks):
    """Exact discrete time averages: Ebar_r = trace(K_r P_r S P_r^T)/2."""
    projected = overlap @ moment @ overlap.T
    result = np.array([.5*np.einsum('ij,ji->', geometric_matrix[:r, :r], projected[:r, :r])
                       for r in ranks])
    tolerance = 1e-10*max(float(np.max(np.abs(result))), np.finfo(float).tiny)
    if not np.isfinite(result).all() or np.min(result) < -tolerance:
        raise ValueError('Invalid time-averaged capillary energy.')
    return np.maximum(result, 0.)


def field_energies(field_m, x_m, y_m, geometry='surface'):
    """Geometric quadratic/exact excess per frame with the stiffness discretization.

    s/(sqrt(1+s)+1) is algebraically sqrt(1+s)-1 without cancellation at small slopes.
    The physical energies are gamma times these geometric integrals.
    """
    wx, wy = quadrature_weights(x_m), quadrature_weights(y_m)
    if geometry == 'surface':
        gx = np.gradient(field_m, x_m, axis=-1, edge_order=2)
        gy = np.gradient(field_m, y_m, axis=-2, edge_order=2)
        s = gx**2 + gy**2
        weight = np.outer(wy, wx).ravel()
    elif geometry == 'centerline':
        center = int(np.argmin(np.abs(y_m - .5*(y_m[0]+y_m[-1]))))
        gx = np.gradient(field_m[:, center, :], x_m, axis=-1, edge_order=2)
        s = gx**2
        weight = wx
    else:
        raise ValueError(f'Unknown geometry: {geometry}')
    s = s.reshape(len(field_m), -1)
    quad = .5*(s @ weight)
    exact = (s/(np.sqrt(1+s)+1)) @ weight
    if not np.isfinite(quad).all() or not np.isfinite(exact).all():
        raise ValueError('Nonfinite reconstructed surface energy.')
    if np.any(exact < 0) or np.any(exact > quad*(1+1e-12)):
        raise ValueError('Expected 0 <= exact surface energy <= quadratic energy.')
    return quad, exact


def reconstructed_energy_series(aligned_m, basis, x_m, y_m, ranks, geometry, field_batch_size):
    """Reuse nested-rank reconstruction increments, bounding spatial-field memory."""
    quad = np.empty((len(ranks), len(aligned_m)))
    exact = np.empty_like(quad)
    for start in range(0, len(aligned_m), field_batch_size):
        stop = min(start+field_batch_size, len(aligned_m))
        field = np.zeros((stop-start, basis.shape[1]))
        previous = 0
        for ri, rank in enumerate(ranks):
            field += aligned_m[start:stop, previous:rank] @ basis[previous:rank]
            quad[ri, start:stop], exact[ri, start:stop] = field_energies(
                field.reshape(stop-start, len(y_m), len(x_m)), x_m, y_m, geometry)
            previous = rank
    return quad, exact


def select_energy_frames(t, count, seed):
    """Uniform sample without replacement; unbiased estimate of full trapezoidal mean.

    Retain full-grid weights and divide by each frame's inclusion probability.
    Integrating across random gaps would instead give different, random weights.
    """
    weights = temporal_weights(t)
    n = len(t)
    if count is None or count >= n:
        return np.arange(n), weights
    indices = np.sort(np.random.default_rng(seed).choice(n, count, replace=False))
    return indices, weights[indices]*(n/count)


def read_selected_coefficients(dataset, indices, rank):
    """Read each touched HDF5 time chunk once, avoiding slow fancy indexing."""
    result = np.empty((len(indices), rank))
    chunk_size = dataset.chunks[0] if dataset.chunks else 1024
    chunk_ids = indices//chunk_size
    for chunk in np.unique(chunk_ids):
        positions = np.flatnonzero(chunk_ids == chunk)
        start = int(chunk)*chunk_size
        block = np.asarray(dataset[start:min(start+chunk_size, len(dataset)), :rank], dtype=float)
        result[positions] = block[indices[positions]-start]
    return result


def cached_energy_trajectories(group, path, prepared, signature, args):
    """Append trajectories to old quadratic caches; checkpoint every coefficient batch.

    Parent source fingerprint and projection configuration guard this cache too.
    Geometric arrays are gamma independent. Never substitute second moments for
    the nonlinear excess-area integral.
    """
    sampled = args.exact_frames is not None
    series_key = f'sampled_trajectories_n{args.exact_frames}_seed{args.frame_seed}' if sampled else 'trajectories'
    series = group.require_group(series_key)
    if not series.attrs.get('complete', False):
        source, x, y = load_basis(path, args.source_rank)
        _check_grid(*prepared[0]['grid'], x, y, str(path))
        overlaps = [ref['basis'] @ source.T for ref in prepared]
        del source
        with h5py.File(path) as handle:
            t = np.asarray(handle['grid/time'], dtype=float)
            temporal_weights(t)  # Validate before writing or resuming.
            indices, mean_weights = select_energy_frames(t, args.exact_frames, args.frame_seed)
            n_frames = len(indices)
            coefficients = handle['reduced/coefficients']
            if coefficients.shape[0] != len(t) or coefficients.shape[1] < args.source_rank:
                raise ValueError(f'Invalid coefficient shape: {path}')
            if not series.attrs.get('initialized', False):
                for key in list(series):
                    del series[key]
                series.create_dataset('time_seconds', data=t[indices]).attrs['units'] = 's'
                series.create_dataset('frame_indices', data=indices)
                series.create_dataset('mean_weights', data=mean_weights)
                series.attrs['full_frame_count'] = len(t)
                series.attrs['frame_seed'] = args.frame_seed
                series.attrs['time_average_method'] = 'uniform without replacement; full trapezoidal weights / inclusion probability' if sampled else 'full trapezoidal quadrature'
                for key in ['geometric_quad', 'geometric_exact']:
                    ds = series.create_dataset(key, shape=(len(prepared), len(args.rank), n_frames), dtype='f8',
                                               chunks=(1, 1, min(args.batch_size, n_frames)), compression='gzip')
                    ds.attrs['axis_order'] = 'reference, rank, time'
                    ds.attrs['units'] = 'm^2' if args.geometry == 'surface' else 'm'
                series.attrs['completed_samples'] = 0
                series.attrs['algorithm'] = 'v1: nested reconstructed field; finite differences; trapezoidal spatial quadrature'
                series.attrs['initialized'] = True
                group.file.flush()
            start_sample = int(series.attrs['completed_samples'])
            for start in range(start_sample, n_frames, args.batch_size):
                stop = min(start+args.batch_size, n_frames)
                batch = (read_selected_coefficients(coefficients, indices[start:stop], args.source_rank) if sampled else
                         np.asarray(coefficients[start:stop, :args.source_rank], dtype=float))
                if not np.isfinite(batch).all():
                    raise ValueError(f'Nonfinite coefficients: {path}')
                for bi, ref in enumerate(prepared):
                    aligned = (batch @ overlaps[bi].T)*length_scale(signature[5])
                    x, y = ref['grid']
                    quad, exact = reconstructed_energy_series(
                        aligned, ref['basis'], x*length_scale(signature[2]), y*length_scale(signature[3]),
                        args.rank, args.geometry, args.field_batch_size)
                    series['geometric_quad'][bi, :, start:stop] = quad
                    series['geometric_exact'][bi, :, start:stop] = exact
                group.file.flush()  # Persist data before advancing the resume marker.
                series.attrs['completed_samples'] = stop
                group.file.flush()
                print(f'    reconstructed energy: {stop}/{n_frames} frames'+(f' (sampled from {len(t)})' if sampled else ''), flush=True)
            series.attrs['complete'] = True
            group.file.flush()
    weights = series['mean_weights'][:] if 'mean_weights' in series else temporal_weights(series['time_seconds'][:])
    means = {}
    for observable in ['quad', 'exact']:
        geometric = series[f'geometric_{observable}'][:]
        means[observable] = geometric @ weights
        key = f'E_{observable}'
        if key in series:
            del series[key]
        ds = series.create_dataset(key, data=args.gamma*geometric, compression='gzip')
        ds.attrs['units'] = group.file.attrs['energy_units']
        ds.attrs['axis_order'] = 'reference, rank, time'
    if not sampled and not np.allclose(means['quad'], group['geometric_mean_energy'][:], rtol=1e-8, atol=1e-30):
        raise ValueError('Reconstructed quadratic time average disagrees with the stiffness/second-moment calculation.')
    return means


def select_references(pool, count, seed):
    """Uniform selection without replacement; first choice matches the old script."""
    if count > len(pool):
        raise ValueError(f'Requested {count} distinct bases, but only {len(pool)} recordings qualify.')
    rng = np.random.default_rng(seed)
    remaining = list(pool)
    return [remaining.pop(int(rng.integers(len(remaining)))) for _ in range(count)]


def cached_second_moment(cache, power, record, signature, args):
    key = f'{power}_rep{record.number}'
    mark = json.dumps(dict(version=1, source=fingerprint(record.pod_path), source_rank=args.source_rank,
                           signature=signature, weighting='normalized trapezoid; uncentered'), sort_keys=True)
    if key in cache and cache[key].attrs.get('configuration') == mark and cache[key].attrs.get('complete', False):
        group = cache[key]
        return group['second_moment'][:], dict(samples=int(group.attrs['samples']),
                                             duration_seconds=float(group.attrs['duration_seconds'])), 'cached'
    with h5py.File(record.pod_path) as source:
        t = np.asarray(source['grid/time'], dtype=float)
        moment = coefficient_second_moment(source['reduced/coefficients'], t, args.source_rank,
                                           length_scale(signature[5]), args.batch_size)
    if key in cache:
        del cache[key]
    group = cache.create_group(key)
    group.create_dataset('second_moment', data=moment).attrs['units'] = 'm^2'
    group.attrs['configuration'] = mark
    group.attrs['samples'] = len(t)
    group.attrs['duration_seconds'] = float(t[-1]-t[0])
    group.attrs['complete'] = True
    cache.flush()
    return moment, dict(samples=len(t), duration_seconds=float(t[-1]-t[0])), 'computed'


def write_study_report(out, rows, diagnostics, references, settings):
    unit = settings['energy_units']
    powers, ranks = settings['powers'], settings['ranks']
    local = settings['basis_scope'] == 'per-power'
    include_exact = settings['compute_exact']
    sampled = settings['exact_frames'] is not None
    summaries = []
    for ref in references:
        for rank in ranks:
            for power in powers:
                if local and ref['power'] != power:
                    continue
                values = np.array([row['time_averaged_energy'] for row in rows
                                   if row['reference_id']==ref['reference_id'] and row['rank']==rank and row['power']==power])
                exact = np.array([row['Ebar_exact'] for row in rows
                                  if row['reference_id']==ref['reference_id'] and row['rank']==rank and row['power']==power]) if include_exact else None
                summaries.append(dict(reference_id=ref['reference_id'], reference_power=ref['power'],
                                      reference_slot=ref.get('reference_slot', ref['reference_id']),
                                      reference_repetition=ref['repetition'], rank=rank, power=power,
                                      repetitions=len(values), mean=float(values.mean()),
                                      sd=float(values.std(ddof=1)) if len(values)>1 else float('nan'),
                                      mean_quad=float(values.mean()), sd_quad=float(values.std(ddof=1)) if len(values)>1 else float('nan'),
                                      **(dict(mean_exact=float(exact.mean()), sd_exact=float(exact.std(ddof=1)) if len(exact)>1 else float('nan')) if include_exact else {}),
                                      median=float(np.median(values)), units=unit))
    for filename, entries in [('repetition_energy.csv', rows), ('power_energy_summary.csv', summaries),
                              ('stiffness_diagnostics.csv', diagnostics)]:
        write_csv(out/filename, list(entries[0]), [list(row.values()) for row in entries])
    write_csv(out/'references.csv', ['reference_id', 'reference_slot', 'power', 'repetition', 'source_file'],
              [[r['reference_id'], r.get('reference_slot', r['reference_id']), r['power'], r['repetition'], r['source']['path']]
               for r in references])

    notes = [
        'Full surface: E_gamma = gamma/2 integral[(d eta/dx)^2 + (d eta/dy)^2] dx dy.'
        if settings['geometry']=='surface' else 'Centerline: E_gamma = gamma/2 integral[(d eta/dx)^2] dx, per transverse width.',
        'Exact geometric excess energy: E_exact = gamma integral[sqrt(1 + |grad eta|^2) - 1] dA. '
        'Both observables use the same reconstructed field, finite differences and trapezoidal spatial/time quadrature. '
        'Exact refers to the graph-area formula, evaluated numerically on the retained POD field; it is not the energy of unresolved modes or the omitted mean surface.',
        f"Compared {settings['recordings']} repetitions at powers strictly greater than 0.04. "
        f"Selected {len(references)} distinct reference recordings uniformly without replacement using seed {settings['seed']}. "
        + ('Each reference supplies a basis only for repetitions at its own power; two draws per power by default. '
           if local else 'Each reference supplies one shared basis for all repetitions. ')
        + 'Ranks use nested prefixes of the same reference basis.',
        'References: '+ '; '.join(f"basis {r['reference_id']}: {r['power']}/rep{r['repetition']}" for r in references)+'.',
        f"Target ranks: {ranks}; source rank: {settings['source_rank']}. "
        f"gamma={settings['gamma']:g} ({settings['gamma_source']}); output units: {unit}. "
        + ('gamma=1 omits the physical surface-tension prefactor.' if settings['gamma_one_proxy'] else 'Spatial coordinates and displacements are converted to metres.'),
        'For stored coordinates b and overlap P = Phi_reference^T Phi_source, calculate S = integral[b b^T] dt / duration. '
        'Then Ebar_r = trace(K_gamma,r P_r S P_r^T)/2. This equals integrating the quadratic energy trajectory with the same trapezoidal weights, '
        'including endpoints and all cross terms. No extra centering, diagonal covariance assumption or amplitude normalization is used.',
        'For the nonlinear energy, reconstruct every frame in bounded batches, differentiate spatially, '
        'then integrate s/(sqrt(1+s)+1), where s=|grad eta|^2. This stable expression avoids cancellation for small slopes. '
        'Compute E_quad(t) from the same slopes and verify its time average against the existing stiffness result. '
        'No time or spatial subsampling is used. At every frame 0 <= E_exact <= E_quad; they agree in the small-slope limit.',
        'The plot shows the mean of repetition-level time averages; error bars are SD across repetitions, not confidence intervals. '
        'Color denotes rank and line style denotes reference draw. Exact curves use filled markers; quadratic curves use open markers and thinner lines. '
        'The same recordings are reused across curves; these are not independent samples. '
        + ('Draw 1/2 connects independent local reference choices across powers, not one common basis. '
           'References are included among the evaluated repetitions. Agreement of the two draws measures sensitivity to within-power basis selection. '
           'Across-power differences also reflect different retained subspaces, especially at low rank.' if local else ''),
        'These are capillary deformation energies of the retained fluctuating field, not total mechanical energy. '
        'The removed temporal mean shape and its energy/cross-term are not restored. A spatially constant field contributes zero up to numerical precision. '
        'Leading-subspace similarity does not establish similarity at rank 1000. Finite source truncation and derivative cross terms can affect rank comparisons; '
        'energy need not increase monotonically with rank.',
        f"All stiffness matrices pass the PSD check. Space/time RMS slopes range from {min(r['rms_spatial_slope'] for r in rows):.4g} "
        f"to {max(r['rms_spatial_slope'] for r in rows):.4g}; the small-slope approximation requires slopes small compared with one.",
    ]
    if not include_exact:
        notes = [n for n in notes if not n.startswith(('Exact geometric excess energy:', 'For the nonlinear energy,'))]
        notes.insert(1, 'Fast quadratic-only analysis. Exact reconstruction is disabled; add --exact for a targeted validation.')
        notes = [n.replace('Exact curves use filled markers; quadratic curves use open markers and thinner lines. ', '') for n in notes]
    elif sampled:
        notes.insert(1, f"Random-frame estimate: requested {settings['exact_frames']} distinct frames per recording, frame seed {settings['frame_seed']}, independent of basis seed. "
                     'All ranks and reference draws use the same frame indices for a recording. The geometric integrand is exact on the retained field; its time average is estimated. '
                     'Uniform sampling without replacement uses full-grid trapezoidal weights divided by the inclusion probability, giving an unbiased estimate of the original discrete time average. '
                     'The plot retains the full-recording quadratic mean. Ebar_quad in the repetition CSV is the sampled quadratic estimate; time_averaged_energy remains the full quadratic mean. '
                     'Error bars show repetition SD, not uncertainty due to frame sampling.')
        notes = [n.replace('No time or spatial subsampling is used.', 'Only the selected time frames are reconstructed; the full spatial grid is used.')
                 .replace('verify its time average against the existing stiffness result.', 'report its sampling error relative to the full-recording stiffness result.') for n in notes]
    with PdfPages(out/'capillary_energy_report.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
        styles = ['-', '--', ':', '-.']
        markers = ['o', 's', '^', 'D', 'v', 'P']
        colors = plt.get_cmap('tab10').colors
        for ri, rank in enumerate(ranks):
            for bi in range(settings['references_per_scope']):
                selected = [s for s in summaries if s['reference_slot']==bi+1 and s['rank']==rank]
                selected.sort(key=lambda s: float(s['power'].replace('p', '.')))
                label = (f'r={rank}, local basis draw {bi+1}' if local else
                         f"r={rank}, basis {bi+1} ({references[bi]['power']}/rep{references[bi]['repetition']})")
                for observable in (['quad', 'exact'] if include_exact else ['quad']):
                    color = colors[ri % len(colors)]
                    filled = observable == 'exact' or not include_exact
                    ax.errorbar([float(s['power'].replace('p', '.')) for s in selected], [s[f'mean_{observable}'] for s in selected],
                                yerr=[s[f'sd_{observable}'] if np.isfinite(s[f'sd_{observable}']) else 0. for s in selected],
                                color=color, linestyle=styles[bi % len(styles)], marker=markers[bi % len(markers)],
                                markerfacecolor=color if filled else 'none',
                                alpha=1. if filled else .55,
                                markersize=5 if observable=='exact' else 7, capsize=2,
                                linewidth=2 if observable=='exact' else 1., label=f'{label}, {observable}')
        ax.set(xlabel='Power-input setting (Vpp)', ylabel=f'Time-averaged capillary deformation energy [{unit}]',
               title=(('Sampled exact vs full quadratic: ' if sampled else 'Exact vs quadratic: ') if include_exact else 'Quadratic energy: ')+'mean ± SD across repetitions\nColor: rank; line style: '+('local basis draw' if local else 'reference basis')+('; filled markers: exact' if include_exact else ''))
        ax.set_xticks([float(p.replace('p', '.')) for p in powers])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=min(3, len(references)))
        fig.savefig(out/'energy_over_power.png', dpi=180)
        fig.savefig(out/'energy_over_power.pdf')
        pdf.savefig(fig)
        plt.close(fig)
        note_lines = '\n\n'.join(textwrap.fill(n, 120) for n in notes).splitlines()
        for start in range(0, len(note_lines), 40):
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(.06, .95, 'Capillary energy comparison: method and interpretation', fontsize=15, va='top')
            fig.text(.06, .89, '\n'.join(note_lines[start:start+40]), fontsize=9, va='top', linespacing=1.3)
            pdf.savefig(fig)
            plt.close(fig)
    lines = ['# Capillary deformation energy: reference-basis and rank comparison', '', *[n+'\n' for n in notes],
             '## Stiffness diagnostics', '',
             '| Basis | Rank | Symmetry error | Minimum eigenvalue | Maximum eigenvalue | PSD |',
             '|---|---|---|---|---|---|']
    lines += [f"| {d['reference_id']} | {d['rank']} | {d['symmetry_error']:.3e} | {d['minimum_eigenvalue']:.6g} | {d['maximum_eigenvalue']:.6g} | {d['positive_semidefinite']} |" for d in diagnostics]
    lines += ['', 'Outputs: energy_over_power.png/pdf (all curves), capillary_energy_report.pdf, repetition_energy.csv, '
              'power_energy_summary.csv, references.csv, stiffness_diagnostics.csv, configuration.json and capillary_energy.h5.', '',
              'capillary_energy.h5 stores geometric stiffness and K_gamma for each basis (smaller ranks use leading blocks), '
              'plus repetition-level geometric and physical mean energies. When --exact is enabled, each recording has trajectories/time_seconds, '
              'E_quad and E_exact under trajectories (reference, rank, time), with gamma-independent geometric_quad/geometric_exact for cache reuse. '
              'Ebar_quad and Ebar_exact are in the repetition CSV and HDF5. Legacy time_averaged_energy/mean_energy and summary mean/sd remain quadratic.', '',
              'Random-frame runs use sampled_trajectories_n{count}_seed{seed} instead of trajectories. Selected frame_indices and mean_weights are saved there; full trajectories are kept separately.', '',
              f"Source second-moment cache: `{settings['moment_cache']}`. It is shared across reference bases, ranks and seeds "
              'when the source rank is the same. Both caches check source-file fingerprints. Completed recordings are reused after interruption; '
              'changing gamma only rescales cached geometric results. Use a different output directory for a different projection setup.', '',
              'Memory is bounded by reference/source mode matrices and coefficient batches, rather than full projected trajectories. '
              'In per-power mode, only the current power\'s reference modes are loaded at once. '
              'Exact energy requires reconstructing every frame and is substantially more expensive than the quadratic second-moment calculation. '
              'Use --field-batch-size to bound spatial-field memory; --batch-size controls coefficient batches and trajectory checkpoints. '
              'Existing quadratic means remain cached; nonlinear trajectories resume from the last completed coefficient batch. '
              'Progress reports reconstructed frame counts and elapsed seconds per recording; subsequent runs reuse completed work.', '',
              '```bash', 'python pod_capillary_energy.py --rank '+' '.join(map(str, ranks))+
              f" --source-rank {settings['source_rank']} --basis-scope {settings['basis_scope']} --references {settings['references_per_scope']} --seed {settings['seed']} --geometry {settings['geometry']} --gamma {settings['gamma']:g}"+(' --exact' if include_exact else '')+
              (f" --exact-frames {settings['exact_frames']} --frame-seed {settings['frame_seed']}" if sampled else '')+
              ' --powers '+' '.join(powers)+(f" --max-repetitions {settings['max_repetitions']}" if settings['max_repetitions'] else ''), '```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def prepare_references(cache, references, chosen, signature, args, diagnostics, total_references):
    """Load only the current power's bases in per-power mode."""
    prepared = []
    proxy = args.gamma == 1.
    for ref, (_, record) in zip(references, chosen):
        print(f"Preparing basis {ref['reference_id']}/{total_references}: {ref['power']}/rep{ref['repetition']} (rank {max(args.rank)})", flush=True)
        if read_signature(record.pod_path) != signature:
            raise ValueError(f'Inconsistent units/preprocessing: {record.pod_path}')
        group = cache.require_group(f"references/basis_{ref['reference_id']}")
        if not group.attrs.get('complete', False):
            basis, x, y = load_basis(record.pod_path, max(args.rank))
            x_m, y_m = x*length_scale(signature[2]), y*length_scale(signature[3])
            geometric, info = capillary_stiffness(basis.reshape(max(args.rank), len(y), len(x)), x_m, y_m, args.geometry)
            constant, _ = capillary_stiffness(np.ones((1, len(y), len(x))), x_m, y_m, args.geometry)
            for key in list(group):
                del group[key]
            for key, value in [('modes', basis), ('x', x), ('y', y), ('geometric_stiffness', geometric)]:
                group.create_dataset(key, data=value)
            for key, value in info.items():
                group.attrs[key] = value
            group.attrs['constant_field_geometric_stiffness'] = float(constant[0, 0])
            group.attrs['complete'] = True
            cache.flush()
        else:
            basis, x, y = group['modes'][:], group['x'][:], group['y'][:]
            geometric = group['geometric_stiffness'][:]
        if prepared:
            _check_grid(*prepared[0]['grid'], x, y, str(record.pod_path))
        if 'K_gamma' in group:
            del group['K_gamma']
        group.create_dataset('K_gamma', data=args.gamma*geometric).attrs['units'] = (
            ('dimensionless' if args.geometry=='surface' else '1/m') if proxy else
            ('N/m' if args.geometry=='surface' else 'N/m^2'))
        group['geometric_stiffness'].attrs['units'] = 'dimensionless' if args.geometry=='surface' else '1/m'
        group['x'].attrs['units'], group['y'].attrs['units'] = signature[2], signature[3]
        for rank in args.rank:
            diagnostics.append(dict(reference_id=ref['reference_id'], reference_power=ref['power'],
                                    reference_repetition=ref['repetition'], rank=rank,
                                    **matrix_diagnostics(args.gamma*geometric[:rank, :rank]),
                                    constant_field_geometric_stiffness=float(group.attrs['constant_field_geometric_stiffness'])))
        prepared.append(dict(basis=basis, grid=(x, y), matrix=geometric, measure=float(group.attrs['measure'])))
    return prepared


def run(args):
    available = sorted([p.name for p in args.data_root.iterdir()
                        if p.is_dir() and re.fullmatch(r'0p\d+', p.name) and float(p.name.replace('p', '.')) > .04],
                       key=lambda p: float(p.replace('p', '.')))
    powers = available if args.powers == ['all'] else sorted(
        {p for p in args.powers if float(p.replace('p', '.')) > .04}, key=lambda p: float(p.replace('p', '.')))
    if not powers or any(p not in available for p in powers):
        raise ValueError('Need existing power directories strictly above 0.04.')
    discovered = {p: discover_repetitions(args.data_root/p, args.pod_rank) for p in available}
    if args.powers == ['all']:
        powers = [p for p in powers if discovered[p]]
    pool = [(p, record) for p in available for record in discovered[p]]
    records = [(p, record) for p in powers for record in discovered[p][:args.max_repetitions]]
    if not records or any(not discovered[p] for p in powers):
        raise ValueError('Every compared power must contain cached POD recordings.')
    if args.basis_scope == 'per-power':
        rng = np.random.default_rng(args.seed)
        chosen = []
        for power in powers:
            if len(discovered[power]) < args.references:
                raise ValueError(f'{power} needs at least {args.references} distinct reference recordings.')
            chosen.extend(select_references([(power, r) for r in discovered[power]], args.references, rng))
    else:
        chosen = select_references(pool, args.references, args.seed)
    references = [dict(reference_id=i+1, power=p, repetition=r.number, source=fingerprint(r.pod_path))
                  for i, (p, r) in enumerate(chosen)]
    if args.basis_scope == 'per-power':
        for i, ref in enumerate(references):
            ref['reference_slot'] = i % args.references + 1
    signature = read_signature(chosen[0][1].pod_path)
    if signature[4] not in ('seconds', 'second', 's'):
        raise ValueError('Expected time in seconds.')
    projection = dict(version=2, ranks=args.rank, source_rank=args.source_rank, stored_rank=args.pod_rank,
                      geometry=args.geometry, seed=args.seed, references=references, signature=signature,
                      coordinates='stored coefficients; no extra centering or amplitude normalization',
                      time_average='normalized trapezoidal second moment')
    if args.basis_scope == 'per-power':
        projection.update(version=3, basis_scope=args.basis_scope)
    encoded = json.dumps(projection, sort_keys=True)
    proxy = args.gamma == 1.
    unit = ('m²' if args.geometry=='surface' else 'm') if proxy else ('J' if args.geometry=='surface' else 'J/m')
    settings = dict(**projection, powers=powers, recordings=len(records), gamma=args.gamma,
                    gamma_source=args.gamma_source, gamma_one_proxy=proxy, energy_units=unit,
                    moment_cache=str(args.moment_cache.resolve()),
                    inputs=[dict(power=p, repetition=r.number, **fingerprint(r.pod_path)) for p, r in records])
    settings.update(basis_scope=args.basis_scope, references_per_scope=args.references)
    settings.update(observables=['quadratic', 'exact_geometric_excess'] if args.exact else ['quadratic'],
                    compute_exact=args.exact, max_repetitions=args.max_repetitions,
                    exact_frames=args.exact_frames, frame_seed=args.frame_seed,
                    exact_integrand='s/(sqrt(1+s)+1), s=|grad eta|^2',
                    coefficient_batch_size=args.batch_size, field_batch_size=args.field_batch_size,
                    temporal_sampling='uniform without replacement' if args.exact_frames is not None else 'all frames', spatial_subsampling=1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    args.moment_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.moment_cache.resolve() == (out/'capillary_energy.h5').resolve():
        raise ValueError('The moment cache must be separate from capillary_energy.h5.')
    rows, diagnostics = [], []
    started = time.perf_counter()
    with h5py.File(out/'capillary_energy.h5', 'a') as cache, h5py.File(args.moment_cache, 'a') as moments:
        if cache.attrs.get('projection_configuration', encoded) != encoded:
            raise ValueError('Output has a different projection setup. Use another --output-dir.')
        cache.attrs['projection_configuration'] = encoded
        cache.attrs['gamma'] = args.gamma
        cache.attrs['energy_units'] = unit
        cache.attrs['observable'] = 'quadratic and exact geometric capillary excess energies of the retained fluctuating POD field'
        (out/'configuration.json').write_text(json.dumps(settings, indent=2)+'\n')
        if 'ranks' not in cache:
            cache.create_dataset('ranks', data=args.rank)
        print(f'{len(records)} recordings × {args.references} bases per recording × {len(args.rank)} ranks; '
              f'{len(references)} total bases, scope={args.basis_scope}, source rank {args.source_rank}. '+
              ('Quadratic and exact energies.' if args.exact else 'Fast quadratic means only.')+
              ' Completed recordings, trajectory batches and source moments are cached.', flush=True)
        scopes = powers if args.basis_scope == 'per-power' else [None]
        completed = 0
        for scope in scopes:
            scoped_references = [r for r in references if scope is None or r['power'] == scope]
            scoped_chosen = [chosen[r['reference_id']-1] for r in scoped_references]
            scoped_records = [(p, r) for p, r in records if scope is None or p == scope]
            prepared = prepare_references(cache, scoped_references, scoped_chosen, signature,
                                          args, diagnostics, len(references))
            for index, (power, record) in enumerate(scoped_records, start=completed):
                tick = time.perf_counter()
                print(f'[{index+1}/{len(records)}] {power}/rep{record.number}: starting', flush=True)
                if read_signature(record.pod_path) != signature:
                    raise ValueError(f'Inconsistent units/preprocessing: {record.pod_path}')
                key = f'recordings/{power}_rep{record.number}'
                mark = json.dumps(fingerprint(record.pod_path), sort_keys=True)
                if key in cache and cache[key].attrs.get('source_fingerprint') == mark and cache[key].attrs.get('complete', False):
                    group = cache[key]
                    geometric_means, captures = group['geometric_mean_energy'][:], group['reference_subspace_capture'][:]
                    action = 'cached energy'
                else:
                    moment, metadata, moment_action = cached_second_moment(moments, power, record, signature, args)
                    source, x, y = load_basis(record.pod_path, args.source_rank)
                    _check_grid(*prepared[0]['grid'], x, y, str(record.pod_path))
                    geometric_means, captures = [], []
                    for ref in prepared:
                        overlap = ref['basis'] @ source.T
                        geometric_means.append(mean_energies(moment, overlap, ref['matrix'], args.rank))
                        captures.append([float(np.sum(overlap[:r]**2)/r) for r in args.rank])
                    del source, moment, overlap
                    geometric_means, captures = np.asarray(geometric_means), np.asarray(captures)
                    if key in cache:
                        del cache[key]
                    group = cache.create_group(key)
                    group.create_dataset('geometric_mean_energy', data=geometric_means).attrs['units'] = 'm^2' if args.geometry=='surface' else 'm'
                    group.create_dataset('reference_subspace_capture', data=captures)
                    for name, value in metadata.items():
                        group.attrs[name] = value
                    group.attrs['source_fingerprint'] = mark
                    group.attrs['complete'] = True
                    cache.flush()
                    action = f'computed energy; {moment_action} source moment'
                means = cached_energy_trajectories(group, record.pod_path, prepared, signature, args) if args.exact else dict(quad=geometric_means)
                for observable in means:
                    name = f'Ebar_{observable}'
                    if name in group:
                        del group[name]
                    ds = group.create_dataset(name, data=args.gamma*means[observable])
                    ds.attrs['units'] = unit
                    ds.attrs['axis_order'] = 'reference, rank'
                if 'mean_energy' in group:
                    del group['mean_energy']
                group.create_dataset('mean_energy', data=args.gamma*geometric_means).attrs['units'] = unit
                if 'reference_ids' not in group:
                    group.create_dataset('reference_ids', data=[r['reference_id'] for r in scoped_references])
                group['mean_energy'].attrs['axis_order'] = 'reference, rank'
                group['geometric_mean_energy'].attrs['axis_order'] = 'reference, rank'
                for bi, ref in enumerate(scoped_references):
                    for ri, rank in enumerate(args.rank):
                        rows.append(dict(reference_id=ref['reference_id'], reference_power=ref['power'],
                                         reference_slot=ref.get('reference_slot', ref['reference_id']),
                                         reference_repetition=ref['repetition'], power=power, repetition=record.number, rank=rank,
                                         time_averaged_energy=float(args.gamma*geometric_means[bi, ri]),
                                         Ebar_quad=float(args.gamma*means['quad'][bi, ri]),
                                         **(dict(Ebar_exact=float(args.gamma*means['exact'][bi, ri])) if args.exact else {}),
                                         **(dict(exact_evaluated_frames=min(args.exact_frames or int(group.attrs['samples']), int(group.attrs['samples'])),
                                                 quadratic_sampling_relative_error=float(means['quad'][bi, ri]/geometric_means[bi, ri]-1) if geometric_means[bi, ri] else float('nan')) if args.exact else {}),
                                         rms_spatial_slope=float(np.sqrt(2*geometric_means[bi, ri]/prepared[bi]['measure'])),
                                         reference_subspace_capture=float(captures[bi, ri]),
                                         samples=int(group.attrs['samples']), duration_seconds=float(group.attrs['duration_seconds']),
                                         units=unit, source_file=str(record.pod_path)))
                cache.flush()
                print(f'[{index+1}/{len(records)}] {action}; {time.perf_counter()-tick:.1f} s '
                      f'(total {(time.perf_counter()-started)/60:.1f} min)', flush=True)
            completed += len(scoped_records)
            del prepared
    write_study_report(out, rows, diagnostics, references, settings)
    print(f'Saved {out/"capillary_energy_report.pdf"}', flush=True)


if __name__ == '__main__':
    run(parse_args())
