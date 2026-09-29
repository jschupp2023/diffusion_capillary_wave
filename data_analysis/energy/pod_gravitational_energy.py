"""Sample POD gravitational energy for selected power, rank, and repetitions.

Example: python -m data_analysis.energy.pod_gravitational_energy 0p20 --rank 50 --reps 1 2
Uses E = (rho*g/2)*integral(eta**2)dA.
Frames are sampled once per equal-size time-index bin; reported means average those frames.
Optionally, the full spatial-mean trajectory can be high-pass filtered before
the sampled values are restored to the POD reconstruction.
"""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt

from data_analysis.energy.capillary_energy import length_scale, quadrature_weights


def highpass_spatial_mean(values, time, cutoff_hz):
    """Zero-phase, fourth-order Butterworth high-pass of a full mean trajectory."""
    values, time = np.asarray(values, dtype=float), np.asarray(time, dtype=float)
    if values.ndim != 1 or time.ndim != 1 or values.shape != time.shape or len(time) < 2:
        raise ValueError('Spatial mean and time must be matching one-dimensional trajectories')
    if not np.isfinite(values).all() or not np.isfinite(time).all():
        raise ValueError('Spatial mean and time must be finite')
    steps = np.diff(time)
    mean_step = float(np.mean(steps))
    if mean_step <= 0 or not np.allclose(steps, mean_step, rtol=.02, atol=0):
        raise ValueError('High-pass filtering requires a uniformly sampled, increasing time grid')
    sampling_hz = 1/mean_step
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0 or cutoff_hz >= .5*sampling_hz:
        raise ValueError(f'High-pass cutoff must lie between 0 and Nyquist ({.5*sampling_hz:.6g} Hz)')
    sos = butter(4, cutoff_hz, btype='highpass', fs=sampling_hz, output='sos')
    try:
        return sosfiltfilt(sos, values)
    except ValueError as exc:
        raise ValueError(f'Spatial-mean trajectory is too short for zero-phase filtering: {exc}') from exc


def compute(path, rank, frames, seed, rho, gravity, spatial_mean_highpass_hz=None):
    with h5py.File(path) as f:
        modes = f['pod/modes']
        coeff = f['reduced/coefficients']
        time = np.asarray(f['grid/time'])
        if rank < 1 or modes.shape[0] < rank or coeff.shape != (len(time), modes.shape[0]):
            raise ValueError(f'Invalid rank or POD shapes in {path}')
        if frames > len(time):
            raise ValueError(f'Requested {frames} frames from only {len(time)} in {path}')
        if not f.attrs['instantaneous_spatial_mean_removed']:
            raise ValueError(f'Spatial mean was not removed from the POD in {path}')
        if not f.attrs['temporal_mean_field_removed']:
            raise ValueError(f'Temporal mean was not removed from the POD in {path}')
        x, y = (np.asarray(f[f'grid/{axis}'])*length_scale(f[f'grid/{axis}'].attrs['units'])
                for axis in ('x', 'y'))
        weight = np.outer(quadrature_weights(y), quadrature_weights(x)).ravel()
        height_unit = f['reduced/coefficients'].attrs['units']
        scale = length_scale(height_unit)
        for name in ('preprocessing/frame_spatial_mean', 'preprocessing/temporal_mean_field'):
            if f[name].attrs['units'] != height_unit or not f[name].attrs['subtracted_before_pod']:
                raise ValueError(f'Inconsistent preprocessing metadata for {name} in {path}')
        mean_field = np.asarray(f['preprocessing/temporal_mean_field'], dtype=float).ravel()*scale
        spatial_mean = np.asarray(f['preprocessing/frame_spatial_mean'], dtype=float)
        if not np.isfinite(spatial_mean).all():
            raise ValueError(f'Nonfinite spatial mean in {path}')
        if spatial_mean_highpass_hz is not None:
            spatial_mean = highpass_spatial_mean(spatial_mean, time, spatial_mean_highpass_hz)
        spatial_mean *= scale
        basis = np.asarray(modes[:rank], dtype=float).reshape(rank, -1)
        if basis.shape[1] != len(weight) or mean_field.shape != weight.shape:
            raise ValueError(f'Spatial grid and POD shape differ in {path}')

        # One random frame in each equal-size time-index bin covers the recording.
        edges = np.linspace(0, len(time), frames+1, dtype=int)
        rng = np.random.default_rng(seed)
        indices = np.array([rng.integers(a, b) for a, b in zip(edges[:-1], edges[1:])])
        without, with_mean = np.empty(frames), np.empty(frames)
        prefactor = .5*rho*gravity
        for start in range(0, frames, 16):
            stop = min(start+16, frames)
            c = np.asarray(coeff[indices[start:stop], :rank], dtype=float)*scale
            field = c @ basis + mean_field
            if not np.isfinite(field).all():
                raise ValueError(f'Nonfinite reconstruction in {path}')
            unscaled = (field*field) @ weight
            selected_mean = spatial_mean[indices[start:stop]]
            without[start:stop] = prefactor*unscaled
            with_mean[start:stop] = prefactor*(unscaled + 2*selected_mean*(field @ weight)
                                               + selected_mean**2*weight.sum())
        return time[indices], indices, without, with_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('power', help='Experiment folder, e.g. 0p20')
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--reps', type=int, nargs='+', required=True)
    parser.add_argument('--experiment', help='Recording stem, e.g. Ca_ac_0p001660; inferred if unique')
    parser.add_argument('--pod-rank', type=int, default=1000)
    parser.add_argument('--frames', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--rho', type=float, default=998., help='Fluid density [kg/m^3]')
    parser.add_argument('--gravity', type=float, default=9.81, help='Gravitational acceleration [m/s^2]')
    parser.add_argument(
        '--spatial-mean-highpass-hz', type=float,
        help='High-pass the full spatial-mean trajectory at this cutoff [Hz] using a fourth-order '
             'Butterworth filter applied forward-backward; default: no filtering.',
    )
    parser.add_argument('--data-root', type=Path, default=Path('/home/jonas/ucsd_thesis/reduced_data'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/gravitational_energy'))
    args = parser.parse_args()
    if args.rank < 1 or args.pod_rank < args.rank or args.frames < 1 or args.seed < 0 or not args.reps or any(r < 1 for r in args.reps):
        parser.error('Ranks, frames, and repetitions must be positive; seed must be nonnegative.')
    if not np.isfinite([args.rho, args.gravity]).all() or min(args.rho, args.gravity) <= 0:
        parser.error('Density and gravity must be finite and positive.')
    if (args.spatial_mean_highpass_hz is not None
            and (not np.isfinite(args.spatial_mean_highpass_hz) or args.spatial_mean_highpass_hz <= 0)):
        parser.error('--spatial-mean-highpass-hz must be finite and positive.')

    paths = []
    for rep in sorted(set(args.reps)):
        pattern = f'{args.experiment or "*"}_rep{rep}'
        matches = list((args.data_root/args.power).glob(f'{pattern}/pod_2d_r{args.pod_rank}.h5'))
        if len(matches) != 1:
            parser.error(f'Expected one POD file for {pattern} in {args.power}; found {len(matches)}')
        paths.append((rep, matches[0]))

    experiments = {path.parent.name.rsplit('_rep', 1)[0] for _, path in paths}
    if len(experiments) != 1:
        parser.error('Selected repetitions belong to different experiments; set --experiment')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.power}_{experiments.pop()}_r{args.rank}_n{args.frames}_seed{args.seed}_reps_{"_".join(map(str, sorted(set(args.reps))))}'
    if args.spatial_mean_highpass_hz is not None:
        cutoff_tag = f'{args.spatial_mean_highpass_hz:.12g}'.replace('.', 'p').replace('+', '')
        stem += f'_mean_hp{cutoff_tag}Hz'
    summary, samples = [], []
    for rep, path in paths:
        time, indices, without, with_mean = compute(
            path, args.rank, args.frames, args.seed, args.rho, args.gravity,
            args.spatial_mean_highpass_hz,
        )
        summary.append((args.power, path.parent.name, rep, args.rank, args.frames, args.seed, args.rho, args.gravity,
                        args.spatial_mean_highpass_hz,
                        float(without.mean()), float(with_mean.mean())))
        samples.extend((args.power, path.parent.name, rep, args.rank, args.spatial_mean_highpass_hz,
                        int(i), float(t), float(e0), float(e1))
                       for i, t, e0, e1 in zip(indices, time, without, with_mean))
        mean_label = ('with spatial mean' if args.spatial_mean_highpass_hz is None else
                      f'with {args.spatial_mean_highpass_hz:g} Hz high-pass spatial mean')
        print(f'{path.parent.name}: without spatial mean {without.mean():.6g} J; '
              f'{mean_label} {with_mean.mean():.6g} J')
    for name, header, rows in (
        ('summary', ('power', 'experiment', 'rep', 'rank', 'frames', 'seed', 'rho_kg_m3', 'gravity_m_s2',
                     'spatial_mean_highpass_hz',
                     'mean_without_spatial_mean_J', 'mean_with_spatial_mean_J'), summary),
        ('samples', ('power', 'experiment', 'rep', 'rank', 'spatial_mean_highpass_hz', 'frame_index',
                     'time_s', 'without_spatial_mean_J', 'with_spatial_mean_J'), samples),
    ):
        path = args.output_dir/f'{stem}_{name}.csv'
        with path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
