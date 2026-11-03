"""Train-only valley interpolation of a positive-semidefinite spectral background.

Uses saved spectral matrices; no stochastic process, oscillator fit or neural training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def select_peak_centers(f, spectrum, prescribed, coordinates=(0, 1, 2), prominence_db=2.5):
    """Mask the known peaks plus clear additional training-only marginal peaks.

    Nearby detections are merged within 12 Hz, keeping the largest prominence.
    Prescribed peaks are snapped to a detected peak within 12 Hz where possible.
    """
    diagonal = np.diagonal(spectrum, axis1=-2, axis2=-1).real
    candidates = []
    for coordinate in coordinates:
        indices, properties = find_peaks(10*np.log10(diagonal[:, coordinate].clip(1e-30)),
                                         prominence=prominence_db,
                                         distance=max(1, round(15/np.median(np.diff(f)))))
        candidates.extend(dict(frequency_hz=float(f[k]), prominence_db=float(pr), coordinate=int(coordinate))
                          for k, pr in zip(indices, properties['prominences']))
    merged = []
    for row in sorted(candidates, key=lambda r: r['prominence_db'], reverse=True):
        if all(abs(row['frequency_hz']-r['frequency_hz']) > 12 for r in merged):
            merged.append(row)
    centers = [row['frequency_hz'] for row in merged]
    for peak in prescribed:
        if not any(abs(peak-center) <= 12 for center in centers):
            centers.append(float(peak))
    return np.array(sorted(centers)), candidates


def valley_background(f, spectrum, peaks, half_width_hz, coordinates=(0, 1, 2, 3),
                      neighborhood_bins=3, max_segment_hz=80.):
    """Convex interpolation of full matrix averages at common valley anchors.

    Only windows entirely outside all peak masks are eligible. Endpoint extension
    is constant, explicitly flagged unsupported. No entrywise clipping/subtraction.
    """
    f, spectrum, peaks = np.asarray(f), np.asarray(spectrum), np.asarray(peaks)
    if (len(f) < 3 or np.any(np.diff(f) <= 0) or spectrum.ndim != 3
            or spectrum.shape[0] != len(f) or spectrum.shape[1] != spectrum.shape[2]
            or not np.isfinite(spectrum).all() or half_width_hz <= 0
            or neighborhood_bins < 1 or neighborhood_bins % 2 != 1 or max_segment_hz <= 0):
        raise ValueError('Invalid spectral grid, matrix array or anchor configuration')
    masked = (abs(f[:, None]-peaks[None]) <= half_width_hz).any(axis=1)
    eligible = ~masked
    diagonal = np.diagonal(spectrum, axis1=-2, axis2=-1).real
    # Geometric-mean marginal power; scale changes add constants and do not move minima.
    score = np.log(diagonal[:, list(coordinates)].clip(1e-30)).mean(axis=1)
    half = neighborhood_bins//2
    transitions = np.diff(np.r_[False, eligible, False].astype(int))
    segments = list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))
    centers, windows = [], []
    for start, end in segments:
        if end-start < neighborhood_bins:
            continue
        # Long peak-free intervals get multiple local anchors, not one remote endpoint.
        possible = np.arange(start+half, end-half)
        chunks = np.floor((f[possible]-f[possible[0]])/max_segment_hz).astype(int)
        for chunk in np.unique(chunks):
            candidates = possible[chunks == chunk]
            values = [score[i-half:i+half+1].mean() for i in candidates]
            center = int(candidates[np.argmin(values)])
            centers.append(center); windows.append(np.arange(center-half, center+half+1))
    if len(centers) < 2:
        raise ValueError('Fewer than two usable valley anchors; peak mask is too dense')
    centers = np.asarray(centers)
    anchor_f = np.array([f[w].mean() for w in windows])
    anchor_s = np.array([spectrum[w].mean(axis=0) for w in windows])
    right = np.searchsorted(anchor_f, f, side='right').clip(1, len(anchor_f)-1)
    left = right-1
    alpha = ((f-anchor_f[left])/(anchor_f[right]-anchor_f[left])).clip(0, 1)
    background = (1-alpha[:, None, None])*anchor_s[left]+alpha[:, None, None]*anchor_s[right]
    supported = (f >= anchor_f[0]) & (f <= anchor_f[-1])
    # Also expose local support gaps; interpolation is weakest over long masked regions.
    gap = anchor_f[right]-anchor_f[left]
    return dict(background=background, anchor_indices=centers, anchor_frequencies=anchor_f,
                anchor_matrices=anchor_s, anchor_windows=np.stack(windows), masked=masked,
                supported=supported, interpolation_gap_hz=gap)


def _diagonal(s):
    return np.diagonal(s, axis1=-2, axis2=-1).real.clip(1e-30)


def describe(f, spectrum, result, peaks):
    s, b = _diagonal(spectrum), _diagonal(result['background'])
    db = 10*np.log10(b/s)
    off = ~result['masked'] & result['supported']
    eigenvalues = np.linalg.eigvalsh(spectrum-result['background'])
    negative = -np.minimum(eigenvalues, 0).sum(axis=1)
    total = abs(eigenvalues).sum(axis=1).clip(1e-30)
    peak_indices = np.array([np.argmin(abs(f-p)) for p in peaks])
    peak_indices = peak_indices[result['supported'][peak_indices]]
    return dict(anchor_frequencies_hz=result['anchor_frequencies'].tolist(),
        anchor_count=len(result['anchor_indices']),
        supported_interval_hz=[float(result['anchor_frequencies'][0]),float(result['anchor_frequencies'][-1])],
        maximum_anchor_gap_hz=float(np.diff(result['anchor_frequencies']).max()),
        minimum_background_eigenvalue=float(np.linalg.eigvalsh(result['background']).min()),
        per_coordinate=[dict(coordinate=j,
            median_background_minus_train_db_offpeak=float(np.median(db[off, j])) if off.any() else None,
            fraction_supported_bins_background_above_train_by_1db=float(np.mean(db[result['supported'], j] > 1)),
            median_excess_db_at_supported_common_peak_centers=float(np.median(-db[peak_indices, j])) if len(peak_indices) else None,
            background_power_fraction_supported=float(b[result['supported'], j].sum()/s[result['supported'], j].sum())) for j in range(s.shape[1])],
        median_negative_fraction_of_signed_residual_eigenvalue_mass_supported=float(np.median((negative/total)[result['supported']])),
        residual_is_psd_at_every_bin=bool(np.all(eigenvalues >= -1e-12)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/diagnostics/observed_oscillator_conditioning_0p10_v1/full_family'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--half-widths', nargs='+', type=float, default=[10., 18.])
    args = parser.parse_args(argv)
    if len(args.half_widths) != 2 or len(set(args.half_widths)) != 2:
        parser.error('Choose two distinct exclusion half-widths')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty')
    with np.load(args.source/'reference_spectra.npz') as z:
        f, train, validation = z['frequency_hz'], z['train'], z['validation']
        validation_records, mean, scale = z['per_validation_record'], z['mean'], z['scale']
    source_protocol = json.loads((args.source/'protocol.json').read_text())
    peaks, candidates = select_peak_centers(f, train, source_protocol['prescribed_peak_frequencies_hz'])
    estimates = [valley_background(f, train, peaks, width) for width in args.half_widths]
    summaries = [dict(half_width_hz=width, **describe(f, train, result, peaks))
                 for width, result in zip(args.half_widths, estimates)]
    supported = estimates[0]['supported'] & estimates[1]['supported']
    difference = 10*np.log10(_diagonal(estimates[1]['background'])/_diagonal(estimates[0]['background']))
    sensitivity = [dict(coordinate=j, median_absolute_difference_db=float(np.median(abs(difference[supported, j]))),
        p90_absolute_difference_db=float(np.quantile(abs(difference[supported, j]), .9)),
        maximum_absolute_difference_db=float(abs(difference[supported, j]).max())) for j in range(train.shape[1])]
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = dict(frequency_hz=f, train=train, validation=validation,
                  per_validation_record=validation_records, mean=mean, scale=scale, peak_centers_hz=peaks)
    for i, result in enumerate(estimates):
        arrays.update({f'{key}_{i}': value for key, value in result.items()})
    np.savez_compressed(args.output/'backgrounds.npz', **arrays)
    protocol = dict(source=str(args.source.resolve()), source_protocol=source_protocol,
        source_spectra_sha256=hashlib.sha256((args.source/'reference_spectra.npz').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), half_widths_hz=args.half_widths,
        training_only=True, anchor_score_coordinates=[0,1,2,3], additional_peak_coordinates=[0,1,2],
        additional_peak_prominence_db=2.5, merge_distance_hz=12, anchor_neighborhood_bins=3,
        maximum_peak_free_segment_hz=80, peak_centers_hz=peaks.tolist(), detected_training_peaks=candidates,
        selection='All known peaks plus train-only marginal peaks; 12 Hz merging. Three-bin matrix averages at minima of mean log marginal power in common unmasked intervals.',
        interpolation='Linear in frequency and power, common convex weights for all matrix entries. Constant extension outside anchors is marked unsupported.',
        limitations=['Background is a smooth nuisance estimate, not identified physical noise.',
            'Valleys can contain resonance tails; minimum selection can also bias noisy backgrounds downward.',
            'Common spatial anchors need not be valleys in every coordinate.',
            'Background PSD is positive semidefinite; data minus background need not be.',
            'No validation data enter peak selection, anchors or interpolation. Two held-out records are shown only for comparison.',
            'Inherited preprocessing and shared-basis limitations are in source_protocol. No oscillator or neural model was fitted.'])
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(dict(estimates=summaries,sensitivity=sensitivity), indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for zoom in [False, True]:
        fig, axes = plt.subplots(2,2,figsize=(12,7), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(f, 10*np.log10(_diagonal(validation)[:,j]), color='.7', lw=.9, label='validation (not fitted)')
            ax.plot(f, 10*np.log10(_diagonal(train)[:,j]), color='k', lw=1.1, label='training')
            for i, (width, result) in enumerate(zip(args.half_widths, estimates)):
                color = ['#0072B2','#D55E00'][i]
                power = 10*np.log10(_diagonal(result['background'])[:,j])
                support = result['supported']
                ax.plot(f, np.where(support, power, np.nan), color=color, lw=1.8,label=f'background: exclude ±{width:g} Hz')
                ax.plot(f, np.where(~support, power, np.nan), color=color, lw=1.4,ls=':')
                ax.scatter(result['anchor_frequencies'], 10*np.log10(_diagonal(result['anchor_matrices'])[:,j]),
                           color=color,s=13,zorder=5)
            for peak in peaks:
                ax.axvspan(peak-args.half_widths[0],peak+args.half_widths[0],color='.8',alpha=.12)
            ax.set_title('Spatial mean' if j == 0 else f'POD {j}')
            ax.set_ylabel('PSD [dB, standardized units/Hz]'); ax.set_xlabel('Frequency [Hz]')
            ax.set_xlim((80,650) if zoom else (f[0],f[-1]))
        axes[0,0].legend(fontsize=7)
        fig.suptitle('Training-only valley background · dots: matrix-average anchors · dotted ends: unsupported')
        fig.tight_layout();fig.savefig(args.output/('background_zoom.png' if zoom else 'background_comparison.png'),dpi=160);plt.close(fig)
    lines=['# Training-only spectral background diagnostic','',
        '[Full PSD overlay](background_comparison.png) · [80–650 Hz detail](background_zoom.png)','',
        'No oscillator fit, neural training or checkpoint modification. Backgrounds use common frequency anchors for the entire 21×21 complex spectral matrix. Validation is displayed only.', '',
        '| Excluded half-width [Hz] | Anchors | Bracketed interval [Hz] | Largest anchor gap [Hz] |',
        '|---:|---:|---:|---:|']
    for row in summaries:
        lo,hi=row['supported_interval_hz'];lines.append(f"| {row['half_width_hz']:g} | {row['anchor_count']} | {lo:.1f}–{hi:.1f} | {row['maximum_anchor_gap_hz']:.1f} |")
    lines += ['', 'Sensitivity between exclusion widths, over their common bracketed interval:', '',
              '| Coordinate | Median absolute change [dB] | 90th percentile [dB] |', '|---|---:|---:|']
    for row in sensitivity[:4]:
        lines.append(f"| {'Spatial mean' if row['coordinate']==0 else 'POD '+str(row['coordinate'])} | {row['median_absolute_difference_db']:.2f} | {row['p90_absolute_difference_db']:.2f} |")
    lines += ['', 'Dots average three adjacent bins (about 10.55 Hz total bin bandwidth). All averaging bins must lie outside the peak masks. Long peak-free intervals receive additional anchors. Shading shows the narrower exclusion windows; dotted background ends are unsupported constant extensions.', '',
        'Convex interpolation preserves Hermitian positive semidefiniteness, but does not guarantee a matrix lower bound on the measured spectrum. Signed data-minus-background must not be treated as a valid covariance without further modeling.', '',
        'These are heuristic smooth floors, not a unique physical separation. Broad resonance tails can lift the floor; selecting valleys can bias it downward. Common anchors can be imperfect for individual coordinates. Dense peak regions and long anchor gaps are the least constrained.', '',
        'Numerical matrices and masks: `backgrounds.npz`. Per-coordinate diagnostics: `summary.json`. Exact train-only selection, source hashes, scaling, split and preprocessing: `protocol.json`.', '',
        'Reproduce from the repository root:', '', '```bash',
        'MPLCONFIGDIR=/tmp/cwt-mpl /home/jonas/miniforge3/envs/diff-model-mit/bin/python \\',
        '  -m modelling.latent_oscillators.spectral_background \\',
        '  --output runs/diagnostics/spectral_background_0p10_repeat', '```']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(estimates=[{k:r[k] for k in ['half_width_hz','anchor_count','supported_interval_hz','maximum_anchor_gap_hz']} for r in summaries],sensitivity=sensitivity[:4]),indent=2))


if __name__ == '__main__':
    main()
