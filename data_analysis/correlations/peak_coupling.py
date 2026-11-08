"""Repetition-resolved dependence diagnostics for the saved frequency hybrid.

Run from the repository with the diff-model-mit environment. Source HDF5 and
model artifacts are read only. The extraction cache contains physical complex
coefficients, not inferred hidden states. See the generated report for the
statistical protocol, null hypotheses, and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, lfilter, resample_poly, sosfiltfilt

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.frequency_hybrid.local_excess import projected_band
from modelling.frequency_hybrid.local_excess_remainder import load_bank


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / 'best_models/0p10/unified_compact'
DEFAULT_OUTPUT = ROOT / 'runs/diagnostics/unified_compact_peak_coupling_v1'
BANK_SCALES = [1., .65, .90, .85, 1.05, .80, .90, .75, 1.15, .80, 1.10]
OVERLAP = .0578


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def coarse(values, factor):
    # Extra FIR guard after the native filter/Hilbert guard, for every series.
    return resample_poly(values, 1, factor, axis=0, padtype='line')[10:-10]


def filters(models, fs, width=1.):
    result = []
    for model in models:
        low, high = model['edges_hz']
        center = (low + high) / 2
        half = (high - low) * width / 2
        result.append(butter(int(model['filter_order']), [center-half, center+half],
                             btype='bandpass', fs=fs, output='sos'))
    return result


def extract(values, models, fs, edge, *, modal=False):
    """Use the original full-record native filter and complex projection."""
    coefficients, narrow, modal_envelopes = [], [], []
    bank = np.zeros_like(values[edge:-edge])
    for model, sos, sos_narrow in zip(models, filters(models, fs), filters(models, fs, .8)):
        component, coefficient = projected_band(values, sos, model['modes'], edge)
        bank += component
        coefficients.append(coarse(coefficient[:, 0], 32))
        _, small = projected_band(values, sos_narrow, model['modes'], edge)
        narrow.append(coarse(small[:, 0], 32))
        if modal:
            analytic = hilbert(sosfiltfilt(sos, values, axis=0, padlen=edge), axis=0)
            modal_envelopes.append(coarse(np.abs(analytic[edge:-edge]), 128))
    coefficients = np.stack(coefficients, axis=-1)
    narrow = np.stack(narrow, axis=-1)
    # Defined on POD coordinates only; an activity proxy, not physical energy.
    high = butter(4, 1200., btype='highpass', fs=fs, output='sos')
    broadband = sosfiltfilt(high, values[:, 1:], axis=0, padlen=edge)[edge:-edge]
    activity = coarse(np.sum(broadband**2, axis=1), 128)
    result = dict(coefficients=coefficients, narrow_coefficients=narrow,
                  amplitude=coarse(np.abs(coefficients), 4),
                  power=coarse(np.abs(coefficients)**2, 4),
                  narrow_amplitude=coarse(np.abs(narrow), 4),
                  broadband_activity=activity,
                  native_retained_frames=len(bank))
    if modal:
        result['modal_amplitude'] = np.stack(modal_envelopes, axis=-1)
    return result


def simulate_bank(models, length, paths, seed, fs, edge):
    """Exact stationary complex OU recurrence, vectorized with scipy lfilter.

    Same stochastic law as simulate_complex_band, with independent peaks and
    paths. The archived rollout's bank scales and overlap are applied here.
    """
    output = np.zeros((length, paths, len(models[0]['modes'])))
    for band, (model, scale) in enumerate(zip(models, BANK_SCALES)):
        rng = np.random.default_rng(seed + band)
        alpha = np.exp((-float(model['gamma_per_second']) +
                        2j*np.pi*float(model['frequency_hz'])) / fs)
        factor = complex(model['coefficient_factor'].item()) * scale
        noise = (rng.normal(size=(length+2*edge, paths)) +
                 1j*rng.normal(size=(length+2*edge, paths))) / np.sqrt(2)
        noise[1:] *= np.sqrt(1-abs(alpha)**2)
        state = lfilter([1.], [1., -alpha], noise * factor, axis=0)
        low, high = model['edges_hz']; width = high-low
        sos = butter(int(model['filter_order']),
                     [max(10., low-OVERLAP*width), high+OVERLAP*width],
                     btype='bandpass', fs=fs, output='sos')
        state = sosfiltfilt(sos, state, axis=0, padlen=edge)[edge:-edge]
        output += (state[..., None] * model['modes'][:, 0]).real
    return output


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'coefficients'; cache.mkdir(exist_ok=True)
    config = json.loads((args.model/'checkpoint/config.json').read_text())
    manifest_path = Path(config['data']['temporal_component_manifest'])
    manifest = json.loads(manifest_path.read_text())
    artifacts, models = load_bank(args.model/'oscillators')
    if [digest(p) for p in artifacts] != manifest['local_excess']['artifact_sha256']:
        raise ValueError('Bank differs from original decomposition')
    data = SharedPODTrainingData(**manifest['parent_config'], output_dtype='float64')
    signature = data.signature()
    if signature != manifest['parent_signature']:
        raise ValueError('Current source signature differs from model preparation')
    fs, edge = float(manifest['fs']), int(manifest['edge_samples'])
    identity = dict(schema=1, source_signature=signature,
                    bank_sha256=[digest(p) for p in artifacts],
                    extraction='original native filters and projection; resample32/4 with 10-sample guards',
                    extraction_code_sha256=digest(__file__), seed=args.seed)
    identity_path = cache/'identity.json'
    # Source identity is strict. Analysis-only edits do not invalidate extracted data.
    if identity_path.exists():
        old = json.loads(identity_path.read_text())
        for key in ('schema', 'source_signature', 'bank_sha256', 'extraction', 'seed'):
            if old[key] != identity[key]:
                raise ValueError(f'Cache identity mismatch: {key}')
    else:
        write_json(identity_path, identity)
    for split, names in data.splits.items():
        for name in names:
            path = cache/f'{name}.npz'
            if path.exists():
                continue
            print(f'Extract {split}: {name}', flush=True)
            time, values = data.read_coordinates(name, 0, data.frame_counts[name])
            result = extract(values, models, fs, edge, modal=True)
            np.savez_compressed(path, **result, split=split, fs=fs,
                                native_frames=len(values), time_start=time[edge],
                                time_stop=time[-edge-1])
    if not (cache/'independent_bank_15.npz').exists():
        print('Simulate 16 independent calibrated bank paths', flush=True)
        generated = simulate_bank(models, 100001, 16, args.seed+10000, fs, edge)
        for index in range(16):
            path = cache/f'independent_bank_{index:02d}.npz'
            if not path.exists():
                print(f'Extract independent bank {index+1}/16', flush=True)
                np.savez_compressed(path, **extract(generated[:, index], models, fs, edge))
    # These short saved trajectories are descriptive only, with identical guards.
    with np.load(args.model/'evaluation/rollout.npz') as saved:
        for i, name in enumerate(saved['repetition'].astype(str)):
            for kind, values in [('saved_reference', saved['reference'][i]),
                                 ('saved_hybrid', saved['trajectories'][i, 0])]:
                path = cache/f'{kind}_{name}.npz'
                if not path.exists():
                    np.savez_compressed(path, **extract(values, models, fs, edge))
    meta = dict(model=str(args.model.resolve()), manifest=str(manifest_path),
                source_signature_matches=True, native_fs=fs, edge_samples=edge,
                coefficient_fs=fs/32, envelope_fs=fs/128,
                frequencies_hz=[float(m['frequency_hz']) for m in models],
                linewidths_hz=[float(m['gamma_per_second'])/np.pi for m in models],
                bands_hz=[m['edges_hz'].tolist() for m in models], splits=data.splits,
                train_trim_percent=config['training']['train_trim_percent'],
                bank_scales=BANK_SCALES, bank_overlap=OVERLAP,
                original_training_source='legacy shared POD, separate spatial mean highpass 50 Hz',
                heldout_caveat='shared POD basis was constructed using held-out repetitions too')
    write_json(args.output/'metadata.json', meta)
    return meta


def simulate_hybrid(args):
    """Longer autonomous comparison using the archived rollout calibration."""
    import torch
    from modelling.neural_sde.train import load_model
    from scripts.diagnostics.evaluate_unified_simplified_hybrid import (
        make_highshelf, run_boosted_shaped_rollout,
    )
    cache = args.output/'coefficients'; cache.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.model/'checkpoint/config.json').read_text())
    manifest_path = Path(config['data']['temporal_component_manifest'])
    manifest = json.loads(manifest_path.read_text())
    fs, edge = float(manifest['fs']), int(manifest['edge_samples'])
    _, models = load_bank(args.model/'oscillators')
    output_file = args.output/'long_hybrid_ensemble.npz'
    if output_file.exists():
        with np.load(output_file) as z:
            hybrid = z['trajectories']
    else:
        model, _ = load_model(str(args.model/'checkpoint/best.pt'), 'cpu')
        model.eval()
        initial, history, conditions = [], [], []
        for rep in (13, 14):
            name = f'Ca_ac_0p000830_rep{rep}'
            rem = np.load(manifest_path.parent/manifest['records'][name]['files']['remainder']['path'])
            for index in np.linspace(1000, 50000, 8).astype(int):
                initial.append(rem[index]); history.append(rem[index-63:index][::-1].copy())
                conditions.append(dict(repetition=name, index=int(index)))
        initial = torch.tensor(np.stack(initial), dtype=torch.float32)
        history = torch.tensor(np.stack(history), dtype=torch.float32)
        b, a = make_highshelf(750., 4000., fs)
        scales = np.array([1., .72, .69, 1.70, 1.30, 1.40, 1.05, 1.05, 1.05, 1.05, 1.05])
        print('Simulate 16 full hybrid paths: 150000 steps, discard 50000-step burnin', flush=True)
        with torch.no_grad():
            rem = run_boosted_shaped_rollout(model, initial, history, b, a, scales,
                                            horizon=150000, seed=args.seed+30000, device='cpu')
        remainder = rem[:, 0, 50000:].numpy().astype(np.float64)
        if not np.isfinite(remainder).all():
            raise ValueError('Nonfinite autonomous remainder')
        del rem
        bank = simulate_bank(models, remainder.shape[1], 16, args.seed+40000, fs, edge)
        hybrid = remainder + bank.transpose(1, 0, 2)
        np.savez_compressed(output_file, trajectories=hybrid, fs=fs)
        write_json(args.output/'long_hybrid_metadata.json', dict(
            checkpoint=str((args.model/'checkpoint/best.pt').resolve()),
            checkpoint_sha256=digest(args.model/'checkpoint/best.pt'),
            conditions=conditions, seed=args.seed+30000, bank_seed=args.seed+40000,
            burnin_steps=50000, retained_steps=100001, paths=16, device='cpu',
            mean_noise_shelf_hz=[750., 4000.], remainder_noise_scales=scales.tolist(),
            bank_scales=BANK_SCALES, bank_overlap=OVERLAP,
            history_order='newest first; 63 preceding measured remainder states',
            note='Fresh autonomous ensemble with archived rollout calibration; not an exact replay of archived seeds'))
    for i, values in enumerate(hybrid):
        path = cache/f'full_hybrid_{i:02d}.npz'
        if not path.exists():
            print(f'Extract full hybrid {i+1}/16', flush=True)
            np.savez_compressed(path, **extract(values, models, fs, edge))


def standardized(x):
    x = np.asarray(x, float)
    x = x-x.mean(axis=0)
    scale = np.sqrt(np.mean(x*x, axis=0))
    if np.any(scale <= 0):
        raise ValueError('Constant envelope has undefined correlation')
    return x/scale


def pair_correlations(x, pairs):
    z = standardized(x)
    return np.mean(z[:, pairs[:, 0]]*z[:, pairs[:, 1]], axis=0)


def shifted_null(x, pairs, draws, rng, fs, min_shift=.1):
    """Pairwise circular-shift null; Holm does not assume joint independence."""
    z = standardized(x); n = len(z)
    margin = int(np.ceil(min_shift*fs))
    if n <= 2*margin:
        raise ValueError('Record too short for the predeclared shift null')
    fft = np.fft.rfft(z, axis=0)
    correlations = np.fft.irfft(fft[:, pairs[:, 0]].conj()*fft[:, pairs[:, 1]],
                               n=n, axis=0)/n
    shifts = rng.integers(margin, n-margin+1, size=(draws, len(pairs)))
    return correlations[shifts, np.arange(len(pairs))[None, :]]


def holm(p):
    p = np.asarray(p); order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result = np.empty_like(p); result[order] = np.minimum(1., adjusted)
    return result


def aggregate_test(observed, null, indices, draws, rng, two_sided=True):
    indices = np.asarray(indices)
    actual = np.mean(observed[indices], axis=0)
    sampled = np.zeros((draws, observed.shape[-1]))
    for i in indices:
        sampled += null[i, rng.integers(len(null[i]), size=draws)]/len(indices)
    statistic = np.abs(actual) if two_sided else actual
    samples = np.abs(sampled) if two_sided else sampled
    p = (1+np.sum(samples >= statistic, axis=0))/(draws+1)
    bootstrap = observed[rng.choice(indices, (draws, len(indices)), replace=True)].mean(axis=1)
    return dict(mean=actual.tolist(), ci95=np.quantile(bootstrap, [.025, .975], axis=0).T.tolist(),
                p=p.tolist(), p_holm=holm(p).tolist(),
                null_mean=sampled.mean(axis=0).tolist(),
                null_upper95=np.quantile(sampled, .95, axis=0).tolist())


def lag_correlations(x, pairs, lags):
    result = []
    for lag in lags:
        a, b = (x[:-lag], x[lag:]) if lag > 0 else ((x[-lag:], x[:lag]) if lag < 0 else (x, x))
        a, b = standardized(a), standardized(b)
        result.append(np.mean(a[:, pairs[:, 0]]*b[:, pairs[:, 1]], axis=0))
    return np.stack(result, axis=-1)


def triad_grid(meta, nperseg):
    """Candidate definition uses saved training bank only, never held-out data."""
    f = np.asarray(meta['frequencies_hz']); widths = np.asarray(meta['linewidths_hz'])
    edges = np.asarray(meta['bands_hz']); df = meta['coefficient_fs']/nperseg
    triads, cells, owners = [], [], []
    for i, j in itertools.combinations_with_replacement(range(len(f)), 2):
        for k in range(j+1, len(f)):
            tolerance = max(2*df, .5*(widths[i]+widths[j]+widths[k]))
            if abs(f[i]+f[j]-f[k]) > tolerance:
                continue
            local = []
            for u, v in itertools.product((-1, 0, 1), repeat=2):
                a, b = int(round(f[i]/df))+u, int(round(f[j]/df))+v
                if (edges[i, 0] <= a*df <= edges[i, 1] and
                    edges[j, 0] <= b*df <= edges[j, 1] and
                    edges[k, 0] <= (a+b)*df <= edges[k, 1]):
                    local.append((i, j, k, a, b, a+b))
            if local:
                owners.extend([len(triads)]*len(local)); cells.extend(local)
                triads.append(dict(bands=[i, j, k], frequencies_hz=f[[i, j, k]].tolist(),
                                   detuning_hz=float(f[i]+f[j]-f[k]), tolerance_hz=float(tolerance),
                                   cells=len(local)))
    return triads, np.asarray(cells), np.asarray(owners)


def targeted_bicoherence(c, cells, owners, nperseg):
    """Conventional squared cross-bicoherence, exact bin sums, Hann windows.

    Each cell: |mean(X_i X_j conj(X_k))|^2 /
               (mean(|X_i X_j|^2) mean(|X_k|^2)).
    Average cell b^2 within a fixed 3x3 neighborhood, not complex bispectra.
    """
    if len(c) < 2*nperseg:
        raise ValueError('Need at least three overlapping windows for bicoherence')
    starts = np.arange(0, len(c)-nperseg+1, nperseg//2)
    frames = c[starts[:, None]+np.arange(nperseg)[None, :]]
    frames = frames-frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.fft(frames*np.hanning(nperseg)[None, :, None], axis=1)
    i, j, k, a, b, summed = cells.T
    product = spectrum[:, a, i]*spectrum[:, b, j]
    third = spectrum[:, summed, k]
    numerator = np.mean(product*third.conj(), axis=0)
    denominator = np.mean(abs(product)**2, axis=0)*np.mean(abs(third)**2, axis=0)
    coherency = numerator/np.sqrt(np.maximum(denominator, 1e-300))
    b2 = abs(coherency)**2
    if np.max(b2) > 1+1e-10:
        raise AssertionError('Cauchy-Schwarz normalization violated')
    count = np.bincount(owners)
    result = np.bincount(owners, weights=b2)/count
    return result, coherency, len(starts)


def phase_surrogate(c, rng):
    """Common phase at each frequency preserves every band cross-spectrum.

    Conjugate-symmetric multiplier corresponds to phase randomizing the real
    parent field; this keeps linear overlap/mixing while destroying nonlinear
    frequency-phase organization. DC and Nyquist multipliers are fixed.
    """
    n = len(c)
    positive = np.exp(1j*rng.uniform(-np.pi, np.pi, (n-1)//2))
    multiplier = np.r_[1., positive, ([1.] if n % 2 == 0 else []), positive[::-1].conj()]
    return np.fft.ifft(np.fft.fft(c, axis=0)*multiplier[:, None], axis=0)


def confirmation(train, heldout, metric_count):
    selected = np.flatnonzero(np.asarray(train['p_holm']) < .05)
    p = np.asarray(heldout['p'])[selected]
    return dict(training_selected=selected.tolist(),
                heldout_p_holm_selected=holm(p).tolist() if len(p) else [],
                confirmed=[int(i) for i, adjusted in zip(selected, holm(p)) if adjusted < .05],
                tested_family_size=metric_count)


def analyze(args, meta):
    cache = args.output/'coefficients'
    names = [name for split in ('train', 'validation', 'test') for name in meta['splits'][split]]
    records = []
    for name in names:
        with np.load(cache/f'{name}.npz') as f:
            records.append({k:f[k] for k in f.files})
    controls = []
    for i in range(16):
        with np.load(cache/f'independent_bank_{i:02d}.npz') as f:
            controls.append({k:f[k] for k in f.files})
    hybrids = []
    for i in range(16):
        with np.load(cache/f'full_hybrid_{i:02d}.npz') as f:
            hybrids.append({k:f[k] for k in f.files})
    pairs = np.asarray(list(itertools.combinations(range(11), 2)))
    rng = np.random.default_rng(args.seed)
    envelope = np.stack([pair_correlations(r['amplitude'], pairs) for r in records])
    power = np.stack([pair_correlations(r['power'], pairs) for r in records])
    narrow = np.stack([pair_correlations(r['narrow_amplitude'], pairs) for r in records])
    control = np.stack([pair_correlations(r['amplitude'], pairs) for r in controls])
    hybrid = np.stack([pair_correlations(r['amplitude'], pairs) for r in hybrids])
    null = np.stack([shifted_null(r['amplitude'], pairs, args.envelope_surrogates, rng,
                                  meta['envelope_fs']) for r in records])
    lags = np.arange(-45, 46, 2)
    lags = np.unique(np.r_[lags, 0])
    curves = np.stack([lag_correlations(r['amplitude'], pairs, lags) for r in records])
    control_curves = np.stack([lag_correlations(r['amplitude'], pairs, lags) for r in controls])
    hybrid_curves = np.stack([lag_correlations(r['amplitude'], pairs, lags) for r in hybrids])
    modal = np.stack([[pair_correlations(r['modal_amplitude'][:, m], pairs)
                       for m in range(11)] for r in records])
    envelope_results = {}
    for label, ids in [('train', range(12)), ('validation', range(12, 14)),
                       ('test', range(14, 16)), ('heldout', range(12, 16))]:
        envelope_results[label] = aggregate_test(envelope, null, list(ids), args.group_draws, rng)
    envelope_confirm = confirmation(envelope_results['train'], envelope_results['heldout'], len(pairs))
    # Confirmation includes the training direction, not only a two-sided p value.
    envelope_confirm['confirmed'] = [i for i in envelope_confirm['confirmed']
        if np.sign(envelope[:12, i].mean()) == np.sign(envelope[12:, i].mean())]
    # Choose lag-plot pairs using training magnitude only; no lag significance claims.
    order = np.argsort(-abs(envelope[:12].mean(axis=0)))
    selected = order[:6]
    print('Envelope training top pairs:', [(pairs[i].tolist(), round(envelope[:12, i].mean(), 3))
                                           for i in selected], flush=True)
    print('Envelope confirmed:', [pairs[i].tolist() for i in envelope_confirm['confirmed']], flush=True)
    triads, cells, owners = triad_grid(meta, args.nperseg)
    print(f'Analyze {len(triads)} candidate triads ({len(cells)} exact-sum cells)', flush=True)
    triad_cache = args.output/f'triad_null_n{args.nperseg}_s{args.phase_surrogates}.npz'
    if triad_cache.exists():
        with np.load(triad_cache) as z:
            b2, bnull, coh, control_b2 = [z[k] for k in ('b2', 'null', 'coherency', 'control_b2')]
            phase_envelope_null = z['phase_envelope_null']
    else:
        b2, bnull, coh, control_b2, phase_envelope_null = [], [], [], [], []
        surrogate_rng = np.random.default_rng(args.seed+100)
        for index, (record, model) in enumerate(zip(records, controls)):
            print(f'Phase surrogates repetition {index+1}/16', flush=True)
            c = record['coefficients']
            actual, complex_coh, _ = targeted_bicoherence(c, cells, owners, args.nperseg)
            b2.append(actual); coh.append(complex_coh)
            triad_draws, envelope_draws = [], []
            for _ in range(args.phase_surrogates):
                surrogate = phase_surrogate(c, surrogate_rng)
                triad_draws.append(targeted_bicoherence(surrogate, cells, owners, args.nperseg)[0])
                envelope_draws.append(pair_correlations(coarse(abs(surrogate), 4), pairs))
            bnull.append(triad_draws); phase_envelope_null.append(envelope_draws)
            control_b2.append(targeted_bicoherence(model['coefficients'], cells, owners, args.nperseg)[0])
        b2, bnull, coh, control_b2 = map(np.asarray, (b2, bnull, coh, control_b2))
        phase_envelope_null = np.asarray(phase_envelope_null)
        np.savez_compressed(triad_cache, b2=b2, null=bnull, coherency=coh, control_b2=control_b2,
                            cells=cells, owners=owners, phase_envelope_null=phase_envelope_null)
    phase_envelope_results = {}
    for label, ids in [('train', range(12)), ('heldout', range(12, 16))]:
        # One-sided: excess positive envelope correlation beyond preserved linear mixing.
        phase_envelope_results[label] = aggregate_test(envelope, phase_envelope_null, list(ids),
                                                       args.group_draws, rng, False)
    triad_results = {}
    for label, ids in [('train', range(12)), ('validation', range(12, 14)),
                       ('test', range(14, 16)), ('heldout', range(12, 16))]:
        triad_results[label] = aggregate_test(b2, bnull, list(ids), args.group_draws, rng, False)
    triad_confirm = confirmation(triad_results['train'], triad_results['heldout'], len(triads))
    hybrid_b2 = np.stack([targeted_bicoherence(r['coefficients'], cells, owners, args.nperseg)[0]
                          for r in hybrids])
    # Repetition/path bootstrap for the model discrepancy, independent of null tests.
    measured_draws = envelope[rng.integers(12, 16, (args.group_draws, 4))].mean(axis=1)
    hybrid_draws = hybrid[rng.integers(0, 16, (args.group_draws, 16))].mean(axis=1)
    discrepancy = measured_draws-hybrid_draws
    discrepancy_interval = np.quantile(discrepancy, [.025, .975], axis=0).T
    saved_comparison = {}
    for kind in ('saved_reference', 'saved_hybrid'):
        with_arrays = []
        for name in names[12:14]:
            with np.load(cache/f'{kind}_{name}.npz') as z:
                with_arrays.append(pair_correlations(z['amplitude'], pairs))
                duration = len(z['amplitude'])/meta['envelope_fs']
        saved_comparison[kind] = dict(per_repetition=np.asarray(with_arrays).tolist(),
                                     usable_envelope_seconds=duration)
    report = dict(metadata=meta, settings=dict(seed=args.seed, nperseg=args.nperseg,
                  envelope_surrogates=args.envelope_surrogates, phase_surrogates=args.phase_surrogates,
                  group_draws=args.group_draws, frequency_spacing_hz=meta['coefficient_fs']/args.nperseg,
                  windows_per_repetition=targeted_bicoherence(records[0]['coefficients'], cells,
                                                             owners, args.nperseg)[2]),
                  pairs=pairs.tolist(), envelope=envelope_results, envelope_confirmation=envelope_confirm,
                  envelope_phase_null=phase_envelope_results,
                  envelope_per_repetition=envelope.tolist(), power_per_repetition=power.tolist(),
                  narrow_envelope_per_repetition=narrow.tolist(), independent_bank_envelope=control.tolist(),
                  full_hybrid_envelope=hybrid.tolist(),
                  heldout_minus_hybrid_envelope_ci95=discrepancy_interval.tolist(),
                  full_hybrid_bicoherence=hybrid_b2.tolist(),
                  triads=triads, bicoherence=triad_results, triad_confirmation=triad_confirm,
                  bicoherence_per_repetition=b2.tolist(), independent_bank_bicoherence=control_b2.tolist(),
                  saved_rollout_comparison=saved_comparison,
                  envelope_duration_seconds=len(records[0]['amplitude'])/meta['envelope_fs'],
                  coefficient_duration_seconds=len(records[0]['coefficients'])/meta['coefficient_fs'])
    state_correlations = []
    for record in records:
        c = record['coefficients']
        c = c-c.mean(0)
        covariance = c.T@c.conj()/len(c)
        rms = np.sqrt(np.diag(covariance).real)
        state_correlations.append(covariance/(rms[:, None]*rms[None, :]))
    state_correlations = np.asarray(state_correlations)
    report['complex_state_correlation_magnitude_per_repetition'] = np.abs(
        state_correlations[:, pairs[:, 0], pairs[:, 1]]).tolist()
    np.savez_compressed(args.output/'statistics.npz', pairs=pairs, envelope=envelope, power=power,
                        narrow=narrow, control=control, modal=modal, lags_seconds=lags/meta['envelope_fs'],
                        lag_correlations=curves, control_lag_correlations=control_curves,
                        hybrid=hybrid, hybrid_lag_correlations=hybrid_curves, hybrid_bicoherence=hybrid_b2,
                        envelope_null=null, bicoherence=b2, bicoherence_null=bnull,
                        phase_envelope_null=phase_envelope_null,
                        complex_state_correlations=state_correlations,
                        bicoherence_cells=cells, bicoherence_cell_owner=owners, coherency=coh)
    write_json(args.output/'summary.json', report)
    plots(args.output, meta, report, pairs, envelope, control, narrow, curves, control_curves,
          lags, selected, b2, bnull, control_b2, modal, hybrid, hybrid_curves, hybrid_b2)
    markdown(args.output, report, selected)


def matrix(values, pairs):
    result = np.eye(11)
    result[pairs[:, 0], pairs[:, 1]] = values
    result[pairs[:, 1], pairs[:, 0]] = values
    return result


def plots(output, meta, report, pairs, envelope, control, narrow, curves, control_curves,
          lags, selected, b2, bnull, control_b2, modal, hybrid, hybrid_curves, hybrid_b2):
    labels = [f'{f:.0f}' for f in meta['frequencies_hz']]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for ax, values, title in zip(axes, [envelope[:12].mean(0), envelope[12:].mean(0), hybrid.mean(0), control.mean(0)],
                                  ['Measured: train 1–12', 'Measured: heldout 13–16', 'Full hybrid: 16 paths', 'Independent bank: 16 paths']):
        display = matrix(values, pairs); np.fill_diagonal(display, np.nan)
        im = ax.imshow(display, vmin=-.3, vmax=.3, cmap='RdBu_r')
        ax.set(xticks=range(11), yticks=range(11), xticklabels=labels, yticklabels=labels,
               xlabel='Peak frequency [Hz]', ylabel='Peak frequency [Hz]', title=title)
        ax.tick_params(axis='x', labelrotation=90)
    fig.colorbar(im, ax=axes, label='Mean within-repetition amplitude correlation', shrink=.8)
    fig.savefig(output/'envelope_matrices.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for ax, p in zip(axes.flat, selected):
        for group, color, label in [(slice(0, 12), 'C0', 'Train'), (slice(12, 16), 'C1', 'Heldout')]:
            vals = curves[group, p]
            ax.plot(lags/meta['envelope_fs']*1000, vals.mean(0), color=color, label=label)
            ax.fill_between(lags/meta['envelope_fs']*1000, *np.quantile(vals, [.25, .75], axis=0), color=color, alpha=.15)
        ax.plot(lags/meta['envelope_fs']*1000, control_curves[:, p].mean(0), color='k', ls='--', label='Independent bank')
        ax.plot(lags/meta['envelope_fs']*1000, hybrid_curves[:, p].mean(0), color='C2', ls='-.', label='Full hybrid')
        i, j = pairs[p]
        ax.axhline(0, color='.6', lw=.5); ax.axvline(0, color='.6', lw=.5)
        ax.set(title=f'{labels[i]} → {labels[j]} Hz', xlabel='Lag [ms]; positive: second peak later', ylabel='Envelope correlation', ylim=(-.65, 1))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Pairs selected by training zero-lag magnitude; shading: repetition IQR, not confidence intervals')
    fig.savefig(output/'envelope_lags.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for ax, p in zip(axes.flat, selected):
        ax.plot(range(1, 17), envelope[:, p], 'o-', label='Original bands')
        ax.plot(range(1, 17), narrow[:, p], 's--', label='80% bandwidth')
        ax.axvline(12.5, color='k', lw=.8); ax.axhline(0, color='.6', lw=.5)
        i, j = pairs[p]
        ax.set(title=f'{labels[i]} / {labels[j]} Hz', xlabel='Repetition', ylabel='Amplitude correlation', ylim=(-1, 1), xticks=[1, 4, 8, 12, 13, 14, 15, 16])
    axes.flat[0].legend(fontsize=8)
    fig.savefig(output/'repetition_consistency.png', dpi=180); plt.close(fig)
    score = b2[:12].mean(0)-bnull[:12].mean(axis=(0, 1))
    top = np.argsort(-score)[:8]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, ids, title in zip(axes, [np.arange(12), np.arange(12, 16)], ['Train 1–12', 'Heldout 13–16']):
        avg = b2[ids].mean(0)
        ax.bar(np.arange(len(top))-.18, avg[top], width=.36, label='Measured')
        ax.bar(np.arange(len(top))+.18, bnull[ids].mean(axis=(0, 1))[top], width=.36, label='Phase-randomized null')
        ax.plot(range(len(top)), control_b2.mean(0)[top], 'kx', label='Independent bank')
        ax.plot(range(len(top)), hybrid_b2.mean(0)[top], 'C2o', label='Full hybrid')
        ax.set(xticks=range(len(top)), xticklabels=['+'.join(labels[i] for i in report['triads'][t]['bands'][:2])+'→'+labels[report['triads'][t]['bands'][2]] for t in top],
               ylabel='Mean squared cross-bicoherence', title=title)
        ax.tick_params(axis='x', labelrotation=60); ax.legend(fontsize=8)
    fig.suptitle('Triads ranked on training excess over null; raw b² has substantial finite-window bias')
    fig.savefig(output/'targeted_bicoherence.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, p in zip(axes.flat, selected):
        ax.plot(range(1, 11), modal[:12, 1:, p].mean(0), 'o-', label='Train')
        ax.plot(range(1, 11), modal[12:, 1:, p].mean(0), 's-', label='Heldout')
        i, j = pairs[p]; ax.set(title=f'{labels[i]} / {labels[j]} Hz', xlabel='POD mode', ylabel='Direct band-envelope correlation', ylim=(-1, 1))
        ax.axhline(0, color='.6', lw=.5)
    axes.flat[0].legend()
    fig.savefig(output/'within_pod_envelopes.png', dpi=180); plt.close(fig)
    confirmed = report['envelope_confirmation']['confirmed']
    if confirmed:
        fig, axes = plt.subplots(len(confirmed), 2, figsize=(11, 4*len(confirmed)), squeeze=False,
                                 constrained_layout=True)
        for row, p in enumerate(confirmed):
            i, j = pairs[p]
            ax = axes[row, 0]
            ax.plot(range(1, 17), envelope[:, p], 'o-', label='Measured, original bands')
            ax.plot(range(1, 17), narrow[:, p], 's--', label='Measured, 80% bandwidth')
            ax.axhline(hybrid[:, p].mean(), color='C2', label='Full-hybrid mean')
            ax.axvline(12.5, color='k', lw=.6); ax.axhline(0, color='.7', lw=.6)
            ax.set(title=f'{labels[i]} / {labels[j]} Hz', xlabel='Repetition', ylabel='Envelope correlation',
                   xticks=range(1, 17)); ax.legend(fontsize=8)
            ax = axes[row, 1]
            train = report['envelope']['train']; held = report['envelope']['heldout']
            model_draws = np.random.default_rng(1).choice(hybrid[:, p], (8192, 16), replace=True).mean(1)
            means = [train['mean'][p], held['mean'][p], hybrid[:, p].mean()]
            cis = np.array([train['ci95'][p], held['ci95'][p], np.quantile(model_draws, [.025, .975])])
            ax.errorbar(range(3), means, yerr=np.maximum(0, np.array([means-cis[:, 0], cis[:, 1]-means])), fmt='o', capsize=5)
            ax.axhline(0, color='.7', lw=.6)
            ax.set(xticks=range(3), xticklabels=['Train 1–12', 'Heldout 13–16', 'Full hybrid'], ylabel='Mean envelope correlation',
                   title='95% repetition/path bootstrap intervals')
        fig.savefig(output/'confirmed_envelope.png', dpi=180); plt.close(fig)


def markdown(output, r, selected):
    f = r['metadata']['frequencies_hz']; pairs = r['pairs']
    confirmed_labels = [f'{f[pairs[p][0]]:.0f}/{f[pairs[p][1]]:.0f} Hz'
                        for p in r['envelope_confirmation']['confirmed']]
    lines = ['# Peak dependence: unified_compact', '',
             'Source files and model artifacts were read only. No training or HDF5 changes.', '',
             '## Main finding', '',
             f"Envelope pairs passing training selection and joint heldout confirmation: {', '.join(confirmed_labels) or 'none'}. Primary bicoherence training discoveries: {len(r['triad_confirmation']['training_selected'])}. These tests assess statistical dependence, not physical energy exchange.", '',
             '## Population and protocol', '',
             f"- All 16 source signatures match the checkpoint preparation manifest; {r['envelope_duration_seconds']:.3f} s of envelopes per full repetition after guards.",
             '- Legacy shared-POD coordinates: 10 POD coefficients plus sqrt(N) times the 50 Hz highpassed spatial mean. This is the model population, not a newly cleaned raw-field analysis.',
             '- No training trim (saved train_trim_percent=0). Exact original filters and spatial projections, native edge guard, then anti-aliased decimation with extra guards.',
             '- Discovery: repetitions 1–12. Confirmation: 13–16 jointly, with validation (13–14) and test (15–16) also reported separately in JSON. The shared POD basis itself includes held-out data.',
             '- Primary envelope test: within-repetition Pearson amplitude correlation; circular shifts of at least 100 ms; arithmetic mean across repetitions; two-sided Monte Carlo p; Holm correction across 55 training pairs, then only selected pairs on heldout data with matching sign.',
             '- Bootstrap intervals resample repetitions, not frames; only four heldout repetitions means limited uncertainty resolution. Lag plots are descriptive; no optimization of lag on heldout data.',
             f"- Targeted bicoherence: {len(r['triads'])} candidates fixed by saved centers and half-sum linewidth tolerance (at least two FFT bins); 3x3 local FFT cells restricted to band edges, exact f3=f1+f2; squared normalization. Hann {r['settings']['nperseg']} samples, 50% overlap, {r['settings']['frequency_spacing_hz']:.2f} Hz spacing, {r['settings']['windows_per_repetition']} windows per repetition.",
             '- Phase surrogate uses the same random phase at a given Fourier frequency across all extracted bands, preserving linear cross-spectra and filter-overlap dependence. Whole-record phase randomization is a stationary null and does not preserve envelopes or intermittent structure.',
             '- Bicoherence inference averages per-repetition b², comparing with per-repetition phase-null draws; Holm on all training triads, then selected heldout triads. High raw b² alone is not evidence of coupling.',
             '- Independent bank control uses the archived oscillator scales and 5.78% overlap, stationary independent OU states, 16 independent paths, identical extraction. It is a bank-only control, not a new full neural-SDE simulation.', '',
             '- Full-model comparison: 16 fresh autonomous hybrid paths, original noise shelf/scales and bank calibration, 50000-step burnin, then 100001 samples per path before identical extraction. Initial conditions span validation repetitions 13–14. See long_hybrid_metadata.json.', '',
             '## Envelope results', '',
             f"Training discoveries: {len(r['envelope_confirmation']['training_selected'])}/55. Same-sign heldout confirmations: {len(r['envelope_confirmation']['confirmed'])}.", '',
             '| Peaks [Hz] | Train r | Heldout r | Full hybrid r | Train adjusted p | Heldout raw p | Confirmed |',
             '|---|---:|---:|---:|---:|---:|---|']
    for p in selected:
        i, j = pairs[p]
        lines.append(f"| {f[i]:.0f}, {f[j]:.0f} | {r['envelope']['train']['mean'][p]:.3f} | {r['envelope']['heldout']['mean'][p]:.3f} | {np.mean(r['full_hybrid_envelope'], axis=0)[p]:.3f} | {r['envelope']['train']['p_holm'][p]:.4f} | {r['envelope']['heldout']['p'][p]:.4f} | {'yes' if p in r['envelope_confirmation']['confirmed'] else 'no'} |")
    for p in r['envelope_confirmation']['confirmed']:
        i, j = pairs[p]
        observed = np.asarray(r['envelope_per_repetition'])[:, p]
        narrow = np.asarray(r['narrow_envelope_per_repetition'])[:, p]
        selected_index = r['envelope_confirmation']['training_selected'].index(p)
        phase_train, phase_held = r['envelope_phase_null']['train'], r['envelope_phase_null']['heldout']
        lines += ['', f"For {f[i]:.0f}/{f[j]:.0f} Hz:", '',
                  f"- Positive in {int(np.sum(observed>0))}/16 repetitions, ranging from {observed.min():.3f} to {observed.max():.3f}; this does not mean each individual repetition is statistically significant.",
                  f"- Heldout adjusted p over the training-selected family: {r['envelope_confirmation']['heldout_p_holm_selected'][selected_index]:.4f}.",
                  f"- Validation mean r={r['envelope']['validation']['mean'][p]:.3f}, test-only mean r={r['envelope']['test']['mean'][p]:.3f}. The two test repetitions alone do not establish significance (raw p={r['envelope']['test']['p'][p]:.3f}).",
                  f"- Narrowed bands: training mean r={narrow[:12].mean():.3f}, heldout r={narrow[12:].mean():.3f}.",
                  f"- Spectrum-preserving phase-null envelope means: train {phase_train['null_mean'][p]:.3f}, heldout {phase_held['null_mean'][p]:.3f}; observed excess remains significant even correcting across all 55 pairs (train p={phase_train['p_holm'][p]:.4f}, heldout p={phase_held['p_holm'][p]:.4f}). This argues against simple linear spectral leakage as a sufficient explanation, under this stationary null.",
                  f"- Heldout minus full-hybrid difference: descriptive 95% repetition/path bootstrap interval {r['heldout_minus_hybrid_envelope_ci95'][p][0]:.3f} to {r['heldout_minus_hybrid_envelope_ci95'][p][1]:.3f} (no simultaneous interval adjustment)."]
    lines += ['', 'Bandwidth sensitivity and individual repetitions: `repetition_consistency.png`. Direct individual-POD band envelopes (before rank-one projection): `within_pod_envelopes.png`.', '',
              '## Extracted oscillator states', '',
              'Complex coefficients are recovered with the saved spatial projection and filters. `statistics.npz` also contains their full normalized complex covariance matrices, separately for all repetitions. These are data-derived band coordinates, not saved physical hidden-state observations. Nonzero covariance can arise from overlapping filter tails, which the spectrum-preserving surrogate retains.', '']
    for p in r['envelope_confirmation']['confirmed']:
        i, j = pairs[p]
        state = np.asarray(r['complex_state_correlation_magnitude_per_repetition'])[:, p]
        lines.append(f"For {f[i]:.0f}/{f[j]:.0f} Hz, the mean absolute complex-state correlation is {state.mean():.3f}. The envelope dependence is stronger (about {np.mean(r['envelope_per_repetition'], axis=0)[p]:.3f}) and exceeds the phase-null envelope correlation despite preserving that linear covariance.")
    lines += ['',
              '## Targeted triads', '',
              f"Training discoveries: {len(r['triad_confirmation']['training_selected'])}/{len(r['triads'])}. Heldout confirmations: {len(r['triad_confirmation']['confirmed'])}.", '',
              '| Triad [Hz] | Detuning [Hz] | Train b² | Train null mean | Train adjusted p | Heldout b² | Heldout p |',
              '|---|---:|---:|---:|---:|---:|---:|']
    order = np.argsort(np.asarray(r['bicoherence']['train']['p']))[:10]
    for t in order:
        triad = r['triads'][t]; a,b,c = triad['frequencies_hz']
        lines.append(f"| {a:.0f}+{b:.0f}→{c:.0f} | {triad['detuning_hz']:.1f} | {r['bicoherence']['train']['mean'][t]:.3f} | {r['bicoherence']['train']['null_mean'][t]:.3f} | {r['bicoherence']['train']['p_holm'][t]:.4f} | {r['bicoherence']['heldout']['mean'][t]:.3f} | {r['bicoherence']['heldout']['p'][t]:.4f} |")
    duration = r['saved_rollout_comparison']['saved_hybrid']['usable_envelope_seconds']
    lines += ['', '## Interpretation limits', '',
              f'- The archived combined hybrid/reference trajectories are only 0.303 s. Reapplying the full native filter guards leaves {duration:.3f} s of usable envelopes. Their correlations are saved in JSON as descriptive checks only; they cannot support slow-envelope or triad inference.',
              '- A stable envelope correlation is statistical dependence, not proof of energy exchange or direct causal coupling. Shared forcing and slow common modulation remain alternatives.',
              '- Noncausal band filtering prevents causal interpretation of lag signs. Narrower-band sensitivity is descriptive, not a separately calibrated significance test.',
              '- The narrowest peak has relatively few independent envelope fluctuations in one recording. Circular shifts assume approximate stationarity and can fail for long coherent trends.',
              '- Frequency triads alone do not establish spatial resonance. No wavevector matching, interaction overlaps, or physical energy-transfer rates are estimated here.',
              '- This first pass tests peak–peak triads only. A third component in the broadband remainder and higher-order coupling remain separate questions.', '',
              '## Reproduce', '', '```bash',
              'env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/peak-coupling-mpl \\',
              '  /home/jonas/miniforge3/envs/diff-model-mit/bin/python -m data_analysis.correlations.peak_coupling --simulate-hybrid',
              'env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/peak-coupling-mpl \\',
              '  /home/jonas/miniforge3/envs/diff-model-mit/bin/python -m data_analysis.correlations.peak_coupling',
              '```', '', 'Extraction and phase-surrogate caches are reused. Full numerical results: `summary.json` and `statistics.npz`.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def window_sensitivity(args, meta):
    """Two additional resolutions, with primary candidate triples fixed."""
    primary = json.loads((args.output/'summary.json').read_text())
    allowed = {tuple(t['bands']) for t in primary['triads']}
    names = [name for split in ('train', 'validation', 'test') for name in meta['splits'][split]]
    coefficients = []
    for name in names:
        with np.load(args.output/'coefficients'/f'{name}.npz') as z:
            coefficients.append(z['coefficients'])
    results = {}
    for n in (256, 1024):
        if n == args.nperseg:
            continue
        path = args.output/f'window_sensitivity_n{n}_s{args.phase_surrogates}.npz'
        triples, cells, owners = triad_grid(meta, n)
        keep = [i for i, t in enumerate(triples) if tuple(t['bands']) in allowed]
        remap = {old:new for new,old in enumerate(keep)}
        mask = np.isin(owners, keep); cells = cells[mask]
        owners = np.array([remap[o] for o in owners[mask]])
        triples = [triples[i] for i in keep]
        if path.exists():
            with np.load(path) as z:
                observed, null = z['observed'], z['null']
        else:
            rng = np.random.default_rng(args.seed+n)
            observed, null = [], []
            for rep, c in enumerate(coefficients):
                print(f'Window sensitivity n={n}: repetition {rep+1}/16', flush=True)
                observed.append(targeted_bicoherence(c, cells, owners, n)[0])
                null.append([targeted_bicoherence(phase_surrogate(c, rng), cells, owners, n)[0]
                             for _ in range(args.phase_surrogates)])
            observed, null = np.asarray(observed), np.asarray(null)
            np.savez_compressed(path, observed=observed, null=null)
        rng = np.random.default_rng(args.seed+n+1000)
        train = aggregate_test(observed, null, range(12), args.group_draws, rng, False)
        heldout = aggregate_test(observed, null, range(12, 16), args.group_draws, rng, False)
        results[str(n)] = dict(triads=triples, train=train, heldout=heldout,
                               confirmation=confirmation(train, heldout, len(triples)),
                               spacing_hz=meta['coefficient_fs']/n,
                               windows=targeted_bicoherence(coefficients[0], cells, owners, n)[2])
    write_json(args.output/'window_sensitivity.json', results)
    lines = ['', '## Bicoherence window-length sensitivity', '',
             'Exploratory resolution check of the primary candidate triples. Each resolution uses a new matched phase-null distribution; no extra candidates are selected using heldout data. Corrections are within each resolution, not across all resolutions.', '',
             '| Window samples | Bin spacing [Hz] | Windows per recording | Training discoveries | Heldout confirmations |',
             '|---|---:|---:|---:|---:|']
    for n, result in results.items():
        lines.append(f"| {n} | {result['spacing_hz']:.2f} | {result['windows']} | {len(result['confirmation']['training_selected'])} | {len(result['confirmation']['confirmed'])} |")
    with (args.output/'report.md').open('a') as handle:
        handle.write('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--envelope-surrogates', type=int, default=1999)
    parser.add_argument('--phase-surrogates', type=int, default=499)
    parser.add_argument('--group-draws', type=int, default=8191)
    parser.add_argument('--nperseg', type=int, default=512)
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--simulate-hybrid', action='store_true')
    args = parser.parse_args()
    if args.simulate_hybrid:
        simulate_hybrid(args)
        return
    meta = prepare(args)
    if not args.extract_only:
        if not (args.output/'coefficients/full_hybrid_15.npz').exists():
            simulate_hybrid(args)
        analyze(args, meta)
        window_sensitivity(args, meta)


if __name__ == '__main__':
    main()
