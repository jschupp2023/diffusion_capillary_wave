"""Bounded train-only conditioning sweep for a square observed-state oscillator.

Reuses the existing prepared experimental cache. No neural checkpoint is changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import welch_matrix
from .observed_spectral import ObservedSpectralModel, spectral_loss


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def initial_basis(target, frequency, peaks, copies, seed):
    """Real spectral subspaces are initialization only, not identified eigenmodes."""
    columns = []
    for peak in peaks:
        matrix = target[abs(frequency-peak) < 12].mean(0).real
        _, u = np.linalg.eigh(matrix)
        columns.extend(u[:, -1-i] for i in range(copies))
    columns = np.stack(columns, axis=1)
    u, _, _ = np.linalg.svd(columns, full_matrices=True)
    candidate = np.column_stack((columns, u[:, columns.shape[1]:]))
    left, _, right = np.linalg.svd(candidate)
    basis = left @ right
    rng = np.random.default_rng(seed)
    a = rng.normal(0, .025, basis.shape)
    from scipy.linalg import expm
    return basis @ expm(a-a.T), np.linalg.svd(columns, compute_uv=False)


def metrics(predicted, reference):
    pd = np.diagonal(predicted, axis1=-2, axis2=-1).real.clip(1e-15)
    rd = np.diagonal(reference, axis1=-2, axis2=-1).real.clip(1e-15)
    normalized = (predicted-reference)/np.sqrt(rd[:, :, None]*rd[:, None, :])
    off = ~np.eye(pd.shape[1], dtype=bool)
    return dict(log_psd_rmse_db=float(np.sqrt(np.mean((10*np.log10(pd/rd))**2))),
                focused_log_psd_rmse_db=float(np.sqrt(np.mean((10*np.log10(pd[:, :4]/rd[:, :4]))**2))),
                normalized_cross_rmse=float(np.sqrt(np.mean(abs(normalized[:, off])**2))),
                normalized_imaginary_rmse=float(np.sqrt(np.mean(normalized[:, off].imag**2))),
                band_power_ratios=(pd.sum(0)/rd.sum(0)).tolist())


def peak_metrics(frequency, predicted, training, validation, prescribed):
    """Match distinct peaks; train selection and held-out confirmation are explicit."""
    from scipy.signal import find_peaks
    from scipy.optimize import linear_sum_assignment
    rows = []
    for coordinate in range(4):
        logs = [10*np.log10(s[:, coordinate, coordinate].real.clip(1e-15))
                for s in (training, validation, predicted)]
        found = [find_peaks(x, prominence=.75, distance=max(1, round(15/np.diff(frequency)[0])))[0]
                 for x in logs]
        targets = [k for k in found[0] if np.min(abs(prescribed-frequency[k])) <= 12
                   and np.any(abs(frequency[found[1]]-frequency[k]) <= 12)]
        pairs = []
        if targets and len(found[2]):
            dist = abs(frequency[np.array(targets)][:, None]-frequency[found[2]][None])
            rr, cc = linear_sum_assignment(np.where(dist <= 12, dist, 1e6))
            pairs = [(targets[r], found[2][c]) for r, c in zip(rr, cc) if dist[r, c] <= 12]
        rows.append(dict(coordinate=coordinate, targets=len(targets), hits=len(pairs),
            target_frequencies_hz=frequency[targets].tolist(),
            matched_prediction_frequencies_hz=[float(frequency[k]) for _, k in pairs],
            target_height_errors_db=[float(logs[2][k]-logs[1][k]) for k in targets]))
    return rows


def fit_one(args, target, indices, frequency, scale, peaks, copies, cap, seed):
    basis, candidate_sv = initial_basis(target.numpy(), frequency, peaks, copies, seed)
    model = ObservedSpectralModel(np.repeat(peaks, copies), basis, cap, args.fs, args.nperseg)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.015)
    started = time.monotonic()
    best, best_state, best_step = float('inf'), None, -1
    history = []
    for step in range(args.iterations+1):
        optimizer.zero_grad()
        prediction = model.spectrum(indices)
        loss = spectral_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite spectral objective')
        value = loss.item()
        if value < best:
            best, best_step, best_state = value, step, copy.deepcopy(model.state_dict())
        if step % 25 == 0:
            history.append(dict(step=step, loss=value, elapsed_seconds=time.monotonic()-started))
        if step == args.iterations or time.monotonic()-started >= args.seconds_per_fit:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        model.project()
    model.load_state_dict(best_state)
    with torch.no_grad():
        prediction = model.spectrum(indices).numpy()
        t, k, d, q = [x.numpy() for x in model.operators()]
        w, g = [x.numpy()*1000 for x in model.poles()]
    full_a = np.block([[np.zeros_like(k), np.eye(len(k))], [-k, -d]])
    result = dict(cap=cap, copies_per_peak=copies, seed=seed,
        iterations=step, best_step=best_step, train_objective=best,
        elapsed_seconds=time.monotonic()-started, history=history,
        condition_standardized=float(np.linalg.cond(t)),
        singular_values_standardized=np.linalg.svd(t, compute_uv=False).tolist(),
        condition_physical=float(np.linalg.cond(scale[:, None]*t)),
        initial_candidate_singular_values=candidate_sv.tolist(),
        maximum_drift_eigenvalue_real_per_second=float(np.linalg.eigvals(full_a).real.max()))
    return model, prediction, result, dict(T=t, K=k, D=d, acceleration_covariance=q,
                                          omega_per_second=w, gamma_per_second=g)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-study', default='runs/diagnostics/latent_oscillators_0p10_two_family_v1')
    p.add_argument('--atlas', default='runs/diagnostics/spectral_peak_atlas_0p10_v1')
    p.add_argument('--frequency-source', choices=['atlas', 'fitted-family'], default='atlas')
    p.add_argument('--output', required=True)
    p.add_argument('--caps', nargs='+', type=float, default=[1, 3, 10])
    p.add_argument('--copies', nargs='+', type=int, default=[1, 2])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--iterations', type=int, default=150)
    p.add_argument('--seconds-per-fit', type=float, default=25.)
    p.add_argument('--nperseg', type=int, default=4096)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args(argv)
    if args.iterations < 1 or args.seconds_per_fit <= 0 or min(args.copies) < 1 or min(args.caps) < 1:
        p.error('Require positive iterations, time, copies and caps >= 1')
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(args.source_study)/'config.json').read_text())
    cache = Path(config['audit_dir'])
    provenance = json.loads((cache/'provenance.json').read_text())
    train_ids, val_ids = provenance['training_repetitions'], provenance['validation_repetitions']
    if train_ids != list(range(1, 13)) or val_ids != [13, 14] or provenance['test_repetitions_used']:
        raise ValueError('Unexpected cache split/order; inspect cache producer before proceeding')
    with np.load(cache/'signals.npz') as z:
        records = z['reference']
    if len(records) != len(train_ids)+len(val_ids):
        raise ValueError('Cache population mismatch')
    nt = len(train_ids)
    args.fs = provenance['single_rollout']['sampling_frequency_hz']/provenance['downsample_factor']
    mean, scale = records[:nt].mean((0, 1)), records[:nt].std((0, 1))
    print('Computing equal-record full-band Welch matrices from prepared experimental cache', flush=True)
    f, spectra = welch_matrix((records-mean)/scale, args.fs, args.nperseg)
    mask = (f >= 80) & (f <= 1500)
    indices = torch.tensor(np.flatnonzero(mask))
    frequency, spectra = f[mask], spectra[:, mask]
    train, validation = spectra[:nt].mean(0), spectra[nt:].mean(0)
    atlas = json.loads((Path(args.atlas)/'summary.json').read_text())
    peaks = np.array([s['frequency_hz'] for s in atlas['sites'] if s['kind'] == 'peak'])
    if args.frequency_source == 'fitted-family':
        with np.load(Path(args.source_study)/'two_family/model.npz') as saved:
            # Earlier train-only fitted natural frequencies -> isolated displacement PSD peaks.
            peaks = np.sqrt(saved['frequencies_hz']**2-2*(saved['damping_per_second']/(2*np.pi))**2)
    if max(args.copies)*len(peaks) >= len(scale):
        raise ValueError('Need at least one nonresonant coordinate')
    write_json(out/'protocol.json', dict(arguments=vars(args), source_provenance=provenance,
        cache_sha256=hashlib.sha256((cache/'signals.npz').read_bytes()).hexdigest(),
        atlas_sha256=hashlib.sha256((Path(args.atlas)/'summary.json').read_bytes()).hexdigest(),
        frequency_model_sha256=(hashlib.sha256((Path(args.source_study)/'two_family/model.npz').read_bytes()).hexdigest()
                                if args.frequency_source == 'fitted-family' else None),
        source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ['observed_spectral.py', 'observed_spectral_study.py', 'model.py']},
        prescribed_peak_frequencies_hz=peaks.tolist(),
        frequency_constraints='Train-atlas peaks or earlier train-only fitted two-family frequencies; individual white-forcing peak parameters within +/-8 Hz. Natural frequencies account for damping. Broad complement 30–6000 Hz, damping ratio 0.8–0.99.',
        objective='Mean squared natural-log PSD error plus 2 times mean squared complex cross-spectrum error normalized by training marginal PSD; uniform fitted bins and coordinates.',
        expected_spectrum='Exact sampled stationary covariance and periodic-Hann finite-window expectation; constant detrending immaterial for bins >=2.',
        state='21 measured positions and their velocities. No extra OU/readout states. Full constant real acceleration covariance.',
        scaling='Train-only temporal standard deviations of prepared cache, not native neural checkpoint scales. T columns have unit Euclidean norm in these standardized coordinates.',
        limits=['Prepared, downsampled/filtered signal model; not a native-rate neural checkpoint.',
                'Only 80–1500 Hz fitted; no full-band or conditional forecasting claim.',
                'Prior shared spatial basis includes held-out repetitions.',
                'Bounded nonconvex fits, not a proof of impossibility or an optimal conditioning bound.',
                'PSD/cross-spectral metrics score distinct properties; SPOD is initialization only.']))
    np.savez_compressed(out/'reference_spectra.npz', frequency_hz=frequency, train=train,
        validation=validation, per_validation_record=spectra[nt:], mean=mean, scale=scale)
    results, predictions = [], []
    for copies in args.copies:
        for seed in args.seeds:
            for cap in args.caps:
                name = f'copies{copies}_cap{cap:g}_seed{seed}'
                print('Fitting', name, flush=True)
                model, prediction, result, arrays = fit_one(args, torch.tensor(train), indices,
                                                           frequency, scale, peaks, copies, cap, seed)
                result.update(name=name, train=metrics(prediction, train),
                    validation=metrics(prediction, validation),
                    peaks=peak_metrics(frequency, prediction, train, validation, peaks),
                    per_validation_record=[metrics(prediction, s) for s in spectra[nt:]])
                folder = out/name; folder.mkdir()
                write_json(folder/'summary.json', result)
                np.savez_compressed(folder/'model.npz', **arrays, mean=mean, scale=scale,
                    frequency_hz=frequency, predicted_spectrum=prediction, fs=args.fs,
                    nperseg=args.nperseg, condition_cap=cap, prescribed_peaks_hz=peaks)
                torch.save(model.state_dict(), folder/'parameters.pt')
                results.append(result); predictions.append(prediction)
                write_json(out/'results.json', results)
                print(name, 'steps', result['iterations'], 'seconds', round(result['elapsed_seconds'], 2),
                      'condition', round(result['condition_standardized'], 2),
                      'validation dB', round(result['validation']['log_psd_rmse_db'], 3), flush=True)
    # Select seed by TRAINING objective for each prescribed model family only.
    selected = [min([i for i, r in enumerate(results) if r['copies_per_peak']==copies and r['cap']==cap],
                    key=lambda i:results[i]['train_objective']) for copies in args.copies for cap in args.caps]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for coordinate, ax in enumerate(axes.flat):
        ax.plot(frequency, 10*np.log10(train[:, coordinate, coordinate].real), color='.6', label='train')
        ax.plot(frequency, 10*np.log10(validation[:, coordinate, coordinate].real), color='k', label='validation')
        for i in selected:
            ax.plot(frequency, 10*np.log10(predictions[i][:, coordinate, coordinate].real.clip(1e-15)),
                    lw=1, alpha=.8, label=results[i]['name'])
        ax.set_title('Spatial mean' if coordinate==0 else f'POD {coordinate}')
        ax.set_ylabel('PSD [dB, standardized units/Hz]'); ax.set_xlabel('Frequency [Hz]')
    axes[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(out/'psd_comparison.png', dpi=160); plt.close(fig)
    lines = ['# Observed-state oscillator conditioning study', '',
        'Frozen experimental spectra; no neural training or checkpoint modifications. Seeds selected by training objective only.', '',
        '| Resonances | Cap | Actual condition | Train objective | Validation PSD RMSE [dB] | Mean/POD1–3 RMSE [dB] | Complex cross RMSE |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for i in selected:
        r=results[i]; v=r['validation']
        lines.append(f"| {len(peaks)*r['copies_per_peak']} | {r['cap']:g} | {r['condition_standardized']:.2f} | {r['train_objective']:.3f} | {v['log_psd_rmse_db']:.3f} | {v['focused_log_psd_rmse_db']:.3f} | {v['normalized_cross_rmse']:.3f} |")
    lines += ['', '[PSD comparison](psd_comparison.png). Per-seed results: `results.json`; exact limits and provenance: `protocol.json`.', '',
        'The remaining position directions are broad, strongly damped second-order modes within the same square T. They are not additive OU states.',
        'T has unit columns and a hard condition cap in training-standardized coordinates. Physical-coordinate conditions are reported separately.',
        'A full correlated white acceleration covariance permits imaginary cross-spectra, unlike independent real-readout oscillators.',
        'Fits use exact expected Hann spectra on the prepared sampling grid. All fits are bounded local optimizations; a poor fit is not an impossibility proof.',
        'Only two held-out recordings. The shared POD basis predates this test and includes held-out data. Native-rate implementation and conditional prediction remain untested.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
