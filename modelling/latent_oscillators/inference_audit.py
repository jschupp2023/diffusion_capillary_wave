"""Frozen-model hidden-state recovery and held-out conditional forecast audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.signal import welch

from .inference import (StateSpace, filter_schedule, filter_observations,
                        forecast, simulate_observations, smooth_endpoint)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    path.write_text(json.dumps(value, default=convert, indent=2, allow_nan=False)+'\n')


def recovery(system, truth, estimate, covariance, scale):
    error = estimate-truth
    result = {}
    for name, indices in [('q', system.q_indices), ('u', system.u_indices)]:
        e = error[:, indices]
        variance = np.diag(covariance)[indices]
        target, predicted = truth[:, indices], estimate[:, indices]
        target = target-target.mean(0)
        predicted = predicted-predicted.mean(0)
        denominator = np.sqrt(np.sum(target**2, 0)*np.sum(predicted**2, 0))
        correlation = np.divide(np.sum(target*predicted, 0), denominator,
                                out=np.zeros_like(denominator), where=denominator > 0)
        result[name] = dict(rmse=float(np.sqrt(np.mean(e**2))),
            expected_rmse=float(np.sqrt(np.mean(variance))),
            per_state_rmse=np.sqrt(np.mean(e**2, 0)),
            per_state_expected_rmse=np.sqrt(np.maximum(variance, 0)),
            per_state_correlation=correlation,
            coverage95=float(np.mean(abs(e) <= 1.95996398454*np.sqrt(variance))),
            mse_over_posterior_variance=float(np.mean(e**2/variance)))
    for name, h in [('resonant_standardized', system.oscillator_readout),
                    ('resonant_physical_pod', scale[1:, None]*system.oscillator_readout[1:]),
                    ('resonant_mean', system.oscillator_readout[:1])]:
        prior_power = np.sum(h**2)
        result[name] = dict(
            nrmse=float(np.sqrt(np.mean(np.sum((error@h.T)**2, 1))/prior_power)),
            expected_nrmse=float(np.sqrt(np.trace(h@covariance@h.T)/prior_power)))
    return result


def forecast_scores(target, mean, covariance):
    error = target-mean
    result = {'mse_per_coordinate': np.mean(error**2, axis=0)}
    for label, indices in [('all21', slice(None)), ('first6', slice(0, 6)),
                           ('spatial_mean', slice(0, 1)), ('pod1_2', slice(1, 3))]:
        e = error[:, indices]
        result[label] = {'mse': float(np.mean(e**2))}
        if covariance is not None:
            cov = covariance[indices, indices]
            factor = cholesky(cov, lower=True)
            white = solve_triangular(factor, e.T, lower=True).T
            result[label].update(
                mahalanobis_per_dimension=float(np.mean(white**2)),
                gaussian_nll_per_dimension=float(.5*(np.mean(white**2)+np.log(2*np.pi)
                    +2*np.log(np.diag(factor)).mean())),
                coverage95=float(np.mean(abs(e) <= 1.95996398454*np.sqrt(np.diag(cov)))))
    return result


def innovation_metrics(white, fs):
    result = []
    for record in white:
        centered = record-record.mean(0)
        acf = {}
        for lag in sorted(set([1, round(.001*fs), round(.005*fs), round(.01*fs)])):
            a, b = centered[:-lag], centered[lag:]
            acf[str(lag)] = np.sum(a*b, axis=0)/np.sqrt(np.sum(a*a, 0)*np.sum(b*b, 0))
        result.append(dict(mean_per_coordinate=record.mean(0),
            second_moment_per_coordinate=np.mean(record**2, 0),
            mean_squared_whitened_innovation=float(np.mean(record**2)),
            coverage95=float(np.mean(abs(record) <= 1.95996398454)),
            autocorrelation=acf))
    return result


def plots(output, synthetic, histories, system, forecasts, horizons, white, frequencies):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.asarray(histories)
    for key, label in [('q', 'individual q'), ('u', 'individual u'),
                       ('resonant_standardized', 'combined resonant output')]:
        metric = 'rmse' if key in ('q', 'u') else 'nrmse'
        expected = 'expected_'+metric
        line, = axes[0].plot(x, [s[key][metric] for s in synthetic[:-1]], 'o-', label=label)
        axes[0].plot(x, [s[key][expected] for s in synthetic[:-1]], '--', alpha=.5,
                     color=line.get_color())
    axes[0].set(xscale='symlog', xlim=(-.05, max(x)*1.1), ylim=(0, 1.1),
                xlabel='Past observation span (ms; 0 = snapshot)', ylabel='RMSE / stationary RMS',
                title='Synthetic known-state recovery')
    axes[0].legend(fontsize=8)
    axes[0].text(.03, .04, 'Dashed: exact model uncertainty', transform=axes[0].transAxes, fontsize=8)
    order = np.argsort(frequencies)
    for row, label in [(synthetic[0], 'snapshot'), (synthetic[-2], '200 ms history'),
                        (synthetic[-1], 'history + 50 ms future')]:
        axes[1].plot(frequencies[order], row['q']['per_state_expected_rmse'][order], 'o-', label=label)
    axes[1].set(xlabel='Natural frequency (Hz)', ylabel='Posterior q standard deviation',
                title='Individual poles: remaining uncertainty', ylim=(0, 1.05))
    axes[1].legend(fontsize=8)
    fig.savefig(output/'synthetic_recovery.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, group in zip(axes, ['all21', 'first6']):
        for name in ['persistence', 'snapshot', 'history_10ms', 'history_50ms', 'history_200ms']:
            selected = [r for r in forecasts if r['method'] == name]
            ratios = []
            for horizon in horizons:
                a = [r['metrics'][group]['mse'] for r in selected if r['horizon_samples'] == horizon]
                b = [r['metrics'][group]['mse'] for r in forecasts
                     if r['method'] == 'stationary_mean' and r['horizon_samples'] == horizon]
                ratios.append(np.sqrt(np.mean(a)/np.mean(b)))
            ax.plot(np.asarray(horizons)/system.fs*1000, ratios, 'o-', label=name)
        ax.axhline(1, color='.5', linestyle='--')
        ax.set(xscale='log', xlabel='Forecast horizon (ms)', ylabel='RMSE / stationary-mean RMSE',
               title=f'Validation 13–14, {group} (standardized)')
        ax.legend(fontsize=8)
    fig.savefig(output/'validation_forecasts.png', dpi=170); plt.close(fig)

    freq, power = welch(white, fs=system.fs, nperseg=2048, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for r in range(len(white)):
        for ax, coordinates in zip(axes, [slice(None), slice(0, 6)]):
            ax.semilogx(freq[1:], (power[r, :, coordinates].mean(axis=1)*system.fs/2)[1:],
                        label=f'rep {13+r}')
    for ax, label in zip(axes, ['all 21 coordinates', 'first 6 coordinates']):
        ax.axhline(1, color='.5', linestyle='--', label='Calibrated white innovations')
        ax.axvspan(80, 1500, alpha=.08, color='C0')
        ax.set(xlim=(20, system.fs/2), xlabel='Frequency (Hz)', ylabel='PSD / unit-white PSD',
               title=f'Pre-update whitened innovations: {label}')
        ax.legend(fontsize=8)
    fig.savefig(output/'validation_innovations.png', dpi=170); plt.close(fig)
    np.savez_compressed(output/'innovation_spectra.npz', frequency_hz=freq, power=power)


def report(output, protocol, synthetic, forecasts, innovations):
    lines = ['# Frozen oscillator hidden-state inference', '',
        'No parameters were refitted. The exact sampled oscillator + colored OU model is used ',
        'with a stationary Gaussian prior, noiseless readout, and a causal Kalman filter.', '',
        f"Synthetic: {protocol['synthetic_paths']} independent stationary paths. "
        'Errors are against the actual hidden state. q and u have unit prior variance. ',
        'Combined-output error is normalized by its exact stationary RMS.', '',
        '| Observation history | q RMSE | u RMSE | Resonant output NRMSE | Expected output NRMSE |',
        '|---|---:|---:|---:|---:|']
    for row in synthetic:
        label = row['method']
        lines.append(f"| {label} | {row['q']['rmse']:.3f} | {row['u']['rmse']:.3f} | "
                     f"{row['resonant_standardized']['nrmse']:.3f} | "
                     f"{row['resonant_standardized']['expected_nrmse']:.3f} |")
    longest = synthetic[-2]
    lines += ['', 'The last row uses future observations and is **not a causal forecast method**.', '',
        f"With 200 ms of past observations, the exact model still assigns an average q variance of "
        f"{longest['q']['expected_rmse']**2:.3f} (prior 1), and a combined resonant-output error variance "
        f"of {longest['resonant_standardized']['expected_nrmse']**2:.3f} relative to its prior power. "
        'History helps substantially, but does not reveal a precise hidden trajectory even when the model is correct.', '',
        '## Held-out forecasts', '',
        'Only repetitions 13–14 are scored. All history lengths use identical forecast origins; ',
        'no observations after each origin enter its forecast. Errors use the saved training ',
        'standardization and the full prepared-grid signal (no additional bandpass). ',
        'The following RMSE ratios use the stationary training mean as baseline; lower is better.', '',
        '| Horizon (ms) | Persistence | Snapshot | 10 ms history | 50 ms history | 200 ms history |',
        '|---:|---:|---:|---:|---:|---:|']
    for horizon in protocol['forecast_samples']:
        base = np.mean([r['metrics']['all21']['mse'] for r in forecasts
                        if r['method'] == 'stationary_mean' and r['horizon_samples'] == horizon])
        values = []
        for method in ['persistence', 'snapshot', 'history_10ms', 'history_50ms', 'history_200ms']:
            mse = np.mean([r['metrics']['all21']['mse'] for r in forecasts
                           if r['method'] == method and r['horizon_samples'] == horizon])
            values.append(f'{np.sqrt(mse/base):.3f}')
        lines.append(f"| {1000*horizon/protocol['fs']:.3f} | "+' | '.join(values)+' |')
    lines += ['', '## Calibration and interpretation limits', '']
    for name, row in zip(protocol['validation_repetitions'], innovations):
        lines.append(f"- Repetition {name}: mean squared whitened one-step innovation "
                     f"{row['mean_squared_whitened_innovation']:.3f} (ideal 1); marginal 95% coverage "
                     f"{100*row['coverage95']:.1f}%; coordinate-averaged lag-one correlation "
                     f"{np.mean(row['autocorrelation']['1']):.3f} (ideal 0).")
    lines += ['', '- Real hidden states are unknown. Synthetic recovery and real forecast skill are different tests.',
        '- Synthetic posterior uncertainty assumes this model is correct; it is not calibrated experimental state uncertainty.',
        '- The OU components remain in the filter. Replacing them with a neural residual changes the inference problem.',
        '- The fit used 80–1500 Hz only. Full-band conditional forecasts test behavior beyond that fitting objective.',
        '- Only two independent validation recordings; forecast windows overlap, so their count is not a replication count.',
        '- Causality is relative to the prepared cache. Existing offline anti-alias/high-pass preprocessing is not raw-camera causal inference.',
        '- u = p/omega. Readouts and state errors use the fitted normalization; physical POD-only output error is also in metrics.json.',
        '- No neural SDE has been changed or trained; inferred residuals are diagnostic estimates, not verified training labels.', '',
        '## Artifacts', '',
        '- `protocol.json`: hashes, frozen settings, data lineage and exact sample counts.',
        '- `metrics.json`: per-state recovery, uncertainty, coverage, per-record forecast and innovation scores.',
        '- `synthetic_recovery.png`, `validation_forecasts.png`, `validation_innovations.png`.',
        '- `inference_arrays.npz`: synthetic endpoint truths/estimates, validation filtered means, innovations and matched forecasts.',
        '- `innovation_spectra.npz`: per-repetition Welch spectra of whitened innovations.', '',
        'Reproduce from the repository root (choose a fresh output directory):', '',
        '```bash',
        'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \\',
        '/home/jonas/miniforge3/envs/diff-model-mit/bin/python \\',
        '  -m modelling.latent_oscillators.inference_audit \\',
        f"  {Path(protocol['model']).parent.parent} \\",
        '  --output runs/diagnostics/latent_oscillator_inference_0p10_repeat',
        '```', '']
    (output/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paths', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261026)
    args = parser.parse_args()
    if args.paths < 2:
        parser.error('--paths must be at least two')
    config = json.loads((args.study/'config.json').read_text())
    model_path = args.study/'two_family/model.npz'
    cache_path = Path(config['audit_dir'])/'signals.npz'
    cache_hash = digest(cache_path)
    if cache_hash != config['cache_sha256']:
        raise ValueError('Cache differs from the fitted model lineage')
    provenance = config['source_provenance']
    if provenance['training_repetitions'] != list(range(1, 13)) or provenance['validation_repetitions'] != [13, 14]:
        raise ValueError('This audit expects the documented 12/2 train/validation split')
    with np.load(model_path) as z:
        saved = dict(z)
    with np.load(cache_path) as z:
        validation = z['reference'][12:].copy()
    if len(validation) != 2:
        raise ValueError('Expected exactly two validation records')
    system = StateSpace.from_saved(saved)
    if not np.isclose(system.fs, config['fs'], rtol=1e-12):
        raise ValueError('Model/cache sampling frequencies differ')
    validation = (validation-saved['mean'])/saved['scale']
    history_ms = [0, 1, 5, 10, 20, 50, 100, 200]
    counts = [1+round(ms*system.fs/1000) for ms in history_ms]
    horizons = sorted(set([1]+[round(ms*system.fs/1000) for ms in [1, 5, 10, 20, 50]]))
    future_count = round(.05*system.fs)
    stride = round(.01*system.fs)
    origins = np.arange(max(counts)-1, validation.shape[1]-max(horizons), stride)
    if not len(origins):
        raise ValueError('Records are too short for the requested history and forecast')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(model=str(model_path.resolve()), model_sha256=digest(model_path),
        code_sha256={name: digest(Path(__file__).with_name(name))
                     for name in ['inference.py', 'inference_audit.py', 'model.py']},
        source_config=str((args.study/'config.json').resolve()), source_provenance=provenance,
        cache=str(cache_path), cache_sha256=cache_hash, seed=args.seed, synthetic_paths=args.paths,
        fs=system.fs, state_dimension=len(system.transition), observed_dimension=len(system.readout),
        history_requested_ms=history_ms, history_samples=counts,
        history_actual_span_ms=(np.asarray(counts)-1)/system.fs*1000,
        forecast_samples=horizons, forecast_actual_ms=np.asarray(horizons)/system.fs*1000,
        smoother_future_samples=future_count, validation_repetitions=[13, 14],
        validation_steps_per_record=validation.shape[1], origins_per_record=origins,
        origin_stride_samples=stride, innovation_discard_samples=max(counts),
        prior='Stationary N(0,I), including colored OU states', measurement_noise=0,
        fitting='None; all model parameters and diagnostics settings frozen before validation scoring',
        covariance_recursion='Exact finite-history Riccati recursion; no steady-state gain approximation')
    write_json(args.output/'protocol.json', protocol)
    print('Computing exact Kalman covariance/gain schedule...', flush=True)
    schedule = filter_schedule(system, validation.shape[1], checkpoints=counts)
    print('Simulating independent stationary paths with known endpoint states...', flush=True)
    observations, truth = simulate_observations(system, max(counts)+future_count, args.paths,
                                               max(counts)-1, args.seed)
    synthetic, forecasts, estimates, real_predictions = [], [], [], []
    real_states = []
    for ms, count in zip(history_ms, counts):
        method = 'snapshot' if ms == 0 else f'history_{ms}ms'
        print(f'Filtering {method}: {count} observations...', flush=True)
        mean, _, _ = filter_observations(system, observations[:, max(counts)-count:max(counts)], schedule)
        covariance = schedule.posterior[count]
        estimates.append(mean)
        synthetic.append(dict(method=method, history_samples=count,
            **recovery(system, truth, mean, covariance, saved['scale'])))
        windows = validation[:, origins[:, None]+np.arange(1-count, 1)[None, :], :]
        real_mean, _, _ = filter_observations(system, windows.reshape(-1, count, len(system.readout)), schedule)
        real_states.append(real_mean.reshape(2, len(origins), -1))
        predictions = []
        for horizon in horizons:
            prediction, predictive_cov = forecast(system, real_mean, covariance, horizon)
            prediction = prediction.reshape(2, len(origins), -1)
            predictions.append(prediction)
            for r, repetition in enumerate([13, 14]):
                target = validation[r, origins+horizon]
                forecasts.append(dict(method=method, repetition=repetition, horizon_samples=horizon,
                    metrics=forecast_scores(target, prediction[r], predictive_cov)))
        real_predictions.append(predictions)
    print('Computing noncausal synthetic endpoint smoother and validation innovations...', flush=True)
    smoothed, smooth_cov = smooth_endpoint(system, estimates[-1], schedule.posterior[max(counts)],
                                          observations[:, max(counts):])
    synthetic.append(dict(method='history_200ms_plus_future_50ms_NONCAUSAL',
        **recovery(system, truth, smoothed, smooth_cov, saved['scale'])))
    for horizon in horizons:
        for r, repetition in enumerate([13, 14]):
            target = validation[r, origins+horizon]
            for method, prediction, covariance in [
                    ('stationary_mean', np.zeros_like(target), system.readout@system.readout.T),
                    ('persistence', validation[r, origins], None)]:
                forecasts.append(dict(method=method, repetition=repetition, horizon_samples=horizon,
                    metrics=forecast_scores(target, prediction, covariance)))
    _, filtered, white = filter_observations(system, validation, schedule, keep_path=True)
    innovations = innovation_metrics(white[:, max(counts):], system.fs)
    np.savez_compressed(args.output/'inference_arrays.npz',
        synthetic_truth=truth, synthetic_estimates=np.asarray(estimates), synthetic_smoothed=smoothed,
        synthetic_posterior_covariances=np.asarray([schedule.posterior[n] for n in counts]),
        synthetic_smoothed_covariance=smooth_cov, frequencies_hz=saved['frequencies_hz'],
        history_samples=counts, forecast_samples=horizons, forecast_origins=origins,
        validation_forecasts=np.asarray(real_predictions), validation_origin_states=np.asarray(real_states),
        validation_normalized=validation, validation_filtered_states=filtered,
        validation_filtered_resonant_output=filtered@system.oscillator_readout.T,
        validation_whitened_innovations=white, normalization_mean=saved['mean'],
        normalization_scale=saved['scale'])
    metrics = dict(synthetic=synthetic, validation_forecasts=forecasts, validation_innovations=innovations)
    write_json(args.output/'metrics.json', metrics)
    plots(args.output, synthetic, history_ms, system, forecasts, horizons,
          white[:, max(counts):], saved['frequencies_hz'])
    report(args.output, protocol, synthetic, forecasts, innovations)
    print(f'Finished: {args.output / "README.md"}', flush=True)


if __name__ == '__main__':
    main()
