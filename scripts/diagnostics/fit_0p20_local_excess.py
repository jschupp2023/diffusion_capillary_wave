"""Fit 11-band local-excess oscillator bank for 0p20."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from modelling.data_preparation.frequency_components import settling_samples
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.frequency_hybrid.local_excess import (
    local_excess_modes, projected_band, complex_covariance,
    fit_filtered_band, simulate_complex_band
)
from modelling.frequency_hybrid.study import (
    center_weights, matrices, projected, spectral_comparison, write_json
)
from modelling.frequency_hybrid.evaluate import describe

# Refined 11 peak centers for 0p20
CENTERS_0P20 = [
    73.83, 126.56, 200.39, 274.22, 362.11,
    450.00, 537.89, 636.33, 741.80, 857.81, 980.86
]

def make_bands_0p20(centers, low_edge=50.0):
    c = np.asarray(centers, dtype=float)
    middle = (c[:-1] + c[1:]) / 2.0
    upper = c[-1] + (c[-1] - middle[-1])
    edges = np.r_[low_edge, middle, upper]
    bands = np.column_stack((edges[:-1], edges[1:]))
    return bands

def run_fit(output_dir, energy_fraction=0.95, rank_limit=1,
            experiment='0p20', rank=10, data_root=None, shared_basis=None,
            spatial_mean_highpass_hz=None):
    output = Path(output_dir).resolve()
    if output.exists():
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {experiment} dataset...")
    kwargs = dict(
        data_root=data_root,
        shared_basis=shared_basis,
        validation_reps=[13, 14],
        test_reps=[15, 16],
        spatial_mean_highpass_hz=spatial_mean_highpass_hz,
        output_dtype='float64'
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    data = SharedPODTrainingData(experiment, rank, **kwargs)
    fs = 1.0 / data.native_dt
    centers = np.asarray(CENTERS_0P20, dtype=float)
    bands = make_bands_0p20(centers, low_edge=50.0)
    
    # Band 0 uses order 2, bands 1-10 use order 4
    orders = np.r_[2, np.full(len(centers) - 1, 4, dtype=int)]
    print("Bands:")
    for i, (c, b, o) in enumerate(zip(centers, bands, orders)):
        print(f"  Band {i:02d}: center={c:.2f} Hz, edges=[{b[0]:.2f}, {b[1]:.2f}] Hz, order={o}")

    filters = [butter(int(order), band, btype='bandpass', fs=fs, output='sos')
               for order, band in zip(orders, bands)]
    edge = max(int(np.ceil(0.1 * fs)), *(settling_samples(s, fs) for s in filters))
    print(f"Settling edge samples: {edge} ({edge / fs:.3f} s)")

    names = data.train_repetitions + data.validation_repetitions
    raw = {name: data.read_coordinates(name, 0, data.frame_counts[name]) for name in names}
    train = [raw[name][1] for name in data.train_repetitions]
    frequency, train_spectrum = matrices(train, fs, 32768)
    weights = center_weights(data)

    retained = {name: (time[edge:-edge], values[edge:-edge]) for name, (time, values) in raw.items()}
    length = min(50001, *(len(retained[name][0]) for name in data.validation_repetitions))
    measured_sum = {name: np.zeros((length, data.n_coordinates)) for name in data.validation_repetitions}
    generated_sum = np.zeros((2, 4, length, data.n_coordinates))

    band_rows = []
    for index, (center, band, order, sos) in enumerate(zip(centers, bands, orders, filters)):
        excess = local_excess_modes(frequency, train_spectrum, weights, center, band,
                                    energy_fraction=energy_fraction)
        if rank_limit is not None and excess['rank'] > rank_limit:
            excess['rank'] = rank_limit
            excess['modes'] = excess['modes'][:, :rank_limit]
            
        train_components = []
        train_coefficients = []
        for name in data.train_repetitions:
            comp, coeffs = projected_band(raw[name][1], sos, excess['modes'], edge)
            train_components.append(comp)
            train_coefficients.append(coeffs)
            
        band_freq, band_spec = matrices(train_components, fs, 32768)
        temporal = fit_filtered_band(train_components, band_freq, band_spec, fs, band, sos)
        coeff_cov, coeff_factor = complex_covariance(train_coefficients)
        
        # Compensate latent variance for filter attenuation
        coeff_factor = coeff_factor / np.sqrt(temporal['filter_passed_variance_fraction'])
        
        model = dict(
            modes=excess['modes'],
            coefficient_covariance=coeff_cov,
            coefficient_factor=coeff_factor,
            frequency_hz=temporal['frequency_hz'],
            gamma_per_second=temporal['gamma_per_second'],
            fs=fs,
            edges_hz=band,
            filter_order=int(order),
            filter_passed_variance_fraction=temporal['filter_passed_variance_fraction']
        )
        np.savez(output / f"band_{index:02d}.npz", **model)

        validation = []
        for name in data.validation_repetitions:
            comp, _ = projected_band(raw[name][1], sos, excess['modes'], edge)
            comp = comp[:length]
            validation.append(comp)
            measured_sum[name] += comp

        generated = simulate_complex_band(model, length + 2 * edge, paths=8, seed=401 + index)
        generated = sosfiltfilt(sos, generated, axis=1, padlen=edge)[:, edge:-edge]
        generated = generated.reshape(2, 4, length, data.n_coordinates)
        generated_sum += generated

        fg, sg = matrices(list(generated.reshape(-1, length, data.n_coordinates)), fs, 32768)
        _, sr = matrices(validation, fs, 32768)
        comparison = spectral_comparison(fg, sg, sr, weights, bands.tolist(), evaluation_band=band.tolist())
        
        band_rows.append(dict(
            index=index,
            nominal_center_hz=float(center),
            edges_hz=band.tolist(),
            selected_peak_hz=excess['peak_frequency_hz'],
            valley_frequencies_hz=excess['valley_frequencies_hz'].tolist(),
            rank=excess['rank'],
            positive_mass_fraction=excess['positive_mass_fraction'],
            retained_positive_fraction=excess['retained_positive_fraction'],
            coordinate_participation=excess['participation'].tolist(),
            fitted_frequency_hz=float(temporal['frequency_hz']),
            fitted_fwhm_hz=float(temporal['approximate_fwhm_hz']),
            heldout=comparison
        ))
        print(f"[{index+1:02d}/{len(centers):02d}] nominal {center:6.1f} Hz -> peak {excess['peak_frequency_hz']:6.1f} Hz, "
              f"fitted {temporal['frequency_hz']:6.1f} Hz, rank {excess['rank']}, "
              f"pos_mass {excess['positive_mass_fraction']:.2f}", flush=True)

    full = np.stack([retained[name][1][:length] for name in data.validation_repetitions])
    times = np.stack([retained[name][0][:length] for name in data.validation_repetitions])
    measured_bands = np.stack([measured_sum[name] for name in data.validation_repetitions])
    remainder = full - measured_bands
    reconstruction_error = float(np.linalg.norm(remainder + measured_bands - full) / np.linalg.norm(full))

    f, sfull = matrices(list(full), fs, 32768)
    _, srem = matrices(list(remainder), fs, 32768)
    _, smeasured = matrices(list(measured_bands), fs, 32768)
    _, sgenerated = matrices(list(generated_sum.reshape(-1, length, data.n_coordinates)), fs, 32768)
    pfull = projected(sfull, weights)
    prem = projected(srem, weights)
    region = (f >= bands[0, 0]) & (f <= bands[-1, 1])
    remainder_fraction = float(prem[region].sum() / pfull[region].sum())
    bank_report = spectral_comparison(f, sgenerated, smeasured, weights, bands.tolist(),
                                      evaluation_band=(float(bands[0, 0]), float(bands[-1, 1])))

    report = dict(
        experiment='0p20',
        method='Local sideband matrix interpolation, positive excess eigenspace, band-limited complex OU bands',
        training_only_identification=True,
        energy_fraction=energy_fraction,
        total_complex_rank=sum(x['rank'] for x in band_rows),
        reconstruction_relative_error=reconstruction_error,
        bands=band_rows,
        remainder_center_power_fraction_oscillator_region=remainder_fraction,
        oscillator_bank_vs_measured_projected_bands=bank_report
    )
    write_json(output / 'summary.json', report)
    np.savez_compressed(
        output / 'spectra.npz',
        frequency_hz=f, full_center=pfull,
        remainder_center=prem, measured_band_center=projected(smeasured, weights),
        generated_bank_center=projected(sgenerated, weights)
    )

    # Plot decomposition
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained')
    mask = (f >= 50) & (f <= 1600)
    axes[0].semilogy(f[mask], pfull[mask], color='black', label='Held-out full signal')
    axes[0].semilogy(f[mask], prem[mask], color='tab:orange', label='0p20 Local-excess remainder')
    axes[1].semilogy(f[mask], projected(smeasured, weights)[mask], color='black', label='Measured projected bands')
    axes[1].semilogy(f[mask], projected(sgenerated, weights)[mask], color='tab:blue', label='0p20 Complex oscillator bank')
    for ax in axes:
        ax.set(xlabel='Frequency [Hz]', ylabel='Center PSD [µm²/Hz]')
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(output / 'decomposition.png', dpi=180)
    plt.close(fig)

    ranks = np.asarray([x['rank'] for x in band_rows])
    participation = np.asarray([x['coordinate_participation'] for x in band_rows])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
    axes[0].bar(centers, ranks, width=35)
    axes[0].set(xlabel='Peak [Hz]', ylabel=f'{energy_fraction:g} excess rank')
    image = axes[1].imshow(participation.T, aspect='auto', origin='lower', cmap='magma',
                           extent=[0, len(centers), -0.5, data.n_coordinates - 0.5])
    coord_labels = (['mean'] + [f'POD{i}' for i in range(1, data.rank + 1)]
                    if data.include_spatial_mean else
                    [f'POD{i}' for i in range(1, data.n_coordinates + 1)])
    axes[1].set(xticks=np.arange(len(centers)) + 0.5, xticklabels=[f'{x:g}' for x in centers],
                yticks=np.arange(data.n_coordinates), yticklabels=coord_labels,
                xlabel='Nominal peak [Hz]', ylabel='Coordinate')
    fig.colorbar(image, ax=axes[1], label='Excess-mode participation')
    fig.savefig(output / 'ranks_and_participation.png', dpi=180)
    plt.close(fig)

    print("\nFitting Complete!")
    print(f"Total complex rank: {report['total_complex_rank']}")
    print(f"Remainder fraction in oscillator region: {remainder_fraction:.3f}")
    print(f"Oscillator bank center PSD MAE: {bank_report['center_psd_mae_db']:.2f} dB")
    print(f"Oscillator bank center power ratio: {bank_report['center_band_power_ratio']:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default='runs/diagnostics/frequency_hybrid_0p20_local_excess_v1/oscillators')
    parser.add_argument('--energy-fraction', type=float, default=0.95)
    parser.add_argument('--rank-limit', type=int, default=1)
    parser.add_argument('--experiment', default='0p20')
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--shared-basis', type=Path, default=None)
    parser.add_argument('--spatial-mean-highpass-hz', type=float, default=None)
    args = parser.parse_args()
    run_fit(args.output, args.energy_fraction, rank_limit=args.rank_limit,
            experiment=args.experiment, rank=args.rank, data_root=args.data_root,
            shared_basis=args.shared_basis, spatial_mean_highpass_hz=args.spatial_mean_highpass_hz)

