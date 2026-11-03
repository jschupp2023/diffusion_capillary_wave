"""Export saved unconditional oscillator trajectories to the common plot schema.

The reference cache has sqrt(N)-scaled physical POD coefficients. This exporter
restores the common physical coefficient convention; it never filters or
calibrates the generated trajectories. Reference times are relative cache times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from data_analysis.rollout import Rollout


def pack_arrays(generated, reference, n_space, fs, repetitions):
    """Group independent draws evenly across reference records, restoring units."""
    generated=np.asarray(generated)
    reference=np.asarray(reference)
    if (generated.ndim!=3 or reference.ndim!=3 or generated.shape[1:]!=reference.shape[1:]
            or len(reference)==0 or len(generated)%len(reference)!=0
            or len(repetitions)!=len(reference) or n_space<=0 or fs<=0):
        raise ValueError('Need compatible paths, equal grouping, positive grid size and fs')
    if not np.isfinite(generated).all() or not np.isfinite(reference).all():
        raise ValueError('Cannot export nonfinite trajectories')
    scale=np.sqrt(n_space)
    return dict(trajectories=generated.reshape(len(reference),-1,*generated.shape[1:])*scale,
        reference=reference*scale,
        reference_time=np.broadcast_to(np.arange(reference.shape[1])/fs,reference.shape[:2]).copy(),
        repetition=np.asarray(repetitions))


def export(variant, output=None):
    variant=Path(variant).resolve()
    config=json.loads((variant.parent/'config.json').read_text())
    provenance=config['source_provenance']
    basis=Path(provenance['single_rollout']['shared_basis'])
    audit=Path(config['audit_dir'])
    output=Path(output).resolve() if output else variant/'standard_rollout_validation_2x8'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    with np.load(variant/'stationary_rollout.npz') as z:
        generated=z['generated'].copy();fs=float(z['fs'])
    if not np.isclose(fs,config['fs']):
        raise ValueError('Saved simulation and study sampling rates disagree')
    train_reps=provenance['training_repetitions'];val_reps=provenance['validation_repetitions']
    if train_reps!=list(range(1,13)) or val_reps!=[13,14] or provenance['test_repetitions_used']:
        raise ValueError('Unexpected cache split; explicit adaptation required')
    with np.load(audit/'signals.npz') as z:
        reference=z['reference'][len(train_reps):].copy()
    with h5py.File(basis) as h:
        n_space=len(h['grid/x'])*len(h['grid/y'])
    repetitions=[f'Ca_ac_0p000830_rep{rep}' for rep in val_reps]
    arrays=pack_arrays(generated,reference,n_space,fs,repetitions)
    meta=dict(model_type='latent_oscillator_stationary_baseline',
        source_shared_basis=str(basis),rank=reference.shape[-1]-1,include_spatial_mean=True,
        time_units='s',signal_units=provenance['single_rollout']['signal_units'],
        physical_lag=1/fs,lag_steps=1,spatial_mean_highpass_hz=50.,
        source_model=str(variant/'model.npz'),source_simulation=str(variant/'stationary_rollout.npz'),
        source_audit=str(audit),reference_split='validation',split='validation',
        initialization='Unconditional stationary draws; no shared initial state with reference',
        pairing='Independent generated paths grouped evenly with validation repetitions for distributional comparison only',
        pointwise_error_interpretation='Pointwise errors are not conditional forecast errors',
        reference_time_origin='Seconds from the start of each cached record, not raw recording timestamps',
        coefficient_conversion=f'Cached coefficients multiplied by sqrt({n_space}); coordinate zero is sqrt(N) times physical spatial mean',
        generated_filtering='None; generated spatial mean has NOT been high-pass filtered',
        temporal_preprocessing=dict(downsample_factor=provenance['downsample_factor'],
            raw_dt=1/provenance['single_rollout']['sampling_frequency_hz'],prepared_dt=1/fs,
            anti_alias=provenance['anti_alias'],
            edge_discard_decimated_samples=provenance['edge_discard_decimated_samples'],
            reference='Existing anti-aliased cache, including 50Hz high-pass spatial mean',
            generated='Simulated directly on the prepared time grid; no postprocessing'),
        reference_window_trim_policy=dict(training_trim_available=False,enforced=False,
            train_trim_percent=0.,normalized_increment_norm_cutoff=None,
            generated_trajectories_trimmed=False),
        fitting_band_hz=config['fit_band_hz'],
        warning='The oscillator fit constrained 80–1500Hz only; full-band diagnostics assess uncalibrated frequencies too')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'rollout.npz',**arrays)
    (output/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
    checked=Rollout(output)
    np.testing.assert_allclose(checked.generated.reshape(generated.shape)/np.sqrt(n_space),generated)
    np.testing.assert_allclose(checked.reference/np.sqrt(n_space),reference)
    (output/'README.md').write_text(
        '# Standard oscillator rollout diagnostics\n\n'
        'Existing stationary simulations, exported without refitting or filtering. '
        'Two validation records (13–14), with eight independent generated paths per record. '
        'Grouping is for distributional metrics; paths do not share reference initial states. '
        'Pointwise errors are not conditional forecast scores.\n\n'
        'Reference mean has the source 50Hz high-pass preprocessing; generated mean is unfiltered. '
        'All other source anti-aliasing and coefficient conversions are recorded in metrics.json. '
        'No increment trimming is used. Full PSD extends to the prepared-grid Nyquist (7.2kHz), '
        'not the native experimental Nyquist. The fitted band was only 80–1500Hz.\n')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    print(export(args.variant,args.output))


if __name__=='__main__':
    main()
