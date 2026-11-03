"""Matched-length and matched-band comparison of saved oscillator study outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .model import welch_matrix
from .study import acf, diagonal, peaks_report, spectral_metrics, write_json


def band_acf(records, fs, maxlag, low=100., high=1100.):
    sos=butter(4,[low,high],fs=fs,btype='bandpass',output='sos')
    edge=int(np.ceil(.05*fs))
    if records.shape[1]<=2*edge+maxlag:
        raise ValueError('Insufficient samples after bandpass edge exclusion')
    filtered=sosfiltfilt(sos,records,axis=1)[:,edge:-edge]
    return acf(filtered,maxlag)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    args=parser.parse_args()
    config=json.loads((args.study/'config.json').read_text())
    fs=config['fs'];nperseg=config['nperseg']
    with np.load(Path(config['audit_dir'])/'signals.npz') as z:
        train=z['reference'][:12].copy();val=z['reference'][12:].copy()
        tcn=z['generated'].reshape(-1,z['generated'].shape[-2],z['generated'].shape[-1]).copy()
    length=tcn.shape[1]
    # Freeze training targets using all training samples, independent of evaluation duration.
    _,full_training_spectra=welch_matrix(train,fs,nperseg)
    peak_training=full_training_spectra.mean(0)
    # Equal analyzed duration for every record, preserving record boundaries.
    train=train[:,:length];val=val[:,:length]
    f,tr=welch_matrix(train,fs,nperseg);_,vr=welch_matrix(val,fs,nperseg)
    training=tr.mean(0);reference=vr.mean(0)
    _,tcn_s=welch_matrix(tcn,fs,nperseg)
    spectra={'native_tcn':tcn_s.mean(0)}
    times=np.arange(round(.03*fs)+1)/fs
    a_reference=band_acf(val,fs,len(times)-1)
    acfs={'native_tcn':band_acf(tcn,fs,len(times)-1)}
    per_rep_spectra=vr
    expected={}
    refreshed_summaries={}
    for variant in config['variants']:
        with np.load(args.study/variant/'stationary_rollout.npz') as z:
            generated=z['generated'][:,:length].copy()
        _,s=welch_matrix(generated,fs,nperseg)
        spectra[variant]=s.mean(0)
        acfs[variant]=band_acf(generated,fs,len(times)-1)
        with np.load(args.study/variant/'spectra.npz') as z:
            expected[variant]=z['expected'].copy()
        # Correct older study summaries if produced before the subset metric fix.
        summary=json.loads((args.study/variant/'summary.json').read_text())
        with np.load(args.study/variant/'spectra.npz') as z:
            for key,array in [('expected',z['expected']),('generated',z['generated'].mean(0)),('native_tcn',z['native_tcn'].mean(0))]:
                summary[key+'_first6']=spectral_metrics(f,array,z['validation'].mean(0),range(6))
                summary[key+'_peaks']=peaks_report(f,array,z['validation'].mean(0),z['training'].mean(0))
        write_json(args.study/variant/'summary.json',summary)
        refreshed_summaries[variant]=summary
    write_json(args.study/'summary.json',refreshed_summaries)
    time_mask=(times>=.001)&(times<=.025)
    result=dict(protocol=dict(fs=fs,nperseg=nperseg,steps_per_record=length,
        training_records=len(train),validation_records=len(val),test_records=0,
        bandpass='Fourth-order Butterworth 100–1100 Hz, zero phase, 50 ms removed per edge',
        acf_estimator='Per-record demeaning and variance normalization, then equal-record average',
        selection='No refitting or checkpoint selection in this audit',
        limitation='Two independent validation repetitions; generated paths are unconditional, TCN starts overlap'),
        variants={},validation_mean_acf_14p3ms=float(a_reference[round(.0143*fs),0]))
    for name,s in spectra.items():
        r=dict(first6=spectral_metrics(f,s,reference,range(6)),all21=spectral_metrics(f,s,reference),
            peaks=peaks_report(f,s,reference,peak_training),
            band_acf_rmse_first6=float(np.sqrt(np.mean((acfs[name][time_mask,:6]-a_reference[time_mask,:6])**2))),
            band_mean_acf_14p3ms=float(acfs[name][round(.0143*fs),0]),
            per_validation_record=[spectral_metrics(f,s,v,range(6)) for v in per_rep_spectra])
        if name in expected:
            r['expected_first6']=spectral_metrics(f,expected[name],reference,range(6))
            r['expected_peaks']=peaks_report(f,expected[name],reference,peak_training)
        result['variants'][name]=r
    # Quantify the irreducible imaginary-part error of a real displacement readout.
    mask=(f>=100)&(f<=1100)
    def imaginary_fraction(s):
        diag=diagonal(s);norm=np.sqrt(diag[:,:,None]*diag[:,None,:])
        coherency=s/np.maximum(norm,1e-30)
        off=~np.eye(s.shape[-1],dtype=bool)
        return float(np.sum(coherency[mask][:,off].imag**2)/np.sum(abs(coherency[mask][:,off])**2))
    result['imaginary_fraction_of_normalized_cross_spectral_power']={
        'training':imaginary_fraction(training),'validation':imaginary_fraction(reference)}
    write_json(args.study/'matched_comparison.json',result)
    np.savez_compressed(args.study/'matched_comparison_arrays.npz',f=f,reference=reference,training=training,
        times=times,reference_acf=a_reference,**{'psd_'+k:v for k,v in spectra.items()},
        **{'acf_'+k:v for k,v in acfs.items()})
    colors={'native_tcn':'C0','smooth':'.6','one_family':'C2','two_family':'C1','two_family_rank2':'C3'}
    show=[name for name in ('native_tcn','two_family','two_family_rank2') if name in spectra]
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    view=(f>=90)&(f<=1150)
    for c,ax in enumerate(axs.flat):
        ax.semilogy(f[view],diagonal(reference)[view,c],color='black',label='Validation')
        for name in show:
            ax.semilogy(f[view],diagonal(spectra[name])[view,c],color=colors[name],label=name,lw=1.1)
        ax.set(title='Spatial mean' if c==0 else f'POD {c}',xlabel='Frequency [Hz]',ylabel='PSD [µm²/Hz]')
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.savefig(args.study/'comparison_psd.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for c,ax in enumerate(axs.flat):
        ax.plot(times*1000,a_reference[:,c],color='black',label='Validation')
        for name in show:
            ax.plot(times*1000,acfs[name][:,c],color=colors[name],label=name,lw=1.1)
        ax.set(title='Spatial mean' if c==0 else f'POD {c}',xlabel='Lag [ms]',ylabel='100–1100 Hz band ACF',ylim=(-.7,1))
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.savefig(args.study/'comparison_band_acf.png',dpi=180);plt.close(fig)
    lines=['# Matched-duration oscillator comparison','',
        f'Every PSD uses {length} samples per record, fs={fs:.6f} Hz, nperseg={nperseg}, Hann and 50% overlap.',
        'Training repetitions 1–12 only fitted the model. Evaluation uses the first matched-duration interval of validation repetitions 13–14.',
        'Correlation scores below apply the same 100–1100 Hz filter and 50 ms edge exclusion to every signal. This isolates the fitted spectral region.',
        '', '| Model | PSD RMSE [dB] | Ripple RMSE [dB] | Band ACF RMSE | Mean peak matches | POD 1 peak matches | POD 2 peak matches |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in [*config['variants'],'native_tcn']:
        r=result['variants'][name];m=r['first6'];peaks=r['peaks']
        count=['%d/%d'%(v['hits'],v['targets']) for v in peaks[:3]]
        lines.append(f"| {name} | {m['log_psd_rmse_db']:.3f} | {m['ripple_rmse_db']:.3f} | {r['band_acf_rmse_first6']:.3f} | {' | '.join(count)} |")
    lines+=['','PSD, ripple and ACF errors aggregate spatial mean plus POD 1–5. Peak matching is descriptive (0.75 dB prominence, +/-22 Hz), and finite realizations can create incidental peaks.',
        'Exact expected-model peak counts are also saved in matched_comparison.json to distinguish persistent resonances from random rollout ripples.',
        '',f"Normalized cross-spectral imaginary fraction: training {result['imaginary_fraction_of_normalized_cross_spectral_power']['training']:.3f}, validation {result['imaginary_fraction_of_normalized_cross_spectral_power']['validation']:.3f}. Real displacement readout cannot reproduce this component. These descriptive fractions are not corrected for finite-sample spectral noise.",
        '', 'These are stationary spectral baselines. They are not conditional forecasts, full-frequency calibrated models, or identified physical eigenmodes.',
        'The native TCN comparison uses an existing checkpoint with a different training objective. No neural-SDE training or controller state was changed.']
    (args.study/'matched_comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
