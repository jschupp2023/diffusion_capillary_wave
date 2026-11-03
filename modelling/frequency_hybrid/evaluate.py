"""Held-out component and unfiltered-sum diagnostics using the common schema."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

from modelling.neural_sde.train import load_model
from data_analysis.rollout import Rollout
from data_analysis.rollout_metrics import analyze as quantitative
from data_analysis.correlations.rollout_state_decorrelation import mean_path_acf
from data_analysis.psd.rollout_spectral_peaks import peak_metrics
from .study import component_data, center_weights, matrices, projected, spectral_comparison, write_json
from .oscillators import simulate_band, oscillator_psd


def save_rollout(path, gen, ref, times, data, description, *, repetitions=None,
                 start_indices=None,
                 initialization='Oscillators stationary; TCN from training history, 10000 generated burn-in steps',
                 pairing='Independent stationary samples; no pointwise forecast claim', **extra):
    path.mkdir(parents=True,exist_ok=True)
    if repetitions is None:
        repetitions=np.asarray(data.validation_repetitions[:len(ref)])
    if start_indices is None:
        start_indices=np.zeros(len(ref),dtype=np.int64)
    np.savez_compressed(path/'rollout.npz',trajectories=gen,reference=ref,
        reference_time=times,repetition=np.asarray(repetitions),
        start_index=np.asarray(start_indices,dtype=np.int64))
    write_json(path/'metrics.json',dict(rank=data.rank,include_spatial_mean=data.include_spatial_mean,
        source_shared_basis=str(data.basis_path),time_units='s',signal_units=data.signal_units,
        physical_lag=data.native_dt,lag_steps=1,spatial_mean_highpass_hz=data.spatial_mean_highpass_hz,
        temporal_preprocessing=data.temporal_preprocessing(),model_type=description,
        initialization=initialization, pairing=pairing, split='validation',
        generated_filtering='None: actual generated components and their unfiltered sum',**extra))


def validate_generated(gen):
    finite=np.isfinite(gen).all(axis=-1)
    if not finite.all():
        return dict(finite=False,first_nonfinite_step=int(np.where(~finite)[-1].min()))
    return dict(finite=True,max_abs=float(np.max(abs(gen))))


def tcn_rollout(out, component, length, ensemble=4, burnin=10000):
    target=out/f'tcn_{component}'
    model,payload=load_model(target/'best.pt','cpu');model.eval()
    data=component_data(out,component)
    initial=[];histories=[]
    h=model.config.history_steps
    for n in data.train_repetitions[:2]:
        _,x=data.read_coordinates(n,0,h+1)
        initial.append(x[h]);histories.append(x[h-1::-1])
    started=time.monotonic()
    with torch.no_grad():
        # Smoke screen catches nonfinite trajectories before long evaluation.
        screen=model.rollout(torch.tensor(np.asarray(initial)).float(),
            history_states=torch.tensor(np.asarray(histories)).float(),
            num_trajectories=ensemble,horizon=5000,seed=41)
        status=validate_generated(screen.numpy())
        if not status['finite']:
            np.savez_compressed(target/'failed_screen.npz',trajectories=screen.numpy())
            return None,status
        gen=model.rollout(torch.tensor(np.asarray(initial)).float(),
            history_states=torch.tensor(np.asarray(histories)).float(),
            num_trajectories=ensemble,horizon=burnin+length-1,seed=41).numpy()
    status=validate_generated(gen);status['seconds']=time.monotonic()-started
    # Preserve evidence of every simulated sample, including failures/burn-in.
    np.savez_compressed(target/'stationary_generation.npz',trajectories=gen,
                        burnin=burnin,training_initial_repetitions=np.asarray(data.train_repetitions[:2]))
    print(component,'generation',status,flush=True)
    return (gen[:,:,burnin:] if status['finite'] else None),status


def describe(path, gen, ref, times, data, label, bands, *, rollout_metadata=None):
    weights=center_weights(data);fs=1/data.native_dt
    save_rollout(path,gen,ref,times,data,label,**(rollout_metadata or {}))
    rollout=Rollout(path)
    quantitative(rollout,nperseg=8192,psd_lower_frequency=100.,overwrite=True)
    q=json.loads((path/'quantitative/summary.json').read_text())
    flat=gen.reshape(-1,*gen.shape[-2:])
    f,sg=matrices(list(flat),fs);_,sr=matrices(list(ref),fs)
    result=spectral_comparison(f,sg,sr,weights,bands)
    hi_f,hi_g=matrices(list(flat),fs,32768)
    _,hi_r=matrices(list(ref),fs,32768)
    result['high_resolution']=spectral_comparison(hi_f,hi_g,hi_r,weights,bands)
    np.savez_compressed(path/'high_resolution_spectra.npz',frequency_hz=hi_f,
        generated_center_psd=projected(hi_g,weights),reference_center_psd=projected(hi_r,weights),
        generated_modal_psd=np.diagonal(hi_g,axis1=-2,axis2=-1).real,
        reference_modal_psd=np.diagonal(hi_r,axis1=-2,axis2=-1).real)
    if label.startswith('Band '):
        index=int(label.split()[1])-1
        result['within_design_band']=spectral_comparison(
            hi_f,hi_g,hi_r,weights,[bands[index]],evaluation_band=bands[index])
        # Wide-band log errors are large because reference filtering suppresses
        # tails that an unfiltered second-order oscillator cannot reproduce.
        result['wideband_log_error_caution']='Includes near-zero reference stopbands; see within_design_band and tail power separately'
    pg=projected(sg,weights);pr=projected(sr,weights)
    result['peaks']=peak_metrics(f,pr,pg,band=(100.,1500.))
    acfg=mean_path_acf(flat,4000);acfr=mean_path_acf(ref,4000)
    result['acf_rmse_first_4000_lags']=float(np.sqrt(np.mean((acfg-acfr)**2)))
    result['increment_variance_ratio']=float(np.var(np.diff(flat.astype(float),axis=1),axis=(0,1)).sum()/
                                           np.var(np.diff(ref,axis=1),axis=(0,1)).sum())
    result['generated_state_rms']=float(np.sqrt(np.mean(gen.astype(float)**2)))
    result['reference_state_rms']=float(np.sqrt(np.mean(ref**2)))
    result['common_metrics']=str(path/'quantitative/summary.json')
    # Non-overlapping early/late halves make remaining transient drift visible.
    mid=gen.shape[2]//2
    result['late_over_early_second_moment']=float(np.mean(gen[:,:,mid:].astype(float)**2)/
                                                np.mean(gen[:,:,:mid].astype(float)**2))
    np.savez_compressed(path/'spectral_diagnostics.npz',frequency_hz=f,
        generated_spectral_matrix=sg,reference_spectral_matrix=sr,
        generated_acf=acfg,reference_acf=acfr,
        generated_center_psd=pg,reference_center_psd=pr)
    write_json(path/'diagnostics.json',result)
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    mask=(f>=100)&(f<=1500)
    axes[0].semilogy(f[mask],pr[mask],color='black',label='Held-out reference')
    axes[0].semilogy(f[mask],pg[mask],label=label)
    axes[0].set(xlabel='Frequency [Hz]',ylabel='Center PSD [µm²/Hz]')
    axes[1].plot(np.arange(4001)/fs*1000,acfr[:,0],color='black',label='Reference mean ACF')
    axes[1].plot(np.arange(4001)/fs*1000,acfg[:,0],label='Generated mean ACF')
    axes[1].set(xlabel='Lag [ms]',ylabel='ACF')
    for ax in axes: ax.grid(alpha=.2);ax.legend()
    fig.savefig(path/'comparison.png',dpi=170);plt.close(fig)
    print(label,{k:result[k] for k in ('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags')},flush=True)
    return result


def evaluate(out):
    out=Path(out);data=component_data(out,'full');fs=1/data.native_dt
    length=min(50001,min(data.frame_counts[n] for n in data.validation_repetitions))
    bands=data.manifest['bands_hz'];ensemble=4
    refs={};times=[]
    for component in ['full','remainder']+[f'band{i}' for i in range(3)]:
        source=component_data(out,component)
        refs[component]=np.stack([source.read_coordinates(n,0,length)[1] for n in source.validation_repetitions])
    times=np.stack([data.read_coordinates(n,0,length)[0] for n in data.validation_repetitions])
    report=dict(length=length,ensemble_per_record=ensemble,fs=fs,models={},survival={},
        stationary_warning='Training histories then burn-in; stationarity is diagnosed, not assumed.',
        held_out='Validation reps13-14 only. Test15-16 untouched.')
    oscillators=[]
    for i in range(3):
        with np.load(out/f'oscillator_{i}.npz') as z: model=dict(z)
        gen=simulate_band(model,length,paths=2*ensemble,seed=101+i).reshape(2,ensemble,length,11)
        oscillators.append(gen)
        report['models'][f'band{i}']=describe(out/f'validation_band{i}',gen,refs[f'band{i}'],times,data,f'Band {i+1} oscillator',bands)
        f=np.fft.rfftfreq(32768,1/fs)
        density=oscillator_psd(float(model['frequency_hz']),float(model['gamma_per_second']),fs,32768)
        inside=(f>=bands[i][0])&(f<=bands[i][1])
        report['models'][f'band{i}']['expected_outside_design_band_power_fraction']=float(1-density[inside].sum()/density.sum())
    for component in ('full','remainder'):
        generated,status=tcn_rollout(out,component,length,ensemble)
        report['survival'][component]=status
        if generated is not None:
            report['models'][component]=describe(out/f'validation_{component}',generated,
                refs[component],times,data,f'TCN {component}',bands)
            if component=='remainder':
                hybrid=generated.astype(float)+sum(oscillators)
                report['models']['hybrid']=describe(out/'validation_hybrid',hybrid,refs['full'],times,data,'Additive hybrid',bands)
        write_json(out/'evaluation.json',report)
    # A counterfactual using the actual remainder isolates oscillator-fit error.
    oracle=np.repeat(refs['remainder'][:,None],ensemble,axis=1)+sum(oscillators)
    report['models']['oracle_remainder']=describe(out/'validation_oracle_remainder',oracle,refs['full'],times,data,'Observed remainder + independent oscillators',bands)
    root=Path(__file__).resolve().parents[2]
    baseline=root/'runs/adaptive/neural_sde_0p10_tcn63_untrimmed_structured/trials/untrimmed_variance_w0p001_lr1e6/rollout_99937_validation_1'
    if (baseline/'rollout.npz').exists():
        with np.load(baseline/'rollout.npz') as z:
            offset=int(data.component_edge)-int(z['start_index'][0])
            old=z['trajectories'][:,:,offset:offset+length,:11].copy()
            old_ref=z['reference'][:,offset:offset+length,:11].copy()
            old_times=z['reference_time'][:,offset:offset+length].copy()
        np.testing.assert_allclose(old_ref,refs['full'][:1],rtol=1e-5,atol=1e-3)
        report['models']['existing_rank20_projected']=describe(out/'validation_existing_baseline',old,old_ref,old_times,data,
            'Established rank20 TCN, first10 modes + mean',bands)
        report['models']['existing_rank20_projected']['caution']='One saved path and rep13; longer training lineage; not a matched rank10 training ablation'
    write_json(out/'evaluation.json',report)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for key,label in [('full','Matched rank10 TCN'),('hybrid','Hybrid'),('oracle_remainder','Observed remainder + oscillators'),('existing_baseline','Established rank20 TCN')]:
        p=out/f'validation_{key}/high_resolution_spectra.npz'
        if not p.exists():continue
        with np.load(p) as z:
            f=z['frequency_hz'];mask=(f>=100)&(f<=1500)
            if key=='full':ax.semilogy(f[mask],z['reference_center_psd'][mask],color='black',lw=2,label='Reference reps13–14')
            ax.semilogy(f[mask],z['generated_center_psd'][mask],label=label)
    ax.set(xlabel='Frequency [Hz]',ylabel='Center PSD [µm²/Hz]',title='Unfiltered sums and neural-SDE controls — Welch spacing 3.52 Hz')
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    return report
