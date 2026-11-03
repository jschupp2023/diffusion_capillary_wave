"""Small reproducible offline temporal-separation experiment; no HDF5 writes."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, sosfreqz

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.frequency_components import (
    DEFAULT_BANDS, FrequencyComponentData, prepare)
from modelling.latent_oscillators.model import welch_matrix
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.train import train, load_model
from .oscillators import fit_band, simulate_band, oscillator_psd


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def component_data(output, component):
    path=Path(output)/'data/manifest.json'
    meta=json.loads(path.read_text())
    return FrequencyComponentData(temporal_component_manifest=path,
        temporal_component=component, **meta['parent_config'])


def matrices(records, fs, nperseg=8192):
    total=None
    for values in records:
        f,s=welch_matrix(values[None],fs,nperseg)
        total=s[0] if total is None else total+s[0]
    return f,total/len(records)


def records(data, split):
    return [data.read_coordinates(n,0,data.frame_counts[n])[1] for n in data.splits[split]]


def center_weights(data):
    with h5py.File(data.basis_path) as h:
        modes=h['pod/modes'][:data.rank]
    return np.r_[1/np.sqrt(data.n_space), modes[:,modes.shape[1]//2,modes.shape[2]//2]]


def projected(s, weights):
    return np.einsum('i,fij,j->f',weights,s,weights).real


def spectral_comparison(f, estimate, truth, weights, bands, evaluation_band=(100.,1500.)):
    mask=(f>=evaluation_band[0])&(f<=evaluation_band[1])
    p=np.maximum(projected(estimate,weights),1e-30)
    t=np.maximum(projected(truth,weights),1e-30)
    diagonal=np.diagonal(truth,axis1=-2,axis2=-1).real
    normal=np.sqrt(np.maximum(diagonal[:,:,None]*diagonal[:,None,:],1e-60))
    # Restrict coherence evaluation to bins/coordinates carrying real power.
    dp=np.maximum(np.diagonal(estimate,axis1=-2,axis2=-1).real,1e-30)
    coh_p=estimate/np.sqrt(dp[:,:,None]*dp[:,None,:])
    coh_t=truth/normal
    off=~np.eye(len(weights),dtype=bool)
    result=dict(evaluation_band_hz=list(evaluation_band),
        center_psd_mae_db=float(np.mean(abs(10*np.log10(p[mask]/t[mask])))),
        center_band_power_ratio=float(p[mask].sum()/t[mask].sum()),
        normalized_csd_rmse=float(np.sqrt(np.mean(abs((estimate-truth)[mask]/normal[mask])**2))),
        coherence_complex_rmse=float(np.sqrt(np.mean(abs((coh_p-coh_t)[mask][:,off])**2))),
        selected_bands=[])
    for low,high in bands:
        inside=(f>=low)&(f<=high);idx=np.flatnonzero(inside)
        result['selected_bands'].append(dict(edges_hz=[low,high],
            power_ratio=float(p[inside].sum()/t[inside].sum()),
            reference_max_hz=float(f[idx[np.argmax(t[inside])]]),
            generated_max_hz=float(f[idx[np.argmax(p[inside])]])))
    return result


def prepare_and_fit(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    data=SharedPODTrainingData('0p10',10,validation_reps=[13,14],test_reps=[15,16],
        spatial_mean_highpass_hz=50.,output_dtype='float64')
    manifest_path=out/'data/manifest.json'
    if not manifest_path.exists():
        prepare(data,out/'data',bands=args.bands,order=args.order,edge_seconds=args.edge_seconds)
    else:
        saved=json.loads(manifest_path.read_text())
        if saved['bands_hz']!=args.bands or saved['filter_order']!=args.order:
            raise ValueError('Existing decomposition has different filters; choose a new output directory')
    full=component_data(out,'full');weights=center_weights(full);fs=1/full.native_dt
    bands=np.asarray(full.manifest['bands_hz']);n=len(bands)
    report=dict(protocol=full.temporal_preprocessing(),split='training only for fitting',
                coordinates=11,rank=10,spatial_mean_included=True,
                uncertainty='existing shared POD basis includes held-out repetitions',
                splits={},oscillators=[])
    arrays={};targets={}
    for split in ('train','validation'):
        f,sfull=matrices(records(full,split),fs)
        independent=np.zeros_like(sfull)
        targets[split]={}
        for name in ['remainder']+[f'band{i}' for i in range(n)]:
            source=component_data(out,name);xs=records(source,split)
            _,s=matrices(xs,fs);independent+=s;targets[split][name]=s
            arrays[f'{split}_{name}_center']=projected(s,weights)
            if split=='train' and name.startswith('band'):
                i=int(name[4:]);fit_f,fit_s=matrices(xs,fs,32768)
                fitted=fit_band(xs,fit_f,fit_s,fs,bands[i])
                np.savez(out/f'oscillator_{i}.npz',**fitted)
                report['oscillators'].append({k:(v.tolist() if isinstance(v,np.ndarray) else v)
                    for k,v in fitted.items() if k not in ('factor','covariance')})
        report['splits'][split]=spectral_comparison(f,independent,sfull,weights,bands.tolist())
        report['splits'][split]['trace_power_ratio']=float(
            np.trace(independent,axis1=-2,axis2=-1).real.sum()/
            np.trace(sfull,axis1=-2,axis2=-1).real.sum())
        arrays[f'{split}_full_center']=projected(sfull,weights)
        arrays[f'{split}_independent_center']=projected(independent,weights)
    arrays['frequency_hz']=f
    report['reconstruction_max_relative_error']=max(r['reconstruction_relative_error'] for r in full.manifest['records'].values())
    report['padding_sensitivity_max_relative']=np.max([r['padding_sensitivity_relative'] for r in full.manifest['records'].values()]).item()
    write_json(out/'preflight.json',report);np.savez_compressed(out/'preflight_spectra.npz',**arrays)
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    mask=(f>=100)&(f<=1500)
    axes[0].semilogy(f[mask],arrays['train_full_center'][mask],label='Original train signal',color='black')
    axes[0].semilogy(f[mask],arrays['train_independent_center'][mask],label='Sum of measured component autospectra',ls='--')
    for name in ['remainder']+[f'band{i}' for i in range(n)]:
        axes[1].semilogy(f[mask],arrays[f'train_{name}_center'][mask],label=name)
    for ax in axes:
        ax.legend();ax.grid(alpha=.2);ax.set(xlabel='Frequency [Hz]',ylabel='Center PSD [µm²/Hz]')
    fig.savefig(out/'decomposition.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for i,sos in enumerate(full.manifest['sos']):
        ff,h=sosfreqz(sos,worN=32768,fs=fs);m=(ff>=100)&(ff<=1500)
        ax.plot(ff[m],abs(h[m])**2,label=f'band {i+1} effective amplitude')
    ax.set(xlabel='Frequency [Hz]',ylabel='Forward/backward amplitude response');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'filters.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


def train_pair(args):
    out=args.output
    for component in ('full','remainder'):
        target=out/f'tcn_{component}'
        if (target/'best.pt').exists():
            logged=[json.loads(line) for line in (target/'metrics.jsonl').read_text().splitlines()]
            if len(logged)<args.epochs:
                raise RuntimeError(f'Partial training in {target}; continue explicitly or choose a new study output')
            print(f'Using existing checkpoint {target}',flush=True);continue
        data=component_data(out,component)
        cfg=SDEConfig(data.n_coordinates,data.native_dt,state_variable='velocity',
            drift_type='damped_residual',residual_architecture='tcn',
            tcn_context_frames=63,tcn_channels=64,diffusion_type='diagonal')
        start=time.monotonic()
        train(data,target,cfg,epochs=args.epochs,batch_size=256,learning_rate=1e-4,
            device=args.device,seed=0,train_trim_percent=0.,indexed_training_batches=True,
            tcn_rollout_cache=True,tcn_rollout_compile=False,
            multistep_weight=.03,multistep_objective='gmm_nll',multistep_target='velocity',
            multistep_horizons=(2,5,10,20),multistep_particles=8,
            multistep_window_fraction=.005,multistep_window_sampling='batches',
            stationary_mean_weight=.03,stationary_variance_weight=.001,
            stationary_mean_horizons=(20,50,100),stationary_mean_trajectories=64,
            stationary_mean_batch_fraction=.0125)
        print(f'{component} training seconds: {time.monotonic()-start:.1f}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','evaluate'])
    p.add_argument('--output',type=Path,default=Path('runs/diagnostics/frequency_hybrid_0p10_rank10_v1'))
    p.add_argument('--bands',type=float,nargs=6,default=np.asarray(DEFAULT_BANDS).ravel().tolist())
    p.add_argument('--order',type=int,default=4);p.add_argument('--edge-seconds',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--device',default='cuda')
    args=p.parse_args();args.bands=np.asarray(args.bands).reshape(3,2).tolist()
    torch.set_num_threads(1)
    if args.stage=='prepare': prepare_and_fit(args)
    elif args.stage=='train': train_pair(args)
    else:
        from .evaluate import evaluate
        evaluate(args.output)


if __name__=='__main__': main()
