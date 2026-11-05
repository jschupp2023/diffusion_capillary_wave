"""Prepare and evaluate the rank-one local-excess neural remainder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch

from modelling.data_preparation.frequency_components import sha256, settling_samples
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.neural_sde.train import evaluate as evaluate_sde
from .evaluate import describe
from .local_excess import projected_band, projected_band_array, simulate_complex_band
from .study import component_data, write_json


def load_bank(path, *, expected=None):
    """Load and validate the fixed complex oscillator artifacts."""
    path=Path(path).resolve();artifacts=sorted(path.glob('band_*.npz'))
    if expected is not None and len(artifacts)!=expected:
        raise ValueError(f'Expected {expected} band artifacts in {path}, found {len(artifacts)}')
    if not artifacts:
        raise ValueError(f'No band artifacts found in {path}')
    models=[]
    for artifact in artifacts:
        with np.load(artifact) as saved:model={key:saved[key] for key in saved.files}
        modes=np.asarray(model['modes'])
        if modes.ndim != 2 or modes.shape[1] != 1:
            raise ValueError(f'{artifact} is not a rank-one model')
        models.append(model)
    return artifacts,models


def bank_filters(models, fs, default_order=4):
    filters=[]
    for model in models:
        if not np.isclose(float(model['fs']),fs,rtol=1e-10):
            raise ValueError('Oscillator and dataset sampling frequencies differ')
        order=int(model.get('filter_order',default_order))
        filters.append(butter(order,np.asarray(model['edges_hz'],dtype=float),
                              btype='bandpass',fs=fs,output='sos'))
    return filters


def decompose(values, models, filters, edge):
    """Exact local-excess decomposition of one complete coordinate record."""
    values=np.asarray(values,dtype=np.float64)
    bands=[projected_band(values,sos,model['modes'],edge)[0]
           for model,sos in zip(models,filters)]
    full=values[edge:-edge];band_sum=sum(bands,start=np.zeros_like(full))
    remainder=full-band_sum
    error=float(np.linalg.norm(remainder+band_sum-full)/max(np.linalg.norm(full),1e-30))
    if error>1e-12:
        raise RuntimeError('Local-excess reconstruction failed')
    return full,remainder,error


def remainder_filter(values, models, filters, edge):
    """Apply the fixed local-excess complement to generated trajectories."""
    values=np.asarray(values,dtype=np.float64)
    full=values[...,edge:-edge,:]
    projected=sum((projected_band_array(values,sos,model['modes'],edge)[0]
                   for model,sos in zip(models,filters)),start=np.zeros_like(full))
    return full-projected


def prepare_data(output, oscillators, *, experiment='0p10', include_test=False, edge_seconds=.1,
                 data_root=None, shared_basis=None, rank=10, spatial_mean_highpass_hz=None):
    output=Path(output).resolve();manifest=output/'data/manifest.json'
    artifacts,models=load_bank(oscillators)
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        expected=[sha256(path) for path in artifacts]
        if saved.get('local_excess',{}).get('artifact_sha256')!=expected:
            raise ValueError('Existing cache uses different local-excess artifacts')
        print(manifest);return manifest
    kwargs = dict(
        data_root=data_root,
        shared_basis=shared_basis,
        validation_reps=[13, 14],
        test_reps=[15, 16],
        spatial_mean_highpass_hz=spatial_mean_highpass_hz,
        output_dtype='float64',
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    data=SharedPODTrainingData(experiment,rank,**kwargs)
    fs=1/data.native_dt;filters=bank_filters(models,fs)
    edge=max(int(np.ceil(edge_seconds*fs)),*(settling_samples(sos,fs) for sos in filters))
    target=output/'data';target.mkdir(parents=True,exist_ok=False)
    bands=[np.asarray(model['edges_hz'],dtype=float).tolist() for model in models]
    value=dict(version=1,parent_config=data.source_config,parent_signature=data.signature(),
        bands_hz=bands,filter_order=[int(model.get('filter_order',4)) for model in models],
        sos=[sos.tolist() for sos in filters],
        edge_samples=edge,fs=fs,filter_method='sosfiltfilt + Hilbert + complex rank-one projection',
        initialization='offline decomposition; not a causal forecast',stored_band_arrays=False,
        local_excess=dict(oscillator_directory=str(Path(oscillators).resolve()),
            artifacts=[path.name for path in artifacts],artifact_sha256=[sha256(path) for path in artifacts],
            complex_ranks=[int(np.asarray(model['modes']).shape[1]) for model in models],
            definition='full minus sum of the ten filtered complex spatial projections'),records={})
    names=data.train_repetitions+data.validation_repetitions
    if include_test:names+=data.test_repetitions
    try:
        for name in names:
            time,values=data.read_coordinates(name,0,data.frame_counts[name])
            full,remainder,error=decompose(values,models,filters,edge)
            files={}
            for component,array in [('time',time[edge:-edge]),('full',full),('remainder',remainder)]:
                path=target/f'{name}_{component}.npy';np.save(path,array,allow_pickle=False)
                files[component]=dict(path=path.name,sha256=sha256(path))
            value['records'][name]=dict(frames=len(full),files=files,
                reconstruction_relative_error=error,padding_sensitivity_relative=[])
            print(f'Prepared {name}: {len(full)} frames, error={error:.2g}',flush=True)
        manifest.write_text(json.dumps(value,indent=2)+'\n')
    except Exception:
        # A partial directory must never be mistaken for a valid immutable cache.
        if manifest.exists():manifest.unlink()
        raise
    print(manifest);return manifest


def simulate_bank(models, filters, edge, length, shape, seed):
    paths=int(np.prod(shape[:-2]));total=np.zeros((paths,length,shape[-1]),dtype=np.float64)
    for index,(model,sos) in enumerate(zip(models,filters)):
        generated=simulate_complex_band(model,length+2*edge,paths=paths,seed=seed+index)
        generated=sosfiltfilt(sos,generated,axis=-2,padlen=edge)[...,edge:-edge,:]
        total+=generated
    return total.reshape(*shape[:-2],length,shape[-1])


def evaluate(checkpoint, output, study, oscillators, horizon, conditions, ensemble,
             seed, device, diffusion_scale):
    checkpoint,output,study=map(Path,(checkpoint,output,study))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True,exist_ok=True)
    raw=output/'raw_rollout'
    evaluate_sde(checkpoint,raw,split='validation',num_conditions=conditions,
        ensemble_size=ensemble,horizon=horizon,batch_size=min(conditions,4),seed=seed,
        device=device,metrics=False,diffusion_scale=diffusion_scale)
    with np.load(raw/'rollout.npz') as saved:
        generated=saved['trajectories'].astype(np.float64)
        remainder_reference=saved['reference'].astype(np.float64)
        times=saved['reference_time'].copy();repetitions=saved['repetition'].astype(str)
        starts=saved['start_index'].astype(np.int64)
    remainder=component_data(study,'remainder');full=component_data(study,'full')
    manifest=remainder.manifest;edge=int(manifest['edge_samples'])
    artifacts,models=load_bank(oscillators);filters=bank_filters(models,float(manifest['fs']))
    expected=manifest['local_excess']['artifact_sha256']
    if [sha256(path) for path in artifacts]!=expected:
        raise ValueError('Evaluation bank differs from the decomposition bank')
    if generated.shape[-2]<=2*edge+32768:
        raise ValueError('Rollout is too short for filter guards plus high-resolution PSD')
    filtered=remainder_filter(generated,models,filters,edge)
    raw_cropped=generated[...,edge:-edge,:]
    rem_ref=remainder_reference[:,edge:-edge];scored_times=times[:,edge:-edge]
    length=filtered.shape[-2]
    full_ref=np.stack([full.read_coordinates(name,int(start)+edge,int(start)+edge+length)[1]
                       for name,start in zip(repetitions,starts)])
    bank=simulate_bank(models,filters,edge,length,filtered.shape,seed+1000)
    meta=dict(repetitions=repetitions,start_indices=starts+edge,
        checkpoint=str(checkpoint.resolve()),diffusion_scale=float(diffusion_scale),
        filterbank_bands_hz=manifest['bands_hz'],output_filter_edge_samples=edge,
        decomposition='ten local spectral-excess rank-one complex projections',
        pairing='Matched neural paths and starts; oscillator draws are independent stationary samples')
    reports={}
    cases=[
        ('remainder_raw',raw_cropped,rem_ref,remainder,'Raw generated local-excess remainder','none'),
        ('remainder_filtered',filtered,rem_ref,remainder,'Projected generated local-excess remainder','fixed local-excess complement'),
        ('hybrid_raw',raw_cropped+bank,full_ref,full,'Raw remainder + complex oscillator bank','none'),
        ('hybrid_filtered',filtered+bank,full_ref,full,'Projected remainder + complex oscillator bank','fixed local-excess complement')]
    for key,gen,ref,data,label,filtering in cases:
        reports[key]=describe(output/key,gen,ref,scored_times,data,label,manifest['bands_hz'],
            rollout_metadata={**meta,'initialization':'Held-out remainder history; autonomous neural rollout; stationary oscillator draws',
                'generated_output_filter':filtering})
    summary=dict(checkpoint=str(checkpoint.resolve()),horizon=horizon,scored_length=length,
        conditions=conditions,ensemble=ensemble,diffusion_scale=float(diffusion_scale),
        decomposition_manifest=str((study/'data/manifest.json').resolve()),reports=reports)
    write_json(output/'summary.json',summary)
    print(json.dumps({key:{metric:value[metric] for metric in
        ('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags',
         'increment_variance_ratio')} for key,value in reports.items()},indent=2),flush=True)


def augment_rollout(rollout, output, study, oscillator, seed):
    """Add one independently fitted complex oscillator to an existing hybrid rollout."""
    rollout,output,study,oscillator=map(Path,(rollout,output,study,oscillator))
    if rollout.is_dir():rollout=rollout/'rollout.npz'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    with np.load(rollout) as saved:
        generated=saved['trajectories'].astype(np.float64)
        reference=saved['reference'].astype(np.float64)
        times=saved['reference_time'].copy();repetitions=saved['repetition'].astype(str)
        starts=saved['start_index'].astype(np.int64)
    with np.load(oscillator) as saved:model={key:saved[key] for key in saved.files}
    fs=float(model['fs']);order=int(model.get('filter_order',4))
    sos=butter(order,np.asarray(model['edges_hz'],dtype=float),btype='bandpass',fs=fs,output='sos')
    edge=max(int(np.ceil(.1*fs)),settling_samples(sos,fs))
    paths=int(np.prod(generated.shape[:-2]));length=generated.shape[-2]
    added=simulate_complex_band(model,length+2*edge,paths=paths,seed=seed)
    added=sosfiltfilt(sos,added,axis=-2,padlen=edge)[...,edge:-edge,:].reshape(generated.shape)
    data=component_data(study,'full');bands=data.manifest['bands_hz']
    new_band=np.asarray(model['edges_hz'],dtype=float).tolist()
    if new_band[1]>bands[0][0]+1e-9:
        raise ValueError('Added oscillator overlaps the existing first band')
    report=describe(output,generated+added,reference,times,data,
        f"Existing hybrid + {float(model['frequency_hz']):.1f} Hz oscillator",
        [new_band]+bands,rollout_metadata=dict(repetitions=repetitions,start_indices=starts,
            initialization='Saved hybrid plus one independent stationary complex oscillator',
            pairing='Same reference and existing hybrid paths; only the added oscillator draw is new',
            source_rollout=str(rollout.resolve()),added_oscillator_artifact=str(oscillator.resolve()),
            generated_output_filter='Existing hybrid unchanged; fixed filter applied to added oscillator'))
    print(json.dumps({key:report[key] for key in ('center_psd_mae_db','center_band_power_ratio',
        'acf_rmse_first_4000_lags','increment_variance_ratio')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--oscillators',type=Path,required=True);prep.add_argument('--experiment',default='0p10')
    prep.add_argument('--rank',type=int,default=10)
    prep.add_argument('--data-root',type=Path,default=None)
    prep.add_argument('--shared-basis',type=Path,default=None)
    prep.add_argument('--spatial-mean-highpass-hz',type=float,default=None)
    prep.add_argument('--include-test',action='store_true')
    ev=sub.add_parser('evaluate');ev.add_argument('--checkpoint',type=Path,required=True)
    ev.add_argument('--output',type=Path,required=True);ev.add_argument('--study',type=Path,required=True)
    ev.add_argument('--oscillators',type=Path,required=True);ev.add_argument('--horizon',type=int,default=60000)
    ev.add_argument('--conditions',type=int,default=2);ev.add_argument('--ensemble',type=int,default=4)
    ev.add_argument('--seed',type=int,default=41);ev.add_argument('--device',default='cuda')
    ev.add_argument('--diffusion-scale',type=float,default=1.)
    aug=sub.add_parser('augment-rollout');aug.add_argument('--rollout',type=Path,required=True)
    aug.add_argument('--output',type=Path,required=True);aug.add_argument('--study',type=Path,required=True)
    aug.add_argument('--oscillator',type=Path,required=True);aug.add_argument('--seed',type=int,default=1551)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='prepare':prepare_data(args.output,args.oscillators,experiment=args.experiment,
                                            rank=args.rank,data_root=args.data_root,shared_basis=args.shared_basis,
                                            spatial_mean_highpass_hz=args.spatial_mean_highpass_hz,
                                            include_test=args.include_test)
    elif args.command=='evaluate':
        evaluate(args.checkpoint,args.output,args.study,args.oscillators,args.horizon,
                 args.conditions,args.ensemble,args.seed,args.device,args.diffusion_scale)
    else:augment_rollout(args.rollout,args.output,args.study,args.oscillator,args.seed)


if __name__=='__main__':main()
