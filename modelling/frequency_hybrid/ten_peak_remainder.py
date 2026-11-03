"""Prepare and evaluate the contiguous ten-peak neural remainder target."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt
import torch

from modelling.data_preparation.frequency_components import prepare
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.neural_sde.train import evaluate as evaluate_sde
from .evaluate import describe
from .oscillators import simulate_band
from .study import component_data, write_json
from .ten_peak_oracle import CENTERS_HZ, midpoint_bands


def prepare_data(output):
    output=Path(output)
    manifest=output/'data/manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        np.testing.assert_allclose(saved['bands_hz'],midpoint_bands(CENTERS_HZ),rtol=0,atol=1e-12)
        if saved.get('stored_band_arrays',True):
            raise ValueError('Expected compact remainder-only component cache')
        print(manifest.resolve());return manifest
    data=SharedPODTrainingData('0p10',10,validation_reps=[13,14],test_reps=[15,16],
        spatial_mean_highpass_hz=50.,output_dtype='float64')
    return prepare(data,output/'data',bands=midpoint_bands(CENTERS_HZ),order=4,
                   edge_seconds=.1,store_bands=False)


def remainder_filter(values,sos,edge):
    values=np.asarray(values,dtype=np.float64)
    bands=sum((sosfiltfilt(np.asarray(section),values,axis=-2,padlen=edge)
               for section in sos),start=np.zeros_like(values))
    return (values-bands)[...,edge:-edge,:]


def evaluate(checkpoint,output,study,oscillators,horizon,conditions,ensemble,
             seed,device,diffusion_scale):
    checkpoint,output,study,oscillators=map(Path,(checkpoint,output,study,oscillators))
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
    if generated.shape[-2] <= 2*edge+32768:
        raise ValueError('Rollout is too short for filter guards plus high-resolution PSD')
    filtered=remainder_filter(generated,manifest['sos'],edge)
    raw_cropped=generated[...,edge:-edge,:]
    rem_ref=remainder_reference[:,edge:-edge]
    scored_times=times[:,edge:-edge];length=filtered.shape[-2]
    full_ref=np.stack([full.read_coordinates(name,int(start)+edge,int(start)+edge+length)[1]
                       for name,start in zip(repetitions,starts)])
    bank=np.zeros_like(filtered)
    paths=conditions*ensemble
    artifacts=sorted(oscillators.glob('oscillator_*.npz'))
    if len(artifacts)!=10:
        raise ValueError(f'Expected ten oscillator artifacts, found {len(artifacts)}')
    for index,path in enumerate(artifacts):
        with np.load(path) as saved:model=dict(saved)
        bank+=simulate_band(model,length,paths=paths,seed=201+index).reshape(filtered.shape)
    meta=dict(repetitions=repetitions,start_indices=starts+edge,
        checkpoint=str(checkpoint.resolve()),diffusion_scale=float(diffusion_scale),
        filterbank_bands_hz=manifest['bands_hz'],output_filter_edge_samples=edge,
        pairing='Matched neural paths and starts; oscillator draws are independent stationary samples')
    reports={}
    for key,gen,ref,data,label,initialization in [
        ('remainder_raw',raw_cropped,rem_ref,remainder,'Raw generated remainder',
         'Held-out component history; raw autonomous neural output'),
        ('remainder_filtered',filtered,rem_ref,remainder,'Output-filtered generated remainder',
         'Held-out component history; offline fixed remainder filter applied after generation'),
        ('hybrid_raw',raw_cropped+bank,full_ref,full,'Raw remainder + ten oscillators',
         'Raw neural remainder plus independent stationary oscillators'),
        ('hybrid_filtered',filtered+bank,full_ref,full,'Filtered remainder + ten oscillators',
         'Offline-filtered neural remainder plus independent stationary oscillators')]:
        reports[key]=describe(output/key,gen,ref,scored_times,data,label,manifest['bands_hz'],
            rollout_metadata={**meta,'initialization':initialization,
                'generated_output_filter':'none' if 'raw' in key else
                'same zero-phase subtractive remainder operator as training target; offline only'})
    summary=dict(checkpoint=str(checkpoint.resolve()),horizon=horizon,scored_length=length,
        conditions=conditions,ensemble=ensemble,diffusion_scale=float(diffusion_scale),
        output_filter='offline zero-phase subtractive ten-band remainder operator',reports=reports)
    write_json(output/'summary.json',summary)
    print(json.dumps({key:{metric:value[metric] for metric in
        ('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags')}
        for key,value in reports.items()},indent=2),flush=True)


def evaluate_full_rollout(rollout,output,study,oscillators,seed):
    """Filter an established full-signal rollout once, then add the peak bank."""
    rollout,output,study,oscillators=map(Path,(rollout,output,study,oscillators))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True,exist_ok=True)
    with np.load(rollout) as saved:
        generated=saved['trajectories'].astype(np.float64)[...,:11]
        source_reference=saved['reference'].astype(np.float64)[...,:11]
        source_times=saved['reference_time'].copy()
        repetitions=saved['repetition'].astype(str)
        starts=saved['start_index'].astype(np.int64)
    if generated.shape[-1] != 11 or source_reference.shape[-1] != 11:
        raise ValueError('The source rollout must contain spatial mean plus at least ten POD modes')
    remainder=component_data(study,'remainder');full=component_data(study,'full')
    manifest=remainder.manifest;edge=int(manifest['edge_samples'])
    if generated.shape[-2] <= 2*edge+32768:
        raise ValueError('Rollout is too short for filter guards plus high-resolution PSD')
    # R is applied exactly once to the ordinary full-signal SDE trajectory.
    filtered=remainder_filter(generated,manifest['sos'],edge)
    source_cropped=generated[...,edge:-edge,:]
    length=filtered.shape[-2]
    full_records=[];remainder_records=[];scored_times=[]
    for name,start in zip(repetitions,starts):
        time,values=full.read_coordinates(name,int(start),int(start)+length)
        full_records.append(values);scored_times.append(time)
        remainder_records.append(remainder.read_coordinates(name,int(start),int(start)+length)[1])
    full_ref=np.stack(full_records);rem_ref=np.stack(remainder_records)
    scored_times=np.stack(scored_times)
    np.testing.assert_allclose(source_reference[:,edge:-edge],full_ref,rtol=1e-5,atol=1e-3)
    np.testing.assert_allclose(source_times[:,edge:-edge],scored_times,rtol=0,atol=1e-12)
    filtered_source_reference=remainder_filter(source_reference,manifest['sos'],edge)
    reference_filter_error=float(np.linalg.norm(filtered_source_reference-rem_ref)/
                                 max(np.linalg.norm(rem_ref),1e-30))
    paths=generated.shape[0]*generated.shape[1]
    bank=np.zeros_like(filtered)
    artifacts=sorted(oscillators.glob('oscillator_*.npz'))
    if len(artifacts)!=10:
        raise ValueError(f'Expected ten oscillator artifacts, found {len(artifacts)}')
    for index,path in enumerate(artifacts):
        with np.load(path) as saved:model=dict(saved)
        bank+=simulate_band(model,length,paths=paths,seed=seed+index).reshape(filtered.shape)
    oracle=np.repeat(rem_ref[:,None],generated.shape[1],axis=1)+bank
    meta=dict(repetitions=repetitions,start_indices=starts,
        source_rollout=str(rollout.resolve()),source_coordinate_count=21,
        evaluated_coordinates='spatial mean plus first ten POD modes',
        filterbank_bands_hz=manifest['bands_hz'],output_filter_edge_samples=edge,
        source_reference_filter_relative_error=reference_filter_error,
        pairing='Matched validation segment; oscillator draws are independent stationary samples')
    reports={}
    for key,gen,ref,data,label,initialization,filtering in [
        ('source_full',source_cropped,full_ref,full,'Established full-signal TCN',
         'Existing saved autonomous rollout','none; only guard samples cropped'),
        ('source_remainder',filtered,rem_ref,remainder,'Once-filtered established TCN',
         'Existing saved autonomous rollout','one offline ten-band subtractive remainder filter'),
        ('hybrid',filtered+bank,full_ref,full,'Once-filtered TCN + ten oscillators',
         'Existing saved autonomous rollout plus independent stationary oscillators',
         'one offline ten-band subtractive remainder filter before addition'),
        ('oracle',oracle,full_ref,full,'Observed remainder + ten oscillators',
         'Held-out measured remainder plus independent stationary oscillators','none')]:
        reports[key]=describe(output/key,gen,ref,scored_times,data,label,manifest['bands_hz'],
            rollout_metadata={**meta,'initialization':initialization,
                'generated_output_filter':filtering})
    summary=dict(source_rollout=str(rollout.resolve()),scored_length=length,
        conditions=generated.shape[0],ensemble=generated.shape[1],
        source_reference_filter_relative_error=reference_filter_error,reports=reports)
    write_json(output/'summary.json',summary)
    print(json.dumps({'reference_filter_relative_error':reference_filter_error,
        **{key:{metric:value[metric] for metric in
        ('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags',
         'increment_variance_ratio')} for key,value in reports.items()}},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--checkpoint',type=Path,required=True)
    ev.add_argument('--output',type=Path,required=True);ev.add_argument('--study',type=Path,required=True)
    ev.add_argument('--oscillators',type=Path,required=True);ev.add_argument('--horizon',type=int,default=60000)
    ev.add_argument('--conditions',type=int,default=2);ev.add_argument('--ensemble',type=int,default=4)
    ev.add_argument('--seed',type=int,default=41);ev.add_argument('--device',default='cuda')
    ev.add_argument('--diffusion-scale',type=float,default=1.)
    full=sub.add_parser('evaluate-full-rollout')
    full.add_argument('--rollout',type=Path,required=True)
    full.add_argument('--output',type=Path,required=True)
    full.add_argument('--study',type=Path,required=True)
    full.add_argument('--oscillators',type=Path,required=True)
    full.add_argument('--seed',type=int,default=301)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='prepare':prepare_data(args.output)
    elif args.command=='evaluate':
        evaluate(args.checkpoint,args.output,args.study,args.oscillators,args.horizon,
                 args.conditions,args.ensemble,args.seed,args.device,args.diffusion_scale)
    else:
        evaluate_full_rollout(args.rollout,args.output,args.study,args.oscillators,args.seed)


if __name__=='__main__':main()
