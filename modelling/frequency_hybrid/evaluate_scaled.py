"""Evaluate a diffusion-scaled remainder rollout and its oscillator sum."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modelling.neural_sde.train import evaluate as evaluate_sde
from .evaluate import describe
from .oscillators import simulate_band
from .study import component_data, write_json


def evaluate_scaled(study, checkpoint, output, diffusion_scale, horizon=50000,
                    conditions=2, ensemble=4, seed=41, device='cuda'):
    study,checkpoint,output=map(Path,(study,checkpoint,output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {output}')
    output.mkdir(parents=True,exist_ok=True)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    data_cfg=payload.get('data_config',{})
    if data_cfg.get('temporal_component')!='remainder':
        raise ValueError('Checkpoint is not trained on the frequency remainder')
    raw=output/'remainder_raw'
    evaluate_sde(checkpoint,raw,split='validation',num_conditions=conditions,
        ensemble_size=ensemble,horizon=horizon,batch_size=min(conditions,8),seed=seed,
        device=device,metrics=True,diffusion_scale=diffusion_scale)
    with np.load(raw/'rollout.npz') as saved:
        generated=saved['trajectories'].astype(float)
        remainder_reference=saved['reference'].astype(float)
        times=saved['reference_time'].copy()
        repetitions=saved['repetition'].astype(str)
        starts=saved['start_index'].astype(np.int64)
    remainder=component_data(study,'remainder');full=component_data(study,'full')
    length=generated.shape[2]
    full_reference=np.stack([full.read_coordinates(name,int(start),int(start)+length)[1]
                             for name,start in zip(repetitions,starts)])
    oscillators=[]
    paths=generated.shape[0]*generated.shape[1]
    for i in range(3):
        with np.load(study/f'oscillator_{i}.npz') as saved:
            model=dict(saved)
        oscillators.append(simulate_band(model,length,paths=paths,seed=101+i).reshape(generated.shape))
    hybrid=generated+sum(oscillators)
    common=dict(repetitions=repetitions,start_indices=starts,
        checkpoint=str(checkpoint.resolve()),diffusion_scale=float(diffusion_scale),
        initialization='Neural remainder conditioned on held-out validation history; no burn-in',
        pairing='Generated remainder shares the saved conditional start; oscillators are independent stationary draws')
    rem=describe(output/'remainder',generated,remainder_reference,times,remainder,
        f'Remainder TCN, diffusion scale {diffusion_scale:g}',full.manifest['bands_hz'],
        rollout_metadata=common)
    hyb=describe(output/'hybrid',hybrid,full_reference,times,full,
        f'Hybrid, diffusion scale {diffusion_scale:g}',full.manifest['bands_hz'],
        rollout_metadata=common)
    summary=dict(diffusion_scale=float(diffusion_scale),checkpoint=str(checkpoint.resolve()),
        conditions=conditions,ensemble=ensemble,horizon=horizon,remainder=rem,hybrid=hyb)
    write_json(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path('runs/diagnostics/frequency_hybrid_0p10_rank10_v1'))
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--diffusion-scale',type=float,required=True)
    parser.add_argument('--horizon',type=int,default=50000)
    parser.add_argument('--conditions',type=int,default=2)
    parser.add_argument('--ensemble',type=int,default=4)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.diffusion_scale<=0 or args.horizon<1 or min(args.conditions,args.ensemble)<1:
        parser.error('Positive scale, horizon, condition count and ensemble size required')
    torch.set_num_threads(1)
    evaluate_scaled(args.study,args.checkpoint,args.output,args.diffusion_scale,
                    args.horizon,args.conditions,args.ensemble,args.seed,args.device)


if __name__=='__main__':
    main()
