"""Training-only band-width sensitivity at common retained endpoints."""
import argparse
from pathlib import Path
import numpy as np
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.frequency_components import design_filters,settling_samples,decompose
from .oscillators import fit_band
from .study import component_data, matrices, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('runs/diagnostics/frequency_hybrid_0p10_rank10_v1'))
    args=p.parse_args();full=component_data(args.output,'full');fs=1/full.native_dt
    parent=SharedPODTrainingData(**full.manifest['parent_config'])
    nominal=np.asarray(full.manifest['bands_hz']);centers=nominal.mean(1);half=np.diff(nominal,axis=1)[:,0]/2
    designs={}
    for scale in (.8,1.,1.2):
        bands=np.column_stack((centers-scale*half,centers+scale*half))
        sos=design_filters(fs,bands,full.manifest['filter_order'])
        designs[scale]=(bands,sos)
    edge=max(settling_samples(s,fs) for _,ss in designs.values() for s in ss)
    cached={scale:[[],[],[]] for scale in designs}
    for n in parent.train_repetitions:
        _,x=parent.read_coordinates(n,0,parent.frame_counts[n])
        for scale,(_,sos) in designs.items():
            parts,_=decompose(x,sos,edge)
            for i in range(3):cached[scale][i].append(parts[f'band{i}'])
    report=dict(split='train',edge_samples=edge,variants={})
    for scale,(bands,_) in designs.items():
        fits=[]
        for i in range(3):
            f,s=matrices(cached[scale][i],fs,32768)
            fit=fit_band(cached[scale][i],f,s,fs,bands[i])
            fits.append(dict(frequency_hz=fit['frequency_hz'],gamma_per_second=fit['gamma_per_second'],
                             effective_fwhm_hz=fit['approximate_fwhm_hz']))
        report['variants'][str(scale)]=fits
    write_json(args.output/'filter_width_sensitivity.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
