"""Complete the valley background with explicitly selected training-only anchors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def interpolate(f, anchor_f, anchor_s):
    right = np.searchsorted(anchor_f, f, side='right').clip(1, len(anchor_f)-1)
    alpha = ((f-anchor_f[right-1])/(anchor_f[right]-anchor_f[right-1])).clip(0,1)
    return (1-alpha[:,None,None])*anchor_s[right-1]+alpha[:,None,None]*anchor_s[right]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('runs/diagnostics/spectral_background_0p10_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--anchors',nargs='+',type=float,default=[88.,116.,148.,183.])
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory')
    with np.load(args.source/'backgrounds.npz') as a:
        f,train,val=a['frequency_hz'],a['train'],a['validation']
        wide_f,wide_s=a['anchor_frequencies_1'],a['anchor_matrices_1']
        scale,mean,peaks=a['scale'],a['mean'],a['peak_centers_hz']
    centers=np.array([np.argmin(abs(f-x)) for x in args.anchors])
    if np.any(centers<2) or np.any(centers>len(f)-3) or np.any(np.diff(centers)<=2) or f[centers[-1]+2]>=wide_f[0]:
        raise ValueError('Manual anchors must be distinct, interior, and below the first automatic anchor')
    def estimate(indices):
        windows=indices[:,None]+np.arange(-1,2)[None,:]
        af=np.r_[f[windows].mean(1),wide_f]
        matrices=np.concatenate([train[windows].mean(1),wide_s])
        return interpolate(f,af,matrices),af,matrices,windows
    background,af,matrices,windows=estimate(centers)
    variations=[]
    for j in range(len(centers)):
        for shift in [-1,1]:
            moved=centers.copy();moved[j]+=shift
            variations.append(estimate(moved)[0])
    variants=np.stack(variations)
    diagonal=lambda x:np.diagonal(x,axis1=-2,axis2=-1).real.clip(1e-30)
    db=10*np.log10(diagonal(background))
    variant_db=10*np.log10(diagonal(variants))
    uncertainty=np.max(abs(variant_db-db[None]),axis=0)
    supported=(f>=af[0])&(f<=af[-1])
    # Common support across all shifted anchors, limited to affected low frequencies.
    low=(f>=f[centers[0]+1])&(f<=wide_f[0])
    eig=np.linalg.eigvalsh(background)
    assert eig.min()>=-1e-12
    np.testing.assert_allclose(background,background.conj().transpose(0,2,1),atol=1e-12)
    assert np.allclose(np.diagonal(background,axis1=-2,axis2=-1).imag,0)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'background.npz',frequency_hz=f,background=background,
        train=train,validation=val,mean=mean,scale=scale,supported=supported,
        anchor_frequencies_hz=af,anchor_matrices=matrices,manual_anchor_windows=windows,
        perturbation_backgrounds=variants,peak_centers_hz=peaks)
    summary=dict(manual_anchors_hz=f[centers].tolist(),supported_interval_hz=[float(af[0]),float(af[-1])],
        minimum_background_eigenvalue=float(eig.min()),sensitivity_interval_hz=[float(f[low][0]),float(f[low][-1])],
        per_coordinate=[dict(coordinate=j,median_worst_shift_db=float(np.median(uncertainty[low,j])),
                            p90_worst_shift_db=float(np.quantile(uncertainty[low,j],.9)),
                            maximum_worst_shift_db=float(uncertainty[low,j].max())) for j in range(4)])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'protocol.json').write_text(json.dumps(dict(source=str(args.source.resolve()),
        source_sha256=hashlib.sha256((args.source/'backgrounds.npz').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_protocol=json.loads((args.source/'protocol.json').read_text()),
        manual_anchors_requested_hz=args.anchors,manual_anchors_actual_hz=f[centers].tolist(),
        selection='Selected manually from training PSDs of mean/POD1-3; identical frequency windows for full matrix. No validation selection.',
        high_frequency='Unchanged automatic +/-18 Hz exclusion anchors, starting at 260.16 Hz.',
        averaging='Three neighboring bins, full complex matrix average, then convex linear interpolation.',
        sensitivity='Each manual anchor moved one Welch bin (~3.52 Hz) left/right separately, eight variants; automatic anchors fixed.',
        limitations=['Heuristic background, not uniquely identified physical noise.',
                    'Constant endpoint extensions outside the supported interval are excluded from the recommended fitting mask.',
                    'Positive background does not imply data minus background is positive semidefinite.',
                    'Inherited uncertainty of the high-frequency background remains; no oscillator/neural training.']),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for zoom in [True,False]:
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        for j,ax in enumerate(axes.flat):
            ax.plot(f,10*np.log10(diagonal(val)[:,j]),color='.7',lw=.8,label='validation (not fitted)')
            ax.plot(f,10*np.log10(diagonal(train)[:,j]),color='k',lw=1,label='training')
            ax.fill_between(f,variant_db[:,:,j].min(0),variant_db[:,:,j].max(0),color='#0072B2',alpha=.22,label='anchor shift sensitivity')
            ax.plot(f,np.where(supported,db[:,j],np.nan),color='#0072B2',lw=2,label='working background')
            ax.plot(f,np.where(~supported,db[:,j],np.nan),color='#0072B2',ls=':',lw=1)
            ax.scatter(af,10*np.log10(diagonal(matrices)[:,j]),color='#0072B2',s=18,zorder=4)
            ax.scatter(f[centers],db[centers,j],color='#D55E00',marker='s',s=30,zorder=5,label='manual anchors')
            ax.set_title('Spatial mean' if j==0 else f'POD {j}');ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('PSD [dB, standardized units/Hz]');ax.set_xlim(80,400 if zoom else 1500)
            if zoom:
                local=(f>=80)&(f<=400)
                lo=min(db[local,j].min(),10*np.log10(diagonal(train)[local,j]).min())
                hi=10*np.log10(diagonal(train)[local,j]).max();ax.set_ylim(lo-2,hi+2)
        axes[0,0].legend(fontsize=7);fig.tight_layout()
        fig.savefig(args.output/('low_frequency.png' if zoom else 'background_comparison.png'),dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
