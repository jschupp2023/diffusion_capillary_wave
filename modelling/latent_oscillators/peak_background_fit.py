"""Fit condition-constrained resonances above a frozen empirical background.

The additive spectral background is an identification aid, not a dynamical OU
state or a guarantee that a future neural residual will preserve these spectra.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .observed_spectral import ObservedSpectralModel
from .observed_spectral_study import initial_basis, metrics, peak_metrics, write_json


class PeakModel(ObservedSpectralModel):
    def __init__(self, peaks, gamma, basis, condition_cap, fs, nperseg):
        super().__init__(peaks, basis, condition_cap, fs, nperseg)
        gamma = torch.as_tensor(gamma, dtype=torch.float64)
        if gamma.shape != self.peaks.shape or not torch.all(gamma > 0):
            raise ValueError('One positive damping rate per prescribed peak required')
        self.register_buffer('initial_gamma', gamma/1000)
        mask = torch.zeros_like(self.force); mask[:self.m,:self.m] = 1
        self.register_buffer('force_mask', mask)
        with torch.no_grad():
            self.width.zero_()
        self.broad_frequency.requires_grad_(False)
        self.broad_damping.requires_grad_(False)

    def force_factor(self):
        return self.force*self.force_mask

    def poles(self):
        peak = self.peaks+8*torch.tanh(self.peak_shift)
        gamma = self.initial_gamma*(1+.25*torch.tanh(self.width))
        omega = torch.sqrt((2*np.pi*peak/1000)**2+2*gamma**2)
        broad = 2*np.pi*self.broad_frequency.exp()/1000
        return torch.cat((omega,broad)),torch.cat((gamma,broad*(.8+.19*torch.sigmoid(self.broad_damping))))


def fit_weights(f, peaks, dimension):
    # Equal mass per peak neighborhood, plus a full-band leakage control.
    windows = abs(f[:,None]-peaks[None,:]) <= 20
    if np.any(windows.sum(axis=0)==0):
        raise ValueError('Every prescribed peak must have a sampled fitting neighborhood')
    frequency = .3/len(f)+.7*(windows/windows.sum(axis=0)).mean(axis=1)
    coordinate = np.ones(dimension); coordinate[:min(4,dimension)] = 5
    return torch.tensor(frequency),torch.tensor(coordinate/coordinate.sum())


def loss_function(total, reference, frequency_weights, coordinate_weights):
    diag = reference.diagonal(dim1=-2,dim2=-1).real.clamp_min(1e-15)
    pred = total.diagonal(dim1=-2,dim2=-1).real.clamp_min(1e-15)
    log = ((pred.log()-diag.log()).square()*coordinate_weights).sum(-1)
    pairs = coordinate_weights[:,None]*coordinate_weights[None,:]
    pairs = pairs*(1-torch.eye(len(pairs),dtype=pairs.dtype,device=pairs.device))
    pairs = pairs/pairs.sum()
    cross = (((total-reference).abs().square()/(diag[:,:,None]*diag[:,None,:]))*pairs).sum((-1,-2))
    return ((log+2*cross)*frequency_weights).sum()


def fit_one(f, train, bg, scale, peaks, gamma, amplitude, cap, seed, args, indices):
    basis, _ = initial_basis(train-bg,f,peaks,1,seed)
    model = PeakModel(peaks,gamma,basis,cap,args.fs,args.nperseg)
    with torch.no_grad():
        model.force.zero_()
        model.force[:model.m,:model.m].copy_(torch.diag(torch.tensor(amplitude)))
    reference, background = torch.tensor(train),torch.tensor(bg)
    weights = fit_weights(f,peaks,len(scale))
    history=[];started=time.monotonic();best=float('inf');state=None;best_step=0
    # Fixed poles first; then bounded frequency/linewidth refinement.
    for step in range(args.iterations+1):
        if step in (0,args.fixed_steps):
            free = step > 0
            model.width.requires_grad_(free);model.peak_shift.requires_grad_(free)
            optimizer = torch.optim.Adam([
                dict(params=[model.rotation,model.force]+([model.stretch] if cap>1 else []),lr=.012 if not free else .005),
                dict(params=[model.width,model.peak_shift] if free else [],lr=.003)])
        optimizer.zero_grad()
        resonance=model.spectrum(indices)
        loss=loss_function(background+resonance,reference,*weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite spectral objective')
        value=loss.item()
        if value < best:
            best=value;state=copy.deepcopy(model.state_dict());best_step=step
        if step%50==0:
            history.append(dict(step=step,loss=value,seconds=time.monotonic()-started))
        if step==args.iterations or time.monotonic()-started>=args.seconds_per_fit:
            break
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step();model.project()
    model.load_state_dict(state)
    with torch.no_grad():
        resonance=model.spectrum(indices).numpy()
        t,k,d,q=[x.numpy() for x in model.operators()]
        w,g=[x.numpy()*1000 for x in model.poles()]
    a=np.block([[np.zeros_like(k),np.eye(len(k))],[-k,-d]])
    result=dict(cap=cap,seed=seed,train_objective=best,iterations=step,best_step=best_step,
        elapsed_seconds=time.monotonic()-started,history=history,
        condition_standardized=float(np.linalg.cond(t)),condition_physical=float(np.linalg.cond(scale[:,None]*t)),
        singular_values=np.linalg.svd(t,compute_uv=False).tolist(),
        maximum_drift_eigenvalue_real_per_second=float(np.linalg.eigvals(a).real.max()),
        isolated_peak_frequencies_hz=(np.sqrt(w[:len(peaks)]**2-2*g[:len(peaks)]**2)/(2*np.pi)).tolist(),
        gamma_per_second=g[:len(peaks)].tolist())
    arrays=dict(T=t,K=k,D=d,acceleration_covariance=q,omega_per_second=w,gamma_per_second=g,
                resonant_spectrum=resonance,total_spectrum=bg+resonance)
    return model,result,arrays


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--background',type=Path,default=Path('runs/diagnostics/spectral_background_0p10_manual_v1'))
    p.add_argument('--prior',type=Path,default=Path('runs/diagnostics/latent_oscillators_0p10_two_family_v1/two_family/model.npz'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,help='Keep spatial mean and POD1 through this rank; refit T in this space.')
    p.add_argument('--caps',type=float,nargs='+',default=[1,3,10])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--fixed-steps',type=int,default=500)
    p.add_argument('--seconds-per-fit',type=float,default=35)
    args=p.parse_args(argv);torch.set_num_threads(2)
    if not 0<args.fixed_steps<args.iterations or args.seconds_per_fit<=0 or min(args.caps)<1:
        p.error('Require 0 < fixed steps < iterations, positive time, and cap >= 1')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Output must be empty')
    with np.load(args.background/'background.npz') as a:
        support=a['supported'];f=a['frequency_hz'][support];train=a['train'][support]
        val=a['validation'][support];bg=a['background'][support];scale=a['scale'];mean=a['mean']
        variants=a['perturbation_backgrounds'][:,support]
    source_dimension = len(scale)
    if args.rank is not None:
        if not 3 <= args.rank < source_dimension:
            p.error('rank must retain at least POD1-3 and not exceed the background rank')
        dimension = args.rank+1
        train, val, bg = [s[:, :dimension, :dimension] for s in (train, val, bg)]
        variants = variants[:, :, :dimension, :dimension]
        scale, mean = scale[:dimension], mean[:dimension]
    else:
        dimension = source_dimension
    with np.load(args.prior) as a:
        gamma=a['damping_per_second'];peaks=np.sqrt(a['frequencies_hz']**2-2*(gamma/(2*np.pi))**2)
        amplitude=np.linalg.norm(a['mixing'][:,:dimension,0]*a['scale'][None,:dimension]/scale[None,:],axis=1)
    if len(peaks) > dimension:
        p.error('Need at least one observed coordinate per resonant direction')
    bg_protocol=json.loads((args.background/'protocol.json').read_text())
    study_protocol=bg_protocol['source_protocol']['source_protocol']
    args.fs=study_protocol['arguments']['fs'];args.nperseg=study_protocol['arguments']['nperseg']
    full_f=np.fft.rfftfreq(args.nperseg,1/args.fs)
    indices=np.array([np.argmin(abs(full_f-fi)) for fi in f])
    np.testing.assert_allclose(full_f[indices],f)
    indices=torch.tensor(indices)
    args.output.mkdir(parents=True,exist_ok=True)
    weights=fit_weights(f,peaks,len(scale))
    zero_loss=loss_function(torch.tensor(bg),torch.tensor(train),*weights).item()
    write_json(args.output/'protocol.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        background_sha256=hashlib.sha256((args.background/'background.npz').read_bytes()).hexdigest(),
        prior_sha256=hashlib.sha256(args.prior.read_bytes()).hexdigest(),background_protocol=bg_protocol,
        source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ['peak_background_fit.py','observed_spectral.py','observed_spectral_study.py']},
        coordinate_selection=dict(kind='leading_with_spatial_mean', source_dimension=source_dimension,
                                  dimension=dimension, pod_rank=dimension-1),
        nominal_peaks_hz=peaks.tolist(),initial_gamma_per_second=gamma.tolist(),
        objective='Frozen B + resonant PSD against training matrices. 70% frequency weight equally split across +/-20 Hz peak neighborhoods, 30% uniform supported band. First four coordinates weight5, others1. Weighted log PSD MSE +2 weighted normalized complex off-diagonal CSD MSE.',
        parameters='T and correlated resonant acceleration covariance; any completion directions unforced. First stage fixes frequencies/damping. Second stage limits isolated peak shifts to +/-8 Hz, gamma to 75–125% of prior.',
        limits=['Additive spectral identification surrogate, not a combined neural dynamical model.',
                'Only the earlier 20 two-family poles are prescribed, not every measured peak.',
                'Background can overshoot data; no positive-part clipping or subtraction labels.',
                'Bounded local fits. Validation excluded from optimization, weights, checkpoint and seed selection.',
                'Inherited two-record validation, downsampling, spatial-basis and background uncertainties remain.']))
    np.savez_compressed(args.output/'targets.npz',frequency_hz=f,train=train,validation=val,background=bg,
                        frequency_weights=weights[0].numpy(),coordinate_weights=weights[1].numpy())
    rows=[];predictions=[]
    for seed in args.seeds:
        for cap in args.caps:
            name=f'cap{cap:g}_seed{seed}';print('Fitting',name,flush=True)
            model,row,arrays=fit_one(f,train,bg,scale,peaks,gamma,amplitude,cap,seed,args,indices)
            prediction=arrays['total_spectrum']
            row.update(name=name,train=metrics(prediction,train),validation=metrics(prediction,val),
                       peaks=peak_metrics(f,prediction,train,val,peaks))
            folder=args.output/name;folder.mkdir()
            write_json(folder/'summary.json',row)
            np.savez_compressed(folder/'model.npz',**arrays,mean=mean,scale=scale,frequency_hz=f,background=bg,
                                prescribed_peaks_hz=peaks,initial_gamma_per_second=gamma,fs=args.fs,nperseg=args.nperseg)
            torch.save(model.state_dict(),folder/'parameters.pt')
            rows.append(row);predictions.append(prediction);write_json(args.output/'results.json',rows)
            print(name,round(row['elapsed_seconds'],2),'s, condition',round(row['condition_standardized'],2),
                  'loss',round(row['train_objective'],3),'focused validation dB',round(row['validation']['focused_log_psd_rmse_db'],3),flush=True)
    selected=[min([i for i,r in enumerate(rows) if r['cap']==cap],key=lambda i:rows[i]['train_objective']) for cap in args.caps]
    best=min(selected,key=lambda i:rows[i]['train_objective'])
    baseline=dict(train_objective=zero_loss,validation=metrics(bg,val),peaks=peak_metrics(f,bg,train,val,peaks))
    write_json(args.output/'background_only.json',baseline)
    write_json(args.output/'sensitivity.json',dict(model=rows[best]['name'],
        note='Fixed fitted resonances; swap each of eight background perturbations without refitting. Not parameter uncertainty.',
        variants=[dict(variant=i,train_objective=loss_function(torch.tensor(v+predictions[best]-bg),torch.tensor(train),*weights).item(),
                       validation=metrics(v+predictions[best]-bg,val)) for i,v in enumerate(variants)]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for zoom in [False,True]:
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
        for j,ax in enumerate(axes.flat):
            for spectrum,color,label in [(val,'.65','validation'),(train,'k','training'),(bg,'#999933','frozen background')]:
                ax.plot(f,10*np.log10(spectrum[:,j,j].real.clip(1e-30)),color=color,lw=1,label=label)
            for i in selected:
                ax.plot(f,10*np.log10(predictions[i][:,j,j].real.clip(1e-30)),lw=1.3,label=f"background + peaks, cap {rows[i]['cap']:g}")
            ax.set_title('Spatial mean' if j==0 else f'POD {j}');ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('PSD [dB, standardized units/Hz]');ax.set_xlim(90,650 if zoom else f[-1])
        axes[0,0].legend(fontsize=7);fig.tight_layout()
        fig.savefig(args.output/('peak_zoom.png' if zoom else 'psd_comparison.png'),dpi=160);plt.close(fig)
    lines=['# Resonances above a frozen empirical background','',
        '[PSD overlay](psd_comparison.png) · [Low-frequency detail](peak_zoom.png)','',
        'No neural training. The background is fixed; only the resonant model is fitted. Seed selection uses training objective only.', '',
        '| Model | Actual condition | Train objective | Validation mean/POD1–3 RMSE [dB] | Complex cross RMSE | Peak hits/targets |',
        '|---|---:|---:|---:|---:|---:|']
    def hits(r):return f"{sum(p['hits'] for p in r['peaks'])}/{sum(p['targets'] for p in r['peaks'])}"
    lines.append(f"| Background only | — | {zero_loss:.3f} | {baseline['validation']['focused_log_psd_rmse_db']:.3f} | {baseline['validation']['normalized_cross_rmse']:.3f} | {hits(baseline)} |")
    for i in selected:
        r=rows[i];v=r['validation']
        lines.append(f"| cap {r['cap']:g} | {r['condition_standardized']:.2f} | {r['train_objective']:.3f} | {v['focused_log_psd_rmse_db']:.3f} | {v['normalized_cross_rmse']:.3f} | {hits(r)} |")
    lines += ['',f"Training-selected candidate: `{rows[best]['name']}`. Total fitting time: {sum(r['elapsed_seconds'] for r in rows):.1f} seconds.", '',
        'Peak counts are coordinate-specific, training-selected and validation-confirmed, not independent physical mode counts. Background-only peaks can count too; inspect peak heights and plots.',
        'This is a spectral identification surrogate. The frozen matrix background is not an added OU dynamical process. Adding a neural residual to these operators does not imply additive or preserved PSDs.',
        'Only the earlier 20 resonances are modeled. Frequencies/damping are fixed initially, then tightly bounded. Any remaining directions are unforced; when dimension equals pole count there is no completion direction. See `protocol.json` for exact weights, data lineage and limitations.',
        'Saved operators use seconds and standardized prepared positions/true velocities; they are not directly compatible with separately scaled native-frame neural increments.',
        'Sensitivity swaps the background with saved anchor perturbations at fixed fitted resonance parameters; it does not refit or quantify parameter confidence.', '',
        'Reproduce:', '', '```bash',
        'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \\',
        '/home/jonas/miniforge3/envs/diff-model-mit/bin/python -m modelling.latent_oscillators.peak_background_fit \\',
        f'  --output runs/diagnostics/peak_background_fit_0p10_repeat --caps '+ ' '.join(map(str,args.caps)) +
        ' --seeds '+ ' '.join(map(str,args.seeds)) +
        (f' --rank {args.rank}' if args.rank is not None else '') +
        f' --iterations {args.iterations} --fixed-steps {args.fixed_steps} --seconds-per-fit {args.seconds_per_fit}', '```']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
