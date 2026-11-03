"""Training-only spectral identification and held-out unconditional rollout audit.

Uses the existing harmonic-dynamics audit cache, never validation-selected peaks.
This is an exploratory band-limited model, not a neural-SDE checkpoint converter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import find_peaks

from .model import ResonanceModel, simulate, welch_matrix


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def select_families(records, fs):
    """Peak positions chosen from training-only mean and first POD spectra."""
    f, matrices = welch_matrix(records, fs, 1024)
    mean = matrices.mean(axis=0)
    parameters, counts, selections = [], [], []
    for coordinate in (0, 1):
        power = mean[:,coordinate,coordinate].real
        indices, props = find_peaks(10*np.log10(np.maximum(power, 1e-30)), prominence=1., distance=4)
        indices = indices[(f[indices]>=100)&(f[indices]<=1100)]
        peaks = f[indices]
        if len(peaks) < 4:
            raise ValueError(f'Only {len(peaks)} training peaks for coordinate {coordinate}')
        n = np.arange(len(peaks))
        initial = np.polyfit(n, peaks**(2/3), 1)
        result = least_squares(lambda ab:(ab[0]*n+ab[1])**1.5-peaks, initial, bounds=(1e-8,np.inf))
        parameters.append(result.x); counts.append(len(peaks))
        selections.append(dict(coordinate=coordinate, peaks_hz=peaks.tolist(),
            initial_ab=result.x.tolist(), initial_rmse_hz=float(np.sqrt(np.mean(result.fun**2)))))
    return parameters, counts, selections


def objective(predicted, target, mask, cross_weight=2.):
    p = predicted[mask]
    s = target[mask]
    diagonal = s.diagonal(dim1=-2,dim2=-1).real.clamp_min(1e-15)
    pd = p.diagonal(dim1=-2,dim2=-1).clamp_min(1e-15)
    log_error = (pd.log()-diagonal.log()).square().mean()
    norm = (diagonal[:,:,None]*diagonal[:,None,:]).sqrt()
    off = ~torch.eye(p.shape[-1], dtype=torch.bool)
    # Imaginary target part is included and reported: real mixing cannot fit it.
    cross_error = (((p-s)/norm).abs().square()[:,off]).mean()
    return log_error+cross_weight*cross_error, log_error, cross_error


def initialize(model, target, frequency, seed):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        osc, _ = model.correlations()
        from .model import expected_welch
        templates = expected_welch(osc, model.fs, model.taper).numpy()
        for j, f in enumerate(model.frequencies().numpy()):
            k = np.argmin(abs(frequency-f))
            spectral = target[max(0,k-1):k+2].real.mean(axis=0)
            values, vectors = np.linalg.eigh(spectral)
            order = np.argsort(values)[::-1][:model.rank]
            # Assign partial local spectral power to each resonance; fit background jointly.
            c = vectors[:,order]*np.sqrt(np.maximum(values[order],1e-12)/max(templates[j,k],1e-12)*.6)
            model.mixing[j].copy_(torch.tensor(c))
        model.background.add_(torch.tensor(rng.normal(0,.002,model.background.shape)))


def fit(records, fs, nperseg, families, counts, rank, iterations, seed, output):
    mean = records.mean(axis=(0,1))
    scale = records.std(axis=(0,1))
    if np.any(scale <= 0):
        raise ValueError('All coordinates need positive training variance')
    f, spectra = welch_matrix((records-mean)/scale, fs, nperseg)
    target = torch.tensor(spectra.mean(axis=0))
    model = ResonanceModel(families, counts, len(mean), rank=rank, fs=fs, nperseg=nperseg)
    initialize(model, target.numpy(), f, seed)
    # Band and weights are fixed before validation; lower than 80Hz is explicitly not fit.
    mask = torch.tensor((f>=80)&(f<=1500))
    optimizer = torch.optim.Adam([
        dict(params=[model.mixing,model.background],lr=.012),
        dict(params=[model.log_family],lr=.0015),
        dict(params=[model.log_width],lr=.012)])
    history = []
    best, best_state = np.inf, None
    for step in range(iterations+1):
        optimizer.zero_grad()
        total, diagonal, cross = objective(model.spectral_matrix(), target, mask)
        if not torch.isfinite(total):
            raise RuntimeError(f'Nonfinite objective at iteration {step}')
        if total.item() < best:
            best, best_state = total.item(), copy.deepcopy(model.state_dict())
        if step%100==0 or step==iterations:
            entry = dict(step=step, total=total.item(), log_psd_mse=diagonal.item(), cross_mse=cross.item())
            history.append(entry)
            print(output.name, entry, flush=True)
        if step<iterations:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
    model.load_state_dict(best_state)
    output.mkdir(parents=True,exist_ok=True)
    model.save(output/'model.npz',mean,scale)
    write_json(output/'training_history.json',history)
    return model,mean,scale,f,spectra


def diagonal(s):
    return np.diagonal(s,axis1=-2,axis2=-1).real


def spectral_metrics(f, prediction, reference, coordinates=None):
    coords = np.arange(reference.shape[-1]) if coordinates is None else np.asarray(coordinates)
    mask = (f>=100)&(f<=1100)
    p = diagonal(prediction)[:,coords]
    r = diagonal(reference)[:,coords]
    lp = 10*np.log10(np.maximum(p,1e-30)); lr = 10*np.log10(np.maximum(r,1e-30))
    # 40Hz Gaussian background scale, applied identically to log PSDs.
    sigma = 40/(f[1]-f[0])
    hp = lp-gaussian_filter1d(lp,sigma,axis=0)
    hr = lr-gaussian_filter1d(lr,sigma,axis=0)
    df = f[1]-f[0]
    ratios = (p[mask].sum(axis=0)*df)/(r[mask].sum(axis=0)*df)
    denominator = np.sqrt(r[:,:,None]*r[:,None,:])
    predicted_sub=prediction[:,coords][:,:,coords]
    reference_sub=reference[:,coords][:,:,coords]
    normalized = (predicted_sub-reference_sub)/np.maximum(denominator,1e-30)
    off = ~np.eye(len(coords),dtype=bool)
    return dict(log_psd_rmse_db=float(np.sqrt(np.mean((lp[mask]-lr[mask])**2))),
        ripple_rmse_db=float(np.sqrt(np.mean((hp[mask]-hr[mask])**2))),
        band_power_ratios=ratios.tolist(), coordinates=coords.tolist(),
        normalized_cross_rmse=float(np.sqrt(np.mean(abs(normalized[mask][:,off])**2))))


def peaks_report(f, prediction, reference, training):
    result = []
    for c in range(min(6,training.shape[-1])):
        spectra = [10*np.log10(np.maximum(diagonal(s)[:,c],1e-30)) for s in (training,reference,prediction)]
        detected=[]
        for s in spectra:
            k,_=find_peaks(s,prominence=.75,distance=max(1,round(40/(f[1]-f[0]))))
            detected.append(k[(f[k]>=100)&(f[k]<=1100)])
        targets = [k for k in detected[0] if np.any(abs(f[detected[1]]-f[k])<=22)]
        # One predicted peak cannot receive credit for two nearby reference peaks.
        hits = 0
        if len(targets) and len(detected[2]):
            distance=abs(f[np.asarray(targets)][:,None]-f[detected[2]][None,:])
            rows,cols=linear_sum_assignment(np.where(distance<=22,distance,1e6))
            hits=int(np.sum(distance[rows,cols]<=22))
        result.append(dict(coordinate=c, reference_confirmed_training_peaks_hz=f[targets].tolist(),
            predicted_peaks_hz=f[detected[2]].tolist(), hits=hits, targets=len(targets),
            recall=float(hits/len(targets)) if len(targets) else None,
            unmatched_predicted_peaks=len(detected[2])-hits))
    return result


def acf(records, maxlag):
    x=records-records.mean(axis=1,keepdims=True)
    n=x.shape[1]
    z=np.fft.rfft(x,n=2*n,axis=1)
    c=np.fft.irfft(z*z.conj(),n=2*n,axis=1)[:,:maxlag+1]
    # Same finite-record estimator for every population, equal record weights.
    return (c/c[:,:1]).mean(axis=0)


def evaluate(output, model, mean, scale, train, validation, baseline, fs, nperseg, paths, seed):
    with np.load(output/'model.npz') as z:
        saved=dict(z)
    generated=simulate(saved,validation.shape[1],paths,seed)
    f,sp=welch_matrix(generated,fs,nperseg)
    _,tr=welch_matrix(train,fs,nperseg)
    _,va=welch_matrix(validation,fs,nperseg)
    _,ba=welch_matrix(baseline,fs,nperseg)
    train_s,valid_s,gen_s,base_s=tr.mean(0),va.mean(0),sp.mean(0),ba.mean(0)
    expected=model.spectral_matrix().detach().numpy()*scale[None,:,None]*scale[None,None,:]
    report=dict(validation_records=len(validation),generated_paths=paths,steps=validation.shape[1],
        initialization='Independent stationary draws; not a conditional forecast',
        fit_band_hz=[80,1500],evaluation_band_hz=[100,1100],
        expected=spectral_metrics(f,expected,valid_s),generated=spectral_metrics(f,gen_s,valid_s),
        native_tcn=spectral_metrics(f,base_s,valid_s),
        expected_first6=spectral_metrics(f,expected,valid_s,range(6)),
        generated_first6=spectral_metrics(f,gen_s,valid_s,range(6)),
        native_tcn_first6=spectral_metrics(f,base_s,valid_s,range(6)),
        expected_peaks=peaks_report(f,expected,valid_s,train_s),
        generated_peaks=peaks_report(f,gen_s,valid_s,train_s),
        native_tcn_peaks=peaks_report(f,base_s,valid_s,train_s),
        frequencies_hz=saved['frequencies_hz'].tolist(),damping_per_second=saved['damping_per_second'].tolist(),
        family_ab=saved['family_ab'].tolist())
    times=np.arange(round(.03*fs)+1)/fs
    acfs={name:acf(x,len(times)-1) for name,x in [('training',train),('validation',validation),('generated',generated),('native_tcn',baseline)]}
    report['acf_14p3ms']={name:a[round(.0143*fs),:6].tolist() for name,a in acfs.items()}
    report['acf_rmse_first6_1_25ms']={name:float(np.sqrt(np.mean((a[(times>=.001)&(times<=.025),:6]-acfs['validation'][(times>=.001)&(times<=.025),:6])**2))) for name,a in acfs.items() if name!='validation'}
    write_json(output/'summary.json',report)
    np.savez_compressed(output/'spectra.npz',frequency_hz=f,training=tr,validation=va,generated=sp,
        native_tcn=ba,expected=expected,times=times,**{'acf_'+k:v for k,v in acfs.items()})
    np.savez_compressed(output/'stationary_rollout.npz',generated=generated,fs=fs,seed=seed,
        note='Physical scaled POD coordinates; not common neural-SDE rollout schema')
    fig,axs=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    view=(f>=90)&(f<=1150)
    for c,ax in enumerate(axs.flat):
        for label,s,color,style in [('Training',train_s,'.65','-'),('Validation',valid_s,'black','-'),
                ('Oscillator expectation',expected,'C3','--'),('Oscillator rollout',gen_s,'C1','-'),('Existing native TCN',base_s,'C0','-')]:
            ax.semilogy(f[view],diagonal(s)[view,c],color=color,ls=style,label=label,lw=1.1)
        ax.set(title='Spatial mean' if c==0 else f'POD {c}',xlabel='Frequency [Hz]',ylabel='PSD [µm²/Hz]')
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7)
    fig.suptitle(output.name+': training-only fit, held-out validation')
    fig.savefig(output/'modal_psd.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for c,ax in enumerate(axs.flat):
        for label,color in [('validation','black'),('generated','C1'),('native_tcn','C0')]:
            ax.plot(times*1000,acfs[label][:,c],label=label,color=color)
        ax.set(title='Spatial mean' if c==0 else f'POD {c}',xlabel='Lag [ms]',ylabel='ACF',ylim=(-.6,1.))
        ax.grid(alpha=.2)
    axs[0,0].legend();fig.savefig(output/'acf.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    weights=np.sum(saved['mixing']**2,axis=-1).T*scale[:,None]**2
    relative=weights/np.maximum(weights.sum(axis=1,keepdims=True),1e-30)
    if relative.shape[1]==0:
        ax.text(.5,.5,'Smooth control: no resonant latent states',ha='center',transform=ax.transAxes)
        fig.savefig(output/'mixing.png',dpi=160);plt.close(fig)
        return report
    im=ax.imshow(relative,aspect='auto',origin='lower',cmap='magma',vmin=0)
    ax.set(xticks=np.arange(len(weights.T)),xticklabels=[f'{v:.0f}' for v in saved['frequencies_hz']],
        yticks=np.arange(len(mean)),yticklabels=['Mean']+[str(c) for c in range(1,len(mean))],
        xlabel='Latent natural frequency [Hz], grouped by family',ylabel='Observed POD coordinate',
        title='Fraction of modeled resonant variance per coordinate (background excluded)')
    ax.tick_params(axis='x',rotation=90);fig.colorbar(im,ax=ax)
    fig.savefig(output/'mixing.png',dpi=160);plt.close(fig)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=1800)
    p.add_argument('--paths',type=int,default=16)
    p.add_argument('--nperseg',type=int,default=2048)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--variants',nargs='+',choices=['smooth','one_family','two_family','two_family_rank2'],
                   default=['smooth','one_family','two_family','two_family_rank2'])
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose an empty output directory to preserve previous studies')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    cache=args.audit_dir/'signals.npz'
    provenance=json.loads((args.audit_dir/'provenance.json').read_text())
    if provenance['training_repetitions']!=list(range(1,13)) or provenance['validation_repetitions']!=[13,14] or provenance['test_repetitions_used']:
        raise ValueError('Unexpected audit split; refusing implicit population changes')
    with np.load(cache) as z:
        train=z['reference'][:12].copy();validation=z['reference'][12:].copy()
        baseline=z['generated'].reshape(-1,z['generated'].shape[-2],z['generated'].shape[-1]).copy()
    if len(validation)!=2 or train.shape[-1]!=21:
        raise ValueError('Expected 12 training / 2 validation records and 21 coordinates')
    fs=float(provenance['single_rollout']['sampling_frequency_hz'])/int(provenance['downsample_factor'])
    families,counts,selections=select_families(train,fs)
    write_json(args.output/'config.json',dict(audit_dir=str(args.audit_dir.resolve()),
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),source_provenance=provenance,
        fs=fs,nperseg=args.nperseg,iterations=args.iterations,paths=args.paths,seed=20261023,
        family_selection=selections,variants=args.variants,fit_band_hz=[80,1500],
        objective='mean squared log autospectral error + 2 * normalized offdiagonal complex spectral MSE',
        caveats=['Real readout has zero imaginary cross spectrum; its error remains in the objective.',
          'Band-limited diagnostic: no claim about total energy, below-80Hz behavior, or conditional forecasting.',
          'Native TCN baseline has shorter records and overlapping starts; reference has two independent validation repetitions.',
          'No validation-based hyperparameter or checkpoint selection; lowest training objective checkpoint retained.']))
    results={}
    for variant in args.variants:
        nfamilies=0 if variant=='smooth' else (1 if variant=='one_family' else 2)
        rank=2 if variant=='two_family_rank2' else 1
        out=args.output/variant
        model,mean,scale,_,_=fit(train,fs,args.nperseg,families[:nfamilies],counts[:nfamilies],rank,
            args.iterations,20261023,out)
        results[variant]=evaluate(out,model,mean,scale,train,validation,baseline,fs,args.nperseg,args.paths,20261023)
    write_json(args.output/'summary.json',results)
    lines=['# Shared latent oscillator baseline', '',
        'Training: repetitions 1–12. Validation: 13–14. Test excluded. All models fit jointly to 21 coordinates.',
        'The fit uses 80–1500 Hz; metrics below use 100–1100 Hz. Unconditional stationary generation, not initial-condition forecasts.',
        'Peak/ripple scores measure spectral structure; they do not identify capillary physics or prove deterministic phase predictability.', '',
        '| Model | First-six PSD RMSE [dB] | Ripple RMSE [dB] | ACF RMSE 1–25 ms | Mean 14.3 ms ACF |',
        '|---|---:|---:|---:|---:|']
    for name,r in results.items():
        m=r['generated_first6']
        lines.append(f"| {name} | {m['log_psd_rmse_db']:.3f} | {m['ripple_rmse_db']:.3f} | {r['acf_rmse_first6_1_25ms']['generated']:.3f} | {r['acf_14p3ms']['generated'][0]:.3f} |")
    r=next(iter(results.values()));m=r['native_tcn_first6']
    lines.append(f"| Existing native TCN | {m['log_psd_rmse_db']:.3f} | {m['ripple_rmse_db']:.3f} | {r['acf_rmse_first6_1_25ms']['native_tcn']:.3f} | {r['acf_14p3ms']['native_tcn'][0]:.3f} |")
    lines += ['',f"Validation spatial-mean ACF at 14.3 ms: {r['acf_14p3ms']['validation'][0]:.3f}.",
        '', 'Each variant contains model.npz, stationary_rollout.npz, spectra.npz, summary.json, training_history.json, modal_psd.png, acf.png and mixing.png.',
        'The smooth control has no mixing.png resonance columns. Inspect exact model expectation as well as finite-rollout estimates.',
        'Imaginary cross-spectral mismatch and low-frequency mismatch are explicit limitations of this real-readout band-limited baseline.']
    (args.output/'quick_summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
