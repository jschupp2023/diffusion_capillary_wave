"""Read-only imported-oscillator/TCN audit with a matched baseline rollout.

Component projections are descriptive. Averaged local frequency responses are
linearizations about sampled histories, not identified global stochastic poles.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

from modelling.neural_sde.train import load_model
from modelling.acdm.conditional_edm.data import checkpoint_data
from data_analysis.rollout import Rollout
from data_analysis.energy.capillary_energy import capillary_stiffness, length_scale
from data_analysis.psd.rollout_spectral_peaks import peak_metrics


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def histories(values,indices,length):
    return values[indices],values[indices[:,None]-np.arange(1,length+1)]


def physical_core(model):
    dt=model.config.dt
    if model.config.drift_type=='imported_oscillator_residual':
        return [a.detach().cpu().numpy() for a in (model.drift_model.physical_stiffness,model.drift_model.physical_damping)]
    su=model.target_std.detach().cpu().numpy();sx=model.state_std.detach().cpu().numpy()
    k=model.drift_model.stiffness_matrix().detach().cpu().numpy()
    d=model.drift_model.damping_matrix().detach().cpu().numpy()
    return su[:,None]*k/sx[None,:]/dt**2,su[:,None]*d/su[None,:]/dt


@torch.no_grad()
def components(model,values,indices,batch_size=256):
    result={name:[] for name in ('restoring','damping','residual','total')}
    device=model.state_mean.device;d=model.config.reduced_dim
    for start in range(0,len(indices),batch_size):
        x,h=histories(values[:,:d],indices[start:start+batch_size],model.config.history_steps)
        x,h=[torch.as_tensor(v,dtype=torch.float32,device=device) for v in (x,h)]
        r=model.nonlinear_drift(x,h)*model.target_std
        if model.config.drift_type=='imported_oscillator_residual':
            m=model.drift_model
            restoring=-model.config.dt**2*(x-m.equilibrium.float())@m.physical_stiffness.float().T
            damping=-model.config.dt*(x-h[:,0])@m.physical_damping.float().T
        else:
            inputs=model._drift_input(x,h)
            restoring=-(inputs[:,:d]@model.drift_model.stiffness_matrix().T)*model.target_std
            damping=-(inputs[:,d:2*d]@model.drift_model.damping_matrix().T)*model.target_std
        for name,v in zip(result,(restoring,damping,r,restoring+damping+r)):
            result[name].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def component_summary(terms,scale):
    terms={k:v[:,:len(scale)]/scale for k,v in terms.items()}
    f=terms['restoring']+terms['damping'];r=terms['residual'];eps=1e-30
    def stats(a,b):
        return dict(residual_parallel_coefficient=float(np.sum(a*b)/(np.sum(a*a)+eps)),
                    cosine=float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)+eps)))
    return dict(rms={k:float(np.sqrt(np.mean(v*v))) for k,v in dict(**terms,fixed=f).items()},
        rms_per_coordinate={k:np.sqrt(np.mean(v*v,axis=0)).tolist() for k,v in dict(**terms,fixed=f).items()},
        residual_vs_fixed=stats(f,r),residual_vs_restoring=stats(terms['restoring'],r),
        residual_vs_damping=stats(terms['damping'],r))


def residual_jacobian(model,values,indices):
    """J[lag, output, input], exact autograd averaged over histories."""
    d=model.config.reduced_dim
    x,h=histories(values[:,:d],indices,model.config.history_steps)
    x,h=[torch.as_tensor(v,dtype=model.state_mean.dtype,device=model.state_mean.device).requires_grad_() for v in (x,h)]
    r=model.nonlinear_drift(x,h)*model.target_std
    rows=[]
    for j in range(d):
        gx,gh=torch.autograd.grad(r[:,j].sum(),(x,h),retain_graph=j<d-1)
        rows.append(torch.cat((gx[:,None],gh),dim=1).mean(0).detach().cpu().numpy())
    return np.stack(rows,axis=1).astype(float)


def residual_transfer(jacobian,frequency,dt):
    return np.einsum('fl,loi->foi',np.exp(-2j*np.pi*np.asarray(frequency)[:,None]*dt*np.arange(len(jacobian))),jacobian)


def response_psd(k,damping,jacobian,noise,frequency,dt,alpha=1.):
    phase=np.exp(-2j*np.pi*np.asarray(frequency)*dt)
    fixed=-dt**2*k[None]-dt*damping[None]*(1-phase[:,None,None])
    residual=residual_transfer(jacobian,frequency,dt)
    identity=np.eye(len(k))
    operator=(1/phase-2+phase)[:,None,None]*identity-fixed-alpha*residual
    transfer=np.linalg.solve(operator,np.broadcast_to(noise,operator.shape))
    return 2*dt*np.sum(abs(transfer)**2,axis=-1)


def modal_response(jacobian,transform,omega,gamma,frequency,dt):
    r=residual_transfer(jacobian,frequency,dt)
    modal=np.linalg.inv(transform)[None]@r@transform[None]
    diagonal=np.array([modal[i,i,i] for i in range(len(frequency))])
    angle=2*np.pi*np.asarray(frequency)*dt
    extra_d=-diagonal.imag/np.sin(angle)
    extra_k=-diagonal.real-extra_d*(1-np.cos(angle))
    return dict(frequencies_hz=np.asarray(frequency).tolist(),
        effective_damping_over_fixed=(1+extra_d/(2*gamma*dt)).tolist(),
        effective_stiffness_over_fixed=(1+extra_k/(omega**2*dt**2)).tolist(),
        note='Frequency-specific diagonal modal response of averaged local Jacobian. Cross-mode coupling and other frequencies remain; these are not fitted global poles.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hybrid',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--hybrid-rollout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--jacobian-samples',type=int,default=32)
    args=p.parse_args(argv);torch.set_num_threads(2)
    if args.output.exists():raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    started=time.monotonic()
    models={};checkpoints={}
    for label,path in [('hybrid',args.hybrid),('baseline',args.baseline)]:
        models[label],checkpoints[label]=load_model(path,args.device)
        for parameter in models[label].parameters():parameter.requires_grad_(False)
    hm,bm=models['hybrid'],models['baseline'];hc,bc=checkpoints['hybrid'],checkpoints['baseline']
    assert hm.config.reduced_dim==20 and bm.config.reduced_dim==21
    assert hm.config.history_steps==bm.config.history_steps==63
    for key in ('shared_basis','experiment','spatial_mean_highpass_hz','validation_reps','test_reps'):
        assert hc['data_config'].get(key)==bc['data_config'].get(key),(key,hc['data_config'],bc['data_config'])
    assert hc['training']['train_trim_percent']==bc['training']['train_trim_percent']==0
    data=checkpoint_data(bc)
    roll=Rollout(args.hybrid_rollout)
    assert Path(roll.metadata['checkpoint']).resolve()==args.hybrid.resolve()
    scale=hm.target_std.detach().cpu().numpy();sx=hm.state_std.detach().cpu().numpy();dt=hm.config.dt
    reference_blocks=[];blocks=[];block_indices=[];populations=[]
    for rep in data.validation_repetitions:
        for start in (1000,45000):
            _,values=data.read_coordinates(rep,start-63,start+8192+1)
            values=values.astype(np.float32)
            blocks.append(values);block_indices.append(np.arange(63,63+8192));reference_blocks.append(values[63:63+8192])
            populations.append(dict(repetition=rep,start=start,samples=8192))
    generated=roll.generated[0,0].astype(np.float32)
    report=dict(hybrid=str(args.hybrid.resolve()),baseline=str(args.baseline.resolve()),hybrid_rollout=str(args.hybrid_rollout.resolve()),
        checkpoint_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('hybrid',args.hybrid),('baseline',args.baseline)]},
        common_coordinates='spatial mean + POD1-19; baseline retains its POD20 internally',
        conditional_populations=populations,normalization='component RMS divided by hybrid native-velocity std; shared first 20 coordinates',
        caveats=['Ranks and training objectives differ: descriptive model comparison, not a matched retraining ablation.',
                 'Baseline has learned diagonal restoring/damping, not the imported resonance operators.',
                 'Local averaged linearization is not a global nonlinear PSD prediction.'],models={})
    artifact=hc['training']['imported_oscillator']['source_artifact']
    with np.load(artifact) as a:
        transform=np.sqrt(data.n_space)*a['scale'][:,None]*a['T']
        omega=a['omega_per_second'];gamma=a['gamma_per_second']
    poles=np.sqrt(omega**2-2*gamma**2)/(2*np.pi)
    # Fixed common sampled states, independent of either model output.
    rng=np.random.default_rng(23)
    chosen=np.linspace(64,len(blocks[0])-2,args.jacobian_samples//len(blocks),dtype=int)
    jacobians={};component_arrays={}
    frequency=np.arange(50.,1500.1,2.)
    spectra={}
    for label,model in models.items():
        print('Conditional components',label,flush=True)
        terms=[components(model,v,idx) for v,idx in zip(blocks,block_indices)]
        merged={key:np.concatenate([a[key] for a in terms]) for key in terms[0]}
        component_arrays[label]=merged
        jac=np.mean([residual_jacobian(model,v,chosen) for v in blocks],axis=0)
        jacobians[label]=jac
        k,damping=physical_core(model)
        noise=model.diffusion_matrix().detach().cpu().numpy()
        spectra[label+'_local']=response_psd(k,damping,jac,noise,frequency,dt)
        spectra[label+'_core']=response_psd(k,damping,jac,noise,frequency,dt,alpha=0.)
        m=dict(reference_components=component_summary(merged,scale),
            normalized_diffusion=model.normalized_diffusion_matrix().diag().detach().cpu().tolist(),
            diffusion_per_common_velocity_std=(np.diag(noise)[:20]/scale).tolist(),
            training={key:checkpoints[label]['training'].get(key) for key in ('train_mode','epochs','learning_rate','multistep_weight','multistep_objective','multistep_horizons','stationary_mean_weight','stationary_variance_weight','initial_checkpoint')})
        if label=='hybrid':
            m['local_modal_response_reference']=modal_response(jac,transform,omega,gamma,poles,dt)
            gidx=np.sort(rng.choice(np.arange(63,len(generated)-1),4096,replace=False))
            gt=components(model,generated,gidx)
            m['generated_components']=component_summary(gt,scale)
            gj=residual_jacobian(model,generated,gidx[np.linspace(0,len(gidx)-1,args.jacobian_samples,dtype=int)])
            jacobians['hybrid_generated']=gj
            m['local_modal_response_generated']=modal_response(gj,transform,omega,gamma,poles,dt)
            spectra['hybrid_generated_local']=response_psd(k,damping,gj,noise,frequency,dt)
            for alpha in (.25,.5):
                spectra[f'hybrid_local_alpha{alpha}']=response_psd(k,damping,jac,noise,frequency,dt,alpha)
        report['models'][label]=m
        save(args.output/'report.json',report)
    # One paired deterministic sinusoidal perturbation independently checks autograd transfer.
    model=copy.deepcopy(hm).double();v=blocks[0];idx=chosen;xx,hh=histories(v[:,:20],idx,63)
    direction=transform[:,0];direction=direction/np.linalg.norm(direction)*np.linalg.norm(sx)
    theta=2*np.pi*poles[0]*dt;epsilon=1e-4
    perturb=np.cos(theta*np.arange(64))[:,None]*direction[None]
    with torch.no_grad():
        results=[]
        for sign in (1,-1):
            x=torch.tensor(xx+sign*epsilon*perturb[0],dtype=torch.float64,device=args.device)
            h=torch.tensor(hh+sign*epsilon*perturb[None,1:],dtype=torch.float64,device=args.device)
            results.append((model.nonlinear_drift(x,h)*model.target_std).mean(0).cpu().numpy())
    jfirst=residual_jacobian(model,v,chosen)
    predicted=np.einsum('loi,li->o',jfirst,perturb)
    actual=(results[0]-results[1])/(2*epsilon)
    report['jacobian_sinusoid_check_relative_error']=float(np.linalg.norm(actual-predicted)/np.linalg.norm(predicted))
    report['jacobian_check_precision']='float64 finite differences; float32 phase differences suffer subtractive cancellation'
    report['float32_jacobian_relative_error']=float(np.linalg.norm(residual_jacobian(hm,v,chosen)-jfirst)/np.linalg.norm(jfirst))
    np.savez_compressed(args.output/'conditional_arrays.npz',frequency_hz=frequency,**spectra,**{key+'_jacobian':v for key,v in jacobians.items()},
        poles_hz=poles,reference_positions=np.concatenate(reference_blocks),
        **{label+'_'+key:v for label,terms in component_arrays.items() for key,v in terms.items()})
    save(args.output/'report.json',report)
    print('Conditional analysis complete; generating matched baseline rollout',flush=True)
    # Match the existing hybrid reference repetition, exact start, and horizon.
    rep=str(roll.repetitions[0]);start=int(roll.start_index[0]);steps=roll.generated.shape[2]-1
    times,full=data.read_coordinates(rep,start-63,start+steps+1)
    np.testing.assert_allclose(full[63:,:20],roll.reference[0],rtol=1e-5,atol=1e-4)
    bm.to('cpu');bm.tcn_rollout_cache=True;torch.set_num_threads(1)
    x=torch.tensor(full[63:64],dtype=torch.float32)
    h=torch.tensor(full[:63][::-1].copy()[None],dtype=torch.float32)
    before=time.monotonic()
    with torch.no_grad():
        baseline=bm.rollout(x,horizon=steps,num_trajectories=1,history_states=h,seed=0)[0,0].numpy()
    print('Matched baseline rollout seconds',time.monotonic()-before,flush=True)
    np.savez_compressed(args.output/'matched_baseline.npz',trajectories=baseline,reference=full[63:],reference_time=times[63:])
    reference=roll.reference[0].astype(float)
    kenergy,_=capillary_stiffness(roll.modes,roll.x*length_scale(roll.x_units),roll.y*length_scale(roll.y_units),gamma=.0728)
    def energy(v):return np.einsum('ni,ij,nj->n',v[:,1:],kenergy,v[:,1:]).mean()
    f,pr=welch(reference,fs=1/dt,nperseg=8192,axis=0)
    psds=dict(reference=pr)
    report['matched_rollout']=dict(repetition=rep,start_index=start,horizon=steps,baseline_seed=0,
        note='Same physical reference window and first 20 output coordinates; one realization each, not identical random innovations.',models={})
    for label,values in [('hybrid',generated),('baseline',baseline[:,:20])]:
        _,psd=welch(values,fs=1/dt,nperseg=8192,axis=0);psds[label]=psd
        rr=np.var(reference,axis=0);vv=np.var(values,axis=0)
        row=dict(energy_ratio=float(energy(values)/energy(reference)),pod_covariance_trace_ratio=float(vv[1:].sum()/rr[1:].sum()),
            mean_variance_ratio=float(vv[0]/rr[0]),variance_ratios=(vv/rr).tolist(),
            peaks={str(j):peak_metrics(f,pr[:,j],psd[:,j]) for j in range(4)})
        report['matched_rollout']['models'][label]=row
    report['elapsed_seconds']=time.monotonic()-started
    np.savez_compressed(args.output/'matched_psds.npz',frequency_hz=f,**psds)
    save(args.output/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ('matched','response'):
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for j,ax in enumerate(axes.flat):
            for label,psd in psds.items():
                ax.plot(f,psd[:,j]/sx[j]**2,label=label,lw=1.)
            if kind=='response':
                for label in ('hybrid_core','hybrid_local','hybrid_generated_local'):
                    ax.plot(frequency,spectra[label][:,j]/sx[j]**2,label=label,lw=1.,ls='--')
            ax.set(xlim=(80,1500),yscale='log',xlabel='Frequency [Hz]',ylabel='PSD / hybrid position variance',title='Spatial mean' if j==0 else f'POD{j}')
        axes[0,0].legend(fontsize=7);fig.savefig(args.output/(kind+'_psd.png'),dpi=160);plt.close(fig)
    print(args.output/'report.json',flush=True)


if __name__=='__main__':main()
