"""Fit one fixed stable, minimum-phase modal biquad notch per oscillator pole."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

def local_ratio(frequency, spectrum, poles, inner=50., outer=150.):
    result=[]
    for mode,pole in enumerate(poles):
        neighbors=((np.abs(frequency-pole)>=inner)&(np.abs(frequency-pole)<=outer))
        result.append(np.interp(pole,frequency,spectrum[:,mode])/np.median(spectrum[neighbors,mode]))
    return np.asarray(result)

def response(frequency,pole,dt,zero_radius,pole_radius):
    theta=2*np.pi*frequency*dt; angle=2*np.pi*pole*dt
    numerator=np.abs(1-2*zero_radius*np.cos(angle)*np.exp(-1j*theta)+zero_radius**2*np.exp(-2j*theta))**2
    denominator=np.abs(1-2*pole_radius*np.cos(angle)*np.exp(-1j*theta)+pole_radius**2*np.exp(-2j*theta))**2
    return numerator/denominator

def fit_filter(frequency,spectrum,poles,dt):
    target=local_ratio(frequency,spectrum,poles)
    zero_radii=[]; pole_radii=[]; gains=[]; achieved=[]
    for mode,(pole,desired) in enumerate(zip(poles,target)):
        neighbors=((np.abs(frequency-pole)>=50)&(np.abs(frequency-pole)<=150))
        def values(log10_alpha):
            alpha=10**log10_alpha; rz=1-alpha; rp=max(0.,1-20*alpha)
            power=response(frequency,pole,dt,rz,rp)
            ratio=np.interp(pole,frequency,power)/np.median(power[neighbors])
            return rz,rp,power,ratio
        def error(log10_alpha):
            return (np.log(values(log10_alpha)[-1])-np.log(desired))**2
        grid=np.linspace(-9.,-2.,1000)
        center=int(np.argmin([error(value) for value in grid]))
        lower,upper=grid[max(0,center-2)],grid[min(len(grid)-1,center+2)]
        fit=minimize_scalar(error,bounds=(lower,upper),method='bounded',options={'xatol':1e-13})
        rz,rp,power,_=values(fit.x); gain=1/np.sqrt(np.median(power[neighbors]))
        zero_radii.append(rz); pole_radii.append(rp); gains.append(gain)
        achieved.append(np.interp(pole,frequency,gain**2*power))
    rz=np.asarray(zero_radii); rp=np.asarray(pole_radii); gains=np.asarray(gains)
    angles=2*np.pi*poles*dt
    return (gains,gains*-2*rz*np.cos(angles),gains*rz**2,
            -2*rp*np.cos(angles),rp**2,rz,rp,target,np.asarray(achieved))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--oscillator-artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv); args.output.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.audit/'report.json').read_text())
    with np.load(args.audit/'spectral_budget.npz',allow_pickle=False) as saved:
        poles=np.asarray(saved['poles_hz'],float); train_f=np.asarray(saved['train_frequency_hz'],float)
        train_psd=np.asarray(saved['train_residual_psd'],float)
        validation_f=np.asarray(saved['validation_frequency_hz'],float)
        validation_psd=np.asarray(saved['validation_residual_psd'],float)
    dt=float(report['dt_seconds'])
    h0,h1,h2,d1,d2,rz,rp,target,achieved=fit_filter(train_f,train_psd,poles,dt)
    validation_target=local_ratio(validation_f,validation_psd,poles)
    artifact_hash=hashlib.sha256(args.oscillator_artifact.read_bytes()).hexdigest()
    path=args.output/'modal_biquad.npz'
    np.savez_compressed(path,h0=h0,h1=h1,h2=h2,d1=d1,d2=d2,zero_radii=rz,pole_radii=rp,
        poles_hz=poles,train_target_pole_over_local=target,fitted_pole_over_local=achieved,
        validation_target_pole_over_local=validation_target,dt_seconds=dt,dimension=len(poles),
        oscillator_artifact_sha256=np.asarray(artifact_hash),source_spectral_budget_sha256=np.asarray(
        hashlib.sha256((args.audit/'spectral_budget.npz').read_bytes()).hexdigest()))
    summary=dict(artifact=str(path.resolve()),dimension=len(poles),dt_seconds=dt,
        source_audit=str(args.audit.resolve()),oscillator_artifact=str(args.oscillator_artifact.resolve()),
        oscillator_artifact_sha256=artifact_hash,train_target_median=float(np.median(target)),
        fitted_median=float(np.median(achieved)),validation_target_median=float(np.median(validation_target)),
        maximum_relative_fit_error=float(np.max(np.abs(achieved/target-1))),
        zero_radius_range=[float(rz.min()),float(rz.max())],pole_radius_range=[float(rp.min()),float(rp.max())],
        inverse_memory_scale_range=[float((1/(1-rz)).min()),float((1/(1-rz)).max())],
        definition='stable minimum-phase modal biquad; zero radius is fitted, pole bandwidth is 20 times wider, sideband gain is one, and the training residual pole/sideband ratio is matched')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'README.md').write_text('# Modal forcing notch filter\n\n'
        f"The fitted biquad filter matches the training median pole/sideband ratio ({summary['train_target_median']:.4g} target, {summary['fitted_median']:.4g} fitted). The held-out observed ratio is {summary['validation_target_median']:.4g}.\n\n"
        'The filter is fixed. Neural drift and diffusion model its causally inverse-filtered forcing; rollout applies this filter before the frozen oscillator update.\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
