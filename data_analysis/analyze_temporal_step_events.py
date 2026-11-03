"""0p20 temporal-step audit: local trend evidence and matched PSD comparisons.

No HDF5 writes. Reuses the existing spatial restoration and temporal audit helpers.
The selective variants are sensitivity experiments, not ground-truth reconstructions.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis.pod.pod_2d import inspect_input, _frame_key
from data_analysis.pod.phase_unwrap_gradient_poisson import restore_frame_poisson
from data_analysis.validate_temporal_cycle_alignment import medians, save_csv, offsets


def channels(a):
    y,x=np.array(a.shape)//2
    return np.array([a.mean(),a[y,x],a[:y,:x].mean(),a[:y,x:].mean(),
                     a[y:,:x].mean(),a[y:,x:].mean()])


def extract_chunk(rawpath, cleanpath, start, stop, ids, q, tau):
    n=stop-start
    result={name:np.empty((n,6)) for name in ['raw','spatial','final']}
    result['median_increments']=np.zeros((n,5))
    result['exported_offsets']=np.zeros(n,dtype=np.int64)
    result['spatial_modified_fraction']=np.zeros(n)
    result['export_residual_cycles']=np.zeros(n)
    with h5py.File(rawpath,'r') as r,h5py.File(cleanpath,'r') as c:
        for i in range(max(0,start-1),stop):
            a=r['main'][_frame_key(int(ids[i]))][:]
            res=restore_frame_poisson(a,q=q,tau=tau,method='integer')
            s=res.restored.astype(float)
            if i>=start:
                j=i-start
                b=c['main'][_frame_key(int(ids[i]))][:].astype(float)
                k_float=(s-b)/q
                k=int(np.rint(np.median(k_float)))
                result['raw'][j]=channels(a.astype(float))
                result['spatial'][j]=channels(s)
                result['final'][j]=channels(b)
                result['exported_offsets'][j]=k
                result['spatial_modified_fraction'][j]=res.pct_pixels_modified/100
                result['export_residual_cycles'][j]=np.max(abs(k_float-k))
                if i:
                    result['median_increments'][j]=medians(s-prev)
            prev=s
    return start,stop,result


def local_fit_evidence(traces, delta_cycles, q, widths=(4,8,16)):
    """Fit independent lines on each side, extrapolated to the boundary midpoint.

    Evidence requires >=3/4 quadrants to predict the SAME existing integer shift,
    with <0.35q mismatch, <0.5q fit RMS and <0.25q/sample slope change.
    Only >=3-cycle events are eligible. Bounds are predeclared, not PSD-tuned.
    """
    n=len(traces)
    counts=np.zeros((n,len(widths)),dtype=int)
    jumps=np.full((n,len(widths),4),np.nan)
    eligible=np.flatnonzero(abs(delta_cycles)>=3)
    for col,w in enumerate(widths):
        left_x=np.arange(-w,0)+.5
        right_x=np.arange(w)+.5
        L=np.column_stack([np.ones(w),left_x])
        R=np.column_stack([np.ones(w),right_x])
        lp,rp=np.linalg.pinv(L),np.linalg.pinv(R)
        for t in eligible:
            if t<w or t+w>n:
                continue
            yl,yr=traces[t-w:t],traces[t:t+w]
            bl,br=lp@yl,rp@yr
            jump=br[0]-bl[0]
            jumps[t,col]=jump
            rms=np.maximum(np.sqrt(np.mean((yl-L@bl)**2,axis=0)),
                           np.sqrt(np.mean((yr-R@br)**2,axis=0)))
            good=(abs(jump-q*delta_cycles[t])<=.35*q)&(rms<=.5*q)&(abs(br[1]-bl[1])<=.25*q)
            counts[t,col]=np.count_nonzero(good)
    return counts,jumps


def synthetic_checks(q,fs):
    t=np.arange(8192)/fs
    records=[]
    for f,a in [(140,80),(365,80),(1000,20)]:
        physical=a*np.sin(2*np.pi*f*t)
        for inject in [False,True]:
            step=np.zeros(len(t),dtype=int)
            if inject:
                step[2000:4000]=5
                step[4000:6000]=-3
            observed=physical+q*step
            k=offsets(np.diff(observed),q)
            dk=np.r_[0,np.diff(k)]
            counts,_=local_fit_evidence(np.repeat(observed[:,None],4,axis=1),dk,q)
            innovation_counts=innovation_evidence(np.repeat(observed[:,None],4,axis=1),dk,q)
            for label,accept in [('strict',np.all(counts>=3,axis=1)),('two_scales',np.sum(counts>=3,axis=1)>=2),('innovation_supported',np.sum(innovation_counts>=3,axis=1)>=2)]:
                shifts=np.where(accept,dk,0)
                corrected=observed-q*np.cumsum(shifts)
                err=corrected-physical
                records.append(dict(frequency_hz=f,amplitude_um=a,injected_steps=inject,
                                    variant=label,accepted_events=int(np.count_nonzero(shifts)),
                                    max_error_um=float(np.max(abs(err))),
                                    rms_error_um=float(np.sqrt(np.mean(err**2)))))
    return records


def innovation_evidence(traces,delta_cycles,q):
    """Fallback for overlapping bursts: robust neighboring increments, excluding
    the candidate transition. Supports a large common jump to within ONE cycle,
    not a unique integer choice. No filtered values are written to the signal.
    """
    ds=np.vstack([np.zeros((1,4)),np.diff(traces,axis=0)])
    result=np.zeros((len(traces),3),dtype=int)
    eligible=np.flatnonzero(abs(delta_cycles)>=3)
    for col,w in enumerate([4,8,16]):
        for t in eligible:
            if t<=w or t+w>=len(traces):continue
            neighbours=np.vstack([ds[t-w:t],ds[t+1:t+w+1]])
            prediction=np.median(neighbours,axis=0)
            scale=1.4826*np.median(abs(neighbours-prediction),axis=0)
            innovation=ds[t]-prediction
            support=(abs(innovation-q*delta_cycles[t])<=q)&(abs(innovation)>=3*q)&(abs(innovation)>=4*np.maximum(scale,.1*q))
            result[t,col]=np.count_nonzero(support)
    return result


def analyze(z,out,q):
    times=z['time']; fs=1/np.mean(np.diff(times)); n=len(times)
    dk=np.r_[0,np.diff(z['exported_offsets'])]
    counts,jumps=local_fit_evidence(z['spatial'][:,2:],dk,q)
    innovation_counts=innovation_evidence(z['spatial'][:,2:],dk,q)
    masks={'strict_steps':np.all(counts>=3,axis=1),
           'two_scale_steps':np.sum(counts>=3,axis=1)>=2,
           'innovation_supported':np.sum(innovation_counts>=3,axis=1)>=2,
           'large_only':abs(dk)>=3}
    sig={'raw':z['raw'],'spatial_only':z['spatial'],'full_temporal':z['final']}
    accepted={}
    for name,mask in masks.items():
        step=np.where(mask,dk,0)
        accepted[name]=step
        sig[name]=z['spatial']-q*np.cumsum(step)[:,None]
    records=[]
    for t in np.flatnonzero(dk):
        records.append(dict(timestep=int(t),time_sec=float(times[t]),cycles=int(dk[t]),
                            spatial_mean_increment_um=float(z['spatial'][t,0]-z['spatial'][t-1,0]),
                            final_mean_increment_um=float(z['final'][t,0]-z['final'][t-1,0]),
                            quadrants_agree_w4=int(counts[t,0]),quadrants_agree_w8=int(counts[t,1]),
                            quadrants_agree_w16=int(counts[t,2]),strict=bool(masks['strict_steps'][t]),
                            two_scales=bool(masks['two_scale_steps'][t]),
                            innovation_support_w4=int(innovation_counts[t,0]),
                            innovation_support_w8=int(innovation_counts[t,1]),
                            innovation_support_w16=int(innovation_counts[t,2]),
                            innovation_supported=bool(masks['innovation_supported'][t])))
    save_csv(out/'event_classification.csv',records)
    # All PSD variants receive exactly the same filtering and Welch settings.
    sos=butter(4,50,btype='highpass',fs=fs,output='sos')
    psds={}
    for name,a in sig.items():
        arr=np.column_stack([a[:,0],a[:,1],a[:,1]-a[:,0]])
        filtered=sosfiltfilt(sos,arr,axis=0)
        f,p=welch(filtered,fs=fs,nperseg=8192,noverlap=4096,axis=0)
        psds[name]=p
    bands=[]
    for ch,label in enumerate(['mean','center','center_minus_mean']):
        for lo,hi in [(100,500),(500,2000),(2000,4000),(10000,50000)]:
            select=(f>=lo)&(f<hi)
            powers={name:float(p[select,ch].sum()*(f[1]-f[0])) for name,p in psds.items()}
            for name,power in powers.items():
                den=powers['spatial_only']-powers['full_temporal']
                bands.append(dict(channel=label,low_hz=lo,high_hz=hi,variant=name,
                                  power_um2=power,power_relative_to_spatial_only=power/powers['spatial_only'],
                                  fraction_of_full_power_reduction=(powers['spatial_only']-power)/den if abs(den)>1e-9 else None))
    save_csv(out/'band_power_comparison.csv',bands)
    colours={'raw':'0.6','spatial_only':'C0','strict_steps':'C2','two_scale_steps':'C4','innovation_supported':'C2','large_only':'C3','full_temporal':'C1'}
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    for ch,ax in enumerate(axes):
        for name,p in psds.items():
            if name in ['strict_steps','two_scale_steps']:continue
            ax.loglog(f[1:],p[1:,ch],label=name,color=colours[name],lw=1)
        ax.set(xlim=(50,10000),xlabel='Frequency [Hz]',ylabel='PSD [um²/Hz]',title=['Spatial mean','Center height','Center minus spatial mean'][ch])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out/'selective_vs_full_psd.png',dpi=170);plt.close(fig)
    # Largest accepted and rejected events, plus a one-cycle example selected by
    # strongest consistent mean trend (rather than by attractive PSD results).
    strong=np.flatnonzero(masks['innovation_supported'])
    rejected=np.flatnonzero((abs(dk)>=3)&~masks['innovation_supported'])
    small=np.flatnonzero(abs(dk)==1)
    choices=[]
    if len(strong):choices.extend(strong[np.argsort(abs(dk[strong]))[-2:]].tolist())
    if len(rejected):choices.extend(rejected[np.argsort(abs(dk[rejected]))[-2:]].tolist())
    if len(small):choices.append(int(small[len(small)//2]))
    choices=list(dict.fromkeys(choices))
    fig,axes=plt.subplots(len(choices),2,figsize=(12,3*len(choices)),squeeze=False,constrained_layout=True)
    for row,t in enumerate(choices):
        sl=slice(max(0,t-30),min(n,t+31)); tx=(times[sl]-times[t])*1e6
        for j in range(4):
            v=z['spatial'][sl,j+2].copy();v-=z['spatial'][t-1,j+2]
            axes[row,0].plot(tx,v,label=f'Q{j+1}')
        for name in ['spatial_only','innovation_supported','large_only','full_temporal']:
            v=sig[name][sl,0].copy();v-=sig[name][t-1,0]
            axes[row,1].plot(tx,v,label=name,color=colours[name])
        for ax in axes[row]:ax.axvline(0,color='black',ls=':');ax.set(xlabel='Time relative to transition [us]',ylabel='Relative height [um]');ax.legend(fontsize=7)
        axes[row,0].set_title(f't={t}, current shift={dk[t]}q; regional traces')
        axes[row,1].set_title(f'Mean; innovation support at 4/8/16: {innovation_counts[t].tolist()}')
    fig.savefig(out/'step_event_examples.png',dpi=145);plt.close(fig)
    checks=synthetic_checks(q,fs)
    save_csv(out/'selective_synthetic_checks.csv',checks)
    smallmask=(abs(dk)>0)&(abs(dk)<=2)
    runs=[]; current=0;prev=0
    run_intervals=[]
    for i,val in enumerate(np.where(smallmask,np.sign(dk),0)):
        if val and val==prev:current+=1
        else:
            if current:
                runs.append(current)
                run_intervals.append((current,i-current,i))
            current=1 if val else 0
        prev=val
    if current:
        runs.append(current)
        run_intervals.append((current,n-current,n))
    longest=sorted(run_intervals,reverse=True)[:3]
    fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    run_records=[]
    for row,(length,start,end) in enumerate(longest):
        sl=slice(max(0,start-35),min(n,end+35));tx=(times[sl]-times[start])*1e6
        for j in range(4):
            a=z['spatial'][sl,j+2]-z['spatial'][start-1,j+2]
            axes[row,0].plot(tx,a,label=f'Q{j+1}')
        for name in ['spatial_only','innovation_supported','large_only','full_temporal']:
            a=sig[name][sl,0]-sig[name][start-1,0]
            axes[row,1].plot(tx,a,label=name,color=colours[name])
        for ax in axes[row]:
            ax.axvspan(0,(times[end-1]-times[start])*1e6,color='0.8',alpha=.3)
            ax.set(xlabel='Time from run onset [us]',ylabel='Relative height [um]')
            ax.legend(fontsize=7)
        axes[row,0].set_title(f'{start}–{end-1}: {length} consecutive small corrections')
        axes[row,1].set_title('Spatial mean')
        run_records.append(dict(start=start,stop_exclusive=end,length=length,
                                net_cycles=int(np.sum(dk[start:end])),
                                spatial_mean_change_um=float(z['spatial'][end-1,0]-z['spatial'][start-1,0]),
                                final_mean_change_um=float(z['final'][end-1,0]-z['final'][start-1,0])))
    fig.savefig(out/'consecutive_small_corrections.png',dpi=155);plt.close(fig)
    summary=dict(frames=n,q_um=q,fs_hz=float(fs),tau_um=4.,
                 total_offset_changes=int(np.count_nonzero(dk)),
                 one_or_two_cycle_changes=int(np.sum(smallmask)),
                 changes_at_least_three_cycles=int(np.sum(abs(dk)>=3)),
                 selected_events={name:int(np.count_nonzero(v)) for name,v in accepted.items()},
                 longest_consecutive_same_sign_small_cycle_run=max(runs,default=0),
                 longest_small_cycle_runs=run_records,
                 maximum_export_residual_cycles=float(np.max(z['export_residual_cycles'])),
                 selected_thresholds=dict(min_cycles=3,quadrants=3,jump_residual_q=.35,fit_rms_q=.5,slope_change_q_per_frame=.25,widths_frames=[4,8,16]),
                 innovation_thresholds=dict(min_cycles=3,quadrants=3,required_scales=2,jump_residual_q=1.,minimum_robust_sigma=4.,scale_floor_q=.1),
                 qualifications=['Local linear continuation is an assumption; rejection is not evidence that a jump is physical.',
                                 'Burst overlap can cause missed events; accepted subsets are not complete reconstructions.',
                                 'Power differences include cross terms and need not add; reduction fraction can be negative or exceed one.',
                                 'large_only is an amplitude-only sensitivity bracket, not a validated selector.'],
                 example_timesteps=choices)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'selective_signals.npz',time=times,offset_change_cycles=dk,
                        quadrant_support=counts,innovation_support=innovation_counts,estimated_regional_jumps=jumps,
                        **{name:a for name,a in sig.items()},freq=f,**{'psd_'+name:p for name,p in psds.items()})
    print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--q',type=float,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--output',type=Path,default=Path('runs/temporal_alignment_validation/step_events_0p20'))
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    base=Path('/home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true')
    rawpath=Path(str(base)+'.hdf5')
    cleanpath=Path(str(base)+'_cleaned.hdf5')
    info=inspect_input(rawpath,0,None)
    with h5py.File(cleanpath,'r') as f:
        assert float(f.attrs['q_microns'])==args.q and float(f.attrs['tau_microns'])==4
        assert np.array_equal(f['meta/t'][:],info.time)
    provenance={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in [rawpath,cleanpath,Path('data_analysis/pod/phase_unwrap_gradient_poisson.py')]}
    provenance.update(q=args.q,tau=4.,frames=info.n_frames)
    cache=out/'full_recording_traces.npz';meta=out/'trace_provenance.json'
    if cache.exists() and meta.exists() and json.loads(meta.read_text())==provenance:
        print('Using verified trace cache',flush=True)
        with np.load(cache) as f:z={k:f[k] for k in f.files}
    else:
        n=info.n_frames
        z={name:np.empty((n,6)) for name in ['raw','spatial','final']}
        z.update(time=info.time,median_increments=np.zeros((n,5)),exported_offsets=np.zeros(n,dtype=np.int64),spatial_modified_fraction=np.zeros(n),export_residual_cycles=np.zeros(n))
        chunks=[(s,min(n,s+1000)) for s in range(0,n,1000)]
        started=time.monotonic();completed=0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(extract_chunk,str(rawpath),str(cleanpath),s,e,info.frame_numbers,args.q,4.) for s,e in chunks]
            for job in as_completed(jobs):
                s,e,part=job.result()
                for key,a in part.items():z[key][s:e]=a
                completed+=e-s
                print(f'{completed}/{n} frames; elapsed {time.monotonic()-started:.0f}s',flush=True)
        np.savez_compressed(cache,**z);meta.write_text(json.dumps(provenance,indent=2)+'\n')
    analyze(z,out,args.q)


if __name__=='__main__':main()
