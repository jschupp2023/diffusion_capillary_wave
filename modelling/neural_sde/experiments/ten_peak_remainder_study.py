"""Resumable fixed controller for the ten-peak neural remainder pilot."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

from .adaptive_controller import read_json, write_json, run_logged


FRESH_BASE = [
    '--experiment','0p10','--rank','10','--validation-reps','13','14','--test-reps','15','16',
    '--spatial-mean-highpass-hz','50','--state-variable','velocity','--lag','1',
    '--history-offsets','1','--drift-type','damped_residual','--damped-operator','diagonal',
    '--stiffness-init','0.01','--damping-init','0.01','--residual-architecture','tcn',
    '--tcn-context-frames','63','--tcn-channels','64','--tcn-kernel-size','3',
    '--tcn-dilations','1','2','4','8','16','--diffusion-type','diagonal','--diffusion-init','0.1',
    '--train-mode','joint','--train-trim-percent','0','--batch-size','256','--seed','0',
    '--tcn-rollout-cache','--indexed-training-batches','--no-tcn-rollout-compile',
    '--stationary-mean-weight','0','--stationary-variance-weight','0',
    '--stationary-lagged-covariance-weight','0']


def scale_token(value):
    return str(value).replace('.','p')


def compact_report(path):
    report=read_json(path/'summary.json')['reports']
    keys=('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags',
          'increment_variance_ratio','generated_state_rms','reference_state_rms')
    result={name:{key:values[key] for key in keys} for name,values in report.items()}
    for name in ('remainder_raw','hybrid_raw'):
        summary=path/name/'normal_rollout_plots/quantitative/summary.json'
        if not summary.exists():continue
        standard=read_json(summary)
        moments=standard['coefficient_statistics']['untrimmed']
        psd=standard['psd']['signals']
        result[name]['standard_rollout']={
            'energy_ratio':standard['energy']['untrimmed']['generated_over_reference'],
            'pod_covariance_trace_ratio':moments['generated_covariance_trace']/moments['reference_covariance_trace'],
            'spatial_mean_variance_ratio':moments['generated_variance'][0]/moments['reference_variance'][0],
            'center_full_psd_log10_mae':psd['center']['mean_absolute_log10_band_power_error_decades'],
            'pod_only_full_psd_log10_mae':psd['center_without_spatial_mean']['mean_absolute_log10_band_power_error_decades'],
            'spatial_mean_full_psd_log10_mae':psd['spatial_mean']['mean_absolute_log10_band_power_error_decades']}
    return result


def run_study(path,dry_run=False):
    path=Path(path).resolve();study=read_json(path);repo=Path(study['repository'])
    root=repo/study['output_root'];python=study['python'];manifest=root/'data/manifest.json'
    workflow=study.get('workflow_module','modelling.frequency_hybrid.ten_peak_remainder')
    exp=study.get('experiment','0p10')
    prep=[python,'-m',workflow,'prepare','--output',str(root),'--experiment',exp]
    if study.get('preparation_uses_oscillators',False):
        prep+=['--oscillators',str(repo/study['oscillators'])]
    commands=[]
    fresh_base=list(FRESH_BASE)
    if '--experiment' in fresh_base:
        fresh_base[fresh_base.index('--experiment')+1]=exp
    for arm in study['arms']:
        run=root/'trials'/arm['id']
        train=[python,'-m','modelling.neural_sde','train']
        if arm['mode']=='fresh':
            train+=fresh_base+['--temporal-component-manifest',str(manifest),
                              '--temporal-component','remainder']
        else:
            source=(repo/arm['checkpoint'] if arm.get('checkpoint') else
                    root/'trials'/arm['parent']/'best.pt')
            train+=['--checkpoint',str(source),'--train-mode','joint',
                    '--batch-size','256','--seed','0','--tcn-rollout-cache','--indexed-training-batches',
                    '--no-tcn-rollout-compile']
        train+=arm['options']+['--output',str(run)];commands.append((f"train_{arm['id']}",train))
        for scale in study['evaluation']['diffusion_scales']:
            ev=study['evaluation'];output=run/f'evaluation_scale{scale_token(scale)}'
            command=[python,'-m',workflow,'evaluate',
                '--checkpoint',str(run/'best.pt'),'--output',str(output),'--study',str(root),
                '--oscillators',str(repo/study['oscillators']),'--horizon',str(ev['horizon']),
                '--conditions',str(ev['conditions']),'--ensemble',str(ev['ensemble']),
                '--seed',str(ev['seed']),'--device',ev['device'],'--diffusion-scale',str(scale)]
            commands.append((f"evaluate_{arm['id']}_scale{scale_token(scale)}",command))
    if dry_run:
        print('PREP',prep)
        for label,command in commands:print(label,command)
        return
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('A controller is already running this study') from None
    digest=hashlib.sha256(path.read_bytes()).hexdigest();state_path=root/'state.json'
    state=read_json(state_path) if state_path.exists() else dict(version=1,status='active',
        study_sha256=digest,started_at_unix=time.time(),preparation='planned',trials={})
    if state['study_sha256']!=digest:raise ValueError('Study changed after launch; use a new output root')
    if state['status'] not in ('active','stopped'):
        print(f"Study is {state['status']}; refusing to restart");return
    state['status']='active';state['controller_pid']=os.getpid()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLCONFIGDIR=str(root/'.mplconfig'),PYTHONUNBUFFERED='1')
    bin_dir=str(Path(python).parent);env['PATH']=bin_dir+os.pathsep+env.get('PATH','')
    ptxas=Path(bin_dir)/'ptxas'
    if ptxas.exists():env['TRITON_PTXAS_PATH']=str(ptxas)
    deadline=state['started_at_unix']+study['total_timeout_seconds']
    def save():
        state['elapsed_seconds']=time.time()-state['started_at_unix'];write_json(state_path,state)
    def job(command,label,timeout):
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('Whole-study time budget exhausted')
        state['current_job']=label;save()
        run_logged(command,root/'logs'/f'{label}.log',repo,env,timeout_seconds=min(timeout,remaining))
    def stop(signum,frame):raise KeyboardInterrupt
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGHUP)}
    save()
    try:
        if not manifest.exists():
            state['preparation']='running';save();job(prep,'prepare',900)
        state['preparation']='completed';save()
        for arm in study['arms']:
            row=state['trials'].setdefault(arm['id'],dict(status='planned',description=arm['description'],evaluations={}))
            parent=arm.get('parent')
            if parent and state['trials'].get(parent,{}).get('status') not in (
                    'trained','completed','completed_with_evaluation_failures'):
                row['status']='cancelled_dependency';save();continue
            run=root/'trials'/arm['id']
            try:
                if not (run/'best.pt').exists():
                    row['status']='training';save()
                    command=next(command for label,command in commands if label==f"train_{arm['id']}")
                    job(command,f"train_{arm['id']}",study['training_timeout_seconds'])
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                row.update(status='failed',error=repr(error));save()
                continue
            row['status']='trained';row.pop('error',None);save()
            for scale in study['evaluation']['diffusion_scales']:
                token=scale_token(scale);evrow=row['evaluations'].setdefault(token,dict(status='planned'))
                if evrow['status'] in ('failed','skipped'):
                    continue
                output=run/f'evaluation_scale{token}'
                try:
                    if not (output/'summary.json').exists():
                        evrow['status']='running';save()
                        label=f"evaluate_{arm['id']}_scale{token}"
                        command=next(command for name,command in commands if name==label)
                        job(command,label,study['evaluation_timeout_seconds'])
                    if study.get('normal_rollout_diagnostics',False):
                        for case in ('remainder_raw','hybrid_raw'):
                            plot_output=output/case/'normal_rollout_plots'
                            if (plot_output/'quantitative/summary.json').exists():continue
                            plot=[python,'-m','data_analysis.plot_rollout',str(output/case),
                                  '--output',str(plot_output),'--nperseg','8192',
                                  '--peak-nperseg','32768','--overwrite']
                            job(plot,f"plot_{arm['id']}_scale{token}_{case}",
                                study['evaluation_timeout_seconds'])
                    evrow.update(status='completed',metrics=compact_report(output));save()
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                    evrow.update(status='failed',error=repr(error));save()
            row['status']=('completed' if all(ev['status']=='completed' for ev in row['evaluations'].values())
                           else 'completed_with_evaluation_failures')
            save()
        state['status']='completed' if all(row['status']=='completed' for row in state['trials'].values()) else 'completed_with_failures'
    except TimeoutError as error:state.update(status='budget_exhausted',error=str(error))
    except KeyboardInterrupt:
        state['status']='stopped';raise
    except Exception as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['current_job']=None;save()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        lines=['# Neural-remainder controller','',f"Status: {state['status']}",'',
            'Each scale evaluates the identical checkpoint both raw and after the configured fixed remainder operator.','',
            '| Trial | Status | Evaluations |','|---|---|---|']
        for name,row in state['trials'].items():lines.append(f"| {name} | {row['status']} | {', '.join(row['evaluations'])} |")
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(state_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study',type=Path)
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args();run_study(args.study,args.dry_run)


if __name__=='__main__':main()
