"""Read-only tests of the exported cleaner's median temporal integer alignment.

Uses real low-forcing backgrounds, known uniform motion and known integer jumps.
Spatial processing is evaluated separately on selected full-resolution 0p20 windows.
The scalar sweep is the exact real-arithmetic reduction of the exporter recurrence;
a direct float32 frame recurrence is also checked. No filtering is used in these tests.
"""
from pathlib import Path
import argparse
import csv
import json
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis.pod.pod_2d import inspect_input, _frame_key
from data_analysis.pod.phase_unwrap_gradient_poisson import restore_frame_poisson


def medians(d):
    y, x = np.array(d.shape) // 2
    return np.array([np.median(d), np.median(d[:y, :x]),
                     np.median(d[:y, x:]), np.median(d[y:, :x]),
                     np.median(d[y:, x:])])


def offsets(increments, q):
    return np.r_[0, np.cumsum(np.rint(increments / q).astype(np.int64))]


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def inspect_events(data_root, out, q, tau):
    """Inspect representative large transitions and the rounding boundary."""
    name = '11272025_c_0.2vpp_data_roi-none_cal-true'
    events = [1563, 23147, 70100, 60126]
    fig, axes = plt.subplots(len(events), 3, figsize=(12, 11), constrained_layout=True)
    records = []
    with h5py.File(data_root/(name+'.hdf5'), 'r') as r, h5py.File(data_root/(name+'_cleaned.hdf5'), 'r') as c:
        for row,t in enumerate(events):
            raw = [r['main'][_frame_key(i)][:] for i in [t-1,t]]
            spatial = [restore_frame_poisson(a,q=q,tau=tau).restored for a in raw]
            final = [c['main'][_frame_key(i)][:] for i in [t-1,t]]
            d = spatial[1].astype(float)-spatial[0]
            df = final[1].astype(float)-final[0]
            med = medians(d)
            votes = np.rint(med/q).astype(int)
            actual_k = int(np.rint(float(np.median(spatial[1]-final[0]))/q))
            reconstructed = spatial[1].copy()
            if actual_k:
                reconstructed -= actual_k*q
            records.append(dict(timestep=t,region_increment_medians_um=med.tolist(),
                                region_cycle_votes=votes.tolist(),
                                mean_final_increment_um=float(df.mean()),
                                median_final_increment_um=float(np.median(df)),
                                direct_exporter_reproduces_frame=bool(np.array_equal(reconstructed,final[1]))))
            lim = max(5.,float(np.percentile(abs(d),99)))
            for j,(field,title) in enumerate([(d,'Spatial-only increment'),(df,'After global alignment')]):
                im=axes[row,j].imshow(field,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim)
                axes[row,j].set(title=f'{t-1} → {t}: {title}',xlabel='x pixel',ylabel='y pixel')
                fig.colorbar(im,ax=axes[row,j],label='um',shrink=.7)
            for j,(ys,xs) in enumerate([(slice(0,100),slice(0,100)),(slice(0,100),slice(100,None)),(slice(100,None),slice(0,100)),(slice(100,None),slice(100,None))]):
                axes[row,2].hist(d[ys,xs].ravel(),bins=80,density=True,histtype='step',label=f'Q{j+1}: {votes[j+1]} cycles')
            axes[row,2].axvline(med[0],color='black',ls='--',label=f'Global: {votes[0]} cycles')
            axes[row,2].set(xlabel='Spatial-only increment [um]',ylabel='Density')
            axes[row,2].legend(fontsize=7)
    fig.savefig(out/'regional_event_inspection.png',dpi=150)
    plt.close(fig)
    (out/'event_inspection.json').write_text(json.dumps(records,indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--q', type=float, required=True, help='Micrometres per phase cycle')
    ap.add_argument('--tau', type=float, default=4.)
    ap.add_argument('--data-root', type=Path, default=Path('/home/jonas/ucsd_thesis'))
    ap.add_argument('--output', type=Path, default=Path('runs/temporal_alignment_validation'))
    ap.add_argument('--control-frames', type=int, default=8192)
    args = ap.parse_args()
    q = args.q
    if not np.isfinite(q) or q <= 0:
        raise ValueError('q must be positive and finite')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    controls, direct_checks = {}, []
    for case, filename in [('0p04', '11222025_j_0.04vpp_data_roi-none_cal-true.hdf5'),
                           ('0p08', '11222025_h_0.08vpp_data_roi-none_cal-true.hdf5')]:
        path = args.data_root / filename
        info = inspect_input(path, 0, args.control_frames)
        times = np.asarray(info.time, dtype=float)
        ds, means, small = [], [], []
        max_edge = 0.
        with h5py.File(path, 'r') as h:
            ids = h['meta/frames'][:len(times)]
            for i, frame_id in enumerate(ids):
                a = h['main'][_frame_key(int(frame_id))][:].astype(float)
                means.append(float(a.mean()))
                max_edge = max(max_edge, float(np.max(abs(np.diff(a, axis=0)))),
                               float(np.max(abs(np.diff(a, axis=1)))))
                if i:
                    ds.append(medians(a - prev))
                if i < 256:
                    small.append(a.astype(np.float32))
                prev = a
        ds, means = np.array(ds), np.array(means)
        controls[case] = dict(times=times, increments=ds[:, 0], means=means,
                              max_edge=max_edge, offsets=offsets(ds[:, 0], q))
        # Direct exporter recurrence, full frames, float32; known q jumps and
        # a subthreshold 140 Hz oscillation. Compare with scalar reduction.
        n = len(small)
        added = 10 * np.sin(2*np.pi*140*(times[:n]-times[0]))
        jumps = np.zeros(n, dtype=int)
        jumps[n//4:n//2] = 5
        jumps[n//2:3*n//4] = -3
        jumps[3*n//4:] = 9
        scalar = offsets(ds[:n-1, 0] + np.diff(added + q*jumps), q)
        ks, max_error = [0], 0.
        for i, a in enumerate(small):
            current = (a.astype(float) + added[i] + q*jumps[i]).astype(np.float32)
            k = 0 if i == 0 else int(np.rint(float(np.median(current-previous))/q))
            if k:
                current -= k*q
            expected = a.astype(float) + added[i] + q*jumps[i] - q*scalar[i]
            max_error = max(max_error, float(np.max(abs(current-expected))))
            previous = current
            if i:
                ks.append(k)
        assert np.array_equal(ks, scalar), 'Scalar and frame recurrence branch disagreement'
        direct_checks.append(dict(case=case, frames=n, max_float32_error_um=max_error,
                                  branch_labels_equal=True))
        print(case, 'control loaded', len(times), 'max edge', max_edge,
              'temporal offset changes', np.count_nonzero(np.diff(controls[case]['offsets'])), flush=True)

    frequencies = [70, 140, 197, 365, 500, 1000]
    amplitudes = [1, 5, 10, 20, 40, 80, 160]
    sweep, examples = [], {}
    for case, c in controls.items():
        t = c['times'] - c['times'][0]
        n = len(t)
        jumps = np.zeros(n, dtype=int)
        jumps[n//5:2*n//5] = 5
        jumps[2*n//5:3*n//5] = -3
        jumps[3*n//5:4*n//5] = 9
        baseline = c['means']-q*c['offsets']
        for f in frequencies:
            design = np.column_stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t),
                                      np.ones(n), (t-t.mean())/np.ptp(t)])
            for a in amplitudes:
                motion = a*np.sin(2*np.pi*f*t)
                k = offsets(c['increments']+np.diff(motion), q)
                kj = offsets(c['increments']+np.diff(motion+q*jumps), q)
                recovered = c['means']+motion-q*k-baseline
                beta = np.linalg.lstsq(design, recovered, rcond=None)[0]
                gain = np.hypot(beta[0], beta[1])/a
                wrong = k-c['offsets']
                sweep.append(dict(case=case, frequency_hz=f, amplitude_um=a,
                                  max_physical_increment_um=float(np.max(abs(np.diff(motion)))),
                                  added_motion_amplitude_ratio=float(gain),
                                  phase_shift_deg=float(np.degrees(np.arctan2(beta[1], beta[0]))),
                                  unintended_offset_changes=int(np.count_nonzero(np.diff(wrong))),
                                  max_error_um=float(np.max(abs(recovered-motion))),
                                  rms_error_um=float(np.sqrt(np.mean((recovered-motion)**2))),
                                  injected_jump_recovery_fraction=float(np.mean(kj-k == jumps))))
                if case == '0p04' and (f,a) in [(140,20),(1000,10),(1000,20),(365,80)]:
                    examples[f'{f} Hz, {a} um'] = (t, motion, recovered)
    save_csv(out/'motion_injection_sweep.csv', sweep)
    print('Injection sweep complete', flush=True)

    # Full-resolution spatial proposals + saved final fields: no reset of the
    # exported recording's global offset. Compare transition offsets, not totals.
    filename = '11272025_c_0.2vpp_data_roi-none_cal-true'
    info = inspect_input(args.data_root/(filename+'.hdf5'), 0, None)
    rows, windows = [], []
    starts = [1400, 2200, 22900, 40000, 60000, 69950, 75000, 90000]
    with h5py.File(args.data_root/(filename+'.hdf5'), 'r') as raw, h5py.File(args.data_root/(filename+'_cleaned.hdf5'), 'r') as clean:
        assert abs(float(clean.attrs['q_microns'])-q)<1e-10
        assert float(clean.attrs['tau_microns']) == args.tau
        ids = raw['meta/frames'][:]
        max_uniform_residual = 0.
        for start in starts:
            local = []
            for i in range(start, start+301):
                key = _frame_key(int(ids[i]))
                a = raw['main'][key][:]
                b = restore_frame_poisson(a, q=q, tau=args.tau, method='integer').restored.astype(float)
                final = clean['main'][key][:].astype(float)
                k_float = (b-final)/q
                k_export = int(np.rint(np.median(k_float)))
                max_uniform_residual = max(max_uniform_residual, float(np.max(abs(k_float-k_export))))
                if i > start:
                    delta = b-prev
                    med = medians(delta)
                    votes = np.rint(med/q).astype(int)
                    export_step = k_export-prev_k
                    row = dict(window_start=start, timestep=i,
                               global_step_cycles=int(votes[0]), exported_step_cycles=export_step,
                               quadrants_agree=int(np.sum(votes[1:]==votes[0])),
                               quadrant_vote_span=int(np.ptp(votes[1:])),
                               median_increment_um=float(med[0]),
                               final_mean_increment_um=float(np.mean(final-prev_final)),
                               max_raw_gradient_um=float(max(np.max(abs(np.diff(a,axis=0))),np.max(abs(np.diff(a,axis=1))))),
                               rounding_margin_um=float(q/2-abs(med[0]-q*votes[0])))
                    rows.append(row)
                    local.append(row)
                prev, prev_final, prev_k = b, final, k_export
            windows.append(dict(start=start, transitions=len(local),
                                offset_changes=sum(r['global_step_cycles']!=0 for r in local),
                                all_quadrants_agree_fraction=float(np.mean([r['quadrants_agree']==4 for r in local])),
                                majority_quadrants_agree_fraction=float(np.mean([r['quadrants_agree']>=3 for r in local])),
                                exporter_matches=sum(r['global_step_cycles']==r['exported_step_cycles'] for r in local)))
            print('0p20 window', windows[-1], flush=True)
    save_csv(out/'regional_agreement.csv', rows)
    changes = [r for r in rows if r['global_step_cycles'] != 0]
    summary = dict(q_um=q,tau_um=args.tau,
                   scope='8192-frame controls; selected 0p20 windows, not a population estimate; synthetic uniform motion is not asserted physical ground truth for 0p20',
                   direct_float32_checks=direct_checks,
                   controls={name: dict(frames=len(c['times']),max_edge_um=c['max_edge'],
                                        temporal_offset_changes=int(np.count_nonzero(np.diff(c['offsets']))),
                                        max_abs_median_increment_um=float(max(abs(c['increments'])))) for name,c in controls.items()},
                   windows=windows, transitions=len(rows),offset_change_transitions=len(changes),
                   changed_all_quadrants_agree_fraction=float(np.mean([r['quadrants_agree']==4 for r in changes])),
                   changed_majority_quadrants_agree_fraction=float(np.mean([r['quadrants_agree']>=3 for r in changes])),
                   total_exporter_branch_mismatches=sum(r['global_step_cycles']!=r['exported_step_cycles'] for r in rows),
                   max_exported_uniform_integer_residual_cycles=max_uniform_residual,
                   min_injected_jump_recovery_fraction=min(r['injected_jump_recovery_fraction'] for r in sweep))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,(title,(t,m,r)) in zip(axes.flat,examples.items()):
        mask=t<.02
        ax.plot(t[mask]*1000,m[mask],label='Known added motion')
        ax.plot(t[mask]*1000,r[mask],label='Motion retained after alignment',alpha=.8)
        ax.set(title=title,xlabel='Time [ms]',ylabel='Height [um]')
        ax.legend(fontsize=8)
    fig.savefig(out/'known_motion_recovery.png',dpi=160)
    plt.close(fig)
    inspect_events(args.data_root, out, q, args.tau)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
