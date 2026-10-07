"""Plot paired raw/POD exact capillary energies and energy ratios from saved results.

No height-field reconstruction is performed. Ratios are computed for each matched
recording before averaging across repetitions. They are not POD variance capture.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path('/home/jonas/ucsd_thesis/reduced_data/capillary_energy')


def read_csv(path):
    return list(csv.DictReader(path.read_text().splitlines()))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_comparison(pod_dir, raw_dir):
    pod_config = json.loads((pod_dir/'configuration.json').read_text())
    raw_config = json.loads((raw_dir/'configuration.json').read_text())
    if pod_config['geometry'] != 'surface' or not np.isclose(pod_config['gamma'], raw_config['gamma_N_per_m'], rtol=1e-12):
        raise ValueError('Need full-surface energies with the same surface tension.')
    if pod_config['basis_scope'] not in ('per-power', 'per-recording') or len({r.get('reference_slot', 1) for r in pod_config['references']}) != 1:
        raise ValueError('Expected one local POD reference per power or each recording\'s own basis.')
    pod_rows = read_csv(pod_dir/'repetition_energy.csv')
    raw_rows = read_csv(raw_dir/'recording_summary.csv')
    raw_lookup = {}
    for row in raw_rows:
        key = (row['power'], int(row['repetition']))
        if key in raw_lookup:
            raise ValueError(f'Duplicate raw recording {key}')
        raw_lookup[key] = row
    ranks = sorted({int(r['rank']) for r in pod_rows})
    powers = sorted({r['power'] for r in pod_rows}, key=lambda p: float(p.replace('p', '.')))
    pod_h5_candidates = [p for p in [pod_dir/'capillary_energy.h5', pod_dir/'capillary_energy_pod.h5'] if p.is_file()]
    if len(pod_h5_candidates) != 1:
        raise ValueError('Expected one POD energy archive (capillary_energy.h5 or capillary_energy_pod.h5).')
    paired, checked, filename_checks = [], set(), 0
    seen = set()
    with h5py.File(pod_h5_candidates[0]) as pod_cache, h5py.File(raw_dir/'raw_capillary_energy.h5') as raw_cache:
        for row in pod_rows:
            key = (row['power'], int(row['repetition']))
            rank = int(row['rank'])
            if pod_config['basis_scope'] == 'per-recording' and (row['reference_power'], int(row['reference_repetition'])) != key:
                raise ValueError(f'Own-basis reference does not match recording {key}')
            if (*key, rank) in seen:
                raise ValueError(f'Duplicate POD result {key}, rank {rank}')
            seen.add((*key, rank))
            if key not in raw_lookup:
                raise ValueError(f'No raw recording for {key}')
            raw = raw_lookup[key]
            if Path(row['source_file']).parent.name != raw['realization']:
                raise ValueError(f'Raw/POD realization identifiers disagree for {key}')
            if row['units'] != 'J' or float(raw['Ebar_exact_J']) <= 0 or not np.isclose(float(raw['gamma_N_per_m']), pod_config['gamma']):
                raise ValueError(f'Invalid units, gamma or raw energy for {key}')
            pod_group = pod_cache[f'recordings/{key[0]}_rep{key[1]}']
            sample_name = f"sampled_trajectories_n{pod_config['exact_frames']}_seed{pod_config['frame_seed']}"
            sample = pod_group[sample_name]
            if bool(sample.attrs.get('temporal_mean_restored', False)) != bool(pod_config.get('restore_temporal_mean', False)):
                raise ValueError(f'Cached/configured mean restoration disagree for {key}')
            raw_group = raw_cache[raw['recording_key']]
            bi = list(pod_group['reference_ids'][:]).index(int(row['reference_id']))
            ri = list(pod_cache['ranks'][:]).index(rank)
            if key not in checked:
                np.testing.assert_array_equal(sample['frame_indices'][:], raw_group['frame_indices'][:])
                np.testing.assert_allclose(sample['time_seconds'][:], raw_group['time_seconds'][:], rtol=0, atol=1e-12)
                np.testing.assert_allclose(sample['mean_weights'][:], raw_group['mean_weights'][:], rtol=1e-12, atol=0)
                ref = pod_cache[f"references/basis_{row['reference_id']}"]
                from data_analysis.energy.pod_capillary_energy import length_scale
                np.testing.assert_allclose(ref['x'][:]*length_scale(pod_config['signature'][2]), raw_group['x_m'][:], rtol=1e-12)
                np.testing.assert_allclose(ref['y'][:]*length_scale(pod_config['signature'][3]), raw_group['y_m'][:], rtol=1e-12)
                if int(row['samples']) != int(raw['full_frame_count']) or int(row['exact_evaluated_frames']) != int(raw['frames_evaluated']):
                    raise ValueError(f'Frame counts differ for {key}')
                source = Path(row['source_file'])
                if source.is_file():
                    with h5py.File(source) as source_cache:
                        if Path(str(source_cache.attrs['source_file'])).name != raw['raw_filename']:
                            raise ValueError(f'Raw source filenames disagree for {key}')
                    filename_checks += 1
                np.testing.assert_allclose(raw_group['E_exact'][:] @ raw_group['mean_weights'][:], float(raw['Ebar_exact_J']), rtol=1e-10)
                checked.add(key)
            pod_energy = float(row['Ebar_exact'])
            raw_energy = float(raw['Ebar_exact_J'])
            if not np.isfinite([pod_energy, raw_energy]).all() or pod_energy < 0:
                raise ValueError(f'Nonfinite/negative energy for {key}')
            np.testing.assert_allclose(sample['E_exact'][bi, ri] @ sample['mean_weights'][:], pod_energy, rtol=1e-10)
            paired.append(dict(power=key[0], repetition=key[1], realization=raw['realization'], rank=rank,
                               raw_exact_mean_J=raw_energy, pod_exact_mean_J=pod_energy,
                               pod_over_raw_percent=100*pod_energy/raw_energy,
                               reference_power=row['reference_power'], reference_repetition=int(row['reference_repetition']),
                               raw_filename=raw['raw_filename']))
    for key in checked:
        if {r['rank'] for r in paired if (r['power'], r['repetition']) == key} != set(ranks):
            raise ValueError(f'Missing ranks for {key}')
    audit = dict(matched_recordings=len(checked), ranks=ranks, common_powers=powers, basis_scope=pod_config['basis_scope'],
                 frame_indices_times_weights_and_grids_match=True, raw_filename_checks=filename_checks,
                 raw_only_recordings=[dict(power=k[0], repetition=k[1]) for k in raw_lookup if k not in checked],
                 raw_temporal_mean_removed=raw_config['temporal_mean_removed'],
                 pod_temporal_mean_removed=bool(pod_config['signature'][1]) and not pod_config.get('restore_temporal_mean', False),
                 pod_temporal_mean_restored=bool(pod_config.get('restore_temporal_mean', False)),
                 gamma_N_per_m=pod_config['gamma'], sampled_frames=pod_config['exact_frames'],
                 pod_directory=str(pod_dir), raw_directory=str(raw_dir))
    return paired, audit


def summarize(paired):
    rows = []
    for power in sorted({r['power'] for r in paired}, key=lambda p: float(p.replace('p', '.'))):
        for rank in sorted({r['rank'] for r in paired}):
            selected = [r for r in paired if r['power'] == power and r['rank'] == rank]
            raw = np.array([r['raw_exact_mean_J'] for r in selected])
            pod = np.array([r['pod_exact_mean_J'] for r in selected])
            ratio = 100*pod/raw
            rows.append(dict(power=power, rank=rank, repetitions=len(selected),
                             raw_mean_J=float(raw.mean()), raw_sd_J=float(raw.std(ddof=1)) if len(raw)>1 else 0.,
                             pod_mean_J=float(pod.mean()), pod_sd_J=float(pod.std(ddof=1)) if len(pod)>1 else 0.,
                             mean_pod_over_raw_percent=float(ratio.mean()),
                             sd_pod_over_raw_percent=float(ratio.std(ddof=1)) if len(ratio)>1 else 0.,
                             ratio_of_power_means_percent=float(100*pod.mean()/raw.mean())))
    return rows


def compare_basis_scopes(native, shared):
    """Paired comparison: same recording, rank and raw denominator."""
    lookup = {(r['power'], r['repetition'], r['rank']): r for r in shared}
    if set(lookup) != {(r['power'], r['repetition'], r['rank']) for r in native}:
        raise ValueError('Own/shared studies must contain identical recordings and ranks.')
    pairs = []
    for row in native:
        previous = lookup[row['power'], row['repetition'], row['rank']]
        np.testing.assert_allclose(row['raw_exact_mean_J'], previous['raw_exact_mean_J'], rtol=1e-12, atol=0)
        own_energy, shared_energy = row['pod_exact_mean_J'], previous['pod_exact_mean_J']
        pairs.append(dict(power=row['power'], repetition=row['repetition'], rank=row['rank'],
                          own_exact_mean_J=own_energy, shared_exact_mean_J=shared_energy,
                          shared_reference_repetition=previous['reference_repetition'],
                          own_over_raw_percent=row['pod_over_raw_percent'],
                          shared_over_raw_percent=previous['pod_over_raw_percent'],
                          shared_over_own_percent=100*shared_energy/own_energy if own_energy > 0 else float('nan')))
    summaries = []
    for power, rank in sorted({(r['power'], r['rank']) for r in pairs}):
        selected = [r for r in pairs if r['power'] == power and r['rank'] == rank]
        values = np.array([r['shared_over_own_percent'] for r in selected])
        summaries.append(dict(power=power, rank=rank, repetitions=len(selected),
                              own_mean_J=float(np.mean([r['own_exact_mean_J'] for r in selected])),
                              shared_mean_J=float(np.mean([r['shared_exact_mean_J'] for r in selected])),
                              mean_shared_over_own_percent=float(values.mean()),
                              sd_shared_over_own_percent=float(values.std(ddof=1)) if len(values)>1 else 0.,
                              mean_own_over_raw_percent=float(np.mean([r['own_over_raw_percent'] for r in selected])),
                              mean_shared_over_raw_percent=float(np.mean([r['shared_over_raw_percent'] for r in selected]))))
    return pairs, summaries


def run(args):
    paired, audit = load_comparison(args.pod_dir, args.raw_dir)
    summaries = summarize(paired)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'paired_recording_energies.csv', paired)
    write_csv(out/'power_rank_summary.csv', summaries)
    (out/'matching_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    ranks = audit['ranks']
    basis_label = 'Own POD basis per recording' if audit['basis_scope'] == 'per-recording' else 'Shared POD basis per power'
    if audit['pod_temporal_mean_restored']:
        basis_label += ' · temporal mean restored'
    if not audit['raw_temporal_mean_removed'] and not audit['pod_temporal_mean_removed']:
        mean_note = 'Raw and POD both retain their temporal mean surfaces.'
    elif audit['raw_temporal_mean_removed'] and audit['pod_temporal_mean_removed']:
        mean_note = 'Raw and POD both exclude their temporal mean surfaces.'
    else:
        mean_note = ('Raw '+('excludes' if audit['raw_temporal_mean_removed'] else 'includes')+' the temporal mean surface; POD '+
                     ('excludes' if audit['pod_temporal_mean_removed'] else 'includes')+' it. Mean treatment differs.')
    shared_summaries = None
    if args.shared_pod_dir is not None:
        shared_pairs, shared_audit = load_comparison(args.shared_pod_dir, args.raw_dir)
        if audit['basis_scope'] != 'per-recording' or shared_audit['basis_scope'] != 'per-power':
            raise ValueError('--shared-pod-dir requires a per-recording primary study and a per-power comparison study.')
        if audit['pod_temporal_mean_removed'] != shared_audit['pod_temporal_mean_removed']:
            raise ValueError('Own/shared basis comparison requires the same temporal-mean treatment.')
        basis_pairs, basis_summaries = compare_basis_scopes(paired, shared_pairs)
        write_csv(out/'own_vs_shared_recording_energies.csv', basis_pairs)
        write_csv(out/'own_vs_shared_power_summary.csv', basis_summaries)
        shared_summaries = summarize(shared_pairs)
        (out/'shared_matching_audit.json').write_text(json.dumps(shared_audit, indent=2)+'\n')
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False}):
        with PdfPages(out/'raw_pod_energy_overview.pdf') as pdf:
            fig, ax = plt.subplots(figsize=(9, 5.8), layout='constrained')
            raw = [r for r in summaries if r['rank'] == ranks[0]]
            x = np.array([float(r['power'].replace('p', '.')) for r in raw])
            ax.errorbar(x, np.array([r['raw_mean_J'] for r in raw])*1e9,
                        yerr=np.array([r['raw_sd_J'] for r in raw])*1e9, color='#222222',
                        fmt='o-', linewidth=2.2, capsize=3, label='Raw measured surface')
            for i, rank in enumerate(ranks):
                selected = [r for r in summaries if r['rank']==rank]
                ax.errorbar(x, np.array([r['pod_mean_J'] for r in selected])*1e9,
                            yerr=np.array([r['pod_sd_J'] for r in selected])*1e9,
                            color=colors[i % len(colors)], fmt='o-', capsize=3, label=f'POD rank {rank}')
            ax.set(xlabel='Input amplitude (Vpp)', ylabel='Mean exact capillary excess energy (nJ)',
                   title='Raw and POD surface energy\n'+basis_label, xticks=x)
            ax.grid(alpha=.2); ax.legend(loc='upper left')
            fig.suptitle(f"Matched repetitions · mean ± SD · {audit['sampled_frames']:,} sampled frames", fontsize=10, color='#555555')
            fig.savefig(out/'energy_over_power.pdf'); fig.savefig(out/'energy_over_power.png', dpi=200); pdf.savefig(fig)
            if np.any(x <= .04):
                low = [r for r in summaries if float(r['power'].replace('p', '.')) <= .08]
                limits = [r[f'{kind}_mean_J']+sign*r[f'{kind}_sd_J'] for r in low for kind in ['raw', 'pod'] for sign in [-1, 1]]
                lower, upper = min(0., min(limits))*1e9, max(limits)*1e9
                padding = max(upper-lower, np.finfo(float).tiny)*.08
                ax.set(xlim=(float(x.min())-.005, min(.08, float(x.max()))+.005), ylim=(lower-padding, upper+padding),
                       title='Zero / low-input surface energy\n'+basis_label)
                fig.savefig(out/'energy_low_power_zoom.pdf'); fig.savefig(out/'energy_low_power_zoom.png', dpi=200); pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(9, 5.8), layout='constrained')
            for i, rank in enumerate(ranks):
                selected = [r for r in summaries if r['rank']==rank]
                ax.errorbar(x, [r['mean_pod_over_raw_percent'] for r in selected],
                            yerr=[r['sd_pod_over_raw_percent'] for r in selected],
                            color=colors[i % len(colors)], fmt='o-', capsize=3, label=f'POD rank {rank}')
            ax.axhline(100, color='#666666', ls='--', lw=1, label='Equal raw and POD energy')
            ax.set(xlabel='Input amplitude (Vpp)', ylabel='POD / raw exact energy (%)',
                   title='POD energy relative to the matched raw recording\n'+basis_label, xticks=x)
            ax.grid(alpha=.2); ax.legend(loc='best', fontsize=10)
            fig.suptitle('Ratio computed per repetition, then mean ± SD across repetitions', fontsize=10, color='#555555')
            fig.text(.5, -.025, mean_note+' Energy ratios are not bounded by 100%.',
                     ha='center', fontsize=9, color='#555555')
            fig.savefig(out/'pod_raw_energy_percent.pdf', bbox_inches='tight'); fig.savefig(out/'pod_raw_energy_percent.png', dpi=200, bbox_inches='tight'); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            if shared_summaries is not None:
                fig, axes = plt.subplots((len(ranks)+1)//2, 2, figsize=(11, 8), squeeze=False, layout='constrained')
                for i, (rank, ax) in enumerate(zip(ranks, axes.flat)):
                    own = [r for r in summaries if r['rank'] == rank]
                    shared = [r for r in shared_summaries if r['rank'] == rank]
                    for entries, key, color, style, label in [
                            (own, 'raw', '#777777', ':', 'Raw surface'),
                            (own, 'pod', colors[i % len(colors)], '-', 'Own POD basis'),
                            (shared, 'pod', colors[i % len(colors)], '--', 'Shared basis per power')]:
                        ax.errorbar(x, np.array([r[f'{key}_mean_J'] for r in entries])*1e9,
                                    yerr=np.array([r[f'{key}_sd_J'] for r in entries])*1e9,
                                    color=color, linestyle=style, marker='o', markersize=4, capsize=2, label=label)
                    ax.set(title=f'Rank {rank}', xlabel='Input amplitude (Vpp)', ylabel='Mean exact energy (nJ)', xticks=x)
                    ax.grid(alpha=.2); ax.legend(fontsize=8)
                for ax in list(axes.flat)[len(ranks):]:
                    ax.set_visible(False)
                fig.suptitle('Effect of using each recording’s own POD basis\nSame recordings and sampled frames · mean ± SD')
                fig.savefig(out/'own_vs_shared_basis.pdf'); fig.savefig(out/'own_vs_shared_basis.png', dpi=180); pdf.savefig(fig); plt.close(fig)
    powers = audit['common_powers']
    table = '| Power | '+' | '.join(f'Rank {r}' for r in ranks)+' |\n|---|'+'---:|'*len(ranks)+'\n'
    for power in powers:
        table += '| '+power+' | '+' | '.join(f"{next(r['mean_pod_over_raw_percent'] for r in summaries if r['power']==power and r['rank']==rank):.1f}%" for rank in ranks)+' |\n'
    text = f'''# Raw versus POD exact capillary-energy overview

Matched {audit['matched_recordings']} recordings at {len(powers)} common powers; ranks {ranks}. {basis_label}.
Both estimates use gamma = {audit['gamma_N_per_m']} N/m and {audit['sampled_frames']} selected frames. Matched frame indices, selected times, integration weights and spatial grids were verified in the archives. CSV energies were checked against the weighted archived time samples.
Matching uses power/repetition and realization directory, with {audit['raw_filename_checks']} additional original raw-filename checks against available source POD metadata.

## Plots

- `energy_over_power.pdf/png`: exact geometric energy, averaged across matched repetitions, in nJ. Bars show repetition SD.
- `pod_raw_energy_percent.pdf/png`: 100 × POD exact mean / matched raw exact mean, computed separately for each repetition and then averaged. Bars show SD of these ratios, not propagated marginal SD or confidence intervals.
- `raw_pod_energy_overview.pdf`: both figures.

The {len(powers)} common powers are shown: {', '.join(powers)}. Any raw-only recordings ({len(audit['raw_only_recordings'])}) are listed in matching_audit.json and excluded from these comparisons. When zero/low-input powers are present, `energy_low_power_zoom.pdf/png` also shows powers up to 0p08 on an expanded energy scale. Those recordings provide a baseline that may contain both measurement noise and static mean-surface structure; neither contribution is subtracted here.

## Interpretation

{mean_note} The percentage is a POD/raw energy ratio. Truncation and raw spatial measurement noise can contribute to the difference. In per-power mode there is an additional cross-recording subspace projection; per-recording mode uses the original coefficients directly. When restored, each full stored mean surface is added before differentiation and energy evaluation, including its interactions with the fluctuating slopes.
This is not cumulative POD displacement variance. The spatial derivative energy metric and nonlinear geometric formula do not guarantee monotonic ratios with rank or a bound of 100%; values and error bars have not been clipped.
Both are exact geometric formulas evaluated on their respective sampled fields; the time means are estimates from {audit['sampled_frames']:,} frames. Repetition SD does not separately quantify frame-sampling uncertainty. No significance claim is made.

Mean of paired repetition-level percentages:

{table}

Tables: `paired_recording_energies.csv` and `power_rank_summary.csv`. The latter also saves the ratio of power-level mean energies as a separate statistic; the figure uses the mean of individual recording ratios.

Inputs:
- POD: `{args.pod_dir}`
- Raw: `{args.raw_dir}`

```bash
python compare_raw_pod_capillary_energy.py --pod-dir {args.pod_dir} --raw-dir {args.raw_dir} --output-dir {out}
```
'''
    (out/'README.md').write_text(text)
    if shared_summaries is not None:
        with (out/'README.md').open('a') as handle:
            handle.write('\n## Own versus shared basis\n\n'
                         '`own_vs_shared_basis.pdf/png` compares raw, own-basis and shared-basis energies at each rank. '
                         'The same recordings, frame indices, times, spatial grids and integration weights are checked against the same raw archive for both studies. '
                         'The own-basis calculation directly reconstructs each recording from its stored coefficients; '
                         'the shared calculation additionally projects those coefficients into another repetition’s subspace. '
                         'The own basis maximizes retained displacement variance, not capillary energy. Neither energy nor their ratio is guaranteed to be monotonic or bounded by the other.\n\n'
                         'Tables: `own_vs_shared_recording_energies.csv` and `own_vs_shared_power_summary.csv`. '
                         'Shared/own percentages are computed per recording before averaging.\n\n'
                         f'Shared study: `{args.shared_pod_dir}`. Reproduce with the command above plus `--shared-pod-dir {args.shared_pod_dir}`.\n')
    print(f'Saved {out}; matched {audit["matched_recordings"]} recordings.', flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pod-dir', type=Path, default=ROOT/'surface_per_power_ranks_10_100_200_1000_frames_2000')
    p.add_argument('--raw-dir', type=Path, default=ROOT/'raw_capillary_energy_2000frames')
    p.add_argument('--shared-pod-dir', type=Path, help='Optional previous per-power study to compare with a per-recording study.')
    p.add_argument('--output-dir', type=Path, default=ROOT/'raw_vs_pod_energy_overview')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
