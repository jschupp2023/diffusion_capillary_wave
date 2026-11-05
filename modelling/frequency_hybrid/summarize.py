"""Plot selected peaks and modal spectra from saved evaluation artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(out):
    from .study import component_data
    out=Path(out)
    full_data=component_data(out,'full')
    n_space=full_data.n_space
    preflight=json.loads((out/'preflight.json').read_text())
    bands=[entry['edges_hz'] for entry in preflight['oscillators']]
    centers=[entry['frequency_hz'] for entry in preflight['oscillators']]
    labels=dict(full='Matched TCN',remainder='Remainder TCN',hybrid='Hybrid',
                oracle_remainder='Measured remainder + oscillators',
                existing_baseline='Established TCN (one path)')
    spectra={}
    for key in labels:
        path=out/f'validation_{key}/high_resolution_spectra.npz'
        if path.exists():
            with np.load(path) as z: spectra[key]=dict(z)
    if 'oracle_remainder' in spectra:
        reference = spectra['oracle_remainder']
    elif 'hybrid' in spectra:
        reference = spectra['hybrid']
    elif 'full' in spectra:
        reference = spectra['full']
    else:
        print('No evaluated spectra found in', out)
        return []
    f=reference['frequency_hz'];truth=reference['reference_center_psd']
    result=[]
    fig,axes=plt.subplots(1,len(bands),figsize=(4.5*len(bands),4),squeeze=False,layout='constrained')
    for ax,band,center in zip(axes.flat,bands,centers):
        low,high=band
        peak=abs(f-center)<=10
        shoulders=((f>=low-25)&(f<=low-5))|((f>=high+5)&(f<=high+25))
        def contrast(p):
            return float(10*np.log10(p[peak].mean()/p[shoulders].mean()))
        entry=dict(band_hz=band,training_fitted_center_hz=center,
            definition='Mean PSD within fitted center +/-10 Hz divided by mean PSD in two 20 Hz shoulders, separated from design edges by 5 Hz',
            reference_contrast_db=contrast(truth),models={})
        window=(f>=low-35)&(f<=high+35)
        ax.semilogy(f[window],truth[window],color='black',lw=2,label=f'Held-out rank{full_data.rank}')
        for key,s in spectra.items():
            pg=s['generated_center_psd']
            entry['models'][key]=dict(contrast_db=contrast(pg),
                center_window_power_over_full_reference=float(pg[peak].sum()/truth[peak].sum()))
            if key=='remainder':
                entry['models'][key]['center_window_power_over_remainder_reference']=float(
                    pg[peak].sum()/s['reference_center_psd'][peak].sum())
            if key in ('full','hybrid','oracle_remainder'):
                ax.semilogy(f[window],pg[window],label=labels[key])
        ax.axvspan(low,high,alpha=.07,color='gray')
        ax.set(xlabel='Frequency [Hz]',ylabel='Center PSD [µm²/Hz]',title=f'Band {low:g}–{high:g} Hz')
        ax.grid(alpha=.2);result.append(entry)
    axes.flat[0].legend(fontsize=7)
    fig.savefig(out/'selected_peaks.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    f_min = max(10.0, float(np.min(bands)) - 50.0)
    f_max = min(full_data.native_dt**-1 / 2.0, float(np.max(bands)) + 50.0)
    mask=(f>=f_min)&(f<=f_max)
    for i,ax in enumerate(axes.flat):
        if i >= full_data.n_coordinates:
            ax.set_visible(False)
            continue
        scale=1/n_space if (i==0 and full_data.include_spatial_mean) else 1.
        ax.semilogy(f[mask],reference['reference_modal_psd'][mask,i]*scale,color='black',label='Reference')
        for key in ('full','hybrid','oracle_remainder'):
            if key in spectra:
                ax.semilogy(f[mask],spectra[key]['generated_modal_psd'][mask,i]*scale,label=labels[key])
        title = 'Spatial mean' if (i==0 and full_data.include_spatial_mean) else f'POD mode {i if full_data.include_spatial_mean else i+1}'
        ax.set(xlabel='Frequency [Hz]',ylabel='PSD [µm²/Hz]',title=title)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.savefig(out/'modal_comparison.png',dpi=180);plt.close(fig)
    (out/'selected_peak_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/diagnostics/frequency_hybrid_0p10_rank10_v1'))
    summarize(p.parse_args().output)
