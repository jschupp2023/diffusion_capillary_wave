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
    n_space=component_data(out,'full').n_space
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
    reference=spectra['oracle_remainder']
    f=reference['frequency_hz'];truth=reference['reference_center_psd']
    result=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,band,center in zip(axes,bands,centers):
        low,high=band
        peak=abs(f-center)<=10
        shoulders=((f>=low-25)&(f<=low-5))|((f>=high+5)&(f<=high+25))
        def contrast(p):
            return float(10*np.log10(p[peak].mean()/p[shoulders].mean()))
        entry=dict(band_hz=band,training_fitted_center_hz=center,
            definition='Mean PSD within fitted center +/-10 Hz divided by mean PSD in two 20 Hz shoulders, separated from design edges by 5 Hz',
            reference_contrast_db=contrast(truth),models={})
        window=(f>=low-35)&(f<=high+35)
        ax.semilogy(f[window],truth[window],color='black',lw=2,label='Held-out rank10 + mean')
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
    axes[0].legend(fontsize=7)
    fig.savefig(out/'selected_peaks.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    mask=(f>=100)&(f<=1500)
    for i,ax in enumerate(axes.flat):
        scale=1/n_space if i==0 else 1.
        ax.semilogy(f[mask],reference['reference_modal_psd'][mask,i]*scale,color='black',label='Reference')
        for key in ('full','hybrid','oracle_remainder'):
            if key in spectra:
                ax.semilogy(f[mask],spectra[key]['generated_modal_psd'][mask,i]*scale,label=labels[key])
        ax.set(xlabel='Frequency [Hz]',ylabel='PSD [µm²/Hz]',title='Spatial mean' if i==0 else f'POD mode {i}')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.savefig(out/'modal_comparison.png',dpi=180);plt.close(fig)
    (out/'selected_peak_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/diagnostics/frequency_hybrid_0p10_rank10_v1'))
    summarize(p.parse_args().output)
