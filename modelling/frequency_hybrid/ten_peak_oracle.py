"""Training-fitted contiguous ten-peak oscillator-bank oracle experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from data_analysis.correlations.rollout_state_decorrelation import mean_path_acf
from modelling.data_preparation.frequency_components import settling_samples
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from .evaluate import describe
from .oscillators import fit_band, oscillator_psd, simulate_band
from .study import center_weights, matrices, projected, spectral_comparison, write_json


# Raw-center family documented by the training-only harmonic audit. Exact values
# use the native-equivalent Welch grid where available.
CENTERS_HZ = np.asarray([
    126.5625, 210.9375, 281.25, 365.625, 450.0,
    548.4375, 646.875, 759.375, 871.875, 984.375,
])


def midpoint_bands(centers):
    centers=np.asarray(centers,float)
    middle=(centers[:-1]+centers[1:])/2
    lower=centers[0]-(centers[1]-centers[0])/2
    upper=centers[-1]+(centers[-1]-centers[-2])/2
    return np.column_stack((np.r_[lower,middle],np.r_[middle,upper]))


def main():
    out=Path('runs/diagnostics/frequency_hybrid_0p10_ten_peak_oracle_v1')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    data=SharedPODTrainingData('0p10',10,validation_reps=[13,14],test_reps=[15,16],
        spatial_mean_highpass_hz=50.,output_dtype='float64')
    fs=1/data.native_dt;bands=midpoint_bands(CENTERS_HZ)
    filters=[butter(4,band,btype='bandpass',fs=fs,output='sos') for band in bands]
    edge=max(int(np.ceil(.1*fs)),*(settling_samples(s,fs) for s in filters))
    names=data.train_repetitions+data.validation_repetitions
    raw={name:data.read_coordinates(name,0,data.frame_counts[name]) for name in names}
    retained={name:(t[edge:-edge],x[edge:-edge]) for name,(t,x) in raw.items()}
    length=min(50001,*(len(retained[n][0]) for n in data.validation_repetitions))
    weights=center_weights(data)
    measured_sum={n:np.zeros((length,data.n_coordinates)) for n in data.validation_repetitions}
    generated_sum=np.zeros((2,4,length,data.n_coordinates))
    fits=[];band_reports=[]
    for i,(band,sos) in enumerate(zip(bands,filters)):
        training=[sosfiltfilt(sos,raw[n][1],axis=0,padlen=edge)[edge:-edge]
                  for n in data.train_repetitions]
        f,s=matrices(training,fs,32768)
        fit=fit_band(training,f,s,fs,band)
        np.savez(out/f'oscillator_{i:02d}.npz',**fit)
        validation=[]
        for n in data.validation_repetitions:
            values=sosfiltfilt(sos,raw[n][1],axis=0,padlen=edge)[edge:-edge][:length]
            validation.append(values);measured_sum[n]+=values
        generated=simulate_band(fit,length,paths=8,seed=201+i).reshape(2,4,length,data.n_coordinates)
        generated_sum+=generated
        fg,sg=matrices(list(generated.reshape(-1,length,data.n_coordinates)),fs,32768)
        _,sr=matrices(validation,fs,32768)
        comparison=spectral_comparison(fg,sg,sr,weights,bands.tolist(),evaluation_band=band.tolist())
        density=oscillator_psd(fit['frequency_hz'],fit['gamma_per_second'],fs,32768)
        inside=(fg>=band[0])&(fg<=band[1])
        item=dict(index=i,nominal_center_hz=float(CENTERS_HZ[i]),edges_hz=band.tolist(),
            fitted_frequency_hz=float(fit['frequency_hz']),effective_fwhm_hz=float(fit['approximate_fwhm_hz']),
            spectral_fit_relative_rmse=float(fit['spectral_fit_relative_rmse']),
            expected_outside_band_power_fraction=float(1-density[inside].sum()/density.sum()),
            heldout=comparison)
        fits.append(item);band_reports.append((fg,projected(sg,weights),projected(sr,weights)))
        print(f"{i+1}/10: {fit['frequency_hz']:.2f} Hz, width {fit['approximate_fwhm_hz']:.2f} Hz",flush=True)
    full=np.stack([retained[n][1][:length] for n in data.validation_repetitions])
    times=np.stack([retained[n][0][:length] for n in data.validation_repetitions])
    measured_bands=np.stack([measured_sum[n] for n in data.validation_repetitions])
    remainder=full-measured_bands
    oracle=remainder[:,None]+generated_sum
    meta=dict(repetitions=np.asarray(data.validation_repetitions),
        start_indices=np.full(2,edge),filterbank_bands_hz=bands.tolist(),
        filterbank_centers_hz=CENTERS_HZ.tolist(),filter_order=4,filter_edge_samples=edge,
        initialization='Measured held-out remainder plus independent stationary oscillator draws',
        pairing='Oracle only: measured remainder is retained; this is not a forecast')
    oracle_report=describe(out/'validation_oracle',oracle,full,times,data,
        'Measured remainder + ten oscillators',bands.tolist(),rollout_metadata=meta)
    fb,sbank=matrices(list(generated_sum.reshape(-1,length,data.n_coordinates)),fs,32768)
    _,smeasured=matrices(list(measured_bands),fs,32768)
    _,srem=matrices(list(remainder),fs,32768)
    _,sfull=matrices(list(full),fs,32768)
    bank_report=spectral_comparison(fb,sbank,smeasured,weights,bands.tolist(),
                                    evaluation_band=(float(bands[0,0]),float(bands[-1,1])))
    pfull=projected(sfull,weights);prem=projected(srem,weights)
    remainder_fractions={}
    for label,(lo,hi) in {'oscillator_region':(bands[0,0],bands[-1,1]),
                          'above_1500_hz':(1500.,fs/2)}.items():
        mask=(fb>=lo)&(fb<=hi)
        remainder_fractions[label]=float(prem[mask].sum()/pfull[mask].sum())
    measured_acf=mean_path_acf(measured_bands,4000)[:,0]
    generated_acf=mean_path_acf(generated_sum.reshape(-1,length,data.n_coordinates),4000)[:,0]
    bank_report['spatial_mean_acf_rmse_first_4000_lags']=float(np.sqrt(np.mean((measured_acf-generated_acf)**2)))
    report=dict(source_peak_family='training-only raw-center family from harmonic_dynamics_audit',
        centers_hz=CENTERS_HZ.tolist(),bands_hz=bands.tolist(),fs=fs,filter_order=4,
        edge_samples=edge,edge_seconds=edge/fs,retained_frames_per_record=len(next(iter(retained.values()))[0]),
        validation_length=length,oscillators=fits,bank_vs_measured_bands=bank_report,
        oracle=oracle_report,remainder_center_power_fraction=remainder_fractions,
        limitations=['Adjacent Butterworth bands meet at midpoint cutoffs but are not an orthogonal filter bank.',
                     'Fitted widths describe filtered targets and are not physical damping estimates.',
                     'Measured-remainder oracle tests oscillator replacement only; it does not test neural-SDE learning.'])
    write_json(out/'summary.json',report)
    np.savez_compressed(out/'spectra.npz',frequency_hz=fb,full_center=projected(sfull,weights),
        remainder_center=projected(srem,weights),measured_band_center=projected(smeasured,weights),
        generated_bank_center=projected(sbank,weights),measured_band_acf=measured_acf,
        generated_bank_acf=generated_acf)
    fig,axes=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    mask=(fb>=50)&(fb<=1600)
    axes[0].semilogy(fb[mask],projected(sfull,weights)[mask],color='black',label='Held-out full signal')
    axes[0].semilogy(fb[mask],projected(srem,weights)[mask],label='Measured ten-band remainder')
    axes[0].semilogy(fb[mask],projected(smeasured,weights)[mask],label='Measured removed bands')
    axes[1].semilogy(fb[mask],projected(smeasured,weights)[mask],color='black',label='Measured removed bands')
    axes[1].semilogy(fb[mask],projected(sbank,weights)[mask],label='Ten generated oscillators')
    with np.load(out/'validation_oracle/high_resolution_spectra.npz') as saved:
        fo=saved['frequency_hz'];mo=(fo>=50)&(fo<=1600)
        axes[2].semilogy(fo[mo],saved['reference_center_psd'][mo],color='black',label='Held-out full signal')
        axes[2].semilogy(fo[mo],saved['generated_center_psd'][mo],label='Measured remainder + oscillator bank')
    for ax in axes:
        for boundary in np.r_[bands[:,0],bands[-1,1]]:ax.axvline(boundary,color='gray',lw=.4,alpha=.3)
        ax.set(xlabel='Frequency [Hz]',ylabel='Center PSD [µm²/Hz]');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'oracle_psd.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    lag=np.arange(4001)/fs*1000
    ax.plot(lag,measured_acf,color='black',label='Measured removed-band spatial mean')
    ax.plot(lag,generated_acf,label='Generated oscillator-bank spatial mean')
    ax.set(xlabel='Lag [ms]',ylabel='ACF');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'bank_acf.png',dpi=180);plt.close(fig)
    print(json.dumps({'bank':bank_report,'oracle':{k:oracle_report[k] for k in
        ('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags')},
        'remainder_fractions':remainder_fractions},indent=2),flush=True)


if __name__=='__main__':main()
