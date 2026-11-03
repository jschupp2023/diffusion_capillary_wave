"""Local spectral-excess decomposition with low-rank complex band modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, hilbert, sosfiltfilt, sosfreqz

from modelling.data_preparation.frequency_components import settling_samples
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from .evaluate import describe
from .oscillators import oscillator_psd
from .study import center_weights, matrices, projected, spectral_comparison, write_json
from .ten_peak_oracle import CENTERS_HZ, midpoint_bands


def _average_matrix(spectrum, index, half_bins):
    lo=max(0,index-half_bins);hi=min(len(spectrum),index+half_bins+1)
    return spectrum[lo:hi].mean(axis=0)


def local_excess_modes(frequency, spectrum, weights, center, edges, *,
                       energy_fraction=.95, valley_window_hz=14., average_half_bins=1):
    """Positive eigenspace of a peak matrix minus local sideband interpolation."""
    frequency=np.asarray(frequency);spectrum=np.asarray(spectrum);weights=np.asarray(weights)
    if not 0 < energy_fraction <= 1 or edges[0] >= center or center >= edges[1]:
        raise ValueError('Invalid energy fraction or peak interval')
    power=projected(spectrum,weights)
    inside=np.flatnonzero((frequency>=edges[0])&(frequency<=edges[1]))
    if len(inside)<5:
        raise ValueError('Peak interval contains too few spectral bins')
    peak=int(inside[np.argmax(power[inside])])
    valleys=[]
    for boundary in edges:
        candidates=np.flatnonzero(abs(frequency-boundary)<=valley_window_hz)
        if not len(candidates):
            raise ValueError('No spectral bins in valley search window')
        valleys.append(int(candidates[np.argmin(power[candidates])]))
    left,right=valleys
    if not left < peak < right:
        raise ValueError('Selected valleys do not bracket the peak')
    peak_matrix=_average_matrix(spectrum,peak,average_half_bins)
    left_matrix=_average_matrix(spectrum,left,average_half_bins)
    right_matrix=_average_matrix(spectrum,right,average_half_bins)
    alpha=(frequency[peak]-frequency[left])/(frequency[right]-frequency[left])
    background=(1-alpha)*left_matrix+alpha*right_matrix
    excess=peak_matrix-background
    excess=(excess+excess.conj().T)/2
    eigenvalues,eigenvectors=np.linalg.eigh(excess)
    order=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[order];eigenvectors=eigenvectors[:,order]
    positive=np.maximum(eigenvalues,0.)
    if positive.sum()<=0:
        raise ValueError('Local excess has no positive eigenvalues')
    rank=int(np.searchsorted(np.cumsum(positive),energy_fraction*positive.sum())+1)
    modes=eigenvectors[:,:rank]
    retained=positive[:rank]
    participation=(abs(modes)**2*retained[None]).sum(axis=1)/retained.sum()
    return dict(modes=modes,eigenvalues=eigenvalues,rank=rank,
        participation=participation,peak_index=peak,left_index=left,right_index=right,
        peak_frequency_hz=float(frequency[peak]),
        valley_frequencies_hz=np.asarray([frequency[left],frequency[right]]),
        positive_mass_fraction=float(positive.sum()/np.abs(eigenvalues).sum()),
        retained_positive_fraction=float(retained.sum()/positive.sum()),
        background=background,peak_matrix=peak_matrix)


def projected_band_array(values, sos, modes, edge):
    """Project arrays with shape ``[..., time, coordinate]`` onto one band subspace."""
    values=np.asarray(values,dtype=np.float64);modes=np.asarray(modes)
    if values.ndim < 2 or values.shape[-1] != len(modes) or edge < 1:
        raise ValueError('Expected [..., time, coordinate] data and a positive edge')
    if values.shape[-2] <= 2*edge:
        raise ValueError('Filter guard removes the complete trajectory')
    filtered=sosfiltfilt(sos,values,axis=-2,padlen=edge)
    analytic=hilbert(filtered,axis=-2)
    coefficients=analytic@modes.conj()
    component=(coefficients@modes.T).real
    return component[...,edge:-edge,:],coefficients[...,edge:-edge,:]


def projected_band(values, sos, modes, edge):
    """Bandpass one trajectory and retain a complex spatial subspace."""
    values=np.asarray(values,dtype=np.float64)
    if values.ndim != 2:
        raise ValueError('Expected [time, coordinate] data')
    return projected_band_array(values,sos,modes,edge)


def complex_covariance(records, shrinkage=.01):
    covariance=0
    for values in records:
        centered=values-values.mean(axis=0,keepdims=True)
        covariance=covariance+centered.conj().T@centered/len(centered)
    covariance=covariance/len(records)
    covariance=(covariance+covariance.conj().T)/2
    covariance=(1-shrinkage)*covariance+shrinkage*np.diag(np.diag(covariance).real)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    factor=eigenvectors*np.sqrt(np.maximum(eigenvalues,0.))
    return covariance,factor


def fit_filtered_band(records, frequency, spectral_matrix, fs, edges, sos):
    """Fit the latent oscillator through the fixed zero-phase output filter."""
    covariance=np.mean([np.cov(values,rowvar=False,bias=True) for values in records],axis=0)
    diagonal=np.diagonal(spectral_matrix,axis1=-2,axis2=-1).real
    target=np.mean(diagonal/np.maximum(np.diag(covariance),1e-30),axis=-1)
    _,response=sosfreqz(sos,worN=2*np.pi*np.asarray(frequency)/fs)
    response=abs(response)**4  # single-pass H, then forward/backward filtering and PSD
    low,high=edges;width=high-low;nperseg=2*(len(frequency)-1)
    mask=(frequency>=max(1.,low-width))&(frequency<=high+width)
    scale=max(float(target[mask].max()),1e-30)
    peak=float(frequency[np.flatnonzero(mask)[np.argmax(target[mask])]])
    def density(params):
        raw=oscillator_psd(params[0],np.exp(params[1]),fs,nperseg)
        passed=float(np.sum(raw*response)/np.sum(raw))
        return raw*response/max(passed,1e-30),passed
    def residual(params):
        return (density(params)[0][mask]-target[mask])/scale
    result=least_squares(residual,[np.clip(peak,low+.1,high-.1),np.log(np.pi*width/3)],
        bounds=([low,np.log(.5)],[high,np.log(min(np.pi*width,2*np.pi*low*.9))]))
    if not result.success:
        raise RuntimeError(result.message)
    fitted_frequency=float(result.x[0]);gamma=float(np.exp(result.x[1]))
    _,passed=density(result.x)
    return dict(frequency_hz=fitted_frequency,gamma_per_second=gamma,
        approximate_fwhm_hz=gamma/np.pi,filter_passed_variance_fraction=passed,
        spectral_fit_relative_rmse=float(np.sqrt(np.mean(result.fun**2))))


def simulate_complex_band(model, steps, paths=8, seed=0):
    """Stationary complex OU envelope mapped through complex spatial modes."""
    rng=np.random.default_rng(seed)
    modes=np.asarray(model['modes']);factor=np.asarray(model['coefficient_factor'])
    dt=1/float(model['fs']);gamma=float(model['gamma_per_second'])
    omega=2*np.pi*float(model['frequency_hz'])
    alpha=np.exp((-gamma+1j*omega)*dt)
    innovation=np.sqrt(max(0.,1-abs(alpha)**2))
    def normal(shape):
        return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
    state=normal((paths,len(factor)))@factor.T
    output=np.empty((paths,steps,len(modes)),dtype=np.float64)
    for index in range(steps):
        output[:,index]=(state@modes.T).real
        state=alpha*state+innovation*(normal(state.shape)@factor.T)
    return output


def run(output, energy_fraction=.95, *, low_peak_center_hz=None,
        low_peak_lower_hz=50., low_peak_order=2):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    data=SharedPODTrainingData('0p10',10,validation_reps=[13,14],test_reps=[15,16],
        spatial_mean_highpass_hz=50.,output_dtype='float64')
    fs=1/data.native_dt;centers=CENTERS_HZ.copy();bands=midpoint_bands(centers)
    orders=np.full(len(centers),4,dtype=int)
    if low_peak_center_hz is not None:
        if not low_peak_lower_hz < low_peak_center_hz < bands[0,0]:
            raise ValueError('Low peak must lie between its lower edge and the first existing band')
        centers=np.r_[float(low_peak_center_hz),centers]
        bands=np.vstack(([float(low_peak_lower_hz),bands[0,0]],bands))
        orders=np.r_[int(low_peak_order),orders]
    filters=[butter(int(order),band,btype='bandpass',fs=fs,output='sos')
             for order,band in zip(orders,bands)]
    edge=max(int(np.ceil(.1*fs)),*(settling_samples(s,fs) for s in filters))
    names=data.train_repetitions+data.validation_repetitions
    raw={name:data.read_coordinates(name,0,data.frame_counts[name]) for name in names}
    train=[raw[name][1] for name in data.train_repetitions]
    frequency,train_spectrum=matrices(train,fs,32768);weights=center_weights(data)
    retained={name:(time[edge:-edge],values[edge:-edge]) for name,(time,values) in raw.items()}
    length=min(50001,*(len(retained[name][0]) for name in data.validation_repetitions))
    measured_sum={name:np.zeros((length,data.n_coordinates)) for name in data.validation_repetitions}
    generated_sum=np.zeros((2,4,length,data.n_coordinates))
    band_rows=[]
    for index,(center,band,order,sos) in enumerate(zip(centers,bands,orders,filters)):
        excess=local_excess_modes(frequency,train_spectrum,weights,center,band,
                                  energy_fraction=energy_fraction)
        train_components=[];train_coefficients=[]
        for name in data.train_repetitions:
            component,coefficients=projected_band(raw[name][1],sos,excess['modes'],edge)
            train_components.append(component);train_coefficients.append(coefficients)
        band_frequency,band_spectrum=matrices(train_components,fs,32768)
        temporal=fit_filtered_band(train_components,band_frequency,band_spectrum,fs,band,sos)
        coefficient_covariance,coefficient_factor=complex_covariance(train_coefficients)
        # Coefficients were measured after filtering; compensate latent variance
        # so applying the observation filter restores their fitted covariance.
        coefficient_factor=coefficient_factor/np.sqrt(temporal['filter_passed_variance_fraction'])
        model=dict(modes=excess['modes'],coefficient_covariance=coefficient_covariance,
            coefficient_factor=coefficient_factor,frequency_hz=temporal['frequency_hz'],
            gamma_per_second=temporal['gamma_per_second'],fs=fs,edges_hz=band,
            filter_order=int(order),
            filter_passed_variance_fraction=temporal['filter_passed_variance_fraction'])
        np.savez(output/f'band_{index:02d}.npz',**model)
        validation=[]
        for name in data.validation_repetitions:
            component,_=projected_band(raw[name][1],sos,excess['modes'],edge)
            component=component[:length];validation.append(component);measured_sum[name]+=component
        # Match the measured component's observation filter. Otherwise the
        # oscillator's Lorentzian tails leak outside its assigned band.
        generated=simulate_complex_band(model,length+2*edge,paths=8,seed=401+index)
        generated=sosfiltfilt(sos,generated,axis=1,padlen=edge)[:,edge:-edge]
        generated=generated.reshape(2,4,length,data.n_coordinates)
        generated_sum+=generated
        fg,sg=matrices(list(generated.reshape(-1,length,data.n_coordinates)),fs,32768)
        _,sr=matrices(validation,fs,32768)
        comparison=spectral_comparison(fg,sg,sr,weights,bands.tolist(),evaluation_band=band.tolist())
        band_rows.append(dict(index=index,nominal_center_hz=float(center),edges_hz=band.tolist(),
            selected_peak_hz=excess['peak_frequency_hz'],
            valley_frequencies_hz=excess['valley_frequencies_hz'].tolist(),rank=excess['rank'],
            positive_mass_fraction=excess['positive_mass_fraction'],
            retained_positive_fraction=excess['retained_positive_fraction'],
            coordinate_participation=excess['participation'].tolist(),
            fitted_frequency_hz=float(temporal['frequency_hz']),
            fitted_fwhm_hz=float(temporal['approximate_fwhm_hz']),heldout=comparison))
        print(f"{index+1}/{len(centers)}: peak {excess['peak_frequency_hz']:.1f} Hz, rank {excess['rank']}, "
              f"positive mass {excess['positive_mass_fraction']:.2f}",flush=True)
    full=np.stack([retained[name][1][:length] for name in data.validation_repetitions])
    times=np.stack([retained[name][0][:length] for name in data.validation_repetitions])
    measured_bands=np.stack([measured_sum[name] for name in data.validation_repetitions])
    remainder=full-measured_bands
    reconstruction_error=float(np.linalg.norm(remainder+measured_bands-full)/np.linalg.norm(full))
    oracle=remainder[:,None]+generated_sum
    meta=dict(repetitions=np.asarray(data.validation_repetitions),start_indices=np.full(2,edge),
        filterbank_bands_hz=bands.tolist(),filterbank_centers_hz=centers.tolist(),
        filter_orders=orders.tolist(),filter_edge_samples=edge,local_excess_energy_fraction=energy_fraction,
        initialization='Measured held-out projected remainder plus independent complex oscillators',
        pairing='Oracle only: measured remainder is retained; this is not a forecast')
    oracle_report=describe(output/'validation_oracle',oracle,full,times,data,
        'Local-excess remainder + complex oscillators',bands.tolist(),rollout_metadata=meta)
    f,sfull=matrices(list(full),fs,32768);_,srem=matrices(list(remainder),fs,32768)
    _,smeasured=matrices(list(measured_bands),fs,32768)
    _,sgenerated=matrices(list(generated_sum.reshape(-1,length,data.n_coordinates)),fs,32768)
    pfull=projected(sfull,weights);prem=projected(srem,weights)
    region=(f>=bands[0,0])&(f<=bands[-1,1])
    remainder_fraction=float(prem[region].sum()/pfull[region].sum())
    bank_report=spectral_comparison(f,sgenerated,smeasured,weights,bands.tolist(),
                                    evaluation_band=(float(bands[0,0]),float(bands[-1,1])))
    current_path=Path('runs/diagnostics/frequency_hybrid_0p10_ten_peak_oracle_v1/summary.json')
    current=json.loads(current_path.read_text()) if current_path.exists() else None
    report=dict(method='Local sideband matrix interpolation, positive excess eigenspace, band-limited complex OU bands',
        training_only_identification=True,energy_fraction=energy_fraction,total_complex_rank=sum(x['rank'] for x in band_rows),
        reconstruction_relative_error=reconstruction_error,bands=band_rows,
        remainder_center_power_fraction_oscillator_region=remainder_fraction,
        oscillator_bank_vs_measured_projected_bands=bank_report,oracle=oracle_report,
        current_contiguous_reference=(None if current is None else dict(
            total_scalar_band_coordinates=110,
            remainder_center_power_fraction_oscillator_region=current['remainder_center_power_fraction']['oscillator_region'],
            oracle={key:current['oracle'][key] for key in ('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags')})),
        limitations=['Local sideband subtraction is a diagnostic estimate, not a unique physical background.',
                     'All retained directions within a band share one fitted frequency and damping rate.',
                     'Generated oscillators use the same offline zero-phase band filters as measured components.',
                     'The oracle retains the measured remainder and does not test neural-SDE learning.'])
    write_json(output/'summary.json',report)
    np.savez_compressed(output/'spectra.npz',frequency_hz=f,full_center=pfull,
        remainder_center=prem,measured_band_center=projected(smeasured,weights),
        generated_bank_center=projected(sgenerated,weights))
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    mask=(f>=50)&(f<=1600)
    axes[0].semilogy(f[mask],pfull[mask],color='black',label='Held-out full signal')
    axes[0].semilogy(f[mask],prem[mask],label='Local-excess projected remainder')
    old=Path('runs/diagnostics/frequency_hybrid_0p10_ten_peak_oracle_v1/spectra.npz')
    if old.exists():
        with np.load(old) as saved:
            axes[0].semilogy(saved['frequency_hz'][mask],saved['remainder_center'][mask],
                            label='Previous contiguous-band remainder',alpha=.75)
    axes[1].semilogy(f[mask],projected(smeasured,weights)[mask],color='black',label='Measured projected bands')
    axes[1].semilogy(f[mask],projected(sgenerated,weights)[mask],label='Complex oscillator bank')
    for ax in axes:
        ax.set(xlabel='Frequency [Hz]',ylabel='Center PSD [µm²/Hz]');ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'decomposition.png',dpi=180);plt.close(fig)
    ranks=np.asarray([x['rank'] for x in band_rows]);participation=np.asarray([x['coordinate_participation'] for x in band_rows])
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    axes[0].bar(centers,ranks,width=35);axes[0].set(xlabel='Peak [Hz]',ylabel=f'{energy_fraction:g} excess rank')
    image=axes[1].imshow(participation.T,aspect='auto',origin='lower',cmap='magma',
        extent=[0,len(centers),-.5,data.n_coordinates-.5])
    axes[1].set(xticks=np.arange(len(centers))+.5,xticklabels=[f'{x:g}' for x in centers],
                yticks=np.arange(data.n_coordinates),yticklabels=['mean']+[f'POD{i}' for i in range(1,11)],
                xlabel='Nominal peak [Hz]',ylabel='Coordinate')
    fig.colorbar(image,ax=axes[1],label='Excess-mode participation')
    fig.savefig(output/'ranks_and_participation.png',dpi=180);plt.close(fig)
    print(json.dumps({'total_complex_rank':report['total_complex_rank'],
        'reconstruction_relative_error':reconstruction_error,
        'remainder_center_power_fraction_oscillator_region':remainder_fraction,
        'bank':{k:bank_report[k] for k in ('center_psd_mae_db','center_band_power_ratio','coherence_complex_rmse')},
        'oracle':{k:oracle_report[k] for k in ('center_psd_mae_db','center_band_power_ratio','acf_rmse_first_4000_lags')}},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--energy-fraction',type=float,default=.95)
    parser.add_argument('--low-peak-center-hz',type=float)
    parser.add_argument('--low-peak-lower-hz',type=float,default=50.)
    parser.add_argument('--low-peak-order',type=int,default=2)
    args=parser.parse_args(argv)
    run(args.output,args.energy_fraction,low_peak_center_hz=args.low_peak_center_hz,
        low_peak_lower_hz=args.low_peak_lower_hz,low_peak_order=args.low_peak_order)


if __name__=='__main__':
    main()
