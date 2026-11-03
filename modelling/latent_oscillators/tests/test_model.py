import numpy as np
import pytest
import torch
from scipy.signal import welch, csd

from modelling.latent_oscillators.model import (
    ResonanceModel, expected_welch, oscillator_transition, simulate,
    taper_correlation, welch_matrix,
)
from modelling.latent_oscillators.study import objective, spectral_metrics, peaks_report


@pytest.mark.parametrize('n', [255,256])
def test_welch_matches_scipy_and_keeps_records_separate(n):
    x=np.random.default_rng(31).normal(size=(3,1200,2))
    # Offsets make accidental joining or missing per-segment detrending conspicuous.
    x+=np.array([100.,-200.,300.])[:,None,None]
    f,s=welch_matrix(x,1000,n)
    for r in range(3):
        fr,p=welch(x[r,:,0],fs=1000,nperseg=n,noverlap=n//2)
        _,cross=csd(x[r,:,1],x[r,:,0],fs=1000,nperseg=n,noverlap=n//2)
        np.testing.assert_allclose(f,fr)
        np.testing.assert_allclose(s[r,:,0,0].real,p,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(s[r,:,0,1],cross,rtol=1e-12,atol=1e-15)


@pytest.mark.parametrize('dt', [1e-6,1/14400,.01])
def test_exact_transition_preserves_stationary_covariance(dt):
    a,l=oscillator_transition(450,60,dt)
    np.testing.assert_allclose(a@a.T+l@l.T,np.eye(2),atol=1e-12)
    assert max(abs(np.linalg.eigvals(a)))<1


def test_expected_welch_against_explicit_covariance_quadratic_form():
    n=128;fs=2000.
    lag=np.arange(n)/fs
    rho=np.exp(-30*lag)*np.cos(2*np.pi*150*lag)
    from scipy.linalg import toeplitz
    from scipy.signal import get_window
    w=get_window('hann',n)
    actual=expected_welch(torch.tensor(rho),fs,torch.tensor(taper_correlation(n))).numpy()
    for k in (2,9,10,30,64):
        h=w*np.exp(-2j*np.pi*k*np.arange(n)/n)
        exact=(h@toeplitz(rho)@h.conj()).real/(fs*(w@w))*(1 if k==64 else 2)
        np.testing.assert_allclose(actual[k],exact,rtol=1e-10,atol=1e-14)


def test_stationary_rollout_resonance_and_cross_spectrum(tmp_path):
    torch.set_num_threads(1)
    model=ResonanceModel([[1.,450**(2/3)]],[1],2,fs=4000,nperseg=512)
    with torch.no_grad():
        model.mixing[0,:,0]=torch.tensor([1.,-.7])
        model.background.zero_()
        model.log_width.fill_(np.log(20.))
    model.save(tmp_path/'m.npz',np.zeros(2),np.ones(2))
    with np.load(tmp_path/'m.npz') as z:
        records=simulate(dict(z),8192,paths=24,seed=14)
    f,s=welch_matrix(records,4000,512)
    s=s.mean(0)
    peak=f[np.argmax(s[:,0,0].real)]
    assert abs(peak-450)<10
    assert abs(records[:,:,0].var()-1)<.08
    expected=model.spectral_matrix().detach().numpy()
    band=(f>300)&(f<600)
    assert np.mean(abs(s[band,0,0].real/expected[band,0,0]-1))<.1
    assert s[np.argmax(s[:,0,0].real),0,1].real<0
    np.testing.assert_allclose(records[:,:,1],-.7*records[:,:,0],atol=1e-12)


def test_joint_spectral_fit_recovers_displaced_frequency_and_width():
    torch.set_num_threads(1)
    truth=ResonanceModel([[1.,450**(2/3)]],[1],2,fs=4000,nperseg=512)
    model=ResonanceModel([[1.,425**(2/3)]],[1],2,fs=4000,nperseg=512)
    with torch.no_grad():
        for m in [truth,model]:
            m.mixing[0,:,0]=torch.tensor([1.,-.7])
        truth.log_width.fill_(np.log(30.))
        model.log_width.fill_(np.log(50.))
    target=truth.spectral_matrix().detach().to(torch.complex128)
    f=np.fft.rfftfreq(512,1/4000)
    mask=torch.tensor((f>200)&(f<800))
    opt=torch.optim.Adam([model.log_family,model.log_width],lr=.01)
    start=objective(model.spectral_matrix(),target,mask)[0].item()
    for _ in range(180):
        opt.zero_grad()
        loss=objective(model.spectral_matrix(),target,mask)[0]
        loss.backward();opt.step()
    assert loss.item()<start*.005
    assert abs(model.frequencies().item()-450)<1
    assert abs(model.log_width.exp().item()-30)<1


def test_no_oscillator_control_and_positive_spectral_matrix():
    model=ResonanceModel([],[],3,fs=4000,nperseg=128)
    s=model.spectral_matrix()
    assert s.shape==(65,3,3)
    assert torch.linalg.eigvalsh(s).min()>0
    total=s.sum();total.backward()
    assert torch.isfinite(model.background.grad).all()


def test_reject_invalid_frequency_and_data():
    with pytest.raises(ValueError): oscillator_transition(-1,2,.001)
    with pytest.raises(ValueError): welch_matrix(np.zeros((1,3,2)),100,128)


def test_coordinate_subset_metrics_do_not_include_other_coordinates():
    f=np.arange(0,2001,10.)
    reference=np.tile(np.eye(3),(len(f),1,1)).astype(complex)
    prediction=reference.copy()
    prediction[:,2,0]=prediction[:,0,2]=.9
    prediction[:,2,2]=100
    metrics=spectral_metrics(f,prediction,reference,[0,1])
    assert metrics['log_psd_rmse_db']==0
    assert metrics['normalized_cross_rmse']==0
    assert spectral_metrics(f,prediction,reference)['normalized_cross_rmse']>0


def test_peak_matching_is_one_to_one():
    f=np.arange(0,1201,dtype=float)
    def make(peaks):
        p=np.ones(len(f))
        for center in peaks:
            p+=4*np.exp(-.5*((f-center)/2)**2)
        return p[:,None,None]
    result=peaks_report(f,make([221]),make([200,242]),make([200,242]))[0]
    assert result['targets']==2
    assert result['hits']==1
    assert result['recall']==.5


def test_common_rollout_export_restores_coefficient_units_and_time():
    from modelling.latent_oscillators.export_rollout import pack_arrays
    generated=np.arange(4*20*3).reshape(4,20,3)/10
    reference=np.ones((2,20,3))
    arrays=pack_arrays(generated,reference,40000,14400.,['rep13','rep14'])
    assert arrays['trajectories'].shape==(2,2,20,3)
    np.testing.assert_allclose(arrays['trajectories'].reshape(4,20,3)/200,generated)
    np.testing.assert_allclose(arrays['reference']/200,reference)
    np.testing.assert_allclose(np.diff(arrays['reference_time']),1/14400)
    with pytest.raises(ValueError):
        pack_arrays(generated[:3],reference,40000,14400.,['rep13','rep14'])
