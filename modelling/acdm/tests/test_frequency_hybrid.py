import json
import h5py
import numpy as np
import pytest
import torch

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.frequency_components import (
    prepare, decompose, design_filters, FrequencyComponentData, sha256)
from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.train import train, load_model, evaluate, _sde_transition_batches
from modelling.neural_sde.batches import indexed_velocity_batches
from modelling.frequency_hybrid.oscillators import simulate_band, fit_band
from modelling.frequency_hybrid.local_excess import (
    local_excess_modes, projected_band, projected_band_array, complex_covariance,
    simulate_complex_band)
from modelling.frequency_hybrid.local_excess_remainder import remainder_filter
from modelling.latent_oscillators.model import welch_matrix


@pytest.fixture
def component_source(shared_data, tmp_path):
    for i,path in enumerate(shared_data.source_paths.values()):
        n=1801+i*20;t=np.arange(n)*.01
        rng=np.random.default_rng(i)
        with h5py.File(path,'r+') as h:
            h.attrs['n_frames']=n
            for key,array,units in [
                ('grid/time',t,'seconds'),
                ('reduced/coefficients',np.column_stack([
                    np.sin(2*np.pi*5*t),np.cos(2*np.pi*15*t),np.sin(2*np.pi*25*t)])
                     + .1*rng.normal(size=(n,3)),'microns'),
                ('preprocessing/frame_spatial_mean',np.sin(2*np.pi*5*t)+i,'microns')]:
                del h[key];h.create_dataset(key,data=array).attrs['units']=units
    data=SharedPODTrainingData(**shared_data.source_config)
    path=prepare(data,tmp_path/'components',bands=[(3,7),(13,17),(23,27)],edge_seconds=.05)
    return data,path


def open_component(source, component):
    data,path=source
    return FrequencyComponentData(**data.source_config,
        temporal_component_manifest=path,temporal_component=component)


def test_exact_reconstruction_intervals_and_train_only_scaling(component_source):
    data,path=component_source
    hashes={p:sha256(p) for p in data.source_paths.values()}
    full=open_component(component_source,'full')
    components=[open_component(component_source,x) for x in ('remainder','band0','band1','band2')]
    for n in full.train_repetitions+full.validation_repetitions:
        t,x=full.read_coordinates(n,0,full.frame_counts[n]);edge=full.component_edge
        orig_t,orig_x=data.read_coordinates(n,edge,data.frame_counts[n]-edge)
        np.testing.assert_array_equal(t,orig_t)
        np.testing.assert_allclose(x,orig_x,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(sum(c.read_coordinates(n,0,c.frame_counts[n])[1] for c in components),x,atol=1e-12)
        np.testing.assert_array_equal(full.read_coordinates(n,10,30)[1],x[10:30])
    rem=components[0].fit_standardization()
    expected=np.concatenate([rem.read_coordinates(n,0,rem.frame_counts[n])[1] for n in rem.train_repetitions])
    np.testing.assert_allclose(rem.mean,expected.mean(0),atol=1e-12)
    np.testing.assert_allclose(rem.std,expected.std(0),rtol=1e-10)
    assert all(sha256(p)==v for p,v in hashes.items())
    with pytest.raises(ValueError,match='not prepared'):
        rem.read_coordinates(rem.test_repetitions[0],0,5)


def test_filter_config_and_invalid_guard():
    with pytest.raises(ValueError): design_filters(100,[(2,20),(10,30)])
    assert len(design_filters(100,[(2,10),(10,20)])) == 2
    with pytest.raises(ValueError): decompose(np.ones((30,2)),design_filters(100,[(2,10)]),20)


def test_checkpoint_identity_and_indexed_sampler(component_source,tmp_path):
    torch.set_num_threads(1)
    data=open_component(component_source,'remainder')
    cfg=SDEConfig(data.n_coordinates,data.native_dt,state_variable='velocity',
                  drift_type='damped_residual',residual_architecture='tcn',
                  tcn_context_frames=7,tcn_dilations=(1,2),tcn_channels=4)
    left=next(_sde_transition_batches(data,cfg,'train',64,seed=1))
    right=next(indexed_velocity_batches(data,cfg,'train',64,seed=1))
    for key in ('current_state','next_state','history_states'):
        torch.testing.assert_close(left[key],right[key])
    target=tmp_path/'train'
    train(data,target,cfg,epochs=1,batch_size=256,device='cpu',indexed_training_batches=True,tcn_rollout_cache=True)
    model,payload=load_model(target/'best.pt');restored=checkpoint_data(payload)
    assert restored.component=='remainder'
    assert restored.signature()==data.signature()
    evaluation=tmp_path/'evaluation'
    evaluate(target/'best.pt',evaluation,split='validation',num_conditions=1,
             ensemble_size=2,horizon=25,device='cpu',metrics=False)
    with np.load(evaluation/'rollout.npz') as saved:
        name=str(saved['repetition'][0]);start=int(saved['start_index'][0])
        expected_t,expected=restored.read_coordinates(name,start,start+26)
        np.testing.assert_allclose(saved['reference'][0],expected,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(saved['reference_time'][0],expected_t)
    n=data.train_repetitions[0];_,x=restored.read_coordinates(n,0,30)
    with torch.no_grad():
        generated=model.rollout(torch.tensor(x[7:8]).float(),history_states=torch.tensor(x[6::-1][None].copy()).float(),horizon=20,num_trajectories=2,seed=7)
    assert torch.isfinite(generated).all()
    entry=data.manifest['records'][n]['files']['remainder']
    p=data.manifest_path.parent/entry['path'];a=np.load(p).copy();a[0,0]+=1;np.save(p,a)
    with pytest.raises(ValueError,match='checksum'):
        checkpoint_data(payload)


def test_vector_oscillator_stationarity_and_fit():
    covariance=np.array([[2.,-.7],[-.7,1.]])
    model=dict(frequency_hz=100.,gamma_per_second=35.,fs=2000.,factor=np.linalg.cholesky(covariance))
    samples=simulate_band(model,16000,paths=12,seed=21)
    np.testing.assert_allclose(np.cov(samples.reshape(-1,2),rowvar=False),covariance,rtol=.12,atol=.1)
    f,s=welch_matrix(samples,2000,2048)
    fit=fit_band(list(samples),f,s.mean(0),2000,(70.,130.))
    assert abs(fit['frequency_hz']-100)<2
    assert abs(fit['gamma_per_second']/35-1)<.3
    assert np.linalg.eigvalsh(fit['covariance']).min()>0


def test_local_excess_projection_and_complex_oscillator():
    fs=200.;n=4096;f=np.linspace(0,100,101);d=3
    background=np.eye(d)[None]*.2
    shape=np.exp(-.5*((f-30)/3)**2)
    direction=np.array([1.,-1j,0.])/np.sqrt(2)
    spectrum=background+shape[:,None,None]*np.outer(direction,direction.conj())[None]
    result=local_excess_modes(f,spectrum,np.array([1.,0.,0.]),30.,(20.,40.),
                              valley_window_hz=2.,average_half_bins=0)
    assert result['rank']==1
    assert result['retained_positive_fraction']>.99
    t=np.arange(n)/fs
    values=np.column_stack((np.cos(2*np.pi*30*t),np.sin(2*np.pi*30*t),np.zeros(n)))
    sos=design_filters(fs,[(20,40)])[0]
    component,coefficients=projected_band(values,sos,result['modes'],100)
    assert component.shape==(n-200,d) and coefficients.shape==(n-200,1)
    assert np.linalg.norm(component)/np.linalg.norm(values[100:-100])>.9
    covariance,factor=complex_covariance([coefficients])
    model=dict(modes=result['modes'],coefficient_factor=factor,frequency_hz=30.,
               gamma_per_second=10.,fs=fs,filter_order=2)
    generated=simulate_complex_band(model,12000,paths=8,seed=4)
    assert generated.shape==(8,12000,d) and np.isfinite(generated).all()
    np.testing.assert_allclose(np.var(generated,axis=(0,1))[:2],
                               np.var(component,axis=0)[:2],rtol=.2,atol=.03)
    batched=np.stack([values,2*values]).reshape(1,2,n,d)
    projected,_=projected_band_array(batched,sos,result['modes'],100)
    np.testing.assert_allclose(projected[0,0],component,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(projected[0,1],2*component,rtol=1e-12,atol=1e-12)
    model={'modes':result['modes']}
    rem=remainder_filter(batched,[model],[sos],100)
    np.testing.assert_allclose(rem,batched[...,100:-100,:]-projected,rtol=1e-12,atol=1e-12)
