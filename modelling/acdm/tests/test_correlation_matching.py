from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from modelling.neural_sde.losses import StationaryMeanLoss
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.correlation import fit_correlation_reference
from modelling.neural_sde.correlation_gradient_diagnostic import noise_components
from modelling.neural_sde.train import train, load_model


def test_explicit_selection_matches_legacy_and_fixed_scaling():
    torch.manual_seed(2)
    model=NeuralSDE(SDEConfig(3,.01,state_variable='velocity',history_steps=1,hidden_layers=(4,)))
    current=torch.randn(12,3);history=torch.randn(12,1,3);endpoints=torch.randn(12,2,3)
    def stats(loss):
        return loss.per_horizon_statistics_with_lagged_covariance(model,current,endpoints,
            history_states=history,generator=torch.Generator().manual_seed(3))
    legacy=stats(StationaryMeanLoss((1,3),pod_coordinate_offset=1,lagged_covariance_mode_count=2))
    explicit=stats(StationaryMeanLoss((1,3),lagged_covariance_coordinates=(1,2)))
    for h in legacy:
        torch.testing.assert_close(legacy[h][4],explicit[h][4])
    var=current.var(0,correction=1)
    frozen=stats(StationaryMeanLoss((1,3),lagged_covariance_coordinates=(0,),lagged_covariance_reference_variance=var))
    batch=stats(StationaryMeanLoss((1,3),lagged_covariance_coordinates=(0,)))
    scaled=stats(StationaryMeanLoss((1,3),lagged_covariance_coordinates=(0,),lagged_covariance_reference_variance=2*var))
    for h in frozen:
        torch.testing.assert_close(frozen[h][4],batch[h][4])
        torch.testing.assert_close(scaled[h][5],frozen[h][5]/2,atol=1e-5,rtol=1e-5)
    with pytest.raises(ValueError,match='OR'):
        StationaryMeanLoss(lagged_covariance_coordinates=(0,),lagged_covariance_mode_count=1)
    with pytest.raises(ValueError,match='positive'):
        StationaryMeanLoss(lagged_covariance_reference_variance=torch.tensor([0.]))


def test_reference_is_train_only_and_gap_safe():
    values=np.c_[np.arange(30,dtype=float),np.arange(30,dtype=float)**2]
    values[12:,0]+=100
    data=SimpleNamespace(n_coordinates=2,splits={'train':['rep1'],'validation':['never']},
        frame_counts={'rep1':30},_coordinate_cache={'rep1':(None,values)})
    model=NeuralSDE(SDEConfig(2,.01,state_variable='velocity',history_steps=2))
    variance,report=fit_correlation_reference(data,model,(1,3),2.,[1.,100.])
    ix=np.array([i for i in range(2,27) if not (i-2<12<=i+3)])
    np.testing.assert_allclose(variance.numpy(),values[ix].var(0,ddof=1),rtol=1e-6)
    assert report['total_starts']==len(ix) and set(report['counts'])=={'rep1'}


class OscillatorOrAR:
    def __init__(self, ar=False):
        self.config=SimpleNamespace(state_variable='velocity',reduced_dim=1,history_steps=1,
            lag_steps=1,velocity_oscillator_periods=None,velocity_innovation_offsets=None)
        self.state_mean=torch.zeros(1);self.state_std=torch.ones(1)
        self.theta=torch.tensor(.2,requires_grad=True);self.ar=ar
    def _frame_velocity(self,state,history): return state-history[...,0,:]
    def velocity_base_step_distribution(self,state,history,velocity,**kwargs):
        if self.ar:
            # Stationary AR has the same unit variance but no narrow spectral peak.
            r=torch.exp(-self.theta)
            return (r-1)*state,torch.sqrt(1-r*r).reshape(1,1)
        next_state=2*torch.cos(self.theta)*state-history[...,0,:]
        return next_state-state,torch.zeros(1,1)


def test_loss_detects_missing_oscillation_despite_equal_variance():
    phase=torch.arange(8192)*2*torch.pi/8192
    current=(2**.5*torch.cos(phase))[:,None]
    history=(2**.5*torch.cos(phase-.2))[:,None,None]
    horizons=(4,8,16,24)
    endpoints=torch.stack([2**.5*torch.cos(phase+.2*h) for h in horizons],1)[...,None]
    objective=StationaryMeanLoss(horizons,lagged_covariance_coordinates=(0,),
        lagged_covariance_reference_variance=torch.ones(1))
    values=[]
    for ar in (False,True):
        model=OscillatorOrAR(ar)
        stats=objective.per_horizon_statistics_with_lagged_covariance(model,current,endpoints,
            history_states=history,generator=torch.Generator().manual_seed(11))
        loss=torch.stack([v[4] for v in stats.values()]).mean()
        values.append(float(loss.detach()))
        if ar:
            grad=torch.autograd.grad(loss,model.theta)[0]
            assert torch.isfinite(grad) and abs(grad)>.01
    assert values[0]<1e-8 and values[1]>.2


def test_noise_separates_starts_and_particles():
    x=np.array([[[0.,0.],[2.,0.]],[[4.,0.],[6.,0.]]])
    r=noise_components(x)
    assert r['mean_norm']==3 and r['within_batch_noise_rms']==1 and r['between_batch_rms']==2


def test_train_saves_fixed_reference_and_explicit_coordinates(shared_data,tmp_path):
    config=SDEConfig(shared_data.n_coordinates,shared_data.native_dt,state_variable='velocity',
        history_steps=1,hidden_layers=(4,))
    output=tmp_path/'fit'
    train(shared_data,output,config,epochs=1,batch_size=8,stationary_lagged_covariance_weight=.01,
        stationary_lagged_covariance_coordinates=(0,1),stationary_lagged_covariance_normalization='training',
        stationary_mean_horizons=(1,),stationary_lagged_covariance_horizons=(2,3),
        stationary_mean_weight=.001,stationary_mean_trajectories=2,stationary_mean_batch_fraction=1.)
    _,checkpoint=load_model(output/'best.pt')
    tr=checkpoint['training']
    assert tr['stationary_lagged_covariance_coordinates']==[0,1]
    assert tr['stationary_lagged_covariance_normalization']=='training'
    assert tr['stationary_lagged_covariance_reference']['split']=='train'
    assert tr['stationary_lagged_covariance_reference']['maximum_lag_steps']==3
    assert tr['stationary_mean_horizons']==[1]
    assert tr['stationary_lagged_covariance_horizons']==[2,3]
    row=json.loads((output/'metrics.jsonl').read_text())
    for split in ('train','validation'):
        assert row[f'{split}_stationary_mean_loss']==pytest.approx(row[f'{split}_stationary_mean_loss_h1'])
        assert row[f'{split}_stationary_lagged_covariance_loss']==pytest.approx(
            (row[f'{split}_stationary_lagged_covariance_loss_h2']+
             row[f'{split}_stationary_lagged_covariance_loss_h3'])/2)


def test_independent_lags_preserve_endpoint_objective():
    torch.manual_seed(1)
    model=NeuralSDE(SDEConfig(2,.01,history_steps=1,state_variable='velocity',hidden_layers=(4,)))
    x=torch.randn(4,2);hist=torch.randn(4,1,2);ends=torch.randn(4,3,2)
    loss=StationaryMeanLoss((1,),lagged_covariance_horizons=(2,3))
    assert loss.horizons==(1,2,3) and loss.endpoint_horizons==(1,)
    a=loss.statistics(model,x,ends,history_states=hist,generator=torch.Generator().manual_seed(4))
    b=StationaryMeanLoss(1).statistics(model,x,ends[:,0],history_states=hist,generator=torch.Generator().manual_seed(4))
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb)


def test_full_gradient_audit_with_gmm_parent(shared_data,tmp_path,monkeypatch):
    from modelling.neural_sde import correlation_gradient_diagnostic as diagnostic
    model=NeuralSDE(SDEConfig(shared_data.n_coordinates,shared_data.native_dt,history_steps=1,
        state_variable='velocity',hidden_layers=(4,)))
    path=tmp_path/'parent.pt'
    torch.save(dict(model_config=model.config.to_dict(),model_state=model.state_dict(),
        training=dict(multistep_objective='gmm_nll',multistep_horizons=[1,2],multistep_particles=2,
                      stationary_mean_horizons=[1],multistep_weight=.03)),path)
    monkeypatch.setattr(diagnostic,'checkpoint_data',lambda checkpoint:shared_data)
    args=SimpleNamespace(checkpoint=str(path),output=str(tmp_path/'audit'),split='train',
        coordinates=[0,1],horizons=[2,3],starts=2,window_seeds=[0,1],noise_seeds=[0,1],
        weight=.01,fraction=.1,device='cpu',threads=1)
    before=path.read_bytes()
    report=diagnostic.diagnose(args)
    assert path.read_bytes()==before
    assert report['reference']['split']=='train'
    assert report['objectives']['correlation']['groups']['diffusion']['expected_weighted_gradient_norm']==0
    assert set(report['per_horizon'])=={'2','3'}
    assert report['objectives']['correlation']['noise_components_drift']['between_batch_rms'] is not None
