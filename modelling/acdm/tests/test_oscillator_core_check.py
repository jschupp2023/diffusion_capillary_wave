import numpy as np
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.oscillator_core_check import (
    core_matrix, stationary_noise, simulate, discrete_psd,
)


def operators():
    t=np.array([[1.,.4],[.2,1.]])
    omega=np.array([200.,600.]);gamma=np.array([15.,25.])
    return t,omega,gamma,(t*omega**2)@np.linalg.inv(t),(t*2*gamma)@np.linalg.inv(t)


def test_core_recurrence_matches_neural_sde_in_observed_coordinates():
    _,_,_,k,damping=operators();dt=.001
    model=NeuralSDE(SDEConfig(reduced_dim=2,dt=dt,state_variable='velocity',
        history_steps=1,drift_type='imported_oscillator_residual')).double()
    model.drift_model.configure(k,damping,np.zeros(2),model)
    x=torch.tensor([[.3,-.4]],dtype=torch.float64)
    u=torch.tensor([[.02,.03]],dtype=torch.float64)
    h=(x-u)[:,None]
    mean,_=model.velocity_step_distribution(x,h,u)
    positions,velocity=simulate(k,damping,dt,np.r_[x.numpy()[0],u.numpy()[0]],0.,1,np.random.default_rng(0))
    np.testing.assert_allclose(velocity,mean.detach().numpy()[0],rtol=1e-12)
    np.testing.assert_allclose(positions[-1],(x+mean).detach().numpy()[0],rtol=1e-12)
    np.testing.assert_allclose(np.r_[positions[-1],velocity],core_matrix(k,damping,dt)@np.r_[x.numpy()[0],u.numpy()[0]])


def test_discrete_density_integrates_to_stationary_covariance():
    t,omega,gamma,k,damping=operators();dt=.001
    f=core_matrix(k,damping,dt)
    sigma,p,error=stationary_noise(f)
    assert np.linalg.eigvals(f).real.max()<1
    assert error<1e-10
    assert abs(np.diag(p)[:2].mean()-1)<1e-12
    frequency=np.linspace(0,1/(2*dt),32769)
    density=discrete_psd(t,omega,gamma,dt,sigma,frequency)
    # Endpoint factors are one-sided; midpoint-grid rectangular integration
    # exactly counts DC/Nyquist once and the interior frequencies twice.
    integrated=density.sum(axis=0)*(frequency[1]-frequency[0])
    np.testing.assert_allclose(integrated,np.diag(p)[:2],rtol=1e-9)
    trajectory,velocity=simulate(k,damping,dt,np.ones(4),0.,2000,np.random.default_rng(0))
    assert np.linalg.norm(np.r_[trajectory[-1],velocity])<1e-9
