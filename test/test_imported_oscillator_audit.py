import numpy as np

from data_analysis.correlations.imported_oscillator_audit import modal_response, response_psd
from modelling.neural_sde.oscillator_core_check import discrete_psd


def test_frequency_response_recovers_known_stiffness_and_damping_additions():
    dt=1/115200
    t=np.array([[1.,.3],[.2,1.]])
    omega=np.array([700.,2100.]);gamma=np.array([20.,45.])
    added_k=np.array([.4,-.2])*omega**2*dt**2
    added_d=np.array([2.,3.])*2*gamma*dt
    inv=np.linalg.inv(t)
    jac=np.stack((-(t*(added_k+added_d))@inv,(t*added_d)@inv))
    result=modal_response(jac,t,omega,gamma,omega/(2*np.pi),dt)
    np.testing.assert_allclose(result['effective_stiffness_over_fixed'],[1.4,.8],rtol=1e-10)
    np.testing.assert_allclose(result['effective_damping_over_fixed'],[3.,4.],rtol=1e-10)


def test_core_frequency_response_agrees_with_independent_modal_formula():
    dt=1/115200;t=np.array([[1.,.3],[.2,1.]])
    omega=np.array([700.,2100.]);gamma=np.array([20.,45.]);sigma=.0003
    inv=np.linalg.inv(t);k=(t*omega**2)@inv;d=(t*2*gamma)@inv
    f=np.arange(50.,1500.,5.)
    result=response_psd(k,d,np.zeros((64,2,2)),np.eye(2)*sigma,f,dt,alpha=0.)
    expected=discrete_psd(t,omega,gamma,dt,sigma,f)
    np.testing.assert_allclose(result,expected,rtol=1e-9,atol=1e-14)
