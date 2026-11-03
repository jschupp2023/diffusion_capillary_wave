import numpy as np
import torch

from modelling.latent_oscillators.peak_background_fit import PeakModel,fit_weights,loss_function


def test_only_resonant_directions_forced_and_poles_bounded():
    m=PeakModel([130.,550.],[20.,80.],np.eye(3),3,14400,128)
    with torch.no_grad():
        m.force.fill_(.4);m.peak_shift.copy_(torch.tensor([-100.,100.]))
        m.width.copy_(torch.tensor([-100.,100.]))
    w,g,q,p=m.modal_statistics()
    np.testing.assert_allclose(g[:2].detach(),np.array([20*.75,80*1.25])/1000)
    np.testing.assert_allclose(np.sqrt(w[:2].detach().numpy()**2-2*g[:2].detach().numpy()**2)*1000/(2*np.pi),[122,558])
    assert torch.count_nonzero(q[2])==0 and torch.count_nonzero(p[2])==0
    assert np.linalg.eigvalsh(m.spectrum().detach().numpy()).min()>-1e-12


def test_weighted_peak_loss_has_finite_gradients_with_frozen_background():
    m=PeakModel([130.],[30.],np.eye(3),3,2048,128)
    f=np.fft.rfftfreq(128,1/2048);indices=np.flatnonzero((f>80)&(f<600))
    weights=fit_weights(f[indices],np.array([130.]),3)
    np.testing.assert_allclose(weights[0].sum(),1)
    bg=torch.eye(3,dtype=torch.complex128)[None].repeat(len(indices),1,1)*.0001
    target=(bg+m.spectrum(indices)*1.4).detach()
    loss=loss_function(bg+m.spectrum(indices),target,*weights);loss.backward()
    assert m.force.grad.isfinite().all() and m.force.grad.abs().sum()>0
    assert torch.count_nonzero(m.force.grad[1:])==0
    assert not bg.requires_grad


def test_all_coordinates_can_be_resonant_without_completion():
    m=PeakModel([130.,550.],[20.,80.],np.eye(2),3,14400,128)
    assert m.broad_frequency.numel()==m.broad_damping.numel()==0
    m.project()
    t,k,d,q=m.operators()
    assert torch.linalg.matrix_rank(k)==torch.linalg.matrix_rank(d)==2
    assert torch.linalg.eigvalsh(q).min()>0
    spectrum=m.spectrum()
    assert torch.isfinite(spectrum).all()
    spectrum.real.sum().backward()
    assert m.force.grad.isfinite().all()
