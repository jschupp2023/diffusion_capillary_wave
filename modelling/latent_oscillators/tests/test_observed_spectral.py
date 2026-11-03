import numpy as np
import torch
from scipy.linalg import block_diag, expm, solve_continuous_lyapunov

from modelling.latent_oscillators.observed_spectral import ObservedSpectralModel, spectral_loss
from modelling.latent_oscillators.model import taper_correlation


def test_spectrum_matches_independent_state_space_welch():
    model = ObservedSpectralModel([130.], np.eye(3), condition_cap=3, nperseg=64)
    with torch.no_grad():
        model.force[0, 1] = .3
        model.stretch[0, 1] = .6
    model.project()
    w, g, q, _ = [x.detach().numpy() for x in model.modal_statistics()]
    a = block_diag(*[np.array([[0., 1.], [-wi**2, -2*gi]]) for wi, gi in zip(w, g)])
    force = np.zeros((6, 6)); force[1::2, 1::2] = q
    p = solve_continuous_lyapunov(a, -force)
    t = model.transform().detach().numpy()
    readout = np.zeros((3, 6)); readout[:, ::2] = t
    c = np.array([readout @ expm(a*k*1000/model.fs) @ p @ readout.T for k in range(64)])
    taper = taper_correlation(64)
    result = []
    for k in range(33):
        pos = np.einsum('l,lij->ij', taper[1:]*np.exp(-2j*np.pi*k*np.arange(1, 64)/64), c[1:])
        result.append((c[0]+pos+pos.conj().T)/model.fs*(1 if k in (0, 32) else 2))
    np.testing.assert_allclose(model.spectrum().detach(), result, rtol=1e-9, atol=1e-12)
    indices = torch.tensor([2, 5, 12, 32])
    np.testing.assert_allclose(model.spectrum(indices).detach(), np.asarray(result)[indices], rtol=1e-9, atol=1e-12)
    assert np.max(abs(np.imag(result))) > 1e-6  # Tests cross-spectrum phase/sign.
    assert np.linalg.eigvalsh(result).min() > -1e-12


def test_condition_gauge_stability_and_operator_transform():
    for cap in [1, 3, 10]:
        model = ObservedSpectralModel([130., 550.], np.eye(4), condition_cap=cap, nperseg=32)
        with torch.no_grad():
            torch.manual_seed(1)
            model.stretch.copy_(torch.randn(4, 4)*4)
            model.rotation.copy_(torch.randn(4, 4))
        model.project()
        t, k, d, q = [v.detach().numpy() for v in model.operators()]
        assert np.linalg.cond(t) <= cap*(1+1e-8)
        np.testing.assert_allclose(np.linalg.norm(t, axis=0), 1, atol=1e-12)
        w, g = [x.detach().numpy() for x in model.poles()]
        np.testing.assert_allclose(k @ t, t*(w*w*1e6), rtol=1e-10, atol=1e-7)
        np.testing.assert_allclose(d @ t, t*(2*g*1000), rtol=1e-10, atol=1e-7)
        a = np.block([[np.zeros((4, 4)), np.eye(4)], [-k, -d]])
        assert np.linalg.eigvals(a).real.max() < 0
        assert np.linalg.eigvalsh(q).min() > 0


def test_spectral_gradient_matches_finite_difference():
    model = ObservedSpectralModel([130.], np.eye(3), condition_cap=3, nperseg=64)
    with torch.no_grad():
        model.force[0, 1] = .2
    reference = model.spectrum().detach()[2:]*1.2
    loss = spectral_loss(model.spectrum()[2:], reference)
    loss.backward()
    for parameter, index in [(model.stretch, (0, 1)), (model.peak_shift, (0,)), (model.force, (0, 1))]:
        analytical = parameter.grad[index].item()
        values = []
        for delta in [1e-5, -1e-5]:
            with torch.no_grad():
                parameter[index] += delta
                values.append(spectral_loss(model.spectrum()[2:], reference).item())
                parameter[index] -= delta
        np.testing.assert_allclose(analytical, (values[0]-values[1])/2e-5, rtol=1e-4, atol=1e-7)


def test_peak_metrics_require_prescribed_training_and_validation_peaks():
    from modelling.latent_oscillators.observed_spectral_study import peak_metrics
    f = np.arange(80., 301.)
    def spectra(peaks):
        power = sum(np.exp(-((f-p)/3)**2)*5 for p in peaks)+np.ones_like(f)
        return power[:, None, None]*np.eye(4)[None]
    result = peak_metrics(f, spectra([130, 250]), spectra([130, 200, 250]),
                          spectra([130, 250]), np.array([130, 200]))
    assert all(row['targets'] == 1 and row['hits'] == 1 for row in result)
    assert all(row['target_frequencies_hz'] == [130.] for row in result)
