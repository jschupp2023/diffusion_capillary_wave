import numpy as np
import pytest

from modelling.latent_oscillators.spectral_background import valley_background, select_peak_centers


def test_linear_matrix_background_recovered_between_anchors():
    f = np.arange(80., 401., 2.)
    matrix = np.array([[2., .3j], [-.3j, 1.]])
    true = (2-.001*f)[:, None, None]*matrix[None]
    direction = np.array([1., 1j])
    peak = sum(np.maximum(0, 1-abs(f-p)/6)*5 for p in [160, 260])
    spectrum = true+peak[:, None, None]*np.outer(direction, direction.conj())
    result = valley_background(f, spectrum, [160,260], 10, coordinates=(0,1))
    np.testing.assert_allclose(result['background'][result['supported']],true[result['supported']],atol=1e-12)
    assert np.linalg.eigvalsh(result['background']).min() > 0
    np.testing.assert_allclose(result['background'], result['background'].conj().transpose(0,2,1))
    assert not result['masked'][result['anchor_windows']].any()
    assert not result['supported'][0]  # Endpoint extension is explicitly flagged.


def test_dense_peak_mask_rejected_and_input_unmodified():
    f = np.arange(100., 201.)
    spectrum = np.tile(np.eye(2), (len(f),1,1)).astype(complex)
    original = spectrum.copy()
    with pytest.raises(ValueError, match='Fewer than two'):
        valley_background(f, spectrum, [150], 100, coordinates=(0,1))
    np.testing.assert_array_equal(spectrum, original)


def test_peak_union_keeps_known_and_adds_training_peaks():
    f = np.arange(80., 401.)
    power = 1+10*np.exp(-((f-150)/3)**2)+10*np.exp(-((f-250)/3)**2)
    spectrum = power[:,None,None]*np.eye(2)
    centers, _ = select_peak_centers(f, spectrum, [151, 350], coordinates=(0,1))
    np.testing.assert_allclose(centers, [150,250,350])
