"""Physical and statistical checks for frequency-resolved peak modes."""
import numpy as np
import pytest
from scipy.signal import welch

from data_analysis.pod.spectral_peak_atlas import (
    analyze_matrices, capture, eigensystem, select_sites, similarity,
    spatial_patterns, spectral_matrices, metric_eigensystem,
)


def test_welch_diagonal_and_complex_traveling_pattern():
    fs = 1024.
    n = 1024
    t = np.arange(4096)/fs
    # Fourier coefficient has phase +phi, which must survive reconstruction.
    pattern = np.array([1., np.exp(.7j), .6*np.exp(-1.2j)])
    x = np.real(np.exp(2j*np.pi*80*t)[:, None]*pattern)
    sites = [dict(frequency_hz=80., half_width_hz=1.)]
    bands, neighbors, info = spectral_matrices(x, fs, n, sites)
    f, psd = welch(x, fs=fs, nperseg=n, noverlap=n//2, axis=0)
    np.testing.assert_allclose(np.diag(neighbors[0, 1]).real, psd[80], atol=1e-12)
    np.testing.assert_allclose(np.diag(bands[0]).real, psd[79:82].sum(0), atol=1e-12)
    ev, v = eigensystem(bands[0])
    expected = pattern/np.linalg.norm(pattern)
    assert abs(np.vdot(expected, v[:, 0])) > 1-1e-12
    assert abs(np.vdot(expected.conj(), v[:, 0])) < .8
    assert ev[0]/ev.sum() > 1-1e-12
    assert info['windows'] == 7


def test_real_matrix_phase_invariance_and_degenerate_subspace():
    _, v = eigensystem(np.diag([3., 3., 1.]))
    rotated = v[:, :2] @ (np.array([[1, 1], [-1, 1]])/np.sqrt(2))
    assert similarity(v, rotated) == pytest.approx(.5)
    assert similarity(v, rotated, 2) == pytest.approx(1.)
    assert similarity(v, v*np.exp(1.3j)) == pytest.approx(1.)
    assert capture(np.diag([3., 3., 1.]), v, 2) == pytest.approx(6/7)


def test_nonorthogonal_mean_companion_matches_pixel_space_covariance():
    h = np.array([[1., .2], [0., 1.], [.3, -.4]])
    s = np.array([[2., .3j], [-.3j, 1.]])
    values, vectors, _ = metric_eigensystem(s, h.T @ h)
    physical = h @ vectors
    np.testing.assert_allclose(physical.conj().T @ physical, np.eye(2), atol=1e-12)
    direct, _ = eigensystem(h @ s @ h.T)
    np.testing.assert_allclose(values, direct[:2], atol=1e-12)
    np.testing.assert_allclose((h @ s @ h.T) @ physical, physical*values, atol=1e-12)


def test_spatial_mapping_preserves_norm_and_does_not_conjugate():
    rng = np.random.default_rng(12)
    q, _ = np.linalg.qr(np.column_stack([np.ones(16), rng.normal(size=(16, 3))]))
    modes = q[:, 1:].T.reshape(3, 4, 4)
    vector = np.array([1., 1j, -.5j]); vector /= np.linalg.norm(vector)
    vectors = np.stack([vector, vector.conj()], axis=-1)[None]
    fields, err = spatial_patterns(modes, vectors)
    expected = q[:, 1:] @ vector
    assert abs(np.vdot(expected, fields[0, 0].ravel())) == pytest.approx(1.)
    assert err < 1e-12
    assert np.linalg.norm(fields[0, 0]) == pytest.approx(1.)
    full = np.zeros((1, 4, 2), complex); full[0, 0, 0] = 1; full[0, 1, 1] = 1
    fields, _ = spatial_patterns(modes, full, include_mean=True)
    np.testing.assert_allclose(fields[0, 0], .25)
    with pytest.raises(ValueError, match='not orthonormal'):
        spatial_patterns(modes*2, vectors)


def test_training_selection_and_band_separation():
    f = np.arange(501.)
    p = np.ones((501, 4))
    for channel, centers in [(0, [100, 200, 300]), (1, [130, 230, 330])]:
        for center in centers:
            p[:, channel] += 10*np.exp(-.5*((f-center)/2)**2)
    sites, candidates = select_sites(f, p, low=80, high=350, per_channel=3,
                                     valley_count=2, half_width_hz=10)
    assert [s['frequency_hz'] for s in sites if s['kind']=='peak'] == [100,130,200,230,300,330]
    assert len(candidates) == 6
    for site in sites:
        for candidate in candidates:
            if candidate['bin'] != site['bin']:
                assert abs(candidate['frequency_hz']-site['frequency_hz']) > site['half_width_hz']


def test_validation_does_not_fit_modes_and_equal_repetition_weighting():
    # Two training records differ in total power; never normalize each CSD by trace.
    train = [np.diag([1., 9., 1., .1]), np.diag([1., 1., 3., .1])]
    validation = np.diag([1., .1, 100., .1])
    bands = np.asarray(train+[validation], complex)[:, None]
    neighbors = np.repeat(bands[:, :, None], 3, axis=2)
    rows, vectors, _, _ = analyze_matrices(bands, neighbors, 2, 3, [2,3], bootstrap=8)
    assert abs(vectors[0, 0, 0]) == pytest.approx(1.)
    assert rows[0]['eta1'] == pytest.approx(5/7.1)
    assert rows[0]['validation_capture1'] == pytest.approx(.1/100.2)
    assert rows[0]['validation_mode_mac'] == pytest.approx(0.)
    modified = bands.copy(); modified[-1] *= 1000
    _, next_vectors, _, _ = analyze_matrices(modified, neighbors, 2, 3, [2,3], bootstrap=8)
    np.testing.assert_allclose(vectors, next_vectors)
