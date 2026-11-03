import numpy as np

from data_analysis.correlations.oscillator_conditional_compatibility import (
    modal_sequences, resonance_rows, spectral_budget,
)


def test_modal_sequence_identity_and_core_conversion():
    rng = np.random.default_rng(4)
    d, n, dt = 3, 2000, .001
    mixing = rng.normal(size=(d, d))
    while abs(np.linalg.det(mixing)) < .2:
        mixing = rng.normal(size=(d, d))
    omega = np.array([20., 31., 45.])
    gamma = np.array([.4, .8, 1.2])
    stiffness = (mixing * omega**2) @ np.linalg.inv(mixing)
    damping = (mixing * (2 * gamma)) @ np.linalg.inv(mixing)
    q = rng.normal(size=(n, d)).cumsum(axis=0) * .01
    equilibrium = np.array([2., -1., .5])
    values = equilibrium + q @ mixing.T
    result = modal_sequences(values, mixing, equilibrium, stiffness, damping, dt)
    expected_core = (-dt**2 * result["q"] * omega**2
                     -dt * result["u"] * (2 * gamma))
    np.testing.assert_allclose(result["core"], expected_core, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result["acceleration"],
                               result["core"] + result["residual"], atol=1e-14)


def test_spectral_budget_closes_with_signed_cross_term():
    rng = np.random.default_rng(5)
    n, d = 8192, 2
    core = rng.normal(size=(n, d))
    residual = -.75 * core + .2 * rng.normal(size=(n, d))
    acceleration = core + residual
    sequence = dict(q=rng.normal(size=(n, d)), u=rng.normal(size=(n, d)),
                    core=core, residual=residual, acceleration=acceleration)
    frequency, spectra, _ = spectral_budget(sequence, fs=1000., nperseg=1024)
    closure = spectra["core"] + spectra["residual"] + 2 * spectra["core_residual"].real
    np.testing.assert_allclose(closure, spectra["acceleration"], rtol=1e-12, atol=1e-14)
    rows = resonance_rows(frequency, spectra, np.array([100., 200.]),
                          2*np.pi*np.array([100., 200.]), np.array([1., 1.]), .001)
    assert all(row["signed_cross_over_acceleration"] < 0 for row in rows)
    assert all(abs(row["budget_sum_over_acceleration"] - 1) < 1e-12 for row in rows)
