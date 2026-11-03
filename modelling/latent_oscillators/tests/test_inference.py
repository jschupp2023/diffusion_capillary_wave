import numpy as np
from modelling.latent_oscillators.inference import (
    StateSpace, filter_schedule, filter_observations, forecast,
    simulate_observations, smooth_endpoint,
)


def saved_model(rank=1):
    return dict(frequencies_hz=np.array([37., 81.]),
                damping_per_second=np.array([12., 25.]),
                mixing=np.array([[[1.], [.4]], [[.2], [.8]]]).repeat(rank, axis=2),
                background=np.array([[[.4, .1], [.1, .3]]]),
                background_hz=np.array([110.]), fs=1000.)


def stationary_cross(system, first, second):
    f = np.linalg.matrix_power(system.transition, abs(first-second))
    return f if first >= second else f.T


def batch_condition(system, observations, endpoint):
    """Independent dense joint-Gaussian reference, without a Kalman recursion."""
    h = system.readout
    n = observations.shape[1]
    yy = np.block([[h@stationary_cross(system, t, s)@h.T for s in range(n)]
                   for t in range(n)])
    xy = np.hstack([stationary_cross(system, endpoint, s)@h.T for s in range(n)])
    gain = np.linalg.solve(yy, xy.T).T
    return observations.reshape(len(observations), -1)@gain.T, np.eye(len(system.transition))-gain@xy.T


def test_readout_covariance_matches_saved_poles_and_rank_copies():
    saved = saved_model(rank=2)
    system = StateSpace.from_saved(saved)
    np.testing.assert_allclose(system.transition@system.transition.T+system.noise,
                               np.eye(10), atol=1e-14)
    for lag in [0, 1, 19]:
        f = np.linalg.matrix_power(system.transition, lag)
        expected = sum(saved['mixing'][j]@saved['mixing'][j].T*f[4*j, 4*j]
                       for j in range(2))
        expected += saved['background'][0]@saved['background'][0].T*np.exp(-2*np.pi*110*lag/1000)
        np.testing.assert_allclose(system.readout@f@system.readout.T, expected, atol=1e-13)


def test_filter_matches_dense_gaussian_conditioning_and_exact_observation():
    system = StateSpace.from_saved(saved_model())
    y = np.random.default_rng(4).normal(size=(4, 9, 2))
    schedule = filter_schedule(system, 9)
    actual, _, _ = filter_observations(system, y, schedule)
    expected, covariance = batch_condition(system, y, 8)
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_allclose(schedule.posterior[9], covariance, atol=2e-13)
    np.testing.assert_allclose(actual@system.readout.T, y[:, -1], atol=2e-13)
    np.testing.assert_allclose(system.readout@schedule.posterior[9]@system.readout.T, 0, atol=2e-13)
    assert np.linalg.eigvalsh(schedule.posterior[9]).min() > -1e-12


def test_endpoint_smoother_matches_dense_conditioning_and_reduces_uncertainty():
    system = StateSpace.from_saved(saved_model())
    y = np.random.default_rng(5).normal(size=(4, 10, 2))
    schedule = filter_schedule(system, 6)
    mean, _, _ = filter_observations(system, y[:, :6], schedule)
    actual, covariance = smooth_endpoint(system, mean, schedule.posterior[6], y[:, 6:])
    expected, expected_covariance = batch_condition(system, y, 5)
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_allclose(covariance, expected_covariance, atol=2e-13)
    assert np.linalg.eigvalsh(schedule.posterior[6]-covariance).min() > -1e-12


def test_forecast_matches_joint_gaussian_and_does_not_assimilate_future():
    system = StateSpace.from_saved(saved_model())
    rng = np.random.default_rng(6)
    y = rng.normal(size=(4, 7, 2))
    schedule = filter_schedule(system, 7)
    mean, _, _ = filter_observations(system, y, schedule)
    expected_state, covariance = batch_condition(system, y, 13)
    predicted, predictive_covariance = forecast(system, mean, schedule.posterior[7], 7)
    np.testing.assert_allclose(predicted, expected_state@system.readout.T, atol=2e-13)
    np.testing.assert_allclose(predictive_covariance, system.readout@covariance@system.readout.T, atol=2e-13)
    # A longer Riccati schedule must not import information beyond the input prefix.
    long_schedule = filter_schedule(system, 20)
    other, _, _ = filter_observations(system, y, long_schedule)
    np.testing.assert_array_equal(other, mean)


def test_synthetic_filter_uncertainty_and_innovations_are_calibrated():
    system = StateSpace.from_saved(saved_model())
    y, truth = simulate_observations(system, 30, 2048, 29, seed=37)
    schedule = filter_schedule(system, 30)
    estimate, _, white = filter_observations(system, y, schedule, keep_path=True)
    diagonal = np.diag(schedule.posterior[30])
    normalized_error = (truth-estimate)/np.sqrt(diagonal)
    assert abs(np.mean(normalized_error**2)-1) < .06
    assert abs(np.mean(abs(normalized_error) <= 1.95996398454)-.95) < .01
    assert abs(np.mean(white**2)-1) < .02
    assert abs(np.mean(white[:, 1:]*white[:, :-1])) < .02
