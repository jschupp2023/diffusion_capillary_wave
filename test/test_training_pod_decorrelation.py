import numpy as np

from data_analysis.correlations.training_pod_decorrelation import (
    analyze_increment_path, clean_runs, decorrelation_summary,
    segmented_correlation)


def direct_segmented_correlation(segments, max_lag):
    result = []
    for lag in range(max_lag + 1):
        x = np.concatenate([segment[:len(segment)-lag] for segment in segments
                            if len(segment) > lag])
        y = np.concatenate([segment[lag:] for segment in segments
                            if len(segment) > lag])
        result.append([np.corrcoef(x[:, mode], y[:, mode])[0, 1]
                       for mode in range(x.shape[1])])
    return np.asarray(result)


def test_segmented_correlation_matches_direct_pairs_without_crossing_gaps():
    rng = np.random.default_rng(17)
    segments = [rng.normal(size=(31, 3)), rng.normal(size=(19, 3)) + 8]
    actual, count = segmented_correlation(segments, 12, min_pairs=2)
    np.testing.assert_allclose(actual, direct_segmented_correlation(segments, 12), atol=1e-12)
    np.testing.assert_array_equal(count, [sum(max(len(x)-lag, 0) for x in segments)
                                          for lag in range(13)])


def test_model_normalization_is_explicit_and_correlation_invariant():
    rng = np.random.default_rng(2)
    increments = rng.normal(size=(80, 2))
    values = np.vstack((np.zeros((1, 2)), np.cumsum(increments, axis=0)))
    good = np.ones(80, dtype=bool)
    normalized, _, _ = analyze_increment_path(
        values, good, np.array([2., -3.], dtype=np.float32),
        np.array([4., .2], dtype=np.float32), 10, 2)
    raw, _ = segmented_correlation([increments], 10, 2)
    np.testing.assert_allclose(normalized, raw, rtol=2e-6, atol=2e-6)


def test_trimmed_gap_is_never_compressed_into_adjacent_increment_pair():
    good = np.array([True, True, False, True, True, True])
    assert clean_runs(good) == [(0, 2), (3, 6)]
    values = np.arange(7, dtype=float)[:, None] ** 2
    _, count, _ = analyze_increment_path(
        values, good, np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32), 3, 1)
    np.testing.assert_array_equal(count, [5, 3, 1, 0])


def test_persistent_horizon_requires_complete_run_and_reports_rebound():
    rho = np.array([1., .2, .05, .02, .08, .11, .04, .03, .02])
    result = decorrelation_summary(rho, threshold=.1, consecutive=3)
    assert result["first_crossing_frames"] == 2
    assert result["persistent_frames"] == 2
    assert result["later_rebound_after_K"] is True

    no_run = decorrelation_summary([1., .05, .2, .05], threshold=.1, consecutive=2)
    assert no_run["persistent_frames"] is None
