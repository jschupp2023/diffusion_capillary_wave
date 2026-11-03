import numpy as np
import pytest

from data_analysis.psd.rollout_experimental_psd import _centered_clean_windows


def test_centered_clean_windows_preserve_gaps_and_use_one_window_per_run():
    signals = np.stack((np.arange(10), 100 + np.arange(10)))
    keep = np.array([
        [False, True, True, True, True, False, True, True, True, True],
        [True, True, True, True, True, False, False, False, False, False],
    ])

    windows = _centered_clean_windows(signals, keep, nperseg=4)

    np.testing.assert_array_equal(
        windows,
        np.array([
            [1, 2, 3, 4],
            [6, 7, 8, 9],
            [100, 101, 102, 103],
        ]),
    )


def test_centered_clean_windows_rejects_missing_eligible_run():
    with pytest.raises(ValueError, match="no contiguous clean run"):
        _centered_clean_windows(
            np.arange(6, dtype=float)[None, :],
            np.array([[True, True, False, True, True, False]]),
            nperseg=3,
        )
