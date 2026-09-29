import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_analysis.pod.compare_raw_highpass_pod import (
    _highpass_columns,
    _sample_treatment,
)


class CompareRawHighpassPodTests(unittest.TestCase):
    def test_mean_only_filter_preserves_spatial_fluctuations(self):
        time = np.arange(2_000, dtype=np.float64) / 1_000.0
        constant = 4.0 * np.sin(2 * np.pi * 5 * time)
        high = np.sin(2 * np.pi * 100 * time)
        pattern = np.array([-1.5, -0.5, 0.5, 1.5])
        raw = constant[:, None] + high[:, None] * pattern[None, :]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "filtered.dat"
            filtered = np.memmap(path, mode="w+", dtype=np.float32, shape=raw.shape)
            _highpass_columns(raw, time, 50.0, filtered, column_batch=2)
            raw_mean = raw.mean(axis=1)
            from data_analysis.energy.pod_gravitational_energy import highpass_spatial_mean

            filtered_mean = highpass_spatial_mean(raw_mean, time, 50.0)
            positions = np.arange(len(time))
            mean_only = _sample_treatment(
                "filtered_mean_only",
                raw,
                filtered,
                positions,
                raw_mean,
                filtered_mean,
            )
            raw_sample = _sample_treatment(
                "raw", raw, filtered, positions, raw_mean, filtered_mean
            )

            np.testing.assert_allclose(
                mean_only - mean_only.mean(axis=1, keepdims=True),
                raw_sample - raw_sample.mean(axis=1, keepdims=True),
                rtol=2e-6,
                atol=2e-6,
            )
            self.assertLess(
                np.mean(mean_only.mean(axis=1) ** 2),
                1e-3 * np.mean(raw_sample.mean(axis=1) ** 2),
            )


if __name__ == "__main__":
    unittest.main()
