"""Check the plotted states against shared-POD training normalization."""
import json

import numpy as np

from data_analysis.plot_normalized_coordinates import increment_mahalanobis, main
from modelling.acdm.tests.conftest import shared_data


def test_increment_mahalanobis_handles_singular_covariance():
    state = np.column_stack((np.arange(20.0), 2 * np.arange(20.0)))
    state[10:, 0] += 3
    state[10:, 1] += 6
    result = increment_mahalanobis(state, lag=1, percentile=95)
    assert result["effective_covariance_rank"] == 1
    assert np.isfinite(result["distance"]).all()
    np.testing.assert_allclose(result["threshold"], np.percentile(result["distance"], 95))


def test_plot_selected_model_input_coordinates(shared_data, tmp_path):
    output = tmp_path / "coordinates"
    main(["0p20", "--rep", "3", "--rank", "2", "--start", "2", "--timesteps", "4",
          "--increment-lag", "2",
          "--modes", "0", "2", "--data-root", shared_data.source_config["data_root"],
          "--output", str(output)])

    shared_data.fit_standardization()
    name = next(name for name in shared_data.frame_counts if name.endswith("_rep3"))
    time, physical = shared_data.read_coordinates(name, 2, 7)
    expected = ((physical.astype(np.float32) - shared_data.mean.astype(np.float32))
                / shared_data.scale.astype(np.float32))[:, [0, 2]]
    with np.load(output / "normalized_coordinates.npz") as saved:
        np.testing.assert_array_equal(saved["step"], np.arange(2, 7))
        np.testing.assert_array_equal(saved["modes"], [0, 2])
        np.testing.assert_allclose(saved["time_seconds"], time - time[0])
        np.testing.assert_allclose(saved["recording_time_seconds"], time)
        np.testing.assert_array_equal(saved["coordinates"], expected)
    with np.load(output / "normalized_increments_lag2.npz") as saved:
        np.testing.assert_array_equal(saved["start_step"], np.arange(2, 5))
        np.testing.assert_array_equal(saved["end_step"], np.arange(4, 7))
        np.testing.assert_allclose(saved["time_seconds"], time[2:] - time[0])
        np.testing.assert_allclose(saved["recording_time_seconds"], time[2:])
        np.testing.assert_array_equal(saved["increments"], expected[2:] - expected[:-2])
    with np.load(output / "increment_mahalanobis_lag2.npz") as saved:
        assert saved["distance"].shape == (3,)
        assert saved["full_trajectory_distance"].shape == (15,)
        np.testing.assert_array_equal(saved["distance"], saved["full_trajectory_distance"][2:5])
        np.testing.assert_array_equal(saved["coordinate_indices"], [0, 1, 2])
        np.testing.assert_allclose(
            saved["threshold"], np.percentile(saved["full_trajectory_distance"], 95)
        )
        np.testing.assert_array_equal(saved["above_threshold"],
                                      saved["distance"] > saved["threshold"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["coordinate_zero"] == "sqrt(number of spatial points) * frame spatial mean"
    assert summary["normalization_train_repetitions"] == list(shared_data.train_repetitions)
    assert summary["increment_lag_steps"] == 2
    assert summary["increment_time_label"] == "ending frame"
    assert summary["first_step"] == 2
    assert summary["last_step"] == 6
    assert summary["increment_score"] == "mahalanobis"
    assert summary["increment_score_coordinates"] == "all model-visible coordinates 0..2"
    assert summary["increment_score_threshold_percentile"] == 95
    assert (output / "normalized_coordinates.png").is_file()
    assert (output / "normalized_increments_lag2.png").is_file()
    assert (output / "increment_mahalanobis_lag2.png").is_file()

    l2_output = tmp_path / "coordinates_l2"
    main(["0p20", "--rep", "3", "--rank", "2", "--start", "2", "--timesteps", "4",
          "--increment-lag", "2", "--increment-score", "l2",
          "--modes", "0", "2", "--data-root", shared_data.source_config["data_root"],
          "--output", str(l2_output)])
    with np.load(l2_output / "increment_l2_norm_lag2.npz") as saved:
        np.testing.assert_array_equal(saved["norm"], saved["full_trajectory_norm"][2:5])
        np.testing.assert_allclose(
            saved["threshold"], np.percentile(saved["full_trajectory_norm"], 95)
        )
        assert str(saved["score_kind"]) == "l2"
    l2_summary = json.loads((l2_output / "summary.json").read_text())
    assert l2_summary["increment_score"] == "l2"
    assert "mahalanobis_covariance" not in l2_summary
    assert (l2_output / "increment_l2_norm_lag2.png").is_file()
