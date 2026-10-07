"""Checks for the uncentered, all-valid-increment Q(h) diagnostic."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from data_analysis.correlations import increment_covariance as diagnostic
from data_analysis.correlations import increment_covariance_batch as batch_diagnostic
from modelling.acdm.tests.conftest import shared_data


def test_estimator_uses_all_increments_and_keeps_cross_covariance():
    z = np.column_stack((np.arange(12, dtype=float) ** 2,
                         3 * np.arange(12, dtype=float) + 1))
    h, q = diagnostic.estimate_increment_covariance(z, 0.25, batch_size=3)
    np.testing.assert_allclose(h, np.arange(1, 11) * 0.25)
    for k in range(1, 11):
        delta = z[k:] - z[:-k]
        expected = delta.T @ delta / ((len(z) - k) * k * 0.25)
        np.testing.assert_allclose(q[k - 1], expected, rtol=1e-14)
    assert q[0, 0, 1] > 0
    assert q[-1, 0, 1] > 0
    selected_h, selected_q = diagnostic.estimate_increment_covariance(
        z, 0.25, min_lag=3, max_lag=5, batch_size=3)
    np.testing.assert_allclose(selected_h, h[2:5])
    np.testing.assert_allclose(selected_q, q[2:5])


def test_checkpoint_normalization_and_saved_matrices(monkeypatch, tmp_path):
    physical = np.column_stack((np.arange(12.) + 20, np.arange(12.) ** 2 + 100))
    name = "Ca_ac_0p002490_rep3"
    basis = tmp_path / "shared.h5"
    source = tmp_path / "source.h5"
    checkpoint = dict(model_config={"reduced_dim": 2, "dt": .5},
                      data_config={"experiment": "0p30", "rank": 1},
                      model_state={"state_mean": np.array([20., 100.]),
                                   "state_std": np.array([2., 5.]),
                                   "diffusion_model.raw_diagonal": np.zeros(2)})
    data = SimpleNamespace(native_dt=.5, frame_counts={name: len(physical)},
                           source_paths={name: source}, basis_path=basis,
                           signal_units="microns", train_repetitions=(name,),
                           read_coordinates=lambda selected, start, stop:
                               (np.arange(start, stop) * .5, physical[start:stop]))
    monkeypatch.setattr(diagnostic, "load_checkpoint", lambda path, device: checkpoint)
    monkeypatch.setattr(diagnostic, "checkpoint_data", lambda saved, **kwargs: data)
    z, dt, metadata = diagnostic.load_normalized_trajectory("0p30", 3, 1, tmp_path / "best.pt",
                                                            batch_size=4)
    np.testing.assert_allclose(z, (physical - [20., 100.]) / [2., 5.])
    assert metadata["coordinate_count"] == 2
    h, q = diagnostic.estimate_increment_covariance(z, dt, min_lag=3, max_lag=10)
    output = diagnostic.save_diagnostic(h, q, metadata, tmp_path / "diagnostic")
    with np.load(output / "Q.npz") as saved:
        np.testing.assert_allclose(saved["Q"], q)
        np.testing.assert_allclose(saved["trace"], np.trace(q, axis1=1, axis2=2))
        np.testing.assert_array_equal(saved["k"], np.arange(3, 11))
        np.testing.assert_array_equal(saved["increment_counts"], len(z) - np.arange(3, 11))
    assert (output / "increment_covariance.png").exists()
    with pytest.raises(FileExistsError, match="--overwrite"):
        diagnostic.save_diagnostic(h, q, metadata, output)
    raw_output = diagnostic.save_diagnostic(h, q, metadata, tmp_path / "unscaled", unscaled=True)
    with np.load(raw_output / "second_moment.npz") as saved:
        np.testing.assert_allclose(saved["second_moment"], q * h[:, None, None])
        np.testing.assert_allclose(saved["trace"], np.trace(q, axis1=1, axis2=2) * h)
    assert (raw_output / "increment_second_moment.png").exists()


def test_cli_uses_shared_pod_training_normalization_without_checkpoint(shared_data, tmp_path):
    output = tmp_path / "direct"
    diagnostic.main([shared_data.experiment, "--rep", "3", "--rank", str(shared_data.rank),
                     "--data-root", shared_data.source_config["data_root"],
                     "--max-lag", "3", "--output", str(output)])
    metadata = json.loads((output / "summary.json").read_text())
    assert metadata["checkpoint"] is None
    assert metadata["normalization_train_repetitions"] == list(shared_data.train_repetitions)
    shared_data.fit_standardization()
    name = next(name for name in shared_data.frame_counts if name.endswith("_rep3"))
    _, physical = shared_data.read_coordinates(name, 0, shared_data.frame_counts[name])
    mean = shared_data.mean.astype(np.float32).astype(np.float64)
    std = shared_data.scale.astype(np.float32).astype(np.float64)
    expected_z = (physical - mean) / std
    with np.load(output / "Q.npz") as saved:
        for k in range(1, 4):
            delta = expected_z[k:] - expected_z[:-k]
            np.testing.assert_allclose(saved["Q"][k - 1], delta.T @ delta / (len(delta) * k * shared_data.native_dt))


def test_vector_norm_percentile_trim_uses_one_mask_for_full_matrix(tmp_path):
    rng = np.random.default_rng(11)
    steps = rng.normal(size=(200, 2))
    steps[34] = [50., -40.]
    z = np.vstack((np.zeros((1, 2)), np.cumsum(steps, axis=0)))
    h, q, retained, thresholds, excluded_energy_percent = diagnostic.estimate_increment_covariance(
        z, .02, max_lag=3, batch_size=17, trim_percentiles=(1, 99), return_details=True)
    for index, k in enumerate(range(1, 4)):
        delta = z[k:] - z[:-k]
        norms = np.linalg.norm(delta, axis=1)
        bounds = np.percentile(norms, [1, 99])
        keep = (norms >= bounds[0]) & (norms <= bounds[1])
        np.testing.assert_allclose(thresholds[index], bounds)
        assert retained[index] == keep.sum()
        np.testing.assert_allclose(q[index], delta[keep].T @ delta[keep] / (keep.sum() * h[index]))
        assert excluded_energy_percent[index] == pytest.approx(
            100 * np.sum(norms[~keep] ** 2) / np.sum(norms ** 2))
    metadata = dict(n_samples=len(z), dt_seconds=.02)
    output = diagnostic.save_diagnostic(h, q, metadata, tmp_path / "trimmed",
                                        retained_counts=retained, norm_thresholds=thresholds,
                                        trim_percentiles=(1, 99),
                                        excluded_energy_percent=excluded_energy_percent)
    with np.load(output / "Q.npz") as saved:
        np.testing.assert_array_equal(saved["retained_increment_counts"], retained)
        np.testing.assert_allclose(saved["norm_thresholds"], thresholds)
        np.testing.assert_allclose(saved["excluded_increment_energy_percent"], excluded_energy_percent)


def test_batch_analyzes_all_repetitions_with_upper_only_cutoff(shared_data):
    output = batch_diagnostic.analyze_batch(
        shared_data.experiment, shared_data.rank, [3, 1, 3], 25,
        data_root=shared_data.source_config["data_root"])
    assert output.parent == (shared_data.source_paths[shared_data.train_repetitions[0]].parents[2]
                             / "increments_analysis" / shared_data.experiment)
    assert (output / "combined_trace.png").is_file()
    assert (output / "combined_excluded_energy.png").is_file()
    report = json.loads((output / "summary.json").read_text())
    assert report["lags"] == [1, 3]
    assert report["upper_cutoff_percent"] == 25
    assert report["repetitions"] == [1, 2, 3, 4]
    with np.load(output / "combined.npz") as combined:
        assert combined["trace_Q"].shape == (4, 2)
        assert combined["excluded_increment_energy_percent"].shape == (4, 2)
        with np.load(output / "rep3/Q.npz") as individual:
            np.testing.assert_allclose(combined["trace_Q"][2], individual["trace"])
            np.testing.assert_allclose(combined["excluded_increment_energy_percent"][2],
                                       individual["excluded_increment_energy_percent"])
            shared_data.fit_standardization()
            name = next(name for name in shared_data.frame_counts if name.endswith("_rep3"))
            _, physical = shared_data.read_coordinates(name, 0, shared_data.frame_counts[name])
            z = ((physical - shared_data.mean.astype(np.float32))
                 / shared_data.scale.astype(np.float32))
            delta = z[1:] - z[:-1]
            norms = np.linalg.norm(delta, axis=1)
            keep = norms <= np.percentile(norms, 75)
            np.testing.assert_allclose(individual["Q"][0],
                                       delta[keep].T @ delta[keep] / (keep.sum() * shared_data.native_dt))
            assert keep[np.argmin(norms)]
    per_rep = json.loads((output / "rep3/summary.json").read_text())
    assert per_rep["selection"]["method"] == "upper_tail_normalized_increment_vector_norm_per_lag"
