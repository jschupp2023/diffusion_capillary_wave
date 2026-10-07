from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.data import fit_normalization, transition_batches
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData


def test_preparation_owns_disjoint_whole_repetition_splits(shared_data):
    data = shared_data
    assert list(map(len, data.splits.values())) == [2, 1, 1]
    assert len(set(sum((list(v) for v in data.splits.values()), []))) == 4
    assert data.native_dt == pytest.approx(.01)
    assert data.n_coordinates == 3  # rank 2 plus mean, while inputs retain 3 modes


def test_physical_projection_includes_all_local_modes_and_scaled_mean(shared_data):
    name = shared_data.splits["train"][0]
    _, actual = shared_data.read_coordinates(name, 0, 7)
    with h5py.File(shared_data.source_paths[name]) as source, h5py.File(shared_data.basis_path) as basis:
        a = source["reduced/coefficients"][:7]
        local = source["pod/modes"][:].reshape(3, 9)
        shared = basis["pod/modes"][:2].reshape(2, 9)
        expected = np.column_stack((np.sqrt(9) * source["preprocessing/frame_spatial_mean"][:7],
                                    a @ local @ shared.T))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_highpass_spatial_mean_selection_is_explicit_and_checkpointable(highpass_shared_data):
    data = highpass_shared_data
    assert data.spatial_mean_highpass_hz == 20.0
    assert data.source_config["spatial_mean_highpass_hz"] == 20.0
    assert data.signature()["spatial_mean_dataset"].endswith("cutoff_20_hz")
    name = data.splits["train"][0]
    _, actual = data.read_coordinates(name, 0, 7)
    with h5py.File(data.source_paths[name]) as source:
        expected = np.sqrt(data.n_space) * source[data.spatial_mean_dataset][:7]
    np.testing.assert_allclose(actual[:, 0], expected, atol=0, rtol=0)

    with pytest.raises(ValueError, match="high-pass cutoff"):
        SharedPODTrainingData(
            **data.source_config, include_spatial_mean=False,
        )
    unavailable = dict(data.source_config, spatial_mean_highpass_hz=30.0)
    with pytest.raises(ValueError, match="Invalid source dimensions"):
        SharedPODTrainingData(**unavailable)


def test_pod_only_preparation_omits_mean_and_keeps_repetition_splits(shared_data):
    pod_only = SharedPODTrainingData(**shared_data.source_config, include_spatial_mean=False,
                                     batch_size=5, output_dtype="float64")
    assert pod_only.n_coordinates == shared_data.rank
    assert pod_only.splits == shared_data.splits
    assert pod_only.source_config["include_spatial_mean"] is False
    assert pod_only.signature() != shared_data.signature()
    for name in shared_data.train_repetitions + shared_data.validation_repetitions:
        _, full = shared_data.read_coordinates(name, 0, 7)
        _, reduced = pod_only.read_coordinates(name, 0, 7)
        np.testing.assert_allclose(reduced, full[:, 1:])
    batch = next(pod_only.iter_transitions("train", batch_size=3, history_steps=2))
    assert batch[1]["current_state"].shape == (3, shared_data.rank)
    stats = fit_normalization(pod_only, EDMConfig(reduced_dim=shared_data.rank))
    full_stats = fit_normalization(shared_data, EDMConfig(reduced_dim=shared_data.n_coordinates))
    torch.testing.assert_close(stats.state_mean, full_stats.state_mean[1:])
    torch.testing.assert_close(stats.target_std, full_stats.target_std[1:])


def test_transition_halos_stride_and_shuffle_preserve_every_pair(shared_data):
    kwargs = dict(lag_steps=2, stride_steps=2, history_steps=2, batch_size=3)
    expected_ids, observed = set(), set()
    for name in shared_data.splits["train"]:
        _, full = shared_data.read_coordinates(name, 0, shared_data.frame_counts[name])
        expected_ids.update((name, i) for i in range(4, len(full) - 2, 2))
    for name, batch in shared_data.iter_transitions("train", seed=3, **kwargs):
        _, full = shared_data.read_coordinates(name, 0, shared_data.frame_counts[name])
        indices = batch["start_index"]
        assert all((name, int(i)) not in observed for i in indices)
        observed.update((name, int(i)) for i in indices)
        np.testing.assert_allclose(batch["current_state"], full[indices])
        np.testing.assert_allclose(batch["next_state"], full[indices + 2])
        np.testing.assert_allclose(batch["history_states"], full[indices[:, None] - [2, 4]])
    assert observed == expected_ids


def test_streaming_statistics_match_explicit_training_data(shared_data):
    cfg = EDMConfig(reduced_dim=3, lag_steps=2, stride_steps=2, history_conditioning=True, history_steps=2)
    stats = fit_normalization(shared_data, cfg)
    states, increments = [], []
    for name in shared_data.splits["train"]:
        _, values = shared_data.read_coordinates(name, 0, shared_data.frame_counts[name])
        states.append(values)
        indices = np.arange(4, len(values) - 2, 2)
        increments.append(values[indices + 2] - values[indices])
    for prefix, values in (("state", np.concatenate(states)), ("target", np.concatenate(increments))):
        np.testing.assert_allclose(getattr(stats, prefix + "_mean"), values.mean(0), rtol=1e-6)
        np.testing.assert_allclose(getattr(stats, prefix + "_std"), values.std(0), rtol=1e-6)
    # Changing held-out values cannot affect either training scaler.
    for split in ("validation", "test"):
        with h5py.File(shared_data.source_paths[shared_data.splits[split][0]], "r+") as f:
            f["reduced/coefficients"][:] = 1e10
    changed = fit_normalization(shared_data, cfg)
    for name in stats.to_dict():
        torch.testing.assert_close(getattr(stats, name), getattr(changed, name))
    future = fit_normalization(shared_data, replace(cfg, target_mode="future_state"))
    torch.testing.assert_close(future.state_mean, future.target_mean)
    batch = next(transition_batches(shared_data, cfg, "train", 3))
    assert batch["current_state"].dtype == torch.float32
    assert shared_data.mean is None  # No second preparation scaler was fitted.


def test_no_prepared_files_created_and_short_history_rejected(shared_data):
    directory = shared_data.basis_path.parents[2]
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}
    list(transition_batches(shared_data, EDMConfig(reduced_dim=3), "train", 4))
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}
    assert before == after
    with pytest.raises(ValueError, match="too short"):
        list(shared_data.iter_transitions("train", lag_steps=20, history_steps=2))
