"""Anti-aliasing, physical time, batch boundaries, and checkpoint round trips."""
import h5py
import numpy as np
import pytest
import torch
from scipy.signal import resample_poly

from data_analysis.rollout import Rollout
from data_analysis.plot_rollout import main as plot_rollout
from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.neural_sde.train import load_model, main


@pytest.fixture
def long_data(shared_data):
    with h5py.File(shared_data.basis_path, "r+") as f:
        for group in f["repetitions"].values():
            group.attrs["signal_units"] = "microns"
    for rep, path in enumerate(shared_data.source_paths.values()):
        count = 2051 + rep * 17
        t = np.arange(count) * .01
        with h5py.File(path, "r+") as f:
            f.attrs["n_frames"] = count
            arrays = {"grid/time": (t, "seconds"),
                      "reduced/coefficients": (np.column_stack([
                          np.sin(2 * np.pi * 2 * t), np.cos(2 * np.pi * 3 * t),
                          np.sin(2 * np.pi * 40 * t)]) + rep, "microns"),
                      "preprocessing/frame_spatial_mean": (
                          rep + np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 40 * t), "microns")}
            for name, (values, units) in arrays.items():
                del f[name]
                f.create_dataset(name, data=values).attrs["units"] = units
    return SharedPODTrainingData(**shared_data.source_config)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_chunk_cache_and_time_equivalent_to_whole_record(long_data, factor):
    native = long_data
    coarse = SharedPODTrainingData(**native.source_config, downsample_factor=factor)
    assert coarse.native_dt == pytest.approx(native.native_dt * factor)
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in native.source_paths.values()}
    for name in native.frame_counts:
        time, values = native.read_coordinates(name, 0, native.frame_counts[name])
        expected = resample_poly(values, 1, factor, axis=0)[32:-32]
        expected_time = time[::factor][32:-32]
        observed = list(coarse.read_coordinates(name, i, min(i + 7, len(expected)))
                        for i in range(0, len(expected), 7))
        np.testing.assert_allclose(np.concatenate([x[1] for x in observed]), expected, atol=1e-12)
        np.testing.assert_array_equal(np.concatenate([x[0] for x in observed]), expected_time)
        np.testing.assert_allclose(coarse.read_coordinates(name, 0, len(expected))[1], expected, atol=1e-12)
    coarse.fit_standardization()
    train = np.concatenate([coarse.read_coordinates(n, 0, coarse.frame_counts[n])[1]
                            for n in coarse.train_repetitions])
    np.testing.assert_allclose(coarse.mean, train.mean(0), atol=1e-12)
    np.testing.assert_allclose(coarse.std, train.std(0), atol=1e-12)
    assert coarse.training_frames == len(train)
    assert coarse.cache_coordinates(max_gib=.01)["enabled"]
    for name in coarse.train_repetitions:
        cached = coarse.read_coordinates(name, 0, coarse.frame_counts[name])[1].copy()
        coarse.cache_coordinates(0)
        np.testing.assert_allclose(coarse.read_coordinates(name, 0, len(cached))[1], cached, atol=1e-12)
    restored = SharedPODTrainingData(**coarse.source_config)
    assert restored.signature() == coarse.signature()
    assert native.signature() != coarse.signature()
    assert before == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in native.source_paths.values()}


def test_alias_rejection_and_low_frequency_retention(long_data):
    data = SharedPODTrainingData(**long_data.source_config, downsample_factor=4)
    name = data.train_repetitions[0]
    t, a = data.read_coordinates(name, 0, data.frame_counts[name])
    # 40 Hz would alias to 10 Hz at 25 Hz sampling. Fit both frequencies to
    # distinguish suppression from changes in mean/variance or leakage.
    design = np.column_stack([np.ones(len(t)), np.sin(2*np.pi*2*t), np.cos(2*np.pi*2*t),
                              np.sin(2*np.pi*10*t), np.cos(2*np.pi*10*t)])
    weights = np.linalg.lstsq(design, a[:, 0] / np.sqrt(data.n_space), rcond=None)[0]
    assert np.linalg.norm(weights[1:3]) == pytest.approx(1., abs=.003)
    assert np.linalg.norm(weights[3:5]) < .003


def test_transitions_and_training_only_statistics(long_data):
    data = SharedPODTrainingData(**long_data.source_config, downsample_factor=8)
    name, batch = next(data.iter_transitions("train", history_steps=7, batch_size=9))
    _, full = data.read_coordinates(name, 0, data.frame_counts[name])
    i = batch["start_index"]
    np.testing.assert_allclose(batch["history_states"], full[i[:, None] - np.arange(1, 8)])
    np.testing.assert_allclose(batch["next_state"] - batch["current_state"], full[i+1] - full[i])
    data.fit_standardization()
    previous = data.mean.copy(), data.std.copy()
    for name in data.validation_repetitions + data.test_repetitions:
        with h5py.File(data.source_paths[name], "r+") as f:
            f["reduced/coefficients"][:] = 1e10
            f["preprocessing/frame_spatial_mean"][:] = -1e10
    changed = SharedPODTrainingData(**data.source_config).fit_standardization()
    np.testing.assert_array_equal(changed.mean, previous[0])
    np.testing.assert_array_equal(changed.std, previous[1])


def test_tcn_cli_checkpoint_rollout_and_warm_start(long_data, tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path / "model"
        main(["train", "--experiment", long_data.experiment, "--rank", "2",
              "--data-root", long_data.source_config["data_root"], "--downsample-factor", "8",
              "--state-variable", "velocity", "--residual-architecture", "tcn",
              "--tcn-context-frames", "7", "--tcn-channels", "4", "--tcn-dilations", "1", "2",
              "--epochs", "1", "--batch-size", "64", "--train-trim-percent", "0",
              "--device", "cpu", "--output", str(output)])
        model, payload = load_model(output / "best.pt")
        data = checkpoint_data(payload)
        assert model.config.dt == pytest.approx(.08)
        assert model.config.lag_steps == 1
        assert data.downsample_factor == 8
        evaluation = tmp_path / "evaluation"
        main(["evaluate", "--checkpoint", str(output / "best.pt"), "--split", "validation",
              "--num-conditions", "2", "--ensemble-size", "2", "--horizon", "25",
              "--device", "cpu", "--output", str(evaluation)])
        rollout = Rollout(evaluation)
        assert rollout.fs == pytest.approx(12.5)
        assert rollout.metadata["temporal_preprocessing"]["raw_start_index"] == 256
        for name, start, reference, time in zip(rollout.repetitions, rollout.start_index,
                                                rollout.reference, rollout.time):
            expected_t, expected = data.read_coordinates(str(name), int(start), int(start) + len(time))
            np.testing.assert_allclose(reference, expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(time, expected_t)
        plot_rollout([str(evaluation), "--nperseg", "8", "--peak-nperseg", "8",
                      "--peak-band", ".5", "5", "--psd-lower-frequency", ".5"])
        assert (evaluation / "quantitative" / "quick_summary.md").is_file()
        warm = ["train", "--checkpoint", str(output), "--epochs", "1", "--batch-size", "64",
                "--device", "cpu", "--train-trim-percent", "0", "--output", str(tmp_path / "warm")]
        with pytest.raises(SystemExit):
            main(warm + ["--downsample-factor", "4"])
        main(warm + ["--multistep-weight", ".1", "--multistep-horizons", "1", "2",
                     "--multistep-particles", "3"])
        _, continued = load_model(tmp_path / "warm" / "best.pt")
        assert continued["data_config"]["downsample_factor"] == 8
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("kwargs", [{"downsample_factor": 0}, {"downsample_factor": 1.5},
                                    {"downsample_factor": True},
                                    {"downsample_factor": 8, "downsample_edge_trim": 2}])
def test_invalid_preparation(shared_data, kwargs):
    with pytest.raises(ValueError, match="downsample"):
        SharedPODTrainingData(**shared_data.source_config, **kwargs)


def test_short_record_rejected(shared_data):
    with pytest.raises(ValueError, match="too short"):
        SharedPODTrainingData(**shared_data.source_config, downsample_factor=8)
