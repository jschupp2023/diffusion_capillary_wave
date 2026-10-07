"""Small shared-POD experiment with independent, unequal-length repetitions."""
import h5py
import numpy as np
import pytest

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData


@pytest.fixture
def shared_data(tmp_path):
    rng = np.random.default_rng(41)
    root = tmp_path / "reduced"
    directory = root / "0p20"
    path = directory / "shared_pod/shared_pod_r3.h5"
    path.parent.mkdir(parents=True)
    q, _ = np.linalg.qr(np.column_stack((np.ones(9), rng.normal(size=(9, 3)))))
    u = q[:, 1:]
    with h5py.File(path, "w") as basis:
        basis.attrs.update(experiment="0p20", instantaneous_spatial_mean_removed=True,
                           temporal_mean_field_removed=True)
        basis.create_dataset("pod/modes", data=u.T.reshape(3, 3, 3))
        for axis, grid in (("x", np.arange(3)), ("y", np.arange(3))):
            basis.create_dataset(f"grid/{axis}", data=grid).attrs["units"] = "mm"
        for rep, length in enumerate((19, 23, 17, 21), 1):
            name = f"Ca_ac_0p001660_rep{rep}"
            source_path = directory / name / "pod_2d_r3.h5"
            source_path.parent.mkdir()
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            local = u @ rotation
            a = rng.normal(size=(length, 3)).cumsum(axis=0) + rep * 100
            with h5py.File(source_path, "w") as source:
                source.attrs.update(n_frames=length, instantaneous_spatial_mean_removed=True,
                                    temporal_mean_field_removed=True)
                for axis in ("x", "y"):
                    basis.copy(f"grid/{axis}", source.require_group("grid"), name=axis)
                source.create_dataset("grid/time", data=2 + np.arange(length) * .01).attrs["units"] = "seconds"
                source.create_dataset("pod/modes", data=local.T.reshape(3, 3, 3))
                source.create_dataset("reduced/coefficients", data=a).attrs["units"] = "microns"
                source.create_dataset("preprocessing/frame_spatial_mean", data=rep + np.arange(length)**2 * .01).attrs["units"] = "microns"
                source.create_dataset("preprocessing/temporal_mean_field", data=np.zeros((3, 3))).attrs["units"] = "microns"
            group = basis.create_group(f"repetitions/{name}")
            group.attrs.update(source_pod_file=str(source_path), input_rank=3)
            group.create_dataset("projection", data=u.T @ local)
    return SharedPODTrainingData("0p20", rank=2, data_root=root, batch_size=5, output_dtype="float64")


@pytest.fixture
def highpass_shared_data(shared_data):
    cutoff_hz = 20.0
    dataset_path = (
        "preprocessing/frame_spatial_mean_highpass/cutoff_20_hz"
    )
    for path in shared_data.source_paths.values():
        with h5py.File(path, "r+") as source:
            raw = np.asarray(source["preprocessing/frame_spatial_mean"], dtype=np.float64)
            # Distinct deterministic values make selection of the stored signal explicit.
            filtered = 0.25 * (raw - raw.mean())
            dataset = source.create_dataset(dataset_path, data=filtered)
            dataset.attrs.update(
                units="microns",
                source_dataset="/preprocessing/frame_spatial_mean",
                filter_type="highpass",
                cutoff_hz=cutoff_hz,
            )
    return SharedPODTrainingData(
        **shared_data.source_config,
        spatial_mean_highpass_hz=cutoff_hz,
        batch_size=5,
        output_dtype="float64",
    )
