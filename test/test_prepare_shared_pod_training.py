"""Validate read-only shared POD preparation and training-only batch scaling."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from modelling.data_preparation.prepare_shared_pod_training import (
    RunningMoments, SharedPODTrainingData, prepare_training_data,
)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / "0p20"
        shared_path = folder / "shared_pod/shared_pod_r2.h5"
        shared_path.parent.mkdir(parents=True)
        rng = np.random.default_rng(76)
        # Orthonormal modes whose spatial mean is zero.
        candidates = rng.normal(size=(6, 3))
        candidates[:, 0] = 1
        q, _ = np.linalg.qr(candidates)
        shared = q[:, 1:3]
        self.expected = {}
        self.frames = {}
        with h5py.File(shared_path, "w") as basis:
            basis.attrs.update(experiment="0p20", instantaneous_spatial_mean_removed=True,
                               temporal_mean_field_removed=True)
            basis.create_dataset("pod/modes", data=shared.T.reshape(2, 2, 3))
            basis.create_dataset("pod/singular_values", data=[10, 5])
            for axis, values in (("x", np.arange(3)), ("y", np.arange(2))):
                basis.create_dataset(f"grid/{axis}", data=values).attrs["units"] = "mm"
            for rep, n_frames in enumerate((7, 11, 5), start=1):
                name = f"Ca_ac_0p001660_rep{rep}"
                path = folder / name / "pod_2d_r2.h5"
                path.parent.mkdir()
                rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
                local = shared @ rotation
                coefficients = rng.normal(size=(n_frames, 2)) + (100 if rep == 3 else rep)
                spatial_mean = rng.normal(size=n_frames) + (200 if rep == 3 else rep)
                temporal_mean = shared[:, 0] * 2
                self.frames[name] = coefficients @ local.T + spatial_mean[:, None] + temporal_mean
                self.expected[name] = np.column_stack((np.sqrt(6) * spatial_mean,
                                                      coefficients @ local.T @ shared))
                with h5py.File(path, "w") as source:
                    source.attrs.update(n_frames=n_frames, instantaneous_spatial_mean_removed=True,
                                        temporal_mean_field_removed=True)
                    for axis in ("x", "y"):
                        basis.copy(f"grid/{axis}", source.require_group("grid"), name=axis)
                    source.create_dataset("grid/time", data=np.arange(n_frames) * 0.01).attrs["units"] = "seconds"
                    source.create_dataset("pod/modes", data=local.T.reshape(2, 2, 3))
                    source.create_dataset("reduced/coefficients", data=coefficients).attrs["units"] = "microns"
                    source.create_dataset("preprocessing/frame_spatial_mean", data=spatial_mean).attrs["units"] = "microns"
                    source.create_dataset("preprocessing/temporal_mean_field", data=temporal_mean.reshape(2, 3)).attrs["units"] = "microns"
                group = basis.create_group(f"repetitions/{name}")
                # A relocated bundle must resolve files in --data-root.
                group.attrs.update(source_pod_file=f"/old/machine/{name}/{path.name}", input_rank=2)
                group.create_dataset("projection", data=shared.T @ local)
        self.options = dict(data_root=self.root, batch_size=3, output_dtype="float64", test_reps=[])

    def test_projection_scaler_loader_and_reconstruction(self):
        data = prepare_training_data("0p20", rank=2, **self.options)
        names = sorted(self.expected)
        self.assertEqual(data.train_repetitions, tuple(names[:2]))
        self.assertEqual(data.validation_repetitions, tuple(names[2:]))
        train = np.concatenate([self.expected[n] for n in names[:2]])
        np.testing.assert_allclose(data.mean, train.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(data.std, train.std(axis=0), atol=1e-12)
        self.assertEqual(data.training_frames, 18)
        with h5py.File(data.basis_path) as basis:
            spatial = np.vstack((np.ones(6) / np.sqrt(6), basis["pod/modes"][:].reshape(2, 6)))
        for split in ("train", "validation"):
            physical = {}
            for name, _, values in data.iter_batches(split):
                physical.setdefault(name, []).append(data.inverse_transform(values))
            for name, chunks in physical.items():
                coefficients = np.concatenate(chunks)
                np.testing.assert_allclose(coefficients, self.expected[name], atol=1e-12)
                with h5py.File(data.source_paths[name]) as source:
                    temporal_mean = source["preprocessing/temporal_mean_field"][:].ravel()
                np.testing.assert_allclose(coefficients @ spatial + temporal_mean, self.frames[name], atol=1e-12)
        batches = list(data.iter_batches(batch_size=4))
        standardized = np.concatenate([values for _, _, values in batches])
        np.testing.assert_allclose(standardized.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(standardized.std(axis=0), 1, atol=1e-12)
        for _, times, values in batches:
            self.assertLessEqual(len(times), 4)
            self.assertEqual(len(times), len(values))
            self.assertTrue(np.all(np.diff(times) > 0))
        validation = np.concatenate([v for _, _, v in data.iter_batches("validation")])
        self.assertGreater(abs(validation[:, 0].mean()), 10)

    def test_no_files_written_and_source_projection_reused(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        original_file = h5py.File
        original_getitem = h5py.Dataset.__getitem__

        def readonly_file(name, mode="r", *args, **kwargs):
            self.assertEqual(mode, "r", "Preparation must never open a file for writing.")
            return original_file(name, mode, *args, **kwargs)

        def no_local_modes(dataset, key):
            if dataset.name == "/pod/modes" and "shared_pod" not in dataset.file.filename:
                self.fail("Local modes must not be read: use the already-saved projections.")
            return original_getitem(dataset, key)

        with patch("modelling.data_preparation.prepare_shared_pod_training.h5py.File", side_effect=readonly_file), \
                patch.object(h5py.Dataset, "__getitem__", no_local_modes):
            data = prepare_training_data("0p20", rank=2, **self.options)
            for split in ("train", "validation"):
                list(data.iter_batches(split))
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_epochs_repeat_without_refitting_scaler(self):
        data = prepare_training_data("0p20", rank=2, **self.options)
        with patch.object(data, "fit_standardization", side_effect=AssertionError("Unexpected refit")):
            first = list(data.iter_batches())
            second = list(data.iter_batches())
        for (name1, t1, a1), (name2, t2, a2) in zip(first, second):
            self.assertEqual(name1, name2)
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(a1, a2)

    def test_explicit_holdout_and_higher_rank_basis(self):
        data = prepare_training_data("0p20", rank=1, validation_reps=[1], **self.options)
        self.assertEqual(data.n_coordinates, 2)
        self.assertEqual(data.validation_repetitions, (sorted(self.expected)[0],))
        expected = np.concatenate([self.expected[n][:, :2] for n in sorted(self.expected)[1:]])
        np.testing.assert_allclose(data.mean, expected.mean(axis=0), atol=1e-12)

    def test_rejects_unknown_or_all_validation_repetitions(self):
        for reps in ([99], [1, 2, 3], [1, 1]):
            with self.assertRaises(ValueError):
                prepare_training_data("0p20", rank=2, validation_reps=reps, **self.options)

    def test_unfitted_physical_batches_and_empty_validation(self):
        data = SharedPODTrainingData("0p20", rank=2, validation_reps=[], **self.options)
        self.assertIsNone(data.mean)
        self.assertEqual(data.validation_repetitions, ())
        with self.assertRaises(RuntimeError):
            next(data.iter_batches())
        physical = np.concatenate([v for _, _, v in data.iter_batches(standardized=False)])
        np.testing.assert_allclose(physical, np.concatenate(list(self.expected.values())), atol=1e-12)
        self.assertEqual(list(data.iter_batches("validation", standardized=False)), [])
        data.fit_standardization()
        self.assertEqual(data.training_frames, 23)

    def test_output_dtype_and_invalid_batch_options(self):
        data = prepare_training_data("0p20", rank=2, data_root=self.root)
        self.assertEqual(next(data.iter_batches())[2].dtype, np.float32)
        with self.assertRaises(ValueError):
            list(data.iter_batches("wrong"))
        with self.assertRaises(ValueError):
            list(data.iter_batches(batch_size=0))
        with self.assertRaises(ValueError):
            data.inverse_transform(np.zeros((3, 4)))

    def test_bad_projection_and_nonfinite_source_fail(self):
        basis_path = self.root / "0p20/shared_pod/shared_pod_r2.h5"
        with h5py.File(basis_path, "r+") as basis:
            name = sorted(self.expected)[0]
            basis[f"repetitions/{name}/projection"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite projection"):
            SharedPODTrainingData("0p20", rank=2, **self.options)

    def test_constant_coordinate_and_large_offset_statistics(self):
        values = np.column_stack((np.ones(17) * 5, 1e9 + np.arange(17) * 0.5))
        moments = RunningMoments(2)
        for block in np.array_split(values, 4):
            moments.update(block)
        mean, std, scale, constant = moments.finish()
        np.testing.assert_allclose(mean, values.mean(axis=0))
        np.testing.assert_allclose(std, values.std(axis=0), atol=1e-12)
        np.testing.assert_array_equal(constant, [True, False])
        self.assertEqual(scale[0], 1)


if __name__ == "__main__":
    unittest.main()
