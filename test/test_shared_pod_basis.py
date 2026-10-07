"""Check pooled POD against explicit snapshots, including truncated inputs."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from modelling.data_preparation.shared_pod_basis import (
    assemble_weighted_modes, compute_shared_basis, discover_repetitions,
    inspect_inputs, main,
)


class SharedPodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "0p20"
        self.snapshots = []
        rng = np.random.default_rng(87)
        for rep, n_frames in enumerate((17, 23), start=1):
            x = rng.normal(size=(6, n_frames)) * np.arange(1, 7)[:, None]
            x -= x.mean(axis=1, keepdims=True)
            self.snapshots.append(x)
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            path = self.directory / f"Ca_ac_0p001660_rep{rep}" / "pod_2d_r1000.h5"
            path.parent.mkdir(parents=True)
            with h5py.File(path, "w") as f:
                f.attrs.update(n_frames=n_frames, instantaneous_spatial_mean_removed=False,
                               temporal_mean_field_removed=True)
                for axis, values in (("x", np.arange(3)), ("y", np.arange(2))):
                    f.create_dataset(f"grid/{axis}", data=values).attrs["units"] = "mm"
                f.create_dataset("pod/modes", data=u.T.reshape(6, 2, 3))
                f.create_dataset("pod/singular_values", data=s)
                f["pod"].attrs.update(total_preprocessed_snapshot_energy=np.sum(x**2),
                                      spatial_inner_product="unweighted Euclidean sum over pixels")
                f.create_dataset("reduced/coefficients", data=vt.T * s).attrs["units"] = "um"
        self.paths = discover_repetitions(self.directory, "pod_2d_r1000.h5")

    def test_exact_matches_pooled_snapshots(self):
        records, _ = inspect_inputs(self.paths, None)
        matrix = assemble_weighted_modes(records, 6)
        u, s = compute_shared_basis(matrix, 4)
        expected_u, expected_s, _ = np.linalg.svd(np.concatenate(self.snapshots, axis=1),
                                                 full_matrices=False)
        np.testing.assert_allclose(s, expected_s[:4], rtol=1e-12)
        np.testing.assert_allclose(u @ u.T, expected_u[:, :4] @ expected_u[:, :4].T,
                                   atol=1e-12)

    def test_truncated_inputs_match_retained_reconstructions(self):
        records, _ = inspect_inputs(self.paths, 2)
        matrix = assemble_weighted_modes(records, 6)
        retained = []
        for x in self.snapshots:
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            retained.append((u[:, :2] * s[:2]) @ vt[:2])
        pooled = np.concatenate(retained, axis=1)
        np.testing.assert_allclose(matrix @ matrix.T, pooled @ pooled.T, atol=1e-11)

    def test_randomized_leading_subspace(self):
        rng = np.random.default_rng(42)
        left, _ = np.linalg.qr(rng.normal(size=(50, 20)))
        right, _ = np.linalg.qr(rng.normal(size=(35, 20)))
        spectrum = np.geomspace(100, 0.001, 20)
        matrix = (left * spectrum) @ right.T
        u, s = compute_shared_basis(matrix, 5, "randomized", 5, 2)
        np.testing.assert_allclose(s, spectrum[:5], rtol=1e-8)
        np.testing.assert_allclose(u @ u.T, left[:, :5] @ left[:, :5].T, atol=1e-6)

    def test_output_projection_and_energy(self):
        output = self.root / "shared.h5"
        main(["0p20", "--data-root", str(self.root), "--rank", "4", "--output", str(output)])
        with h5py.File(output, "r") as f:
            u = f["pod/modes"][:].reshape(4, 6).T
            for path, x in zip(self.paths, self.snapshots):
                projection = f[f"repetitions/{path.parent.name}/projection"][:]
                with h5py.File(path, "r") as source:
                    a = source["reduced/coefficients"][:]
                np.testing.assert_allclose(a @ projection.T, x.T @ u, atol=1e-12)
            pooled = np.concatenate(self.snapshots, axis=1)
            expected = np.cumsum(np.linalg.svd(pooled, compute_uv=False)[:4]**2) / np.sum(pooled**2)
            np.testing.assert_allclose(f["pod/cumulative_energy_fraction"][:], expected)
        with self.assertRaises(FileExistsError):
            main(["0p20", "--data-root", str(self.root), "--rank", "4", "--output", str(output)])

    def test_rejects_mismatched_preprocessing_and_grid(self):
        with h5py.File(self.paths[1], "r+") as f:
            f.attrs["temporal_mean_field_removed"] = False
        with self.assertRaisesRegex(ValueError, "mismatch"):
            inspect_inputs(self.paths, None)
        with h5py.File(self.paths[1], "r+") as f:
            f.attrs["temporal_mean_field_removed"] = True
            f["grid/x"][0] = -1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            inspect_inputs(self.paths, None)

    def test_missing_repetition_is_not_silently_skipped(self):
        self.paths[1].unlink()
        with self.assertRaises(FileNotFoundError):
            discover_repetitions(self.directory, "pod_2d_r1000.h5")

    def test_rejects_impossible_rank_and_zero_energy(self):
        with self.assertRaises(ValueError):
            inspect_inputs(self.paths, 7)
        with self.assertRaises(ValueError):
            compute_shared_basis(np.ones((3, 2)), 3)
        with self.assertRaises(ValueError):
            compute_shared_basis(np.zeros((3, 2)), 1)


if __name__ == "__main__":
    unittest.main()
