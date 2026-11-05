"""Shared read-only adapter for ACDM rollout.npz and its metrics.json metadata."""
from pathlib import Path
import json

import h5py
import numpy as np

from data_analysis.psd.pod_center_psd import infer_sampling_frequency
from modelling.data_preparation.prepare_shared_pod_training import (
    SPATIAL_MEAN_EMBEDDED,
    SPATIAL_MEAN_OMITTED,
    SPATIAL_MEAN_REPRESENTATIONS,
    SPATIAL_MEAN_SEPARATE,
)


class Rollout:
    def __init__(self, path, *, shared_basis=None, discard=0):
        self.path = Path(path).expanduser().resolve()
        if self.path.is_dir():
            self.path /= "rollout.npz"
        self.metadata = json.loads(self.path.with_name("metrics.json").read_text())
        with np.load(self.path, allow_pickle=False) as f:
            self.generated = f["trajectories"]
            self.reference = f["reference"]
            self.time = f["reference_time"]
            self.repetitions = f["repetition"].astype(str)
            self.start_index = (f["start_index"].astype(np.int64)
                                if "start_index" in f else None)
        g, r = self.generated, self.reference
        if (g.ndim != 4 or min(g.shape) < 1
                or r.shape != (g.shape[0], g.shape[2], g.shape[3])
                or self.time.shape != r.shape[:2] or self.repetitions.shape != (len(r),)):
            raise ValueError("Expected generated [condition, ensemble, time, coordinate] and matching references.")
        if self.start_index is not None and self.start_index.shape != (len(r),):
            raise ValueError("Rollout start_index must have one entry per condition.")
        if any(not np.isfinite(a).all() for a in (g, r, self.time)):
            raise ValueError("Rollout contains nonfinite values.")
        if not isinstance(discard, int) or not 0 <= discard <= g.shape[2] - 4:
            raise ValueError("Discard must leave at least four samples per trajectory.")
        self.rank = int(self.metadata["rank"])
        saved_include_spatial_mean = bool(self.metadata.get("include_spatial_mean", True))
        cutoff = self.metadata.get("spatial_mean_highpass_hz")
        self.spatial_mean_highpass_hz = None if cutoff is None else float(cutoff)
        if self.metadata["time_units"] not in ("s", "second", "seconds"):
            raise ValueError("Expected rollout times in seconds.")
        dt = float(self.metadata["physical_lag"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid physical sampling interval.")
        self.fs = 1 / dt
        self.lag_steps = int(self.metadata.get("lag_steps", 1))
        if self.lag_steps < 1:
            raise ValueError("Rollout lag_steps must be positive.")
        for t in self.time:
            fs, jitter = infer_sampling_frequency(t)
            if not np.isclose(fs, self.fs, rtol=1e-3) or jitter > .05:
                raise ValueError("Reference timestamps disagree with the model sampling interval.")
        self.generated = g[:, :, discard:]
        self.reference = r[:, discard:]
        self.time = self.time[:, discard:]
        if self.start_index is not None:
            self.start_index = self.start_index + discard * self.lag_steps
        self.discard = discard
        self.basis_path = Path(shared_basis or self.metadata["source_shared_basis"]).expanduser().resolve()
        with h5py.File(self.basis_path, "r") as basis:
            if len(basis["pod/modes"]) < self.rank:
                raise ValueError("Shared basis must have enough modes.")
            means_removed = bool(basis.attrs["instantaneous_spatial_mean_removed"])
            representation = self.metadata.get("spatial_mean_representation")
            if representation is None:
                representation = (
                    SPATIAL_MEAN_SEPARATE if saved_include_spatial_mean else
                    SPATIAL_MEAN_OMITTED if means_removed else
                    SPATIAL_MEAN_EMBEDDED)
            if representation not in SPATIAL_MEAN_REPRESENTATIONS:
                raise ValueError("Unknown rollout spatial_mean_representation.")
            if means_removed != (representation != SPATIAL_MEAN_EMBEDDED):
                raise ValueError(
                    "Rollout spatial-mean representation disagrees with the shared basis.")
            self.spatial_mean_representation = representation
            self.include_spatial_mean = representation == SPATIAL_MEAN_SEPARATE
            self.spatial_mean_available = representation != SPATIAL_MEAN_OMITTED
            self.pod_coordinate_offset = int(self.include_spatial_mean)
            if g.shape[-1] != self.rank + self.pod_coordinate_offset:
                raise ValueError(
                    "Metadata rank and spatial-mean representation do not match coordinate count.")
            if (self.spatial_mean_highpass_hz is not None
                    and representation != SPATIAL_MEAN_SEPARATE):
                raise ValueError(
                    "A saved spatial-mean high-pass cutoff requires a separate mean coordinate.")
            self.modes = np.asarray(basis["pod/modes"][:self.rank], dtype=float)
            self.x, self.y = basis["grid/x"][:], basis["grid/y"][:]
            self.x_units, self.y_units = str(basis["grid/x"].attrs["units"]), str(basis["grid/y"].attrs["units"])
            units = {str(basis[f"repetitions/{name}"].attrs["signal_units"]) for name in self.repetitions}
            if len(units) != 1:
                raise ValueError("Reference repetitions must share displacement units.")
            self.units = units.pop()
        if self.modes.shape[1:] != (len(self.y), len(self.x)) or not np.isfinite(self.modes).all():
            raise ValueError("Invalid shared spatial modes or grid shape.")
        self.n_space = len(self.x) * len(self.y)
        self.spatial_mean_weights = self.modes.reshape(self.rank, -1).mean(axis=1)

    def pod_coefficients(self, coordinates):
        values = np.asarray(coordinates)
        expected = self.rank + self.pod_coordinate_offset
        if values.shape[-1] != expected:
            raise ValueError(f"Expected {expected} rollout coordinates on the last axis.")
        return values[..., self.pod_coordinate_offset:]

    def spatial_mean(self, coordinates):
        """Return the physical frame spatial mean in rollout signal units."""
        values = np.asarray(coordinates)
        if self.spatial_mean_representation == SPATIAL_MEAN_SEPARATE:
            return values[..., 0] / np.sqrt(self.n_space)
        if self.spatial_mean_representation == SPATIAL_MEAN_EMBEDDED:
            return self.pod_coefficients(values) @ self.spatial_mean_weights
        raise ValueError("The spatial mean was omitted from this rollout.")

    def reconstruct(self, coordinates, modes=None):
        """Reconstruct rank-limited fields without restoring the temporal mean."""
        selected_modes = self.modes if modes is None else np.asarray(modes)
        fields = self.pod_coefficients(coordinates) @ selected_modes.reshape(self.rank, -1)
        fields = fields.reshape(*np.asarray(coordinates).shape[:-1], *selected_modes.shape[1:])
        if self.spatial_mean_representation == SPATIAL_MEAN_SEPARATE:
            fields = fields + self.spatial_mean(coordinates)[..., None, None]
        return fields

    def provenance(self):
        result = dict(rollout=str(self.path), shared_basis=str(self.basis_path), rank=self.rank,
                      conditions=len(self.reference), ensemble_size=self.generated.shape[1],
                      samples=self.generated.shape[2], discarded_samples=self.discard,
                      sampling_frequency_hz=self.fs, signal_units=self.units,
                      reference_repetitions=self.repetitions.tolist(),
                      coefficient_convention=("physical; coordinate 0 = sqrt(N) * spatial mean"
                                              if self.include_spatial_mean else
                                              "physical shared POD coefficients; spatial mean embedded"
                                              if self.spatial_mean_representation == SPATIAL_MEAN_EMBEDDED else
                                              "physical shared POD coefficients; spatial mean omitted"),
                      spatial_mean_representation=self.spatial_mean_representation,
                      spatial_mean_available=self.spatial_mean_available,
                      temporal_mean_restored=False)
        result["spatial_mean_highpass_hz"] = self.spatial_mean_highpass_hz
        if "temporal_preprocessing" in self.metadata:
            result["temporal_preprocessing"] = self.metadata["temporal_preprocessing"]
        if "reference_window_filter" in self.metadata:
            result["reference_window_filter"] = self.metadata["reference_window_filter"]
        if "reference_window_trim_policy" in self.metadata:
            result["reference_window_trim_policy"] = self.metadata["reference_window_trim_policy"]
        return result


def output_directory(rollout, name, output=None, overwrite=False):
    path = Path(output) if output is not None else rollout.path.parent / name
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Nonempty output directory {path}; use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_rollout_arguments(parser):
    parser.add_argument("rollout", type=Path, help="Rollout directory or rollout.npz (metrics.json alongside).")
    parser.add_argument("--shared-basis", type=Path, help="Override the saved basis path, e.g. after relocation.")
    parser.add_argument("--discard", type=int, default=0, help="Discard this many initial samples from generated and reference trajectories.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
