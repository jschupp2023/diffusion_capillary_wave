"""Prepare shared-POD coordinates and transitions in memory; never save data.

SharedPODTrainingData(experiment, rank) owns train/validation/test splits.
By default the last 15% of repetitions are test and the preceding 15% are
validation (rounded, at least one each). Explicit validation_reps/test_reps
choose holdouts; test_reps=[] disables test. At least one training rep is
required. Splits exclude scaler/model fitting, not prior shared-basis fitting.

By default coordinate zero is sqrt(N)*frame_spatial_mean, for the unit constant
mode. Set spatial_mean_highpass_hz to select a high-pass trajectory previously
stored under preprocessing/frame_spatial_mean_highpass instead. Set
include_spatial_mean=False to return only the first `rank` shared POD
coefficients. These project all retained local modes using saved basis-to-basis
projections. Removed temporal mean fields remain removed. The model dimension
is rank+1 by default or rank when the spatial mean is excluded.

read_coordinates and iter_transitions return physical float64 coordinates.
Transitions preserve history and never cross repetitions. cache_coordinates
optionally keeps train/validation coordinates in RAM within a size budget;
otherwise data are projected in bounded blocks. No files are written.

For consumers outside ACDM, prepare_training_data fits a training-only scaler
and iter_batches returns standardized coordinates (float32 by default).
ACDM reads physical transitions and applies its own normalization once.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import h5py
import numpy as np

from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT, PREPROCESSING_FLAGS


def select_basis(directory: Path, rank: int, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = []
    for path in (directory / "shared_pod").glob("shared_pod_r*.h5"):
        match = re.fullmatch(r"shared_pod_r(\d+)\.h5", path.name)
        if match and int(match[1]) >= rank:
            candidates.append((int(match[1]), path))
    if not candidates:
        raise FileNotFoundError(f"No shared basis with at least {rank} modes in {directory / 'shared_pod'}.")
    return min(candidates)[1].resolve()


def spatial_mean_dataset_path(cutoff_hz):
    if cutoff_hz is None:
        return "preprocessing/frame_spatial_mean"
    tag = f"{float(cutoff_hz):.12g}".replace(".", "p").replace("+", "")
    return f"preprocessing/frame_spatial_mean_highpass/cutoff_{tag}_hz"


def inspect_sources(basis, directory, experiment, rank, validation_reps,
                    spatial_mean_dataset, spatial_mean_highpass_hz):
    if str(basis.attrs["experiment"]) != experiment:
        raise ValueError("Shared basis belongs to a different experiment.")
    if not bool(basis.attrs["instantaneous_spatial_mean_removed"]):
        raise ValueError("Shared POD must remove instantaneous spatial means before adding a constant mode.")
    modes = basis["pod/modes"]
    if modes.ndim != 3 or not 1 <= rank <= len(modes):
        raise ValueError("Requested rank exceeds the saved shared basis or modes have invalid dimensions.")
    shape = modes.shape[1:]
    if shape != (len(basis["grid/y"]), len(basis["grid/x"])):
        raise ValueError("Shared mode shape does not match its spatial grid.")
    records, units = [], None
    seen = set()
    for name in sorted(basis["repetitions"], key=lambda s: int(s.rsplit("_rep", 1)[1])):
        group = basis[f"repetitions/{name}"]
        rep = int(name.rsplit("_rep", 1)[1])
        if rep in seen:
            raise ValueError("Duplicate repetition number in shared basis.")
        seen.add(rep)
        recorded_path = Path(str(group.attrs["source_pod_file"]))
        # Prefer the requested data tree so bundles can be moved between machines.
        local_path = directory / name / recorded_path.name
        path = local_path if local_path.is_file() else recorded_path
        with h5py.File(path, "r") as source:
            for flag in PREPROCESSING_FLAGS:
                if bool(source.attrs[flag]) != bool(basis.attrs[flag]):
                    raise ValueError(f"Preprocessing mismatch in {path}: {flag}.")
            for axis in ("x", "y"):
                if (not np.array_equal(source[f"grid/{axis}"][:], basis[f"grid/{axis}"][:])
                        or source[f"grid/{axis}"].attrs["units"] != basis[f"grid/{axis}"].attrs["units"]):
                    raise ValueError(f"Spatial grid or units mismatch in {path}.")
            local_rank = int(group.attrs["input_rank"])
            n_frames = int(source.attrs["n_frames"])
            coefficients = source["reduced/coefficients"]
            if (n_frames < 1 or coefficients.ndim != 2 or coefficients.shape[0] != n_frames
                    or coefficients.shape[1] < local_rank
                    or source["pod/modes"].shape[1:] != shape
                    or source["pod/modes"].shape[0] < local_rank
                    or spatial_mean_dataset not in source
                    or source[spatial_mean_dataset].shape != (n_frames,)
                    or source["preprocessing/temporal_mean_field"].shape != shape
                    or source["grid/time"].shape != (n_frames,)):
                raise ValueError(f"Invalid source dimensions in {path}.")
            signal_units = str(coefficients.attrs["units"])
            if (str(source[spatial_mean_dataset].attrs.get("units", "")) != signal_units
                    or str(source["preprocessing/temporal_mean_field"].attrs["units"]) != signal_units):
                raise ValueError(f"Spatial mean and POD coefficients have different units in {path}.")
            if spatial_mean_highpass_hz is not None:
                selected = source[spatial_mean_dataset]
                saved_cutoff = selected.attrs.get("cutoff_hz")
                if (saved_cutoff is None
                        or not np.isclose(float(saved_cutoff), spatial_mean_highpass_hz,
                                          rtol=1e-12, atol=0)
                        or str(selected.attrs.get("source_dataset", ""))
                        != "/preprocessing/frame_spatial_mean"
                        or str(selected.attrs.get("filter_type", "")) != "highpass"):
                    raise ValueError(
                        f"High-pass spatial-mean metadata do not match "
                        f"{spatial_mean_highpass_hz:g} Hz in {path}."
                    )
            this_units = (signal_units, str(source["grid/time"].attrs["units"]))
            if units is not None and units != this_units:
                raise ValueError("Repetitions have different signal/time units.")
            units = this_units
            times = source["grid/time"][:]
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"Timestamps must be finite and strictly increasing in {path}.")
        records.append(dict(name=name, path=path.resolve(), rep=rep, n_frames=n_frames,
                            local_rank=local_rank, split="validation" if rep in validation_reps else "train"))
    if not records or not any(r["split"] == "train" for r in records):
        raise ValueError("At least one training repetition is required.")
    if set(validation_reps) - seen:
        raise ValueError(f"Unknown validation repetitions: {sorted(set(validation_reps) - seen)}.")
    return records, units


def project_blocks(source, projection, batch_size, n_space, include_spatial_mean=True,
                   spatial_mean_dataset="preprocessing/frame_spatial_mean"):
    """Stream physical coordinates without constructing spatial snapshots."""
    coefficients = source["reduced/coefficients"]
    for start in range(0, len(coefficients), batch_size):
        stop = min(start + batch_size, len(coefficients))
        offset = int(include_spatial_mean)
        block = np.empty((stop - start, len(projection) + offset), dtype=np.float64)
        if include_spatial_mean:
            block[:, 0] = np.sqrt(n_space) * source[spatial_mean_dataset][start:stop]
        block[:, offset:] = np.asarray(coefficients[start:stop, :projection.shape[1]], dtype=np.float64) @ projection.T
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite projected data in {source.filename}, rows {start}:{stop}.")
        yield start, stop, block


class RunningMoments:
    """Merge batch-centered moments without cancellation from large offsets."""

    def __init__(self, n_coordinates):
        self.count = 0
        self.mean = np.zeros(n_coordinates, dtype=np.float64)
        self.m2 = np.zeros(n_coordinates, dtype=np.float64)

    def update(self, values):
        count = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        m2 = np.einsum("ij,ij->j", centered, centered)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta**2 * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def finish(self):
        if not self.count:
            raise ValueError("Cannot fit standardization without training samples.")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0))
        # Treat variation at roundoff relative to a large mean as constant.
        constant = std <= 10 * np.finfo(np.float64).eps * np.maximum(1, np.abs(self.mean))
        scale = np.where(constant, 1.0, std)
        return self.mean, std, scale, constant


class SharedPODTrainingData:
    """Read-only source-backed batch preparation for one experiment.

    Construction validates input metadata and loads the saved basis-to-basis
    projections, but does not scan coefficient histories. Call
    fit_standardization() once before requesting standardized batches.
    Source POD files must be the same decompositions used to build the shared
    basis; regenerated modes require regenerating that basis and projections.
    Files are opened within each iterator, with no persistent HDF5 handles.
    """

    def __init__(self, experiment, rank, *, data_root=DEFAULT_DATA_ROOT,
                 shared_basis=None, validation_reps=None, test_reps=None, batch_size=2048,
                 output_dtype="float32", include_spatial_mean=True,
                 spatial_mean_highpass_hz=None):
        if not isinstance(rank, (int, np.integer)) or rank < 1:
            raise ValueError("Rank must be a positive integer.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
        if not experiment or Path(experiment).name != experiment or experiment in (".", ".."):
            raise ValueError("Experiment must be a single folder name.")
        self.output_dtype = np.dtype(output_dtype)
        if self.output_dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Output dtype must be float32 or float64.")
        self.experiment = experiment
        self.rank = int(rank)
        self.include_spatial_mean = bool(include_spatial_mean)
        if spatial_mean_highpass_hz is not None:
            spatial_mean_highpass_hz = float(spatial_mean_highpass_hz)
            if not np.isfinite(spatial_mean_highpass_hz) or spatial_mean_highpass_hz <= 0:
                raise ValueError("spatial_mean_highpass_hz must be finite and positive.")
            if not self.include_spatial_mean:
                raise ValueError("A spatial-mean high-pass cutoff requires include_spatial_mean=True.")
        self.spatial_mean_highpass_hz = spatial_mean_highpass_hz
        self.spatial_mean_dataset = spatial_mean_dataset_path(spatial_mean_highpass_hz)
        self.n_coordinates = self.rank + int(self.include_spatial_mean)
        self.batch_size = int(batch_size)
        directory = Path(data_root).expanduser() / experiment
        self.basis_path = select_basis(directory, self.rank,
                                       Path(shared_basis) if shared_basis is not None else None)
        with h5py.File(self.basis_path, "r") as basis:
            numbers = sorted(int(name.rsplit("_rep", 1)[1]) for name in basis["repetitions"])
            count = max(1, round(.15 * len(numbers)))
            if test_reps is None:
                test_reps = [n for n in numbers if n not in (validation_reps or [])][-count:]
            if validation_reps is None:
                validation_reps = [n for n in numbers if n not in test_reps][-count:]
            for selected in (validation_reps, test_reps):
                if (any(not isinstance(rep, (int, np.integer)) for rep in selected)
                        or len(set(selected)) != len(selected) or set(selected) - set(numbers)):
                    raise ValueError("Holdout repetitions must be unique known integer numbers.")
            if set(validation_reps) & set(test_reps):
                raise ValueError("Validation and test repetitions must be disjoint.")
            records, units = inspect_sources(
                basis, directory, experiment, self.rank, validation_reps,
                self.spatial_mean_dataset, self.spatial_mean_highpass_hz,
            )
            for record in records:
                if record["rep"] in test_reps:
                    record["split"] = "test"
            if not any(r["split"] == "train" for r in records):
                raise ValueError("At least one training repetition is required.")
            self._records = tuple(records)
            self.signal_units, self.time_units = units
            self.frame_shape = basis["pod/modes"].shape[1:]
            self.n_space = int(np.prod(self.frame_shape))
            self.temporal_mean_field_removed = bool(basis.attrs["temporal_mean_field_removed"])
            self.shared_basis_includes_validation = bool(validation_reps)
            # Check the constant direction in small blocks without keeping the spatial basis.
            self.constant_mode_overlap = np.empty(self.rank)
            for start in range(0, self.rank, 64):
                stop = min(start + 64, self.rank)
                modes = np.asarray(basis["pod/modes"][start:stop], dtype=np.float64).reshape(stop - start, -1)
                if not np.isfinite(modes).all():
                    raise ValueError("Nonfinite shared spatial modes.")
                self.constant_mode_overlap[start:stop] = modes.sum(axis=1) / np.sqrt(self.n_space)
            if np.max(np.abs(self.constant_mode_overlap)) > 1e-3:
                raise ValueError("Shared modes contain a significant constant component; cannot append an independent mean mode.")
            self._projections = {}
            for record in self._records:
                group = basis[f"repetitions/{record['name']}"]
                if "projection" not in group:
                    raise ValueError(f"Missing saved projection for {record['name']}; regenerate the shared basis file.")
                saved = group["projection"]
                if saved.ndim != 2 or saved.shape != (len(basis["pod/modes"]), record["local_rank"]):
                    raise ValueError(f"Invalid projection dimensions for {record['name']}.")
                projection = np.asarray(saved[:self.rank], dtype=np.float64)
                if not np.isfinite(projection).all():
                    raise ValueError(f"Nonfinite projection for {record['name']}.")
                projection.setflags(write=False)
                self._projections[record["name"]] = projection
        self.train_repetitions = tuple(r["name"] for r in self._records if r["split"] == "train")
        self.validation_repetitions = tuple(r["name"] for r in self._records if r["split"] == "validation")
        self.test_repetitions = tuple(r["name"] for r in self._records if r["split"] == "test")
        self.splits = {s: getattr(self, s + "_repetitions") for s in ("train", "validation", "test")}
        self.frame_counts = {r["name"]: r["n_frames"] for r in self._records}
        self.source_paths = {r["name"]: r["path"] for r in self._records}
        self.mean = self.std = self.scale = self.constant = None
        self.training_frames = 0
        self._coordinate_cache = {}
        self.source_config = dict(experiment=experiment, rank=self.rank,
                                  data_root=str(directory.parent.resolve()), shared_basis=str(self.basis_path),
                                  validation_reps=list(map(int, validation_reps)), test_reps=list(map(int, test_reps)))
        if not self.include_spatial_mean:
            self.source_config["include_spatial_mean"] = False
        if self.spatial_mean_highpass_hz is not None:
            self.source_config["spatial_mean_highpass_hz"] = self.spatial_mean_highpass_hz
        intervals = []
        for path in self.source_paths.values():
            with h5py.File(path, "r") as source:
                raw_time = source["grid/time"][:]
            time = raw_time.astype(np.float64)
            if len(time) < 2:
                raise ValueError("Trajectories need at least two timestamps.")
            dt = float(np.mean(np.diff(time)))
            # Source timestamps may have been computed in float32, then stored
            # in float64. Allow that rounding, bounded by 5% of one interval.
            rounding = min(.05 * dt, 4 * np.finfo(np.float32).eps * np.max(np.abs(time)))
            if np.max(np.abs(np.diff(time) - dt)) > max(dt * 1e-4, rounding):
                raise ValueError("Trajectories must be uniformly sampled.")
            intervals.append(dt)
        self.native_dt = intervals[0]
        if not np.allclose(intervals, self.native_dt, rtol=1e-4, atol=0):
            raise ValueError("Repetitions must have the same sampling interval.")

    def read_coordinates(self, name, start, stop):
        """Return timestamps and physical float64 coordinates for one interval."""
        if name not in self.frame_counts or not 0 <= start < stop <= self.frame_counts[name]:
            raise ValueError("Invalid repetition or coordinate interval.")
        if name in self._coordinate_cache:
            time, values = self._coordinate_cache[name]
            return time[start:stop], values[start:stop]
        with h5py.File(self.source_paths[name], "r") as source:
            projection = self._projections[name]
            offset = int(self.include_spatial_mean)
            values = np.empty((stop - start, self.n_coordinates), dtype=np.float64)
            if self.include_spatial_mean:
                values[:, 0] = np.sqrt(self.n_space) * source[self.spatial_mean_dataset][start:stop]
            values[:, offset:] = np.asarray(source["reduced/coefficients"][start:stop, :projection.shape[1]],
                                       dtype=np.float64) @ projection.T
            time = source["grid/time"][start:stop]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite projected data in {name}.")
        return time, values

    def cache_coordinates(self, max_gib=2.):
        """Cache train/validation in RAM if the full cache fits; never write data.

        Float64 preserves small physical increments before normalization. The
        budget limits the cache, not total process memory. Zero clears it.
        """
        if not np.isfinite(max_gib) or max_gib < 0:
            raise ValueError("Cache budget must be finite and nonnegative.")
        self._coordinate_cache.clear()
        names = self.train_repetitions + self.validation_repetitions
        required = sum(self.frame_counts[n] for n in names) * (self.n_coordinates + 1) * 8
        budget = max_gib * 2**30
        try:
            available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))
            budget = min(budget, available * .25)
        except (OSError, StopIteration):
            pass
        if required > budget:
            return dict(enabled=False, required_gib=required / 2**30, cached_gib=0.)
        try:
            for name in names:
                time = np.empty(self.frame_counts[name], dtype=np.float64)
                values = np.empty((len(time), self.n_coordinates), dtype=np.float64)
                for start in range(0, len(time), self.batch_size):
                    stop = min(start + self.batch_size, len(time))
                    time[start:stop], values[start:stop] = self.read_coordinates(name, start, stop)
                time.setflags(write=False)
                values.setflags(write=False)
                self._coordinate_cache[name] = time, values
        except MemoryError:
            self._coordinate_cache.clear()
        return dict(enabled=bool(self._coordinate_cache), required_gib=required / 2**30,
                    cached_gib=sum(t.nbytes + v.nbytes for t, v in self._coordinate_cache.values()) / 2**30)

    def iter_transitions(self, split="train", *, lag_steps=1, stride_steps=1,
                         history_steps=0, history_spacing_steps=None,
                         include_next_history=False, include_state_path=False,
                         batch_size=256, seed=None):
        history_spacing = lag_steps if history_spacing_steps is None else history_spacing_steps
        if (split not in self.splits or min(lag_steps, stride_steps, history_spacing, batch_size) < 1
                or history_steps < 0 or (include_next_history and not history_steps)):
            raise ValueError("Invalid split, lag, stride, history or batch size.")
        first = history_steps * history_spacing
        width = min(batch_size, max(1, self.batch_size // stride_steps))
        jobs = []
        for name in self.splits[split]:
            last = self.frame_counts[name] - lag_steps
            if first >= last:
                raise ValueError(f"Repetition {name} is too short for this history and lag.")
            jobs.extend((name, start, min(start + width * stride_steps, last))
                        for start in range(first, last, width * stride_steps))
        rng = np.random.default_rng(seed)
        if seed is not None:
            rng.shuffle(jobs)
        for name, start, stop in jobs:
            indices = np.arange(start, stop, stride_steps)
            offset = start - first
            time, values = self.read_coordinates(name, offset, int(indices[-1]) + lag_steps + 1)
            if seed is not None:
                rng.shuffle(indices)
            local = indices - offset
            batch = dict(current_state=values[local], next_state=values[local + lag_steps],
                         start_index=indices, time=time[local])
            if history_steps:
                offsets = history_spacing * np.arange(1, history_steps + 1)
                batch["history_states"] = values[local[:, None] - offsets]
                if include_next_history:
                    batch["next_history_states"] = values[local[:, None] + lag_steps - offsets]
            if include_state_path:
                path_offsets = np.arange(-first, lag_steps + 1)
                batch["state_path"] = values[local[:, None] + path_offsets]
            yield name, batch

    def signature(self):
        """Portable source identity; coefficient contents are guarded by size/mtime."""
        digest = hashlib.sha256()
        with h5py.File(self.basis_path, "r") as basis:
            for start in range(0, self.rank, 64):
                digest.update(np.asarray(basis["pod/modes"][start:min(start + 64, self.rank)], dtype=np.float64).tobytes())
        sources = {}
        for record in self._records:
            name, path = record["name"], record["path"]
            digest.update(self._projections[name].tobytes())
            stat = path.stat()
            with h5py.File(path, "r") as source:
                time_hash = hashlib.sha256(source["grid/time"][:].tobytes()).hexdigest()
            sources[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                 frames=record["n_frames"], input_rank=record["local_rank"], time_sha256=time_hash)
        signature = dict(basis_sha256=digest.hexdigest(), rank=self.rank, n_space=self.n_space,
                    frame_shape=list(self.frame_shape), native_dt=self.native_dt,
                    units=[self.signal_units, self.time_units], splits={k: list(v) for k, v in self.splits.items()},
                    temporal_mean_removed=self.temporal_mean_field_removed,
                    coordinate_order=("sqrt(N)*spatial_mean, shared POD coefficients" if self.include_spatial_mean
                                      else "shared POD coefficients only"), sources=sources)
        if not self.include_spatial_mean:
            signature["include_spatial_mean"] = False
        if self.spatial_mean_highpass_hz is not None:
            signature["spatial_mean_highpass_hz"] = self.spatial_mean_highpass_hz
            signature["spatial_mean_dataset"] = self.spatial_mean_dataset
        return signature

    def _iter_physical_batches(self, split, batch_size):
        if split not in self.splits:
            raise ValueError("Split must be 'train', 'validation' or 'test'.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
        for record in self._records:
            if record["split"] != split:
                continue
            if record["name"] in self._coordinate_cache:
                for start in range(0, record["n_frames"], batch_size):
                    time, values = self.read_coordinates(record["name"], start, min(start + batch_size, record["n_frames"]))
                    yield record["name"], time, values
                continue
            with h5py.File(record["path"], "r") as source:
                for start, stop, block in project_blocks(
                        source, self._projections[record["name"]], batch_size, self.n_space,
                        self.include_spatial_mean, self.spatial_mean_dataset):
                    yield record["name"], source["grid/time"][start:stop], block

    def fit_standardization(self):
        """Fit training-only mean/std in one bounded-memory pass; return self.

        All epochs reuse these arrays. Calling this method again explicitly
        refits them; iter_batches never refits or uses validation to fit.
        """
        moments = RunningMoments(self.n_coordinates)
        for _, _, values in self._iter_physical_batches("train", self.batch_size):
            moments.update(values)
        self.mean, self.std, self.scale, self.constant = moments.finish()
        self.training_frames = moments.count
        return self

    def iter_batches(self, split="train", *, batch_size=None, standardized=True):
        """Yield (repetition_name, timestamps, coordinates) in temporal order.

        Make a new iterator each epoch. Each batch belongs to one repetition.
        standardized=False exposes physical coefficients without fitting a
        scaler. Scaling is applied in float64 before the output dtype cast.
        """
        if standardized and self.mean is None:
            raise RuntimeError("Call fit_standardization() before requesting standardized batches.")
        size = self.batch_size if batch_size is None else batch_size
        for name, time, values in self._iter_physical_batches(split, size):
            if standardized:
                values = (values - self.mean) / self.scale
            yield name, time, values.astype(self.output_dtype, copy=False)

    def inverse_transform(self, coordinates):
        """Undo standardization and return physical coefficients as float64."""
        if self.mean is None:
            raise RuntimeError("Call fit_standardization() before inverse_transform().")
        values = np.asarray(coordinates, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != self.n_coordinates:
            raise ValueError(f"Expected {self.n_coordinates} coordinates on the last axis.")
        return values * self.scale + self.mean


def prepare_training_data(experiment, rank, *, data_root=DEFAULT_DATA_ROOT,
                          shared_basis=None, validation_reps=None, test_reps=None,
                          batch_size=2048, output_dtype="float32", include_spatial_mean=True,
                          spatial_mean_highpass_hz=None):
    """Return a fitted SharedPODTrainingData for use across training epochs.

    One streaming training pass computes scaling in memory. Subsequent calls
    to the returned object's iter_batches() read and transform source batches
    directly. This function never saves coefficients, scaling, or other data.
    """
    return SharedPODTrainingData(
        experiment, rank, data_root=data_root, shared_basis=shared_basis,
        validation_reps=validation_reps, test_reps=test_reps, batch_size=batch_size,
        output_dtype=output_dtype, include_spatial_mean=include_spatial_mean,
        spatial_mean_highpass_hz=spatial_mean_highpass_hz,
    ).fit_standardization()
