"""Offline temporal components in the unchanged shared spatial POD basis.

Cache only derived NPY arrays. Source HDF5 files are never modified. Filtering
is per complete recording, before edge exclusion and before window sampling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from .prepare_shared_pod_training import SharedPODTrainingData


DEFAULT_BANDS = ((255., 300.), (425., 480.), (620., 680.))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def design_filters(fs, bands=DEFAULT_BANDS, order=4):
    if not np.isfinite(fs) or fs <= 0 or not isinstance(order, int) or order < 1:
        raise ValueError('Positive sampling frequency and integer order required')
    bands = np.asarray(bands, dtype=float)
    if (bands.ndim != 2 or bands.shape[1] != 2 or len(bands) == 0
            or not np.isfinite(bands).all() or np.any(bands[:, 0] <= 0)
            or np.any(bands[:, 1] >= fs / 2)
            or np.any(bands[:, 0] >= bands[:, 1])
            or np.any(bands[1:, 0] < bands[:-1, 1])):
        raise ValueError('Ordered, separated positive bands below Nyquist required')
    return [butter(order, band, btype='bandpass', fs=fs, output='sos') for band in bands]


def settling_samples(sos, fs, tolerance=1e-6):
    """Conservative guard from the single-pass impulse-response tail energy."""
    impulse = np.zeros(int(np.ceil(4 * fs)))
    impulse[0] = 1
    response = sosfilt(sos, impulse)
    tail = np.cumsum(response[::-1] ** 2)[::-1]
    indices = np.flatnonzero(tail <= tolerance * tail[0])
    if not len(indices):
        raise ValueError('Filter has not settled within four seconds')
    return max(1, int(indices[0]))


def decompose(values, filters, edge):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or edge < 1:
        raise ValueError('Finite [time,coordinate] data and positive edge required')
    if len(values) <= 2 * edge + 128:
        raise ValueError('Filter guard leaves insufficient usable trajectory')
    bands = [sosfiltfilt(sos, values, axis=0, padlen=edge) for sos in filters]
    remainder = values - sum(bands)
    sl = slice(edge, -edge)
    result = {'full': values[sl], 'remainder': remainder[sl]}
    result.update({f'band{i}': band[sl] for i, band in enumerate(bands)})
    reconstructed = result['remainder'] + sum(result[f'band{i}'] for i in range(len(bands)))
    error = np.linalg.norm(reconstructed - result['full']) / max(np.linalg.norm(result['full']), 1e-30)
    if error > 1e-12:
        raise RuntimeError('Component reconstruction failed')
    return result, float(error)


def prepare(data, output, bands=DEFAULT_BANDS, order=4, edge_seconds=.1,
            include_test=False, store_bands=True):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite component cache: {output}')
    output.mkdir(parents=True, exist_ok=True)
    fs = 1 / data.native_dt
    filters = design_filters(fs, bands, order)
    edge = max(int(np.ceil(edge_seconds * fs)),
               *(settling_samples(sos, fs) for sos in filters))
    manifest = dict(version=1, parent_config=data.source_config,
                    parent_signature=data.signature(), bands_hz=np.asarray(bands).tolist(),
                    filter_order=order, sos=[s.tolist() for s in filters],
                    edge_samples=edge, fs=fs, filter_method='sosfiltfilt',
                    initialization='offline; not a causal forecast',
                    stored_band_arrays=bool(store_bands), records={})
    names = data.train_repetitions + data.validation_repetitions
    if include_test:
        names += data.test_repetitions
    for name in names:
        time, values = data.read_coordinates(name, 0, data.frame_counts[name])
        parts, error = decompose(values, filters, edge)
        # Padding sensitivity uses a distinct, longer extension, same interior.
        alternate_pad = min(2 * edge, len(values) - 1)
        sensitivity = []
        for i, sos in enumerate(filters):
            alternate = sosfiltfilt(sos, values, axis=0, padlen=alternate_pad)[edge:-edge]
            sensitivity.append(float(np.linalg.norm(alternate-parts[f'band{i}']) /
                                     max(np.linalg.norm(parts[f'band{i}']), 1e-30)))
        files = {}
        stored = {'time': time[edge:-edge], 'full': parts['full'],
                  'remainder': parts['remainder']}
        if store_bands:
            stored.update({key:value for key,value in parts.items() if key.startswith('band')})
        for component, array in stored.items():
            path = output / f'{name}_{component}.npy'
            np.save(path, array, allow_pickle=False)
            files[component] = dict(path=path.name, sha256=sha256(path))
        manifest['records'][name] = dict(frames=len(parts['full']), files=files,
            reconstruction_relative_error=error, padding_sensitivity_relative=sensitivity)
        print(f'Prepared {name}: {len(parts["full"])} frames, error={error:.2g}', flush=True)
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


class FrequencyComponentData(SharedPODTrainingData):
    """Source-compatible adapter with immutable, separately scaled components."""
    def __init__(self, *, temporal_component_manifest, temporal_component, **kwargs):
        super().__init__(**kwargs)
        self.manifest_path = Path(temporal_component_manifest).resolve()
        self.component = temporal_component
        manifest = json.loads(self.manifest_path.read_text())
        if manifest['version'] != 1 or manifest['parent_signature'] != super().signature():
            raise ValueError('Component cache parent signature does not match source data')
        self.manifest = manifest
        self.component_edge = int(manifest['edge_samples'])
        self._component_files = {}
        for name, record in manifest['records'].items():
            if self.component not in record['files']:
                raise ValueError(f'Unknown temporal component: {self.component}')
            entry = record['files'][self.component]
            time_entry = record['files']['time']
            path, time_path = (self.manifest_path.parent / entry['path'],
                               self.manifest_path.parent / time_entry['path'])
            if sha256(path) != entry['sha256'] or sha256(time_path) != time_entry['sha256']:
                raise ValueError('Component cache checksum mismatch')
            time, values = np.load(time_path, mmap_mode='r'), np.load(path, mmap_mode='r')
            if values.shape != (record['frames'], self.n_coordinates) or len(time) != len(values):
                raise ValueError('Invalid cached component dimensions')
            self._coordinate_cache[name] = (time, values)
            self._component_files[name] = (time_path, path)
            self.frame_counts[name] = len(time)
        for name in self.train_repetitions + self.validation_repetitions:
            if name not in self._component_files:
                raise ValueError(f'Missing required component recording: {name}')
        self.source_config.update(temporal_component_manifest=str(self.manifest_path),
                                  temporal_component=self.component)

    def read_coordinates(self, name, start, stop):
        if name not in self._component_files:
            raise ValueError('Recording was not prepared; test data require explicit preparation')
        return super().read_coordinates(name, start, stop)

    def cache_coordinates(self, max_gib=2.):
        # Arrays are already read-only memory maps. Never replace with raw data.
        return dict(enabled=True, memory_mapped=True, cached_gib=0.)

    def _iter_physical_batches(self, split, batch_size):
        if split not in self.splits or batch_size < 1:
            raise ValueError('Invalid split or batch size')
        for name in self.splits[split]:
            for start in range(0, self.frame_counts[name], batch_size):
                time, values = self.read_coordinates(name, start,
                    min(start + batch_size, self.frame_counts[name]))
                yield name, time, values

    def signature(self):
        return dict(**super().signature(), temporal_component=dict(
            manifest_sha256=sha256(self.manifest_path), component=self.component,
            prepared_frame_counts={name: self.frame_counts[name] for name in self._component_files}))

    def temporal_preprocessing(self):
        result = super().temporal_preprocessing()
        result.update(temporal_component=self.component,
            component_manifest=str(self.manifest_path), bands_hz=self.manifest['bands_hz'],
            component_edge_samples=self.component_edge,
            component_start_time_offset_seconds=self.component_edge*self.native_dt,
            raw_start_index=result['raw_start_index']+self.component_edge*self.downsample_factor,
            method='offline_frequency_component', time_alignment='offline; not causal')
        return result
