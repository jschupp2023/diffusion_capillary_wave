"""Frozen observed-coordinate oscillator drift plus the ordinary neural residual."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from .model import MLPDrift


class ImportedOscillatorResidual(nn.Module):
    def __init__(self, dimension, hidden_layers, activation, input_blocks, dt, residual=None,
                 forcing_filter="none"):
        super().__init__()
        self.dimension, self.dt = dimension, float(dt)
        self.forcing_filter = forcing_filter
        self.residual = residual if residual is not None else MLPDrift(
            dimension, hidden_layers, activation, input_blocks=input_blocks)
        self.residual.zero_output()
        # Physical operators use seconds and native coefficient units. Keep
        # double precision during the affine normalization conversion.
        for name in ('physical_stiffness', 'physical_damping'):
            self.register_buffer(name, torch.zeros(dimension, dimension, dtype=torch.float64))
        self.register_buffer('equilibrium', torch.zeros(dimension, dtype=torch.float64))
        for name in ('stiffness', 'damping'):
            self.register_buffer(name, torch.zeros(dimension, dimension))
        self.register_buffer('bias', torch.zeros(dimension))
        self.register_buffer('configured', torch.tensor(False))
        if forcing_filter == "modal_biquad":
            self.register_buffer('modal_transform', torch.zeros(
                dimension, dimension, dtype=torch.float64))
            self.register_buffer('modal_inverse', torch.zeros(
                dimension, dimension, dtype=torch.float64))
            for name in ('forcing_h0', 'forcing_h1', 'forcing_h2',
                         'forcing_d1', 'forcing_d2'):
                self.register_buffer(name, torch.zeros(dimension, dtype=torch.float64))
            self.register_buffer('forcing_filter_configured', torch.tensor(False))
        self.ready = False
        self.filter_ready = False

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.ready = bool(self.configured.item())
        self.filter_ready = bool(
            getattr(self, 'forcing_filter_configured', torch.tensor(False)).item())

    @torch.no_grad()
    def configure(self, stiffness, damping, equilibrium, model, modal_transform=None):
        for name, value in [('physical_stiffness', stiffness), ('physical_damping', damping),
                            ('equilibrium', equilibrium)]:
            buffer = getattr(self, name)
            value = torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device)
            if value.shape != buffer.shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid imported {name}')
            buffer.copy_(value)
        self.configured.fill_(True)
        self.ready = True
        if self.forcing_filter == "modal_biquad":
            transform = torch.as_tensor(
                modal_transform, dtype=self.modal_transform.dtype,
                device=self.modal_transform.device)
            if (transform.shape != self.modal_transform.shape
                    or not torch.isfinite(transform).all()):
                raise ValueError('Invalid imported oscillator modal transform')
            inverse = torch.linalg.inv(transform)
            self.modal_transform.copy_(transform)
            self.modal_inverse.copy_(inverse)
        self.refresh_normalization(model)

    @torch.no_grad()
    def configure_forcing_filter(self, h0, h1, h2, d1, d2):
        if self.forcing_filter != "modal_biquad" or not self.ready:
            raise RuntimeError('Modal forcing filter requires a configured oscillator')
        for name, value in (('forcing_h0', h0), ('forcing_h1', h1),
                            ('forcing_h2', h2), ('forcing_d1', d1),
                            ('forcing_d2', d2)):
            buffer = getattr(self, name)
            value = torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device)
            if (value.shape != buffer.shape or not torch.isfinite(value).all()
                    or (name == 'forcing_h0' and torch.any(value <= 0))):
                raise ValueError(f'Invalid {name}')
            buffer.copy_(value)
        self.forcing_filter_configured.fill_(True)
        self.filter_ready = True

    def _filter_buffers(self, value):
        dtype, device = value.dtype, value.device
        return (self.modal_transform.to(dtype=dtype, device=device),
                self.modal_inverse.to(dtype=dtype, device=device),
                self.forcing_h0.to(dtype=dtype, device=device),
                self.forcing_h1.to(dtype=dtype, device=device),
                self.forcing_h2.to(dtype=dtype, device=device),
                self.forcing_d1.to(dtype=dtype, device=device),
                self.forcing_d2.to(dtype=dtype, device=device))

    def forcing_to_modal(self, value):
        if not self.filter_ready:
            raise RuntimeError('Modal forcing filter is not configured')
        _, inverse, *_ = self._filter_buffers(value)
        return value @ inverse.T

    def forcing_from_modal(self, value):
        if not self.filter_ready:
            raise RuntimeError('Modal forcing filter is not configured')
        transform, *_ = self._filter_buffers(value)
        return value @ transform.T

    def filter_forcing(self, raw, raw_history, filtered_history):
        """Apply the fixed modal biquad; both histories are newest first."""
        if (raw_history.shape[-2:] != (2, self.dimension)
                or filtered_history.shape[-2:] != (2, self.dimension)):
            raise ValueError('forcing histories must contain two newest-first values')
        transform, inverse, h0, h1, h2, d1, d2 = self._filter_buffers(raw)
        modal = raw @ inverse.T
        history = raw_history @ inverse.T
        filtered_past = filtered_history @ inverse.T
        filtered = (h0 * modal + h1 * history[..., 0, :] + h2 * history[..., 1, :]
                    - d1 * filtered_past[..., 0, :] - d2 * filtered_past[..., 1, :])
        return filtered @ transform.T

    def inverse_filter_forcing(self, filtered, raw_history, filtered_history):
        """Recover the current raw forcing from filtered forcing and past raw values."""
        if (raw_history.shape[-2:] != (2, self.dimension)
                or filtered_history.shape[-2:] != (2, self.dimension)):
            raise ValueError('forcing histories must contain two newest-first values')
        transform, inverse, h0, h1, h2, d1, d2 = self._filter_buffers(filtered)
        modal = filtered @ inverse.T
        history = raw_history @ inverse.T
        filtered_past = filtered_history @ inverse.T
        raw_modal = ((modal + d1 * filtered_past[..., 0, :]
                      + d2 * filtered_past[..., 1, :]
                      - h1 * history[..., 0, :] - h2 * history[..., 1, :]) / h0)
        return raw_modal @ transform.T

    def filter_direct_matrix(self, reference):
        """Physical-coordinate direct map from current raw to filtered forcing."""
        transform, inverse, h0, *_ = self._filter_buffers(reference)
        return (transform * h0) @ inverse

    @torch.no_grad()
    def refresh_normalization(self, model):
        if not self.ready:
            return
        mean, sx, um, su = [getattr(model, k).double() for k in
                           ('state_mean', 'state_std', 'target_mean', 'target_std')]
        k, d = self.physical_stiffness.double(), self.physical_damping.double()
        # u = x[n]-x[n-1], not dx/dt. Network outputs Delta u / std(u).
        self.stiffness.copy_(self.dt**2 * k*sx[None, :]/su[:, None])
        self.damping.copy_(self.dt * d*su[None, :]/su[:, None])
        self.bias.copy_(-(self.dt**2*(k @ (mean-self.equilibrium.double()))
                          + self.dt*(d @ um))/su)

    def linear(self, normalized_input):
        if not self.ready:
            raise RuntimeError('Imported oscillator must be configured or loaded from a checkpoint')
        d = self.dimension
        return (-normalized_input[..., :d] @ self.stiffness.T
                - normalized_input[..., d:2*d] @ self.damping.T + self.bias)

    def forward(self, normalized_input, sequence_input=None, residual_output=None):
        residual = (self.residual(normalized_input if sequence_input is None else sequence_input)
                    if residual_output is None else residual_output)
        return self.linear(normalized_input)+residual

    def zero_output(self):
        """Diffusion warm-up zeros the neural residual, retaining the fixed prior."""
        self.residual.zero_output()


def configure_imported_oscillator(model, data, artifact):
    """Validate the peak-fit artifact lineage and convert to native coordinates."""
    artifact = Path(artifact).resolve()
    protocol_path = artifact.parent.parent/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    source = protocol['background_protocol']['source_protocol']['source_protocol']['source_provenance']
    single = source['single_rollout']
    if model.config.reduced_dim != data.n_coordinates:
        raise ValueError('Imported oscillator dimension differs from training coordinates')
    if Path(single['shared_basis']).resolve() != Path(data.basis_path).resolve():
        raise ValueError('Imported oscillator POD basis differs from training data')
    selection = protocol.get('coordinate_selection')
    source_rank = single['rank']
    if selection is not None:
        if (selection.get('kind') != 'leading_with_spatial_mean'
                or selection.get('source_dimension') != source_rank+1
                or selection.get('dimension') != data.n_coordinates
                or selection.get('pod_rank') != data.rank or data.rank > source_rank):
            raise ValueError('Invalid imported oscillator coordinate-selection provenance')
        source_rank = selection['pod_rank']
    if source_rank != data.rank or not data.include_spatial_mean:
        raise ValueError('Imported oscillator requires the same POD rank and spatial-mean coordinate')
    if single['signal_units'] != data.signal_units:
        raise ValueError('Imported oscillator coefficient units differ from training data')
    if single.get('spatial_mean_highpass_hz') != data.spatial_mean_highpass_hz:
        raise ValueError('Imported oscillator spatial-mean preprocessing differs from training data')
    if (getattr(data, 'downsample_factor', 1) != 1
            or not math.isclose(model.config.dt, data.native_dt, rel_tol=1e-5)
            or not math.isclose(model.config.dt, 1/single['sampling_frequency_hz'], rel_tol=1e-5)):
        raise ValueError('Imported oscillator training requires the matching native sampling interval')
    def numbers(names):
        return [int(re.search(r'rep(\d+)$', name).group(1)) for name in names]
    if (numbers(data.train_repetitions) != source['training_repetitions']
            or numbers(data.validation_repetitions) != source['validation_repetitions']):
        raise ValueError('Imported oscillator training/validation splits differ from the source fit')
    unit_match = re.fullmatch(r'all coefficients divided by sqrt\(N\)=([\d.]+); coordinate 0 is physical spatial mean',
                             source['coordinate_units'])
    if unit_match is None or not math.isclose(float(unit_match.group(1)), math.sqrt(data.n_space)):
        raise ValueError('Unrecognized spectral-cache coefficient scaling or spatial grid size')
    with np.load(artifact, allow_pickle=False) as saved:
        if 'resonant_spectrum' not in saved.files:
            raise ValueError('Expected a peak-background-fit artifact')
        arrays = {k:np.array(saved[k], dtype=float) for k in
                  ('T','K','D','omega_per_second','gamma_per_second','mean','scale','prescribed_peaks_hz')}
        fit_fs = float(saved['fs'])
    d = model.config.reduced_dim
    t, omega, gamma = arrays['T'], arrays['omega_per_second'], arrays['gamma_per_second']
    if (any(not np.isfinite(v).all() for v in arrays.values()) or t.shape != (d,d)
            or any(arrays[k].shape != (d,d) for k in ('K','D'))
            or any(arrays[k].shape != (d,) for k in ('omega_per_second','gamma_per_second','mean','scale'))
            or np.any(arrays['scale'] <= 0) or np.any(omega <= 0) or np.any(gamma <= 0)):
        raise ValueError('Invalid imported oscillator shapes or coefficients')
    if not math.isclose(fit_fs, single['sampling_frequency_hz']/source['downsample_factor'],rel_tol=1e-5):
        raise ValueError('Imported oscillator sampling provenance is inconsistent')
    modes = len(arrays['prescribed_peaks_hz'])
    if not 0 < modes <= d or len(protocol['nominal_peaks_hz']) != modes:
        raise ValueError('Invalid resonant/completion direction count')
    if not np.allclose(arrays['prescribed_peaks_hz'], protocol['nominal_peaks_hz']):
        raise ValueError('Imported oscillator pole order differs from provenance')
    inverse = np.linalg.solve(t,np.eye(d))
    if (not np.allclose((t*omega**2) @ inverse, arrays['K'], rtol=1e-6,atol=1e-5)
            or not np.allclose((t*(2*gamma)) @ inverse, arrays['D'],rtol=1e-6,atol=1e-5)):
        raise ValueError('Saved oscillator operators disagree with modal parameters')
    active = np.arange(d) < modes
    k = (t*(omega**2*active)) @ inverse
    damping = (t*(2*gamma*active)) @ inverse
    scale = math.sqrt(data.n_space)*arrays['scale']
    equilibrium = math.sqrt(data.n_space)*arrays['mean']
    physical_k = scale[:,None]*k/scale[None,:]
    physical_d = scale[:,None]*damping/scale[None,:]
    modal_transform = scale[:, None] * t
    model.drift_model.configure(
        physical_k, physical_d, equilibrium, model,
        modal_transform=modal_transform if model.config.forcing_filter == 'modal_biquad' else None)
    return dict(source_artifact=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                basis_path=str(Path(data.basis_path).resolve()), coordinate_selection=selection, resonant_directions=modes,
                excluded_completion_directions=d-modes, source_condition_number=float(np.linalg.cond(t)),
                native_dt=model.config.dt, source_fs=fit_fs, cache_to_native_factor=math.sqrt(data.n_space),
                source_position_mean=arrays['mean'].tolist(),source_position_std=arrays['scale'].tolist(),
                omega_per_second=omega[:modes].tolist(),gamma_per_second=gamma[:modes].tolist(),
                formula='Delta u = -dt^2 K_physical (x-equilibrium) - dt D_physical u; affine normalized conversion includes position/velocity offsets',
                fixed=True, background_and_spectral_forcing_imported=False,
                caveat='Prepared downsampled spectral operators used as a native-rate prior; any excluded completion is neutral, residual stability and peak preservation are not guaranteed')


def configure_modal_forcing_filter(model, artifact, oscillator_report):
    """Load a fixed training-only fitted modal biquad forcing filter."""
    if model.config.forcing_filter != 'modal_biquad':
        raise ValueError('forcing-filter artifact requires forcing_filter modal_biquad')
    artifact = Path(artifact).resolve()
    with np.load(artifact, allow_pickle=False) as saved:
        required = {'h0', 'h1', 'h2', 'd1', 'd2', 'dt_seconds', 'dimension',
                    'oscillator_artifact_sha256'}
        if not required.issubset(saved.files):
            raise ValueError('Incomplete modal forcing-filter artifact')
        h0 = np.asarray(saved['h0'], dtype=float)
        h1 = np.asarray(saved['h1'], dtype=float)
        h2 = np.asarray(saved['h2'], dtype=float)
        d1 = np.asarray(saved['d1'], dtype=float)
        d2 = np.asarray(saved['d2'], dtype=float)
        dt = float(saved['dt_seconds'])
        dimension = int(saved['dimension'])
        source_hash = str(saved['oscillator_artifact_sha256'])
    if (dimension != model.config.reduced_dim
            or not math.isclose(dt, model.config.dt, rel_tol=1e-10)
            or source_hash != oscillator_report['artifact_sha256']):
        raise ValueError('Forcing-filter artifact differs from oscillator dimension, dt, or lineage')
    model.drift_model.configure_forcing_filter(h0, h1, h2, d1, d2)
    return dict(
        type='modal_biquad', source_artifact=str(artifact),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        oscillator_artifact_sha256=source_hash,
        burnin=model.config.forcing_filter_burnin,
        direct_gain=h0.tolist(), numerator_lag1=h1.tolist(),
        numerator_lag2=h2.tolist(), denominator_lag1=d1.tolist(),
        denominator_lag2=d2.tolist(),
        fixed=True,
        definition=('D(z)e_modal = H(z)raw_modal, fixed second-order notch; '
                    'the TCN and diffusion model raw forcing'))


@torch.no_grad()
def prepare_modal_forcing_cache(data, model):
    """Precompute causal inverse-filtered forcing for train/validation records.

    Cache entries use the current-position index: entry ``t`` is the forcing in
    the transition from ``x[t]`` to ``x[t+1]``.  Nothing is written to HDF5.
    """
    if not model.has_forcing_filter or not model.drift_model.filter_ready:
        raise RuntimeError('Configured modal forcing filter required before caching')
    cache = {}
    burnin = model.config.forcing_filter_burnin
    d = model.config.reduced_dim
    k = model.drift_model.physical_stiffness.detach().cpu().numpy()
    damping = model.drift_model.physical_damping.detach().cpu().numpy()
    equilibrium = model.drift_model.equilibrium.detach().cpu().numpy()
    transform = model.drift_model.modal_transform.detach().cpu().numpy()
    inverse = model.drift_model.modal_inverse.detach().cpu().numpy()
    h0 = model.drift_model.forcing_h0.detach().cpu().numpy()
    h1 = model.drift_model.forcing_h1.detach().cpu().numpy()
    h2 = model.drift_model.forcing_h2.detach().cpu().numpy()
    d1 = model.drift_model.forcing_d1.detach().cpu().numpy()
    d2 = model.drift_model.forcing_d2.detach().cpu().numpy()
    modal_rms = []
    reconstruction_errors = []
    retained = total = 0
    raw_sum = np.zeros(d, dtype=np.float64)
    raw_square_sum = np.zeros(d, dtype=np.float64)
    raw_count = 0
    for name in data.train_repetitions + data.validation_repetitions:
        values = np.asarray(data.read_coordinates(name, 0, data.frame_counts[name])[1],
                            dtype=np.float64)
        velocity = np.diff(values, axis=0)
        core = (-model.config.dt**2 * (values[1:-1] - equilibrium) @ k.T
                -model.config.dt * velocity[:-1] @ damping.T)
        filtered = velocity[1:] - velocity[:-1] - core
        filtered_modal = filtered @ inverse.T
        raw_modal = np.empty_like(filtered_modal)
        previous1 = np.zeros(d, dtype=np.float64)
        previous2 = np.zeros(d, dtype=np.float64)
        for index, current in enumerate(filtered_modal):
            filtered1 = filtered_modal[index - 1] if index >= 1 else 0.
            filtered2 = filtered_modal[index - 2] if index >= 2 else 0.
            raw_modal[index] = ((current + d1 * filtered1 + d2 * filtered2
                                - h1 * previous1 - h2 * previous2) / h0)
            previous2, previous1 = previous1, raw_modal[index]
        raw = raw_modal @ transform.T
        recovered_modal = (h0 * raw_modal
                           + h1 * np.vstack((np.zeros((1, d)), raw_modal[:-1]))
                           + h2 * np.vstack((np.zeros((2, d)), raw_modal[:-2]))
                           - d1 * np.vstack((np.zeros((1, d)), filtered_modal[:-1]))
                           - d2 * np.vstack((np.zeros((2, d)), filtered_modal[:-2])))
        reconstruction_errors.append(float(
            np.linalg.norm(recovered_modal - filtered_modal)
            / max(np.linalg.norm(filtered_modal), np.finfo(float).tiny)))
        full = np.full((len(values), d), np.nan, dtype=np.float64)
        full[1:-1] = raw
        # The inverse recursion begins from zero.  Exclude its transient and
        # the final point, which has no following observed velocity.
        full[:max(1, burnin)] = np.nan
        full.setflags(write=False)
        cache[name] = full
        valid = np.isfinite(full).all(axis=1)
        retained += int(valid.sum())
        total += len(valid)
        if name in data.train_repetitions and valid.any():
            modal_rms.append(np.sqrt(np.mean((full[valid] @ inverse.T) ** 2, axis=0)))
            raw_sum += full[valid].sum(axis=0)
            raw_square_sum += np.square(full[valid]).sum(axis=0)
            raw_count += int(valid.sum())
    data._modal_forcing_cache = cache
    raw_mean = raw_sum / raw_count
    raw_variance = np.maximum(raw_square_sum / raw_count - raw_mean**2,
                              np.finfo(np.float64).tiny)
    raw_std = np.sqrt(raw_variance)
    model.set_forcing_normalization(raw_mean, raw_std)
    rms = np.sqrt(np.mean(np.square(modal_rms), axis=0)) if modal_rms else np.zeros(d)
    return dict(
        cached_repetitions=list(cache), burnin=burnin,
        retained_count=retained, total_count=total,
        retained_fraction=retained / total,
        inverse_reconstruction_relative_error=max(reconstruction_errors),
        training_raw_mean=raw_mean.tolist(),
        training_raw_std=raw_std.tolist(),
        training_raw_modal_rms=rms.tolist(),
        definition=('cache[t] is inverse-filtered physical forcing for the '
                    'transition x[t] to x[t+1]; initialization transient excluded'))
