"""Experimental frequency-resolved spatial modes; no model fitting or HDF5 writes.

Run ``python -m data_analysis.pod.spectral_peak_atlas --help``. Physical shared
POD coordinates are never standardized. Convention: S_ij = E[X_i conj(X_j)],
so a coefficient eigenvector v reconstructs as phi @ v (not phi @ conj(v)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.signal import find_peaks, get_window, welch

from data_analysis.pod.complex_band_modes import _load_shared_spatial_modes
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.data_preparation.shared_pod_basis import DEFAULT_DATA_ROOT


def eigensystem(matrix):
    """Descending Hermitian eigensystem with deterministic global phases."""
    matrix = np.asarray(matrix, dtype=complex)
    matrix = (matrix + matrix.conj().T) / 2
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    for j in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, j]))
        vectors[:, j] *= np.exp(-1j * np.angle(vectors[pivot, j]))
    return values, vectors


def similarity(first, second, count=1):
    """Mean squared principal-angle cosines; k=1 is phase-invariant MAC."""
    a, _ = np.linalg.qr(first[:, :count])
    b, _ = np.linalg.qr(second[:, :count])
    return float(np.clip(np.linalg.norm(a.conj().T @ b, "fro")**2 / count, 0, 1))


def capture(matrix, vectors, count=1):
    total = np.trace(matrix).real
    if total <= 0:
        return 0.
    v = vectors[:, :count]
    return float(np.clip(np.trace(v.conj().T @ matrix @ v).real / total, 0, 1))


def metric_eigensystem(matrix, gram):
    """SPOD for physical field H a when G=H^H H is not exactly identity.

    Returns coefficient vectors with V^H G V=I and their physical orthonormal
    coordinates G^(1/2) V. Eigenvalues sum to trace(G S).
    """
    values, basis = np.linalg.eigh(gram)
    if values.min() <= 0:
        raise ValueError('Spatial Gram matrix must be positive definite')
    root = (basis*np.sqrt(values)) @ basis.conj().T
    eigenvalues, orthogonal = eigensystem(root @ matrix @ root)
    return eigenvalues, np.linalg.solve(root, orthogonal), orthogonal


def select_sites(frequency, training_psd, *, low=100., high=1100.,
                 prominence_db=1., spacing_hz=40., per_channel=3,
                 valley_count=3, half_width_hz=10.):
    """Select observed peaks in mean/POD1, then valleys in deformation trace.

    Family labels are intentionally not assigned. All candidate peaks, not just
    displayed peaks, constrain band widths. No validation data enter selection.
    """
    f = np.asarray(frequency)
    df = f[1] - f[0]
    candidates = []
    selected = {}
    for channel in (0, 1):
        power = 10 * np.log10(np.maximum(training_psd[:, channel], 1e-300))
        indices, props = find_peaks(power, prominence=prominence_db,
                                   distance=max(1, int(np.ceil(spacing_hz / df))))
        valid = (f[indices] >= low) & (f[indices] <= high)
        indices, prominences = indices[valid], props['prominences'][valid]
        if len(indices) < per_channel:
            raise ValueError(f"Only {len(indices)} peaks in channel {channel}; "
                             "reduce --peaks-per-channel or adjust detection settings")
        for index, prominence in zip(indices, prominences):
            candidates.append(dict(bin=int(index), frequency_hz=float(f[index]),
                                   channel=channel, prominence_db=float(prominence)))
        targets = np.linspace(f[indices[0]], f[indices[-1]], per_channel)
        for index in indices[[np.argmin(abs(f[indices]-target)) for target in targets]]:
            selected.setdefault(int(index), []).append(channel)
    all_bins = np.unique([c['bin'] for c in candidates])

    def site(index, kind, channels):
        other = all_bins[all_bins != index]
        nearest = np.min(np.abs(f[other] - f[index])) if len(other) else np.inf
        half = min(half_width_hz, .4 * nearest)
        bins = np.flatnonzero(np.abs(f - f[index]) <= half + 1e-9)
        return dict(kind=kind, bin=int(index), frequency_hz=float(f[index]),
                    selection_channels=channels, half_width_hz=float(half),
                    bins=bins.tolist(), band_bin_centers_hz=[float(f[bins[0]]), float(f[bins[-1]])])

    sites = [site(i, 'peak', channels) for i, channels in sorted(selected.items())]
    # Valley must lie between detected peaks and away from either peak center.
    trace = training_psd[:, 1:].sum(axis=1)
    minima, _ = find_peaks(-np.log(np.maximum(trace, 1e-300)))
    valleys = []
    for left, right in zip(all_bins[:-1], all_bins[1:]):
        guard = max(2*df, half_width_hz)
        choices = minima[(f[minima]-f[left] > guard) & (f[right]-f[minima] > guard)]
        if not len(choices):
            continue
        index = int(choices[np.argmin(trace[choices])])
        valleys.append(index)
    if valley_count and len(valleys) < valley_count:
        raise ValueError("Not enough resolved inter-peak intervals for valley controls")
    if valley_count:
        for k in np.rint(np.linspace(0, len(valleys)-1, valley_count)).astype(int):
            sites.append(site(valleys[k], 'valley', []))
    for i, s in enumerate(sites):
        s['id'] = f"{i+1:02d}_{s['kind']}_{s['frequency_hz']:.2f}Hz"
    return sites, candidates


def spectral_matrices(record, fs, nperseg, sites):
    """Streaming selected-bin Welch CSD, bounded in frequency-space memory.

    Returns integrated bands and three adjacent single-bin density matrices per
    site. Hann, constant detrend, 50% overlap, one-sided density; no zero padding.
    """
    x = np.asarray(record, dtype=np.float64)
    if x.ndim != 2 or len(x) < nperseg or not np.isfinite(x).all():
        raise ValueError("Need a finite (time,coordinate) record at least nperseg long")
    f = np.fft.rfftfreq(nperseg, 1/fs)
    band_indices, nearby = [], []
    for site in sites:
        center = int(np.argmin(abs(f-site['frequency_hz'])))
        bins = np.flatnonzero(abs(f-site['frequency_hz']) <= site['half_width_hz'] + 1e-9)
        if not len(bins):
            bins = np.array([center])
        band_indices.append(bins)
        nearby.append(np.clip(np.arange(center-1, center+2), 0, len(f)-1))
    required = np.unique(np.concatenate(band_indices+nearby))
    matrices = np.zeros((len(required), x.shape[1], x.shape[1]), complex)
    window = get_window('hann', nperseg)
    starts = range(0, len(x)-nperseg+1, nperseg-nperseg//2)
    for start in starts:
        segment = x[start:start+nperseg]
        z = np.fft.rfft((segment-segment.mean(axis=0))*window[:, None], axis=0)[required]
        matrices += np.einsum('fi,fj->fij', z, z.conj())
    factors = np.full(len(required), 2.)
    factors[required == 0] = 1
    if nperseg % 2 == 0:
        factors[required == nperseg//2] = 1
    matrices *= factors[:, None, None] / (len(starts)*fs*np.dot(window, window))
    bands = np.stack([matrices[np.searchsorted(required, bins)].sum(axis=0)*fs/nperseg
                      for bins in band_indices])
    neighbors = np.stack([matrices[np.searchsorted(required, bins)] for bins in nearby])
    return bands, neighbors, dict(windows=len(starts),
                                 band_bins=[b.tolist() for b in band_indices],
                                 band_frequencies_hz=[f[b].tolist() for b in band_indices])


def compare_ranks(matrix, reference_vectors, primary_rank, ranks):
    results = {}
    for rank in ranks:
        values, vectors = eigensystem(matrix[1:rank+1, 1:rank+1])
        dim = max(rank, primary_rank)
        a = np.zeros((dim, 2), complex)
        b = np.zeros_like(a)
        a[:primary_rank] = reference_vectors[:, :2]
        b[:rank] = vectors[:, :2]
        results[str(rank)] = dict(mac=similarity(a, b), subspace2=similarity(a, b, 2),
                                 eta1=float(values[0]/values.sum()),
                                 deformation_band_variance=float(values.sum()))
    return results


def analyze_matrices(bands, neighbors, train_count, primary_rank, ranks,
                     bootstrap=100, seed=20261026, full_gram=None):
    """Train-only eigensystems plus held-out capture and repetition stability."""
    rng = np.random.default_rng(seed)
    if full_gram is None:
        full_gram = np.eye(primary_rank+1)
    mean = bands[:train_count].mean(axis=0)
    validation = bands[train_count:].mean(axis=0)
    bootstrap_indices = rng.integers(0, train_count, (bootstrap, train_count))
    summaries, vectors_list, full_list, eigenvalues_list = [], [], [], []
    for j in range(len(mean)):
        matrix = mean[j, 1:primary_rank+1, 1:primary_rank+1]
        values, vectors = eigensystem(matrix)
        val = validation[j, 1:primary_rank+1, 1:primary_rank+1]
        _, val_vectors = eigensystem(val)
        full_values, full_vectors, _ = metric_eigensystem(
            mean[j, :primary_rank+1, :primary_rank+1], full_gram)
        row = dict(eta1=float(values[0]/values.sum()),
                   eta12=float(values[:2].sum()/values.sum()),
                   lambda2_over_lambda1=float(values[1]/values[0]),
                   deformation_band_variance=float(values.sum()),
                   validation_capture1=capture(val, vectors),
                   validation_capture2=capture(val, vectors, 2),
                   validation_mode_mac=similarity(vectors, val_vectors),
                   validation_subspace2=similarity(vectors, val_vectors, 2),
                   mean_inclusive_eta1=float(full_values[0]/full_values.sum()),
                   mean_inclusive_leading_uniform_fraction=float(abs(full_gram[0] @ full_vectors[:, 0])**2),
                   ranks=compare_ranks(mean[j], vectors, primary_rank, ranks))
        row['repetitions'] = []
        for k in range(len(bands)):
            rep = bands[k, j, 1:primary_rank+1, 1:primary_rank+1]
            # Training reproducibility uses a leave-one-repetition-out reference.
            ref = ((train_count*matrix-rep)/(train_count-1) if k < train_count else matrix)
            _, ref_vectors = eigensystem(ref)
            _, rep_vectors = eigensystem(rep)
            row['repetitions'].append(dict(mac=similarity(ref_vectors, rep_vectors),
                subspace2=similarity(ref_vectors, rep_vectors, 2),
                capture1=capture(rep, ref_vectors), capture2=capture(rep, ref_vectors, 2)))
        boot = []
        for indices in bootstrap_indices:
            sample = bands[indices, j, 1:primary_rank+1, 1:primary_rank+1].mean(axis=0)
            sv, vv = eigensystem(sample)
            boot.append([sv[0]/sv.sum(), similarity(vectors, vv), similarity(vectors, vv, 2)])
        row['bootstrap'] = dict(draws=bootstrap, columns=['eta1', 'mac_to_pooled', 'subspace2_to_pooled'],
                               quantiles_025_50_975=np.quantile(boot, [.025, .5, .975], axis=0).tolist())
        row['nearby_bins'] = []
        for offset in range(3):
            near = neighbors[:train_count, j, offset, 1:primary_rank+1, 1:primary_rank+1].mean(axis=0)
            ev, vv = eigensystem(near)
            row['nearby_bins'].append(dict(offset=offset-1, eta1=float(ev[0]/ev.sum()),
                                          mac_to_band=similarity(vectors, vv),
                                          subspace2_to_band=similarity(vectors, vv, 2)))
        summaries.append(row)
        vectors_list.append(vectors[:, :2])
        full_list.append(full_vectors[:, :2])
        eigenvalues_list.append(values)
    return summaries, np.stack(vectors_list), np.stack(full_list), np.stack(eigenvalues_list)


def spatial_patterns(modes, vectors, include_mean=False):
    """Return (site, mode, y, x), checking the spatial norm explicitly."""
    phi = modes.reshape(len(modes), -1).T
    if include_mean:
        phi = np.column_stack([np.ones(len(phi))/np.sqrt(len(phi)), phi])
    error = np.max(abs(phi.T @ phi-np.eye(phi.shape[1])))
    if error > 1e-6 and not include_mean:
        raise ValueError(f"Spatial basis is not orthonormal (error {error:g}); weighted SPOD required")
    norms = np.einsum('sdm,de,sem->sm', vectors.conj(), phi.T @ phi, vectors).real
    if not np.allclose(norms, 1, atol=1e-6):
        raise ValueError('Coefficient eigenvectors must have unit physical spatial norm')
    fields = np.einsum('xd,sdm->smx', phi, vectors).reshape(
        vectors.shape[0], vectors.shape[2], *modes.shape[1:])
    # A display-only phase rotation; cross-peak MACs use unaltered eigenvectors.
    for site in fields:
        for field in site:
            pivot = np.argmax(abs(field))
            field *= np.exp(-1j*np.angle(field.flat[pivot]))
    return fields, float(error)


def plot_spatial(path, fields, title, extent, units, phase_threshold=.1):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for j, field in enumerate(fields):
        amplitude = abs(field)
        bound = amplitude.max()
        phase = np.ma.masked_where(amplitude < phase_threshold*bound, np.angle(field))
        images = [amplitude, phase, field.real, -field.imag]
        labels = ['Amplitude', 'Phase (amplitude mask)', 'Phase cycle: 0°', 'Phase cycle: 90°']
        for k, (values, label) in enumerate(zip(images, labels)):
            limits = (dict(vmin=-np.pi, vmax=np.pi, cmap='twilight') if k == 1 else
                      dict(vmin=-bound, vmax=bound, cmap='RdBu_r') if k > 1 else dict(cmap='viridis'))
            im = axes[j, k].imshow(values, origin='lower', extent=extent, **limits)
            axes[j, k].set(title=f'Mode {j+1}\n{label}', xlabel=f'x [{units[0]}]', ylabel=f'y [{units[1]}]')
            axes[j, k].title.set_fontsize(10)
            fig.colorbar(im, ax=axes[j, k], shrink=.75)
    fig.suptitle(title+'\nUnit spatial norm; arbitrary global phase; phase cycles are illustrative')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def animate_modes(path, fields, title, extent, frames=24):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    images = []
    for j, (ax, field) in enumerate(zip(axes, fields)):
        bound = abs(field).max()
        images.append(ax.imshow(field.real, origin='lower', extent=extent,
                                cmap='RdBu_r', vmin=-bound, vmax=bound))
        ax.set(title=f'Mode {j+1}', xlabel='x', ylabel='y')
    fig.suptitle(title+'\nIllustrative phase cycle, not a measured realization')
    def update(frame):
        for im, field in zip(images, fields):
            im.set_data(np.real(field*np.exp(2j*np.pi*frame/frames)))
        return images
    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    animation.save(path, writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)


def plot_overview(out, frequency, training_psd, sites, summaries, eigenvalues, vectors,
                  full_vectors, train_count, mean_highpass_hz=50.):
    labels = [f"{s['frequency_hz']:.1f}"+(' V' if s['kind']=='valley' else '') for s in sites]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].semilogy(frequency, training_psd[:, 0], label=f'{mean_highpass_hz:g} Hz high-pass spatial mean component')
    axes[0].semilogy(frequency, training_psd[:, 1], label='POD 1')
    axes[0].semilogy(frequency, training_psd[:, 1:].sum(1), label='Summed deformation PSD')
    for s in sites:
        color = 'C3' if s['kind']=='peak' else '.5'
        axes[0].axvspan(s['frequency_hz']-s['half_width_hz'], s['frequency_hz']+s['half_width_hz'], color=color, alpha=.15)
        axes[0].axvline(s['frequency_hz'], color=color, alpha=.5, lw=.7)
    axes[0].set(xlim=(max(0,min(s['frequency_hz'] for s in sites)-60),max(s['frequency_hz'] for s in sites)+60), xlabel='Frequency [Hz]', ylabel='Physical coefficient PSD [signal²/Hz]',
                title='Training-only peak selection; red: peaks, grey: valley controls')
    shown=(frequency>=axes[0].get_xlim()[0]) & (frequency<=axes[0].get_xlim()[1])
    displayed=np.column_stack([training_psd[:,0],training_psd[:,1],training_psd[:,1:].sum(1)])[shown]
    axes[0].set_ylim(.5*displayed[displayed>0].min(),2*displayed.max())
    axes[0].legend(fontsize=8)
    x = np.arange(len(sites))
    for key, label, marker in [('eta1','Training first mode','o'), ('eta12','Training first two','s'),
                             ('validation_capture1','Held-out capture by training first mode','^'),
                             ('validation_capture2','Held-out capture by training first two','v')]:
        axes[1].plot(x, [r[key] for r in summaries], marker+'-', label=label)
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0,1.03), xlabel='Band center [Hz]; V = valley',
                ylabel='Fraction of deformation spectral variance')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out/'overview.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    matrices = []
    for container, count in [(vectors,1),(vectors,2),(full_vectors,1)]:
        matrices.append(np.array([[similarity(a,b,count) for b in container] for a in container]))
    for ax, mat, title in zip(axes, matrices, ['Deformation mode 1 MAC','Deformation first-two subspaces','Mean-inclusive mode 1 MAC']):
        im = ax.imshow(mat, vmin=0,vmax=1,cmap='viridis')
        ax.set(xticks=x,yticks=x,xticklabels=labels,yticklabels=labels,title=title)
        ax.tick_params(axis='x',rotation=90)
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(out/'cross_peak_similarity.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for i, vals in enumerate(eigenvalues):
        axes[0].semilogy(np.arange(1,min(12,len(vals))+1),vals[:12]/vals.sum(),'.-',label=labels[i])
    axes[0].set(xlabel='SPOD mode index',ylabel='Fraction of band variance',title='Training eigenvalue spectra')
    axes[0].legend(fontsize=7,ncol=2)
    repro=np.array([[r['mac'] for r in s['repetitions']] for s in summaries])
    im=axes[1].imshow(repro,vmin=0,vmax=1,aspect='auto')
    axes[1].axvline(train_count-.5,color='red',lw=1)
    axes[1].set(yticks=x,yticklabels=labels,xlabel='Repetition index (train then validation)',
                title='Mode 1 MAC: training leave-one-out / held-out')
    fig.colorbar(im,ax=axes[1])
    fig.savefig(out/'eigenvalues_and_reproducibility.png',dpi=160)
    plt.close(fig)
    return matrices


def write_report(out, protocol, sites, summaries):
    lines=['# Experimental spectral peak atlas','',
           'Training-only complex spectral modes in physical shared-POD coordinates. '
           'No oscillator model, power-law frequency constraint, or per-coordinate standardization is used.','',
           '[Overview](overview.png) · [Spatial similarity](cross_peak_similarity.png) · '
           '[Eigenvalues and repetition stability](eigenvalues_and_reproducibility.png)','',
           '| Band [Hz] | Selection | First-mode fraction | First-two fraction | Validation capture, first mode | Validation mode MAC | Validation two-mode subspace |',
           '|---|---|---:|---:|---:|---:|---:|']
    for s,r in zip(sites,summaries):
        lines.append(f"| [{s['frequency_hz']:.2f}]({s['id']}/spatial_modes.png) | {s['kind']} | "
                     f"{r['eta1']:.3f} | {r['eta12']:.3f} | {r['validation_capture1']:.3f} | {r['validation_mode_mac']:.3f} | {r['validation_subspace2']:.3f} |")
    peaks=[i for i,s in enumerate(sites) if s['kind']=='peak']
    fractions=[summaries[i]['eta12'] for i in peaks]
    with np.load(out/'spectral_modes.npz') as arrays:
        pair=arrays['cross_peak_subspace2'][np.ix_(peaks,peaks)]
    off=pair[~np.eye(len(peaks),dtype=bool)]
    lines += ['', '## Initial reading','',
              f"The leading two deformation modes explain {100*min(fractions):.1f}–{100*max(fractions):.1f}% "
              'of the selected peaks\' training band variance.']
    if len(off):
        lines.append(f"Their pairwise two-mode subspace similarities across peaks range from {off.min():.3f} "
                     f"to {off.max():.3f}. A value near one means similar spatial subspaces, even if individual "
                     'leading eigenvectors have different orientations. This is a comparison within the retained '
                     'POD subspace and measured field of view, not evidence that physical eigenmodes are identical.')
    for i in peaks:
        row=summaries[i]
        if row['lambda2_over_lambda1']>.9:
            lines.append(f"At {sites[i]['frequency_hz']:.2f} Hz, lambda2/lambda1={row['lambda2_over_lambda1']:.3f}: "
                         f"the first two modes are nearly degenerate. The alternate-window individual-mode MAC "
                         f"is {row['alternate_window']['mac']:.3f}, while the two-mode subspace similarity is "
                         f"{row['alternate_window']['subspace2']:.5f}. Interpret the subspace, not a unique mode orientation.")
    lines += ['', '## Spatial patterns','',
              'Each band has two deformation modes and a mean-inclusive companion. '
              'Patterns have unit pixel-sum spatial norm; color amplitude does not indicate relative excitation. '
              'Excitation is given by the saved eigenvalues. Phase is masked below 10% of peak amplitude. '
              'Global phase is arbitrary. Animations show a phase cycle, not an experimental realization.','']
    for s in sites:
        links=f"[Deformation]({s['id']}/spatial_modes.png), [mean-inclusive]({s['id']}/mean_inclusive_modes.png)"
        if (out/s['id']/'phase_cycle.gif').exists():links+=f", [animation]({s['id']}/phase_cycle.gif)"
        lines.append(f"- **{s['frequency_hz']:.2f} Hz ({s['kind']})**: {links}.")
    lines += ['', '## Stability diagnostics','',
              '| Band [Hz] | Bootstrap mode MAC 2.5–97.5% | Rank 20 MAC | Rank 100 MAC | Alternate-window MAC | Alternate-window subspace | lambda2/lambda1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for s,r in zip(sites,summaries):
        q=np.asarray(r['bootstrap']['quantiles_025_50_975'])
        def rank_mac(k):return f"{r['ranks'][str(k)]['mac']:.3f}" if str(k) in r['ranks'] else '—'
        lines.append(f"| {s['frequency_hz']:.2f} | {q[0,1]:.3f}–{q[2,1]:.3f} | {rank_mac(20)} | {rank_mac(100)} | "
                     f"{r['alternate_window']['mac']:.3f} | {r['alternate_window']['subspace2']:.5f} | {r['lambda2_over_lambda1']:.3f} |")
    lines += ['', 'Bootstrap quantiles describe orientation stability under repetition resampling; '
              'they are not independent-window confidence intervals. Nearly equal eigenvalues make individual '
              'vectors ambiguous; first-two-subspace similarities are saved in summary.json. '
              'Band-integrated, center-bin and neighboring-bin comparisons are also saved.','',
              '## Protocol and limits','',
              f"- Primary POD rank: {protocol['primary_rank']}; comparison ranks: {protocol['ranks']}.",
              f"- Effective sampling: {protocol['fs_hz']:.6f} Hz; Hann window: {protocol['nperseg']} samples "
              f"({1000*protocol['nperseg']/protocol['fs_hz']:.2f} ms), 50% overlap; no zero padding.",
              f"- Training: {', '.join(protocol['training_repetitions'])}.",
              f"- Validation: {', '.join(protocol['validation_repetitions'])}; test trajectories are excluded.",
              '- Each repetition has equal weight; windows never cross recording boundaries.',
              '- Eigenvalues are displacement variance (integrated bands), not capillary energy. '
              'The spatial norm is an unweighted pixel sum on the uniform grid.',
              '- The mean-inclusive decomposition uses the measured spatial Gram matrix, correcting the small '
              'nonorthogonality of the constant direction and saved POD modes. Its MACs use the same physical norm.',
              '- The mean-inclusive field uses the stored high-pass mean; deformation coefficients receive no added high-pass filter.',
              '- No increment trimming or generated trajectories enter this analysis. Anti-alias preparation removes filter edges.',
              '- The existing shared basis includes held-out repetitions. Spectral fitting and peak selection are train-only, '
              'but validation is not independent of the prior spatial basis construction.',
              '- The number of Welch windows limits single-bin spectral rank; overlapping windows and adjacent frequencies '
              'are not independent experimental repetitions.',
              '- Different modes at different frequencies need not be orthogonal. SPOD does not establish physical eigenmodes '
              'or nonlinear phase coupling. Spatial patterns are restricted to the observed field of view and retained POD subspace.',
              '', 'Numerical data: [summary.json](summary.json), [spectral_modes.npz](spectral_modes.npz), '
              '[per-repetition matrices](spectral_matrices.npz), [protocol.json](protocol.json).']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def render_saved(out, no_animations=False):
    """Refresh presentation from saved results without rereading source data."""
    out=Path(out)
    protocol=json.loads((out/'protocol.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    sites,rows=summary['sites'],summary['metrics']
    with np.load(out/'spectral_modes.npz') as a:
        vectors=a['coefficient_vectors'];full=a['mean_inclusive_vectors']
        gram=a['mean_inclusive_spatial_gram'];ev=a['eigenvalues']
        fields=a['spatial_modes'];full_fields=a['mean_inclusive_spatial_modes']
        x,y=a['grid_x'],a['grid_y']
    with np.load(out/'spectral_matrices.npz') as a:
        frequency=a['frequency_hz'];nt=int(a['train_count'])
        training_psd=a['per_repetition_psd'][:nt].mean(axis=0)
    values,basis=np.linalg.eigh(gram)
    root=(basis*np.sqrt(values))@basis.T
    orthogonal=np.einsum('de,sem->sdm',root,full)
    plot_overview(out,frequency,training_psd,sites,rows,ev,vectors,orthogonal,nt,protocol['mean_highpass_hz'])
    extent=[x[0],x[-1],y[0],y[-1]]
    for s,r,field,full_field in zip(sites,rows,fields,full_fields):
        title=f"{protocol['experiment']}: {s['frequency_hz']:.2f} Hz {s['kind']}; "
        print('Rendering',s['id'],flush=True)
        plot_spatial(out/s['id']/'spatial_modes.png',field,title+f"deformation eta1={r['eta1']:.2f}",extent,protocol['grid_units'])
        plot_spatial(out/s['id']/'mean_inclusive_modes.png',full_field,title+'including high-pass spatial mean',extent,protocol['grid_units'])
        if not no_animations and s['kind']=='peak':
            animate_modes(out/s['id']/'phase_cycle.gif',field,title+'deformation',extent)
    write_report(out,protocol,sites,rows)
    (out/'rendering.json').write_text(json.dumps(dict(
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source='Saved spectral_modes.npz, spectral_matrices.npz, summary.json, protocol.json',
        recomputed_spectral_statistics=False),indent=2)+'\n')
    return out


def run(args):
    if args.render_only:
        return render_saved(args.output,args.no_animations)
    if (args.rank < 2 or min(args.compare_ranks) < 2 or args.bootstrap < 2
            or args.nperseg < 8 or args.alternate_nperseg < 8
            or args.peaks_per_channel < 1 or args.valleys < 0 or args.band_half_width_hz <= 0):
        raise ValueError('Invalid rank, bootstrap, window, or selection settings')
    ranks=sorted(set([args.rank]+args.compare_ranks)); max_rank=max(ranks)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a new or empty output directory; results are never overwritten')
    data=SharedPODTrainingData(args.experiment,max_rank,data_root=args.data_root,
        shared_basis=args.shared_basis,validation_reps=args.validation_reps,test_reps=args.test_reps,
        include_spatial_mean=True,spatial_mean_highpass_hz=args.mean_highpass_hz,
        downsample_factor=args.downsample,downsample_edge_trim=args.edge_trim,output_dtype='float64')
    names=list(data.train_repetitions+data.validation_repetitions); nt=len(data.train_repetitions)
    if nt<2 or not data.validation_repetitions:
        raise ValueError('Need at least two training repetitions and one validation repetition')
    fs=1/data.native_dt
    if args.high_hz>=fs/2 or args.low_hz<=0 or args.low_hz>=args.high_hz:
        raise ValueError('Peak interval must lie strictly within (0, Nyquist)')
    cache_bytes=sum(data.frame_counts[n] for n in names)*(max_rank+1)*8
    if cache_bytes > args.cache_gib*2**30:
        raise MemoryError(f'Prepared trajectory cache needs {cache_bytes/2**30:.3f} GiB; raise --cache-gib')
    records=[]
    for i,name in enumerate(names):
        print(f'Projecting {i+1}/{len(names)}: {name}',flush=True)
        a=np.empty((data.frame_counts[name],max_rank+1))
        for start in range(0,len(a),args.block_size):
            end=min(start+args.block_size,len(a));_,a[start:end]=data.read_coordinates(name,start,end)
        if len(a)<max(args.nperseg,args.alternate_nperseg):
            raise ValueError(f'{name} is too short for the requested Welch windows')
        records.append(a)
    psds=[]
    for record in records:
        f,p=welch(record[:,:args.rank+1],fs=fs,window='hann',nperseg=args.nperseg,
                  noverlap=args.nperseg//2,detrend='constant',scaling='density',axis=0)
        psds.append(p)
    psds=np.stack(psds);training_psd=psds[:nt].mean(axis=0)
    sites,candidates=select_sites(f,training_psd,low=args.low_hz,high=args.high_hz,
        prominence_db=args.prominence_db,spacing_hz=args.spacing_hz,per_channel=args.peaks_per_channel,
        valley_count=args.valleys,half_width_hz=args.band_half_width_hz)
    print('Selected bands:',[(s['kind'],round(s['frequency_hz'],2)) for s in sites],flush=True)
    bands=[];neighbors=[];window_info=[]
    for i,record in enumerate(records):
        print(f'Spectral matrices {i+1}/{len(records)}',flush=True)
        b,n,info=spectral_matrices(record,fs,args.nperseg,sites)
        bands.append(b);neighbors.append(n);window_info.append(info)
    bands=np.stack(bands);neighbors=np.stack(neighbors)
    modes,x,y,xunits,yunits=_load_shared_spatial_modes(data)
    phi=modes[:args.rank].reshape(args.rank,-1).T
    full_phi=np.column_stack([np.ones(len(phi))/np.sqrt(len(phi)),phi])
    full_gram=full_phi.T@full_phi
    print('Eigendecompositions and repetition bootstrap',flush=True)
    summaries,vectors,full_vectors,eigenvalues=analyze_matrices(
        bands,neighbors,nt,args.rank,ranks,args.bootstrap,args.seed,full_gram)
    alternate=[]
    for record in records[:nt]:
        b,_,alt_info=spectral_matrices(record[:,:args.rank+1],fs,args.alternate_nperseg,sites)
        alternate.append(b)
    alternate=np.mean(alternate,axis=0)
    for i,row in enumerate(summaries):
        ev,vv=eigensystem(alternate[i,1:,1:])
        row['alternate_window']=dict(nperseg=args.alternate_nperseg,eta1=float(ev[0]/ev.sum()),
            mac=similarity(vectors[i],vv),subspace2=similarity(vectors[i],vv,2),
            frequency_bins_hz=alt_info['band_frequencies_hz'][i])
    fields,gram_error=spatial_patterns(modes[:args.rank],vectors)
    full_fields,full_gram_error=spatial_patterns(modes[:args.rank],full_vectors,True)
    with h5py.File(data.basis_path,'r') as source:
        basis_reps=list(source['repetitions'])
    source_files=[Path(data.basis_path)]+[Path(data.source_paths[n]) for n in names]
    protocol=dict(experiment=args.experiment,primary_rank=args.rank,ranks=ranks,fs_hz=fs,
        raw_fs_hz=1/data.raw_native_dt,nperseg=args.nperseg,alternate_nperseg=args.alternate_nperseg,
        preprocessing=data.source_config,training_repetitions=list(data.train_repetitions),
        validation_repetitions=list(data.validation_repetitions),excluded_test_repetitions=list(data.test_repetitions),
        basis_repetitions=basis_reps,signal_units=data.signal_units,grid_units=[xunits,yunits],
        mean_dataset=data.spatial_mean_dataset,downsample=args.downsample,edge_trim_samples=args.edge_trim,
        mean_highpass_hz=args.mean_highpass_hz,bootstrap_seed=args.seed,bootstrap_draws=args.bootstrap,
        gram_error=gram_error,mean_inclusive_gram_error=full_gram_error,window_info=window_info,
        selection=vars(args),source_files=[dict(path=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in source_files],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        convention='S_ij=E[X_i conj(X_j)]; spatial mode = phi @ eigenvector; no coordinate standardization',
        preparation='Existing centered anti-alias FIR per repetition, decimation and edge exclusion; no increment trimming',
        spatial_norm='Euclidean sum over pixels; uniform grid; mean coefficient sqrt(N) times physical mean',
        band_units='signal squared; sum of one-sided Welch density times df',
        independence='Equal repetition weights; overlapping windows are not independent repetitions')
    out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'spectral_matrices.npz',band_matrices=bands,neighbor_density_matrices=neighbors,
        repetition_names=names,train_count=nt,frequency_hz=f,per_repetition_psd=psds,
        alternate_training_band_matrices=alternate)
    gvalues,gbasis=np.linalg.eigh(full_gram)
    groot=(gbasis*np.sqrt(gvalues))@gbasis.T
    full_orthogonal=np.einsum('de,sem->sdm',groot,full_vectors)
    matrices=plot_overview(out,f,training_psd,sites,summaries,eigenvalues,vectors,full_orthogonal,nt,args.mean_highpass_hz)
    np.savez_compressed(out/'spectral_modes.npz',coefficient_vectors=vectors,mean_inclusive_vectors=full_vectors,
        eigenvalues=eigenvalues,spatial_modes=fields,mean_inclusive_spatial_modes=full_fields,
        grid_x=x,grid_y=y,cross_peak_mac=matrices[0],cross_peak_subspace2=matrices[1],
        mean_inclusive_spatial_gram=full_gram,
        mean_inclusive_cross_peak_mac=matrices[2],site_frequency_hz=[s['frequency_hz'] for s in sites])
    extent=[x[0],x[-1],y[0],y[-1]]
    for s,row,field,full_field in zip(sites,summaries,fields,full_fields):
        print('Rendering',s['id'],flush=True)
        folder=out/s['id'];folder.mkdir()
        title=f"{args.experiment}: {s['frequency_hz']:.2f} Hz {s['kind']}; "
        plot_spatial(folder/'spatial_modes.png',field,title+f"deformation eta1={row['eta1']:.2f}",extent,[xunits,yunits])
        plot_spatial(folder/'mean_inclusive_modes.png',full_field,title+'including high-pass spatial mean',extent,[xunits,yunits])
        if not args.no_animations and s['kind']=='peak':
            animate_modes(folder/'phase_cycle.gif',field,title+'deformation',extent)
    (out/'summary.json').write_text(json.dumps(dict(sites=sites,candidate_peaks=candidates,
        metrics=summaries,repetition_names=names),indent=2,allow_nan=False)+'\n')
    write_report(out,protocol,sites,summaries)
    print('Atlas:',out.resolve()/'README.md',flush=True)
    return out


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',default='0p10')
    p.add_argument('--data-root',default=str(DEFAULT_DATA_ROOT))
    p.add_argument('--shared-basis')
    p.add_argument('--rank',type=int,default=50)
    p.add_argument('--compare-ranks',type=int,nargs='+',default=[20,100])
    p.add_argument('--validation-reps',type=int,nargs='+',default=[13,14])
    p.add_argument('--test-reps',type=int,nargs='+',default=[15,16])
    p.add_argument('--downsample',type=int,default=8)
    p.add_argument('--edge-trim',type=int,default=32)
    p.add_argument('--mean-highpass-hz',type=float,default=50.)
    p.add_argument('--nperseg',type=int,default=4096)
    p.add_argument('--alternate-nperseg',type=int,default=2048)
    p.add_argument('--low-hz',type=float,default=100.)
    p.add_argument('--high-hz',type=float,default=1100.)
    p.add_argument('--prominence-db',type=float,default=2.)
    p.add_argument('--spacing-hz',type=float,default=40.)
    p.add_argument('--peaks-per-channel',type=int,default=3)
    p.add_argument('--valleys',type=int,default=3)
    p.add_argument('--band-half-width-hz',type=float,default=10.)
    p.add_argument('--bootstrap',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--cache-gib',type=float,default=2.)
    p.add_argument('--block-size',type=int,default=2048)
    p.add_argument('--no-animations',action='store_true')
    p.add_argument('--render-only',action='store_true',help='Refresh figures/report from saved numerical results only')
    p.add_argument('--output',required=True)
    return p


if __name__=='__main__':
    run(parser().parse_args())
