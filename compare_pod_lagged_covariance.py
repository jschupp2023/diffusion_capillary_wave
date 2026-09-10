"""Compare temporal POD statistics across powers in shared spatial bases.

Example (all powers >= 0p08, three seeded reference recordings)::

    python compare_pod_lagged_covariance.py

Quick subset::

    python compare_pod_lagged_covariance.py --powers 0p20 0p35 \
        --max-repetitions 2 --references 1 --output-dir /tmp/pod_lagged_check

Simple single-delay correlation comparison (10 samples is the primary lag)::

    python compare_pod_lagged_covariance.py --single-lag-samples 0 5 10 20 50 100

The target dimension is always --rank. --source-ranks checks convergence of
the projection using progressively larger stored reconstructions. Statistics
use the largest source rank; this is an approximation to raw-field projection.
Lagged covariances are linearly interpolated between adjacent sample lags onto
one physical-time grid. No symmetry is imposed at nonzero lag.

Outputs include a resumable HDF5 cache, PDF/PNG figures, labeled distance CSVs,
projection diagnostics, repetition-held-out classification, and a Markdown
report. An interrupted run resumes automatically if its configuration and
input file metadata match. Existing per-power analyses are never modified.

Full zero-lag whitening is derived directly from the cached POD covariance
matrices, excluding the spatial mean. --whitening-eps-rel controls its relative
eigenvalue floor; derived settings are saved in summary_configuration.json
without invalidating the original projection cache/configuration.json.
The whitened statistic is additionally compared using diagonal entries only
and directed off-diagonal entries only, with RMS over each retained entry set.
All components reuse the same whitening, physical lag grid, and holdouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import textwrap

# Avoid BLAS oversubscription for small coordinate matrices. Environment
# settings supplied by the caller take precedence.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "2")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.spatial.distance import pdist, squareform

from pod_analysis_report import _check_grid, load_basis
from pod_center_psd import infer_sampling_frequency
from pod_lagged_covariance import discover_repetitions, lagged_covariances, normalize


ROOT = Path("/home/jonas/ucsd_thesis/reduced_data")
DEFAULT_POWERS = ("0p08", "0p10", "0p15", "0p18", "0p20", "0p25", "0p30", "0p35")
SCHEMA_VERSION = 1
WHITENED = "full_whitened_correlation"
DIAG_ONLY = "diag_only"
OFFDIAG_ONLY = "offdiag_only"
POD_METRICS = ("correlation", "covariance", WHITENED)
METRIC_TITLES = {"correlation": "Normalized correlation", "covariance": "Raw covariance",
                 WHITENED: "Full zero-lag whitening", DIAG_ONLY: "Whitened diagonal only",
                 OFFDIAG_ONLY: "Whitened off-diagonal only"}
COMPONENT_LABELS = {WHITENED: "full_whitened", DIAG_ONLY: DIAG_ONLY, OFFDIAG_ONLY: OFFDIAG_ONLY}
WHITENING_IDENTITY_TOL = 1e-6


def default_lags_ms() -> np.ndarray:
    return np.r_[np.arange(0, 1.001, .05), np.arange(1.2, 5.001, .2),
                 np.arange(6, 20.001, 1.)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reduced-root", type=Path, default=ROOT)
    p.add_argument("--powers", nargs="+", default=list(DEFAULT_POWERS))
    p.add_argument("--rank", type=int, default=10, help="Common target dimension.")
    p.add_argument("--pod-rank", type=int, default=1000)
    p.add_argument("--source-ranks", nargs="+", type=int, default=[10, 30, 100])
    p.add_argument("--references", type=int, default=3)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--lags-ms", nargs="+", type=float,
                   help="Physical lag grid; default: 56 points from 0 to 20 ms.")
    p.add_argument("--max-repetitions", type=int)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--whitening-eps-rel", type=float, default=1e-10,
                   help="Relative eigenvalue floor for full POD whitening (default: 1e-10). "
                   "Changing this regenerates derived summaries using the existing covariance cache.")
    p.add_argument("--single-lag-samples", type=int, nargs="+",
                   help="Run the simple unwhitened diagnostic at these exact sample delays, "
                   "each scored separately. Zero and the primary lag are always included.")
    p.add_argument("--primary-lag-samples", type=int, default=10,
                   help="Primary delay for the simple diagnostic (default: 10 samples).")
    a = p.parse_args()
    if a.single_lag_samples is not None and a.lags_ms is not None:
        p.error("Use --single-lag-samples or --lags-ms, not both.")
    a.powers = sorted(set(a.powers), key=lambda s: float(s.replace("p", ".")))
    a.source_ranks = sorted(set(a.source_ranks))
    a.lags_ms = (default_lags_ms() if a.lags_ms is None
                 else np.asarray(sorted(set(a.lags_ms)), dtype=float))
    if (a.rank < 1 or not a.source_ranks or a.source_ranks[0] < a.rank
            or a.source_ranks[-1] > a.pod_rank):
        p.error("Require 1 <= rank <= source ranks <= pod rank.")
    if len(a.source_ranks) < 2:
        p.error("Supply at least two source ranks for the convergence check.")
    if (len(a.powers) < 2 or min(a.references, a.batch_size) < 1
            or (a.max_repetitions is not None and a.max_repetitions < 2)):
        p.error("Need >=2 powers/repetitions and positive references/batch size/threads.")
    if (len(a.lags_ms) < 2 or not np.isfinite(a.lags_ms).all()
            or a.lags_ms[0] != 0 or a.lags_ms[-1] <= 0):
        p.error("Lags must be finite, nonnegative, include zero and a positive lag.")
    if not np.isfinite(a.whitening_eps_rel) or not 0 < a.whitening_eps_rel < 1:
        p.error("--whitening-eps-rel must be finite and strictly between zero and one.")
    if a.single_lag_samples is not None:
        if min(a.single_lag_samples) < 0 or a.primary_lag_samples < 1:
            p.error("Sample lags must be nonnegative and the primary lag must be positive.")
        a.single_lag_samples = sorted(set([0, a.primary_lag_samples, *a.single_lag_samples]))
    if a.output_dir is None:
        a.output_dir = (a.reduced_root / "center_psd_comparisons" /
                        "lagged_covariance" / (f"single_lag_rank_{a.rank}" if a.single_lag_samples is not None
                                               else f"cross_power_rank_{a.rank}"))
    return a


def physical_lag_covariance(x: np.ndarray, fs: float, seconds: np.ndarray) -> np.ndarray:
    """C_ij(tau)=mean[x_i(t) x_j(t+tau)], with N-lag normalization.

    FFTs compute the exact integer-lag sums with zero padding. Fractional
    sample lags interpolate the two normalized covariance estimates.
    Input must already be centered. Output at zero is symmetric, generally
    output at a positive lag is not.
    """
    n, k = x.shape
    sample_lags = np.asarray(seconds) * fs
    # Remove floating-point noise when a physical lag is an integer sample.
    nearest = np.rint(sample_lags)
    sample_lags = np.where(np.abs(sample_lags - nearest) < 1e-8, nearest, sample_lags)
    lo = np.floor(sample_lags).astype(int)
    hi = np.ceil(sample_lags).astype(int)
    if np.any(lo < 0) or np.any(hi > n - 2):
        raise ValueError("Lags must leave at least two sample pairs.")
    fraction = sample_lags - lo
    fft_size = next_fast_len(2 * n - 1)
    spectrum = rfft(x, n=fft_size, axis=0)
    out = np.empty((len(seconds), k, k))
    for i in range(k):
        sums = irfft(spectrum[:, i:i+1].conj() * spectrum[:, i:],
                      n=fft_size, axis=0)
        out[:, i, i:] = ((1 - fraction[:, None]) * sums[lo] / (n-lo)[:, None]
                         + fraction[:, None] * sums[hi] / (n-hi)[:, None])
        reverse = ((1 - fraction[:, None]) * sums[(-lo) % fft_size] / (n-lo)[:, None]
                   + fraction[:, None] * sums[(-hi) % fft_size] / (n-hi)[:, None])
        out[:, i:, i] = reverse
    return out


def lag_weights(seconds: np.ndarray) -> np.ndarray:
    """Trapezoidal weights on strictly positive lags, normalized to sum to 1.

    Excludes zero lag to avoid scoring static covariance as temporal behavior.
    Physical-time weighting prevents the dense early grid dominating distance.
    """
    t = seconds[seconds > 0]
    if len(t) == 1:
        return np.ones(1)
    dt = np.diff(t)
    w = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    return w / w.sum()


def feature_vectors(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Flatten matrices with physical-lag weights and per-entry RMS scaling."""
    dim = values.shape[-1]
    return (values * np.sqrt(weights)[None, :, None, None] / dim).reshape(len(values), -1)


def entry_mask(rank: int, metric: str) -> np.ndarray:
    """Select all directed entries; positive-lag matrices are not symmetrized."""
    diagonal = np.eye(rank, dtype=bool)
    if metric == DIAG_ONLY:
        return diagonal
    if metric == OFFDIAG_ONLY:
        return ~diagonal
    return np.ones((rank, rank), dtype=bool)


def metric_features(values: np.ndarray, weights: np.ndarray, metric: str) -> np.ndarray:
    """RMS over retained entries and the unchanged physical-lag weights.

    diag_only retains k diagonal entries; offdiag_only retains k*(k-1)
    directed off-diagonal entries. Dropped entries do not dilute either RMS.
    Selecting the off-diagonal entries is equivalent to forming K-diag(diag(K))
    and omitting its structural zeros. No input matrix is modified.
    """
    if metric not in (DIAG_ONLY, OFFDIAG_ONLY):
        return feature_vectors(values, weights)
    mask = entry_mask(values.shape[-1], metric)
    if not mask.any():
        raise ValueError("Off-diagonal comparison needs target rank >= 2.")
    selected = values[..., mask]
    return (selected * np.sqrt(weights)[None, :, None] / np.sqrt(mask.sum())).reshape(len(values), -1)


def whiten_lagged_covariance(covariance: np.ndarray, eps_rel: float = 1e-10) -> tuple:
    """Symmetric (ZCA) whitening of a single recording/reference POD covariance.

    Zero lag must be the first matrix. All positive-lag matrices keep their
    direction and asymmetry. Clipping regularizes W but does NOT imply K(0)=I:
    its eigenvalues are lambda_i / max(lambda_i, floor). Report that residual
    without silently replacing the measured zero-lag matrix by an identity.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    if (covariance.ndim != 3 or not len(covariance) or covariance.shape[1] < 1
            or covariance.shape[1] != covariance.shape[2]
            or not np.isfinite(covariance).all()):
        raise ValueError("Whitening needs finite, square lagged covariance matrices.")
    if not np.isfinite(eps_rel) or not 0 < eps_rel < 1:
        raise ValueError("Whitening eps_rel must lie strictly between zero and one.")
    c0 = .5 * (covariance[0] + covariance[0].T)
    eigenvalues, q = np.linalg.eigh(c0)
    minimum, maximum = float(eigenvalues[0]), float(eigenvalues[-1])
    if maximum <= 0 or minimum < -1e-10 * maximum:
        raise ValueError("Zero-lag covariance is not positive semidefinite with positive variance.")
    floor = eps_rel * maximum
    clipped = np.maximum(eigenvalues, floor)
    w = (q * (1 / np.sqrt(clipped))) @ q.T
    w = .5 * (w + w.T)
    kernel = w @ covariance @ w
    kernel[0] = w @ c0 @ w
    error = float(np.linalg.norm(kernel[0] - np.eye(len(c0)), ord="fro"))
    count = int(np.count_nonzero(eigenvalues < floor))
    diagnostics = dict(minimum_eigenvalue=minimum, maximum_eigenvalue=maximum,
                       condition_number=maximum/minimum if minimum > 0 else float("inf"),
                       regularized_condition_number=maximum/float(clipped[0]),
                       eigenvalue_floor=floor, clipped_eigenvalues=count,
                       identity_frobenius_error=error,
                       identity_verified=error <= WHITENING_IDENTITY_TOL)
    return kernel, w, eigenvalues, diagnostics


def whitened_identity_baseline(values: np.ndarray, clipped_counts: np.ndarray) -> tuple:
    """Remove floating-point-only identity residuals for the *static control*.

    Positive-lag features and saved K(0) are never altered. Running a classifier
    on roundoff-sized identity residuals would measure numerical artifacts.
    If clipping/residuals invalidate identity, keep the measured baseline and
    explicitly report that static structure has not been fully removed.
    """
    error = np.linalg.norm(values - np.eye(values.shape[-1]), axis=(-2, -1))
    verified = bool(np.all(error <= WHITENING_IDENTITY_TOL) and not np.any(clipped_counts))
    if verified:
        return np.broadcast_to(np.eye(values.shape[-1]), values.shape).copy(), True
    return values, False


def add_whitening_summary(summary: h5py.Group, covariance: np.ndarray,
                           eps_rel: float, labels: list, references: np.ndarray,
                           output: Path) -> tuple:
    """Derive whitening directly from cached C(tau); no projections are reread."""
    group = summary.create_group("whitening")
    group.attrs["eps_rel"] = eps_rel
    group.attrs["identity_frobenius_tolerance"] = WHITENING_IDENTITY_TOL
    group.attrs["definition"] = "K(tau) = W C(tau) W; W = symmetric inverse square root of symmetrized C(0)"
    group.attrs["axes"] = "recording, reference, lag, coordinate, coordinate"
    kernels = np.empty_like(covariance)
    transforms = np.empty(covariance.shape[:2] + covariance.shape[-2:])
    eigenvalues = np.empty(covariance.shape[:2] + (covariance.shape[-1],))
    diagnostic_arrays = {}
    csv_rows = []
    for i, label in enumerate(labels):
        for ref in range(len(references)):
            k, w, eig, diagnostics = whiten_lagged_covariance(covariance[i, ref], eps_rel)
            kernels[i, ref], transforms[i, ref], eigenvalues[i, ref] = k, w, eig
            for name, value in diagnostics.items():
                if name not in diagnostic_arrays:
                    diagnostic_arrays[name] = np.empty(covariance.shape[:2], dtype=np.asarray(value).dtype)
                diagnostic_arrays[name][i, ref] = value
            csv_rows.append([ref+1, labels[references[ref]], label, eps_rel, *diagnostics.values()])
    for name, values in {"lagged_kernel": kernels, "inverse_sqrt_covariance": transforms,
                         "eigenvalues": eigenvalues, **diagnostic_arrays}.items():
        group.create_dataset(name, data=values, compression="gzip")
    write_csv(output/"whitening_diagnostics.csv", ["reference", "reference_recording", "recording",
              "eps_rel", *diagnostic_arrays], csv_rows)
    return kernels, diagnostic_arrays


def held_out_centroid(features: np.ndarray, groups: np.ndarray,
                      eligible: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    """Leave one whole recording out; all reference recordings are excluded.

    Descriptive validation of fixed features, with no fitted scaling or tuning.
    """
    confusion = np.zeros((count, count), dtype=int)
    for i in np.flatnonzero(eligible):
        training = eligible.copy()
        training[i] = False
        if any(not np.any(training & (groups == p)) for p in range(count)):
            raise ValueError("Too few non-reference recordings for held-out classification.")
        # Anchored means are algebraically the same centroids, but preserve
        # identical features exactly for unequal class sizes. Summing a common
        # 0.1 repeatedly can otherwise break identity-control ties by roundoff.
        centroids = []
        for p in range(count):
            members = features[training & (groups == p)]
            centroids.append(members[0] + (members - members[0]).mean(axis=0))
        centroids = np.stack(centroids)
        predicted = int(np.argmin(np.sum((centroids - features[i])**2, axis=1)))
        confusion[groups[i], predicted] += 1
    totals = confusion.sum(axis=1)
    score = float(np.mean(np.diag(confusion) / totals)) if np.all(totals) else float("nan")
    return confusion, score


def write_csv(path: Path, header: list, rows) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def read_signature(path: Path) -> tuple:
    with h5py.File(path, "r") as f:
        attrs = ("instantaneous_spatial_mean_removed", "temporal_mean_field_removed")
        if any(key not in f.attrs for key in attrs):
            raise ValueError(f"Missing preprocessing metadata: {path}")
        return tuple(bool(f.attrs[key]) for key in attrs) + tuple(
            str(f[name].attrs.get("units", "")) for name in
            ("grid/x", "grid/y", "grid/time", "reduced/coefficients",
             "preprocessing/frame_spatial_mean"))


def checked_sampling_frequency(t: np.ndarray) -> tuple[float, float]:
    """Reject gaps/nonuniform sampling, allowing stored float32 time rounding."""
    fs, _ = infer_sampling_frequency(t)
    error = float(np.max(np.abs((t-t[0])*fs - np.arange(len(t)))))
    if error > .05:
        raise ValueError(f"Time grid departs from uniform sampling by {error:.3g} samples.")
    return fs, error


def analyze_recording(path: Path, reference_modes: np.ndarray,
                      grid: tuple, signature: tuple, args: argparse.Namespace) -> dict:
    source_rank = args.source_ranks[-1]
    modes, x, y = load_basis(path, source_rank)
    _check_grid(*grid, x, y, str(path))
    if read_signature(path) != signature:
        raise ValueError(f"Preprocessing/units differ from the reference: {path}")
    overlap = reference_modes @ modes.T  # reference, target mode, source mode
    with h5py.File(path, "r") as f:
        t = np.asarray(f["grid/time"], dtype=float)
        fs, grid_error = checked_sampling_frequency(t)
        source = np.empty((len(t), source_rank))
        dataset = f["reduced/coefficients"]
        if dataset.shape[0] != len(t) or dataset.shape[1] < source_rank:
            raise ValueError(f"Coefficient shape inconsistent with time or source rank: {path}")
        for start in range(0, len(t), args.batch_size):
            stop = min(start + args.batch_size, len(t))
            source[start:stop] = dataset[start:stop, :source_rank]
        mean = np.asarray(f["preprocessing/frame_spatial_mean"], dtype=float)
    if (mean.shape != (len(t),) or not np.isfinite(source).all()
            or not np.isfinite(mean).all()):
        raise ValueError(f"Invalid coordinates in {path}")
    source -= source.mean(axis=0)
    mean -= mean.mean()
    covariance, correlations, deltas, capture, leading_cosines = [], [], [], [], []
    sample_lags = getattr(args, "single_lag_samples", None)
    if sample_lags is not None:
        if not np.isclose(fs, args.single_lag_reference_fs, rtol=1e-5, atol=0):
            raise ValueError("Single-sample-lag comparison requires matching sampling frequencies.")
        def covariance_at_lags(values):
            return lagged_covariances(values, np.asarray(sample_lags))
    else:
        def covariance_at_lags(values):
            return physical_lag_covariance(values, fs, args.lags_ms / 1000)
    mean_cov = covariance_at_lags(mean[:, None])
    mean_corr = normalize(mean_cov, np.sqrt(np.mean(mean**2))[None])
    for o in overlap:
        aligned = source @ o.T
        energy = float(np.sum(aligned**2))
        if energy <= 0:
            raise ValueError(f"Zero projected energy: {path}")
        # Each error is relative to the largest-source-rank trajectory.
        deltas.append([np.sqrt(np.sum((source[:, :m] @ o[:, :m].T - aligned)**2) / energy)
                       for m in args.source_ranks])
        capture.append([float(np.sum(o[:, :m]**2) / args.rank)
                        for m in args.source_ranks])
        leading_cosines.append(np.linalg.svd(o[:, :args.rank], compute_uv=False))
        c = covariance_at_lags(aligned)
        covariance.append(c)
        correlations.append(normalize(c, np.sqrt(np.mean(aligned**2, axis=0))))
    return dict(covariance=np.asarray(covariance), correlation=np.asarray(correlations),
                mean_covariance=mean_cov, mean_correlation=mean_corr,
                relative_projection_error=np.asarray(deltas),
                reference_subspace_capture=np.asarray(capture),
                leading_principal_cosines=np.asarray(leading_cosines),
                sampling_frequency_hz=fs, sample_count=len(t),
                maximum_time_grid_error_samples=grid_error,
                duration_seconds=t[-1]-t[0])


def distance_plot(distance: np.ndarray, groups: np.ndarray, powers: list[str],
                  ax, title: str, colorbar_label: str = "RMS matrix difference over physical lag") -> None:
    im = ax.imshow(distance, cmap="viridis", vmin=0)
    centers = [np.flatnonzero(groups == p).mean() for p in range(len(powers))]
    ax.set_xticks(centers, powers, rotation=45, ha="right")
    ax.set_yticks(centers, powers)
    for end in np.flatnonzero(np.diff(groups)) + .5:
        ax.axhline(end, color="white", lw=.4)
        ax.axvline(end, color="white", lw=.4)
    ax.set_title(title)
    ax.set_xlabel("Recording, grouped by power")
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, shrink=.75, label=colorbar_label)


def single_matrix_features(matrices: np.ndarray) -> np.ndarray:
    """Euclidean feature distance equals RMS difference of one matrix pair."""
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Expected one square matrix per recording.")
    return matrices.reshape(len(matrices), -1) / matrices.shape[-1]


def summarize_single_lags(cache: h5py.File, args: argparse.Namespace,
                          records: list, references: np.ndarray) -> None:
    """One ordinary correlation matrix at a time, with lag 10 fixed as primary."""
    out = args.output_dir
    labels = [f"{p}/rep{r.number}" for p, r in records]
    groups = np.array([args.powers.index(p) for p, _ in records])
    eligible = np.ones(len(records), bool)
    eligible[references] = False
    correlations = np.stack([cache[f"recordings/{i}/correlation"][:] for i in range(len(records))])
    if "single_lag_summary" in cache:
        del cache["single_lag_summary"]
    summary = cache.create_group("single_lag_summary")
    summary.attrs["primary_lag_samples"] = args.primary_lag_samples
    (out/"summary_configuration.json").write_text(json.dumps(
        dict(primary_lag_samples=args.primary_lag_samples, metric="ordinary_single_lag_correlation",
             distance="RMS matrix difference", whitening=False, combines_lags=False), indent=2)+"\n")
    summary.attrs["distance"] = "sqrt(mean_ij((R_a(ell)-R_b(ell))^2)); one sample lag ell at a time"
    upper = np.triu(np.ones((len(records), len(records)), bool), 1)
    same_power = groups[:, None] == groups[None, :]
    rows, pair_rows, primary_distances, primary_pairs = [], [], [], []
    primary_index = args.single_lag_samples.index(args.primary_lag_samples)
    for ref in range(len(references)):
        for index, lag in enumerate(args.single_lag_samples):
            features = single_matrix_features(correlations[:, ref, index])
            distance = squareform(pdist(features))
            within = distance[upper & same_power]
            between = distance[upper & ~same_power]
            ratio = float(np.median(between)/np.median(within)) if np.median(within)>0 else float("nan")
            try:
                confusion, accuracy = held_out_centroid(features, groups, eligible, len(args.powers))
            except ValueError:
                confusion = np.zeros((len(args.powers), len(args.powers)), int)
                accuracy = float("nan")
            g = summary.create_group(f"reference_{ref+1}/lag_{lag}")
            g.create_dataset("distance", data=distance)
            g.create_dataset("held_out_confusion", data=confusion)
            rows.append([ref+1, lag, args.lags_ms[index], np.median(within), np.median(between), ratio, accuracy])
            for p in range(len(args.powers)):
                for q in range(p, len(args.powers)):
                    selected = upper & (groups[:, None] == p) & (groups[None, :] == q)
                    values = distance[selected]
                    pair_rows.append([ref+1, lag, args.powers[p], args.powers[q], len(values), np.median(values)])
            if lag == args.primary_lag_samples:
                primary_distances.append(distance)
                primary_pairs.append((within, between))
                write_csv(out/f"reference_{ref+1}_lag_{lag}_distances.csv", ["recording", *labels],
                          ([label, *row] for label, row in zip(labels, distance)))
                write_csv(out/f"reference_{ref+1}_lag_{lag}_confusion.csv", ["actual_power / predicted_power", *args.powers],
                          ([power, *row] for power, row in zip(args.powers, confusion)))
    write_csv(out/"single_lag_summary.csv", ["reference", "lag_samples", "lag_ms", "median_within",
              "median_between", "between_within_ratio", "held_out_balanced_accuracy"], rows)
    write_csv(out/"power_pair_distances.csv", ["reference", "lag_samples", "power_a", "power_b",
              "pair_count", "median_distance"], pair_rows)
    with PdfPages(out/"single_lag_comparison.pdf") as pdf:
        fig, axes = plt.subplots(1, len(references), figsize=(6*len(references), 6), squeeze=False, layout="constrained")
        for ref, ax in enumerate(axes.flat):
            distance_plot(primary_distances[ref], groups, args.powers, ax,
                          f"Basis {ref+1}: {labels[references[ref]]}",
                          "RMS difference of two correlation matrices")
        fig.suptitle(f"Single-lag correlation comparison: {args.primary_lag_samples} samples "
                     f"({args.lags_ms[primary_index]:.4f} ms)\nDark = similar; diagonal blocks compare repetitions at the same power")
        pdf.savefig(fig)
        fig.savefig(out/"primary_lag_distance_matrices.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.boxplot([v for pair in primary_pairs for v in pair], showfliers=False)
        ax.set_xticks(np.arange(1, 2*len(references)+1),
                      [f"Basis {r+1}\n{label}" for r in range(len(references))
                       for label in ("Same power", "Different powers")])
        ax.set_ylabel("RMS difference of two correlation matrices")
        ax.set_title(f"Lag {args.primary_lag_samples}: are repetitions more similar than different powers?\n"
                     "Pairwise distances share recordings; distributions can overlap")
        pdf.savefig(fig)
        fig.savefig(out/"primary_lag_within_between.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ref in range(len(references)):
            selected = [r for r in rows if r[0] == ref+1]
            for ax, column in zip(axes, (5, 6)):
                ax.plot([r[1] for r in selected], [r[column] for r in selected], 'o-', label=f"Basis {ref+1}")
                ax.axvline(args.primary_lag_samples, color="gray", lw=.5, alpha=.5)
        axes[0].axhline(1, color="gray", ls="--", lw=.8)
        axes[0].set_ylabel("Median different-power / median same-power distance")
        axes[1].axhline(1/len(args.powers), color="gray", ls="--", lw=.8)
        axes[1].set_ylabel("Held-out balanced accuracy (secondary check)")
        axes[1].set_ylim(0, 1.02)
        for ax in axes:
            ax.set_xlabel("Single delay (samples); 0 is a static baseline")
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle("Fixed candidate delays, each compared separately\nExploratory sensitivity; primary lag is unchanged")
        pdf.savefig(fig)
        fig.savefig(out/"candidate_lags.png", dpi=160)
        plt.close(fig)
        # Show the actual matrices, not only distances between them.
        for ref in range(len(references)):
            means = np.stack([correlations[groups==p, ref, primary_index].mean(axis=0) for p in range(len(args.powers))])
            sd = np.stack([correlations[groups==p, ref, primary_index].std(axis=0, ddof=1) for p in range(len(args.powers))])
            for name, values, cmap, lo, hi in (("mean", means, "RdBu_r", -1, 1),
                                             ("repetition_sd", sd, "magma", 0, float(sd.max()))):
                columns = min(4, len(args.powers))
                nr = (len(args.powers)+columns-1)//columns
                fig, axes = plt.subplots(nr, columns, figsize=(3.6*columns, 3.5*nr), squeeze=False, layout="constrained")
                for p, ax in enumerate(axes.flat):
                    if p >= len(args.powers):
                        ax.set_visible(False)
                        continue
                    im = ax.imshow(values[p], vmin=lo, vmax=hi, cmap=cmap)
                    ax.set_title(args.powers[p])
                    ticks = sorted(set([0, args.rank//2, args.rank-1]))
                    ax.set_xticks(ticks, np.array(ticks)+1)
                    ax.set_yticks(ticks, np.array(ticks)+1)
                    ax.set_xlabel("Coordinate j, later")
                    ax.set_ylabel("Coordinate i, now")
                fig.colorbar(im, ax=axes, shrink=.7, label="Correlation" if name=="mean" else "SD across repetitions")
                fig.suptitle(f"Lag {args.primary_lag_samples}, common basis {ref+1}: "
                             + ("mean correlation matrices" if name=="mean" else "repetition variability of matrix entries"))
                pdf.savefig(fig)
                fig.savefig(out/f"reference_{ref+1}_primary_{name}_matrices.png", dpi=160)
                plt.close(fig)
    primary = [r for r in rows if r[1] == args.primary_lag_samples]
    lines = ["# A single-lag correlation diagnostic", "",
             f"Primary delay: **{args.primary_lag_samples} samples = {args.lags_ms[primary_index]:.6f} ms**. "
             f"{len(records)} recordings, {len(args.powers)} powers, {len(references)} reference POD bases.", "",
             f"**Explanation:** project each recording onto the same {args.rank}-dimensional POD basis; "
             "calculate the correlation of each "
             f"coordinate now with each coordinate {args.primary_lag_samples} samples later; "
             "compare two recordings by the RMS difference of their correlation matrices.", "",
             "For time-centered coordinates, R_ij(ell) = mean_t[a_i(t) a_j(t+ell)]/(sigma_i sigma_j). "
             "The mean uses N-ell pairs and sigma uses the full recording, matching the existing "
             "lagged-correlation definition. Distance = sqrt(mean_ij[(R_A(ell)-R_B(ell))^2]). "
             "No whitening, diagonal splitting, integration across lags, or lag interpolation is used.", "",
             "| Basis | Same-power median distance | Different-power median distance | Different/same | Held-out balanced accuracy |",
             "|---|---|---|---|---|"]
    for r in primary:
        acc = f"{r[6]:.1%}" if np.isfinite(r[6]) else "unavailable"
        lines.append(f"| {r[0]} | {r[3]:.5f} | {r[4]:.5f} | {r[5]:.3f} | {acc} |")
    lines += ["", "## Fixed alternative delays", "",
              "Each row evaluates a single matrix separately; no combination of delays is fitted.", "",
              "| Delay (samples) | Delay (ms) | Between/within ratio across bases | Balanced accuracy across bases |",
              "|---|---|---|---|"]
    for lag in args.single_lag_samples:
        selected = [r for r in rows if r[1]==lag]
        ratios = [r[5] for r in selected]
        accuracy = [r[6] for r in selected]
        acc = f"{min(accuracy):.1%}–{max(accuracy):.1%}" if np.isfinite(accuracy).all() else "unavailable"
        lines.append(f"| {lag} | {selected[0][2]:.6f} | {min(ratios):.3f}–{max(ratios):.3f} | {acc} |")
    lines += ["", "Keep the primary lag as the stated diagnostic. The alternative-delay comparison is "
              "exploratory: a lag chosen for separation on these recordings has not been independently "
              "validated. Larger between/within ratios support greater typical similarity within a "
              "power, but distributions and neighboring powers can overlap. Classification is only a "
              "secondary check; it holds out whole recordings and excludes all basis-reference recordings.", "",
              "Ordinary normalization preserves zero-lag cross-coordinate structure. This simple "
              "diagnostic measures repeatable lagged correlation patterns; it does not isolate temporal "
              "effects from static covariance. Projection uses the same reference selection and "
              f"{args.source_ranks[-1]} source modes as before, with the existing convergence checks "
              "saved in the cache. Pairwise distances are not independent and no significance test is implied.", "",
              "Reference recordings:", *[f"- Basis {r+1}: {labels[i]}" for r, i in enumerate(references)], "",
              "```bash", "python compare_pod_lagged_covariance.py --single-lag-samples "
              + " ".join(map(str, args.single_lag_samples)) + f" --primary-lag-samples {args.primary_lag_samples}",
              "```", "", "The exact input selection and delays are in configuration.json; the primary plot delay is in "
              "summary_configuration.json. Cached results are reused on reruns."]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    cache.flush()
    print("\n".join(lines[:20]), flush=True)


def whitening_interpretation(rows: list, diagnostics: dict,
                             static_rows: list, args: argparse.Namespace) -> list[str]:
    original = [r for r in rows if r[1] == "pod" and r[2] == "correlation"]
    whitened = [r for r in rows if r[1] == "pod" and r[2] == WHITENED]
    errors = diagnostics["identity_frobenius_error"]
    clipped = diagnostics["clipped_eigenvalues"]
    chance = 1 / len(args.powers)
    def span(values, percent=False):
        if not np.isfinite(values).all():
            return "unavailable (too few non-reference recordings)"
        return (f"{min(values):.1%}–{max(values):.1%}" if percent
                else f"{min(values):.3f}–{max(values):.3f}")
    paragraphs = [
        "Question: does input-power separation remain after all zero-lag covariance structure "
        "has been removed? For each recording/reference, only the POD coordinates are whitened: "
        "C0 = (C(0) + C(0).T)/2, W = C0^(-1/2), and K(tau) = W C(tau) W. "
        "The spatial mean is still scored separately.",
        f"Relative eigenvalue floor: {args.whitening_eps_rel:g}. Across {errors.size} "
        f"recording/reference combinations, median ||K(0)-I||_F = {np.median(errors):.3e}; "
        f"maximum = {errors.max():.3e}. Identity tolerance: {WHITENING_IDENTITY_TOL:g}. "
        f"Clipped eigenvalues: {int(clipped.sum())} across {np.count_nonzero(clipped)} combinations. "
        f"Largest unregularized condition number: {diagnostics['condition_number'].max():.3g}. "
        "Per-recording eigenvalues, condition numbers and residuals are in whitening_diagnostics.csv.",
        f"On positive lags {args.lags_ms[1]:g}–{args.lags_ms[-1]:g} ms, original normalized correlation: "
        f"between/within median-distance ratio {span([r[5] for r in original])}, held-out balanced "
        f"accuracy {span([r[6] for r in original], True)}. Full whitening: ratio "
        f"{span([r[5] for r in whitened])}, accuracy {span([r[6] for r in whitened], True)}. "
        f"Balanced-chance accuracy is {chance:.1%}.",
    ]
    verified = all(r[3] for r in static_rows)
    if verified:
        paragraphs.append(
            f"All zero-lag matrices pass the identity check without clipping. Their largest measured "
            f"pairwise zero-lag RMS distance is {max(r[1] for r in static_rows):.3e}. "
            "For the static classifier control only, verified roundoff-sized deviations are replaced "
            "by exact I; saved K(0) and all positive lags are unchanged. Identical features contain "
            f"no power information and give balanced accuracy {span([r[4] for r in static_rows], True)} "
            "with deterministic centroid ties. The zero-lag distance ratio is undefined (0/0), "
            "not evidence of separation. See whitened_zero_lag_baseline.csv.")
        if not all(np.isfinite(r[6]) for r in whitened):
            paragraphs.append(
                "There are too few non-reference recordings for the held-out classifier. "
                "The distance summaries remain descriptive; missing classification results "
                "cannot be interpreted as a loss of temporal separation.")
        elif all(r[5] > 1 and r[6] > chance for r in whitened):
            paragraphs.append(
                "At the full lag range, every reference retains larger between-power median distances "
                "and above-chance held-out accuracy after whitening. This supports characteristic "
                "temporal dependence in these reduced states beyond their zero-lag covariance. "
                "Judge the strength from the effect sizes and overlap, especially between neighboring "
                "powers; it does not imply perfect separation or identify a unique dynamical law.")
        else:
            paragraphs.append(
                "Whitening does not retain both larger between-power median distances and above-chance "
                "accuracy consistently across references. A substantial drop toward ratio 1 and chance "
                "accuracy is evidence that much of the original separation came from static covariance; "
                "compare the numerical changes and individual references before drawing that conclusion.")
    else:
        paragraphs.append(
            "Identity is not verified without clipping for every recording. With eigenvalue flooring, "
            "K(0) has eigenvalues lambda_i / max(lambda_i, floor), so it need not equal I. "
            "The static baseline uses the measured matrices in affected references. Do not claim "
            "complete removal of zero-lag structure for this configuration; inspect the diagnostics.")
    paragraphs.append(
        "The fixed diagnostic windows end at 0.5, 1, 2, 5, 10 and 20 ms on the default grid; "
        "every window excludes zero lag. All windows are reported, with no best-window selection. "
        "Pairwise distances share recordings and are not independent. No window tuning, independent "
        "test set or formal significance test is implied. Whitening matches instantaneous second "
        "moments; it does not remove higher-order static structure or establish a causal mechanism.")
    return paragraphs


def component_interpretation(rows: list, args: argparse.Namespace) -> list[str]:
    """Describe diagonal/off-diagonal attribution without selecting a lag window."""
    groups = {metric: [r for r in rows if r[1] == metric]
              for metric in COMPONENT_LABELS.values()}
    text = [
        "Question: is power-dependent temporal separation carried by individual-coordinate "
        "autocorrelations, or also by lagged cross-coordinate structure? diag_only retains "
        "K_ii(tau); offdiag_only retains all ordered K_ij(tau), i != j, without symmetrization. "
        "These are whitened POD coordinates: whitening mixes the original POD modes, so the "
        "diagonal/off-diagonal split is coordinate-dependent and is checked across reference bases.",
        f"Distances use the same positive physical lags ({args.lags_ms[1]:g}–{args.lags_ms[-1]:g} ms) "
        f"and trapezoidal weights. RMS is averaged over {args.rank**2} entries for full_whitened, "
        f"{args.rank} for diag_only, and {args.rank*(args.rank-1)} for offdiag_only. Structural "
        "zeros are excluded. For each recording pair, d_full^2 = d_diag^2/k + "
        "(k-1)*d_offdiag^2/k. Within/between ratios and standalone centroid predictions are "
        "unchanged by this constant per-component rescaling; raw distance magnitudes are "
        "per-retained-entry RMS, not fractions of explained separation.",
    ]
    for metric, selected in groups.items():
        if not selected:
            text.append(f"{metric}: unavailable at target rank {args.rank}; no off-diagonal entries exist.")
            continue
        ratios = [r[4] for r in selected]
        accuracies = [r[5] for r in selected]
        accuracy = (f"{min(accuracies):.1%}–{max(accuracies):.1%}"
                    if np.isfinite(accuracies).all() else "unavailable (too few non-reference recordings)")
        text.append(f"{metric}: between/within median-distance ratio {min(ratios):.3f}–{max(ratios):.3f}; "
                    f"held-out balanced accuracy {accuracy} across reference bases.")
    off, diag, full = groups[OFFDIAG_ONLY], groups[DIAG_ONLY], groups['full_whitened']
    if off and all(np.isfinite(r[5]) for r in off):
        if all(r[4] > 1 and r[5] > 1/len(args.powers) for r in off):
            text.append(
                "Off-diagonal entries retain larger between-power median distances and above-chance "
                "classification in every reference. Thus the observed dependence is not confined "
                "to individual-coordinate autocorrelations: lagged cross-coordinate structure also "
                "carries reproducible power information. Assess its strength from the ratios, accuracy "
                "and distribution overlap; this is not a formal significance conclusion.")
        else:
            text.append(
                "Off-diagonal entries do not retain both larger between-power median distances and "
                "above-chance accuracy in every reference. If their ratios approach one and accuracy "
                "approaches chance while diagonal performance stays close to full performance, "
                "the separation is mostly attributable to autocorrelation (and related spectral) structure.")
    if full and diag and all(np.isfinite(r[5]) for r in full+diag):
        difference = np.mean([r[5] for r in diag])-np.mean([r[5] for r in full])
        text.append(
            f"Diagonal-only balanced accuracy differs from the full statistic by {100*difference:+.2f} "
            "percentage points on average over references. Standalone off-diagonal predictability "
            "does not establish an incremental gain beyond the diagonal features or conditional "
            "independence from them. The full metric weights both sets by their entry counts.")
    text.append(
        "The diagonal describes temporal memory/autocorrelation of whitened coordinates and is "
        "closely related to their spectra. Off-diagonal lagged correlations describe cross-coordinate "
        "dependence (also related to cross-spectra), not necessarily causal or nonlinear mode coupling. "
        "All six fixed positive-lag windows are reported without selecting a best window. Pairwise "
        "distances share recordings; no independent test set or formal significance test is implied.")
    return text


def summarize(cache: h5py.File, args: argparse.Namespace, records: list,
              references: np.ndarray) -> None:
    out = args.output_dir
    groups = np.array([args.powers.index(p) for p, _ in records])
    labels = [f"{p}/rep{r.number}" for p, r in records]
    eligible = np.ones(len(records), dtype=bool)
    eligible[references] = False
    data = {name: np.stack([cache[f"recordings/{i}/{name}"][()] for i in range(len(records))])
            for name in ("covariance", "correlation", "mean_covariance", "mean_correlation",
                         "relative_projection_error", "reference_subspace_capture",
                         "leading_principal_cosines")}
    positive = args.lags_ms > 0
    weights = lag_weights(args.lags_ms / 1000)
    if "summary" in cache:
        del cache["summary"]
    summary = cache.create_group("summary")
    # Keep the original projection/cache configuration unchanged. Whitening
    # only depends on the saved covariance matrices, so its settings belong
    # to a separate, reproducible derived-summary configuration.
    summary_config = dict(version=3, whitening_eps_rel=args.whitening_eps_rel,
                          whitening_identity_tolerance=WHITENING_IDENTITY_TOL,
                          static_control="exact identity only if no clipping and numerical identity verified",
                          fixed_window_endpoints_ms=[.5, 1., 2., 5., 10., 20.],
                          whitened_components="full_whitened, diag_only, offdiag_only (rank >= 2)",
                          component_rms="mean over retained entries: k^2 full, k diagonal, k*(k-1) directed off-diagonal")
    summary.attrs["configuration"] = json.dumps(summary_config, sort_keys=True)
    (out/"summary_configuration.json").write_text(json.dumps(summary_config, indent=2)+"\n")
    data[WHITENED], whitening = add_whitening_summary(
        summary, data["covariance"], args.whitening_eps_rel, labels, references, out)
    # All component features select from the same K; no recomputation of
    # projections, whitening, or lag covariances and no duplicate kernel cache.
    split_metrics = (DIAG_ONLY, OFFDIAG_ONLY) if args.rank > 1 else (DIAG_ONLY,)
    for metric in split_metrics:
        data[metric] = data[WHITENED]
    analysis_metrics = POD_METRICS + split_metrics
    component_metrics = (WHITENED,) + split_metrics
    rows, pair_rows, diagnostic_rows, results = [], [], [], []
    for ref in range(len(references)):
        for i, label in enumerate(labels):
            for j, source_rank in enumerate(args.source_ranks):
                diagnostic_rows.append([ref+1, labels[references[ref]], label, source_rank,
                    data["relative_projection_error"][i, ref, j],
                    data["reference_subspace_capture"][i, ref, j],
                    data["leading_principal_cosines"][i, ref].min()])
        for component in ("pod", "spatial_mean"):
            # Spatial mean is independent of the POD reference, so save it once.
            if component == "spatial_mean" and ref:
                continue
            for metric in (analysis_metrics if component == "pod" else ("correlation", "covariance")):
                v = (data[metric][:, ref] if component == "pod"
                     else data[f"mean_{metric}"])
                features = metric_features(v[:, positive], weights, metric)
                distance = squareform(pdist(features))
                key = f"reference_{ref+1}/{component}/{metric}"
                g = summary.create_group(key)
                g.create_dataset("distance", data=distance)
                if metric in component_metrics and component == "pod":
                    mask = entry_mask(args.rank, metric)
                    g.create_dataset("entry_mask", data=mask)
                    g.attrs["retained_entries"] = int(mask.sum())
                    g.attrs["normalization"] = "RMS over retained entries and physical-lag weights"
                try:
                    confusion, accuracy = held_out_centroid(features, groups, eligible, len(args.powers))
                except ValueError:
                    confusion = np.zeros((len(args.powers), len(args.powers)), dtype=int)
                    accuracy = float("nan")
                g.create_dataset("held_out_confusion", data=confusion)
                g.attrs["held_out_balanced_accuracy"] = accuracy
                upper = np.triu(np.ones(distance.shape, dtype=bool), 1)
                within = distance[upper & (groups[:, None] == groups[None, :])]
                between = distance[upper & (groups[:, None] != groups[None, :])]
                ratio = float(np.median(between) / np.median(within)) if np.median(within) > 0 else float("nan")
                rows.append([ref+1, component, metric, np.median(within), np.median(between), ratio, accuracy])
                results.append((ref, component, metric, distance, within, between, accuracy, ratio))
                for p in range(len(args.powers)):
                    for q in range(p, len(args.powers)):
                        selected = upper & (((groups[:, None] == p) & (groups[None, :] == q)) |
                                            ((groups[:, None] == q) & (groups[None, :] == p)))
                        values = distance[selected]
                        pair_rows.append([ref+1, component, metric, args.powers[p], args.powers[q],
                                          len(values), np.mean(values), np.median(values)])
                if ref == 0 or metric in split_metrics:
                    prefix = "" if ref == 0 else f"reference_{ref+1}_"
                    write_csv(out/f"{prefix}{component}_{metric}_distances.csv", ["recording", *labels],
                              ([label, *row] for label, row in zip(labels, distance)))
                write_csv(out/f"reference_{ref+1}_{component}_{metric}_confusion.csv",
                          ["actual_power / predicted_power", *args.powers],
                          ([p, *row] for p, row in zip(args.powers, confusion)))
    write_csv(out/"distance_summary.csv", ["reference", "component", "metric", "median_within",
              "median_between", "between_within_ratio", "held_out_balanced_accuracy"], rows)
    component_rows = [[r[0], COMPONENT_LABELS[r[2]], *r[3:]] for r in rows
                      if r[1] == "pod" and r[2] in component_metrics]
    write_csv(out/"whitened_components_comparison.csv", ["reference", "metric", "median_within",
              "median_between", "between_within_ratio", "held_out_balanced_accuracy"], component_rows)
    write_csv(out/"power_pair_distances.csv", ["reference", "component", "metric", "power_a",
              "power_b", "pair_count", "mean_distance", "median_distance"], pair_rows)
    write_csv(out/"projection_diagnostics.csv", ["reference", "reference_recording", "recording",
              "source_rank", "trajectory_error_relative_to_largest_source_rank",
              "reference_subspace_capture", "minimum_leading_principal_cosine"], diagnostic_rows)
    # Fixed lag-window sensitivity and zero-lag baseline. These are reported
    # together, without picking a window based on classification performance.
    window_rows, static_control_rows = [], []
    windows = sorted(set([0., *[v for v in (.5, 1., 2., 5., 10.) if v < args.lags_ms[-1]],
                          float(args.lags_ms[-1])]))
    for ref in range(len(references)):
        for metric in analysis_metrics:
            for window in windows:
                if metric in split_metrics and window == 0:
                    continue  # Component attribution uses positive lags only.
                selected = ((args.lags_ms > 0) & (args.lags_ms <= window + 1e-9)
                            if window else args.lags_ms == 0)
                if not np.any(selected):
                    continue
                w = lag_weights(args.lags_ms[selected]/1000) if window else np.ones(1)
                values = data[metric][:, ref, selected]
                if metric == WHITENED and window == 0:
                    measured_max_distance = float(pdist(feature_vectors(values, w)).max())
                    values, identity_control = whitened_identity_baseline(
                        values, whitening["clipped_eigenvalues"][:, ref])
                features = metric_features(values, w, metric)
                d = squareform(pdist(features))
                upper = np.triu(np.ones(d.shape, dtype=bool), 1)
                within = d[upper & (groups[:, None] == groups[None, :])]
                between = d[upper & (groups[:, None] != groups[None, :])]
                try:
                    _, accuracy = held_out_centroid(features, groups, eligible, len(args.powers))
                except ValueError:
                    accuracy = float("nan")
                ratio = float(np.median(between)/np.median(within)) if np.median(within)>0 else float("nan")
                window_rows.append([ref+1, metric, window, np.median(within),
                                    np.median(between), ratio, accuracy])
                if metric == WHITENED and window == 0:
                    static_control_rows.append([ref+1, measured_max_distance,
                        float(whitening["identity_frobenius_error"][:, ref].max()),
                        identity_control, accuracy])
    write_csv(out/"lag_window_sensitivity.csv", ["reference", "metric", "maximum_lag_ms_zero_is_static_baseline",
              "median_within", "median_between", "between_within_ratio", "held_out_balanced_accuracy"], window_rows)
    write_csv(out/"whitened_zero_lag_baseline.csv", ["reference", "measured_maximum_zero_lag_rms_distance",
              "maximum_identity_frobenius_error", "identity_control_used", "held_out_balanced_accuracy"], static_control_rows)
    whitening_text = whitening_interpretation(rows, whitening, static_control_rows, args)
    component_text = component_interpretation(component_rows, args)
    with PdfPages(out/"cross_power_lagged_covariance.pdf") as pdf:
        for ref in range(len(references)):
            fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
            for ax, metric in zip(axes, ("correlation", "covariance")):
                d = summary[f"reference_{ref+1}/pod/{metric}/distance"][:]
                distance_plot(d, groups, args.powers, ax, f"POD {metric}")
            fig.suptitle(f"Common basis {ref+1}: {labels[references[ref]]} | target rank {args.rank}\n"
                         f"Source rank {args.source_ranks[-1]}, positive lags through {args.lags_ms[-1]:g} ms")
            pdf.savefig(fig)
            fig.savefig(out/f"reference_{ref+1}_distance_matrices.png", dpi=160)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
            distance_plot(summary[f"reference_{ref+1}/pod/{WHITENED}/distance"][:],
                          groups, args.powers, ax, "POD full zero-lag-whitened statistic")
            fig.suptitle(f"Common basis {ref+1}: {labels[references[ref]]}\n"
                         f"Positive lags {args.lags_ms[1]:g}–{args.lags_ms[-1]:g} ms")
            pdf.savefig(fig)
            fig.savefig(out/f"reference_{ref+1}_whitened_distance_matrix.png", dpi=160)
            plt.close(fig)
            fig, axes = plt.subplots(1, len(component_metrics), figsize=(6*len(component_metrics), 6),
                                     squeeze=False, layout="constrained")
            for ax, metric in zip(axes.flat, component_metrics):
                distance_plot(summary[f"reference_{ref+1}/pod/{metric}/distance"][:],
                              groups, args.powers, ax, COMPONENT_LABELS[metric])
            fig.suptitle(f"Whitened entry comparison | reference {ref+1}: {labels[references[ref]]}\n"
                         "Distances are RMS per retained entry; positive lags only")
            pdf.savefig(fig)
            fig.savefig(out/f"reference_{ref+1}_whitened_component_distances.png", dpi=160)
            plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
        for ax, metric in zip(axes, POD_METRICS):
            chosen = [r for r in results if r[1] == "pod" and r[2] == metric]
            arrays = [a for r in chosen for a in (r[4], r[5])]
            tick_labels = [f"Ref {r[0]+1}\n{kind}" for r in chosen for kind in ("within", "between")]
            ax.boxplot(arrays, showfliers=False)
            ax.set_xticks(np.arange(1, len(tick_labels)+1), tick_labels)
            ax.set_title(f"{METRIC_TITLES[metric]}: all recording pairs")
            ax.set_ylabel("Distance (pairwise values are not independent)")
        pdf.savefig(fig)
        fig.savefig(out/"within_between_distances.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, len(component_metrics), figsize=(6*len(component_metrics), 5),
                                 squeeze=False, layout="constrained")
        for ax, metric in zip(axes.flat, component_metrics):
            chosen = [r for r in results if r[1] == "pod" and r[2] == metric]
            arrays = [a for r in chosen for a in (r[4], r[5])]
            labels_box = [f"Ref {r[0]+1}\n{kind}" for r in chosen for kind in ("within", "between")]
            ax.boxplot(arrays, showfliers=False)
            ax.set_xticks(np.arange(1, len(labels_box)+1), labels_box)
            ax.set_title(COMPONENT_LABELS[metric])
            ax.set_ylabel("RMS distance per retained entry")
        fig.suptitle("Whitened components: within- versus between-power distances\n"
                     "Pairwise values share recordings and are not independent")
        pdf.savefig(fig)
        fig.savefig(out/"whitened_components_within_between.png", dpi=160)
        plt.close(fig)
        for metric, filename in (("correlation", "lag_window_sensitivity.png"),
                                  (WHITENED, "whitened_lag_window_sensitivity.png"),
                                  *((m, f"{m}_lag_window_sensitivity.png") for m in split_metrics)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
            for ref in range(len(references)):
                selected = [r for r in window_rows if r[0] == ref+1 and r[1] == metric]
                axes[0].plot([r[2] for r in selected], [r[5] for r in selected],
                             'o-', label=f"Reference {ref+1}")
                axes[1].plot([r[2] for r in selected], [r[6] for r in selected],
                             'o-', label=f"Reference {ref+1}")
            axes[0].axhline(1, color="gray", ls="--", lw=.8)
            axes[0].set_ylabel("Median between / median within distance")
            axes[1].axhline(1/len(args.powers), color="gray", ls="--", lw=.8)
            axes[1].set_ylabel("Held-out balanced accuracy")
            axes[1].set_ylim(0, 1.02)
            for ax in axes:
                ax.set_xlabel("Maximum included positive lag (ms)" if metric in split_metrics
                              else "Maximum included lag (ms); 0 = static control")
                ax.grid(alpha=.2)
                ax.legend()
            subtitle = ("\nVerified whitening: static control uses exact I to exclude roundoff; its distance ratio is undefined."
                        if metric == WHITENED and all(r[3] for r in static_control_rows) else "")
            fig.suptitle(f"{METRIC_TITLES[metric]}: fixed lag-window sensitivity{subtitle}", fontsize=12)
            pdf.savefig(fig)
            fig.savefig(out/filename, dpi=160)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        ax.set_axis_off()
        cells = [[r[0], r[1], f"{r[2]:.5g}", f"{r[3]:.5g}", f"{r[4]:.3f}",
                  f"{r[5]:.1%}" if np.isfinite(r[5]) else "unavailable"] for r in component_rows]
        table = ax.table(cellText=cells,
                         colLabels=["Reference", "Metric", "Median within", "Median between",
                                    "Between/within", "Balanced accuracy"],
                         colWidths=[.1, .22, .16, .16, .16, .2],
                         cellLoc="center", bbox=[0, .18, 1, .68])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title(f"Whitened components | positive lags {args.lags_ms[1]:g}–{args.lags_ms[-1]:g} ms",
                     fontsize=15)
        ax.text(0, .06, "RMS distances average over retained entries. Whole recordings are held out; reference recordings are excluded.\n"
                "No window selection or formal significance test. Compare component ratios and accuracy, not just raw distances.",
                transform=ax.transAxes, fontsize=10)
        pdf.savefig(fig)
        fig.savefig(out/"whitened_components_comparison.png", dpi=160)
        plt.close(fig)
        for paragraphs in (component_text[:5], component_text[5:]):
            if not paragraphs:
                continue
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(.06, .94, "Autocorrelation versus lagged cross-coordinate structure", fontsize=16, va="top")
            fig.text(.06, .87, "\n\n".join(textwrap.fill(p, width=115) for p in paragraphs),
                     fontsize=10, va="top", linespacing=1.4)
            pdf.savefig(fig)
            plt.close(fig)
        # Interpretation is part of the PDF as well as the Markdown report.
        for paragraphs in (whitening_text[:3], whitening_text[3:]):
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(.06, .94, "Does separation remain after zero-lag whitening?", fontsize=16, va="top")
            body = "\n\n".join(textwrap.fill(p, width=115) for p in paragraphs)
            fig.text(.06, .87, body, fontsize=10, va="top", linespacing=1.4)
            pdf.savefig(fig)
            plt.close(fig)
        # Means and repetition SDs; no inference based on correlated time samples.
        colors = plt.get_cmap("tab10").colors
        for component in ("pod", "spatial_mean"):
            dimensions = args.rank if component == "pod" else 1
            columns = 2 if dimensions > 1 else 1
            nr = (dimensions + columns - 1) // columns
            fig, axes = plt.subplots(nr, columns, figsize=(13, max(4, nr*2.5)),
                                     squeeze=False, layout="constrained", sharex=True)
            values = data["correlation"][:, 0] if component == "pod" else data["mean_correlation"]
            for mode, ax in enumerate(axes.flat):
                if mode >= dimensions:
                    ax.set_visible(False)
                    continue
                for p, power in enumerate(args.powers):
                    curves = values[groups == p, :, mode, mode]
                    mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
                    ax.plot(args.lags_ms, mean, color=colors[p % 10], label=power, lw=1)
                    ax.fill_between(args.lags_ms, mean-sd, mean+sd, color=colors[p % 10], alpha=.10)
                ax.axhline(0, color="gray", lw=.5)
                ax.set_title(f"Mode {mode+1}" if component == "pod" else "Spatial mean")
                ax.set_xlabel("Lag (ms)")
                ax.set_ylabel("Autocorrelation")
                ax.grid(alpha=.2)
            axes[0, 0].legend(ncol=4, fontsize=7)
            fig.suptitle(f"{component.replace('_', ' ').title()}: mean ± SD across repetitions\n"
                         f"Reference 1: {labels[references[0]]}")
            pdf.savefig(fig)
            fig.savefig(out/f"{component}_autocorrelations.png", dpi=160)
            plt.close(fig)
    lines = ["# Shared-basis lagged covariance comparison", "",
             f"{len(records)} recordings; powers: {', '.join(args.powers)}.",
             f"Target rank {args.rank}; source ranks {args.source_ranks}; seed {args.seed}.",
             "", "Reference recordings:", *[f"- {labels[i]}" for i in references], "",
             "Distances are physical-time-weighted RMS matrix differences over positive lags "
             f"({args.lags_ms[1]:g}–{args.lags_ms[-1]:g} ms). Zero lag is excluded. "
             "Correlation uses each recording's zero-lag coordinate standard deviations. "
             "Covariance retains amplitudes. POD and spatial mean are scored separately.", "",
             "| Reference | Component | Metric | Median within | Median between | Between/within | Held-out balanced accuracy |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        accuracy_text = f"{r[6]:.1%}" if np.isfinite(r[6]) else "unavailable (too few non-reference recordings)"
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.5g} | {r[4]:.5g} | {r[5]:.3f} | {accuracy_text} |")
    baseline = [r[6] for r in window_rows if r[1] == "correlation" and r[2] == 0]
    temporal = [r[6] for r in window_rows if r[1] == "correlation" and r[2] == args.lags_ms[-1]]
    if np.isfinite(baseline).all() and np.isfinite(temporal).all():
        lines += ["", f"Zero-lag correlation alone achieves {min(baseline):.1%}–{max(baseline):.1%} "
                  f"balanced accuracy; the full positive-lag comparison achieves "
                  f"{min(temporal):.1%}–{max(temporal):.1%}. Compare these baselines before "
                  "attributing power separation specifically to temporal information."]
    lines += ["", "## Full zero-lag whitening", "", "\n\n".join(whitening_text)]
    lines += ["", "## Diagonal versus off-diagonal whitened structure", "",
              "| Reference | Metric | Median within | Median between | Between/within | Held-out balanced accuracy |",
              "|---|---|---|---|---|---|"]
    for r in component_rows:
        accuracy_text = f"{r[5]:.1%}" if np.isfinite(r[5]) else "unavailable"
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.5g} | {r[3]:.5g} | {r[4]:.3f} | {accuracy_text} |")
    lines += ["", "\n\n".join(component_text), "",
              "See whitened_components_comparison.csv, whitened_components_within_between.png, "
              "reference_*_whitened_component_distances.png, diag_only_lag_window_sensitivity.png "
              "and offdiag_only_lag_window_sensitivity.png. Component rows also appear in "
              "distance_summary.csv, power_pair_distances.csv and lag_window_sensitivity.csv. "
              "Confusion matrices use reference_*_pod_{diag_only,offdiag_only}_confusion.csv."]
    lines += ["", "Classification leaves one entire recording out; every reference recording is "
              "excluded from both training and testing. Features and lag grid are fixed. "
              "No tuning, independent test set, or significance test is implied. "
              "Pair distances share recordings and must not be treated as independent replicates. "
              "lag_window_sensitivity.csv reports all fixed windows and a zero-lag-only baseline; "
              "no best window is selected.", "",
              "## Projection checks", ""]
    for j, rank in enumerate(args.source_ranks[:-1]):
        error = data["relative_projection_error"][:, :, j]
        lines.append(f"- Source rank {rank} versus {args.source_ranks[-1]}: median trajectory RMS error "
                     f"{np.median(error):.3%}, maximum {np.max(error):.3%}.")
    captures = data["reference_subspace_capture"][:, :, -1]
    lines += [f"- Reference subspace capture at largest source rank: {captures.min():.6f}–{captures.max():.6f}.",
              f"- Smallest principal cosine between leading rank-{args.rank} subspaces: "
              f"{data['leading_principal_cosines'].min():.6f}.", "",
              "Projection uses saved truncated reconstructions, not raw fields. The largest source "
              "rank is the convergence reference, not ground truth; increase it if errors remain "
              "material. Grid coordinates, units, preprocessing, orthonormality, and time spacing "
              "are checked. Fractional physical lags linearly interpolate adjacent sample-lag covariances.", "",
              "A between/within ratio above one is descriptive and does not establish separation "
              "of every neighboring power. Inspect power_pair_distances.csv and held-out confusion "
              "matrices. Normalized correlation can still reflect zero-lag cross-coordinate structure; "
              "it does not isolate a unique dynamical law or nonlinear coupling."]
    lines += ["", "## Reproduce", "", "From the CapillaryWaveTurbulence repository, in the capillarywave environment:",
              "", "```bash", "python compare_pod_lagged_covariance.py "
              f"--powers {' '.join(args.powers)} --rank {args.rank} "
              f"--source-ranks {' '.join(map(str, args.source_ranks))} "
              f"--references {args.references} --seed {args.seed} "
              f"--whitening-eps-rel {args.whitening_eps_rel:g}", "```", "",
              "Use configuration.json for the exact input paths, physical lag grid and recording "
              "selection (including any subset). Repeating the same command resumes cached "
              "recordings and regenerates summaries. Use a new --output-dir when changing inputs "
              "or projection/lag settings. Whitening settings are recorded separately in "
              "summary_configuration.json; changing --whitening-eps-rel reuses the original "
              "covariance cache and regenerates derived whitening and summaries."]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    cache.flush()
    print("\n".join(lines[:24]), flush=True)


def run(args: argparse.Namespace) -> None:
    single_lags = getattr(args, "single_lag_samples", None)
    records = []
    for power in args.powers:
        repetitions = discover_repetitions(args.reduced_root / power, args.pod_rank)
        if args.max_repetitions:
            repetitions = repetitions[:args.max_repetitions]
        if len(repetitions) < 2:
            raise ValueError(f"Need at least two repetitions for {power}.")
        records.extend((power, repetition) for repetition in repetitions)
    if args.references >= len(records):
        raise ValueError("Number of references must be smaller than recording count.")
    references = np.random.default_rng(args.seed).choice(len(records), args.references, replace=False)
    reference_bases = [load_basis(records[i][1].pod_path, args.rank) for i in references]
    grid = reference_bases[0][1:]
    signature = read_signature(records[references[0]][1].pod_path)
    if signature[4] not in ("s", "seconds", "second"):
        raise ValueError(f"Expected time in seconds, got {signature[4]!r}")
    for i, (_, x, y) in zip(references, reference_bases):
        _check_grid(*grid, x, y, str(records[i][1].pod_path))
    reference_modes = np.stack([b[0] for b in reference_bases])
    if single_lags is not None:
        with h5py.File(records[references[0]][1].pod_path) as source:
            args.single_lag_reference_fs, _ = checked_sampling_frequency(np.asarray(source["grid/time"], dtype=float))
        args.lags_ms = np.asarray(single_lags, dtype=float)*1000/args.single_lag_reference_fs
    config = dict(schema=SCHEMA_VERSION, powers=args.powers, rank=args.rank,
                  pod_rank=args.pod_rank, source_ranks=args.source_ranks,
                  seed=args.seed, references=references.tolist(), lags_ms=args.lags_ms.tolist(),
                  records=[dict(power=p, repetition=r.number, path=str(r.pod_path),
                                size=r.pod_path.stat().st_size, mtime_ns=r.pod_path.stat().st_mtime_ns)
                           for p, r in records])
    if single_lags is not None:
        config.update(mode="single_lag_correlation", sample_lags=single_lags,
                      sampling_frequency_reference_hz=args.single_lag_reference_fs)
    serialized = json.dumps(config, sort_keys=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / ("single_lag_correlations.h5" if single_lags is not None
                              else "cross_power_lagged_covariance.h5")
    with h5py.File(path, "a") as cache:
        if "configuration" in cache.attrs and cache.attrs["configuration"] != serialized:
            raise ValueError("Output has a different configuration/input metadata. Use another --output-dir.")
        cache.attrs["configuration"] = serialized
        cache.attrs["covariance_definition"] = "C_ij(tau) = mean_t[a_i(t) * a_j(t+tau)]; coordinates time-centered"
        cache.attrs["correlation_definition"] = "C_ij(tau) / sqrt(C_ii(0) * C_jj(0)); per recording"
        cache.attrs["pod_coefficient_units"] = signature[5]
        cache.attrs["spatial_mean_units"] = signature[6]
        cache.attrs["covariance_units"] = "square of corresponding coordinate units"
        cache.attrs["distance_definition"] = ("RMS over matrix entries at one exact sample lag" if single_lags is not None
                                               else "RMS over matrix entries and trapezoid-weighted positive physical lags")
        if "lags_ms" not in cache:
            cache.create_dataset("lags_ms", data=args.lags_ms)
            cache.create_dataset("reference_indices", data=references)
            cache.create_dataset("reference_modes", data=reference_modes, compression="gzip")
            if single_lags is not None:
                cache.create_dataset("lag_samples", data=single_lags)
        cache.require_group("recordings")
        (args.output_dir/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
        print(f"{len(records)} recordings, {len(references)} references, target rank {args.rank}, "
              f"source ranks {args.source_ranks}", flush=True)
        for i in references:
            print(f"Reference: {records[i][0]} / {records[i][1].name}", flush=True)
        for i, (power, repetition) in enumerate(records):
            key = f"recordings/{i}"
            if key in cache and cache[key].attrs.get("complete", False):
                with h5py.File(repetition.pod_path, "r") as source:
                    _, grid_error = checked_sampling_frequency(np.asarray(source["grid/time"], dtype=float))
                if "maximum_time_grid_error_samples" not in cache[key]:
                    cache[key].create_dataset("maximum_time_grid_error_samples", data=grid_error)
                print(f"[{i+1}/{len(records)}] cached {power}/rep{repetition.number}", flush=True)
                continue
            result = analyze_recording(repetition.pod_path, reference_modes, grid, signature, args)
            if key in cache:
                del cache[key]
            g = cache.create_group(key)
            g.attrs["power"], g.attrs["repetition"], g.attrs["source_file"] = power, repetition.number, str(repetition.pod_path)
            for name, values in result.items():
                g.create_dataset(name, data=values, **({"compression": "gzip"} if np.ndim(values) else {}))
            g.attrs["complete"] = True
            cache.flush()
            error = result["relative_projection_error"][:, -2].max()
            print(f"[{i+1}/{len(records)}] {power}/rep{repetition.number}: "
                  f"max {args.source_ranks[-2]}→{args.source_ranks[-1]} projection error {error:.3%}", flush=True)
        if single_lags is not None:
            summarize_single_lags(cache, args, records, references)
        else:
            summarize(cache, args, records, references)
    report_name = "single_lag_comparison.pdf" if single_lags is not None else "cross_power_lagged_covariance.pdf"
    print(f"Saved report: {args.output_dir / report_name}", flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)
