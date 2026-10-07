"""Compare experimental center bispectra across repetitions and forcing powers.

Examples, from the repository with the conda environment active::

    python compare_bispectral_repetitions.py 0p20
    python compare_bispectral_repetitions.py 0p20 0p35

Each experimental repetition is transformed separately and cached. The script
does not use POD coefficients: the POD files supply only the saved spatial mean
for the optional mean-subtracted treatment. Both treatments are the default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pywt
from scipy.spatial.distance import pdist, squareform

import bis_bic
import data_analysis.bispectrum.sparse_bispectrum as sparse_bispectrum
from data_analysis.psd.pod_center_psd import infer_sampling_frequency


METRICS = {
    "bicoherence": "Bicoherence MAE",
    "bispectrum_log": "Log-bispectrum MAE [decades]",
    "bispectrum_shape": "Centered log-bispectrum MAE [decades]",
}
TREATMENTS = {"without_mean": "Spatial mean removed", "with_mean": "Spatial mean retained"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("powers", nargs="+", help="One or more labels, e.g. 0p20 0p35, or 'all'.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "reduced_data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--treatment", choices=["both", *TREATMENTS], default="both")
    parser.add_argument("--num-scales", type=int, default=512)
    parser.add_argument("--axis-samples", type=int, default=0,
                        help="Evaluate up to this many log-spaced axis frequencies on the original grid (0: full grid). Dense caches are reused by slicing.")
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(METRICS),
                        help="Distances and corresponding median/IQR maps to report; both underlying spectra remain cached.")
    parser.add_argument("--domains", nargs="+", choices=["full", "limited_sum"], default=["full", "limited_sum"],
                        help="Frequency domains to compare (use 'full' for full-domain plots only).")
    parser.add_argument("--fb-low", type=float, default=150)
    parser.add_argument("--fb-high", type=float, default=57000)
    parser.add_argument("--wavelet", default="cgau1")
    parser.add_argument("--score-max-sum-frequency", type=float, default=10000,
                        help="Additional scoring domain f1+f2 <= this value; full-domain scores are also produced.")
    parser.add_argument("--reps", type=int, nargs="+", help="Optional repetition numbers; default all cached repetitions.")
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def fingerprint(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def discover(root, powers, repetitions):
    if powers == ["all"]:
        powers = sorted({p.name.split("__")[0] for p in (root / "center_psd_comparisons/raw_psd_cache").glob('*__raw_center_psd.npz')})
    cases = []
    for power in sorted(set(powers), key=lambda p: float(p.replace("p", "."))):
        found = []
        for path in (root / "center_psd_comparisons/raw_psd_cache").glob(f"{power}__*__raw_center_psd.npz"):
            _, experiment, _ = path.name.split("__")
            match = re.search(r"_rep(\d+)$", experiment)
            if match and (repetitions is None or int(match[1]) in repetitions):
                found.append({"power": power, "experiment": experiment, "rep": int(match[1]), "raw_cache": path,
                              "pod_file": root / power / experiment / "pod_2d_r1000.h5"})
        found.sort(key=lambda item: item["rep"])
        if len(found) < 2:
            raise ValueError(f"Need at least two cached repetitions for {power}; found {len(found)}.")
        if repetitions and set(repetitions) != {item["rep"] for item in found}:
            raise ValueError(f"Some requested repetitions are missing for {power}.")
        cases.extend(found)
    return cases


def read_experiment(case, treatments):
    with np.load(case["raw_cache"], allow_pickle=False) as archive:
        t = archive["time"].astype(np.float64)
        z = archive["center_signal"].astype(np.float64)
        point = [int(archive["center_y_index"]), int(archive["center_x_index"])]
        coords = [float(archive["center_y_coordinate"]), float(archive["center_x_coordinate"])]
        units = str(archive["signal_units"])
        source = str(archive["source_file"])
    if z.shape != t.shape or not np.isfinite(z).all():
        raise ValueError(f"Invalid signal in {case['raw_cache']}.")
    signals = {"with_mean": z}
    if "without_mean" in treatments:
        with h5py.File(case["pod_file"], "r") as handle:
            if not np.array_equal(t, handle["grid/time"][:]):
                raise ValueError("Cached signal and saved spatial mean have different timestamps.")
            if source != str(handle.attrs["source_file"]):
                raise ValueError("Cached signal and saved spatial mean identify different sources.")
            if not np.allclose(coords, [handle["grid/y"][point[0]], handle["grid/x"][point[1]]]):
                raise ValueError("Center coordinates differ between cache and POD file.")
            mean = handle["preprocessing/frame_spatial_mean"]
            if str(mean.attrs["units"]) != units:
                raise ValueError("Signal and spatial mean have different units.")
            signals["without_mean"] = z - mean[:]
    fs, jitter = infer_sampling_frequency(t)
    return t, signals, {"sampling_frequency_hz": fs, "timestamp_jitter": jitter, "units": units,
                        "point": point, "coordinates": coords, "n_samples": len(t), "source_file": source}


def compute_case(case, treatments, args, cache_dir, started):
    config = {"num_scales": args.num_scales, "fb_low": args.fb_low, "fb_high": args.fb_high,
              "wavelet": args.wavelet, "pywavelets_version": pywt.__version__,
              "estimator_sha256": hashlib.sha256(Path(bis_bic.__file__).read_bytes()).hexdigest(),
              "demean": True, "trim_edges": True}
    source = {"raw_cache": fingerprint(case["raw_cache"])}
    if "without_mean" in treatments:
        source["spatial_mean_source"] = fingerprint(case["pod_file"])
    paths = {}
    loaded = None
    for treatment in treatments:
        dense_key = {"config": config, "source": source, "treatment": treatment}
        dense_digest = hashlib.sha256(json.dumps(dense_key, sort_keys=True).encode()).hexdigest()[:16]
        dense_path = cache_dir / f"{case['power']}__{case['experiment']}__{treatment}__{dense_digest}.npz"
        this_config = dict(config)
        if args.axis_samples:
            this_config.update({"axis_samples_requested": args.axis_samples,
                               "sparse_estimator_sha256": hashlib.sha256(Path(sparse_bispectrum.__file__).read_bytes()).hexdigest()})
        key = {"config": this_config, "source": source, "treatment": treatment}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        path = cache_dir / f"{case['power']}__{case['experiment']}__{treatment}__{digest}.npz"
        if path.exists() and not args.overwrite_cache:
            with np.load(path, allow_pickle=False) as d:
                if json.loads(str(d["cache_key_json"])) != key:
                    raise ValueError(f"Cache key mismatch: {path}")
            print(f"[{time.perf_counter()-started:.1f}s] Reuse {case['power']} rep {case['rep']} {treatment}", flush=True)
        else:
            if args.axis_samples and dense_path.exists() and not args.overwrite_cache:
                with np.load(dense_path, allow_pickle=False) as d:
                    if json.loads(str(d["cache_key_json"])) != dense_key:
                        raise ValueError(f"Dense cache key mismatch: {dense_path}")
                    metadata = json.loads(str(d["metadata_json"]))
                    grid = sparse_bispectrum.sparse_grid(
                        metadata["sampling_frequency_hz"], args.fb_low, args.fb_high,
                        args.num_scales, args.wavelet, args.axis_samples,
                    )
                    idx = np.ix_(grid.output_positions, grid.output_positions)
                    data = {name: d[name][idx] for name in ["complex_bispectrum", "normalization_magnitude", "bicoherence"]}
                    metadata.update({"derived_from_dense_cache": str(dense_path),
                                     "axis_samples_requested": args.axis_samples,
                                     "axis_samples_actual": len(grid.frequency_hz)})
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **data, frequency_hz=grid.frequency_hz,
                                    transform_frequency_hz=grid.transform_frequency_hz,
                                    metadata_json=np.asarray(json.dumps(metadata)), cache_key_json=np.asarray(json.dumps(key)))
                temporary.replace(path)
                print(f"[{time.perf_counter()-started:.1f}s] Slice dense cache {case['power']} rep {case['rep']} {treatment}", flush=True)
                paths[treatment] = path
                continue
            if loaded is None:
                loaded = read_experiment(case, treatments)
            t, signals, metadata = loaded
            print(f"[{time.perf_counter()-started:.1f}s] Compute {case['power']} rep {case['rep']} {treatment}", flush=True)
            if args.axis_samples:
                result, timing = sparse_bispectrum.sparse_wavelet_metrics(
                    signals[treatment], t, metadata["sampling_frequency_hz"], args.fb_low, args.fb_high,
                    Ns=args.num_scales, wavelet=args.wavelet, axis_samples=args.axis_samples,
                )
                metadata.update(timing)
            else:
                result = bis_bic.wavelet_bispectral_metrics(
                    signals[treatment], t, metadata["sampling_frequency_hz"], args.fb_low, args.fb_high,
                    Ns=args.num_scales, wavelet=args.wavelet,
                )
            metadata.update({"power": case["power"], "rep": case["rep"], "experiment": case["experiment"],
                             "sample_start": result.sample_start, "sample_stop": result.sample_stop})
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary, frequency_hz=result.frequency_hz, transform_frequency_hz=result.transform_frequency_hz,
                complex_bispectrum=result.complex_bispectrum, bicoherence=result.bicoherence,
                normalization_magnitude=result.normalization_magnitude,
                metadata_json=np.asarray(json.dumps(metadata)), cache_key_json=np.asarray(json.dumps(key)),
            )
            temporary.replace(path)
        paths[treatment] = path
    return paths


def log_area_weights(frequency, maximum_sum=None):
    """Log-frequency cell areas, folded onto the unique upper triangle.

    Off-diagonal weights include the mirrored cell. Endpoint cells span half
    their neighboring interval, keeping integration within the sampled bounds.
    """
    f = np.asarray(frequency, dtype=float)
    if f.ndim != 1 or len(f) < 2 or np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("Frequencies must be positive and strictly ascending.")
    x = np.log(f)
    edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
    widths = np.diff(edges)
    weights = widths[:, None] * widths[None, :]
    upper = np.triu_indices(len(f))
    values = weights[upper] * np.where(upper[0] == upper[1], 1, 2)
    if maximum_sum is not None:
        values[f[upper[0]] + f[upper[1]] > maximum_sum] = 0
    if not np.any(values > 0):
        raise ValueError("Scoring domain has no frequency pairs.")
    return upper, values / values.sum()


def distance_matrices(log_bispectrum, bicoherence, weights, metrics=None):
    """Use the same finite domain for every repetition and every distance."""
    valid = (weights > 0) & np.isfinite(log_bispectrum).all(axis=0) & np.isfinite(bicoherence).all(axis=0)
    if not np.any(valid):
        raise ValueError("No common finite frequency pairs remain; spectra may be undefined.")
    coverage = float(weights[valid].sum() / weights.sum())
    w = weights[valid] / weights[valid].sum()
    logs, b = log_bispectrum[:, valid], bicoherence[:, valid]
    centered = logs - (logs @ w)[:, None]
    arrays = {"bicoherence": b, "bispectrum_log": logs, "bispectrum_shape": centered}
    matrices = {name: squareform(pdist(array * w[None, :], metric="cityblock")) for name, array in arrays.items()
                if metrics is None or name in metrics}
    return matrices, {"retained_log_area_fraction": coverage, "valid_unique_frequency_pairs": int(valid.sum()),
                      "bispectrum_mean_log_level_per_repetition": (logs @ w).tolist()}


def group_summary(distance, labels):
    powers = list(dict.fromkeys(labels))
    labels = np.asarray(labels)
    group = np.empty((len(powers), len(powers)))
    within = {}
    for i, power in enumerate(powers):
        idx = np.flatnonzero(labels == power)
        block = distance[np.ix_(idx, idx)]
        values = block[np.triu_indices(len(idx), 1)]
        within[power] = {"repetitions": len(idx), "pairs": len(values), "mean": float(values.mean()),
                         "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}
        group[i, i] = values.mean()
    between = []
    for i, j in itertools.combinations(range(len(powers)), 2):
        p, q = powers[i], powers[j]
        values = distance[np.ix_(labels == p, labels == q)].ravel()
        mean = float(values.mean())
        baseline = (within[p]["mean"] + within[q]["mean"]) / 2
        group[i, j] = group[j, i] = mean
        between.append({"power_a": p, "power_b": q, "pairs": len(values), "mean": mean,
                        "median": float(np.median(values)), "within_baseline": baseline,
                        "between_to_within_ratio": mean / baseline if baseline > 0 else None,
                        "between_minus_within": mean - baseline})
    return {"within": within, "between": between}, group


def separation_statistic(distance, labels):
    """Equal weight to each power and each pair of powers, excluding self pairs."""
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == p) for p in np.unique(labels)]
    within = [distance[np.ix_(g, g)][np.triu_indices(len(g), 1)].mean() for g in groups]
    across = [distance[np.ix_(a, b)].mean() for a, b in itertools.combinations(groups, 2)]
    return float(np.mean(across) - np.mean(within))


def permutation_separation(distance, labels, count=1999, seed=42):
    """Shuffle whole-repetition power labels; never shuffle pairwise distances.

    This tests equality of group distributions under independent/exchangeable
    repetitions. It is not valid unmodified for matched or blocked acquisitions.
    """
    if len(set(labels)) < 2 or count == 0:
        return None
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    observed = separation_statistic(distance, labels)
    null = np.array([separation_statistic(distance, rng.permutation(labels)) for _ in range(count)])
    p = (1 + np.count_nonzero(null >= observed - 1e-14)) / (count + 1)
    return {"between_minus_within": observed, "one_sided_pvalue_unadjusted": float(p),
            "permutations": count, "seed": seed,
            "assumption": "Whole repetitions independent/exchangeable; no acquisition-block adjustment."}


def plot_distances(distance, cases, group, title):
    labels = [c["power"] for c in cases]
    powers = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1, 1]})
    vmax = max(float(distance.max()), 1e-12)
    mesh = axes[0].imshow(distance, cmap="viridis", vmin=0, vmax=vmax)
    counts = [labels.count(p) for p in powers]
    ends = np.cumsum([0] + counts)
    centers = [(ends[i] + ends[i+1] - 1) / 2 for i in range(len(powers))]
    axes[0].set_xticks(centers, powers)
    axes[0].set_yticks(centers, powers)
    for edge in ends[1:-1] - .5:
        axes[0].axhline(edge, color="white", lw=.8)
        axes[0].axvline(edge, color="white", lw=.8)
    axes[0].set_title("All repetition pairs; lower = closer")
    axes[0].set_xlabel("Repetitions ordered by power, then number")
    fig.colorbar(mesh, ax=axes[0], shrink=.8)
    mesh = axes[1].imshow(group, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_xticks(range(len(powers)), powers)
    axes[1].set_yticks(range(len(powers)), powers)
    for i in range(len(powers)):
        for j in range(len(powers)):
            axes[1].text(j, i, f"{group[i,j]:.3g}", ha="center", va="center",
                         color="white" if group[i,j] < .5*vmax else "black")
    axes[1].set_title("Mean distances; diagonal = within power")
    i, j = np.triu_indices(len(cases), 1)
    same = np.asarray(labels)[i] == np.asarray(labels)[j]
    bins = np.linspace(0, vmax * 1.001, 25)
    axes[2].hist(distance[i[same], j[same]], bins=bins, density=True, alpha=.6, label="Within power")
    if np.any(~same):
        axes[2].hist(distance[i[~same], j[~same]], bins=bins, density=True, alpha=.6, label="Between powers")
    axes[2].set_title("Pair distributions (pairs are dependent)")
    axes[2].set_xlabel("Distance")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    fig.suptitle(title, fontsize=13)
    return fig


def plot_power_maps(frequency, logs, bicoherence, cases, treatment, metrics=None):
    metrics = list(METRICS) if metrics is None else metrics
    powers = list(dict.fromkeys(c["power"] for c in cases))
    labels = np.asarray([c["power"] for c in cases])
    figures = []
    for power in powers:
        selected = labels == power
        panels = []
        if any(metric in metrics for metric in ("bispectrum_log", "bispectrum_shape")):
            med_log = np.nanmedian(logs[selected], axis=0)
            spread_log = np.nanquantile(logs[selected], .75, axis=0) - np.nanquantile(logs[selected], .25, axis=0)
            panels.extend([(med_log, "Median log10|B|", "magma", None),
                           (spread_log, "Log10|B| interquartile range", "inferno", None)])
        if "bicoherence" in metrics:
            med_b = np.nanmedian(bicoherence[selected], axis=0)
            spread_b = np.nanquantile(bicoherence[selected], .75, axis=0) - np.nanquantile(bicoherence[selected], .25, axis=0)
            panels.extend([(med_b, "Median bicoherence", "magma", (0,1)),
                           (spread_b, "Bicoherence interquartile range", "inferno", (0,1))])
        fig, axes = plt.subplots(len(panels)//2, 2, figsize=(10, 4*len(panels)//2), layout="constrained")
        for ax, (array, title, cmap, limits) in zip(axes.ravel(), panels):
            kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
            mesh = ax.pcolormesh(frequency/1000, frequency/1000, array, shading="auto", rasterized=True, cmap=cmap, **kwargs)
            ax.set(xscale="log", yscale="log", xlabel="f2 [kHz]", ylabel="f1 [kHz]", title=title)
            fig.colorbar(mesh, ax=ax)
        fig.suptitle(f"{power}: {selected.sum()} experimental repetitions — {TREATMENTS[treatment]}\n"
                     "Pointwise median and spread, not the spectrum of an averaged signal")
        figures.append((power, fig))
    return figures


def analyze(cases, paths, treatments, args, output):
    domains = {"full": None, "limited_sum": args.score_max_sum_frequency}
    summary = {"powers": list(dict.fromkeys(c["power"] for c in cases)), "num_scales": args.num_scales,
               "axis_samples_requested": args.axis_samples, "metrics": args.metrics,
               "wavelet": args.wavelet, "weighting": "Log-frequency cell area; symmetric cells folded into upper triangle",
               "domains": {name: domains[name] for name in args.domains}, "comparisons": {}}
    manifest = [{"power": c["power"], "rep": c["rep"], "experiment": c["experiment"],
                 "spectra": {t: str(p[t]) for t in treatments}} for c, p in zip(cases, paths)]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    labels = [c["power"] for c in cases]
    names = [f"{c['power']}_rep{c['rep']}" for c in cases]
    archives = {}
    with PdfPages(output / "repetition_comparison.pdf") as pdf:
        reference_metadata = None
        for treatment in treatments:
            log_maps, b_maps = [], []
            for path in paths:
                with np.load(path[treatment], allow_pickle=False) as d:
                    meta = json.loads(str(d["metadata_json"]))
                    f = d["frequency_hz"]
                    if reference_metadata is None:
                        reference_metadata, frequency = meta, f
                    if not np.allclose(frequency, f, rtol=1e-12, atol=1e-9):
                        raise ValueError("Frequency grids differ; recompute all repetitions on the same grid.")
                    for key in ["n_samples", "sample_start", "sample_stop", "point", "units"]:
                        if meta[key] != reference_metadata[key]:
                            raise ValueError(f"Repetitions differ in {key}; harmonize before comparison.")
                    if not np.isclose(meta["sampling_frequency_hz"], reference_metadata["sampling_frequency_hz"], rtol=1e-8):
                        raise ValueError("Sampling frequencies differ.")
                    magnitude = np.abs(d["complex_bispectrum"])
                    log = np.full_like(magnitude, np.nan)
                    np.log10(magnitude, out=log, where=magnitude > 0)
                    log_maps.append(log)
                    b_maps.append(d["bicoherence"])
            log_maps, b_maps = np.asarray(log_maps), np.asarray(b_maps)
            summary["axis_samples_actual"] = len(frequency)
            summary["frequency_bounds_hz"] = [float(frequency[0]), float(frequency[-1])]
            summary["sample_start"] = reference_metadata["sample_start"]
            summary["sample_stop"] = reference_metadata["sample_stop"]
            for power, fig in plot_power_maps(frequency, log_maps, b_maps, cases, treatment, args.metrics):
                fig.savefig(output / f"{treatment}_{power}_median_spread.png", dpi=args.dpi)
                pdf.savefig(fig)
                plt.close(fig)
            for domain, maximum in summary["domains"].items():
                upper, weights = log_area_weights(frequency, maximum)
                distances, info = distance_matrices(log_maps[:, upper[0], upper[1]], b_maps[:, upper[0], upper[1]], weights, args.metrics)
                for metric, distance in distances.items():
                    key = f"{treatment}__{domain}__{metric}"
                    group, block = group_summary(distance, labels)
                    group["coverage"] = info
                    group["permutation_test"] = permutation_separation(distance, labels, args.permutations, args.seed)
                    summary["comparisons"][key] = group
                    archives[key] = distance
                    with (output / f"{key}_distances.csv").open("w", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["repetition", *names])
                        writer.writerows([[name, *row] for name, row in zip(names, distance)])
                    domain_title = "Full domain" if maximum is None else f"f1+f2 ≤ {maximum/1000:g} kHz (sensitivity domain)"
                    fig = plot_distances(distance, cases, block, f"{METRICS[metric]} — {TREATMENTS[treatment]}\n{domain_title}; log-frequency area weighting")
                    fig.savefig(output / f"{key}.png", dpi=args.dpi)
                    pdf.savefig(fig)
                    plt.close(fig)
                    ratios = [round(x["between_to_within_ratio"], 3) if x["between_to_within_ratio"] is not None else None for x in group["between"]]
                    print(f"{key}: between/within ratios {ratios}", flush=True)
    summary["permutation_note"] = "P-values exploratory/unadjusted across metrics, domains and treatments. Repetitions, not pairs, are the exchangeable units. Acquisition blocks are not modeled."
    np.savez_compressed(output / "distance_matrices.npz", repetition_names=np.asarray(names), **archives)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_readme(output, summary, args)


def write_readme(output, summary, args):
    metric_notes = {
        "bicoherence": "- Bicoherence: sum(w * abs(b_A - b_B)), in [0,1].",
        "bispectrum_log": "- Log bispectrum: sum(w * abs(L_A - L_B)), where L=log10(abs(B)); units are decades.",
        "bispectrum_shape": "- Bispectrum shape: subtract each map's weighted mean of L=log10(abs(B)), then use weighted MAE in decades. This removes a uniform multiplicative change in bispectrum magnitude, but retains changes in slope and contrast.",
    }
    domain_notes = ["Full domain" if name == "full" else f"f1+f2 <= {args.score_max_sum_frequency:g} Hz"
                    for name in summary["domains"]]
    lines = ["# Experimental repetition consistency", "", "Open **repetition_comparison.pdf** for all maps and distances.", "",
             "## What is compared", "", "Each cached experimental center signal is analyzed separately. No POD coefficients or ROM output enter these results.",
             "The spatial-mean-subtracted treatment uses the measured spatial mean saved in the POD HDF5. Both treatments are temporally demeaned.",
             f"Wavelet: {args.wavelet}; scales requested: {args.num_scales}; transform maximum: {args.fb_high:g} Hz.",
             f"Axis samples: {summary['axis_samples_actual']}; log-spaced target count: {args.axis_samples or 'full original grid'}. Sparse sampling retains the original frequency endpoints and time interval, and evaluates every required sum frequency exactly. Dense cached spectra are reused by slicing.",
             "All repetitions must have the same sampling rate, duration, frequency grid, point index, units, and retained time interval.",
             "", "## Distances", "", "Lower always means more similar. Frequency-cell weights approximate area in (log f1, log f2), normalized to sum to one. Symmetric cells are counted once with their combined area.",
             *(metric_notes[metric] for metric in args.metrics),
             "Median/IQR maps show the selected spectral quantities; bispectrum maps show uncentered log magnitudes even when the distance uses centering.",
             "No per-map min-max normalization or Pearson correlation is used. High correlation alone could hide differences in coupling strength.",
             "The same finite frequency-pair mask is used for all repetitions and all distances in each treatment/domain. Undefined entries are excluded globally; retained log-area coverage is reported. There is no fitted noise floor or significance mask.",
             "", "## Within and between powers", "",
             "Within-power distances include all distinct repetition pairs and exclude self pairs. The group heatmap diagonal is their mean, not zero.",
             "Between-power distances include every repetition pairing across the two powers.",
             "Separation ratio = mean between-power distance / average of the two mean within-power distances. A value above 1 means greater average between-power separation; it does not guarantee disjoint distributions.",
             "Permutation statistic = mean between-power distance (equal weight to each power pair) minus mean within-power distance (equal weight to each power).",
             "Shuffle whole-repetition power labels, preserving group sizes. Pairwise distances are dependent and are never treated as independent replicates. The one-sided Monte Carlo p-value uses (1 + number of null statistics >= observed)/(1 + permutations).",
             "The permutation test assumes independent/exchangeable repetitions. Matched acquisitions, sessions, drift, or blocks require a restricted permutation design. P-values here are exploratory and unadjusted across the reported metrics/domains/treatments.",
             "", "## Frequency-domain sensitivity", "",
             "Reported domains: " + "; ".join(domain_notes) + ".",
             *(["The restricted sum domain is an exploratory sensitivity check, not a validated wavelet/noise cutoff."] if "limited_sum" in summary["domains"] else []),
             "cgau1 coefficients at the highest frequencies are coarsely sampled. More scales refine the frequency grid but do not fix short-wavelet discretization. Separation of powers also does not itself establish significant physical three-wave coupling.",
             "", "## Results", ""]
    for key, group in summary["comparisons"].items():
        for row in group["between"]:
            ratio = row["between_to_within_ratio"]
            lines.append(f"- {key}, {row['power_a']} vs {row['power_b']}: between/within = {ratio:.3f}" if ratio is not None else f"- {key}: ratio undefined (zero within variation).")
    lines += ["", "## Reuse", "", "Run the same command again to regenerate comparisons while reusing cached spectra. Individual caches survive interrupted runs and are keyed by source metadata, estimator code, wavelet version, and transform settings.",
              "Use one power to assess repeatability or several to compare powers. --reps selects repetition numbers; --treatment without_mean selects only the primary fluctuation comparison.",
              "The manifest identifies every cache and its repetition. Distance matrices are saved as NPZ and labeled CSV files. Changing the scoring band or permutation count does not recompute wavelet/bispectral maps.",
              "", "References: https://doi.org/10.1016/j.chaos.2023.113615 ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html"]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.num_scales < 4 or args.permutations < 0 or args.dpi < 1 or args.score_max_sum_frequency <= 0 or args.axis_samples < 0 or args.axis_samples == 1:
        raise ValueError("Invalid grid, permutation count, DPI, or scoring frequency.")
    root = args.data_root.expanduser().resolve()
    cases = discover(root, args.powers, args.reps)
    powers = list(dict.fromkeys(c["power"] for c in cases))
    treatments = list(TREATMENTS) if args.treatment == "both" else [args.treatment]
    base = root / "bispectral_repetition_comparisons"
    name = ("all_powers" if args.powers == ["all"] else "_vs_".join(powers)) + f"_ns{args.num_scales}"
    if args.treatment != "both":
        name += f"_{args.treatment}"
    if args.axis_samples:
        name += f"_sparse{args.axis_samples}"
    output = (args.output_dir or base / name).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} repetitions, {len(treatments)} treatments; spectra cached in {base / 'spectra_cache'}", flush=True)
    started = time.perf_counter()
    paths = [compute_case(case, treatments, args, base / "spectra_cache", started) for case in cases]
    analyze(cases, paths, treatments, args, output)
    print(f"Completed in {time.perf_counter()-started:.1f}s. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
