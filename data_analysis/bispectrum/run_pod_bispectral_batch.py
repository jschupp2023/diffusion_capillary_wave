"""Paired raw/POD bispectral errors by forcing and rank, spatial mean removed.

Run with the capillarywave environment: python run_pod_bispectral_batch.py
Raw spectra are read from the existing sparse repetition manifest, never recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pywt

import bis_bic
import data_analysis.bispectrum.compare_center_bispectrum as compare_center_bispectrum
import data_analysis.psd.pod_center_psd as pod_center_psd
import data_analysis.bispectrum.sparse_bispectrum as sparse_bispectrum
from data_analysis.bispectrum.compare_bispectral_repetitions import METRICS, distance_matrices, fingerprint, log_area_weights, read_experiment
from data_analysis.bispectrum.compare_center_bispectrum import load_signals, log_magnitude


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "reduced_data")
    parser.add_argument("--raw-comparison-dir", type=Path,
                        help="Folder with the existing sparse raw manifest (default: all_powers_ns512_sparse64).")
    parser.add_argument("--powers", nargs="+", default=["all"], help="Default all; e.g. --powers 0p20 0p35")
    parser.add_argument("--reps", type=int, nargs="+", help="Optional repetition subset.")
    parser.add_argument("--ranks", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--score-max-sum-frequency", type=float, default=10000)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--dry-run", action="store_true", help="Validate all selected inputs and report cached/pending counts only.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Recompute POD caches; raw caches stay read-only.")
    args = parser.parse_args()
    if min(args.ranks) < 1 or args.dpi < 1 or not np.isfinite(args.score_max_sum_frequency) or args.score_max_sum_frequency <= 0:
        parser.error("Ranks and DPI must be positive; maximum sum frequency must be finite and positive.")
    if args.reps and min(args.reps) < 1:
        parser.error("Repetition numbers must be positive.")
    args.ranks = sorted(set(args.ranks))
    return args


def select_entries(manifest, powers, reps):
    available = {entry["power"] for entry in manifest}
    requested = available if powers == ["all"] else set(powers)
    if not requested <= available:
        raise ValueError(f"Unknown forcing labels: {sorted(requested - available)}")
    entries = [entry for entry in manifest if entry["power"] in requested
               and (reps is None or entry["rep"] in reps)]
    identities = [(entry["power"], entry["rep"]) for entry in entries]
    if not entries or len(set(identities)) != len(identities):
        raise ValueError("Empty selection or duplicate forcing/repetition entries in raw manifest.")
    for power in requested:
        found = {entry["rep"] for entry in entries if entry["power"] == power}
        if not found or (reps is not None and found != set(reps)):
            raise ValueError(f"Missing requested repetitions for {power}.")
    return sorted(entries, key=lambda entry: (float(entry["power"].replace("p", ".")), entry["rep"]))


def prepare_case(entry, ranks):
    path = Path(entry["spectra"]["without_mean"])
    with np.load(path, allow_pickle=False) as archive:
        key = json.loads(str(archive["cache_key_json"]))
        meta = json.loads(str(archive["metadata_json"]))
        raw = {name: archive[name] for name in ("frequency_hz", "complex_bispectrum", "bicoherence")}
    config = key["config"]
    if key["treatment"] != "without_mean" or not config["demean"] or not config["trim_edges"]:
        raise ValueError(f"Expected spatial-mean-removed, temporally demeaned, edge-trimmed raw spectra: {path}")
    if config.get("axis_samples_requested", 0) < 2:
        raise ValueError(f"Expected a sparse raw cache: {path}")
    expected = {"estimator_sha256": sha256(bis_bic.__file__),
                "sparse_estimator_sha256": sha256(sparse_bispectrum.__file__),
                "pywavelets_version": pywt.__version__}
    for name, value in expected.items():
        if config[name] != value:
            raise ValueError(f"Raw cache {name} differs from the current estimator/environment: {path}")
    for source in key["source"].values():
        if fingerprint(Path(source["path"])) != source:
            raise ValueError(f"Raw cache source has changed: {source['path']}")
    case = {**entry, "raw_spectrum": path,
            "raw_cache": Path(key["source"]["raw_cache"]["path"]),
            "pod_file": Path(key["source"]["spatial_mean_source"]["path"]),
            "config": config, "metadata": meta, "raw": raw}
    _, _, measured = read_experiment(case, ["without_mean"])
    for name in ("power", "rep", "experiment"):
        if meta[name] != entry[name]:
            raise ValueError(f"Raw manifest/cache disagree in {name}: {path}")
    for name in ("source_file", "n_samples", "point", "units", "coordinates", "sampling_frequency_hz"):
        if measured[name] != meta[name]:
            raise ValueError(f"Raw spectrum/signal metadata disagree in {name}: {path}")
    grid = sparse_bispectrum.sparse_grid(meta["sampling_frequency_hz"], config["fb_low"], config["fb_high"],
                                       config["num_scales"], config["wavelet"], config["axis_samples_requested"])
    np.testing.assert_allclose(raw["frequency_hz"], grid.frequency_hz, rtol=1e-12, atol=1e-9)
    wavelet = pywt.ContinuousWavelet(config["wavelet"])
    margin = int(np.ceil(max(abs(wavelet.lower_bound), abs(wavelet.upper_bound)) * grid.scales.max()))
    if (meta["sample_start"], meta["sample_stop"]) != (margin, meta["n_samples"] - margin):
        raise ValueError(f"Raw averaging interval differs from current estimator: {path}")
    for quantity in ("complex_bispectrum", "bicoherence"):
        if raw[quantity].shape != (len(grid.frequency_hz),) * 2:
            raise ValueError(f"Invalid raw map shape: {path}")
    with h5py.File(case["pod_file"], "r") as handle:
        if max(ranks) > len(handle["pod/modes"]):
            raise ValueError(f"Requested rank exceeds stored modes: {case['pod_file']}")
        mean = handle["preprocessing/frame_spatial_mean"]
        if not bool(mean.attrs.get("subtracted_before_pod", handle.attrs.get("instantaneous_spatial_mean_removed", False))):
            raise ValueError(f"POD did not remove the spatial mean: {case['pod_file']}")
    return case


def pod_cache_key(case, rank):
    return {"schema": 1, "rank": rank, "treatment": "without_mean", "config": case["config"],
            "raw_spectrum": fingerprint(case["raw_spectrum"]), "pod_file": fingerprint(case["pod_file"]),
            "signal_loader_sha256": sha256(compare_center_bispectrum.__file__),
            "reconstruction_sha256": sha256(pod_center_psd.__file__)}


def cache_path(cache_dir, case, rank, key):
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return cache_dir / f"{case['power']}__{case['experiment']}__r{rank}__{digest}.npz"


def compute_pod(case, rank, path, key, overwrite):
    if not path.exists() or overwrite:
        config, reference = case["config"], case["metadata"]
        t, _, signals, meta = load_signals(case["raw_cache"], case["pod_file"], rank)
        for name in ("source_file", "n_samples", "sampling_frequency_hz"):
            if meta[name] != reference[name]:
                raise ValueError(f"POD reconstruction and raw spectrum differ in {name}.")
        result, timing = sparse_bispectrum.sparse_wavelet_metrics(
            signals["without_mean_pod"], t, meta["sampling_frequency_hz"], config["fb_low"], config["fb_high"],
            Ns=config["num_scales"], wavelet=config["wavelet"], axis_samples=config["axis_samples_requested"])
        np.testing.assert_allclose(result.frequency_hz, case["raw"]["frequency_hz"], rtol=1e-12, atol=1e-9)
        if (result.sample_start, result.sample_stop) != (reference["sample_start"], reference["sample_stop"]):
            raise ValueError("POD and raw averaging intervals differ.")
        meta.update(timing, sample_start=result.sample_start, sample_stop=result.sample_stop)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, frequency_hz=result.frequency_hz,
                            transform_frequency_hz=result.transform_frequency_hz,
                            complex_bispectrum=result.complex_bispectrum, bicoherence=result.bicoherence,
                            normalization_magnitude=result.normalization_magnitude,
                            removed_temporal_mean=result.removed_temporal_mean,
                            metadata_json=np.asarray(json.dumps(meta)), cache_key_json=np.asarray(json.dumps(key)))
        temporary.replace(path)
        status = f"computed in {timing['total_seconds']:.1f}s"
    else:
        status = "cached"
    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive["cache_key_json"])) != key:
            raise ValueError(f"POD cache key mismatch: {path}")
        np.testing.assert_allclose(archive["frequency_hz"], case["raw"]["frequency_hz"], rtol=1e-12, atol=1e-9)
        maps = {name: archive[name] for name in ("complex_bispectrum", "bicoherence")}
    return maps, status


def paired_errors(cases, ranks, maximum_sum):
    """One global finite mask, then matched raw/POD errors; no cross-repetition pairs."""
    logs, b = [], []
    for case in cases:
        maps = [case["raw"], *[case["pod"][rank] for rank in ranks]]
        logs.append([log_magnitude(np.abs(m["complex_bispectrum"])) for m in maps])
        b.append([m["bicoherence"] for m in maps])
    logs, b = np.asarray(logs), np.asarray(b)
    rows, coverage = [], {}
    for domain, maximum in (("full", None), ("limited_sum", maximum_sum)):
        upper, weights = log_area_weights(cases[0]["raw"]["frequency_hz"], maximum)
        l = logs[:, :, upper[0], upper[1]]
        coherence = b[:, :, upper[0], upper[1]]
        valid = np.isfinite(l).all(axis=(0, 1)) & np.isfinite(coherence).all(axis=(0, 1)) & (weights > 0)
        if not valid.any():
            raise ValueError(f"No common finite frequency pairs in {domain}.")
        coverage[domain] = {"retained_log_area_fraction": float(weights[valid].sum() / weights.sum()),
                            "valid_unique_frequency_pairs": int(valid.sum())}
        common_weights = np.where(valid, weights, 0)
        for i, case in enumerate(cases):
            distances, _ = distance_matrices(l[i], coherence[i], common_weights)
            for j, rank in enumerate(ranks, 1):
                for metric, matrix in distances.items():
                    rows.append({"power": case["power"], "rep": case["rep"], "experiment": case["experiment"],
                                 "rank": rank, "domain": domain, "metric": metric, "error": float(matrix[0, j]),
                                 **coverage[domain]})
    return rows, coverage


def aggregate_errors(rows):
    groups = {}
    for row in rows:
        key = (row["power"], row["rank"], row["domain"], row["metric"])
        groups.setdefault(key, []).append(row["error"])
    return [{"power": p, "rank": r, "domain": d, "metric": m, "n_repetitions": len(values),
             "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
             "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
             "q75": float(np.quantile(values, .75))} for (p, r, d, m), values in groups.items()]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(output, groups, ranks, maximum_sum, dpi):
    with PdfPages(output / "pod_bispectral_errors.pdf") as pdf:
        for domain, title in (("full", "Full frequency domain"),
                               ("limited_sum", f"f1 + f2 ≤ {maximum_sum / 1000:g} kHz")):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
            for ax, metric in zip(axes, METRICS):
                for rank in ranks:
                    selected = sorted([g for g in groups if (g["domain"], g["metric"], g["rank"]) ==
                                      (domain, metric, rank)], key=lambda g: float(g["power"].replace("p", ".")))
                    x = [float(g["power"].replace("p", ".")) for g in selected]
                    line, = ax.plot(x, [g["mean"] for g in selected], "o-", label=f"Rank {rank}")
                    ax.fill_between(x, [g["q25"] for g in selected], [g["q75"] for g in selected],
                                    color=line.get_color(), alpha=.15)
                ax.set(xlabel="Forcing [Vpp]", ylabel=METRICS[metric], title=METRICS[metric])
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.25)
                ax.legend()
            fig.suptitle(f"POD versus matching raw repetition — spatial mean removed\n{title}; lines: mean error; bands: repetition IQR; lower = closer")
            fig.savefig(output / f"{domain}_mean_errors.png", dpi=dpi)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    args = parse_args()
    started = time.perf_counter()
    root = args.data_root.expanduser().resolve()
    raw_dir = (args.raw_comparison_dir or root / "bispectral_repetition_comparisons/all_powers_ns512_sparse64").expanduser().resolve()
    entries = select_entries(json.loads((raw_dir / "manifest.json").read_text()), args.powers, args.reps)
    print(f"Validating {len(entries)} raw spectra and POD sources...", flush=True)
    cases = [prepare_case(entry, args.ranks) for entry in entries]
    first = cases[0]
    for case in cases[1:]:
        if case["config"] != first["config"]:
            raise ValueError("Raw cases use different transform settings.")
        np.testing.assert_allclose(case["raw"]["frequency_hz"], first["raw"]["frequency_hz"], rtol=1e-12, atol=1e-9)
        for name in ("n_samples", "point", "units", "sample_start", "sample_stop", "sampling_frequency_hz"):
            if case["metadata"][name] != first["metadata"][name]:
                raise ValueError(f"Raw cases differ in {name}.")
    config = first["config"]
    base = root / "pod_bispectral_comparisons"
    cache_dir = base / "spectra_cache"
    label = "all_powers" if args.powers == ["all"] else "_vs_".join(dict.fromkeys(c["power"] for c in cases))
    if args.reps:
        label += "_reps" + "-".join(map(str, sorted(set(args.reps))))
    name = label + "_r" + "-".join(map(str, args.ranks)) + f"_ns{config['num_scales']}_sparse{config['axis_samples_requested']}_without_mean"
    output = (args.output_dir or base / name).expanduser().resolve()
    jobs = []
    for case in cases:
        case["pod"], case["pod_paths"] = {}, {}
        for rank in args.ranks:
            key = pod_cache_key(case, rank)
            path = cache_path(cache_dir, case, rank, key)
            jobs.append((case, rank, path, key))
    cached = sum(path.exists() and not args.overwrite_cache for _, _, path, _ in jobs)
    print(f"{len(cases)} recordings, ranks {args.ranks}, {len(first['raw']['frequency_hz'])} axis frequencies. "
          f"POD spectra: {len(jobs)-cached} pending, {cached} cached. Raw spectra are reused.\nOutput: {output}", flush=True)
    if args.dry_run:
        return
    for i, (case, rank, path, key) in enumerate(jobs, 1):
        maps, status = compute_pod(case, rank, path, key, args.overwrite_cache)
        case["pod"][rank], case["pod_paths"][rank] = maps, str(path)
        print(f"[{i}/{len(jobs)}; {time.perf_counter()-started:.1f}s] {case['power']} rep {case['rep']} rank {rank}: {status}", flush=True)
    rows, coverage = paired_errors(cases, args.ranks, args.score_max_sum_frequency)
    groups = aggregate_errors(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_errors.csv", rows)
    write_csv(output / "mean_errors_by_forcing_rank.csv", groups)
    manifest = [{"power": c["power"], "rep": c["rep"], "experiment": c["experiment"],
                 "raw_spectrum": str(c["raw_spectrum"]), "pod_spectra": c["pod_paths"]} for c in cases]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {"config": config, "treatment": "without_mean", "ranks": args.ranks,
               "recordings": len(cases), "raw_comparison_dir": str(raw_dir),
               "domains": {"full": None, "limited_sum": args.score_max_sum_frequency},
               "coverage": coverage, "groups": groups}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render(output, groups, args.ranks, args.score_max_sum_frequency, args.dpi)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    (output / "README.md").write_text(
        "# Paired POD/raw bispectral errors\n\nOpen **pod_bispectral_errors.pdf** for forcing curves at each rank. "
        "Lines are arithmetic means of per-repetition errors; shaded bands are the repetition interquartile range, not confidence intervals. "
        "The CSV files contain individual errors and mean, sample standard deviation, median and quartiles by forcing/rank. "
        "A single-repetition standard deviation is undefined (blank CSV/null JSON).\n\n"
        "Each POD center signal is compared only with its own measured repetition. We remove the instantaneous spatial mean "
        "and temporally demean before transforming. No spatial mean is restored. The optional static temporal-mean field offset "
        "is restored consistently with the existing point reconstruction, then removed by temporal demeaning.\n\n"
        "Raw spectra are reused read-only from the manifest. Transform settings, frequency grid, source fingerprints, units, "
        "sampling rate and retained time interval are checked. Sparse evaluation uses every time sample and exact sum frequencies. "
        "Complex bispectrum, normalization magnitude and bicoherence are cached for every rank/repetition.\n\n"
        "Distances use the existing repetition code's normalized log-frequency area weights and folded symmetry. "
        "Log-bispectrum MAE compares log10(abs(B)) in decades. Centered log-bispectrum MAE subtracts each map's weighted mean "
        "before differencing, removing a uniform magnitude factor. Bicoherence MAE compares b = abs(mean(Q))/mean(abs(Q)); "
        "it is not squared bicoherence. Lower means closer. A single finite mask across every selected raw/POD map "
        "is used within each scoring domain; coverage is recorded. Selecting different cases can change that mask if any entries are undefined. "
        "Errors are calculated before averaging, with equal weight per repetition.\n\n"
        "These are reconstruction errors for measured data, not unseen-data predictions, energy-transfer estimates or significance tests. "
        "POD rank improvements need not be monotonic for nonlinear spectral metrics.\n\n"
        "## Run / resume\n\n```bash\n" + command + "\n```\n\n"
        "Rerun the same command to resume or regenerate summaries from caches. --dry-run validates inputs and counts pending spectra. "
        "--powers and --reps select a subset; --ranks selects ranks. Use --output-dir to separate reports. "
        "Changing the scoring sum-frequency limit does not recompute spectra. --overwrite-cache recomputes POD spectra only.\n")
    print(f"Completed in {time.perf_counter()-started:.1f}s: {output / 'pod_bispectral_errors.pdf'}", flush=True)


if __name__ == "__main__":
    main()
