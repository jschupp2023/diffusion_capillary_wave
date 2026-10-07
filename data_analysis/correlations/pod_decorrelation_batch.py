"""Summarize POD decorrelation across reps, using one reference basis per power.

Example: python pod_decorrelation_batch.py all --rank 100
Or:      python pod_decorrelation_batch.py 0p20 0p35 --rank 100
Uses K consecutive |ACF|<0.01 lags (default 20), reporting the run's start.
K=1 recovers the first crossing from pod_decorrelation.py. Mode 0 is the
spatial mean. Maps each rank-r reconstruction into its power's lowest-rep
rank-r basis, exactly as in pod_lagged_covariance.py. Across powers, equal
mode indices need not describe equal spatial structures.
"""

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

# Small matrix products do not benefit from excessive BLAS threading.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from data_analysis.correlations.pod_decorrelation import decorrelation_lags
from data_analysis.correlations.pod_lagged_covariance import (
    POWER_PATTERN, discover_repetitions, load_coordinates, load_modes,
)


def summarize(values):
    """Mean, sample standard deviation and finite repetition count per mode."""
    count = np.isfinite(values).sum(axis=0)
    mean = np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1], np.nan), where=count > 0)
    variance = np.divide(np.nansum((values - mean) ** 2, axis=0), count - 1,
                         out=np.full_like(mean, np.nan), where=count > 1)
    return mean, np.sqrt(variance), count


def write_csv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def plot_results(output, powers, rank, K=20):
    for power in powers:
        with np.load(output / f"{power}.npz") as data:
            if int(data.get("K", 1)) != K:
                raise ValueError(f"Saved K for {power} differs from requested K={K}.")
    combined, combined_ax = plt.subplots(figsize=(12, 5))
    def format_axis(ax):
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(bottom=0)
        ax.set(xlabel="Reference mode index (0 = spatial mean)",
               ylabel=f"Run-start lag [ms], K={K}, |ACF| < 0.01\n(linear below 0.01 ms, logarithmic above)")
        ax.grid(alpha=0.2, which="both")

    with PdfPages(output / "decorrelation_summary.pdf") as pdf:
        for power in powers:
            with np.load(output / f"{power}.npz") as data:
                modes = data["modes"]
                if not np.array_equal(modes, np.arange(rank + 1)):
                    raise ValueError(f"Saved rank for {power} differs from requested rank.")
                mean, std = data["mean_seconds"] * 1000, data["std_seconds"] * 1000
                nrep, ref = len(data["repetitions"]), int(data["reference_repetition"])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(modes, mean, lw=1.5, label="Repetition mean")
            ax.fill_between(modes, mean - std, mean + std, alpha=0.25,
                            label="±1 sample standard deviation")
            ax.set_title(f"{power}: {nrep} reps, rank {rank}, reference rep{ref}")
            format_axis(ax)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"{power}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            line, = combined_ax.plot(modes, mean, label=power, lw=1.3)
            combined_ax.fill_between(modes, mean - std, mean + std,
                                     color=line.get_color(), alpha=0.10)
        combined_ax.set_title(f"Rank {rank}: repetition mean ± std; separate reference basis per power")
        format_axis(combined_ax)
        combined_ax.legend(ncol=5, fontsize=9)
        combined.tight_layout()
        combined.savefig(output / "all_powers.png", dpi=180)
        pdf.savefig(combined)
        plt.close(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("powers", nargs="+", help="Power folders, or all")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--pod-rank", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, help="Default: half each record")
    parser.add_argument("--K", "--k", type=int, default=20,
                        help="Consecutive lags with |ACF|<0.01 (default: 20; 1=first crossing)")
    parser.add_argument("--reduced-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "reduced_data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true", help="Replot saved NPZ results")
    args = parser.parse_args()
    if not 1 <= args.rank <= args.pod_rank:
        parser.error("Require 1 <= rank <= pod-rank.")
    if args.K < 1 or (args.max_lag is not None and args.K > args.max_lag):
        parser.error("Require 1 <= K <= max-lag.")
    powers = sorted(p.name for p in args.reduced_root.iterdir()
                    if p.is_dir() and POWER_PATTERN.fullmatch(p.name)) if args.powers == ["all"] else list(dict.fromkeys(args.powers))
    if not powers or any(not POWER_PATTERN.fullmatch(p) for p in powers):
        parser.error("Supply valid power folders, or all by itself.")
    output = args.output_dir or (args.reduced_root / "pod_decorrelation" /
                                f"{'_'.join(args.powers)}_r{args.rank}_K{args.K}")
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_results(output, powers, args.rank, args.K)
        print(f"Replotted {output}")
        return
    started = perf_counter()
    summary_rows, detail_rows, diagnostic_rows = [], [], []
    modes = np.arange(args.rank + 1)
    for power in powers:
        repetitions = discover_repetitions(args.reduced_root / power, args.pod_rank)
        if not repetitions:
            raise ValueError(f"No POD files for {power}.")
        reference = repetitions[0]
        reference_modes, shape = load_modes(reference.pod_path, args.rank)
        lag_rows, second_rows = [], []
        for rep in repetitions:
            coordinates, _, cosines, energy_ratio, fs = load_coordinates(
                rep, reference_modes, shape, args.rank, 8192)
            max_lag = len(coordinates) // 2 if args.max_lag is None else args.max_lag
            lags = np.concatenate([
                decorrelation_lags(coordinates[:, start:start + 16], max_lag, args.K)
                for start in range(0, args.rank + 1, 16)])
            lag_rows.append(lags)
            second_rows.append(lags / fs)
            detail_rows.extend((power, rep.number, reference.number, int(mode), lag, lag / fs, args.K)
                               for mode, lag in zip(modes, lags))
            diagnostic_rows.append((power, rep.number, reference.number, fs, max_lag,
                                    energy_ratio, cosines.min(), cosines.mean(), str(rep.pod_path)))
            print(f"{power} rep{rep.number}: {np.isfinite(lags).sum()}/{len(lags)} crossings; "
                  f"projection energy={energy_ratio:.4f}", flush=True)
            del coordinates
        lag_mean, lag_std, count = summarize(np.asarray(lag_rows))
        mean, std, _ = summarize(np.asarray(second_rows))
        summary_rows.extend((power, int(mode), int(n), lm, ls, m, s, args.K)
                            for mode, n, lm, ls, m, s in zip(modes, count, lag_mean, lag_std, mean, std))
        np.savez(output / f"{power}.npz", modes=modes, K=args.K,
                 repetitions=[rep.number for rep in repetitions],
                 reference_repetition=reference.number, lags=lag_rows, seconds=second_rows,
                 mean_seconds=mean, std_seconds=std, valid_count=count)
    write_csv(output / "per_repetition.csv",
              ["power", "rep", "reference_rep", "mode", "lag_timesteps", "lag_seconds", "K"], detail_rows)
    write_csv(output / "summary.csv",
              ["power", "mode", "valid_reps", "mean_timesteps", "std_timesteps", "mean_seconds", "std_seconds", "K"], summary_rows)
    write_csv(output / "projection_diagnostics.csv",
              ["power", "rep", "reference_rep", "fs_hz", "max_lag", "projection_energy_ratio",
               "min_principal_cosine", "mean_principal_cosine", "source"], diagnostic_rows)
    (output / "README.md").write_text(
        f"Rank {args.rank}; stored POD rank {args.pod_rank}; K={args.K}.\n\n"
        "Mode 0 is the instantaneous spatial mean. All coordinates are temporally demeaned. "
        "Each power uses its lowest-numbered available repetition as reference. "
        "With spatial modes stored as rows, overlap = reference_modes @ source_modes.T; "
        "aligned = source_coefficients[:, :rank] @ overlap.T. This projects a rank-r "
        "reconstruction, not the full experimental field. Different powers use different bases.\n\n"
        "ACF(k) = sum(x[t]*x[t+k])/sum(x[t]**2). Report the first positive lag L with "
        "abs(ACF)<0.01 at every lag L..L+K-1. The entire window must fit within max-lag "
        "(default: half the record). Correlation may rise again beyond that window. "
        "K=1 reproduces the old first crossing. Seconds use each repetition's sampling frequency. "
        "NaN means constant signal or no qualifying run. Means exclude NaNs; std uses ddof=1 "
        "and requires two valid repetitions. See valid_reps/count for missing crossings. "
        "Shading represents repetition spread, not uncertainty of the mean. Plots use a linear "
        "scale below 0.01 ms and logarithmic above; negative lower bands are outside the displayed axis.\n\n"
        "per_repetition.csv: individual results; summary.csv: means/std; "
        "projection_diagnostics.csv: basis overlap and retained energy; "
        "one NPZ/PNG per power; all_powers.png and decorrelation_summary.pdf: plots.\n")
    plot_results(output, powers, args.rank, args.K)
    print(f"Saved {output}; elapsed {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
