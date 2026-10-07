"""Quick POD decorrelation: start of first K consecutive lags with |ACF| < 0.01.

ACF(k) = sum_t x(t)*x(t+k) / sum_t x(t)**2, after temporal demeaning.
K defaults to 20; K=1 recovers the first crossing. The entire run must lie in
the search range (first half of record by default). Constant modes or no run
give NaN. Correlation may rise again after the K-lag window.
Mode 0 is the instantaneous spatial mean; modes 1..rank use the repetition's
own POD basis. Each coordinate, including mode 0, is temporally demeaned.

Example: python pod_decorrelation.py 0p20 --rep 1 --rank 100
"""

import argparse
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def decorrelation_lags(coefficients, max_lag, K=20):
    """Return run-start lags for columns, using zero-padded FFT ACF."""
    x = np.array(coefficients, dtype=np.float64, copy=True)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Coefficients must be a finite time-by-mode array.")
    if not 1 <= max_lag < len(x):
        raise ValueError("max_lag must be between 1 and number of samples - 1.")
    if not isinstance(K, (int, np.integer)) or not 1 <= K <= max_lag:
        raise ValueError("K must be an integer between 1 and max_lag.")
    x -= x.mean(axis=0)
    nfft = next_fast_len(2 * len(x) - 1)
    spectrum = rfft(x, n=nfft, axis=0)
    acf = irfft(spectrum * spectrum.conj(), n=nfft, axis=0)[:max_lag + 1]
    energy = np.sum(x * x, axis=0)
    valid = energy > 0
    acf[:, valid] /= energy[valid]
    crossed = (np.abs(acf[1:]) < 0.01) & valid[None, :]
    cumulative = np.vstack((np.zeros((1, x.shape[1]), dtype=np.int64),
                            np.cumsum(crossed, axis=0)))
    runs = cumulative[K:] - cumulative[:-K] == K
    return np.where(runs.any(axis=0), runs.argmax(axis=0) + 1, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("power", help="Power folder, e.g. 0p20")
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--pod-rank", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, help="Default: half the record")
    parser.add_argument("--K", "--k", type=int, default=20,
                        help="Consecutive lags with |ACF|<0.01 (default: 20; 1=first crossing)")
    parser.add_argument("--reduced-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "reduced_data")
    parser.add_argument("--output", type=Path, help="Override output CSV path")
    args = parser.parse_args()
    started = perf_counter()
    matches = sorted((args.reduced_root / args.power).glob(
        f"*_rep{args.rep}/pod_2d_r{args.pod_rank}.h5"))
    if len(matches) != 1:
        parser.error(f"Expected one POD file, found {len(matches)}: {matches}")
    source = matches[0]
    with h5py.File(source, "r") as handle:
        coefficients = handle["reduced/coefficients"]
        time = handle["grid/time"][:]
        if not 1 <= args.rank <= coefficients.shape[1]:
            parser.error(f"Rank must be between 1 and {coefficients.shape[1]}.")
        if (len(time) < 3 or len(time) != coefficients.shape[0]
                or not np.isfinite(time).all() or not np.all(np.diff(time) > 0)):
            parser.error("Expected increasing finite timestamps matching coefficients.")
        dt = (time[-1] - time[0]) / (len(time) - 1)
        max_lag = len(time) // 2 if args.max_lag is None else args.max_lag
        if not 1 <= max_lag < len(time):
            parser.error(f"max-lag must be between 1 and {len(time) - 1}.")
        if not 1 <= args.K <= max_lag:
            parser.error("K must be between 1 and max-lag.")
        spatial_mean = handle["preprocessing/frame_spatial_mean"][:]
        if spatial_mean.shape != time.shape:
            parser.error("Spatial mean and time lengths differ.")
        lags = np.empty(args.rank + 1)
        lags[0] = decorrelation_lags(spatial_mean[:, None], max_lag, args.K)[0]
        for start in range(0, args.rank, 16):
            stop = min(start + 16, args.rank)
            lags[start + 1:stop + 1] = decorrelation_lags(coefficients[:, start:stop], max_lag, args.K)
    output = args.output or (args.reduced_root / "pod_decorrelation" /
                            f"{args.power}_rep{args.rep}_r{args.rank}_K{args.K}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (f"source: {source.resolve()}\n"
              f"dt_seconds: {dt:.16g}; max_lag: {max_lag}; K: {args.K}\n"
              "temporal mean removed; ACF(k)=sum(x[t]*x[t+k])/sum(x[t]**2)\n"
              "first run-start lag with abs(ACF)<0.01 for K consecutive lags; NaN=constant or no run\n"
              "mode 0=instantaneous spatial mean; modes 1..rank=native POD basis\n"
              "mode,decorrelation_timesteps,decorrelation_seconds")
    np.savetxt(output, np.column_stack((np.arange(args.rank + 1), lags, lags * dt)),
               delimiter=",", fmt=["%d", "%.0f", "%.10g"], header=header)
    print(f"Saved spatial mean + {args.rank} POD modes to {output}")
    print(f"{np.isnan(lags).sum()} undefined/no crossing; elapsed {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
