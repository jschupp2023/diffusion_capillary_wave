"""Read-only modal amplitude/timescale comparison of saved native-lag rollouts.

No source HDF5 or checkpoint is modified. Reference paths are never concatenated
or gap-compressed. Output contains exact finite-record structure functions,
per-path demeaned ACFs, long-window PSDs, and explicit reference trim controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import periodogram, welch

from data_analysis.rollout import Rollout
from data_analysis.rollout_metrics import training_trim_specification


def lag_statistics(x, max_lag):
    """[path,time,mode] -> per-path ACF and exact E[(x[t+k]-x[t])^2].

    ACF uses the existing POD decorrelation convention: demean each path and
    divide lagged product sums by its lag-zero sum (biased FFT estimator).
    Structure functions use n-k actual pairs and retain no cross-path pairs.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all() or not 1 <= max_lag < x.shape[1]:
        raise ValueError("finite [path,time,mode] arrays and 1 <= max_lag < time required")
    n = x.shape[1]
    acfs, structures = [], []
    lag = np.arange(max_lag + 1)
    for path in x:
        centered = path - path.mean(axis=0)
        spectrum = rfft(centered, n=next_fast_len(2*n-1), axis=0)
        sums = irfft(spectrum * spectrum.conj(), n=next_fast_len(2*n-1), axis=0)[:max_lag+1]
        energy = np.sum(centered**2, axis=0)
        acfs.append(np.divide(sums, energy, out=np.full_like(sums, np.nan), where=energy > 0))
        prefix = np.concatenate((np.zeros((1, x.shape[-1])), np.cumsum(centered**2, axis=0)))
        d = (prefix[n-lag] + prefix[n] - prefix[lag] - 2*sums) / (n-lag[:, None])
        d[0] = 0.
        structures.append(np.maximum(d, 0.))
    return np.asarray(acfs), np.asarray(structures)


def persistent_crossing(acf, threshold=.01, consecutive=20):
    """First run-start lag with |ACF| below threshold for consecutive lags."""
    values = np.asarray(acf)
    good = np.abs(values[1:]) < threshold
    if len(good) < consecutive:
        return None
    count = np.convolve(good.astype(int), np.ones(consecutive, dtype=int), mode="valid")
    hits = np.flatnonzero(count == consecutive)
    return int(hits[0] + 1) if len(hits) else None


def first_crossing(acf, threshold):
    hits = np.flatnonzero(np.asarray(acf)[1:] <= threshold)
    return int(hits[0]+1) if len(hits) else None


def reference_controls(reference, specification, config, coordinates, lags):
    """Training-scale edge and complete-stencil controls, without gap compression.

    Current velocity is edge index t; future velocity at horizon H is edge t+H.
    Retain every increment from t-history+1 through t+H inclusive, matching
    the training window eligibility for lag=1, except unavailable file edges.
    """
    if specification is None:
        return {}, {}
    delta = np.diff(reference.astype(np.float64), axis=1)
    good = np.linalg.norm(delta / specification["state_std"], axis=-1) <= specification["cutoff"]
    selected = delta[..., coordinates]
    raw_ms = np.mean(selected**2, axis=(0, 1))
    retained_ms = np.mean(selected[good]**2, axis=0)
    out = dict(edge_retained_fraction=float(good.mean()),
               edge_retained_increment_rms=np.sqrt(retained_ms).tolist(),
               removed_increment_square_fraction=(1-(selected[good]**2).sum(0)/(selected**2).sum((0,1))).tolist())
    prefix = np.concatenate((np.zeros((len(good),1), dtype=int), np.cumsum(~good, axis=1)), axis=1)
    history = config["model"]["history_steps"]
    maximum = max(config["training"].get("multistep_horizons", [1])) if config["training"].get("multistep_weight", 0) > 0 else 1
    indices = np.arange(history-1, selected.shape[1]-maximum)
    mask = prefix[:, indices+maximum+1] == prefix[:, indices-history+1]
    out.update(training_window_horizon=maximum,
               training_window_retained_fraction=float(mask.mean()),
               training_window_current_increment_rms=np.sqrt(np.mean(selected[:,indices][mask]**2,axis=0)).tolist(),
               training_window_next_increment_rms=np.sqrt(np.mean(selected[:,indices+1][mask]**2,axis=0)).tolist())
    fixed_correlations={}
    for lag in lags:
        if lag > maximum:
            continue
        a,b=selected[:,indices][mask],selected[:,indices+lag][mask]
        a,b=a-a.mean(0),b-b.mean(0)
        fixed_correlations[str(lag)]=((a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))).tolist()
    out["fixed_training_window_velocity_correlations"]=fixed_correlations
    correlations, counts = [], []
    for lag in lags:
        idx = np.arange(history-1, selected.shape[1]-lag)
        keep = prefix[:,idx+lag+1] == prefix[:,idx-history+1]
        a, b = selected[:,idx][keep], selected[:,idx+lag][keep]
        counts.append(int(keep.sum()))
        if len(a) < 2:
            correlations.append(np.full(len(coordinates), np.nan))
        else:
            a, b = a-a.mean(0), b-b.mean(0)
            correlations.append((a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0)))
    return out, dict(trim_velocity_lags=np.asarray(lags),
                     trim_velocity_correlations=np.asarray(correlations),
                     trim_velocity_pair_counts=np.asarray(counts))


def summarize_population(x, fs, max_lag, nperseg):
    result = {}
    for name, signal in (("position", x), ("velocity", np.diff(x,axis=1)),
                         ("acceleration", np.diff(x,n=2,axis=1))):
        acf, structure = lag_statistics(signal, min(max_lag,signal.shape[1]-1))
        result[f"{name}_acf"] = acf.mean(0)
        result[f"{name}_acf_paths"] = acf
        if name == "position":
            result["structure"] = structure.mean(0)
        result[f"{name}_rms"] = np.sqrt(np.mean(signal**2,axis=(0,1)))
        result[f"{name}_within_variance"] = np.var(signal,axis=1).mean(0)
    result["trajectory_mean_square"] = np.mean(x.mean(1)**2,axis=0)
    for size in sorted(set((1024, nperseg))):
        f, p = welch(x, fs=fs, window="hann", nperseg=min(size,x.shape[1]),
                     noverlap=min(size,x.shape[1])//2, detrend="constant", scaling="density", axis=1)
        result[f"frequency_{size}"] = f
        result[f"psd_{size}"] = p.mean(0)
        result[f"psd_{size}_paths"] = p
    f, p = periodogram(x, fs=fs, window="boxcar", detrend="constant", scaling="density", axis=1)
    result["periodogram_frequency"] = f
    result["periodogram"] = p.mean(0)
    # Sum*df closes exactly to per-path temporal variance (not pooled variance).
    result["periodogram_variance"] = p.mean(0).sum(0) * (f[1]-f[0])
    return result


def plot_run(path, label, modes, fs, arrays, nperseg):
    fig, axes = plt.subplots(4, min(3,len(modes)), figsize=(14,13), squeeze=False)
    colors = {"reference":"black", "generated":"tab:blue"}
    for j, mode in enumerate(modes[:3]):
        for population in colors:
            color=colors[population]
            acf=arrays[f"{population}_position_acf"][:,j]
            axes[0,j].plot(np.arange(len(acf))/fs*1000,acf,color=color,label=population)
            v=arrays[f"{population}_velocity_acf"][:41,j]
            a=arrays[f"{population}_acceleration_acf"][:41,j]
            axes[1,j].plot(np.arange(len(v)),v,color=color,label=population+" velocity")
            axes[1,j].plot(np.arange(len(a)),a,color=color,ls=":",label=population+" acceleration")
            d=arrays[f"{population}_structure"][:,j]
            axes[2,j].loglog(np.arange(1,len(d))/fs*1000,d[1:],color=color,label=population)
            for size, style in ((1024,":"),(nperseg,"-")):
                f=arrays[f"{population}_frequency_{size}"]
                p=arrays[f"{population}_psd_{size}"][:,j]
                axes[3,j].loglog(f[1:],p[1:],color=color,ls=style,label=f"{population}, {size}")
        axes[0,j].set_title(f"POD {mode}")
        for row in range(4):
            axes[row,j].grid(alpha=.2)
        axes[0,j].axhline(0,color="gray",lw=.5)
        axes[1,j].axhline(0,color="gray",lw=.5)
        axes[0,j].set_xlabel("Lag (ms)")
        axes[1,j].set_xlabel("Lag (native frames)")
        axes[2,j].set_xlabel("Lag (ms)")
        axes[3,j].set_xlabel("Frequency (Hz)")
    for row, title in enumerate(("Position ACF", "Velocity / acceleration ACF", "Displacement mean square", "PSD (coefficient units²/Hz)")):
        axes[row,0].set_ylabel(title)
        axes[row,0].legend(fontsize=7)
    fig.suptitle(label,fontsize=11)
    fig.tight_layout()
    fig.savefig(path / "timescales.png",dpi=160)
    plt.close(fig)


def analyze(path, output, modes, discard, max_lag, nperseg):
    rollout=Rollout(path,discard=discard)
    if rollout.lag_steps != 1:
        raise ValueError("This analysis requires native lag=1; coarse displacement is not native velocity")
    config_path=rollout.path.parent.parent / "config.json"
    config=json.loads(config_path.read_text())
    if config["model"]["state_variable"] != "velocity":
        raise ValueError("Velocity model required")
    label=rollout.path.parent.parent.name+"__"+rollout.path.parent.name
    destination=output/label
    destination.mkdir(parents=True,exist_ok=False)
    coordinates=[mode-int(not rollout.include_spatial_mean) for mode in modes]
    generated=rollout.generated[...,coordinates].reshape(-1,rollout.generated.shape[2],len(modes)).astype(np.float64)
    reference=rollout.reference[...,coordinates].astype(np.float64)
    transient_checks={}
    for skip in (0,5000,25000):
        if skip >= generated.shape[1]-2:
            continue
        g,r=generated[:,skip:],reference[:,skip:]
        transient_checks[str(skip)]={
            "position_rms_ratio":np.sqrt(np.mean(g*g,axis=(0,1))/np.mean(r*r,axis=(0,1))).tolist(),
            "velocity_rms_ratio":np.sqrt(np.mean(np.diff(g,axis=1)**2,axis=(0,1))/np.mean(np.diff(r,axis=1)**2,axis=(0,1))).tolist()}
    arrays={}
    for population, x in (("reference",reference),("generated",generated)):
        arrays.update({f"{population}_{key}":value for key,value in summarize_population(
            x,rollout.fs,max_lag,nperseg).items()})
    spec=training_trim_specification(rollout)
    controls,extra=reference_controls(rollout.reference,spec,config,coordinates,[1,2,4,5,10,20,50,100])
    arrays.update(extra)
    fingerprint=hashlib.sha256(np.ascontiguousarray(rollout.reference).tobytes()).hexdigest()
    rows=[]
    for j,mode in enumerate(modes):
        row=dict(mode=mode)
        for quantity in ("position","velocity","acceleration"):
            row[quantity+"_rms_ratio"]=float(arrays[f"generated_{quantity}_rms"][j]/arrays[f"reference_{quantity}_rms"][j])
            for pop in ("reference","generated"):
                acf=arrays[f"{pop}_{quantity}_acf"][:,j]
                row[f"{pop}_{quantity}_acf"]= {str(k):float(acf[k]) for k in (1,2,4,5,10,20,50,100) if k<len(acf)}
                row[f"{pop}_{quantity}_first_1e_crossing"]=first_crossing(acf,1/np.e)
                row[f"{pop}_{quantity}_first_zero"]=first_crossing(acf,0)
                row[f"{pop}_{quantity}_persistent_001_K20"]=persistent_crossing(acf)
        row["within_position_variance_ratio"]=float(arrays["generated_position_within_variance"][j]/arrays["reference_position_within_variance"][j])
        row["displacement_ratio"]={str(k):float(arrays["generated_structure"][k,j]/arrays["reference_structure"][k,j]) for k in (1,2,5,10,20,100,500,1000,5000) if k<=max_lag}
        for pop in ("reference","generated"):
            row[pop+"_trajectory_mean_square"]=float(arrays[pop+"_trajectory_mean_square"][j])
            row[pop+"_within_variance"]=float(arrays[pop+"_position_within_variance"][j])
            row[pop+"_periodogram_closure_ratio"]=float(arrays[pop+"_periodogram_variance"][j]/row[pop+"_within_variance"])
        for low,high in ((0,112.5),(112.5,1000),(1000,10000),(10000,60000)):
            f=arrays["reference_periodogram_frequency"]
            mask=(f>0)&(f>=low)&(f<high)
            powers={pop:float(arrays[pop+"_periodogram"][mask,j].sum()*(f[1]-f[0])) for pop in ("reference","generated")}
            row[f"band_{low:g}_{high:g}_ratio"]=powers["generated"]/powers["reference"]
            row[f"band_{low:g}_{high:g}_variance_excess"]=powers["generated"]-powers["reference"]
        if controls:
            row["velocity_rms_ratio_to_edge_trimmed_reference"]=float(arrays["generated_velocity_rms"][j]/controls["edge_retained_increment_rms"][j])
            row["velocity_rms_ratio_to_training_window_reference"]=float(arrays["generated_velocity_rms"][j]/controls["training_window_current_increment_rms"][j])
            row["reference_increment_square_fraction_removed"]=controls["removed_increment_square_fraction"][j]
        rows.append(row)
    report=dict(label=label,provenance=rollout.provenance(),reference_sha256=fingerprint,
                config_path=str(config_path.resolve()),experiment=config["data"]["experiment"],
                model=config["model"],training={k:v for k,v in config["training"].items() if k in (
                    "epochs","seed","multistep_weight","multistep_horizons","multistep_particles","train_trim_percent","stability")},
                split=rollout.metadata["split"],diffusion_scale=rollout.metadata.get("diffusion_scale",1.),
                trim_controls=controls,discard_sensitivity=transient_checks,modes=rows)
    np.savez_compressed(destination/"curves.npz",**arrays)
    (destination/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    plot_run(destination,label,modes,rollout.fs,arrays,nperseg)
    print(label, [(r["mode"],round(r["position_rms_ratio"],3),round(r["velocity_rms_ratio"],3)) for r in rows[:3]],flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--modes",type=int,nargs="+",default=[1,2,3,4,5])
    parser.add_argument("--discard",type=int,default=0)
    parser.add_argument("--max-lag",type=int,default=8192)
    parser.add_argument("--nperseg",type=int,default=16384)
    args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True,exist_ok=True)
    reports=[analyze(path,args.output,args.modes,args.discard,args.max_lag,args.nperseg) for path in args.rollouts]
    (args.output/"comparison.json").write_text(json.dumps(reports,indent=2)+"\n")
    checks={r["label"]:dict(fixed_training_window_control=r["trim_controls"],
                            discard_sensitivity=r["discard_sensitivity"]) for r in reports}
    (args.output/"training_stencil_and_transient_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    with (args.output/"comparison.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=["run","split","reference_group","diffusion_scale","multistep_weight","mode",
            "position_rms_ratio","velocity_rms_ratio","velocity_rms_ratio_to_training_window_reference","acceleration_rms_ratio","within_position_variance_ratio"])
        writer.writeheader()
        for r in reports:
            for mode in r["modes"]:
                row={key:mode[key] for key in writer.fieldnames if key in mode}
                row.update(run=r["label"],split=r["split"],reference_group=r["reference_sha256"][:12],
                           diffusion_scale=r["diffusion_scale"],multistep_weight=r["training"].get("multistep_weight",0))
                writer.writerow(row)
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k=="rollouts" else v for k,v in vars(args).items()},
        conventions=["Shared-POD physical coefficients; mode numbers exclude spatial mean.",
        "All saved conditions and ensemble paths; equal path weighting; no temporal mean field restored.",
        "ACF: each trajectory demeaned separately, lag sums divided by lag-zero sum; no gap compression.",
        "K20 crossing does not mean independence or permanent decorrelation; it can recross.",
        "Structure functions use exact finite-record n-k pairs, not stationary approximation.",
        "PSD: Hann Welch with constant per-segment detrend at 1024 and requested nperseg; density scaling.",
        "Full-record boxcar periodogram variance uses sum*df for exact Parseval closure after per-path demeaning.",
        "Reference trim uses saved all-coordinate pre-trim scales/cutoff; generated trajectories never trimmed.",
        "Training-window control uses history and maximum enabled NLL horizon; initial unavailable stencils excluded.",
        "Trim correlations keep whole intervening stencils and original time indices; survivor-conditioned, not global ACF.",
        "References can overlap/repeat; ensemble paths share starts. No independence-based confidence intervals.",
        "Equal reference hashes identify exact saved reference arrays, not identical model/training configurations."])
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
