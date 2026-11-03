"""Read-only cached-data review of the advanced selector, not a replacement cleaner."""
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from review_hybrid_temporal_selector import select


def load_selector(source):
    fn = next(node for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef) and node.name == "select_advanced")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<advanced selector>", "exec"), namespace)
    return namespace["select_advanced"]


def main():
    out = Path("runs/advanced_temporal_selector_review")
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs/temporal_alignment_validation/step_events_0p20")
    provenance = json.loads((cache / "trace_provenance.json").read_text())
    for path, item in provenance.items():
        if isinstance(item, dict) and "mtime_ns" in item:
            stat = Path(path).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (item["size"], item["mtime_ns"]), path
    source_path = Path("data_analysis/compare_advanced_temporal_selector.py")
    source = source_path.read_text()
    advanced = load_selector(source)
    # One-factor diagnostic ablation, NOT a recommended production replacement.
    assert source.count("(large_runs <= 2)") == 1
    without_run_gate = load_selector(source.replace("(large_runs <= 2)", "np.ones(n, dtype=bool)"))
    d = np.load(cache / "full_recording_traces.npz")
    time, x, q = d["time"], d["spatial"], provenance["q"]
    fs = 1 / np.mean(np.diff(time))
    dh = np.r_[0, np.diff(x[:, 0])]
    dq = np.vstack([np.zeros((1, 4)), np.diff(x[:, 2:], axis=0)])
    old = np.r_[0, np.diff(d["exported_offsets"])]
    hybrid = select(old, dq, q)
    adv = advanced(dh, dq, q)
    no_run = without_run_gate(dh, dq, q)
    variants = {"spatial_only": np.zeros_like(adv), "greedy": old,
                "hybrid": hybrid, "advanced": adv, "advanced_no_large_run_gate": no_run}
    summary = {"frames": len(time), "q_um": q, "tau_um": provenance["tau"],
               "fs_hz": float(fs), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "advanced_small_events": int(np.count_nonzero((abs(adv) > 0) & (abs(adv) < 3))),
               "advanced_large_events": int(np.count_nonzero(abs(adv) >= 3)),
               "hybrid_only_events": int(np.count_nonzero((hybrid != 0) & (adv == 0))),
               "advanced_only_events": int(np.count_nonzero((hybrid == 0) & (adv != 0))),
               "different_integer_at_shared_events": int(np.count_nonzero((hybrid != 0) & (adv != 0) & (hybrid != adv)))}
    summary["variants"] = {}
    signals, psds = {}, {}
    sos = butter(4, 50, btype="highpass", fs=fs, output="sos")
    for label, shifts in variants.items():
        signals[label] = x[:, :2] - q*np.cumsum(shifts)[:, None]
        increments = dh - q*shifts
        hp = sosfiltfilt(sos, signals[label], axis=0)
        freq, psd = welch(hp, fs=fs, nperseg=16384, noverlap=8192, axis=0)
        psds[label] = psd
        stats = dict(events=int(np.count_nonzero(shifts)),
                     largest_mean_increment_um=float(np.max(abs(increments))),
                     mean_increment_rms_um=float(np.sqrt(np.mean(increments[1:]**2))),
                     counts_above_um={str(v): int(np.count_nonzero(abs(increments) > v)) for v in [10, 20, 40]})
        for low, high in [(100, 500), (500, 2000), (2000, 4000)]:
            mask = (freq >= low) & (freq < high)
            stats[f"power_ratio_{low}_{high}_mean_center"] = (
                psd[mask].sum(axis=0) / psds["spatial_only"][mask].sum(axis=0)).tolist()
        summary["variants"][label] = stats
    bad = (abs(dh-q*adv) > 20)
    summary["large_residual_frames"] = np.flatnonzero(bad).tolist()
    adv_increments = dh-q*adv
    summary["fraction_mean_increment_squared_sum_at_large_residuals"] = float(
        np.sum(adv_increments[bad]**2)/np.sum(adv_increments[1:]**2))
    summary["large_residual_frames_fixed_in_no_run_ablation"] = int(np.count_nonzero(bad & (abs(dh-q*no_run) <= 20)))
    summary["events"] = []
    for t in [1553, 1557, 1563, 17035, 17036, 52254, 52309, 70100, 99064]:
        summary["events"].append(dict(frame=t, mean_increment_um=float(dh[t]),
            quadrant_increments_um=dq[t].tolist(), hybrid_k=int(hybrid[t]),
            advanced_k=int(adv[t]), no_run_k=int(no_run[t]),
            advanced_residual_um=float(dh[t]-q*adv[t])))
    summary["excursions"] = [dict(start=start, end=end,
        changes=int(np.count_nonzero(adv[start:end+1])))
        for start, end in [(10845, 10869), (58433, 58454), (91928, 91946)]]

    # Reuse the actual earlier regression (including its phase), not a new test
    # that misses the threshold but labels the old failure in a legend.
    st = np.arange(8192)/fs
    boundary = 1.0001*q/(4*np.sin(np.pi/128))*np.sin(2*np.pi*(fs/128)*st-np.pi/128)
    tests = {"original_900Hz_threshold_regression": boundary,
             "smooth_1000Hz_20um": 20*np.sin(2*np.pi*1000*st),
             "smooth_1000Hz_100um": 100*np.sin(2*np.pi*1000*st)}
    summary["smooth_controls"] = {}
    for label, signal in tests.items():
        diff = np.r_[0, np.diff(signal)]
        sh = advanced(diff, np.repeat(diff[:, None], 4, axis=1), q)
        summary["smooth_controls"][label] = dict(events=int(np.count_nonzero(sh)),
            max_error_um=float(np.max(abs(q*np.cumsum(sh)))))
    displayed = 20.34504*np.sin(2*np.pi*900*np.arange(250)/fs)
    diff = np.r_[0, np.diff(displayed)]
    displayed_hybrid = select(np.rint(diff/q).astype(int), np.repeat(diff[:, None], 4, axis=1), q)
    summary["comparison_figure_actual_hybrid_synthetic_events"] = int(np.count_nonzero(displayed_hybrid))
    summary["center_minus_mean_invariance_error_um"] = float(np.max(abs(
        np.diff(signals["advanced"], axis=1)-np.diff(x[:, :2], axis=1))))

    fig, axs = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for row, event in enumerate([52309, 17036, 99064]):
        sl = slice(event-6, event+7)
        t = (time[sl]-time[event])*1e6
        ax = axs[row, 0]
        for label in ["spatial_only", "hybrid", "advanced"]:
            y = signals[label][sl, 0]
            ax.plot(t, y-y[0], marker=".", label=label)
        ax.axvline(0, color="k", alpha=.3)
        ax.set(title=f"Frame {event}: mean trajectories, aligned at window start",
               xlabel="Time relative to event [us]", ylabel="Height change [um]")
        ax.legend(fontsize=8)
        ax = axs[row, 1]
        for j in range(4):
            ax.plot(t, dq[sl, j], marker=".", label=f"Quadrant {j+1}")
        ax.axvline(0, color="k", alpha=.3)
        ax.set(title="Spatial-only quadrant increments", xlabel="Time relative to event [us]",
               ylabel="Frame-to-frame increment [um]")
        ax.legend(fontsize=8)
    fig.savefig(out / "real_burst_events.png", dpi=150)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(out / "review_traces.npz", advanced_shifts=adv,
                        advanced_offsets=np.cumsum(adv), no_run_shifts=no_run,
                        frequency=freq, **{f"psd_{k}": v for k, v in psds.items()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
