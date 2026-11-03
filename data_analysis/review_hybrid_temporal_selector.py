"""Independent, trace-only audit; never modifies the cleaner or input HDF5s.

Run from the repository root. Uses the existing full-recording validation cache.
Synthetic uniform motion is a mathematical control, not an asserted model of the
experiment. All decisions use unfiltered increments; filtering is only for PSDs.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def select(dk, quad_diffs, q, large=3, tolerance=0.6):
    """Vectorized equivalent of the reviewed hybrid rule (defaults exact)."""
    signs = np.sign(dk)
    boundaries = np.r_[0, np.flatnonzero(np.diff(signs)) + 1, len(dk)]
    lengths = np.repeat(np.diff(boundaries), np.diff(boundaries))
    consensus = (np.abs(quad_diffs - dk[:, None] * q) <= tolerance*q).sum(axis=1)
    keep = (np.abs(dk) >= large) | (
        (np.abs(dk) > 0) & (np.abs(dk) < large) & (lengths == 1) & (consensus >= 3))
    out = np.where(keep, dk, 0)
    out[0] = 0
    return out


def exact_source_selection(source, dk, quads, q):
    # Only execute the self-contained selector statements: importing that script
    # also writes to an external Antigravity folder and generates unrelated plots.
    block = source[source.index("run_lengths ="):source.index("offsets_hybrid =")]
    scope = dict(np=np, n_frames=len(dk), dk_appA=dk, spatial_quads=quads, q=q)
    exec(compile(block, "<reviewed selector block>", "exec"), scope)
    return scope["shifts_hybrid"]


def synthetic_case(name, truth, slips, q, time, frequency=None):
    observed = truth + q*slips
    increment = np.r_[0, np.diff(observed)]
    dk = np.rint(increment/q).astype(np.int64)
    selected = select(dk, np.repeat(increment[:, None], 4, axis=1), q)
    corrected = observed - q*np.cumsum(selected)
    error = corrected-truth
    row = dict(name=name, selected_transitions=int(np.count_nonzero(selected)),
               max_error_um=float(np.max(np.abs(error))),
               rms_error_um=float(np.sqrt(np.mean(error**2))),
               terminal_error_cycles=float(error[-1]/q),
               max_true_increment_um=float(np.max(np.abs(np.diff(truth)))))
    if frequency is not None:
        design = np.column_stack([np.sin(2*np.pi*frequency*time),
                                  np.cos(2*np.pi*frequency*time), np.ones(len(time))])
        a = np.linalg.lstsq(design, truth, rcond=None)[0][:2]
        b = np.linalg.lstsq(design, corrected, rcond=None)[0][:2]
        row["fundamental_amplitude_ratio"] = float(np.linalg.norm(b)/np.linalg.norm(a))
    return row, observed, corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/hybrid_temporal_selector_review"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs/temporal_alignment_validation/step_events_0p20")
    provenance = json.loads((cache/"trace_provenance.json").read_text())
    for path, record in provenance.items():
        if isinstance(record, dict) and "mtime_ns" in record:
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
                raise RuntimeError(f"Stale trace cache: {path}")
    q = provenance["q"]
    d = np.load(cache/"full_recording_traces.npz")
    time, spatial = d["time"], d["spatial"]
    fs = 1/np.mean(np.diff(time))
    dk = np.r_[0, np.diff(d["exported_offsets"])].astype(np.int64)
    quad_diffs = np.vstack([np.zeros((1, 4)), np.diff(spatial[:, 2:], axis=0)])
    source_path = Path("data_analysis/evaluate_hybrid_temporal_selector.py")
    source = source_path.read_text()
    hybrid = select(dk, quad_diffs, q)
    exact = exact_source_selection(source, dk, spatial[:, 2:], q)
    np.testing.assert_array_equal(hybrid, exact)
    # Check the implementation on inputs unlike the real trace as well.
    rng = np.random.default_rng(514)
    for _ in range(5):
        test_dk = rng.integers(-5, 6, 200)
        test_dk[0] = 0
        test_diffs = test_dk[:, None]*q + rng.normal(0, q, (200, 4))
        test_diffs[0] = 0
        test_quads = np.cumsum(test_diffs, axis=0)
        np.testing.assert_array_equal(select(test_dk, test_diffs, q),
                                      exact_source_selection(source, test_dk, test_quads, q))

    large_mask = np.abs(dk) >= 3
    votes = (np.abs(quad_diffs-dk[:, None]*q) <= 0.6*q).sum(axis=1)
    summary = dict(q_um=q, spatial_tau_um=provenance["tau"], fs_hz=float(fs),
                   frames=len(time), selector_source=str(source_path),
                   selector_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   exact_source_match=True,
                   old_changes=int(np.count_nonzero(dk)),
                   hybrid_changes=int(np.count_nonzero(hybrid)),
                   accepted_small=int(np.count_nonzero((hybrid != 0) & ~large_mask)),
                   accepted_large=int(np.count_nonzero(large_mask)),
                   large_failing_small_event_consensus=int(np.count_nonzero(large_mask & (votes < 3))))
    # Source uses historical float32-exported shifts; quantify drift from a clean
    # native-increment calculation instead of silently treating them as identical.
    native = np.rint(d["median_increments"][:, 0]/q).astype(np.int64)
    summary["exported_vs_native_candidate_mismatch_frames"] = np.flatnonzero(native != dk).tolist()
    native_hybrid = select(native, quad_diffs, q)
    second = select(native-native_hybrid, quad_diffs-q*native_hybrid[:, None], q)
    summary["second_pass_additional_changes_native_arithmetic"] = int(np.count_nonzero(second))
    summary["second_pass_max_extra_offset_cycles"] = int(np.max(np.abs(np.cumsum(second))))
    summary["excursion_windows"] = []
    for start, end in [(10845, 10869), (58433, 58454), (91928, 91946)]:
        summary["excursion_windows"].append(dict(start=start, end=end,
            hybrid_changes=int(np.count_nonzero(hybrid[start:end+1])),
            spatial_mean_change_um=float(spatial[end, 0]-spatial[start-1, 0]),
            hybrid_mean_change_um=float(spatial[end, 0]-spatial[start-1, 0]-q*hybrid[start:end+1].sum())))

    variants = {"spatial_only": np.zeros_like(dk), "old_greedy": dk,
                "large_only": np.where(large_mask, dk, 0), "hybrid": hybrid}
    for threshold in (2, 4):
        variants[f"hybrid_large{threshold}"] = select(dk, quad_diffs, q, large=threshold)
    for tolerance in (0.4, 0.8):
        variants[f"hybrid_tol{tolerance}"] = select(dk, quad_diffs, q, tolerance=tolerance)
    sos = butter(4, 50., btype="highpass", fs=fs, output="sos")
    psds, band_power = {}, {}
    for label, shifts in variants.items():
        corrected = spatial[:, :2]-q*np.cumsum(shifts)[:, None]
        hp = sosfiltfilt(sos, corrected, axis=0)
        freq, psd = welch(hp, fs=fs, nperseg=16384, noverlap=8192, axis=0)
        psds[label] = psd
        band_power[label] = dict(changes=int(np.count_nonzero(shifts)))
        for low, high in [(100, 500), (500, 2000), (2000, 4000)]:
            mask = (freq >= low) & (freq < high)
            power = np.sum(psd[mask], axis=0)*(freq[1]-freq[0])
            band_power[label][f"{low}_{high}_um2"] = power.tolist()
    for label in variants:
        for low, high in [(100, 500), (500, 2000), (2000, 4000)]:
            key = f"{low}_{high}_um2"
            band_power[label][key.replace("um2", "fraction_spatial")] = (
                np.array(band_power[label][key])/band_power["spatial_only"][key]).tolist()
    summary["band_power_mean_then_center"] = band_power

    n = 8192
    st = np.arange(n)/fs
    zero = np.zeros(n)
    cases = []
    for f, a in [(140., 80.), (1000., 20.), (365., 80.), (1000., 100.)]:
        cases.append((f"smooth_{f:g}Hz_{a:g}um", a*np.sin(2*np.pi*f*st), zero, f))
    # Exactly one above-threshold sample per half cycle: a smooth near-threshold
    # sinusoid satisfies regional consensus but gets classified as isolated slips.
    f_boundary = fs/128
    amplitude = 1.0001*q/(4*np.sin(np.pi/128))
    boundary = amplitude*np.sin(2*np.pi*f_boundary*st-np.pi/128)
    cases.append(("smooth_isolated_threshold_crossing", boundary, zero, f_boundary))
    for name, events in [("isolated_true_1q", [(256, 1)]),
                         ("consecutive_true_1q", [(256, 1), (257, 1)]),
                         ("isolated_true_3q", [(256, 3)]),
                         ("adjacent_true_3q_then_1q", [(256, 3), (257, 1)])]:
        steps = np.zeros(n)
        for index, delta in events:
            steps[index:] += delta
        cases.append((name, zero, steps, None))
    cases.append(("true_1q_during_fast_smooth_slope", np.arange(n)*0.7*q,
                  np.where(np.arange(n) >= 256, 1, 0), None))
    synthetic, signals = [], {}
    for name, truth, slips, frequency in cases:
        row, observed, corrected = synthetic_case(name, truth, slips, q, st, frequency)
        synthetic.append(row)
        signals[name] = (truth, observed, corrected)
    summary["synthetic_controls"] = synthetic
    summary["threshold_control"] = dict(frequency_hz=float(f_boundary), amplitude_um=float(amplitude))

    # Integer steps cannot change the modulo-q increment distribution.
    steps = rng.integers(-10, 11, len(time))
    increments = np.diff(spatial[:, 0])
    wrapped = lambda x: (x+q/2) % q-q/2
    summary["modulo_invariance_max_error_um"] = float(np.max(np.abs(
        wrapped(increments+q*np.diff(steps))-wrapped(increments))))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, name, sl in [(axs[0, 0], "smooth_isolated_threshold_crossing", slice(110, 210)),
                         (axs[0, 1], "consecutive_true_1q", slice(245, 280))]:
        truth, observed, corrected = signals[name]
        ax.plot(st[sl]*1e3, observed[sl], color="0.65", lw=3, label="Input")
        ax.plot(st[sl]*1e3, truth[sl], "k--", label="Known truth")
        ax.plot(st[sl]*1e3, corrected[sl], label="Hybrid output")
        ax.set(title=name.replace("_", " "), xlabel="Time [ms]", ylabel="Height [um]")
        ax.legend(fontsize=8)
    for col, title in [(0, "Spatial mean"), (1, "Center height")]:
        ax = axs[1, col]
        mask = (freq >= 50) & (freq <= 4000)
        for label in ["spatial_only", "old_greedy", "large_only", "hybrid"]:
            ax.loglog(freq[mask], psds[label][mask, col], label=label)
        ax.set(title=title+": matched PSDs, not ground truth", xlabel="Frequency [Hz]", ylabel="PSD [um²/Hz]")
        ax.legend(fontsize=8)
    fig.savefig(args.out/"critical_checks.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(args.out/"review_traces.npz", time=time, hybrid_shifts=hybrid,
                        hybrid_offsets=np.cumsum(hybrid), frequency=freq,
                        **{f"psd_{label}": value for label, value in psds.items()})
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
