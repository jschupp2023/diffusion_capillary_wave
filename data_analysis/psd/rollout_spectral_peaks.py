"""Reference-guided center-point PSD peak contrast for saved rollouts.

Each path has its own Welch transform; spectra, never signals, are averaged.
Reference peak and valley bins are frozen before inspecting generated power.
The contrast score is invariant to a global PSD multiplier. Band power is
reported separately. This describes second-order spectra, not phase coupling.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory
from .pod_center_psd import select_point
from .rollout_psd import ensemble_psd, point_signals, fluctuation_signals


def validate_settings(nperseg, band, prominence_db, min_distance_hz):
    if not isinstance(nperseg, (int, np.integer)) or nperseg < 4:
        raise ValueError("Peak PSD nperseg must be an integer >= 4.")
    if len(band) != 2 or not np.isfinite(band).all() or not 0 < band[0] < band[1]:
        raise ValueError("Peak PSD band must have finite 0 < lower < upper frequencies.")
    if not np.isfinite(prominence_db) or prominence_db <= 0:
        raise ValueError("Peak prominence must be finite and positive.")
    if not np.isfinite(min_distance_hz) or min_distance_hz <= 0:
        raise ValueError("Minimum peak distance must be finite and positive.")


def peak_metrics(frequency, reference, generated, *, band=(100., 1500.),
                 prominence_db=2., min_distance_hz=40., sufficient_samples=True):
    """Score path-averaged PSDs using reference-only peak/valley locations.

    Detection uses dB prominence and a minimum separation in Hz. For each
    accepted reference peak, choose the minimum reference bin on either side,
    bounded by neighboring accepted peaks or the analysis-band edges. Contrast
    is the dB PSD at the peak minus the linearly interpolated dB valley baseline.
    Exactly the same three bins and interpolation weights are used for generated
    PSD. No generated peak search or amplitude normalization is performed.
    """
    validate_settings(4, band, prominence_db, min_distance_hz)
    f, r, g = (np.asarray(x, dtype=float) for x in (frequency, reference, generated))
    if (f.ndim != 1 or len(f) < 3 or r.shape != f.shape or g.shape != f.shape
            or any(not np.isfinite(x).all() for x in (f, r, g))
            or np.any(r < 0) or np.any(g < 0) or np.any(np.diff(f) <= 0)):
        raise ValueError("Expected finite nonnegative PSDs on an increasing frequency grid.")
    df = f[1] - f[0]
    if not np.allclose(np.diff(f), df, rtol=1e-6):
        raise ValueError("Expected a uniform Welch frequency grid.")
    indices = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    report = dict(status="ok", peak_contrast_error_db=None, reference_peak_count=None,
                  generated_over_reference_band_power=None, reference_band_power=None,
                  generated_band_power=None, peaks=[])
    if len(indices) < 3 or f[0] > band[0] or f[-1] < band[1]:
        report["status"] = "insufficient_frequency_coverage"
        return report
    rp, gp = float(r[indices].sum() * df), float(g[indices].sum() * df)
    report.update(reference_band_power=rp, generated_band_power=gp,
                  generated_over_reference_band_power=gp / rp if rp > 0 else None)
    if not sufficient_samples:
        report["status"] = "insufficient_samples"
        return report
    if rp <= 0 or gp <= 0:
        report["status"] = "no_reference_power" if rp <= 0 else "no_generated_power"
        return report
    # Signal-relative floors preserve scale invariance, including isolated zero bins.
    rd = 10 * np.log10(np.maximum(r / r.max(), 1e-15))
    gd = 10 * np.log10(np.maximum(g / g.max(), 1e-15))
    positions, properties = find_peaks(
        rd[indices], prominence=prominence_db,
        distance=max(1, int(np.ceil(min_distance_hz / df))))
    peaks = indices[positions]
    for j, p in enumerate(peaks):
        lo = int(peaks[j - 1]) if j else int(indices[0])
        hi = int(peaks[j + 1]) if j + 1 < len(peaks) else int(indices[-1])
        left = lo + int(np.argmin(rd[lo:p]))
        right = p + 1 + int(np.argmin(rd[p + 1:hi + 1]))
        weight = (f[p] - f[left]) / (f[right] - f[left])
        rc = float(rd[p] - ((1 - weight) * rd[left] + weight * rd[right]))
        gc = float(gd[p] - ((1 - weight) * gd[left] + weight * gd[right]))
        report["peaks"].append(dict(
            frequency_hz=float(f[p]), peak_bin=int(p),
            left_valley_hz=float(f[left]), right_valley_hz=float(f[right]),
            left_valley_bin=int(left), right_valley_bin=int(right),
            reference_detection_prominence_db=float(properties["prominences"][j]),
            reference_contrast_db=rc, generated_contrast_db=gc,
            contrast_error_db=abs(gc - rc)))
    report["reference_peak_count"] = len(peaks)
    if len(peaks):
        report["peak_contrast_error_db"] = float(np.mean([p["contrast_error_db"] for p in report["peaks"]]))
    else:
        report["status"] = "no_reference_peaks"
    return report


def summary_text(report, *, detailed=True):
    def value(x):
        return "N/A" if x is None else f"{x:.4g}"
    m = report["metrics"]
    lines = ["## Spectral peaks", "",
             f"{report['signal'].replace('_', ' ')}; {report['band_hz'][0]:g}–{report['band_hz'][1]:g} Hz. "
             f"Reference split: {report['reference_split']}. Status: `{m['status']}`.", "",
             "| Metric | Value |", "|---|---:|",
             f"| Peak-contrast error [dB], lower is better | {value(m['peak_contrast_error_db'])} |",
             f"| Generated/reference band power | {value(m['generated_over_reference_band_power'])} |",
             f"| Reference peaks assessed | {value(m['reference_peak_count'])} |", "",
             f"Requested/used nperseg: {report['requested_nperseg']}/{report['nperseg']}; "
             f"frequency spacing {report['frequency_spacing_hz']:.4g} Hz.", "",
             "Peak and valley frequencies are selected from the reference only. Contrast uses the same "
             "bins in both spectra and a linear baseline in dB between neighboring reference valleys. "
             "The contrast score is invariant to overall amplitude; band power measures amplitude separately.", ""]
    if m["status"] == "insufficient_samples":
        lines += ["The trajectory is shorter than the requested segment length. The plot and band power "
                  "use the available length, but the comparable peak score is withheld.", ""]
    if not detailed:
        lines += ["See `psd_peaks/quick_summary.md` for individual peaks and `psd_peaks/psd_peaks.png` for the comparison."]
        return "\n".join(lines) + "\n"
    if m["peaks"]:
        lines += ["| Reference peak [Hz] | Reference contrast [dB] | Generated contrast [dB] | Error [dB] |",
                  "|---|---:|---:|---:|"]
        for p in m["peaks"]:
            lines.append(f"| {p['frequency_hz']:.2f} | {p['reference_contrast_db']:.3f} | "
                         f"{p['generated_contrast_db']:.3f} | {p['contrast_error_db']:.3f} |")
    lines += ["", "This is a reference-guided spectral-shape diagnostic, not a test of phase coupling. "
              "Extra generated peaks outside the reference locations are not directly scored. "
              "Reference windows may overlap; trajectory counts are not independent experimental sample counts."]
    return "\n".join(lines) + "\n"


def analyze(rollout, *, point=None, nperseg=8192, band=(100., 1500.),
            prominence_db=2., min_distance_hz=40., output=None, overwrite=False):
    validate_settings(nperseg, band, prominence_db, min_distance_hz)
    out = output_directory(rollout, "psd_peaks", output, overwrite)
    selected = select_point(rollout.modes.shape[1:], rollout.x, rollout.y,
                            rollout.x_units, rollout.y_units, point, None)
    if rollout.include_spatial_mean:
        g = point_signals(rollout.generated, rollout.modes, rollout.n_space, selected)["center"]
        r = point_signals(rollout.reference, rollout.modes, rollout.n_space, selected)["center"]
        signal = "center"
    else:
        signal = "center_without_spatial_mean"
        g = fluctuation_signals(rollout.generated, rollout.modes, selected)[signal]
        r = fluctuation_signals(rollout.reference, rollout.modes, selected)[signal]
    f, gp, segment, overlap = ensemble_psd(g, rollout.fs, nperseg, .5)
    _, rp, _, _ = ensemble_psd(r, rollout.fs, nperseg, .5)
    # Equal condition weights; reference is not duplicated for ensemble members.
    gm = gp.mean(axis=(0, 1), dtype=np.float64)
    rm = rp.mean(axis=0, dtype=np.float64)
    metrics = peak_metrics(f, rm, gm, band=band, prominence_db=prominence_db,
                           min_distance_hz=min_distance_hz, sufficient_samples=r.shape[-1] >= nperseg)
    report = rollout.provenance()
    report.update(signal=signal, reference_split=rollout.metadata.get("split", "unspecified"),
                  point=dict(y_index=selected.y_index, x_index=selected.x_index,
                             y=selected.y_coordinate, x=selected.x_coordinate),
                  band_hz=list(band), requested_nperseg=nperseg, nperseg=segment,
                  noverlap=overlap, frequency_spacing_hz=float(f[1] - f[0]),
                  welch_segments_per_path=1 + (r.shape[-1] - segment) // (segment - overlap),
                  estimator="Per-path Hann Welch, constant detrend, density; arithmetic mean of PSDs",
                  reference_population="Saved contiguous reference windows; no additional frame trimming or training-data pooling",
                  prominence_threshold_db=prominence_db, minimum_peak_distance_hz=min_distance_hz,
                  contrast_definition="Fixed reference peak bin minus frequency-interpolated dB baseline between reference valley bins",
                  band_power_definition="Sum of PSD bins whose centers lie inside the band, multiplied by df",
                  density_units=f"{rollout.units}^2/Hz", metrics=metrics)
    np.savez_compressed(out / "psd.npz", frequency_hz=f, generated_psd=gp, reference_psd=rp,
                        generated_mean_psd=gm, reference_mean_psd=rm)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "quick_summary.md").write_text(summary_text(report))
    fig, ax = plt.subplots(figsize=(12, 4.7), constrained_layout=True)
    mask = (f >= band[0]) & (f <= band[1])
    for p, label, color in ((rm, "Reference", "C1"), (gm, "Generated", "C0")):
        ax.semilogy(f[mask], np.maximum(p[mask], np.finfo(float).tiny), label=label, color=color)
    for j, peak in enumerate(metrics["peaks"]):
        i, left, right = (peak[k] for k in ("peak_bin", "left_valley_bin", "right_valley_bin"))
        ax.plot(f[i], rm[i], "v", color="C1", label="Reference peaks" if j == 0 else None)
        ax.plot(f[[left, right]], rm[[left, right]], ":.", color="0.5", lw=.8,
                label="Reference valley baseline" if j == 0 else None)
        ax.annotate(f"{f[i]:.0f}", (f[i], rm[i]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    score = metrics["peak_contrast_error_db"]
    result = f"contrast error {score:.2f} dB; {metrics['reference_peak_count']} peaks" if score is not None else metrics["status"].replace("_", " ")
    ax.set(xlim=band, xlabel="Frequency [Hz]", ylabel=f"PSD [{rollout.units}²/Hz]",
           title=f"{signal.replace('_', ' ').capitalize()} spectral peaks — {result}\n"
                 f"nperseg={segment}; Δf={f[1] - f[0]:.2f} Hz")
    ax.grid(alpha=.18); ax.legend(fontsize=8)
    fig.savefig(out / "psd_peaks.png", dpi=170)
    plt.close(fig)
    return out


def include_in_quantitative_summary(quantitative, peaks):
    """Expose the focused result alongside the central rollout summary."""
    report = json.loads((peaks / "summary.json").read_text())
    path = quantitative / "summary.json"
    central = json.loads(path.read_text())
    central["spectral_peaks"] = report
    path.write_text(json.dumps(central, indent=2) + "\n")
    markdown = quantitative / "quick_summary.md"
    text = markdown.read_text()
    start, end = "<!-- spectral-peaks:start -->", "<!-- spectral-peaks:end -->"
    section = start + "\n" + summary_text(report, detailed=False) + end
    if start in text and end in text:
        before, rest = text.split(start, 1)
        text = before + section + rest.split(end, 1)[1]
    else:
        text += "\n" + section + "\n"
    markdown.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--point", nargs=2, type=int, metavar=("Y", "X"))
    parser.add_argument("--nperseg", type=int, default=8192)
    parser.add_argument("--band", nargs=2, type=float, default=(100., 1500.), metavar=("LOW", "HIGH"))
    parser.add_argument("--prominence-db", type=float, default=2.)
    args = parser.parse_args(argv)
    print(analyze(Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard),
                  point=args.point, nperseg=args.nperseg, band=args.band,
                  prominence_db=args.prominence_db, output=args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
