"""Make plots and quantitative metrics from a saved SDE or EDM rollout.

    python -m data_analysis.plot_rollout runs/0p20_r50_neural_sde/evaluation

The input directory must contain rollout.npz and metrics.json. Plots are saved
in its energy/, modal/, psd/, psd_peaks/, coordinates/, and quantitative/ subdirectories.
Pass --surface-video to additionally create a short side-by-side reference and
generated shared-POD surface video.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.energy.rollout_energy import analyze as plot_energy
from data_analysis.energy.rollout_modal import analyze as plot_modal
from data_analysis.pod.pod_surface_video import make_surface_comparison_video
from data_analysis.psd.rollout_modal_psd import analyze as plot_modal_psd
from data_analysis.psd.rollout_psd import analyze as plot_psd
from data_analysis.psd.rollout_spectral_peaks import (
    analyze as plot_spectral_peaks, include_in_quantitative_summary,
    validate_settings as validate_peak_settings,
)
from data_analysis.rollout import Rollout, add_rollout_arguments, output_directory
from data_analysis.rollout_metrics import analyze as quantitative_metrics


def plot_coordinates(rollout, *, condition=0, ensemble=0, modes=None,
                     output=None, overwrite=False):
    if not 0 <= condition < len(rollout.reference):
        raise ValueError("Condition index is outside the rollout.")
    if not 0 <= ensemble < rollout.generated.shape[1]:
        raise ValueError("Ensemble index is outside the rollout.")
    modes = modes if modes is not None else [m for m in (1, 2, 5, 10, 20, 50) if m <= rollout.rank]
    if not modes or any(m < 1 or m > rollout.rank for m in modes):
        raise ValueError(f"POD modes must lie between 1 and {rollout.rank}.")
    indices = [m - 1 + rollout.pod_coordinate_offset for m in modes]
    mean_label = ("√N × spatial mean" if rollout.include_spatial_mean
                  else "spatial mean (derived from POD modes)")
    if rollout.spatial_mean_highpass_hz is not None:
        mean_label += f" (HP {rollout.spatial_mean_highpass_hz:g} Hz)"
    series = []
    if rollout.spatial_mean_available:
        mean_reference = (rollout.reference[condition, :, 0]
                          if rollout.include_spatial_mean else
                          rollout.spatial_mean(rollout.reference[condition]))
        mean_generated = (rollout.generated[condition, ensemble, :, 0]
                          if rollout.include_spatial_mean else
                          rollout.spatial_mean(rollout.generated[condition, ensemble]))
        series.append((mean_reference, mean_generated,
                       mean_label))
    series.extend((rollout.reference[condition, :, index],
                   rollout.generated[condition, ensemble, :, index], f"POD {mode}")
                  for index, mode in zip(indices, modes, strict=True))
    time = np.arange(rollout.generated.shape[2]) / rollout.fs
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.1 * len(series)),
                             sharex=True, squeeze=False, constrained_layout=True)
    for row, (axis, (reference, generated, label)) in enumerate(zip(axes[:, 0], series, strict=True)):
        axis.plot(time, reference, color="0.2", lw=.8,
                  label="Reference" if row == 0 else None)
        axis.plot(time, generated, color="C3", lw=.8,
                  label="Generated" if row == 0 else None)
        axis.set_ylabel(f"{label}\n[{rollout.units}]")
        axis.grid(alpha=.2)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Time since selected window start [s]")
    fig.suptitle(f"{rollout.repetitions[condition]} · condition {condition}, ensemble {ensemble}")
    out = output_directory(rollout, "coordinates", output, overwrite)
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    return out


def plot_surface_video(rollout, *, condition=0, ensemble=0, steps=500,
                       spatial_points=30, fps=30, output=None, overwrite=False):
    """Render matched rollout/reference fluctuation surfaces with shared limits."""
    if not 0 <= condition < len(rollout.reference):
        raise ValueError("Condition index is outside the rollout.")
    if not 0 <= ensemble < rollout.generated.shape[1]:
        raise ValueError("Ensemble index is outside the rollout.")
    if steps < 1 or spatial_points < 3 or fps < 1:
        raise ValueError("Video steps/fps must be positive and spatial-points at least 3.")
    count = min(steps, rollout.generated.shape[2])
    ny, nx = rollout.modes.shape[1:]
    yidx = np.unique(np.linspace(0, ny - 1, min(spatial_points, ny), dtype=int))
    xidx = np.unique(np.linspace(0, nx - 1, min(spatial_points, nx), dtype=int))
    modes = np.asarray(rollout.modes[:, yidx][:, :, xidx], dtype=np.float32)
    def reconstruct(coordinates):
        values = np.asarray(coordinates[:count], dtype=np.float32)
        return rollout.reconstruct(values, modes=modes)

    reference = reconstruct(rollout.reference[condition])
    generated = reconstruct(rollout.generated[condition, ensemble])
    times = rollout.time[condition, :count] - rollout.time[condition, 0]
    out = output_directory(rollout, "surface_video", output, overwrite)
    return make_surface_comparison_video(
        reference, generated, rollout.x[xidx], rollout.y[yidx], times,
        output=out / "reference_vs_generated.mp4",
        x_unit=rollout.x_units, y_unit=rollout.y_units, z_unit=rollout.units,
        fps=fps,
        title=(f"{rollout.repetitions[condition]} | condition {condition}, "
               f"ensemble {ensemble} | rank {rollout.rank} fluctuations"),
        overwrite=overwrite,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_rollout_arguments(parser)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--ensemble", type=int, default=0)
    parser.add_argument("--modes", nargs="+", type=int, help="POD mode numbers to plot.")
    parser.add_argument("--point", nargs=2, type=int, metavar=("Y", "X"), help="PSD pixel; default grid center.")
    parser.add_argument("--nperseg", type=int, default=1024, help="Maximum Welch segment length.")
    parser.add_argument("--peak-nperseg", type=int, default=8192,
                        help="Independent Welch length for the spectral-peak diagnostic (default: 8192).")
    parser.add_argument("--peak-band", nargs=2, type=float, default=(100., 1500.),
                        metavar=("LOW", "HIGH"), help="Spectral-peak frequency band in Hz (default: 100 1500).")
    parser.add_argument("--peak-prominence-db", type=float, default=2.,
                        help="Minimum reference peak prominence in dB (default: 2).")
    parser.add_argument("--psd-lower-frequency", type=float, default=10.,
                        help="Requested lower frequency for log-band PSD metrics [Hz] (default: 10).")
    parser.add_argument("--bins-per-decade", type=int, choices=(1, 2), default=2,
                        help="Whole- or half-decade PSD metric bands (default: 2).")
    parser.add_argument("--modal-psd", action="store_true",
                        help="Also compare generated/reference Welch PSDs for individual POD modes.")
    parser.add_argument("--modal-psd-modes", nargs="+", type=int,
                        help="POD modes for --modal-psd; providing this option enables the diagnostic.")
    parser.add_argument("--exact", action="store_true", help="Also compute exact geometric capillary energy.")
    parser.add_argument("--surface-video", action="store_true",
                        help="Also render a side-by-side reference/generated surface MP4.")
    parser.add_argument("--video-steps", type=int, default=500,
                        help="Initial rollout timesteps represented in the optional video (default: 500).")
    parser.add_argument("--video-spatial-points", type=int, default=30,
                        help="Approximate grid points per spatial direction in the video (default: 30).")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="Optional surface-video frame rate (default: 30).")
    args = parser.parse_args(argv)
    rollout = Rollout(args.rollout, shared_basis=args.shared_basis, discard=args.discard)
    root = Path(args.output) if args.output else rollout.path.parent
    names = ["energy", "modal", "psd", "psd_peaks", "coordinates", "quantitative"]
    modal_psd_requested = args.modal_psd or args.modal_psd_modes is not None
    if modal_psd_requested:
        names.append("modal_psd")
    if args.surface_video:
        names.append("surface_video")
    outputs = {name: root / name for name in names}
    if not args.overwrite:
        for path in outputs.values():
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"Nonempty output directory {path}; use --overwrite.")
    # Check coordinate selection before running the more expensive analyses.
    if not 0 <= args.condition < len(rollout.reference) or not 0 <= args.ensemble < rollout.generated.shape[1]:
        raise ValueError("Condition or ensemble index is outside the rollout.")
    if args.modes and any(m < 1 or m > rollout.rank for m in args.modes):
        raise ValueError(f"POD modes must lie between 1 and {rollout.rank}.")
    if (args.modal_psd_modes
            and (any(m < 1 or m > rollout.rank for m in args.modal_psd_modes)
                 or len(set(args.modal_psd_modes)) != len(args.modal_psd_modes))):
        raise ValueError(
            f"Modal PSD modes must be unique and lie between 1 and {rollout.rank}.")
    if args.nperseg < 4:
        raise ValueError("nperseg must be at least 4.")
    validate_peak_settings(args.peak_nperseg, args.peak_band, args.peak_prominence_db, 40.)
    if args.video_steps < 1 or args.video_spatial_points < 3 or args.video_fps < 1:
        raise ValueError("Video steps/fps must be positive and spatial-points at least 3.")
    if args.psd_lower_frequency <= 0:
        raise ValueError("PSD lower frequency must be positive.")
    if args.point is not None and not (0 <= args.point[0] < len(rollout.y)
                                       and 0 <= args.point[1] < len(rollout.x)):
        raise ValueError("PSD point is outside the spatial grid.")
    print(plot_energy(rollout, exact=args.exact, output=outputs["energy"], overwrite=args.overwrite))
    print(plot_modal(rollout, output=outputs["modal"], overwrite=args.overwrite))
    print(plot_psd(rollout, point=args.point, nperseg=args.nperseg,
                   output=outputs["psd"], overwrite=args.overwrite))
    print(plot_spectral_peaks(
        rollout, point=args.point, nperseg=args.peak_nperseg,
        band=args.peak_band, prominence_db=args.peak_prominence_db,
        output=outputs["psd_peaks"], overwrite=args.overwrite))
    if modal_psd_requested:
        print(plot_modal_psd(
            rollout, modes=args.modal_psd_modes, nperseg=args.nperseg,
            lower_frequency=args.psd_lower_frequency,
            bins_per_decade=args.bins_per_decade,
            output=outputs["modal_psd"], overwrite=args.overwrite))
    print(quantitative_metrics(
        rollout, point=args.point, nperseg=args.nperseg,
        psd_lower_frequency=args.psd_lower_frequency,
        bins_per_decade=args.bins_per_decade, output=outputs["quantitative"],
        overwrite=args.overwrite))
    include_in_quantitative_summary(outputs["quantitative"], outputs["psd_peaks"])
    print(plot_coordinates(rollout, condition=args.condition, ensemble=args.ensemble,
                           modes=args.modes, output=outputs["coordinates"], overwrite=args.overwrite))
    if args.surface_video:
        print(plot_surface_video(
            rollout, condition=args.condition, ensemble=args.ensemble,
            steps=args.video_steps, spatial_points=args.video_spatial_points,
            fps=args.video_fps, output=outputs["surface_video"],
            overwrite=args.overwrite))


if __name__ == "__main__":
    main()
