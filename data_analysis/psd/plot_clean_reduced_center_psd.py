"""Plot rank-limited center PSDs across the cleaned reduced-data conditions.

Each repetition is reconstructed independently at the geometric center.  Welch
PSDs are computed separately and only then aggregated, so repetition boundaries
are never concatenated.  The plotted line is the arithmetic mean PSD and the
shaded band is the 10th--90th percentile across repetitions.

The reconstruction supports both repository POD conventions:

* legacy PODs, where the instantaneous spatial mean was removed and is restored;
* new cleaned-data PODs, where the spatial mean is already represented by the
  POD modes and must not be added again.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.psd.pod_center_psd import (
    compute_welch,
    infer_sampling_frequency,
    reconstruct_point_signals,
)


CONDITION_PATTERN = re.compile(r"0p\d+")
REALIZATION_PATTERN = re.compile(r"(?:Ca_ac.*|rep\d+)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pod-rank", type=int, default=200)
    parser.add_argument("--reconstruction-rank", type=int, default=200)
    parser.add_argument("--nperseg", type=int, default=8_192)
    parser.add_argument("--overlap-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--min-frequency-hz", type=float, default=50.0)
    parser.add_argument("--max-frequency-hz", type=float)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _natural_key(text: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def _condition_value(name: str) -> float:
    return float(name.replace("p", "."))


def discover(root: Path, pod_rank: int) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for condition in sorted(root.iterdir(), key=lambda path: _natural_key(path.name)):
        if not condition.is_dir() or not CONDITION_PATTERN.fullmatch(condition.name):
            continue
        for realization in sorted(
            condition.iterdir(), key=lambda path: _natural_key(path.name)
        ):
            if (
                realization.is_dir()
                and REALIZATION_PATTERN.fullmatch(realization.name)
                and (realization / f"pod_2d_r{pod_rank}.h5").is_file()
            ):
                grouped[condition.name].append(
                    realization / f"pod_2d_r{pod_rank}.h5"
                )
    return dict(grouped)


def _preprocessing_kind(handle: h5py.File) -> str:
    source = str(handle.attrs.get("source_file", ""))
    if bool(handle.attrs.get("input_full_field_highpass_applied", False)):
        return "raw_full_field_hp50"
    if "cleaned" in source and "hp50" in source:
        return "cleaned_full_field_hp50"
    return "unverified"


def reconstruct_psd(
    path: Path,
    reconstruction_rank: int,
    nperseg: int,
    overlap_fraction: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with h5py.File(path, "r") as handle:
        stored_rank = int(handle.attrs.get("rank", len(handle["pod/modes"])))
        spatial_mean_removed = bool(
            handle.attrs.get("instantaneous_spatial_mean_removed", False)
        )
        temporal_centered = bool(
            handle.attrs.get("temporal_mean_field_removed", False)
        )
        retained_energy = float(
            handle["pod"].attrs.get("retained_energy_fraction", np.nan)
        )
        candidate_energy = float(
            handle["pod"].attrs.get("candidate_subspace_energy_fraction", np.nan)
        )
        orthogonality_error = float(
            handle["pod"].attrs.get("max_orthonormality_error", np.nan)
        )
        preprocessing_kind = _preprocessing_kind(handle)
    if reconstruction_rank > stored_rank:
        raise ValueError(
            f"Requested rank {reconstruction_rank}, but {path} stores {stored_rank}."
        )
    if not temporal_centered:
        raise ValueError(f"Temporal mean field was not removed in {path}.")
    if preprocessing_kind == "unverified":
        raise ValueError(f"Could not verify full-field high-pass provenance for {path}.")

    (
        time_values,
        _modal_signal,
        _spatial_mean,
        physical_signal,
        selected,
        used_rank,
        signal_units,
    ) = reconstruct_point_signals(
        path,
        reconstruction_rank,
        None,
        None,
        batch_size,
        show_progress=False,
    )
    sampling_hz, relative_jitter = infer_sampling_frequency(time_values)
    result, actual_nperseg, noverlap = compute_welch(
        physical_signal,
        sampling_hz,
        nperseg,
        overlap_fraction,
    )
    metadata = {
        "path": str(path.resolve()),
        "stored_rank": stored_rank,
        "reconstruction_rank": used_rank,
        "spatial_mean_removed_then_restored": spatial_mean_removed,
        "temporal_mean_field_removed": temporal_centered,
        "preprocessing_kind": preprocessing_kind,
        "sampling_hz": sampling_hz,
        "maximum_relative_timestep_variation": relative_jitter,
        "nperseg": actual_nperseg,
        "noverlap": noverlap,
        "center_y_index": selected.y_index,
        "center_x_index": selected.x_index,
        "signal_units": signal_units,
        "signal_rms": float(np.sqrt(np.mean(np.square(physical_signal)))),
        "retained_energy_fraction": retained_energy,
        "candidate_subspace_energy_fraction": candidate_energy,
        "max_orthonormality_error": orthogonality_error,
    }
    return result.frequency, result.density, metadata


def _plot(
    frequency: np.ndarray,
    condition_psds: dict[str, np.ndarray],
    output_path: Path,
    min_frequency_hz: float,
    max_frequency_hz: float | None,
    reconstruction_rank: int,
    nperseg: int,
    noverlap: int,
    units: str,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 6.3))
    ordered = sorted(condition_psds, key=_condition_value)
    colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(ordered)))
    for condition, color in zip(ordered, colors, strict=True):
        values = condition_psds[condition]
        mean = np.mean(values, axis=0)
        lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
        valid = (
            (frequency >= min_frequency_hz)
            & (frequency > 0)
            & np.isfinite(mean)
            & (mean > 0)
        )
        if max_frequency_hz is not None:
            valid &= frequency <= max_frequency_hz
        axis.loglog(
            frequency[valid],
            mean[valid],
            color=color,
            linewidth=1.8,
            label=f"{_condition_value(condition):.2f} Vpp",
        )
        axis.fill_between(
            frequency[valid],
            np.maximum(lower[valid], np.finfo(float).tiny),
            upper[valid],
            color=color,
            alpha=0.13,
            linewidth=0,
        )

    density_units = f"{units}$^2$/Hz" if units else "value$^2$/Hz"
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Power spectral density [{density_units}]")
    axis.set_title(
        f"Rank-{reconstruction_rank} center reconstruction: mean PSD across repetitions"
    )
    axis.grid(True, which="both", linestyle="--", alpha=0.26)
    axis.legend(title="Forcing", frameon=False, ncol=2)
    axis.text(
        0.02,
        0.025,
        f"Hann Welch: nperseg={nperseg:,}, overlap={noverlap:,}\n"
        "line: arithmetic mean; shading: 10th--90th percentile (16 repetitions)",
        transform=axis.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "0.78", "alpha": 0.9},
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(args: argparse.Namespace) -> list[Path]:
    root = args.root.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Reduced-data root does not exist: {root}")
    if args.pod_rank < 1 or not 1 <= args.reconstruction_rank <= args.pod_rank:
        raise ValueError("Require 1 <= reconstruction rank <= stored POD rank.")
    if args.nperseg < 2 or not 0 <= args.overlap_fraction < 1:
        raise ValueError("Invalid Welch configuration.")
    if args.batch_size < 1 or args.min_frequency_hz <= 0 or args.dpi < 1:
        raise ValueError("Batch size, minimum frequency, and dpi must be positive.")
    if args.max_frequency_hz is not None and args.max_frequency_hz <= args.min_frequency_hz:
        raise ValueError("Maximum frequency must exceed minimum frequency.")

    grouped = discover(root, args.pod_rank)
    if not grouped:
        raise FileNotFoundError(f"No rank-{args.pod_rank} POD files found below {root}.")
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path = output_directory / (
        f"center_psd_rank{args.reconstruction_rank}_all_conditions.png"
    )
    cache_path = output_directory / (
        f"center_psd_rank{args.reconstruction_rank}_all_conditions.npz"
    )
    summary_path = output_directory / (
        f"center_psd_rank{args.reconstruction_rank}_summary.json"
    )
    existing = [path for path in (figure_path, cache_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}; use --overwrite.")

    reference_frequency: np.ndarray | None = None
    condition_psds: dict[str, np.ndarray] = {}
    all_metadata: dict[str, list[dict[str, object]]] = {}
    units = ""
    actual_nperseg = actual_noverlap = None
    for condition in sorted(grouped, key=_condition_value):
        paths = grouped[condition]
        print(f"{condition}: reconstructing {len(paths)} repetitions", flush=True)
        densities: list[np.ndarray] = []
        metadata: list[dict[str, object]] = []
        for index, path in enumerate(paths, start=1):
            frequency, density, current = reconstruct_psd(
                path,
                args.reconstruction_rank,
                args.nperseg,
                args.overlap_fraction,
                args.batch_size,
            )
            if reference_frequency is None:
                reference_frequency = frequency
            elif not np.allclose(reference_frequency, frequency, rtol=0, atol=1e-10):
                raise ValueError(f"Frequency grid mismatch in {path}.")
            densities.append(density)
            metadata.append(current)
            units = units or str(current["signal_units"])
            actual_nperseg = int(current["nperseg"])
            actual_noverlap = int(current["noverlap"])
            print(f"  {index:2d}/{len(paths)} {path.parent.name}", flush=True)
        condition_psds[condition] = np.stack(densities)
        all_metadata[condition] = metadata

    assert reference_frequency is not None
    assert actual_nperseg is not None and actual_noverlap is not None
    _plot(
        reference_frequency,
        condition_psds,
        figure_path,
        args.min_frequency_hz,
        args.max_frequency_hz,
        args.reconstruction_rank,
        actual_nperseg,
        actual_noverlap,
        units,
        args.dpi,
    )
    np.savez_compressed(
        cache_path,
        frequency_hz=reference_frequency,
        conditions=np.asarray(sorted(condition_psds, key=_condition_value)),
        psd_by_condition=np.stack(
            [condition_psds[name] for name in sorted(condition_psds, key=_condition_value)]
        ),
        reconstruction_rank=np.int64(args.reconstruction_rank),
        nperseg=np.int64(actual_nperseg),
        noverlap=np.int64(actual_noverlap),
    )
    summary = {
        "root": str(root),
        "pod_rank": args.pod_rank,
        "reconstruction_rank": args.reconstruction_rank,
        "welch": {
            "window": "hann",
            "nperseg": actual_nperseg,
            "noverlap": actual_noverlap,
            "detrend": "constant",
            "scaling": "density",
        },
        "aggregation": (
            "Welch per repetition; arithmetic mean with 10th--90th percentile band"
        ),
        "conditions": all_metadata,
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_summary.replace(summary_path)
    print(f"Saved {figure_path}", flush=True)
    print(f"Saved {cache_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    return [figure_path, cache_path, summary_path]


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
