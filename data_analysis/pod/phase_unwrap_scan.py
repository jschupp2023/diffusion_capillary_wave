"""Scan a raw-height recording for spatial phase-unwrapping ambiguity.

This tool is diagnostic only.  It reads the original keyed-frame HDF5 file,
does not unwrap or modify any field, and writes timeline metrics separately.
The required ``q`` is the height represented by one 2*pi phase cycle.

Example
-------
python data_analysis/pod/phase_unwrap_scan.py \
    --raw-input /home/jonas/ucsd_thesis/11272025_c_0.2vpp_data_roi-none_cal-true.hdf5 \
    --start 0 --stop 100001 --stride 10 --q 2.0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .pod_2d import _frame_key, inspect_input
else:
    from pod_2d import _frame_key, inspect_input


METRIC_COLUMNS = (
    "edge_abs_p50",
    "edge_abs_p95",
    "edge_abs_p99",
    "edge_abs_p999",
    "edge_abs_max",
    "edge_fraction_above_q_over_2",
    "edge_fraction_above_q",
    "wrapped_residue_fraction",
)


def _wrap_phase(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def frame_ambiguity_metrics(field: np.ndarray, q: float) -> dict[str, float]:
    """Return edge-tail and wrapped-residue metrics for one finite field."""
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be a finite positive height period.")
    valid = np.isfinite(field)
    valid_x = valid[:, 1:] & valid[:, :-1]
    valid_y = valid[1:, :] & valid[:-1, :]
    edge_x = np.abs(np.diff(field, axis=1))[valid_x]
    edge_y = np.abs(np.diff(field, axis=0))[valid_y]
    edges = np.concatenate((edge_x, edge_y))
    if edges.size == 0:
        raise ValueError("A frame needs at least one valid adjacent pixel pair.")

    safe = np.where(valid, field, 0.0)
    phase = np.angle(np.exp(2j * np.pi * safe / q))
    gradient_x = _wrap_phase(np.diff(phase, axis=1))
    gradient_y = _wrap_phase(np.diff(phase, axis=0))
    circulation = (
        gradient_x[:-1, :]
        + gradient_y[:, 1:]
        - gradient_x[1:, :]
        - gradient_y[:, :-1]
    )
    valid_cell = (
        valid[:-1, :-1]
        & valid[:-1, 1:]
        & valid[1:, :-1]
        & valid[1:, 1:]
    )
    residue = np.abs(np.rint(circulation[valid_cell] / (2 * np.pi))) > 0
    percentiles = np.percentile(edges, (50, 95, 99, 99.9))
    return {
        "edge_abs_p50": float(percentiles[0]),
        "edge_abs_p95": float(percentiles[1]),
        "edge_abs_p99": float(percentiles[2]),
        "edge_abs_p999": float(percentiles[3]),
        "edge_abs_max": float(np.max(edges)),
        "edge_fraction_above_q_over_2": float(np.mean(edges > q / 2)),
        "edge_fraction_above_q": float(np.mean(edges > q)),
        "wrapped_residue_fraction": float(np.mean(residue)) if residue.size else np.nan,
    }


def _robust_high_threshold(values: np.ndarray, z: float) -> float:
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad
    return median + z * robust_sigma


def _group_flagged_positions(
    positions: np.ndarray, flagged: np.ndarray, stride: int
) -> list[list[int]]:
    selected = positions[flagged]
    if selected.size == 0:
        return []
    groups: list[list[int]] = [[int(selected[0])]]
    for position in selected[1:]:
        if int(position) - groups[-1][-1] <= stride:
            groups[-1].append(int(position))
        else:
            groups.append([int(position)])
    return groups


def scan_recording(
    raw_input: Path,
    start: int,
    stop: int | None,
    stride: int,
    q: float,
    *,
    output_dir: Path | None = None,
    robust_z: float = 6.0,
    dpi: int = 160,
    overwrite: bool = False,
) -> Path:
    if start < 0:
        raise ValueError("start must be nonnegative.")
    if stride < 1:
        raise ValueError("stride must be positive.")
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be a finite positive height period.")
    if not np.isfinite(robust_z) or robust_z <= 0:
        raise ValueError("robust_z must be finite and positive.")

    source = Path(raw_input).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        n_total = len(handle["meta/t"])
    actual_stop = n_total if stop is None else stop
    if actual_stop <= start or actual_stop > n_total:
        raise ValueError(f"stop must be in ({start}, {n_total}].")
    info = inspect_input(source, start, actual_stop)
    sample_local = np.arange(0, info.n_frames, stride, dtype=np.int64)
    positions = info.source_positions[sample_local]
    frame_numbers = info.frame_numbers[sample_local]

    if output_dir is None:
        q_label = f"{q:.9g}".replace(".", "p")
        output_dir = (
            Path("runs/phase_unwrap_scan")
            / source.stem
            / f"start_{start}_stop_{actual_stop}_stride_{stride}_q_{q_label}"
        )
    output = Path(output_dir).expanduser().resolve()
    filenames = ("metrics.csv", "summary.json", "timeline.png")
    existing = [output / name for name in filenames if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Scan output already exists ({existing[0]}); pass --overwrite to replace it."
        )

    records: list[dict[str, float | int]] = []
    with h5py.File(source, "r") as handle:
        main = handle["main"]
        for position, frame_number in zip(positions, frame_numbers):
            field = np.asarray(main[_frame_key(int(frame_number))], dtype=np.float64)
            values = frame_ambiguity_metrics(field, q)
            records.append(
                {
                    "source_position": int(position),
                    "hdf5_frame_number": int(frame_number),
                    **values,
                }
            )

    arrays = {
        name: np.asarray([record[name] for record in records], dtype=float)
        for name in METRIC_COLUMNS
    }
    threshold_metrics = (
        "edge_abs_p95",
        "edge_abs_p99",
        "edge_fraction_above_q_over_2",
        "wrapped_residue_fraction",
    )
    thresholds = {
        name: _robust_high_threshold(arrays[name], robust_z)
        for name in threshold_metrics
    }
    flags_by_metric = {
        name: arrays[name] > thresholds[name] for name in threshold_metrics
    }
    flagged = np.logical_or.reduce(tuple(flags_by_metric.values()))
    episodes = _group_flagged_positions(positions, flagged, stride)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]) + ("flagged",))
        writer.writeheader()
        for record, is_flagged in zip(records, flagged):
            writer.writerow({**record, "flagged": int(is_flagged)})

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, constrained_layout=True
    )
    panels = (
        ("edge_abs_p95", "p95 |adjacent height increment|"),
        ("edge_abs_p99", "p99 |adjacent height increment|"),
        ("wrapped_residue_fraction", "wrapped residue fraction"),
    )
    for axes_item, (name, label) in zip(axes, panels):
        axes_item.plot(positions, arrays[name], linewidth=0.8)
        axes_item.axhline(
            thresholds[name], color="tab:red", linestyle="--", label="robust threshold"
        )
        axes_item.scatter(
            positions[flagged], arrays[name][flagged],
            s=8, color="tab:red", alpha=0.6,
        )
        axes_item.set_ylabel(label)
        axes_item.grid(alpha=0.2)
        axes_item.legend(loc="upper right")
    axes[-1].set_xlabel("source timestep position")
    fig.suptitle(
        f"Phase-ambiguity scan: {source.name}; q={q:g}; every {stride} frame(s)"
    )
    fig.savefig(output / "timeline.png", dpi=dpi)
    plt.close(fig)

    top_by_metric: dict[str, list[dict[str, float | int]]] = {}
    for name in threshold_metrics:
        order = np.argsort(arrays[name])[-20:][::-1]
        top_by_metric[name] = [records[int(index)] for index in order]
    summary: dict[str, Any] = {
        "source": str(source),
        "source_opened_read_only": True,
        "start_inclusive": int(start),
        "stop_exclusive": int(actual_stop),
        "stride": int(stride),
        "sample_count": int(len(records)),
        "q": float(q),
        "q_units": info.units["z"] or "same units as stored heights",
        "robust_z": float(robust_z),
        "thresholds": thresholds,
        "flagged_sample_count": int(np.sum(flagged)),
        "flagged_sample_fraction": float(np.mean(flagged)),
        "flagged_episodes_sampled_positions": [
            {"start": group[0], "stop": group[-1], "sample_count": len(group)}
            for group in episodes
        ],
        "top_samples_by_metric": top_by_metric,
        "interpretation": (
            "High edge tails and residue density identify frames where branch selection is "
            "ambiguous; they detect risk, not which integer-cycle correction is physical. "
            "Integer-cycle artifacts disappear on rewrapping, so this scan cannot prove a "
            "correction from reconstructed heights alone. Robust thresholds are estimated "
            "from the selected scan range and therefore depend on that range."
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(
        f"Scanned {len(records)} frames ({start}:{actual_stop}:{stride}); "
        f"flagged {int(np.sum(flagged))}. Saved read-only diagnostics to {output}"
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--robust-z", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    scan_recording(
        args.raw_input,
        args.start,
        args.stop,
        args.stride,
        args.q,
        output_dir=args.output_dir,
        robust_z=args.robust_z,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
