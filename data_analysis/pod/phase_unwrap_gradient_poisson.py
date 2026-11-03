r"""Gradient-Domain Poisson Surface Restoration for Phase-Unwrapped Wave Data.

This module provides a fast, robust, and physically grounded method to eliminate
integer-cycle phase unwrapping discontinuities ($k \cdot q$, where $q \approx 2.0\,\mu\text{m}$)
from experimental capillary wave height fields.

Key Features
------------
1. Non-destructive gating:
   Frames whose maximum spatial gradient is below the physical threshold $\tau$
   are preserved verbatim (bit-for-bit unchanged).
2. Direct gradient-domain jump cancellation:
   Instead of rewrapping to phase (which creates thousands of aliased residue dipoles
   on steep physical wave crests), it removes integer jumps directly from the spatial
   derivative fields:
       $g_x = \Delta_x h - q \cdot \text{round}(\Delta_x h / q)$ for $|\Delta_x h| \ge \tau$.
3. Exact 2D Poisson integration via Discrete Cosine Transform (DCT-II):
   Integrates $\nabla^2 h = \nabla \cdot \mathbf{g}$ with Neumann boundary conditions
   in ~5 milliseconds per frame without stair-stepping or branch-cut tearing.
4. Robust temporal DC alignment:
   Maintains temporal continuity across contiguous burst episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

if __package__:
    from .pod_2d import _frame_key, inspect_input
else:
    from pod_2d import _frame_key, inspect_input


DEFAULT_Q = 1.9969696  # 2 * pi * 3.1782758e-7 m * 1e6 um/m
DEFAULT_TAU = 3.5      # Physical gradient threshold in microns (clears wave slopes <= 3.4 um)


@dataclass(frozen=True)
class RestorationResult:
    restored: np.ndarray
    was_modified: bool
    cycles_inferred: np.ndarray
    orig_max_dx: float
    orig_p99_dx: float
    orig_jump_count: int
    rest_max_dx: float
    rest_p99_dx: float
    rest_jump_count: int
    n_pixels_modified: int = 0
    pct_pixels_modified: float = 0.0


def restore_frame_poisson(
    frame: np.ndarray,
    q: float = DEFAULT_Q,
    tau: float = DEFAULT_TAU,
    method: str = "integer",
) -> RestorationResult:
    """Restore a single 2D height frame using gradient-domain jump cancellation.

    Parameters
    ----------
    frame : np.ndarray
        2D array of surface heights (in microns).
    q : float
        Height corresponding to one full 2*pi phase cycle (default ~1.997 um).
    tau : float
        Physical gradient jump threshold (default 3.5 um).
    method : str
        Restoration method: 'integer' (default, non-destructive plateau cycle
        subtraction leaving clean regions 100% bit-for-bit identical) or
        'poisson' (continuous Neumann Poisson surface integration).
    """
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {frame.shape}")
    if q <= 0 or not np.isfinite(q):
        raise ValueError("q must be positive and finite")
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be positive and finite")
    if method not in ("integer", "poisson"):
        raise ValueError(f"method must be 'integer' or 'poisson', got '{method}'")

    dx = np.diff(frame, axis=1)
    dy = np.diff(frame, axis=0)

    orig_max_dx = float(max(np.max(np.abs(dx)), np.max(np.abs(dy))))
    dxy = np.concatenate([np.abs(dx).ravel(), np.abs(dy).ravel()])
    orig_p99_dx = float(np.percentile(dxy, 99))
    jump_x = np.abs(dx) >= tau
    jump_y = np.abs(dy) >= tau
    orig_jump_count = int(np.sum(jump_x) + np.sum(jump_y))

    # Gating: if no edge exceeds threshold, preserve frame unchanged
    if orig_jump_count == 0:
        return RestorationResult(
            restored=frame.copy(),
            was_modified=False,
            cycles_inferred=np.zeros_like(frame, dtype=int),
            orig_max_dx=orig_max_dx,
            orig_p99_dx=orig_p99_dx,
            orig_jump_count=0,
            rest_max_dx=orig_max_dx,
            rest_p99_dx=orig_p99_dx,
            rest_jump_count=0,
            n_pixels_modified=0,
            pct_pixels_modified=0.0,
        )

    # Cancel integer jumps from gradient fields
    gx = dx.copy()
    gx[jump_x] -= q * np.round(dx[jump_x] / q)

    gy = dy.copy()
    gy[jump_y] -= q * np.round(dy[jump_y] / q)

    M, N = frame.shape
    rho = np.zeros((M, N), dtype=np.float64)
    rho[:, 0] = gx[:, 0]
    rho[:, 1:-1] = gx[:, 1:] - gx[:, :-1]
    rho[:, -1] = -gx[:, -1]
    rho[0, :] += gy[0, :]
    rho[1:-1, :] += gy[1:, :] - gy[:-1, :]
    rho[-1, :] += -gy[-1, :]

    # Invert Laplacian via 2D DCT-II (Neumann boundary conditions)
    rho_dct = fft.dctn(rho, type=2, norm="ortho")
    I, J = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    eigenvalues = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
    eigenvalues[0, 0] = 1.0  # avoid division by zero for DC
    phi_dct = rho_dct / eigenvalues
    phi_dct[0, 0] = 0.0
    h_poisson = fft.idctn(phi_dct, type=2, norm="ortho")

    # Align DC level with original clean pixels
    clean_pixel = np.ones((M, N), dtype=bool)
    clean_pixel[:, :-1] &= ~jump_x
    clean_pixel[:, 1:] &= ~jump_x
    clean_pixel[:-1, :] &= ~jump_y
    clean_pixel[1:, :] &= ~jump_y

    if np.sum(clean_pixel) >= 20:
        offset = float(np.median(frame[clean_pixel] - h_poisson[clean_pixel]))
    else:
        offset = float(np.median(frame - h_poisson))
    h_poisson += offset

    if method == "poisson":
        restored = h_poisson.astype(frame.dtype)
        cycles = np.round((restored - frame) / q).astype(int)
        n_mod = int(np.sum(restored != frame))
    elif method == "integer":
        cycles = np.round((frame - h_poisson) / q).astype(int)
        n_mod = int(np.sum(cycles != 0))
        if n_mod == 0:
            return RestorationResult(
                restored=frame.copy(),
                was_modified=False,
                cycles_inferred=cycles,
                orig_max_dx=orig_max_dx,
                orig_p99_dx=orig_p99_dx,
                orig_jump_count=orig_jump_count,
                rest_max_dx=orig_max_dx,
                rest_p99_dx=orig_p99_dx,
                rest_jump_count=orig_jump_count,
                n_pixels_modified=0,
                pct_pixels_modified=0.0,
            )
        restored = (frame - cycles * q).astype(frame.dtype)

    # Statistics of restored field
    dx_rest = np.diff(restored, axis=1)
    dy_rest = np.diff(restored, axis=0)
    rest_max_dx = float(max(np.max(np.abs(dx_rest)), np.max(np.abs(dy_rest))))
    dxy_rest = np.concatenate([np.abs(dx_rest).ravel(), np.abs(dy_rest).ravel()])
    rest_p99_dx = float(np.percentile(dxy_rest, 99))
    rest_jump_count = int(np.sum(np.abs(dx_rest) >= tau) + np.sum(np.abs(dy_rest) >= tau))
    pct_mod = float(n_mod / (M * N) * 100.0)

    return RestorationResult(
        restored=restored,
        was_modified=True,
        cycles_inferred=cycles,
        orig_max_dx=orig_max_dx,
        orig_p99_dx=orig_p99_dx,
        orig_jump_count=orig_jump_count,
        rest_max_dx=rest_max_dx,
        rest_p99_dx=rest_p99_dx,
        rest_jump_count=rest_jump_count,
        n_pixels_modified=n_mod,
        pct_pixels_modified=pct_mod,
    )


# Backwards compatibility and intuitive aliases
restore_frame = restore_frame_poisson
restore_frame_integer = lambda frame, q=DEFAULT_Q, tau=DEFAULT_TAU: restore_frame_poisson(
    frame, q=q, tau=tau, method="integer"
)


def restore_window(
    frames: np.ndarray,
    q: float = DEFAULT_Q,
    tau: float = DEFAULT_TAU,
    temporally_align: bool = True,
    method: str = "integer",
) -> tuple[np.ndarray, list[RestorationResult]]:
    """Restore a sequence of frames with optional temporal DC continuity alignment."""
    results = [restore_frame_poisson(f, q=q, tau=tau, method=method) for f in frames]
    restored_stack = np.stack([r.restored for r in results], axis=0)

    if temporally_align and len(frames) > 1:
        # Align DC across consecutive frames
        for t in range(1, len(restored_stack)):
            if results[t].was_modified:
                prev = restored_stack[t - 1]
                curr = restored_stack[t]
                # align median height difference to expected physical transition
                dc_diff = np.median(prev - curr)
                if np.abs(dc_diff) > q / 2:
                    k_align = np.round(dc_diff / q) * q
                    restored_stack[t] += k_align

    return restored_stack, results


def plot_comparison(
    orig: np.ndarray,
    res: RestorationResult,
    timestep: int,
    output_path: Path,
    q: float = DEFAULT_Q,
    dpi: int = 160,
) -> None:
    """Render a comprehensive diagnostic comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    # 1. Original surface
    im0 = axes[0, 0].imshow(orig, origin="lower", cmap="viridis")
    axes[0, 0].set_title(f"Original Raw Height (t={timestep})")
    fig.colorbar(im0, ax=axes[0, 0], label="Height (um)")

    # 2. Restored surface
    im1 = axes[0, 1].imshow(res.restored, origin="lower", cmap="viridis")
    axes[0, 1].set_title("Restored Surface (Poisson Gradient-Domain)")
    fig.colorbar(im1, ax=axes[0, 1], label="Height (um)")

    # 3. Inferred integer cycle map
    im2 = axes[0, 2].imshow(res.cycles_inferred, origin="lower", cmap="coolwarm")
    axes[0, 2].set_title("Inferred Cycle Shifts k")
    fig.colorbar(im2, ax=axes[0, 2], label="Cycles (step = q)")

    # 4. Gradient comparison
    dx_orig = np.diff(orig, axis=1)[:-1, :]
    dy_orig = np.diff(orig, axis=0)[:, :-1]
    grad_orig = np.maximum(np.abs(dx_orig), np.abs(dy_orig))

    dx_rest = np.diff(res.restored, axis=1)[:-1, :]
    dy_rest = np.diff(res.restored, axis=0)[:, :-1]
    grad_rest = np.maximum(np.abs(dx_rest), np.abs(dy_rest))

    im3 = axes[1, 0].imshow(grad_orig, origin="lower", cmap="hot", vmax=5.0)
    axes[1, 0].set_title(f"Original |grad| (max={res.orig_max_dx:.1f} um)")
    fig.colorbar(im3, ax=axes[1, 0], label="Gradient (um)")

    im4 = axes[1, 1].imshow(grad_rest, origin="lower", cmap="hot", vmax=5.0)
    axes[1, 1].set_title(f"Restored |grad| (max={res.rest_max_dx:.1f} um)")
    fig.colorbar(im4, ax=axes[1, 1], label="Gradient (um)")

    # 5. Row profile
    row = orig.shape[0] // 2
    axes[1, 2].plot(orig[row], label="Original", color="crimson", alpha=0.7)
    axes[1, 2].plot(res.restored[row], label="Restored", color="royalblue", linewidth=1.5)
    axes[1, 2].set_title(f"Center Row {row} Profile")
    axes[1, 2].set_xlabel("x pixel")
    axes[1, 2].set_ylabel("Height (um)")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, required=True, help="Input HDF5 path")
    parser.add_argument("--timestep", type=int, default=1553, help="Target timestep to diagnose")
    parser.add_argument("--window", type=int, default=1, help="Number of frames around target")
    parser.add_argument("--q", type=float, default=DEFAULT_Q, help="Height period q in microns")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Jump threshold in microns")
    parser.add_argument(
        "--method",
        type=str,
        default="integer",
        choices=["integer", "poisson"],
        help="Restoration method ('integer' or 'poisson')",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase_unwrap_poisson"))
    args = parser.parse_args()

    source = args.raw_input.expanduser().resolve()
    start = max(0, args.timestep - args.window)
    stop = args.timestep + args.window + 1
    info = inspect_input(source, start, stop)

    frames = []
    with h5py.File(source, "r") as h5:
        main_grp = h5["main"]
        for fn in info.frame_numbers:
            frames.append(np.asarray(main_grp[_frame_key(int(fn))], dtype=np.float32))
    frames = np.stack(frames, axis=0)

    restored_stack, results = restore_window(
        frames, q=args.q, tau=args.tau, method=args.method
    )

    target_idx = args.timestep - start
    output_dir = args.output_dir / f"timestep_{args.timestep}"
    plot_comparison(
        frames[target_idx],
        results[target_idx],
        args.timestep,
        output_dir / "restoration_comparison.png",
        q=args.q,
    )

    summary = {
        "source": str(source),
        "timestep": args.timestep,
        "q": args.q,
        "tau": args.tau,
        "was_modified": results[target_idx].was_modified,
        "orig_max_dx": results[target_idx].orig_max_dx,
        "rest_max_dx": results[target_idx].rest_max_dx,
        "orig_jump_count": results[target_idx].orig_jump_count,
        "rest_jump_count": results[target_idx].rest_jump_count,
        "cycle_min": int(results[target_idx].cycles_inferred.min()),
        "cycle_max": int(results[target_idx].cycles_inferred.max()),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Processed timestep {args.timestep}: modified={results[target_idx].was_modified}")
    print(f"  Max dx: {results[target_idx].orig_max_dx:.2f} -> {results[target_idx].rest_max_dx:.2f}")
    print(f"  Jump count: {results[target_idx].orig_jump_count} -> {results[target_idx].rest_jump_count}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
