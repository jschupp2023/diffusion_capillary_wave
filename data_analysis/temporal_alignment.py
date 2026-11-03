"""Reusable, import-safe temporal cycle alignment module.

Implements the Advanced Kinematic Temporal Selector with local-motion veto:
1. Small events (|k| in {1, 2}):
   - Isolated in time (run length == 1 of non-zero candidates).
   - Quadrant consensus (at least 3 of 4 quadrants within 0.6*q).
   - Local step innovation check (|dh - v_pred - k*q| <= tol_small*q),
     which differentiates true isolated step slips from smooth threshold crossings.
2. Large events (|k| >= 3):
   - Directional consensus (at least 3 quadrants match sign of k).
   - Magnitude consensus (at least 3 quadrants have |dq| >= 2.0*q).
   - Local-motion veto: assesses whether surrounding motion supports smoothly
     continuing velocity. Vetoes correction ONLY when local context is demonstrably
     reliable (low acceleration) AND leaving the transition uncorrected produces
     a smaller error than applying k*q.
   - Bursts / ambiguous contexts: when surrounding motion is rapid or bursty,
     the fallback is to apply the consensus integer correction k*q to prevent
     leaving multi-cycle cliffs in the dataset, while recording an explicit
     uncertainty flag.

Candidate estimation note:
Candidate shifts k = round(dh / q) are estimated from the spatial-mean increment
dh = Delta h_mean(t), ensuring consistency with whole-domain elevation tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DEFAULT_Q = 1.9969696
DEFAULT_TAU = 4.0


@dataclass(frozen=True)
class TemporalSelectionResult:
    """Immutable record of temporal alignment decisions."""
    shifts: np.ndarray             # (N,) int64: frame-to-frame shifts delta_K(t)
    offsets: np.ndarray            # (N,) int64: accumulated offsets K(t)
    uncertainty_flags: np.ndarray  # (N,) bool: flags indicating burst/uncertain context
    n_frames: int
    n_corrections: int
    n_large_corrections: int
    n_small_corrections: int
    n_flagged_corrections: int
    max_residual_increment_um: float
    parameters: dict[str, float | int | str]


def select_temporal_kinematic(
    dh_mean: np.ndarray,
    quad_diffs: np.ndarray,
    q: float = DEFAULT_Q,
    tol_small: float = 0.45,
    large_cand_threshold: int = 3,
    quad_mag_threshold_cycles: float = 2.0,
    smooth_accel_threshold_cycles: float = 1.0,
    smooth_err_threshold_cycles: float = 0.6,
) -> TemporalSelectionResult:
    """Compute global integer cycle corrections K(t) from unaligned spatial traces.

    All computations are performed using float64 arithmetic.

    Parameters
    ----------
    dh_mean : np.ndarray
        (N,) float array of frame-to-frame spatial-mean increments: dh[t] = mean[t] - mean[t-1],
        with dh[0] = 0.
    quad_diffs : np.ndarray
        (N, 4) float array of frame-to-frame increments across the four spatial quadrants:
        top-left, top-right, bottom-left, bottom-right.
    q : float
        Optical cycle height in microns (default: 1.9969696 um).
    tol_small : float
        Innovation tolerance fraction for small (|k| in {1, 2}) events (default: 0.45).
    large_cand_threshold : int
        Minimum candidate integer magnitude classified as large (default: 3).
    quad_mag_threshold_cycles : float
        Minimum quadrant increment magnitude in cycles for large events (default: 2.0).
    smooth_accel_threshold_cycles : float
        Maximum local acceleration in cycles for reliable context (default: 1.0).
    smooth_err_threshold_cycles : float
        Maximum uncorrected error from trend in cycles for smooth veto (default: 0.6).

    Returns
    -------
    TemporalSelectionResult
        Result dataclass containing shifts, offsets, uncertainty flags, and metadata.
    """
    dh = np.asarray(dh_mean, dtype=np.float64)
    dq = np.asarray(quad_diffs, dtype=np.float64)
    n = len(dh)
    if dq.shape != (n, 4):
        raise ValueError(f"quad_diffs shape must be ({n}, 4); got {dq.shape}")

    # 1. Candidate integer shift from spatial-mean increment
    k = np.rint(dh / q).astype(np.int64)
    abs_k = np.abs(k)
    signs = np.sign(k)

    # 2. Local trend and acceleration with proper boundary extrapolation
    prev_dh = np.r_[dh[0], dh[:-1]]
    next_dh = np.r_[dh[1:], dh[-1]]
    if n > 1:
        # Prior velocity for frame 1 is estimated from frame 1
        prev_dh[1] = dh[1]
        # Next velocity for frame n-1 is estimated from frame n-1
        next_dh[-1] = dh[-1]

    v_pred = 0.5 * (prev_dh + next_dh)
    local_accel = np.abs(next_dh - prev_dh)

    dh_corr = dh - k * q
    err_uncorr = np.abs(dh - v_pred)
    err_corr = np.abs(dh_corr - v_pred)

    # 3. Large candidate evaluation (|k| >= large_cand_threshold)
    same_sign = (np.sign(dq) == signs[:, None]).sum(axis=1)
    mag_ok = (np.abs(dq) >= quad_mag_threshold_cycles * q).sum(axis=1)
    large_cand = abs_k >= large_cand_threshold
    large_consensus = large_cand & (same_sign >= 3) & (mag_ok >= 3)

    # Local-motion veto:
    # Requires reliable local context (low local acceleration)
    # AND uncorrected transition is close to trend
    # AND applying correction makes it significantly worse
    context_reliable = local_accel <= smooth_accel_threshold_cycles * q
    smooth_veto = (
        context_reliable
        & (err_uncorr <= smooth_err_threshold_cycles * q)
        & (err_corr > err_uncorr + 1.0 * q)
    )
    large_keep = large_consensus & (~smooth_veto)

    # Burst / uncertainty tracking:
    # Flag large events accepted within an active burst window
    burst_context = (
        (local_accel > smooth_accel_threshold_cycles * q)
        | (np.abs(np.rint(prev_dh / q)) >= large_cand_threshold)
        | (np.abs(np.rint(next_dh / q)) >= large_cand_threshold)
    )
    uncertainty_flag = large_keep & burst_context

    # 4. Small candidate evaluation (|k| in {1, 2})
    nonzero_mask = abs_k > 0
    all_runs = np.zeros(n, dtype=int)
    for t in np.flatnonzero(nonzero_mask):
        fwd, bwd = 0, 0
        while t + fwd + 1 < n and nonzero_mask[t + fwd + 1] and signs[t + fwd + 1] == signs[t]:
            fwd += 1
        while t - bwd - 1 >= 0 and nonzero_mask[t - bwd - 1] and signs[t - bwd - 1] == signs[t]:
            bwd += 1
        all_runs[t] = 1 + fwd + bwd

    step_innov = dh - v_pred
    quad_agree_small = (np.abs(dq - k[:, None] * q) <= 0.6 * q).sum(axis=1)
    innov_agree = np.abs(step_innov - k * q) <= tol_small * q
    small_cand = (abs_k >= 1) & (abs_k < large_cand_threshold)
    small_keep = small_cand & (all_runs == 1) & (quad_agree_small >= 3) & innov_agree

    # 5. Assemble final decisions
    shifts = np.where(large_keep | small_keep, k, 0).astype(np.int64)
    shifts[0] = 0
    uncertainty_flag[0] = False
    offsets = np.cumsum(shifts)

    # Residuals
    res_dh = dh - q * shifts
    max_res = float(np.max(np.abs(res_dh[1:]))) if n > 1 else 0.0

    params = {
        "method": "advanced_kinematic_veto_v1",
        "q_um": float(q),
        "tol_small": float(tol_small),
        "large_cand_threshold": int(large_cand_threshold),
        "quad_mag_threshold_cycles": float(quad_mag_threshold_cycles),
        "smooth_accel_threshold_cycles": float(smooth_accel_threshold_cycles),
        "smooth_err_threshold_cycles": float(smooth_err_threshold_cycles),
        "candidate_source": "spatial_mean_increment",
    }

    return TemporalSelectionResult(
        shifts=shifts,
        offsets=offsets,
        uncertainty_flags=uncertainty_flag,
        n_frames=n,
        n_corrections=int(np.count_nonzero(shifts)),
        n_large_corrections=int(np.count_nonzero(shifts[abs_k >= large_cand_threshold])),
        n_small_corrections=int(np.count_nonzero(shifts[(abs_k >= 1) & (abs_k < large_cand_threshold)])),
        n_flagged_corrections=int(np.count_nonzero(uncertainty_flag)),
        max_residual_increment_um=max_res,
        parameters=params,
    )


def extract_spatial_channels(frame: np.ndarray) -> np.ndarray:
    """Compute the 6 spatial tracking channels for a 2D frame.

    Channels:
      0: spatial mean
      1: center point elevation
      2: quadrant 0 (top-left mean)
      3: quadrant 1 (top-right mean)
      4: quadrant 2 (bottom-left mean)
      5: quadrant 3 (bottom-right mean)

    Returns
    -------
    np.ndarray
        (6,) float64 array of channel elevations.
    """
    a = np.asarray(frame, dtype=np.float64)
    ny, nx = a.shape
    cy, cx = ny // 2, nx // 2
    return np.array([
        a.mean(),
        a[cy, cx],
        a[:cy, :cx].mean(),
        a[:cy, cx:].mean(),
        a[cy:, :cx].mean(),
        a[cy:, cx:].mean(),
    ], dtype=np.float64)

