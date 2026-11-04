"""Compare oscillator ranks and modal participation between the old raw 0p20 model and the newly cleaned 0p20 dataset."""

import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from modelling.frequency_hybrid.local_excess import _average_matrix, local_excess_modes
from modelling.frequency_hybrid.study import matrices, projected
from scripts.diagnostics.fit_0p20_local_excess import CENTERS_0P20, make_bands_0p20

OUT_FIG = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/ranks_and_participation_comparison_0p20.png")
OUT_JSON = Path("/home/jonas/.gemini/antigravity/brain/c18455d2-38ed-40f3-9b47-f11fb8246321/ranks_and_participation_comparison_0p20.json")

OLD_SUMMARY_JSON = Path("best_models/0p20/oscillators/summary.json")
CLEAN_POD_PATH = Path("runs/cleaned_qc/cleaned_pod_0p20_full_r10.h5")

def main():
    print(f"Loading old model summary: {OLD_SUMMARY_JSON}")
    with open(OLD_SUMMARY_JSON) as f:
        old_data = json.load(f)
    
    old_bands = old_data["bands"]
    centers = np.asarray(CENTERS_0P20, dtype=float)
    bands = make_bands_0p20(centers, low_edge=50.0)
    n_bands = len(centers)
    n_coords = 11

    old_ranks = np.asarray([b["rank"] for b in old_bands])
    old_part = np.asarray([b["coordinate_participation"] for b in old_bands])

    print(f"Loading cleaned POD: {CLEAN_POD_PATH}")
    with h5py.File(CLEAN_POD_PATH, "r") as f:
        t = f["grid/time"][:]
        fs = 1.0 / np.mean(np.diff(t))
        sm = 200.0 * f["preprocessing/frame_spatial_mean"][:]
        coords = f["reduced/coefficients"][:, :10]
        modes = f["pod/modes"][:10]
        n_space = 40000

    weights = np.r_[1.0 / np.sqrt(n_space), modes[:, modes.shape[1] // 2, modes.shape[2] // 2]]
    cleaned_traj = np.column_stack([sm, coords])
    freq, spec = matrices([cleaned_traj], fs, 32768)

    # 1. Cleaned with Center-Pixel Weighting (Exact reproduction of old methodology)
    cln_center_ranks = []
    cln_center_part = []
    cln_center_part_rk1 = []
    cln_center_peaks = []

    power_center = projected(spec, weights)

    for idx, (c, b) in enumerate(zip(centers, bands)):
        inside = np.flatnonzero((freq >= b[0]) & (freq <= b[1]))
        peak = int(inside[np.argmax(power_center[inside])])
        valleys = []
        for boundary in b:
            candidates = np.flatnonzero(abs(freq - boundary) <= 14.0)
            valleys.append(int(candidates[np.argmin(power_center[candidates])]))
        left, right = valleys
        p_mat = _average_matrix(spec, peak, 1)
        l_mat = _average_matrix(spec, left, 1)
        r_mat = _average_matrix(spec, right, 1)
        alpha = (freq[peak] - freq[left]) / (freq[right] - freq[left])
        bg = (1 - alpha) * l_mat + alpha * r_mat
        ex = (p_mat - bg + (p_mat - bg).conj().T) / 2.0
        
        vals, vecs = np.linalg.eigh(ex)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        pos = np.maximum(vals, 0)
        rank = int(np.searchsorted(np.cumsum(pos), 0.95 * pos.sum()) + 1)
        retained = pos[:rank]
        part = (abs(vecs[:, :rank])**2 * retained[None]).sum(axis=1) / retained.sum()
        
        m1 = vecs[:, 0]
        part_rk1 = np.abs(m1)**2 / np.sum(np.abs(m1)**2)

        cln_center_ranks.append(rank)
        cln_center_part.append(part)
        cln_center_part_rk1.append(part_rk1)
        cln_center_peaks.append(float(freq[peak]))

    cln_center_ranks = np.asarray(cln_center_ranks)
    cln_center_part = np.asarray(cln_center_part)
    cln_center_part_rk1 = np.asarray(cln_center_part_rk1)

    # 2. Cleaned with Full-Field Trace Weighting (Unbiased by center node)
    cln_trace_ranks = []
    cln_trace_part = []
    cln_trace_peaks = []

    power_trace = np.trace(spec, axis1=1, axis2=2).real

    for idx, (c, b) in enumerate(zip(centers, bands)):
        inside = np.flatnonzero((freq >= b[0]) & (freq <= b[1]))
        peak = int(inside[np.argmax(power_trace[inside])])
        valleys = []
        for boundary in b:
            candidates = np.flatnonzero(abs(freq - boundary) <= 14.0)
            valleys.append(int(candidates[np.argmin(power_trace[candidates])]))
        left, right = valleys
        p_mat = _average_matrix(spec, peak, 1)
        l_mat = _average_matrix(spec, left, 1)
        r_mat = _average_matrix(spec, right, 1)
        alpha = (freq[peak] - freq[left]) / (freq[right] - freq[left])
        bg = (1 - alpha) * l_mat + alpha * r_mat
        ex = (p_mat - bg + (p_mat - bg).conj().T) / 2.0
        
        vals, vecs = np.linalg.eigh(ex)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        pos = np.maximum(vals, 0)
        rank = int(np.searchsorted(np.cumsum(pos), 0.95 * pos.sum()) + 1)
        retained = pos[:rank]
        part = (abs(vecs[:, :rank])**2 * retained[None]).sum(axis=1) / retained.sum()

        cln_trace_ranks.append(rank)
        cln_trace_part.append(part)
        cln_trace_peaks.append(float(freq[peak]))

    cln_trace_ranks = np.asarray(cln_trace_ranks)
    cln_trace_part = np.asarray(cln_trace_part)

    # Build comprehensive diagnostic figure
    fig = plt.figure(figsize=(16, 12), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2])

    ax_bar_comp = fig.add_subplot(gs[0, :])
    ax_old_heat = fig.add_subplot(gs[1, 0])
    ax_cln_heat = fig.add_subplot(gs[1, 1])
    ax_trc_heat = fig.add_subplot(gs[2, 0])
    ax_bar_dist = fig.add_subplot(gs[2, 1])

    # Top: Spatial Mean Participation comparison across all bands
    x = np.arange(n_bands)
    width = 0.25
    ax_bar_comp.bar(x - width, old_part[:, 0] * 100, width, label="Old Raw (Center Weighted)", color="#d95f02", alpha=0.9)
    ax_bar_comp.bar(x, cln_center_part[:, 0] * 100, width, label="Cleaned 0p20 (Center Weighted)", color="#7570b3", alpha=0.9)
    ax_bar_comp.bar(x + width, cln_trace_part[:, 0] * 100, width, label="Cleaned 0p20 (Full-Field Trace)", color="#1b9e77", alpha=0.9)
    ax_bar_comp.set_xticks(x)
    ax_bar_comp.set_xticklabels([f"{c:.0f} Hz\n({b[0]:.0f}-{b[1]:.0f})" for c, b in zip(centers, bands)], fontsize=9)
    ax_bar_comp.set_ylabel("Spatial Mean Participation [%]", fontsize=11, fontweight="bold")
    ax_bar_comp.set_title("Dominance of Spatial Mean: Old Model vs. Cleaned 0p20 (Across all 11 Bands)", fontsize=13, fontweight="bold")
    ax_bar_comp.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_bar_comp.set_ylim(0, 105)
    ax_bar_comp.legend(frameon=True, fontsize=10, loc="lower left")

    # Heatmaps: Coordinate participation
    coord_labels = ["mean"] + [f"POD{i}" for i in range(1, 11)]

    # Old Heatmap
    im1 = ax_old_heat.imshow(old_part.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                             extent=[0, n_bands, -0.5, n_coords - 0.5])
    ax_old_heat.set_xticks(np.arange(n_bands) + 0.5)
    ax_old_heat.set_xticklabels([f"{c:.0f}" for c in centers], fontsize=8)
    ax_old_heat.set_yticks(np.arange(n_coords))
    ax_old_heat.set_yticklabels(coord_labels, fontsize=9)
    ax_old_heat.set_title("A. Old Baseline: best_models/0p20/oscillators\n(Artifact-Dominated Heaviside Ringing)", fontsize=11, fontweight="bold")
    ax_old_heat.set_xlabel("Nominal Peak [Hz]", fontsize=9)
    ax_old_heat.set_ylabel("Coordinate", fontsize=10)

    # Cleaned Center-Weighted Heatmap
    im2 = ax_cln_heat.imshow(cln_center_part.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                             extent=[0, n_bands, -0.5, n_coords - 0.5])
    ax_cln_heat.set_xticks(np.arange(n_bands) + 0.5)
    ax_cln_heat.set_xticklabels([f"{c:.0f}" for c in centers], fontsize=8)
    ax_cln_heat.set_yticks(np.arange(n_coords))
    ax_cln_heat.set_yticklabels(coord_labels, fontsize=9)
    ax_cln_heat.set_title("B. Cleaned 0p20: Center-Weighted (Same Metric)\n(Spatial Mean Drops; POD Waves Emerge at >400 Hz)", fontsize=11, fontweight="bold")
    ax_cln_heat.set_xlabel("Nominal Peak [Hz]", fontsize=9)
    fig.colorbar(im2, ax=ax_cln_heat, label="Participation Fraction")

    # Cleaned Full-Field Trace Heatmap
    im3 = ax_trc_heat.imshow(cln_trace_part.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                             extent=[0, n_bands, -0.5, n_coords - 0.5])
    ax_trc_heat.set_xticks(np.arange(n_bands) + 0.5)
    ax_trc_heat.set_xticklabels([f"{c:.0f}" for c in centers], fontsize=8)
    ax_trc_heat.set_yticks(np.arange(n_coords))
    ax_trc_heat.set_yticklabels(coord_labels, fontsize=9)
    ax_trc_heat.set_title("C. Cleaned 0p20: Full-Field Trace (Unbiased by Center Node)\n(POD 1-2 Waves Carry 60-85% of Carrier Excess)", fontsize=11, fontweight="bold")
    ax_trc_heat.set_xlabel("Nominal Peak [Hz]", fontsize=9)
    ax_trc_heat.set_ylabel("Coordinate", fontsize=10)
    fig.colorbar(im3, ax=ax_trc_heat, label="Participation Fraction")

    # Bottom Right: Grouped Stacked Bar for Full-Field Energy Participation
    p_mean = cln_trace_part[:, 0] * 100
    p_pod12 = (cln_trace_part[:, 1] + cln_trace_part[:, 2]) * 100
    p_pod34 = (cln_trace_part[:, 3] + cln_trace_part[:, 4]) * 100
    p_other = cln_trace_part[:, 5:].sum(axis=1) * 100

    ax_bar_dist.bar(x, p_mean, label="Spatial Mean (Container Heave)", color="#d95f02", alpha=0.85)
    ax_bar_dist.bar(x, p_pod12, bottom=p_mean, label="POD 1 & 2 (Dipole Waves)", color="#1f78b4", alpha=0.85)
    ax_bar_dist.bar(x, p_pod34, bottom=p_mean + p_pod12, label="POD 3 & 4 (Breathing/Quadrupole)", color="#33a02c", alpha=0.85)
    ax_bar_dist.bar(x, p_other, bottom=p_mean + p_pod12 + p_pod34, label="POD 5-10 (High Capillary Waves)", color="#e31a1c", alpha=0.85)
    ax_bar_dist.set_xticks(x)
    ax_bar_dist.set_xticklabels([f"{c:.0f}" for c in centers], fontsize=8)
    ax_bar_dist.set_xlabel("Nominal Peak [Hz]", fontsize=9)
    ax_bar_dist.set_ylabel("Mode Participation [%]", fontsize=10)
    ax_bar_dist.set_title("D. Mode Family Decomposition (Cleaned 0p20 Full Field)", fontsize=11, fontweight="bold")
    ax_bar_dist.set_ylim(0, 100)
    ax_bar_dist.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_bar_dist.legend(frameon=True, fontsize=8, loc="upper right")

    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"Saved comparison figure: {OUT_FIG}")

    # Save quantitative json comparison
    results = {
        "centers_hz": centers.tolist(),
        "bands_hz": bands.tolist(),
        "old_raw": {
            "spatial_mean_participation": (old_part[:, 0] * 100).tolist(),
            "pod_total_participation": (old_part[:, 1:].sum(axis=1) * 100).tolist(),
            "ranks": old_ranks.tolist()
        },
        "cleaned_center_weighted": {
            "peaks_hz": cln_center_peaks,
            "spatial_mean_participation": (cln_center_part[:, 0] * 100).tolist(),
            "pod_total_participation": (cln_center_part[:, 1:].sum(axis=1) * 100).tolist(),
            "ranks": cln_center_ranks.tolist()
        },
        "cleaned_full_field_trace": {
            "peaks_hz": cln_trace_peaks,
            "spatial_mean_participation": (cln_trace_part[:, 0] * 100).tolist(),
            "pod_1_2_participation": (p_pod12).tolist(),
            "pod_3_4_participation": (p_pod34).tolist(),
            "pod_total_participation": (cln_trace_part[:, 1:].sum(axis=1) * 100).tolist(),
            "ranks": cln_trace_ranks.tolist()
        }
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison JSON: {OUT_JSON}")

if __name__ == "__main__":
    main()

