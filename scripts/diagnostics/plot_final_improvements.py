"""Generate comprehensive publication-quality comparison plots:
Reference vs Baseline Hybrid vs Final Improved Hybrid (Shaped Noise + 1232 Hz Oscillator).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfiltfilt
from data_analysis.energy.rollout_energy import capillary_stiffness, compute_energies, length_scale

def main():
    out_dir = Path('runs/diagnostics/final_improved_hybrid_v1')
    data_path = out_dir / 'augmented_hybrid_rollout.npz'
    if not data_path.exists():
        print(f'Waiting for {data_path}...')
        return

    with np.load(data_path) as z:
        ref = z['reference'] # [2, 34873, 11]
        aug = z['trajectories'][:, 0, :, :] # [2, 34873, 11]
        base_shaped = z['base_trajectories'][:, 0, :, :] # [2, 34873, 11]

    # Also load the original unshaped baseline hybrid rollout
    orig_rollout = 'runs/adaptive/neural_sde_0p10_local_excess_low77_psd_refinement_v1/trials/mean_variance_w0p03_e2/evaluation_scale1p0/hybrid_raw/rollout.npz'
    with np.load(orig_rollout) as z:
        orig = z['trajectories'][:, 0, :, :] # [2, 34873, 11]

    fs = 115200.0

    # 1. Plot Full-Spectrum PSD (Spatial Mean and POD Trace)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Spatial Mean PSD
    f_w, p_ref_m = signal.welch(ref[0, :, 0], fs=fs, nperseg=8192, noverlap=4096)
    p_ref_m = np.mean([signal.welch(ref[c, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_orig_m = np.mean([signal.welch(orig[c, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_shaped_m = np.mean([signal.welch(base_shaped[c, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)
    p_aug_m = np.mean([signal.welch(aug[c, :, 0], fs=fs, nperseg=8192, noverlap=4096)[1] for c in range(2)], axis=0)

    axes[0].loglog(f_w, p_ref_m, 'k-', lw=1.8, label='Reference (Ground Truth)', alpha=0.9)
    axes[0].loglog(f_w, p_orig_m, 'C3--', lw=1.2, label='Original Baseline Hybrid (Severe HF deficit)', alpha=0.7)
    axes[0].loglog(f_w, p_shaped_m, 'C0-.', lw=1.2, label='Shaped Noise Hybrid (HF restored)', alpha=0.8)
    axes[0].loglog(f_w, p_aug_m, 'C2-', lw=1.8, label='Final Improved Hybrid (+ 1232 Hz mode)', alpha=0.95)
    axes[0].axvspan(1000, 2500, color='gold', alpha=0.15, label='Transition Dip & Harmonics (1-2.5 kHz)')
    axes[0].axvspan(5000, 20000, color='cyan', alpha=0.12, label='High-Frequency Cascade (5-20 kHz)')
    axes[0].set_ylabel(r'PSD $[(\mu\mathrm{m})^2/\mathrm{Hz}]$')
    axes[0].set_title('Spatial Mean Coordinate ($\sqrt{N}\cdot\overline{\eta}$)')
    axes[0].grid(True, which='both', ls=':', alpha=0.4)
    axes[0].legend(loc='lower left', framealpha=0.9)
    axes[0].set_ylim(1e-4, 1e7)

    # POD Modes Trace PSD
    p_ref_pod = np.mean([sum(signal.welch(ref[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)
    p_orig_pod = np.mean([sum(signal.welch(orig[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)
    p_shaped_pod = np.mean([sum(signal.welch(base_shaped[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)
    p_aug_pod = np.mean([sum(signal.welch(aug[c, :, m], fs=fs, nperseg=8192, noverlap=4096)[1] for m in range(1, 11)) for c in range(2)], axis=0)

    axes[1].loglog(f_w, p_ref_pod, 'k-', lw=1.8, label='Reference (Ground Truth)', alpha=0.9)
    axes[1].loglog(f_w, p_orig_pod, 'C3--', lw=1.2, label='Original Baseline Hybrid', alpha=0.7)
    axes[1].loglog(f_w, p_shaped_pod, 'C0-.', lw=1.2, label='Shaped Noise Hybrid', alpha=0.8)
    axes[1].loglog(f_w, p_aug_pod, 'C2-', lw=1.8, label='Final Improved Hybrid (+ 1232 Hz mode)', alpha=0.95)
    axes[1].axvspan(1000, 2500, color='gold', alpha=0.15)
    axes[1].axvspan(5000, 20000, color='cyan', alpha=0.12)
    axes[1].set_ylabel(r'POD Trace PSD $[(\mu\mathrm{m})^2/\mathrm{Hz}]$')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_title(r'Summed POD Modes 1–10 Trace ($\sum_{m=1}^{10} a_m$)')
    axes[1].grid(True, which='both', ls=':', alpha=0.4)
    axes[1].legend(loc='lower left', framealpha=0.9)
    axes[1].set_xlim(50, 25000)
    axes[1].set_ylim(1e-4, 1e7)

    plt.tight_layout()
    psd_fig_path = out_dir / 'final_psd_comparison.png'
    fig.savefig(psd_fig_path, dpi=200)
    plt.close(fig)
    print(f'Saved PSD plot to {psd_fig_path}')

    # 2. Plot Capillary Deformation Energy Comparison (Raw vs 50 Hz HP)
    basis_path = '/home/jonas/ucsd_thesis/reduced_data/0p10/shared_pod/shared_pod_r1000.h5'
    import h5py
    with h5py.File(basis_path, 'r') as h5:
        modes = h5['pod']['modes'][:10]
        x_m = h5['grid']['x'][:] * length_scale('mm')
        y_m = h5['grid']['y'][:] * length_scale('mm')
    gamma = 0.0728
    stiffness, _ = capillary_stiffness(modes, x_m, y_m, gamma=gamma)
    opts = dict(exact=False, batch_size=256, stiffness=stiffness, include_spatial_mean=True)

    E_ref_raw = compute_energies(ref, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']
    E_orig_raw = compute_energies(orig, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']
    E_aug_raw = compute_energies(aug, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']

    sos50 = butter(4, 50.0, btype='highpass', fs=fs, output='sos')
    ref_hp = ref.copy(); ref_hp[..., 1:] = sosfiltfilt(sos50, ref_hp[..., 1:], axis=-2)
    orig_hp = orig.copy(); orig_hp[..., 1:] = sosfiltfilt(sos50, orig_hp[..., 1:], axis=-2)
    aug_hp = aug.copy(); aug_hp[..., 1:] = sosfiltfilt(sos50, aug_hp[..., 1:], axis=-2)

    E_ref_hp = compute_energies(ref_hp, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']
    E_orig_hp = compute_energies(orig_hp, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']
    E_aug_hp = compute_energies(aug_hp, modes, x_m, y_m, length_scale('microns'), gamma, **opts)['quadratic']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    categories = ['Raw Energy\n(All Frequencies)', '50 Hz High-Passed Energy\n(Physical Wave Cascade)']
    ref_vals = [np.mean(E_ref_raw) * 1e11, np.mean(E_ref_hp) * 1e11]
    orig_vals = [np.mean(E_orig_raw) * 1e11, np.mean(E_orig_hp) * 1e11]
    aug_vals = [np.mean(E_aug_raw) * 1e11, np.mean(E_aug_hp) * 1e11]

    x = np.arange(2)
    width = 0.25

    axes[0].bar(x - width, ref_vals, width, label='Reference', color='0.25')
    axes[0].bar(x, orig_vals, width, label='Original Baseline', color='C3')
    axes[0].bar(x + width, aug_vals, width, label='Final Improved', color='C2')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel(r'Mean Deformation Energy $[10^{-11}\ \mathrm{J}]$')
    axes[0].set_title('Capillary Deformation Energy')
    axes[0].legend()
    axes[0].grid(axis='y', ls=':', alpha=0.5)

    # Energy Ratios (Generated / Reference)
    orig_ratios = [orig_vals[0]/ref_vals[0] * 100, orig_vals[1]/ref_vals[1] * 100]
    aug_ratios = [aug_vals[0]/ref_vals[0] * 100, aug_vals[1]/ref_vals[1] * 100]

    axes[1].bar(x - width/2, orig_ratios, width, label='Original Baseline', color='C3')
    axes[1].bar(x + width/2, aug_ratios, width, label='Final Improved', color='C2')
    axes[1].axhline(100, color='k', ls='--', lw=1.5, label='100% Target')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel('Energy Match Percentage [%]')
    axes[1].set_title('Energy Ratio to Reference (%)')
    axes[1].set_ylim(50, 110)
    for i in range(2):
        axes[1].text(x[i] - width/2, orig_ratios[i] + 1.5, f'{orig_ratios[i]:.1f}%', ha='center', fontweight='bold', color='C3')
        axes[1].text(x[i] + width/2, aug_ratios[i] + 1.5, f'{aug_ratios[i]:.1f}%', ha='center', fontweight='bold', color='C2')
    axes[1].legend()
    axes[1].grid(axis='y', ls=':', alpha=0.5)

    plt.tight_layout()
    energy_fig_path = out_dir / 'final_energy_comparison.png'
    fig.savefig(energy_fig_path, dpi=200)
    plt.close(fig)
    print(f'Saved Energy plot to {energy_fig_path}')

if __name__ == '__main__':
    main()
