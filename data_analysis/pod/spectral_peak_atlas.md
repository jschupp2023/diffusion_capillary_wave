# Experimental spectral peak atlas

`spectral_peak_atlas.py` identifies coherent spatial patterns at measured
spectral peaks and nearby valley controls. It analyzes experimental shared-POD
trajectories; it does not use oscillator predictions, fit a frequency power law,
train a neural model, or modify any HDF5 file.

From the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/capillarywave/bin/python \
  -m data_analysis.pod.spectral_peak_atlas \
  --experiment 0p10 --rank 50 --compare-ranks 20 100 \
  --output runs/diagnostics/spectral_peak_atlas_0p10_v1
```

Choose a new output directory for each run. Existing results are never
overwritten. The defaults are intended for the existing 0p10 protocol:
training reps 1–12, validation 13–14, excluded test 15–16, stored 50 Hz high-pass
spatial mean, anti-alias downsampling by eight, and 32 prepared samples removed
at each recording edge. The shared basis itself was previously fitted using
all 16 repetitions. Held-out spectral statistics therefore test transfer
within that fixed spatial representation, not a wholly held-out basis.

## Spectral and spatial conventions

- Physical coefficients are not standardized. Coordinate zero is `sqrt(N)`
  times the stored high-pass spatial mean; its spatial mode is `1/sqrt(N)`.
  Other coordinates describe deformation with their original preprocessing.
- Within each recording, use periodic Hann windows, constant detrending,
  50% overlap, no zero padding, and one-sided spectral density. Default
  `nperseg=4096` is about 284 ms at 14.4 kHz. The alternate window is 2048.
- Form `S_ij = mean(X_i * conj(X_j))`. Average recording-level matrices with
  equal recording weights. No window crosses a recording boundary.
- Select peaks in the pooled **training** spatial-mean and POD 1 PSDs between
  100 and 1100 Hz, using at least 2 dB prominence and 40 Hz within-channel
  spacing. Select candidates nearest evenly spaced low/middle/high frequency
  targets in each channel. Coincident selected bins are merged. Channels are
  selection labels, not identified physical frequency families.
- Valley controls are actual local minima in the summed deformation PSD,
  between adjacent candidate peaks, at least `max(2*df, half_width)` away from
  those peaks. Select controls across the available frequency range.
- Integrate each spectral matrix over a narrow band with default half-width
  10 Hz, capped at 40% of the distance to the nearest other detected candidate.
  Integration is `sum(S(f_bin))*df`; exact included bins are saved. At a
  different Welch resolution, the nearest bin is used if the interval contains
  no bin. Consequently sensitivity comparisons can have different effective
  band widths; these are explicitly recorded.
- Diagonalize the deformation CSD in the physical orthonormal POD basis. For
  `S v = lambda v`, the surface mode is `phi @ v`, **not** `phi @ conj(v)`.
  The existing Hilbert band-mode script uses a different covariance convention;
  its conjugation must not be copied into this calculation.
- For the mean-inclusive companion, measure `G = H^H H`, diagonalize
  `G^(1/2) S G^(1/2)`, and map eigenvectors back with `G^(-1/2)`. This handles
  the small nonorthogonality of the stored POD and constant mode. Similarities
  use the corresponding physical inner product.

Eigenvalues are integrated displacement variance under the pixel-sum norm,
not capillary energy. Dominant-mode fractions do not measure oscillator count
directly. Different frequencies need not have orthogonal spatial modes.

## Outputs and reproducibility checks

Open `README.md` in the output directory. It links to:

- Peak/valley spectra, training mode fractions, and held-out capture by the
  **training** modes, without validation refitting of those modes.
- Two spatial modes per band: amplitude, amplitude-masked phase, phase snapshots,
  and separate mean-inclusive patterns. Peak folders also contain GIF phase
  cycles. These unit-norm animations illustrate coherent geometry, not measured
  realizations or relative peak strength.
- Phase-invariant mode similarity (MAC) and mean squared principal-angle
  cosines between the leading two-dimensional subspaces across bands.
- Repetition reproducibility: training repetitions compared to fits excluding
  that repetition, and validation repetitions compared to the pooled training
  fit. Independent validation eigensystems are diagnostic only.
- Repetition bootstrap quantiles for the leading variance fraction and mode/
  subspace similarity; windows are never resampled as independent recordings.
- POD-rank sensitivity, alternate-Welch-window sensitivity, and center/neighbor
  bin comparisons. Nearly degenerate eigenvalues require subspace comparisons;
  two-mode subspaces can themselves be ambiguous if the second and third
  eigenvalues are close.

`spectral_matrices.npz` saves recording-level integrated CSD matrices and adjacent
single-bin matrices, and `spectral_modes.npz` saves eigenvectors, eigenvalues,
surface maps, spatial coordinates, and similarity matrices. `summary.json`
contains all per-band metrics and candidate-peak metadata. `protocol.json`
records source paths/file sizes/mtimes, source-code hash, splits, spectral
settings, exact frequency bins, and the preparation convention.

Only selected frequency matrices are accumulated, so memory does not scale as
the full frequency grid times the square of the maximum rank. Prepared
train/validation trajectories are cached up to `--cache-gib` (default 2 GiB);
this budget excludes the additional spectral matrices and plotting allocations.
Projection uses existing blockwise source preparation. Test trajectories are
not projected or analyzed.

Useful options: `--peaks-per-channel`, `--valleys`, `--prominence-db`,
`--spacing-hz`, `--band-half-width-hz`, `--bootstrap`, and `--no-animations`.
Use `--help` for the full interface.

To refresh figures and the report from saved numerical results, use
`--render-only --output EXISTING_ATLAS`. Adding `--no-animations` leaves existing
GIFs in place. This mode does not recompute any spectral statistics or reopen
source HDF5 files. A separate `rendering.json` records the rendering code hash;
the original analysis provenance is preserved.

## Verification

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python -m pytest \
  test/test_spectral_peak_atlas.py test/test_complex_band_modes.py -q
```

Tests check the Welch normalization, complex traveling-wave phase convention,
pixel-space reconstruction, nonorthogonal Gram correction, phase/subspace
invariance, selection rules, equal-repetition weighting, and isolation of
training eigensystems from validation statistics.
