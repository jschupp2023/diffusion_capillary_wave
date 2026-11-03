# Square observed-state oscillator feasibility test

Run from the repository root with the existing experimental spectral cache:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
timeout --signal=TERM --kill-after=5s 300s \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.latent_oscillators.observed_spectral_study \
  --output runs/diagnostics/observed_oscillator_conditioning_repeat \
  --frequency-source fitted-family --copies 1 --caps 1 3 10 30 \
  --seeds 0 1 --iterations 1000 --seconds-per-fit 28
```

Use a fresh output directory. The alternative `--frequency-source atlas --copies 1 2`
tests six representative atlas peaks with one or two resonant directions per peak.
That atlas subset does not cover all peaks in the fitted interval. Smaller-family
and complete-family fits have different broad complements and are not a pure T ablation.

The model evolves the observed position and velocity only:

    dz = v dt
    dv = (-K z - D v) dt + G dW
    K = T diag(omega**2) inv(T)
    D = T diag(2*gamma) inv(T)

All fits have 21 position coordinates. T has unit-norm columns and a projected hard
condition-number bound in training-standardized prepared coordinates. T is represented
as an orthogonal matrix times a symmetric matrix exponential followed by column
normalization. The actual normalized T condition is checked after each optimizer step.
The full real acceleration covariance is learned in every variant, allowing nonzero
imaginary cross-spectra. There is no additive background or observation noise.
Nonresonant directions have broad second-order dynamics within this same square T.

Peak locations come either from train-only atlas selection or the earlier train-only
two-family frequency fit. In the latter case, natural frequencies are converted to
isolated white-forcing displacement peak frequencies using saved damping. Each peak
parameter may move by at most 8 Hz; natural frequencies account for learned damping.
Correlated forcing means these isolated-mode peak locations need not equal maxima
of the mixed observed PSD. Six or twelve resonant directions leave fifteen or nine
broad directions; the full twenty-frequency model leaves only one broad direction.

Spectra are the exact expected sampled, finite-window periodic-Hann spectra of this
continuous model, including aliasing. Pairwise 2x2 oscillator Lyapunov equations give
stationary covariance; analytic time evolution and FFTs give complex spectra without
unrolling simulated paths. Constant detrending has no effect at the fitted Hann bins.
Internal time is milliseconds; saved K, D, acceleration_covariance, omega and gamma
use seconds. Saved arrays act on standardized prepared positions and their true
time derivatives, not the neural model's separately scaled native-frame increments.

The cache has 12 training and 2 validation recordings, no test recordings. Training
means/scales, objectives, initialization and best-step/seed selection exclude validation.
Welch uses 4096 samples at about 14.4 kHz, equal recording weights and no cross-record
windows. The 80–1500 Hz objective is log marginal PSD MSE plus twice normalized complex
off-diagonal cross-spectrum MSE. PSD and cross-spectrum metrics are reported separately,
including the spatial mean and POD1–3, and each validation recording separately.
The previous shared spatial basis includes held-out repetitions.

Peak metrics use 0.75 dB prominence, 15 Hz minimum spacing, 12 Hz location tolerance,
training peaks near prescribed frequencies, and confirmation by a validation peak.
Predictions match targets one-to-one. These are exploratory diagnostics; counts across
coordinates are not independent physical modes. Model seed selection never uses these
validation-confirmed targets. PSD plots are needed alongside aggregate errors.

Physical condition numbers use unstandardized coefficient coordinates, not a corrected
spatial Gram metric. Prepared-signal scales differ from native neural checkpoint scales.
The fits do not establish native-rate discretization stability, conditional forecasts,
or the behavior of a future neural residual. Broadening resonances to fit a missing
broadband component is a confound when interpreting these results as a test of T alone.

Tests independently compare complex Welch spectra with full state-space matrix
exponentials and Lyapunov solves, verify condition bounds and stable drift poles,
and compare autograd gradients against finite differences.
