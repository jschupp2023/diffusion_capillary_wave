# Shared latent oscillator baseline

The separate [observed-state oscillator feasibility test](observed_spectral.md)
fits a square, condition-constrained transformation directly in measured coordinates,
without additive OU states. It does not modify this latent baseline or neural checkpoints.

`python -m modelling.latent_oscillators.spectral_background --output <fresh-directory>`
provides a separate train-only valley-background diagnostic from saved spectral matrices,
with two exclusion-width settings, PSD overlays and numerical background matrices.

`python -m modelling.latent_oscillators.peak_background_fit --output <fresh-directory>`
fits the earlier 20 resonances above the frozen manual background. It compares
condition caps 1/3/10 with two initializations, fixes poles initially, then permits
frequency shifts of ±8 Hz and damping changes of ±25%. Only resonant directions
are forced. The additive spectral background is an identification aid, not a
dynamical OU process or a neural-SDE checkpoint conversion.

This exploratory model asks whether the 0p10 resonance peaks and lagged
correlations can be represented by a small, stable linear stochastic system
with shared observations. It does **not** change or fine-tune the neural SDE.

For oscillator j, the implementation uses velocity units `u = p / omega`:

```
dq = omega * u dt
du = (-omega * q - 2 gamma * u) dt + sqrt(4 gamma) dW
y = mean + scale * (sum_j C_j q_j + broadband_background)
omega_j = 2 pi (a_family * n_j + b_family)^(3/2)
```

Each `(q,u)` pair has identity stationary covariance. This fixes the amplitude
ambiguity between oscillator forcing and the mixing matrix: readout columns
contain the excitation amplitudes. All coefficients are real. The background
is a sum of independent vector OU processes with fixed corner frequencies
30, 300 and 3000 Hz and learned covariance factors. `rank=2` gives two
independent copies of each resonant pole; it is not a second harmonic.
With 20 poles and 21 observed coordinates, rank one has 40 oscillator states
plus 63 broadband states (103 total); rank two has 143 total. This is a linear
diagnostic baseline, not yet a claim of a minimal latent realization.

Positive dispersion parameters and damping are learned. Damping is bounded
below the natural frequency to keep the oscillator underdamped. A natural
frequency is not exactly the maximum of a broadened displacement PSD.
Separate families have separate `(a,b)`; their ordered indices are descriptive,
not identified physical mode numbers or boundary conditions.

## Data and fitting

The CLI consumes `signals.npz` and `provenance.json` from the existing
`harmonic_dynamics_audit`. That cache contains physical shared-POD coordinates,
scaled by sqrt(number of image pixels), with coordinate zero the 50 Hz
high-pass spatial mean, anti-alias downsampling by eight and discarded filter
edges. It uses training repetitions 1–12 and validation 13–14; test recordings
are excluded. Input files and HDF5 sources are never modified.

Training-only peak detection on coordinate 0 and POD 1 initializes the two
families. Full 21-coordinate Welch spectral matrices are then fitted jointly.
Training-only means/scales balance different coordinate amplitudes. The loss
is squared natural-log autospectral error plus twice normalized off-diagonal
complex spectral error, averaged over 80–1500 Hz. Fits select the checkpoint
with lowest training objective; validation never selects a checkpoint.

Expected spectra include the finite Hann-window spectral smoothing and exact
sampling of the continuous process. Constant detrending affects only bins
0 and 1 for the periodic Hann window; neither is fitted. This avoids mistaking
Welch resolution for oscillator damping. Independent trajectories are never
concatenated before computing spectra.

Run from the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.latent_oscillators.study \
  --audit-dir runs/adaptive/neural_sde_0p10_tcn63_untrimmed_structured/trials/untrimmed_variance_w0p001_lr1e6/rollout_99937_validation_1/harmonic_dynamics_audit \
  --output runs/diagnostics/latent_oscillators_0p10_two_family_v1 \
  --iterations 1800 --paths 16
```

Use a new output directory for a new run; existing results are not overwritten.
CPU only; no GPU contention with the neural-SDE controller. Four variants run
by default: `smooth`, `one_family`, `two_family`, `two_family_rank2`.

## Evaluation and artifacts

Each variant saves parameters (`model.npz`), convergence history, exact model
spectra, per-record Welch matrices, ACFs, stationary stochastic rollouts,
`modal_psd.png`, `acf.png`, `mixing.png`, and numerical metrics. The parent
directory contains the protocol, cache SHA256, a comparison and summary.

Simulation uses exact matrix-exponential transitions and `Q=I-F F.T` in
stationary normalized oscillator coordinates. Initial states are independent
stationary draws. This tests autonomous distributional behavior, **not**
conditional forecasting or inferred initial phases. The saved rollout is an
explicit standalone array format, not the shared neural-SDE rollout schema.

PSD metrics use 100–1100 Hz. Ripple error compares log spectra after subtracting
the same Gaussian-smoothed envelope (40 Hz standard deviation). Peak recall
uses >=0.75 dB prominence, >=40 Hz spacing and +/-22 Hz matching, for training
peaks also present in validation. These thresholds are descriptive, not
statistical significance tests. Validation has only two independent records.
The existing native TCN has shorter trajectories and overlapping start windows;
it is a reference comparison, not a matched training-objective ablation.

For a stricter comparison with identical analyzed record lengths and identical
100–1100 Hz filtering before the ACF calculation, run:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.latent_oscillators.audit \
  runs/diagnostics/latent_oscillators_0p10_two_family_v1
```

This creates `matched_comparison.md/json`, `comparison_psd.png`, and
`comparison_band_acf.png`. It also refreshes subset metrics and one-to-one
peak counts in each variant's summary; it never refits a model. Peak candidates
remain based on full training records, independent of evaluation duration.

To export a saved simulation and generate the usual full-band PSD, energy,
modal moments, covariance and trajectory plots:

```bash
python -m modelling.latent_oscillators.export_rollout \
  runs/diagnostics/latent_oscillators_0p10_two_family_v1/two_family
python -m data_analysis.plot_rollout \
  runs/diagnostics/latent_oscillators_0p10_two_family_v1/two_family/standard_rollout_validation_2x8 \
  --nperseg 2048 --peak-nperseg 2048 --modal-psd
```

This restores the shared physical coefficient convention by multiplying the
cached coefficients by sqrt(number of spatial pixels). Reference and generated
signals receive no new filtering or trimming. Generated paths are independent
stationary draws, grouped with two validation records only for distributional
comparison; pointwise errors are not forecast accuracy. Full PSD here means
the prepared-grid range up to 7.2kHz, not the native experimental Nyquist.
The standard report intentionally exposes errors outside the fitted band.

## Limits and possible next steps

- The fit is band-limited. Total energy/variance, low-frequency drift and
  conditional state prediction require separate validation.
- Real displacement-only readouts imply real cross-spectra. The imaginary
  mismatch is retained in the loss, but cannot be fixed by its gradients.
  A velocity/quadrature readout would be a separate extension if this matters.
- Smooth OU backgrounds are a convenient approximation, not inferred physics.
- A shared linear stochastic resonance fit does not establish capillary
  eigenmodes, deterministic forcing, or nonlinear phase coupling.
- No neural residual or state-dependent diffusion is added yet. First establish
  what this linear core captures, then test a residual without erasing its poles.

Tests:

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python -m pytest modelling/latent_oscillators/tests -q
```

## Frozen hidden-state inference diagnostic

`inference.py` builds the exact discrete state-space model from a saved fit and
provides a Kalman filter, conditional forecasts and a noncausal endpoint
smoother. Both oscillator and colored OU states are retained, with stationary
prior covariance `I` and no added observation noise. The inference code neither
inverts the mixing matrix nor refits the spectral parameters.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.latent_oscillators.inference_audit \
  runs/diagnostics/latent_oscillators_0p10_two_family_v1 \
  --output runs/diagnostics/latent_oscillator_inference_0p10_v1
```

Use a fresh output directory. The audit checks the fitted cache hash and fixed
12/2 train/validation split. It tests 256 independent synthetic paths with known
hidden states, comparing snapshot conditioning with 1–200 ms observation
histories. A separate 50 ms future-observation smoother quantifies the additional
information available offline; its estimates never enter causal forecasts.

On validation repetitions 13–14, all history lengths forecast at the same
origins, with horizons from one prepared sample to 50 ms. Comparators are the
stationary mean, persistence and instantaneous Gaussian conditioning. Forecasts
consume no observations after their origin. Metrics include per-coordinate and
coordinate-group errors, predictive Gaussian calibration, and pre-update
innovation spectra and correlations. Windows overlap within the two records.
The original offline preprocessing means causality is relative to the prepared
cache, not the native camera stream. No additional temporal filtering is used.

Outputs include `README.md`, numerical metrics, plots, exact protocol/source
hashes, and normalized filtered states and resonant outputs. Array dimensions
are: `synthetic_estimates[history,path,state]`,
`validation_forecasts[history,horizon,repetition,origin,coordinate]`,
`validation_origin_states[history,repetition,origin,state]`, and
`validation_filtered_states[repetition,time,state]`. State order is `(q,u)`
per pole/copy, followed by each vector OU block; `u=p/omega`.
The saved states are uncertain diagnostic estimates, not ground-truth labels
for neural-SDE training. Replacing the OU model requires revisiting inference.
