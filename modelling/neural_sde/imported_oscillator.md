# Imported oscillator drift with a neural residual

## Forcing-history TCN option

Add `--residual-input core_innovation` to a **fresh** imported-oscillator TCN
training run. The default `state` mode preserves existing checkpoints and behavior.
The new mode uses the same observed coordinates, frozen operators, Gaussian
likelihood, multistep objectives, and differentiable cached rollouts, but replaces
the neural inputs with past forcing increments:

```
u[n] = x[n] - x[n-1]
e[n] = u[n] - u[n-1] - Fcore(x[n-1], u[n-1])
u[n+1] = u[n] + Fcore(x[n], u[n])
         + std(u) * TCN(e[n-H+1:n+1] / std(u)) + G * noise[n]
x[n+1] = x[n] + u[n+1]
```

`Fcore` includes the native-step `dt²` and `dt` factors below. Inputs are
chronological, divided by the saved velocity standard deviation, and are **not**
centered by mean velocity. Only transitions ending at or before the current
state enter the TCN. Training and rollout use the same extraction. A 63-token
TCN automatically requests 64 preceding positions. Neither raw position nor
velocity is separately fed to this neural branch. There are no hidden-state
labels or additional loss hyperparameters.

This focused option supports TCN with constant diagonal or full diffusion;
state-dependent diffusion is rejected to keep forcing independent of direct
state feedback. Existing checkpoints cannot be relabeled or warm-started across
the two input modes. Checkpoints embed the mode and operators, and resume/evaluate
inherit them. The TCN output starts at zero; diffusion is fitted jointly.

For a stable linear forcing model, this gives a forcing process followed by the
fixed oscillator response. It does not guarantee visible peaks: forcing can have
spectral notches, and nonlinear forcing can be unstable. Diffusion amplitude and
long-rollout energy still require calibration.

Example fresh rank-19, TCN63, five-epoch experiment (run from repository root):

```bash
source /home/jonas/miniforge3/etc/profile.d/conda.sh
conda activate diff-model-mit
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
python -m modelling.neural_sde train \
  --experiment 0p10 --rank 19 --spatial-mean-highpass-hz 50 \
  --shared-basis /home/jonas/ucsd_thesis/reduced_data/0p10/shared_pod/shared_pod_r1000.h5 \
  --validation-reps 13 14 --test-reps 15 16 \
  --state-variable velocity --lag 1 --history 1 \
  --drift-type imported_oscillator_residual \
  --oscillator-artifact runs/diagnostics/peak_background_fit_0p10_rank19_v1/cap10_seed1/model.npz \
  --residual-architecture tcn --residual-input core_innovation \
  --tcn-context-frames 63 --tcn-channels 64 --tcn-rollout-cache \
  --diffusion-type diagonal --train-mode joint \
  --learning-rate 0.0001 --batch-size 256 --epochs 5 --device cuda \
  --output runs/comparison/neural_sde_0p10_core_innovation_tcn63_5epochs_v1
```

This command has **not** been run. Validation: 105 imported-oscillator/TCN/cache
tests pass, including recovery of known forcing under different oscillator
initial conditions, cached/noncached multistep gradients, CLI training/evaluation,
and checkpoint reload. The bounded [real-data smoke](../../runs/diagnostics/core_innovation_tcn_0p10_smoke_v1/README.md)
completed five updates in about 3 seconds; it verifies integration, not fit quality.

## State-history residual (existing default)

`--drift-type imported_oscillator_residual` uses the fitted resonant operators in the ordinary observed-coordinate velocity SDE. The learned term is the existing MLP or TCN, with the usual history inputs and diagonal, full, or state-dependent diffusion. No latent oscillator state or additional inference step is introduced.

For native-frame velocity `u[n] = x[n] - x[n-1]`, the conditional velocity change is

```
E[u[n+1] - u[n]] = -dt² Kphysical (x[n] - equilibrium)
                   -dt Dphysical u[n] + target_std * residual(normalized inputs).
x[n+1] = x[n] + u[n+1]
```

The existing Gaussian likelihood, multistep objectives, moment losses, cached TCN rollouts, trimming, and checkpoint/evaluation workflows remain available. The first version requires native lag 1 and disallows the separate additive memory extensions. Diagonal-operator orthogonality penalties do not apply to this nonorthogonal frozen operator. The stability regularizer's `auto` setting falls back to its empirical metric; its damped-residual-specific Lyapunov-core option does not apply.

## Import and normalization

The source is `runs/diagnostics/peak_background_fit_0p10_v1/cap10_seed1/model.npz`, with its enclosing study's `protocol.json`. The importer validates basis path, rank, units, spatial-mean preprocessing, training/validation repetitions, sampling, pole ordering, and consistency of saved matrices with T and modal parameters. Historical basis contents cannot be verified from a path alone; the spectral protocol does not contain a basis-content hash.

Only the first 20 fitted resonant directions are imported. The 21st, unforced completion direction's arbitrary stiffness/damping are set to zero before mapping back to observed coordinates. **There is therefore a neutral direction in the fixed core**; its dynamics must be supplied by the residual. The fixed spectral background and fitted acceleration covariance are not imported. Diffusion is learned by the standard SDE procedure.

The fit used `z=(x_cache-mean_fit)/scale_fit` and seconds; `x_native=sqrt(N)*x_cache`, with `sqrt(N)=200` for this source. Define `S=diag(sqrt(N)*scale_fit)`:

```
Kphysical = S Kresonant S^-1
Dphysical = S Dresonant S^-1
equilibrium = sqrt(N) * mean_fit
```

The network's normalized fixed term includes both position and velocity offsets. Conversion is refreshed after fitting normalization, including trimming and warm starts. Physical operators, equilibrium, converted matrices and affine bias are checkpoint buffers; the source artifact is only required for a fresh import. Its path and hashes are recorded in metadata.

The residual's final layer starts at zero. Operators remain fixed in every training mode. `--train-mode diffusion` retains the fixed oscillator and zeros/freezes only the residual; its diffusion covariance diagnostic uses velocity innovations after subtracting the fixed drift. The ordinary diffusion-to-drift staged workflow is supported. A previous MLP/TCN checkpoint cannot be directly relabeled as this model: its learned drift could duplicate or cancel the imported terms.

Each epoch logs fixed and residual normalized RMS on its first retained train/validation batch under `*_imported_*_normalized_rms_first_batch`. These are component diagnostics, not spectral guarantees or epoch-wide estimates. Evaluate exports physical and normalized operators in `metrics.json`.

## Commands

Bounded real-data smoke (48 training windows, 8 validation windows; five updates per architecture, four 1,024-step rollout paths):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.imported_oscillator_smoke \
  --output runs/diagnostics/imported_oscillator_residual_0p10_smoke_repeat
```

This imports only normalization statistics from the existing untrimmed native TCN checkpoint, not its learned neural weights. It writes explicitly labeled smoke checkpoints; they are not full-training resumable candidates.

A fresh joint training command (a full epoch scans the training population; this is not the bounded smoke):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python -m modelling.neural_sde train \
  --experiment 0p10 --rank 20 --spatial-mean-highpass-hz 50 \
  --shared-basis /home/jonas/ucsd_thesis/reduced_data/0p10/shared_pod/shared_pod_r1000.h5 \
  --validation-reps 13 14 --test-reps 15 16 \
  --state-variable velocity --lag 1 --history 1 \
  --drift-type imported_oscillator_residual \
  --oscillator-artifact runs/diagnostics/peak_background_fit_0p10_v1/cap10_seed1/model.npz \
  --hidden-layers 64 64 --diffusion-type diagonal \
  --train-mode joint --learning-rate 0.0001 --batch-size 256 --epochs 1 \
  --device cpu --output runs/comparison/neural_sde_0p10_imported_oscillator_mlp_v1
```

For a TCN, add `--residual-architecture tcn --tcn-context-frames 63 --tcn-channels 64 --tcn-rollout-cache`. Existing history-offset and diffusion options still apply. Frozen operators do not make the resulting PSD additive: the residual and diffusion can alter resonance amplitudes, widths, and stability. In particular, the resonant fit was made on downsampled data and its native-rate conditional behavior needs validation.

The first smoke passes finite-gradient and checkpoint checks but produces unrealistic rollout amplitudes after only five updates. It does not establish a useful trained model. See the [run report](../../runs/diagnostics/imported_oscillator_residual_0p10_smoke_v1/README.md).

## Fully resonant rank-19 control

The peak fitter now accepts `--rank 19`. It keeps spatial mean + POD1–19 by
selecting the corresponding principal submatrices of the saved empirical
spectra and background, then **refits** a 20×20 T for the existing 20 poles.
It does not crop a previously fitted operator. Coordinate selection is saved
in provenance and checked by the importer. Equal pole/coordinate counts are
supported, so this case has no completion direction.

```bash
python -m modelling.latent_oscillators.peak_background_fit \
  --rank 19 --caps 10 --seeds 1 \
  --iterations 1000 --fixed-steps 500 --seconds-per-fit 35 \
  --output runs/diagnostics/peak_background_fit_0p10_rank19_v1
python -m modelling.neural_sde.oscillator_core_check \
  --artifact runs/diagnostics/peak_background_fit_0p10_rank19_v1/cap10_seed1/model.npz \
  --steps 90000 --nperseg 8192 \
  --output runs/diagnostics/oscillator_core_0p10_rank19_v1
```

Use the project's `diff-model-mit` Python and headless plotting environment.
The second command checks strict native-step linear stability, then compares
zero diffusion with constant diagonal diffusion, both with zero neural residual.
It writes `README.md`, `report.json`, `core_rollouts.npz`, `psd.png`, and
`trajectories.png`. These are standalone core diagnostics, not the standard
experimental-reference `rollout.npz` schema.

The diagonal noise uses one scalar amplitude calibrated to unit mean theoretical
variance in the fit's standardized position coordinates. It is not the trained
neural diffusion or the fitted spectral forcing covariance. Both controls start
from the same stationary Gaussian sample. The stationary PSD prediction uses
the exact discrete update; empirical Welch spectra also include finite-window
smoothing. Only the fixed-core mechanism is being checked here, in float64.

Validation: 44 focused tests cover latent oscillators, imported operators, and
the new core check. A real rank-19 CLI smoke with only two fitting updates and
90,000-step controls completed in temporary output; that short fit is not the
requested production refit or evidence of fit quality.

## Fixed modal notch forcing

`--forcing-filter-artifact` enables the focused combined model. The imported
oscillator remains frozen. For each oscillator direction, a stable,
minimum-phase biquad maps a latent broadband forcing into the notched physical
forcing that enters the oscillator. Training causally inverse-filters each full
training/validation repetition, discards the initial filter transient, and fits
the ordinary TCN plus constant diffusion to that raw forcing. The cache is
in-memory only. Checkpoints store the filter coefficients, normalization, and
artifact hashes, so warm starts do not need the source artifact.

The initial inverse transient is long for these narrow notches; the default
`--forcing-filter-burnin 32768` retains about two thirds of each 100,001-frame
record. The current proof-of-concept supports one-step NLL training. Multistep,
stationary-moment, stability, history-noise, orthogonality, indexed-batch, and
streaming-TCN options are rejected explicitly until they carry the biquad state.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python -m modelling.neural_sde train \
  --experiment 0p10 --rank 19 --spatial-mean-highpass-hz 50 \
  --shared-basis /home/jonas/ucsd_thesis/reduced_data/0p10/shared_pod/shared_pod_r1000.h5 \
  --validation-reps 13 14 --test-reps 15 16 \
  --state-variable velocity --lag 1 --history 1 \
  --drift-type imported_oscillator_residual \
  --oscillator-artifact runs/diagnostics/peak_background_fit_0p10_rank19_v1/cap10_seed1/model.npz \
  --forcing-filter-artifact runs/diagnostics/modal_forcing_filter_0p10_rank19_v1/modal_biquad.npz \
  --residual-architecture tcn --residual-input core_innovation \
  --tcn-context-frames 63 --tcn-channels 64 --tcn-kernel-size 3 \
  --tcn-dilations 1 2 4 8 16 --diffusion-type diagonal \
  --train-mode joint --learning-rate 0.0001 --batch-size 256 --epochs 5 \
  --device cuda --output runs/comparison/neural_sde_0p10_modal_notch_tcn63_5epochs_v1
```
