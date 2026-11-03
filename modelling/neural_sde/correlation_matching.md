# Focused correlation matching

The existing two-independent-branch start-to-endpoint covariance loss supports
three opt-in settings. Omitting them preserves the old leading-POD,
batch-variance normalization and shared-horizon behavior:

```text
--stationary-lagged-covariance-coordinates 0 1 2 3 4
--stationary-lagged-covariance-normalization training
--stationary-lagged-covariance-horizons 128 512 1024 2048
```

Coordinate numbers refer to the actual model state: 0 is spatial mean **when
included**, and 1–4 are then POD 1–4. Explicit coordinates and
`--stationary-lagged-covariance-mode-count` are mutually exclusive. The latter
retains its old meaning: leading POD coordinates excluding the spatial mean.
Independent correlation horizons leave `--stationary-mean-horizons` and the
multistep likelihood horizons unchanged. Moment rollouts share the union of
endpoint and correlation horizons; sampled starts must pass the complete union
stencil. This can change the retained population for a trimmed model even
though endpoint horizons remain fixed. In a matched fine-tuning control, use
the same start eligibility when attributing changes to the loss.

## Definition and normalization

With `training` normalization, `fit_correlation_reference` streams all eligible
training starts once, preserving recording boundaries and rejecting the full
history-to-maximum-endpoint stencil under the saved increment trim. It computes
N-1 position variances in the model's normalized coordinates. These variances
stay fixed across optimization and validation batches. They are fitted once
per training invocation and saved with training-only counts and preprocessing
provenance under `training.stationary_lagged_covariance_reference`. A resumed
invocation recomputes them for its actual model normalization and lag stencil.

For coordinate j, lag h and branch r, the error is the centered sample
start-to-endpoint covariance difference divided by this fixed variance plus
epsilon. The loss is the coordinate/lag mean of `error_branch1*error_branch2`,
not the square of the mean error. Its expectation over independent rollout
noise is the squared conditional covariance mismatch for the sampled observed
batch; it is not an unbiased estimator of the population squared loss across
random data batches. Finite-batch values can be negative. Correlation gradients
still route to drift only, with inverse sampling-fraction scaling on scored
updates. Positive and negative covariance lobes are both targeted. Scalar
normalization does not turn this into generated-variance-normalized Pearson ACF.
The current weights within the coordinate and correlation-lag sets are uniform.
Per-lag logging also retains diagnostics at the union's other endpoint lags;
only the configured correlation subset enters the correlation aggregate.

## Gradient preflight

`correlation_gradient_diagnostic` reuses this loss and the objective-balance
gradient helpers. It supports both the saved GMM likelihood and energy score,
keeps the saved parent moment/multistep horizons and coefficients, and compares
all objectives on the same complete-stencil eligible starts. It reports raw,
expected-weighted and selected-update gradient norms by parameter group;
per-lag gradients and per-coordinate covariance errors; alignment with the
existing objective; within-batch rollout-noise and between-start-batch
variability; elapsed time and CUDA peak allocated memory. It never updates a
checkpoint. Train and validation splits are available; use **train** when
calibrating coefficients. Test is deliberately unavailable.

The between-batch statistic includes finite-noise error in each batch mean;
it is not a pure variance-component estimate. Two seeds/batches are a smoke
check, not a reliable gradient signal-to-noise estimate. The NLL is deterministic
within each batch; its duplicated noise-seed entries do not constitute extra
independent evidence. All anchor objectives use the long-stencil population in
this audit, which can differ from their ordinary training populations.

Bounded native-rate preflight (two start batches, two noise seeds, 16 starts):

```bash
PATH=/home/jonas/miniforge3/envs/diff-model-mit/bin:$PATH \
TRITON_PTXAS_PATH=/home/jonas/miniforge3/envs/diff-model-mit/bin/ptxas \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MPLCONFIGDIR=/tmp/cwt-mpl \
timeout --signal=TERM --kill-after=5s 180s \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.correlation_gradient_diagnostic \
  --checkpoint runs/adaptive/neural_sde_0p10_tcn63_untrimmed_structured/trials/untrimmed_variance_w0p001_lr1e6/best.pt \
  --output runs/diagnostics/neural_sde_0p10_correlation_gradient_trial \
  --split train --coordinates 0 1 2 3 4 \
  --horizons 128 512 1024 2048 --starts 16 \
  --window-seeds 0 1 --noise-seeds 0 1 --weight 0.01 --fraction 0.01 --device cuda
```

The diagnostic default additionally reaches 4,096 native steps (35.56 ms).
This is a costlier setting, not a promise of affordable full-epoch training.
At 115,200 Hz, 2,048 steps span 17.78 ms. Sparse lags alone do not identify a
PSD uniquely. Start with measured signed correlation curves and verify the
result against autonomous PSD peaks, energy/covariance and survival. Gradient
norms are Euclidean sensitivities, not Adam update sizes. `--weight .01` and
`--fraction .01` above are audit settings, not tuned training recommendations.

Tests include a unit-variance oscillator versus a stationary unit-variance AR
process with a smooth spectrum: the same loss detects missing oscillations
despite matched variance. This controlled check is not an empirical
peak-smoothing ablation on capillary-wave data; that remains a useful next audit.
