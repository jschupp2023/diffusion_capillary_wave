# Neural SDE baseline

## Optional faster TCN multistep and moment training

Append these execution flags to an otherwise unchanged training/fine-tuning
command (including when loading an older TCN checkpoint):

```bash
--tcn-rollout-cache --indexed-training-batches
```

Both default to off. For additional kernel fusion, optionally also pass
`--tcn-rollout-compile` (requires `--tcn-rollout-cache`). It uses PyTorch
compilation and can take minutes on first use or new shapes; steady-state
benchmarks exclude this startup cost. Compilation defaults off and is not
automatically enabled when loading checkpoints for evaluation. The ordinary
eager cached path remains available without a compiler/toolchain dependency.

These options preserve architecture, tensor dtype, particle schedule,
loss weights, scored modes/horizons, RNG sampling order, and trim eligibility.
The flags are recorded in training metadata; pass them explicitly again when
fine-tuning. Loaded optimized checkpoints also use the TCN cache for evaluation
rollouts. Old checkpoints retain the reference execution path.

- The call-local, differentiable TCN cache reuses causal layer inputs across
  generated steps. It covers Monte Carlo velocity/displacement likelihoods,
  stationary mean/variance/lagged-covariance objectives, and ordinary rollout.
  The measured history is warmed once per condition before expanding to MC
  particles. No cache is detached, and particles are still pruned according to
  the existing schedule. Convolution taps use equivalent linear operations;
  floating-point roundoff can differ from the reference Conv1d kernels.
  In particular cuDNN may use TF32 by default while Linear uses IEEE FP32.
  CUDA equivalence checks disable TF32 for both backends; the implementation
  does not silently change the user's global precision settings. Seeded paths
  are therefore not promised to match bit-for-bit or indefinitely in chaotic
  dynamics, even though the recurrence and loss are mathematically unchanged.
- An old current token must use a position-difference velocity when it becomes
  history, even if its original explicit velocity differed through roundoff.
  The streaming path recomputes that token and retains buffers only through the
  preceding token, preserving the existing recurrence and gradients.
- Streaming is used only when context equals the full receptive field (e.g.
  TCN63). Shorter contexts reset left-padding effects each step and safely fall
  back to the reference path. MLPs also keep their original drift execution.
- Indexed velocity batches cache per-repetition invalid-increment prefix
  counts, retain the exact full-stencil trim test, and gather only the selected
  long-window examples before transfer. Job order, within-job shuffling, small
  final batches, and example-selection RNG are unchanged. No repetition gaps
  are joined and generated paths are never trimmed. Compact coordinates use
  the existing bounded RAM cache; uncached data use bounded job-local reads.

Increasing `--batch-size` is a separate optimization experiment, not part of
these flags: it changes optimizer updates per epoch and rounding of the 2.5%
window count. Measure retained examples/second and validate training quality.
Moment objectives use their own `--stationary-mean-trajectories` and
`--stationary-mean-batch-fraction`; increasing the clean batch size does not
automatically increase the moment ensemble. No particle count or objective
was reduced by these execution optimizations.

Bounded benchmark entry points (neither modifies the source checkpoint):

```bash
python -m modelling.neural_sde.experiments.profile_tcn_batch \
  --checkpoint PATH/best.pt --output NEW_PROFILE_DIRECTORY \
  --tcn-rollout-cache --indexed-training-batches --skip-trace

python -m modelling.neural_sde.experiments.benchmark_tcn_moments \
  --checkpoint PATH/best.pt --output NEW_MOMENT_PROFILE_DIRECTORY \
  --horizons 20 100 500 --trajectories 32
```

The batch profiler now warms the same coordinate cache as production training.
The earlier October 14 profile omitted this warmup, so its ~90 ms CPU fetch
figure should **not** be used as a production-training bottleneck estimate.

## Optional noisy-history drift training

For native-lag velocity models (TCN or MLP, structured or unstructured), add
`--history-noise-weight 0.1 --history-noise-std 0.05` to a training or joint
fine-tuning command. These are initial experimental settings, not calibrated
optima. Weight zero (the default) skips all augmentation and noise RNG work.

This adds a coordinate-mean squared next-velocity prediction error, normalized
by the saved `target_std`, alongside the unchanged clean one-step Gaussian NLL.
Only drift parameters receive the auxiliary gradients; diffusion continues to
learn from the clean NLL. This is not an extra noisy-data likelihood.

For each history window, draw independent native-velocity perturbations with
SD `history_noise_std * target_std` and integrate them backwards, keeping the
current position fixed. Position error at age k is minus the sum of the first
k perturbations; its SD grows as sqrt(k). Velocities are then derived from
these positions, preserving consistency. The next measured position/velocity
target stays unchanged. Trimming and normalization are computed on the original
data before perturbation; augmented inputs are not re-trimmed. No source data
or rollout sampler is modified. This is local robustness training, not a global
stability guarantee.

Clean `train_nll` / `validation_nll` remain separately logged. The additional
`*_history_noise_loss` and `*_history_noise_weighted_contribution` describe the
auxiliary branch. `best.pt` uses the total validation objective, including this
branch when enabled. Validation perturbations repeat using
`--history-noise-validation-seed` (default 0); training uses seed + epoch.
Training options are recorded in the checkpoint/config but must be explicitly
supplied again to enable augmentation in a later continuation.

A matched control uses the same positive weight with `--history-noise-std 0`:
this isolates adding velocity MSE from adding noisy histories. Compare both to
weight zero. Evaluate all arms with clean held-out NLL/innovations and full-noise
rollout survival, means, amplitudes, and PSD. Additive memory models and
diffusion-only/memory-only training are currently unsupported.

Example from the original 0p10 TCN63 checkpoint (does not modify that run):

```bash
python -m modelling.neural_sde train \
  --checkpoint runs/comparison/0p10_tcn63/best.pt \
  --train-mode joint --train-trim-percent 5 --multistep-weight 0 \
  --history-noise-weight 0.1 --history-noise-std 0.05 \
  --learning-rate 0.0001 --batch-size 128 --epochs 2 --seed 0 \
  --device cuda --output runs/comparison/0p10_tcn63_history_noise_s0p05_w0p1
```

## Basic usage

Run from the repository root in the same PyTorch environment as the EDM workflow:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 50 --epochs 20 --batch-size 256 \
  --hidden-layers 64 64 --diffusion-init 0.1 \
  --output runs/0p20_r50_neural_sde

python -m modelling.neural_sde evaluate \
  --checkpoint runs/0p20_r50_neural_sde/best.pt \
  --split test --num-conditions 32 --ensemble-size 16 --horizon 50 \
  --output runs/0p20_r50_neural_sde/evaluation
```

The SDE reads the same shared-POD source, whole-repetition split, and native
one-step transitions as EDM. Default rank 50 means 51 coordinates including
the spatial mean; `--no-spatial-mean` uses 50 POD coordinates. The model fits
training-only state and increment statistics with EDM's existing fitter.
`--dt` defaults to the source sampling interval and must agree with it.

To use one of the full-trajectory high-pass spatial means stored in the POD
files, add (for example) `--spatial-mean-highpass-hz 20` to the training
command. Coordinate zero is then `sqrt(N)` times the stored 20 Hz high-pass
mean. The selected cutoff and dataset identity are saved in the checkpoint;
evaluation reopens the same trajectory automatically and has no cutoff
override. Do not combine this option with `--no-spatial-mean`.

The MLP predicts drift in standardized **per-native-step** units. The trainable
diffusion factor has a positive diagonal, and `--diffusion-init` is its initial
native-step standard deviation in those same standardized units. `--lag k` uses pairs
separated by `k` native steps: the likelihood multiplies drift by `k` and
diffusion standard deviation by `sqrt(k)`. The default is `--lag 1`. The total
validation objective selects `best.pt`: NLL alone by default, plus any enabled
multistep NLL and stability penalty.

By default, `--diffusion-type diagonal` learns one independent noise scale per
coordinate. Use `--diffusion-type full` to learn a constant full covariance:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 50 --diffusion-type full \
  --epochs 20 --batch-size 256 --output runs/0p20_r50_neural_sde_full_diffusion
```

The full model learns a lower-triangular factor `G`, initialized as
`diffusion_init * I`, so the increment covariance `G G^T` is positive definite.
This adds `d(d+1)/2` diffusion parameters instead of `d`, where `d` is the
number of modeled coordinates. The diffusion remains constant in state; only
cross-coordinate noise correlations are added.

Use `--diffusion-type state_diagonal` for multiplicative diagonal noise:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --lag 3 --state-variable increment \
  --diffusion-type state_diagonal --train-trim-percent 3 \
  --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p20_r20_incr_lag3_state_diffusion
```

This diffusion has its own MLP weights but exactly the drift network's input,
hidden widths, and activation. Its softplus-positive outputs form a diagonal
factor `G(x)`. The final layer is initialized to give the constant value from
`--diffusion-init`, after which training can learn state and history dependence.
For increment and velocity models it sees both normalized position and the
modeled increment/velocity; larger history settings use the same complete
conditioning vector as the drift. Training and rollout still use
Euler-Maruyama. This option does not yet apply a Milstein correction.

The unbounded softplus model above can amplify its own rollout excursions when
it extrapolates beyond the training states. The recommended state-dependent
alternative is `bounded_state_diagonal`:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --lag 1 --state-variable increment \
  --diffusion-type bounded_state_diagonal --diffusion-log-range 2 \
  --train-trim-percent 3 --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p20_r20_incr_lag1_bounded_diffusion
```

It learns a positive per-mode baseline `sigma_0` and a separate conditional MLP
`h(x)`, using

`sigma(x) = sigma_0 * exp(diffusion_log_range * tanh(h(x)))`.

Consequently the state-dependent multiplier is smoothly restricted to
`[exp(-diffusion_log_range), exp(diffusion_log_range)]`; the default value 2
gives approximately `[0.135, 7.39]`. The baseline remains trainable, but cannot
grow in response to an out-of-distribution rollout state. The modulation MLP
starts at zero, so the initial diffusion is exactly the constant value selected
by `--diffusion-init`. Use a smaller `--diffusion-log-range` for a tighter
state-dependent range.

To evolve the **increment** instead of the state, use `--state-variable increment`:

```bash
python -m modelling.neural_sde train \
  --experiment 0p15 --rank 20 --lag 3 --state-variable increment \
  --train-trim-percent 1 --epochs 20 \
  --output runs/0p15_r20_increment_sde_lag3
```

At lag `k`, the SDE variable is `v[t] = a[t] - a[t-k]`. The drift sees the
training-normalized current state `a[t]` and current increment `v[t]`, and the
likelihood fits `v[t+k] - v[t]`. Constant diffusion is independent of both
inputs; both state-dependent diffusion types use the same inputs as the drift.
Rollout evolves `v[t+k]` and reconstructs
`a[t+k] = a[t] + v[t+k]`, so evaluation still saves physical shared-POD state
trajectories. Increment mode defaults to `--history 1`; larger history counts
also append older states to the drift input. Warm starts and evaluation must
use the training lag because changing the lag changes the definition of `v`.
Its diffusion covariance diagnostic compares changes in consecutive
increments; `--train-trim-percent` still removes the largest next increments.
The default `--state-variable state` keeps the original state SDE and existing
checkpoints load as before.

For a causal genuine second-order model, use `--state-variable velocity`:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 3 \
  --train-trim-percent 5 --epochs 20 \
  --output runs/neural_sde_0p20_r20_velocity_trimmed
```

Velocity is computed at every native frame, independently of the training lag,
by treating one recorded frame interval as one unit of model time:

```text
u[n] = a[n] - a[n-1].
```

For `--lag k`, training pairs `(a[n], u[n])` with `(a[n+k], u[n+k])`. The
likelihood models `u[n+k] - u[n]`, with frame-time step `h = k`, drift
proportional to `k`, and diffusion standard deviation proportional to
`sqrt(k)`. The velocity definition is therefore never widened to the training
lag. Rollout carries velocity explicitly and uses symplectic Euler-Maruyama:

```text
u[n+1] = u[n] + b(a[n], u[n]) h + G sqrt(h) xi[n]
a[n+1] = a[n] + h u[n+1].
```

At evaluation, `h = --rollout-lag` in frame-time units, which may differ from
the training lag. Initialization always uses the immediately preceding native
frame; the velocity is not redefined at either lag. Physical output timestamps
still use `--rollout-lag * dt`. A rollout step not used for training is an
integration extrapolation and should be checked against a rollout at the
training step subsampled to the same output times.

At native lag 1, velocity mode also supports additional contiguous or sparse
history. Offset 1 is used to construct `u[n]` and is not duplicated as a raw
state input; later offsets are appended as normalized positions. For example,
`--history-offsets 1 2 3 5 10` supplies information equivalent to
`[a[n], u[n], a[n-2], a[n-3], a[n-5], a[n-10]]`. The residual drift MLP and a
state-dependent diffusion MLP receive this full vector. A structured linear
drift still applies its restoring and damping matrices only to `a[n]` and
`u[n]`. Additional velocity history is deliberately restricted to `--lag 1`,
because a rollout that jumps multiple native frames does not generate the
intermediate states needed to update native-frame delay taps exactly.

For a smaller and more controlled memory test, a velocity checkpoint can be
expanded with an explicit diagonal drift-memory kernel:

```text
b_memory[n] = sum_k B_k u[n-k],   B_k diagonal.
```

`--velocity-memory-offsets 1 2 5 10 20 50` selects the native-frame velocity
taps. These taps bypass the residual MLP, whose input width therefore remains
unchanged. With `--train-mode memory --checkpoint ...`, the new coefficients
start at zero and are the only trainable parameters; the source drift,
diffusion, and normalization remain frozen. Rollout reconstructs every past
velocity causally from its generated position buffer. This mode requires a
memory-free joint velocity checkpoint trained at native lag 1.

Alternatively, stable oscillator memory represents a compact recurrent history:

```text
q_j[n+1] = rho_j R(phi_j) q_j[n] + sqrt(1-rho_j^2) B_j u[n+1]
b_osc[n] = C q[n],                 0 < rho_j < 0.9999.
```

Each `--velocity-oscillator-periods` value initializes one two-state rotation;
its radius, phase, input projection, and bounded output coupling are learned.
`--velocity-oscillator-burnin` measured velocities initialize `q` for each
training or evaluation window. Rollout then carries `q` recursively instead of
retaining that long buffer. The output coupling starts at exactly zero, so a
memory-mode warm start initially reproduces its parent checkpoint:

```bash
python -m modelling.neural_sde train \
  --train-mode memory --checkpoint runs/parent/best.pt \
  --velocity-oscillator-periods 8 32 128 \
  --velocity-oscillator-burnin 256 \
  --velocity-oscillator-radius-init 0.98 \
  --velocity-oscillator-norm-cap 0.25 \
  --epochs 5 --output runs/oscillator_memory
```

Autoregressive innovation memory instead models colored stochastic forcing.
For the frozen parent, let
`e[n] = (u[n] - E_parent[u[n] | history]) / target_std`. Then

```text
e_hat[n+1] = sum_k A_k e[n+1-k],
E[u[n+1] | history,e] = E_parent[u[n+1] | history] + target_std e_hat[n+1].
```

The low-rank kernels are selected with `--velocity-innovation-offsets`,
`--velocity-innovation-rank`, and `--velocity-innovation-norm-cap`. Their
summed induced norm is conservatively capped below one. Observed residuals
initialize training windows; sampled residuals update the state during rollout.
All three memory families are mutually exclusive. In memory mode the parent
drift, diffusion, and normalization stay frozen.

## Optional Monte Carlo multistep likelihood

Velocity models can add a closed-loop endpoint likelihood to the unchanged
one-step Euler--Maruyama NLL:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 \
  --multistep-weight 0.1 --multistep-horizons 2 4 \
  --multistep-particles 100 --multistep-validation-seed 0 \
  --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p20_r20_velocity_multistep
```

For a horizon `H`, the objective expands each observed starting state and its
history into `M` particles, applies `H-1` reparameterized stochastic rollout
steps without teacher forcing or gradient detachment, and evaluates the true
endpoint native-frame velocity under the `M` final conditional Gaussians. It
uses the joint-coordinate log density for each particle, then computes
`-logsumexp(logp, particle) + log(M)`. No KDE or separately fitted mixture is
used. The complete position and velocity state is simulated with the same
symplectic Euler--Maruyama update as normal rollout; position is not scored.

To score position instead, add `--multistep-target displacement` when
`--multistep-weight` is positive. The default is `--multistep-target velocity`.
The displacement target is `a[n + H*lag_steps] - a[n]`, relative to the same
observed initial position for every particle. At the last transition, with
`h=lag_steps`, velocity mean `mu_v`, and velocity Cholesky factor `L_v`, each
displacement component has mean `delta_previous + h*mu_v` and factor `h*L_v`.
Thus its covariance is `h**2 * L_v @ L_v.T`. The previous displacement is
accumulated along the sampled path, preserving gradients without subtracting
large absolute positions. Particle weights remain `1/M`; the final noise is
integrated analytically, and the joint-coordinate `logsumexp` reduction is
unchanged. No additional covariance fit, bandwidth, or energy score is used.

For native lag 1 and horizon 1, displacement NLL equals velocity NLL. For
larger model lags, displacement scores the actual measured endpoint position
change; it is not replaced by `h` times the measured endpoint native velocity.
The one-step training term remains velocity NLL in either target mode.

`H` counts applications of the configured model step, so its observed endpoint
is `H * lag_steps` native frames after the start. The scored endpoint velocity
is still the native difference `a[n] - a[n-1]`. Multiple horizons share the
same generated paths. A shorter-horizon distribution is scored before its
sample is used solely to continue to a longer requested horizon.

Training noise is resampled normally. Validation rebuilds its generator from
`--multistep-validation-seed` each epoch, making its Monte Carlo estimate
reproducible. `metrics.jsonl` records the one-step NLL, total combined loss, and
each horizon NLL separately. A saved checkpoint can be rescored at other
particle counts without retraining:

```bash
python -m modelling.neural_sde evaluate \
  --checkpoint runs/neural_sde_0p20_r20_velocity_multistep/best.pt \
  --split validation --num-conditions 1 --ensemble-size 1 --horizon 1 \
  --no-metrics --multistep-horizons 2 4 \
  --multistep-particles 1000 --multistep-seed 0 \
  --multistep-max-windows 10000 --multistep-window-seed 0 \
  --output runs/neural_sde_0p20_r20_velocity_multistep/likelihood_m1000
```

For endpoint-velocity multistep training, the likelihood dimension can decrease
with horizon. Supply one POD-mode count per horizon:

```bash
python -m modelling.neural_sde train \
  --experiment 0p10 --rank 20 --state-variable velocity --lag 1 \
  --spatial-mean-highpass-hz 50 --train-trim-percent 5 \
  --multistep-weight 0.1 --multistep-target velocity \
  --multistep-horizons 10 20 30 --multistep-mode-counts 10 3 3 \
  --multistep-particle-counts 64 24 12 \
  --multistep-window-fraction 0.05 \
  --multistep-window-sampling batches \
  --epochs 20 --batch-size 256 \
  --output runs/comparison/velocity_0p10_selected_mode_multistep
```

At each horizon, `--multistep-mode-counts N...` scores the joint marginal of
the first `N` shared-POD modes. Coordinate 0, when present, is the spatial mean
and is excluded. The model still evolves every coordinate along every latent
path, and the one-step NLL still scores the full reduced state. With full
diffusion, the implementation forms the selected covariance marginal before
its Cholesky factorization, retaining covariance contributions from omitted
coordinates. Omitting the option preserves the previous all-coordinate
multistep likelihood. Counts are saved in the checkpoint and inherited by
evaluation when the saved horizons are used; provide explicit counts when
evaluating different horizons. Mode counts are supported only for the velocity
target.

`--multistep-particle-counts` supplies one nonincreasing count per horizon.
After scoring a horizon, the implementation slices every particle-dependent
state tensor to the next count before continuing the rollout; this reduces the
actual drift/diffusion work rather than only using fewer mixture components at
the end. Without this option, every horizon uses `--multistep-particles` as
before. One-step and multistep terms use separate data iterators. The one-step
NLL always sees every retained one-step transition, so changing the longest
multistep horizon does not silently remove additional one-step examples. The
multistep iterator independently constructs long windows and rejects a window
when any increment along its complete measured path fails the trim rule.
`--multistep-window-fraction` samples from this long-window population only.
Training resamples it each epoch; validation uses the fixed
`--multistep-validation-seed`.

Use `--multistep-objective energy_score` for a density-free conditional
velocity objective. For each measured starting window and horizon it generates
`M` independent complete autoregressive paths and evaluates

```text
mean_m ||(V_m - v_obs) / target_std||
  - sum_{m<n} ||(V_m - V_n) / target_std|| / (M (M - 1)).
```

This is the unbiased distinct-pair estimator of the ordinary energy score:
self-pairs are excluded and distances are Euclidean, not squared. The score is
formed separately for each starting window before averaging, so measured
start/future conditioning is retained. `target_std` is the checkpoint's
training native-velocity scale; there is no additional whitening parameter.
Energy-score schedules require at least two paths at every horizon. For
example:

```bash
python -m modelling.neural_sde train \
  --checkpoint runs/structured_parent/best.pt \
  --multistep-weight 0.1 --multistep-objective energy_score \
  --multistep-target velocity --multistep-horizons 2 5 10 20 \
  --multistep-particle-counts 8 8 4 4 \
  --multistep-window-fraction 0.05 \
  --multistep-gradient-groups drift diffusion \
  --epochs 1 --output runs/structured_parent_energy_score
```

The default remains `--multistep-objective gmm_nll`. Optional multistep
rescoring under `evaluate` remains the MC Gaussian-mixture likelihood
diagnostic, even for a checkpoint trained with the energy score.

With `--multistep-window-sampling examples`, the configured fraction is selected
within each independently drawn long-window batch. Its sample mean estimates the
same batch multistep mean. The one-step NLL and stability term remain evaluated
on their complete one-step batches.

With small fractions, `--multistep-window-sampling examples` can leave too few
windows in every GPU launch. Select `batches` to score a complete minibatch
approximately once per `1/fraction` batches. The loss applies the inverse
fraction correction, so the multistep gradient remains an unbiased estimator
of the configured objective; its variance and per-update magnitude are larger.
At fraction `0.05` and weight `0.1`, an active batch therefore uses coefficient
`2.0`. Training and validation use reproducible systematic sampling rather than
clustered Bernoulli draws.

`--multistep-gradient-groups` controls only which parameter groups receive the
MC multistep gradient. The one-step NLL continues updating every parameter
allowed by `--train-mode`, and diffusion remains active when particles and
final Gaussian densities are generated. The default is `drift`, which excludes
diffusion from the MC backward pass. Available groups are `drift`, `structured`,
`stiffness`, `damping`, `mass`, `residual`, `memory`, and `diffusion`; multiple
groups form their union. For example:

```bash
# MC updates only the learned restoring coefficients.
--multistep-gradient-groups stiffness

# MC updates restoring and damping terms, but not the residual MLP or diffusion.
--multistep-gradient-groups structured

# Reproduce the earlier joint MC-gradient behavior explicitly.
--multistep-gradient-groups drift diffusion
```

The resolved parameter names are saved with the checkpoint. A requested group
that does not exist or is frozen by `--train-mode` is rejected instead of
silently producing no update.

Optional likelihood evaluation accepts the same particle schedule explicitly,
for example `--multistep-particle-counts 512 256 128`. Evaluation does not
inherit the training particle schedule, so a larger fixed-window rescore can
measure finite-particle sensitivity independently.

Training saves `multistep_target` in `config.json`, checkpoints, and active
multistep epoch logs. Evaluation inherits the checkpoint target; old checkpoints
without this field use velocity. `evaluate --multistep-target displacement`
or `--multistep-target velocity` explicitly overrides it. Evaluation reports
the scored target and its definition, so scores for different targets are not
silently compared. For displacement training with the existing trim heuristic:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 \
  --drift-type damped_residual --diffusion-type bounded_state_diagonal \
  --diffusion-log-range 0.5 --spatial-mean-highpass-hz 50 \
  --train-trim-percent 5 \
  --multistep-weight 0.1 --multistep-target displacement \
  --multistep-horizons 2 4 10 20 --multistep-particles 100 \
  --multistep-validation-seed 0 --epochs 5 --batch-size 256 \
  --output runs/comparison/velocity_lag1_multistep_displacement_damped_bounded_diffusion_0p5
```

This writes `multistep_likelihood` to `metrics.json`; repeat with different
`--multistep-particles` while holding `--multistep-seed` fixed to assess
estimator convergence. `--multistep-max-windows` scores a reproducibly shuffled
subset after training-trim filtering; hold `--multistep-window-seed` fixed so
particle-count comparisons use the same retained windows. The cap affects only
this optional likelihood pass, not the separately saved rollout. If the
checkpoint used training trimming, this likelihood evaluation applies its
saved cutoff to the complete scored windows.
The likelihood trim is independent of
`--enforce-train-trim-on-reference-windows`, which controls the separate saved
rollout reference. Generated latent particles are never trimmed or rejected.
Hold the target, horizons, cutoff, window seed, and window cap fixed when
comparing particle counts; NLL estimates remain finite-particle estimates.
The current one-step loss sums the joint velocity-coordinate NLL, so the
mixture loss follows the same convention and is not divided by dimension after
`logsumexp`.

To inspect finite-particle error directly on a trained checkpoint, run:

```bash
python -m modelling.neural_sde.multistep_particle_diagnostic \
  --checkpoint runs/comparison/velocity_lag1_multistep_h2_5_10_20_30_modes20_15_10_3_3_damped_bounded_diffusion_0p5_optimisierung_new2/best.pt \
  --split validation --max-windows 10000 --window-seed 0 \
  --seeds 0 1 2 3 4 \
  --output runs/comparison/velocity_lag1_multistep_h2_5_10_20_30_modes20_15_10_3_3_damped_bounded_diffusion_0p5_optimisierung_new2/particle_diagnostic
```

By default this inherits the checkpoint's target, horizons, marginal POD-mode
counts, particle schedule, lag, and saved training-trim rule. For every fixed
window, horizon, and independent particle seed it records the log likelihood,
ESS, ESS fraction, plug-in squared coefficient of variation, estimated relative
standard error, and largest normalized component contribution. Aggregate and
seed-to-seed summaries are written to `metrics.json`; the complete per-window
arrays, current states, and targets are written to
`per_window_diagnostics.npz`. Use `--particles M` for a fixed count at every
horizon, or `--particle-counts ...` for an explicit nonincreasing schedule.
When scoring a POD marginal, weights are evaluated from that marginal Gaussian;
omitted coordinates are not conditioned on their observed values.

The training-relevant companion diagnostic compares frozen-model gradients on
one fixed retained minibatch while scaling the checkpoint particle schedule:

```bash
python -m modelling.neural_sde.multistep_gradient_diagnostic \
  --checkpoint runs/comparison/velocity_lag1_multistep_h2_5_10_20_30_modes20_15_10_3_3_damped_bounded_diffusion_0p5_optimisierung_new2/best.pt \
  --split validation --windows 256 --microbatch-size 32 --window-seed 0 \
  --particle-multipliers 1 2 4 8 --seeds 0 1 2 \
  --reference-multiplier 8 --reference-seeds 100 101 102 \
  --device auto \
  --output runs/comparison/velocity_lag1_multistep_h2_5_10_20_30_modes20_15_10_3_3_damped_bounded_diffusion_0p5_optimisierung_new2/gradient_diagnostic
```

It inherits `multistep_objective` from the checkpoint, with `gmm_nll` as the
legacy fallback. Pass `--objective energy_score` or `--objective gmm_nll` for
an explicit frozen-checkpoint counterfactual. Energy-score mode uses the same
unbiased distinct-particle-pair estimator as training and requires velocity
targets with at least two paths per horizon. The diagnostic reproduces the
training aggregation (the mean of per-horizon losses), without the scalar
multistep weight or one-step term, and performs no optimizer step.
`metrics.json` records `objective_name` and reports cosine similarity, norm ratio, relative
L2 error to an independent high-particle seed-average, seed-to-seed cosine, and
noise-to-signal ratio for all parameters and separately for drift and
diffusion. It also writes the main comparison table to `quick_summary.md`. The
reference remains a finite-particle approximation, not an exact
infinite-particle gradient. Raw gradient norms also depend on the model
parameterization. `--microbatch-size` accumulates the exact gradient of the
fixed window-mean objective in memory-bounded chunks; use the checkpoint batch
size for `--windows` when testing training-time gradient averaging. Structured
drift models additionally report the combined structured block and separate
stiffness, damping, mass (when present), and residual-MLP gradient groups.

For sparse full-batch sampling, separately measure particle noise within a selected
batch and window-selection noise between independently shuffled batches:

```bash
python -m modelling.neural_sde.multistep_batch_fraction_diagnostic \
  --checkpoint RUN/best.pt --split train --windows-per-batch 256 \
  --microbatch-size 32 --window-seeds 0 1 2 3 4 5 \
  --particle-seeds 100 101 102 --device auto \
  --output RUN/batch_fraction_diagnostic
```

The report maps candidate batch fractions to the expected number of scored batches,
the selected-update scale `1/f`, and the effective coefficient
`multistep_weight/f`. Its epoch-noise projection uses independent-batch
`1/sqrt(K)` scaling; it does not simulate Adam's response to rare rescaled updates.

All requested endpoints come from one contiguous window within one repetition.
When `--train-trim-percent` is active, a multistep example is retained only if
every native increment from its required prehistory through the farthest
endpoint passes the existing cutoff. `--multistep-weight 0` (the default)
skips multistep window construction and random sampling, preserving the legacy
one-step training path. Multistep likelihood is intentionally restricted to
`--state-variable velocity`.

The 5% training cutoff is an upper-tail increment-amplitude heuristic, not a
measurement of how many samples contain discontinuities. It can remove real
large increments and retain smaller discontinuities. The saved cutoff and
pre-trim scale are reused for validation and evaluation; percentiles are not
refitted on those splits. Whole-window removal generally exceeds 5% and grows
with horizon. Training records the removed-window counts and retained windows
scored; likelihood evaluation records its trim rule, cutoff, and window count.
Comparisons characterize this retained population, not guaranteed clean data.
Trimming remains controlled by `--train-trim-percent`; zero means untrimmed.

An independent rollout-mean objective can match generated and measured endpoint
means without fitting another high-dimensional density:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 \
  --train-trim-percent 5 \
  --stationary-mean-weight 0.01 --stationary-variance-weight 0.001 \
  --stationary-mean-horizons 100 200 500 \
  --stationary-mean-trajectories 30 \
  --stationary-mean-batch-fraction 0.05 \
  --epochs 5 --batch-size 256 --output runs/velocity_stationary_mean
```

On a selected batch, up to 30 observed starts each produce two trajectories
with independent noise. The trajectories are generated once through the maximum
horizon, and their states are retained at every configured horizon. Matching
measured endpoints are read at the same horizons. The mean and variance losses
are averaged equally across horizons, so adding horizons does not multiply their
configured objective weights. Each ensemble takes at most one window from a
shuffled source block, and every increment through the maximum endpoint must
pass the saved training cutoff. The legacy `--stationary-mean-horizon` option
still selects one horizon.

Let `mu_A`, `mu_B`, and `mu_D` be the endpoint means of the two generated
branches and measured data. After scaling their differences by the training
state standard deviations, the loss is the coordinate mean of
`(mu_A - mu_D) * (mu_B - mu_D)`. Independent branch noise makes this unbiased
over rollout noise for the squared generated conditional batch-mean error. The
finite-batch value can be negative. All model coordinates, including the
optional high-pass spatial-mean coordinate, participate.

The optional diagonal-variance term applies the same construction to unbiased
sample variances in normalized coordinates. If `s_A^2`, `s_B^2`, and `s_D^2`
are the coordinatewise variances computed with denominator `N-1`, its loss is
the coordinate mean of `(s_A^2 - s_D^2) * (s_B^2 - s_D^2)`. Use
`--stationary-variance-weight` to enable it. Mean-loss gradients are restricted
to drift parameters; variance-loss gradients reach every trainable drift and
diffusion parameter.

By default, the variance reduction includes every modeled coordinate. Set
`--stationary-variance-pod-mode-count N` to restrict it to the spatial-mean
coordinate, when present, plus the first `N` shared-POD modes. `N=0` therefore
selects only the spatial mean and is invalid for data prepared without that
coordinate. The full per-coordinate variance-error vector remains logged, and
the selected-coordinate RMSE is logged separately. Omitting the selector
preserves the legacy all-coordinate objective exactly.

The optional lagged-covariance term uses the same starts, measured endpoints,
generated paths, and horizons. For each selected leading shared-POD mode it
centers the observed starts and endpoints within the retained ensemble, computes
the `N-1` start-to-endpoint sample covariance, and divides the generated-minus-
measured error by the observed start variance plus
`--stationary-lagged-covariance-epsilon`. The spatial-mean coordinate is always
excluded. Two independently noised generated covariance errors are multiplied,
as for the endpoint moment losses, and then averaged equally over selected modes
and horizons. Enable it with `--stationary-lagged-covariance-weight`; restrict it
to the first modes with `--stationary-lagged-covariance-mode-count`. Its gradients
are routed only to trainable drift parameters. For example, a targeted POD-1--3
calibration is:

```bash
python -m modelling.neural_sde train \
  --checkpoint runs/comparison/0p10_damped_residual_orthogonal_adjusted_w1/best.pt \
  --lag 1 --train-trim-percent 5 \
  --stationary-variance-weight 0.001 \
  --stationary-lagged-covariance-weight 0.01 \
  --stationary-lagged-covariance-mode-count 3 \
  --stationary-mean-horizons 25 50 100 150 200 \
  --stationary-mean-trajectories 30 \
  --stationary-mean-batch-fraction 0.05 \
  --learning-rate 0.0001 --epochs 5 \
  --output runs/comparison/0p10_lagged_covariance_finetune
```

This is a transition autocovariance calibration from observed starts, not an
estimate from an independently equilibrated free-running model. Use a new output
directory and compare against a variance-only fine-tune with the same checkpoint,
sampler, horizons, seeds, and training budget.

The default single horizon is 100, trajectory count 30, and selected-batch
fraction 0.05. Selected updates use inverse-fraction correction, so
`--stationary-mean-weight` denotes the expected objective weight rather than an
active-batch coefficient. The stochastic paths use the current diffusion, but
the mean term adds gradients only to drift parameters, while the variance term
adds gradients to trainable drift and diffusion parameters. The one-step and
multistep likelihoods continue training both normally. Training logs the
raw moment losses, their realized weighted contributions, sampled
trajectory/batch counts, and aggregate normalized generated-minus-measured
moment errors and RMS values. Validation uses the fixed
`--stationary-mean-validation-seed`.

The same lag can be fine-tuned from a joint checkpoint. The checkpoint supplies
the data split, architecture, normalization, and initial parameters; use a new
output directory:

```bash
python -m modelling.neural_sde train \
  --checkpoint runs/comparison/velocity_lag1 --lag 1 \
  --train-trim-percent 5 \
  --stationary-mean-weight 0.01 --stationary-variance-weight 0.001 \
  --learning-rate 0.0001 --epochs 5 \
  --output runs/comparison/velocity_lag1_endpoint_moments_finetune
```

### Optional TCN replacement for the nonlinear drift

`--residual-architecture tcn` replaces the nonlinear MLP, rather than adding
a correction to it. The default remains `mlp`, including for old checkpoints.
This option supports native-lag velocity models with either drift structure:

- `--drift-type damped_residual`: normalized drift is
  `-K z_bar - Gamma v_bar + TCN(history)`. The TCN output layer starts at zero;
  the restoring/damping operators and TCN train jointly. The existing
  `--no-structured-damping` and diagonal/SPSD operator choices still apply.
- `--drift-type mlp --residual-architecture tcn`: the entire learned drift is
  `TCN(history)`, with no explicit restoring/damping operators. Here `mlp` is
  the legacy name for the unstructured drift family, not the selected network.

For example, a fresh **one-step-only** structured model:

```bash
python -m modelling.neural_sde train \
  --experiment 0p10 --rank 20 --state-variable velocity --lag 1 \
  --drift-type damped_residual --residual-architecture tcn \
  --tcn-context-frames 63 --tcn-channels 64 \
  --tcn-kernel-size 3 --tcn-dilations 1 2 4 8 16 \
  --diffusion-type full --train-trim-percent 5 --multistep-weight 0 \
  --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p10_r20_tcn63_one_step
```

Use `--drift-type mlp` and a separate output directory for the unstructured arm.
Keep preprocessing, split, trim, diffusion and training options matched to the
comparison run; the example is not a reproduction of an existing checkpoint.

The input is a dense, oldest-to-newest sequence of separately normalized
`[position_s, position_s - position_(s-1)]` tokens. `--tcn-context-frames H`
includes the current token and H-1 earlier tokens, and therefore requires **H
preceding positions** to calculate the oldest velocity. For current through
50 frames back use H=51. Defaults use five residual blocks, each containing
one causal left-padded convolution with kernel size 3 and dilation 1,2,4,8,16;
the receptive field is `1 + (kernel_size - 1) * sum(dilations) = 63` tokens.
The last output token predicts normalized velocity drift. Context cannot exceed
this receptive field; increase dilations for longer contexts. No future samples
or temporal pooling enter the prediction.

`history_steps` is expanded automatically for data windows, full-stencil trimming,
and rollout history. Existing `--history`/`--history-offsets` still select the
flat conditioning taps for state-dependent **diffusion**, independently of the
dense TCN input; their default is one frame in velocity mode. `--hidden-layers`
still controls that diffusion MLP, while `--tcn-channels` controls drift width.
The likelihood, diffusion parameterization and Euler--Maruyama update are unchanged.
One-step, multistep, stationary-moment and structured orthogonality objectives
use the same TCN drift, and all architecture settings are saved in checkpoints.
Changing MLP to TCN requires a fresh run, not an MLP warm start. The controller
treats these options as new-model architecture choices; it is not resumed
automatically by enabling this feature.

For one-step innovation checks use
`data_analysis/correlations/training_innovation_diagnostic.py`; it honors the full
saved history and trim stencil. The older one-history-frame drift/Jacobian audit
does not support TCN. TCN cannot currently be combined with capillary-energy
drift, non-native lag, or the separate additive finite/oscillator/innovation
memory extensions. Longer history changes which trimmed windows survive: for
a strictly matched architecture comparison, give the MLP the same full history
length and report retained populations as well as NLL/innovation correlations.

To impose positive diagonal linear restoring and damping terms while retaining
a nonlinear residual, use the structured velocity drift at native lag 1:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 \
  --drift-type damped_residual --stiffness-init 0.01 --damping-init 0.01 \
  --train-trim-percent 5 --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p20_r20_damped_residual
```

For normalized position `z_bar` and native-frame velocity `v_bar`, its drift is

```text
-diag(k) z_bar - diag(gamma) v_bar + r_theta(z_bar, v_bar).
```

Both diagonals use a softplus parameterization and therefore remain positive.
The residual MLP has the selected hidden layers and activation, and its final
layer starts at zero so initial drift is purely linear. The saved `k` and
`gamma` values act in separately standardized coordinates; they are not
physical stiffness or damping constants without undoing those scalings. The
Euler-Maruyama likelihood and selected diffusion model are otherwise unchanged.
The structured form requires velocity state and at least its first native-frame
prehistory. Lag 1 is recommended because its position update is then exactly
consistent with the measured native-frame velocity definition, and it is
required when supplying any additional history.

For a restoring-only ablation, add `--no-structured-damping`. The drift then is
`-diag(k) z_bar + r_theta(z_bar, v_bar)` and contains no damping parameter. A
joint damped-residual checkpoint can be used with `--checkpoint`; all matching
weights are loaded and only its explicit damping parameter is deliberately
dropped. The choice is stored as `structured_damping: false` in the checkpoint.

To make the diagonal structured terms identifiable against the residual, add
`--restoring-orthogonality-weight LAMBDA_Z` and, when explicit damping is
enabled, `--damping-orthogonality-weight LAMBDA_V`. For each minibatch these add

```text
L_restoring = sum_i mean(z_tilde_i * r_theta_i)^2
              / (mean(z_tilde_i^2) + epsilon),
L_damping = sum_i mean(v_tilde_i * r_theta_i)^2
            / (mean(v_tilde_i^2) + epsilon).
```

With epsilon zero, under the empirical inner product this is exactly the squared
norm of the residual's projections onto the vector-field subspaces spanned by
the coordinatewise fields `e_i z_tilde_i` and `e_i v_tilde_i`; positive epsilon
is the denominator-stable version. These are projection energies, not
cosine-normalized correlations: residual amplitude still matters. All normalized
coordinates owned by the diagonal operators are included, including the
spatial-mean coordinate when present. The two weights are independent, and zero
disables the corresponding term. The stochastic minibatch estimates have the
usual finite-batch noise (and squared sample cross-moments are upward biased
relative to the full-population squares), so compare more than one weight and
seed.

Position-increment correlation can make either marginal penalty confound
restoring and damping behavior. Add
`--restoring-orthogonality-adjust-velocity` (the legacy option name is retained)
to fit both conditional regressions once on the retained normalized training
population:

```text
A = Cov(z_bar, v_bar) pinv(Cov(v_bar, v_bar)),
B = Cov(v_bar, z_bar) pinv(Cov(z_bar, z_bar)),
z_tilde = (z_bar - mean(z_bar)) - A (v_bar - mean(v_bar)),
v_tilde = (v_bar - mean(v_bar)) - B (z_bar - mean(z_bar)).
```

The restoring penalty uses `z_tilde`, while the damping penalty uses `v_tilde`.
Consequently, a diagonal residual term `-a_i z_bar_i - b_i v_bar_i` contributes
only `a_i` to the adjusted restoring cross-moment and only `b_i` to the adjusted
damping cross-moment. The model itself continues to see the original position
and velocity. Both frozen matrices use every coordinate of the conditioning
variable, are reused unchanged for validation, and are stored with both means,
covariance ranks/condition numbers, conditional variances, explained fractions,
and residual cross-correlations. This is a symmetric linear conditional
interpretation; it does not forbid nonlinear dependence or establish physical
stiffness or damping.

Both penalties require `damped_residual` and the diagonal operator. The damping
penalty additionally requires explicit structured damping (the default), while
the restoring penalty also remains available with `--no-structured-damping`.
When both weights are zero no extra forward pass is performed. The configuration
and separate raw/weighted train and validation terms are saved, together with
per-coordinate projection energies and coefficients. For example, to isolate
both learned diagonal operators from the residual:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 --history 1 \
  --hidden-layers 64 64 --activation silu \
  --drift-type damped_residual --damped-operator diagonal \
  --restoring-orthogonality-weight 1 --damping-orthogonality-weight 1 \
  --restoring-orthogonality-adjust-velocity \
  --diffusion-type diagonal --train-trim-percent 5 \
  --learning-rate 0.001 --epochs 5 --batch-size 256 --seed 0 \
  --output runs/velocity_lag1_damped_residual_orthogonal_adjusted_w1
```

A simpler joint alternative is
`--diagonal-linear-orthogonality-weight LAMBDA`. For each output coordinate it
asks how much of the residual is explained jointly by that coordinate's
position and velocity. Although both inputs are already standardized with
retained-training means and standard deviations, the finite minibatch values
and residual are centered before forming

```text
x_i = [z_bar_i, v_bar_i],
C_i = mean(x_i x_i^T),
b_i = mean(x_i r_theta_i),
L_linear_orth = sum_i b_i^T (C_i + epsilon I)^-1 b_i.
```

Thus the loss is the ridge-stabilized squared magnitude of the minibatch
least-squares projection onto `span(z_bar_i, v_bar_i)`. Centering makes it
insensitive to a residual intercept; it does not replace or refit the saved
training normalization. The fitted position and velocity coefficients are
logged separately for every coordinate. This option requires explicit
structured diagonal damping and is intentionally mutually exclusive with the
legacy restoring/damping orthogonality options. Its covariance ridge defaults
to `1e-6` and can be changed with
`--diagonal-linear-orthogonality-epsilon`.

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 --history 1 \
  --hidden-layers 64 64 --activation silu \
  --drift-type damped_residual --damped-operator diagonal \
  --diagonal-linear-orthogonality-weight 1 \
  --diffusion-type diagonal --train-trim-percent 5 \
  --learning-rate 0.001 --epochs 5 --batch-size 256 --seed 0 \
  --output runs/velocity_lag1_damped_residual_joint_orthogonal_w1
```

Use `--damped-operator spsd` to replace each positive diagonal with a learned
full symmetric positive-semidefinite operator:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 \
  --drift-type damped_residual --damped-operator spsd \
  --stiffness-init 0.01 --damping-init 0.01 \
  --train-trim-percent 5 --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p20_r20_damped_residual_spsd
```

The matrices use `K = L_K L_K^T` and `Gamma = L_Gamma L_Gamma^T` with learned
lower-triangular factors. They initialize to `stiffness_init * I` and
`damping_init * I`, respectively. Evaluation saves both complete matrices and
their eigenvalues in `metrics.json`. This parameterization needs a fresh run;
it cannot warm-start from a diagonal structured-drift checkpoint.

The third drift option fixes the restoring geometry from quadratic capillary
energy while retaining learned inertia, SPSD damping, and the same residual:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --state-variable velocity --lag 1 \
  --drift-type capillary_energy --surface-tension 0.0728 \
  --stiffness-init 0.01 --damping-init 0.01 \
  --train-trim-percent 5 --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p20_r20_capillary_energy
```

For POD modes `U`, the physical matrix `K_gamma` is the surface-tension-weighted
gradient Gram matrix, using the same second-order finite differences and 2-D
trapezoidal quadrature as the capillary-energy diagnostics. If
`delta_a = S q` converts standardized POD displacements `q` about the training
mean to coefficient displacements in metres, the fixed model matrix is
`Q = S K_gamma S`. It is constructed after
the final training normalization is fitted and is stored in the checkpoint.
The spatial-mean coordinate is constant in space, so its row and column in `Q`
are exactly zero. That coordinate instead receives its own learned positive
restoring coefficient, initialized by `--stiffness-init`; it is not left as a
coordinate without a restoring force.

The effective mass is positive diagonal. Internally its trainable positive
multiplier is scaled by `diag(Q)` (and by a representative POD energy scale for
the mean coordinate), which avoids poorly scaled raw parameters. Damping uses
`Gamma = D L L^T D`, with `D` the square root of that same fixed energy scale,
so it remains SPSD. The structured acceleration is evaluated with a linear
solve against diagonal `M`; no inverse is formed. The initialization makes the
diagonal ratios `Q_ii/M_ii` and `Gamma_ii/M_ii` approximately
`stiffness_init` and `damping_init`, respectively. The nonlinear residual MLP
is unchanged and starts at zero.

This variant follows the existing velocity convention exactly: `v` is the
separately standardized native-frame displacement `a[n]-a[n-1]`, not a
derivative in SI seconds. Consequently its time unit is one native frame; no
extra `dt` factor is inserted into `Q`, `M`, or `Gamma`. The existing NLL and
rollout apply the selected lag as `h` (and still report physical timestamps
using the source `dt`). Diffusion architecture, loss, rollout integrator, and
evaluation metrics are unchanged. Evaluation only adds `Q`, learned mass,
damping, their eigenvalues, and the mean stiffness to `metrics.json`.

With `--train-trim-percent`, the cutoff is still fitted from native state
increment norms. A lag-`k` velocity pair is removed if any native increment
from its one-frame prehistory through `n+k` exceeds that cutoff. Validation
NLL uses the same path mask. Strict reference-window filtering likewise checks
all native increments, including the initialization interval.

For the state SDE, `--history H` (default `0`) conditions the drift on the current state and `H`
preceding states, each separated by the chosen lag. For example, `--lag 3
--history 2` uses `[a[t], a[t-3], a[t-6]]` to predict `a[t+3]`. Each block uses
the same training-only state normalization. Constant diffusion remains
independent of history; `state_diagonal` and `bounded_state_diagonal` use the
full history-conditioned input. Rollouts begin with measured prehistory and
then shift in generated states. The first `H * lag` starting positions of each
repetition are excluded.

```bash
python -m modelling.neural_sde train \
  --experiment 0p15 --rank 20 --lag 3 --history 2 \
  --train-trim-percent 1 --epochs 20 \
  --output runs/0p15_r20_sde_lag3_h2
```

Use `--history-offsets` instead of `--history` for a sparse multiscale delay
embedding. Offsets are measured in model transitions; velocity models with
additional history require `--lag 1`, so those offsets are also native frames:

```bash
python -m modelling.neural_sde train \
  --experiment 0p20 --rank 20 --lag 1 --state-variable increment \
  --history-offsets 1 2 3 5 10 \
  --diffusion-type bounded_state_diagonal --diffusion-log-range 1 \
  --train-trim-percent 3 --epochs 20 --batch-size 256 \
  --output runs/neural_sde_0p20_r20_incr_sparse_history
```

For an increment model, offset 1 is required and is encoded as the current
increment rather than duplicated as a raw state. The example therefore feeds
information equivalent to `[a[t], delta_a[t], a[t-2], a[t-3], a[t-5],
a[t-10]]`. The rollout retains every intermediate state through the largest
offset, while only the selected taps enter the networks. With `--lag 3`, the
same offsets refer to native frames `t-3`, `t-6`, `t-9`, `t-15`, and `t-30`.
Existing `--history H` configurations remain equivalent to contiguous offsets
`1 ... H`. The two history options are mutually exclusive.

Existing no-history checkpoints still load. A history-conditioned run must
start fresh or warm-start from a checkpoint with the same history count;
changing the count changes the drift network's input width. If evaluation
overrides `--rollout-lag`, both the model step and history spacing change, so
that result extrapolates beyond the spacing used in training.

Evaluation normally uses the checkpoint's training lag. Add
`--rollout-lag 1` to integrate at the native timestep and select matching
native-step reference frames, even if the model was trained at a larger lag.
The checkpoint retains its training lag; `metrics.json` records both
`training_lag_steps` and the rollout `lag_steps`. This is a finer Euler–Maruyama
integration of the learned coefficients; accuracy at the finer step must be
checked against data because training only constrained coarse transitions.
When scoring is enabled, increment statistics for metric normalization are
refitted at the rollout lag; `--no-metrics` skips that extra pass.

Use `--diffusion-scale` for a rollout-only noise-amplitude diagnostic without
changing the checkpoint. For example:

```bash
python -m modelling.neural_sde evaluate \
  --checkpoint runs/neural_sde_0p20_r20_incr_sparse_history/best.pt \
  --diffusion-scale 0.9 --horizon 500 \
  --output runs/neural_sde_0p20_r20_incr_sparse_history/rollout_diffusion_0p9
```

This multiplies the trained diffusion factor `G` by `0.9` at every generated
step, so the injected covariance is multiplied by `0.9^2`. It does not alter
the drift, checkpoint, normalization, reference windows, or trained diffusion
parameters. The default is `1`; `0` produces a deterministic drift-only
rollout. `metrics.json` records the applied `diffusion_scale`.

For two-stage training, use distinct output directories:

```bash
python -m modelling.neural_sde train \
  --experiment 0p30 --rank 50 --train-mode diffusion --lag 1 \
  --diffusion-trim-percent 5 --epochs 20 \
  --output runs/0p30_r50_sde_diffusion_lag1

python -m modelling.neural_sde train \
  --train-mode drift --lag 10 \
  --checkpoint runs/0p30_r50_sde_diffusion_lag1/best.pt \
  --epochs 20 --output runs/0p30_r50_sde_drift_lag10
```

Diffusion mode freezes and zeroes the untrained drift, and logs the full
validation increment covariance comparison at every epoch. For
`state_diagonal` and `bounded_state_diagonal`, the learned side of this comparison
is the validation mean of the conditional covariance `G(x) G(x)^T`. The optional trim
removes only the largest `p` percent of **validation** increment vector norms;
it does not filter training pairs. By default the Frobenius comparison uses
checkpoint-normalized coordinates (`--diffusion-covariance-space physical`
switches both matrices to physical shared-POD units). It logs the relative
error, both traces, removed count and percentage, and removed squared-L2 energy
fraction. Drift mode reloads the Stage 1 checkpoint, freezes diffusion, keeps
its state normalization, and refreshes lag-dependent increment statistics used
by rollout diagnostics. Joint training still starts fresh when no checkpoint is supplied.

Drift-only training can also start without a checkpoint. In that case its
diffusion is fixed at `--diffusion-init` (default `0.1`, a standard deviation
per native step in standardized coordinates), while only the drift MLP is
fitted. With `--diffusion-type full`, the fixed initial factor is still
`diffusion_init * I`:

```bash
python -m modelling.neural_sde train \
  --experiment 0p15 --rank 20 --lag 3 --train-mode drift \
  --diffusion-init 0.1 --train-trim-percent 1 --epochs 20 \
  --output runs/0p15_r20_sde_lag_3_trim1_drift
```

The fixed diffusion also controls noise in later stochastic rollouts, so its
chosen value matters even though it is not optimized in this mode.

To refine a joint model at a larger lag, pass its run directory (or `best.pt`)
as `--checkpoint` while keeping the default `--train-mode joint`:

```bash
python -m modelling.neural_sde train \
  --experiment 0p15 --rank 20 --lag 3 \
  --checkpoint runs/0p15_r20_sde_lag_2_trim1 \
  --train-trim-percent 1 --epochs 20 \
  --output runs/0p15_r20_sde_lag_3_trim1
```

The checkpoint must be from joint training at a smaller lag with the same
source, split, and architecture. Both drift and diffusion weights are loaded
and trained; the optimizer starts fresh. Without training trim, state
normalization is preserved and lag-dependent increment statistics are
refreshed. With `--train-trim-percent`, the loaded state scale first defines
the frozen trim mask and normalization is then refitted on its retained
transitions. The new run records the source checkpoint and selects its own
`best.pt` by validation NLL.

Use `--train-trim-percent 1` to compute a cutoff from the largest 1% of
**training** increment vector norms and exclude pairs above that cutoff from
the NLL in any training mode. The threshold is computed once from the training
repetitions at the selected lag, using
increments divided by the training state standard deviations (the same norm
as the default validation covariance trim). It is applied to every training
epoch and to validation NLL, so best-checkpoint selection uses the same
retained-pair criterion. The saved `config.json` and checkpoint record the
cutoff and excluded counts for both splits. Normalization uses two passes: the
full-training state scale defines and freezes the percentile mask, then state
and target statistics are refitted from the retained training transitions.
The preliminary state scale is saved with the checkpoint so training,
validation, and optional reference-window filtering all apply the same mask.
This option is independent of `--diffusion-trim-percent`, which only changes
the validation covariance comparison.

Evaluation uses the EDM reference-window selector and stochastic rollout
metrics. By default, even a training-trimmed checkpoint permits reference
windows containing excluded increments so that long contiguous rollouts remain
possible; `metrics.json` and plot summaries record this with a caution. Pass
`--enforce-train-trim-on-reference-windows` to instead reject any candidate
reference window containing a transition above the saved training cutoff.
Whole windows are rejected to preserve uniformly sampled signals for Welch PSD;
generated trajectories are never trimmed. When enforcement is active, the
retained-window count and cutoff are saved in `metrics.json` and propagated to
plot summaries. Evaluation saves physical-coordinate `rollout.npz` and `metrics.json`
in the same schema, so the existing `data_analysis.energy.rollout_energy` and
`data_analysis.psd.rollout_psd` commands can read them. To compare with an EDM
run, use the same experiment, rank, spatial-mean setting, history count, split,
condition count, ensemble size, horizon, and evaluation seed. Exact reference-window
matching also requires EDM to use native lag 1 without history conditioning;
history-conditioned EDM excludes early starts when selecting windows. For a
POD-only run, add `--no-metrics` during evaluation, as required by the shared
metric code. The shared POD basis may have included held-out repetitions;
those repetitions remain held out from SDE parameter and normalization fitting.

Generate energy, PSD, marginal-statistics diagnostics, and selected coordinate
trajectories together with:

```bash
python -m data_analysis.plot_rollout runs/0p20_r50_neural_sde/evaluation
```

Plots are saved under `energy/`, `modal/`, `psd/`, and `coordinates/`. A
machine-readable `quantitative/summary.json` additionally records quadratic
energy, reduced-coordinate mean/covariance errors (all coordinates, POD
fluctuations, and spatial mean separately), and Welch band-power errors in
half-decade frequency bands. For a training-trimmed checkpoint, energy and
moments are reported both against the full reference and against reference
frames retained by the checkpoint's saved normalized-increment cutoff and
frozen pre-trim state scale. Generated trajectories are never trimmed. PSDs are
also never frame-trimmed because Welch requires uniformly sampled trajectories.
The modal plot likewise uses the training-trimmed reference as its primary
amplitude and energy comparison and overlays the untrimmed RMS ratios; its JSON
and NPZ outputs retain both populations explicitly.
Use `--overwrite` to regenerate all outputs. Every run also writes
`quantitative/quick_summary.md`, a small table containing only the central
energy, moment, and PSD results.

The same command also writes `psd_peaks/psd_peaks.png`, `summary.json`,
`quick_summary.md`, and per-path spectra in `psd.npz`. This center-point diagnostic
includes the spatial mean when available and zooms to 100–1500 Hz. Its independent
`--peak-nperseg` defaults to 8192; `--peak-band LOW HIGH` and
`--peak-prominence-db` (default 2 dB) control the reference peak selection.
The summary reports mean absolute peak-contrast error in dB (lower is better),
generated/reference band power, and the number of reference peaks. Reference
peak and neighboring-valley bins are frozen and used for both spectra; contrast
uses a frequency-interpolated baseline in dB and is invariant to overall amplitude.
Reference windows are the saved rollout references, without additional trimming
or training-data pooling. Results are also included in the central quantitative
JSON/Markdown summaries. Short records still get a plot and band power, but no
comparable contrast score when the requested segment length cannot fit. A reference
without clear peaks produces an N/A score. Extra generated peaks outside the
reference locations are not directly scored, and this is not a phase-coupling test.
For POD-only inputs, the signal is explicitly labeled center without spatial mean.

To create only this diagnostic for an existing rollout:

```bash
python -m data_analysis.psd.rollout_spectral_peaks ROLLOUT_DIR
```

## Optional discrete Lyapunov regularizer

Velocity training supports a fixed quadratic penalty with either one-step NLL
or the existing one-step-plus-multistep objective. For example:

```bash
python -m modelling.neural_sde train \
  --experiment 0p10 --rank 20 --state-variable velocity --lag 1 \
  --drift-type damped_residual --damped-operator spsd --diffusion-type full \
  --stability-weight 0.1 --stability-a 0.01 --stability-b 0.01 \
  --stability-metric auto --stability-perturb-scale 0.2 \
  --stability-generated-steps 2 --stability-validation-seed 0 \
  --epochs 20 --output runs/neural_sde_0p10_r20_stability
```

These are example hyperparameters, not validated settings for this experiment.
Add the existing `--multistep-weight`, `--multistep-horizons`, and
`--multistep-particles` flags to combine objectives. `--stability-weight 0`
(the default) bypasses all metric fitting, state sampling, and penalty work;
it leaves existing losses, RNG consumption, metrics, and checkpoint schema
unchanged. State and legacy-increment modes reject a positive stability weight.
Python callers pass a frozen `StabilityConfig` from `modelling.neural_sde.stability`
as `train(..., stability=StabilityConfig(...))`.

The metric is **uncentered** in physical shared-POD coordinates:
`s = (z, v)`, `V(s) = s.T P s`. Here `v = z[n]-z[n-1]` is displacement per
native frame, and the model step is `h=lag_steps`, **not** `lag_steps*dt`.
The mean offsets used inside the drift/diffusion networks are still included
in the actual conditional mean. No centering is silently applied to V; when
the data have a large nonzero mean, that mean contributes to V and its scale.

The shared velocity helper supplies `m = v + h*f(z,v,history)` and
`L = sqrt(h)*G(z,v,history)`. Since rollout updates position with the **new**
velocity, its augmented mean and covariance are

```text
mu = [z + h*m; m],      H = [h*L; L],      C = L L^T,
Sigma = H H^T = [[h^2*C, h*C], [h*C, C]].
Delta V = mu^T P mu + tr(P Sigma) - s^T P s.
L_stab = mean(relu(Delta V - a + b*V(s))^2).
L_total = L_existing + weight*L_stab.
```

The expectation is analytic, including both position--velocity covariance
blocks. Sigma is rank deficient because the same noise drives position and
velocity; no augmented Gaussian likelihood or covariance inversion is used.
The trace is evaluated as `tr(H^T P H)`. Drift and diffusion both receive
gradients. The implementation computes the quadratic and trace in float64
using fixed training scales for conditioning.

For the damped-residual core, let `K,C` be its initial stiffness and damping
matrices, `Sz=diag(state_std)`, `Sv=diag(target_std)`, and `R=Sz^-1 Sv`.
In `s_scaled = (Sz^-1 z, Sv^-1 v)`, the homogeneous discrete core is

```text
F_scaled = [[I - h^2 R K, h R (I - h C)],
            [    -h K,       I - h C ]].
```

In physical coordinates, with `D=diag(Sz,Sv)`, `F=D F_scaled D^-1`.
The implementation checks `max(abs(eigenvalues(F_scaled))) < 1`; positive
stiffness and damping alone do **not** establish discrete stability. When
available, it solves `F_scaled.T P_scaled F_scaled - P_scaled = -I`, verifies
positive definiteness and the equation residual, then sets
`P = D^-T P_scaled D^-1 / mean_training_V`. Thus the corresponding physical
`Q = D^-T D^-1 / mean_training_V` is positive definite. Both P and Q are
scaled together. The retained training starting states have mean V approximately
one, using the same lag, multistep-window eligibility, and trim mask as training.
The initial homogeneous core excludes the network residual and the affine
normalization offset; the **penalty includes both** through the actual update.

`--stability-metric auto` uses this construction when possible and otherwise
warns and records a fallback reason. `lyapunov` fails if the construction is
unavailable. `data_scaled` explicitly uses `P_scaled=I` before the same mean-V
rescaling: it is an **empirical regularizer**, without a reference Lyapunov
equation. All model types supported in velocity mode can use this fallback.
Metric fitting occurs after normalization, trimming, warm-start loading, and
any diffusion-stage drift zeroing. A warm-started run constructs its own fixed
reference at its new lag. P is never recomputed as the core learns; a, b, and P
are fixed for the entire run and are not optimizer parameters. Require finite
`a>=0`, `0<b<1`, and a finite nonnegative weight.

Every batch penalizes its observed starting states. Optional sampling adds:

- `--stability-perturb-scale p`: one perturbed copy with independent position
  and velocity perturbations of standard deviation `p*state_std` and
  `p*target_std`. History at native offset j receives `delta_z-j*delta_v`,
  preserving the causal velocity and the full sparse-history input layout.
- `--stability-generated-steps k`: each of k successive stochastic states
  generated from each observed state, using the actual velocity distribution,
  position update, and history shift. Velocity is carried explicitly at lag>1.
  Generated sampling locations are detached; gradients flow through the local
  conditional drift and diffusion, without differentiating the sampling path.

All included states have equal weight: an observed state, an optional perturbed
copy, and each generated step. Generation starts from observed states; optional
perturbations are a separate group. Randomness selects evaluation locations;
it does **not** estimate the conditional expectation. Sampling uses a separate
generator from multistep NLL, with training seed `seed+epoch` and a fixed
`--stability-validation-seed` each validation epoch. Reproducibility assumes the
same batch size and ordering.

`config.json` and both checkpoints store `training.stability`: the settings,
physical P, scaled metric, scales, initial F/eigenvalue diagnostic when
applicable, training-state count, and fallback reason when applicable.
`metrics.jsonl` logs train/validation `stability_penalty`,
`stability_weighted_penalty`, `stability_violation_fraction`, `stability_mean_v`,
`stability_mean_delta_v`, `stability_mean_excess`, and `stability_max_excess`.
Diagnostics are also separated into observed, perturbed, and generated groups.
Violation means `Delta V-a+bV>0`; maxima are maxima over the entire epoch.
Validation uses the frozen training metric and scales. `train_nll` and
`validation_nll` remain the one-step NLL; total-loss fields include all enabled
terms, and total validation loss selects `best.pt`.

These sampled penalties **encourage stability but do not prove a global
bound**. A valid Lyapunov equation applies only to the frozen homogeneous
reference core. The nonlinear learned model, affine offsets, diffusion, and
unsampled states can violate the inequality. With older history conditioning,
the expectation is conditional on the complete supplied history but V omits
that history, so this is not a full-state stability guarantee. Check long
rollouts, nonfinite/survival behavior, energy, and spectral diagnostics separately.

## Training increment decorrelation

Choose candidate mode-specific horizons from the exact reduced coordinates and
training split saved in a native-lag velocity checkpoint:

```bash
MPLCONFIGDIR=/tmp/capillary-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m data_analysis.correlations.training_pod_decorrelation \
  --checkpoint runs/comparison/velocity_lag1_multistep_2_4_10_20/best.pt \
  --max-lag 256 --thresholds 0.3 0.1 0.01 \
  --consecutive 20 --min-pairs 1000 \
  --output runs/comparison/velocity_lag1_multistep_2_4_10_20/training_increment_decorrelation
```

Coordinate 0 inherits the checkpoint's stored spatial mean (50 Hz high-pass in
this example) and `sqrt(N)` scaling; the remaining coordinates are shared-POD
coefficients. The diagnostic explicitly correlates the model input
`(a[t]-a[t-1]-target_mean)/target_std`. Pearson correlation is invariant to
this per-coordinate affine normalization, but the saved arrays retain the exact
model convention. Every pair lies within one run passing the checkpoint's
saved training trim; gaps are never compressed. The 5% upper-tail cutoff is a
heuristic and is not an estimate of the fraction of discontinuous measurements.

`horizons.csv` reports the first threshold crossing and the first start of 20
consecutive supported lags below each absolute-correlation threshold, separately
by mode and then summarized across training repetitions. These are finite-range
marginal-correlation timescales, not proofs of independence or conditional
unpredictability.

## Anti-aliased temporal downsampling

For a **new** model, add `--downsample-factor 8 --lag 1` to the existing
training command. This prepares a coarser time series before normalization,
velocity/increment construction, training trimming, or history sampling. The
TCN architecture is unchanged. Omitting the option preserves existing behavior.

Every selected coordinate, including the spatial mean, is filtered independently
within each repetition with a centered Kaiser-window FIR (beta 5, `20*q+1` taps,
cutoff at the new Nyquist frequency), then decimated by `q`. This is the
`scipy.signal.resample_poly` filter convention. The default excludes 32 **coarse**
samples from each edge; `--downsample-edge-trim` can change this, with a minimum
of 10 to exclude the filter support. Filtering uses aligned input halos so
batch and cache boundaries do not alter the prepared signal. Source HDF5 files
are read-only; no new HDF5 files are created.

The prepared timestep, frame counts, timestamps, training-only normalization,
and references all use the coarse grid. Saved data configuration and signatures
record the factor, edge removal and filter specification. Evaluation and warm
starts inherit these settings from the checkpoint; changing them in a warm
start is rejected. Start a fresh model when changing the factor.

At 115,200 Hz, factor eight gives 14,400 Hz sampling and a 69.44-microsecond
timestep. TCN63 then covers approximately 4.4 ms. All history, loss horizons,
strides, rollout lengths and PSD segment lengths now count **coarse samples**:
20 ms is 288 steps, and `--nperseg 1024 --peak-nperseg 1024` matches the physical
Welch duration of 8192 native samples. Choose longer training horizons explicitly;
downsampling does not automatically change the loss configuration.

Standard `data_analysis.plot_rollout` plots and metrics use the saved, filtered
reference. PSD comparisons should use the retained passband, away from the
anti-alias filter's transition region. The specialized raw-experiment PSD/video
commands currently reject downsampled rollouts rather than interpreting prepared
indices as raw frame indices. The centered filter defines an offline filtered
process; it is not a causal forecasting preprocessing pipeline for raw data.

### Fixed oscillator memory: inexpensive first-stage correction

`python -m modelling.neural_sde.fit_fixed_memory` freezes an existing native-lag
velocity SDE (including diffusion) and fits only a linear drift correction.
The current fitter supports constant diagonal/full diffusion. It reuses the
20 fitted peak frequencies, their damping rates, and rank-one spatial mixing
directions from the latent oscillator model. It does **not** import its OU
background or claim to infer its physical hidden states.

For standardized native-frame velocity `u[t]` and fixed spatial projection `P`,
the complex memory is

```
rho[k] = exp((-gamma[k] + 2*pi*i*f[k]) * dt)
s[t]   = sqrt(1 - abs(rho)**2) * (u[t] @ P)
h[t]   = sum(j=0..L-1) rho**j * s[t-j]
h[t+1] = rho*h[t] + s[t+1] - rho**L*s[t-L+1]
mean_v[t+1] = parent_mean_v[t+1] + target_std * W * features(h[t])
```

The last subtraction makes this an exact finite-history filter. The default
window spans five slowest damping times; the report records its physical
duration and the corresponding omitted exponential envelope. Features contain
the real/imaginary parts, scaled by their **training-only** RMS. `P` uses the
physical peak mixing directions divided by parent velocity standard deviations,
with unit-normalized columns. This assumes identical ordered POD coordinates;
the fitter checks the oscillator's source basis, rank, mean preprocessing and
training repetitions against the parent. Frequencies and damping describe a
driven damped rotation, rather than the original second-order oscillator's exact
transition matrix.

Each training/validation record is filtered once, independently. Eligible
windows include the full memory and next target in the parent's saved trim
stencil. Sampling is uniform within each record with approximately equal record
counts. Parent predictions are computed once on the sampled windows. Fitting
uses ridge regression of the parent residuals whitened by its frozen diffusion:
this is the one-step Gaussian likelihood, with an L2 penalty on the whitened
readout. Validation chooses the ridge penalty **or the unchanged parent**.
There is no stochastic multistep backpropagation. With 20 filters and 21 POD
coordinates only 840 coefficients are fitted.

Example bounded trial (CPU; no controller campaign):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MPLCONFIGDIR=/tmp/cwt-mpl \
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.fit_fixed_memory \
  --parent runs/comparison/neural_sde_0p10_tcn63_downsample8_energy_score/onestep/best.pt \
  --oscillators runs/diagnostics/latent_oscillators_0p10_two_family_v1/two_family/model.npz \
  --output runs/comparison/neural_sde_0p10_fixed_memory_v1 \
  --train-windows 8192 --validation-windows 2048 --max-seconds 180
```

`--max-seconds` is checked between records and parent-inference batches; it is
not a hard process deadline during data loading or file writes. The command
refuses to overwrite an existing nonempty output directory. Outputs include
standard `best.pt`/`config.json`, `report.json` with source hashes, complete
per-record `features.npz`, and sampled regression arrays/window IDs. These caches
are specific to this frozen parent and filter setup. Generic `train` rejects
this configuration; use the dedicated fitter to preserve the frozen parent.

Standard `python -m modelling.neural_sde evaluate` loads the checkpoint and uses
the same recurrence with a forcing ring buffer. Parent TCN histories stay short;
long memory is not reconstructed at each generated step. The sampling interval
cannot be overridden. Reference records must contain the full conditioning
window plus the requested forecast horizon; for the current factor-eight data,
the default 6,449-frame memory leaves roughly 5,988 frames per record for a
forecast. Initialization uses observed history, not a random physical phase.

This is a cheap test of whether fixed resonant history adds useful information
to the neural drift. Stable filter poles do not guarantee stable closed-loop
generation, and lower one-step validation loss does not establish PSD peak
recovery. Follow a successful fit with matched parent/correction rollouts and
check survival, energy/covariance and PSD before a larger experiment.

### Focused correlation matching

The existing two-branch covariance loss now supports explicit coordinates
(including spatial mean), fixed training-only variance normalization and
correlation lags independent of endpoint mean/variance horizons. Defaults
preserve earlier behavior. The frozen gradient audit supports the parent's
GMM likelihood as well as energy score, with per-lag sensitivities and separate
start-batch/noise-seed variability. See [definition, flags and bounded diagnostic
command](correlation_matching.md). No fine-tuning is launched by that diagnostic.

## Imported resonant operators plus neural residual

`--drift-type imported_oscillator_residual --oscillator-artifact <peak-fit/model.npz>`
adds a frozen physical restoring/damping prior to the ordinary MLP/TCN drift and
learned diffusion, with native-frame conversion and self-contained checkpoints.
See [equations, import checks, commands, and smoke limitations](imported_oscillator.md).
