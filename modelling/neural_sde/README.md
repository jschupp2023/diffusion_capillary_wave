# Neural SDE baseline

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
before. `--multistep-window-fraction` uniformly samples that fraction from each
retained long-window batch for the multistep term. Its sample mean estimates the
same batch multistep mean, while the one-step NLL and stability term still use
the complete retained batch. Training resamples the subset each epoch;
validation uses the fixed `--multistep-validation-seed`. The epoch log records
both available and actually scored multistep-window counts.

With small fractions, `--multistep-window-sampling examples` can leave too few
windows in every GPU launch. Select `batches` to score a complete minibatch
approximately once per `1/fraction` batches. The loss applies the inverse
fraction correction, so the multistep gradient remains an unbiased estimator
of the configured objective; its variance and per-update magnitude are larger.
At fraction `0.05` and weight `0.1`, an active batch therefore uses coefficient
`2.0`. Training and validation use reproducible systematic sampling rather than
clustered Bernoulli draws.

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
  --stationary-mean-horizon 100 \
  --stationary-mean-trajectories 30 \
  --stationary-mean-batch-fraction 0.05 \
  --epochs 5 --batch-size 256 --output runs/velocity_stationary_mean
```

On a selected batch, up to 30 observed starts each produce two trajectories
with independent noise. The matching measured endpoint is read at the configured
horizon. Each ensemble takes at most one window from a shuffled source block,
and every increment through the endpoint must pass the saved training cutoff.

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

The default horizon is 100, trajectory count 30, and selected-batch fraction
0.05. Selected updates use inverse-fraction correction, so
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
