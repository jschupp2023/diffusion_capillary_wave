# Adaptive Neural-SDE experiments

## Bounded fixed oscillator-memory screen

`0p10_fixed_memory_study.json` uses the cached readout fitter, via the existing
controller entry point. Its `kind: fixed_memory` dispatch runs three declared
fits (8,192 windows; 32,768 windows with two seeds) and four matched validation
screens, including the zero-readout parent. Validation sampling is held fixed.
There is no stochastic rollout backpropagation or advisor expansion. An official
zero-readout selection is preserved; a separate rejected nonzero probe may be
screened, explicitly labelled exploratory. The entire queue has a 15-minute
budget with process timeouts and forceful cleanup after five seconds.

```bash
python -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_fixed_memory_study.json --dry-run
```

Protocol and launch script: `runs/adaptive/neural_sde_0p10_fixed_memory_v1/README.md`.
Progress is recorded in that directory's `state.json`; final evidence is in
`summary.json`, `RESULTS.md` and standard rollout quantitative/PSD outputs.

## Factor-eight TCN context and peak-timescale factorial

`0p10_ds8_tcn_peak_timescales_study.json` compares contexts 63/511 against
short (2/5/10/20) and long (20/64/128/256/512) joint displacement/velocity
energy-score horizons. At 14.4 kHz the longer context and horizon span about
35 ms. Each architecture first receives a fresh 20-epoch one-step pretrain;
its two horizon arms then start from that same checkpoint with three initial
refinement epochs. The advisor can request bounded matched continuations.

This study opts into `defer_initial_trial_dependencies`: declared children
are queued only once their earlier declared parent has finished and has a
checkpoint. Failed parents cancel their dependents explicitly. The fixed
comparisons precede any advisor decision. Existing studies retain their
original eager queue behavior. Screen/confirmation settings can also specify
`discard` to exclude a common initial prefix from generated/reference plotting
and quantitative metrics; numerical path-health checks still cover all samples.

```bash
python -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_ds8_tcn_peak_timescales_study.json
```

Protocol, preflight gradients, prior h200 evidence, launch script and progress:
`runs/adaptive/neural_sde_0p10_ds8_tcn_peak_timescales_v1/README.md`.
The study permits 12 training jobs including the six fixed stages, and two
confirmations. It keeps validation 13/14 and locks test 15/16.

## Factor-eight 0p10 long joint energy-score calibration

`0p10_ds8_long_joint_es_study.json` continues the existing downsampled TCN63
checkpoint. Both imported one-step and energy/moment checkpoints first receive
matched 6250-step validation screens (5 conditions x 5 paths, nperseg=1024).
Training then compares same-parent no-score, weight-0.3 and weight-1.0 arms at
horizons 20/100/200 coarse steps (up to 13.9 ms). Later bounded trials can change
weight, horizon, particles, sampling fraction or learning rate with matched
controls. Data preparation and architecture are fixed; test data remain locked.

Every passing screen gets the same gradient audit: four start-window seeds,
three particle-noise seeds and horizon-resolved gradients. The initial current
checkpoint also gets an M16 audit matched to M8 to distinguish particle noise
from variability between sampled starts. Audit coefficient 1 is fixed across
checkpoints and must be rescaled when interpreting actual training weights.
Peak-contrast and band-power metrics are included in advisor packets, and the
peak plot is shown alongside the existing plots. Confirmation uses 12000 coarse
steps; 50000 coarse steps exceed the available validation recordings.

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_ds8_long_joint_es_study.json
```

The resumable study allows ten training jobs and two confirmations. Its state
and logs are under `runs/adaptive/neural_sde_0p10_ds8_long_joint_es/`.

`adaptive_controller.py` runs one Neural-SDE experiment at a time. Training,
rollout generation, and plotting use the existing CLIs. No model agent runs
while those processes are active. Once a result is complete, a short-lived,
read-only `codex exec` process receives compact metrics and selected figures and
chooses one of three actions:

- train one bounded matched ablation;
- promote an existing trial from the cheap validation screen to a 50k rollout;
- stop because the evidence or budget does not justify another experiment.

Innovation-focused studies may also request a reviewed model implementation.
That action writes `implementation_request.md`, marks the campaign
`implementation_required`, and stops before changing source code. They can use
the dedicated `memory` training mode to add and fit the existing diagonal
velocity-memory coefficients from a memory-free parent checkpoint.

The 0p10 study is deliberately validation-only. Test repetitions 15 and 16 are
excluded from every rollout used for selection. Its initial ledger imports the
useful completed 0p10 controls, including the failed history-1,2/untrimmed run,
so they are not recomputed. Endpoint-moment losses are excluded from the search
space. Each advisor override must match an explicitly listed alternative in
`0p10_adaptive_study.json`; commands are passed as argument arrays rather than
shell strings.

First validate the setup without launching Codex or training:

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_adaptive_study.json \
  --dry-run
```

Run the campaign in a persistent shell (`tmux` is recommended):

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_adaptive_study.json
```

The controller writes `state.json`, `advisor_context.json`, the last advisor
decision, subprocess logs, and trial directories below
`runs/adaptive/neural_sde_0p10/`. Re-running the same command resumes completed
stages instead of repeating them. An interrupted training directory without a
checkpoint is retained and marked failed rather than silently appended to.
Inspect progress with:

```bash
tail -f runs/adaptive/neural_sde_0p10/trials/*/controller_logs/*.log
```

For every child process, the controller prepends the selected Python
environment's `bin` directory to `PATH`. If that directory contains `ptxas`, it
also sets `TRITON_PTXAS_PATH` explicitly so Triton cannot silently select an
older system CUDA compiler.

Pass `--manual` to create `decision_request.md` and stop at the advisor
boundary. This is useful for reviewing the evidence packet without spending an
advisor call; the automatic mode is the intended unattended workflow.

The follow-up 0p10 innovation campaign evaluates innovations on validation
repetitions 13 and 14 after every checkpoint. It compares ordinary network
capacity, capacity-matched history models, finite velocity-memory kernels,
stable recurrent oscillators, and autoregressive parent-innovation forcing.
Launch it in a separate tmux session with:

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_innovation_study.json
```

Its durable state and logs live below
`runs/adaptive/neural_sde_0p10_innovation/`. The previous campaign is left as a
completed scientific record.

## 0p20 multistep versus endpoint moments

`0p20_objective_comparison_study.json` starts three matched fine-tunes from the
simple `runs/comparison/velocity_lag1/best.pt` checkpoint: a one-step-only
continuation, the established multistep horizon/mode/particle schedule, and an
endpoint mean-plus-variance objective. The queued controls run before the
advisor can select bounded schedule or moment-weight follow-ups. Selection uses
validation repetitions 13 and 14; test repetitions 15 and 16 remain locked.

Validate the queued commands first:

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p20_objective_comparison_study.json \
  --dry-run
```

Then launch the resumable campaign in a persistent shell:

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p20_objective_comparison_study.json
```

Results, state, advisor decisions, and subprocess logs are written below
`runs/adaptive/neural_sde_0p20_objective_comparison/`.

## 0p20 multistep gradient routing

`0p20_multistep_gradient_routing_study.json` trains four fresh, initialization-
matched damped-residual models: one-step-only, MC stiffness-only, MC structured
stiffness-plus-damping, and MC all-drift. Endpoint-moment objectives are off,
and diffusion never receives an MC gradient in the three multistep arms.

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p20_multistep_gradient_routing_study.json \
  --dry-run

/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p20_multistep_gradient_routing_study.json
```

Results are written below
`runs/adaptive/neural_sde_0p20_multistep_gradient_routing/`.

## 0p10 TCN63 rollout recovery

`0p10_tcn63_rollout_study.json` fine-tunes the original unstructured
`runs/comparison/0p10_tcn63/best.pt`, preserving its architecture, normalization,
and saved 5% full-history trimming rule. Four matched three-epoch arms test a
one-step continuation, velocity multistep weights 0.03/0.1, and displacement
multistep weight 0.03. Additional multistep gradients update drift only;
one-step NLL still trains both drift and diffusion. All 21 coordinates are
scored. Examples-level subsampling limits TCN particle/BPTT memory usage.

```bash
/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_tcn63_rollout_study.json --dry-run

/home/jonas/miniforge3/envs/diff-model-mit/bin/python \
  -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_tcn63_rollout_study.json
```

This study opts into **screen-before-innovations** and a hard promotion gate:
1000 native steps, 8 validation starts, 4 paths each, full noise, seed 814.
Any nonfinite path, peak standardized POD RMS above 10, or absolute standardized
coordinate above 50 rejects the candidate before plotting. After plotting,
energy and POD covariance-trace ratios must both lie in [0.25, 4]. These are
broad rejection bounds, not acceptance of physical fidelity. Rejected trials
remain available as fine-tuning parents but cannot be promoted; no generated
data are clipped or trimmed. Passing candidates receive validation innovation
diagnostics, and the advisor can authorize a 50k confirmation with independent
seed 1814. Budget: six training trials total, two confirmations. Test data stay
locked. The imported parent gets an innovation diagnostic for comparison but
its catastrophically inflated historical rollout is not a passing screen.

The new `screen_before_innovations`, `screen_gate`, and `allowed_training_modes`
settings are opt-in; existing studies keep their evaluation ordering. Live
state and rejection evidence are saved under
`runs/adaptive/neural_sde_0p10_tcn63_rollout/state.json`.
