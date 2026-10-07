# Shared-POD conditional EDM

Run from the project root, in an environment with NumPy, SciPy, h5py, and
PyTorch (the existing `diff-model-mit` environment works).

```bash
python -m modelling.acdm.conditional_edm train \
  --experiment 0p20 --rank 100 \
  --epochs 20 --batch-size 256 \
  --lag-steps 1 --history-steps 2 \
  --output runs/0p20_r100
```

`--rank` counts shared fluctuation modes; the model has `rank + 1` coordinates.
Coordinate zero is `sqrt(number_of_pixels) * frame_spatial_mean`. Remaining
coordinates project each repetition's saved POD reconstruction into the shared
basis. All retained input modes contribute, even when requesting fewer shared
modes. Removed temporal mean fields remain removed.

The default data tree is `../reduced_data` relative to the project root.
`--data-root` and `--shared-basis` override it. A higher-rank shared basis can
supply its leading modes. Use module imports, not direct execution of nested
Python files.

## Data preparation and splitting

`modelling.data_preparation.prepare_shared_pod_training.SharedPODTrainingData`
owns the split. By default it assigns the last 15% of repetitions to test and
the preceding 15% to validation (rounded, at least one each). For 16 repetitions:
train 1–12, validation 13–14, test 15–16. To choose explicitly:

```bash
python -m modelling.acdm.conditional_edm train \
  --experiment 0p20 --rank 100 --validation-reps 13 14 --test-reps 15 16 \
  --output runs/0p20_r100
```

The training function accepts the prepared **source object**, never performs
another split, and does not call its standardized-batch API:

```python
import torch
from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from modelling.acdm.conditional_edm.config import EDMConfig, TrainConfig
from modelling.acdm.conditional_edm.train import train

source = SharedPODTrainingData("0p20", rank=100)
train(data=source, output_dir="runs/0p20_r100",
      model_config=EDMConfig(reduced_dim=source.n_coordinates),
      train_config=TrainConfig(), device=torch.device("cpu"))
```

Coordinates are optionally cached in RAM at startup. Every
eligible transition is visited once per epoch; blocks and pairs within blocks
are shuffled deterministically. Pairs never cross repetitions. `--stride-steps`
changes which transitions are used; `--lag-steps` changes the forecast interval.
The source sampling interval is recorded in the model configuration.

Physical coordinates enter the model. Training-only state statistics and,
for increment prediction, separate lagged-increment statistics are fitted in
streaming float64 passes. The model performs normalization exactly once.
Constant coordinates use scale 1. No prepared coefficients are saved; only
model checkpoints and run metadata are written. The standalone preparation
module still offers standardized batches for other consumers.

## Training diagnostics and runtime

New models use `--conditioning-mode joint_noised` by default. Use
`--conditioning-mode clean` for a clean-conditioning comparison. This adapts the
[ACDM conditioning procedure](https://arxiv.org/html/2309.01745v3#S3) and its
[reference code](https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion.py)
to our EDM model; it does not switch the sampler to the paper's DDPM.

Let `y` be the normalized prediction target and `c` the concatenation of the
normalized current state, history increments and any parameters. Training
forms the joint state `x=(c,y)` and draws `x_sigma=x+sigma*eps` at one shared
noise level. The full state enters the backbone with
EDM input scaling `1/sqrt(sigma_data² + sigma²)`. With the default sigma_data=1,
the scaled conditioning is `(c + sigma*eps_c)/sqrt(1+sigma²)`, equivalent to
the DDPM forward-noise marginal under `alpha_bar = 1/(1+sigma²)`.

The output head reconstructs the full clean state with the usual EDM-weighted
squared error per coordinate. Our conditioning features
include normalized history *increments*, rather than the paper's stacked flow
fields. The clean physical increment target stays `next_state - current_state`;
forward diffusion corrupts input copies, not the physical transition itself.

For every predicted physical step and ensemble member, sampling draws a new
target initialization and an independent conditioning-noise vector. That
conditioning vector is **reused across the entire Heun solve**, as in the
reference code: at each sigma, conditioning is rebuilt from its known clean
features plus sigma times that vector. Predicted conditioning is discarded.
Thus the condition is weak at large sigma and nearly clean near zero, without
injecting these corruptions directly into the saved physical trajectory.

`loss` includes both slices; `target_loss` and `conditioning_loss` report
their contributions. `best.pt` uses validation **target_loss** (or `loss` for
legacy clean models). Fixed-sigma MSE remains target-only, with reproducible
target and conditioning noise. New and old losses need not have the same scale;
compare validation rollout statistics to assess whether stability improves.

Old checkpoints lacking this setting load with conditioning noise disabled,
preserving their trained architecture and sampling. Resuming them keeps that
behavior. To test the new scheme, start a fresh run in a separate directory:

```bash
python -m modelling.acdm.conditional_edm train \
  --experiment 0p20 --rank 50 --lag-steps 1 --history-steps 2 \
  --conditioning-mode joint_noised --epochs 100 --batch-size 256 \
  --output runs/0p20_r50_lag1_noisy
```

Each epoch reports unweighted, per-coordinate MSE in **normalized target**
coordinates on 256 fixed validation transitions at sigma 0.03, 0.3 and 3:
`fixed_MSE_low_sigma`, `fixed_MSE_mid_sigma`, `fixed_MSE_high_sigma`.
The same conditions and Gaussian draws are reused across epochs and noise
levels, in evaluation mode. These are denoising errors, not rollout errors.
`--fixed-sigmas LOW MID HIGH` changes levels; `--fixed-mse-batch-size 0` disables.
The probes do not consume training random draws.

For the source-faithful Kohl baseline, add `--diffusion-formulation ddpm`
(with the default `--conditioning-mode joint_noised`). This uses the fixed
20-step linear schedule, predicts joint `(condition,target)` noise without EDM
preconditioning or loss weighting, and reports target-noise MSE at
`fixed_MSE_r1`, `fixed_MSE_r10`, and `fixed_MSE_r20`. Sampling reuses one
conditioning noise tensor per physical transition and reconstructs the known
conditioning marginal at every reverse step; only the target slice is updated.

Every 10 epochs, two trajectories from each of one fixed training and one
fixed validation start are rolled out for 200 steps, with true preceding
history and the model's usual sampler settings. `--rollout-every`,
`--rollout-horizon` and `--rollout-ensemble-size` control this; setting the
interval to 0 disables it. A 1000-step probe costs roughly five times as much
as 200 steps. Defaults keep these probes small; use separate longer rollouts
for PSD evaluation.

The `rollout` record reports quadratic capillary energy in joules, ratios to
the matching reference window, nonfinite states, and the first step exceeding
10 times that window's maximum energy (`--rollout-energy-factor`). This is
an instability alert, not a physical bound. Nonfinite energy summaries are
JSON null and count as failures. Uniform spatial mean contributes no capillary
energy, so its maximum drift is logged separately. Temporal mean fields remain
omitted. The stiffness matrix is computed once using the existing energy
implementation; `--surface-tension` defaults to 0.0728 N/m. `best.pt` still
uses validation denoising loss; the probes do not change checkpoint selection.

Metrics appear in stdout, checkpoint history and `metrics.jsonl` in the run
directory. Each split logs `elapsed_seconds`, `data_seconds` and `examples`;
`diagnostic_seconds` reports monitoring overhead. Data timing measures CPU
batch preparation; GPU work can overlap it, so this is not an exclusive GPU
profile. Metric scalars are copied to CPU together at epoch end.

`--cache-gib 2` is the default cap on cached physical coordinates; the cache
also must fit within 25% of available Linux RAM. If it does not fit, preparation
streams bounded blocks instead. `--cache-gib 0` forces streaming. The cache
uses float64 to preserve small increments, requires about **1.06 GiB** for
rank 100 with 14 repetitions of 100001 frames, and is rebuilt on restart.
It avoids repeated HDF5 reads and projections; it never saves prepared data.
Startup prints the chosen cache size and preparation time.

Diagnostic/cache options may change on resume without changing optimization
settings. Changes to code take effect when starting/resuming a process; an
already running training process continues using the code it loaded.

```bash
python -m modelling.acdm.conditional_edm train \
  --resume runs/0p20_r100/latest.pt --epochs 100 \
  --cache-gib 2 --rollout-every 10 --rollout-horizon 200
```

The supplied shared basis has already seen every contributing repetition,
including the held-out ones. These splits hold out model/scaler fitting, not
basis fitting. Inputs must remain the same POD decompositions used when the
shared basis was generated.

## Resume and evaluation

```bash
python -m modelling.acdm.conditional_edm train \
  --resume runs/0p20_r100/latest.pt --epochs 30

python -m modelling.acdm.conditional_edm evaluate \
  --checkpoint runs/0p20_r100/best.pt --split test \
  --num-conditions 32 --ensemble-size 16 --horizon 50 \
  --output runs/0p20_r100/evaluation
```

Checkpoints contain the normalization buffers, source configuration, explicit
splits, shared-basis content hash, source size/mtime and timestamp hashes,
optimizer state, and RNG state. Resume/evaluation verify the source identity;
source histories are not copied into checkpoints. Size/mtime verification does
not detect deliberate edits preserving both attributes.

Evaluation reads only selected reference windows. `rollout` accepts the same
arguments and skips metric computation. Both save physical-coordinate
`rollout.npz` arrays and a JSON report. This replaces the old single-repetition
window-conversion and `reduced/state` input workflow. Legacy `psd`,
`visualize-segments`, and original-space visualization routes are removed.
The EDM architecture, sampler, numerical metrics, and synthetic checks remain.

## POD-only spatial-mean ablation

`--no-spatial-mean` omits coordinate zero from both the conditioning states
and increment targets. Rank 50 then means 50 model coordinates. The checkpoint
records this choice, so later rollout commands reopen the same preparation.
For this ablation, training skips the standard mean-dependent rollout
diagnostics; use `--no-metrics` when saving a validation rollout.

```bash
python -m modelling.acdm.conditional_edm train \
  --experiment 0p20 --rank 50 --no-spatial-mean \
  --target-mode increment --lag-steps 1 --history-steps 2 \
  --normalization-from-checkpoint runs/0p20_r50_lag1_acdm_edm_incr_joint/best.pt \
  --epochs 10 --max-train-batches 64 --max-validation-batches 4 \
  --batch-size 128 --cache-gib 0 --fixed-mse-batch-size 0 \
  --hidden-dim 128 --num-blocks 2 --sampling-steps 16 \
  --output runs/0p20_r50_no_mean_quick

python -m modelling.acdm.conditional_edm evaluate \
  --checkpoint runs/0p20_r50_no_mean_quick/best.pt \
  --split validation --num-conditions 1 --ensemble-size 1 \
  --horizon 300 --sampling-steps 16 --no-metrics \
  --output runs/0p20_r50_no_mean_quick/validation_rollout

python -m data_analysis.energy.rollout_energy \
  runs/0p20_r50_no_mean_quick/validation_rollout
```

The last command saves `energy.png` and `traces.png`. Capillary deformation
energy does not depend on a uniform spatial mean, so the POD-only generated
and reference energies remain directly comparable at the same rank. The
spatial-mean contribution to gravitational energy cannot be evaluated from
this model's output. Normalization reuse checks the saved basis, source files,
repetition splits, prediction target, and lag before slicing off coordinate
zero; without that option, the trainer computes new training-only statistics.

For any saved rollout, `python -m data_analysis.plot_rollout PATH_TO_ROLLOUT_DIR`
creates the energy, PSD, and selected coordinate trajectory plots together.
POD-only PSD output contains the center fluctuation spectrum; the spatial-mean
spectrum is unavailable. Use `--overwrite` to regenerate existing plots.

```bash
python -m pytest modelling/acdm/tests -q
```
