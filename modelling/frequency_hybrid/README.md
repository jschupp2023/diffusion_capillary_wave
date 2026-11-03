# Frequency-separated POD / neural-SDE experiment

This is an **offline stationary-generation baseline**, with unchanged shared
spatial POD modes. Rank 10 includes ten deformation coordinates plus the stored
50 Hz high-pass spatial mean (11 coordinates). The three default temporal bands
are 255–300, 425–480, and 620–680 Hz. They are initial training-selected regions,
not universal physical mode boundaries.

The implementation consists of:

- `modelling/data_preparation/frequency_components.py`: per-record SOS
  forward/backward filtering, edge guards, exact subtractive remainder,
  checksummed read-only NPY caches, and the existing training-data interface.
- `oscillators.py`: one shared frequency/damping per band, a regularized full
  spatial excitation covariance, exact stable transitions, stationary sampling.
- `study.py`: training-only spectral fitting and cross-term preflight; matched
  five-epoch full-signal and remainder TCNs using the existing trainer.
- `evaluate.py`: component, hybrid, oracle-remainder, and existing-checkpoint
  comparisons, including the common rollout metric schema.
- `sensitivity.py`: training-only effective linewidth sensitivity to band width.

The source HDF5 files are never modified. Test repetitions 15–16 are not read
for component preparation, fitting or evaluation. Metadata/signature validation
reads source timestamps. The existing shared spatial basis includes held-out
recordings; this experiment does not claim an independently fitted spatial basis.

## Reproduce

From the repository root, activate `diff-model-mit`. On this machine Triton
requires the matching assembler:

```bash
export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-frequency-hybrid
python -m modelling.frequency_hybrid.study prepare
python -m modelling.frequency_hybrid.sensitivity
python -m modelling.frequency_hybrid.study train --epochs 5 --device cuda
python -m modelling.frequency_hybrid.study evaluate
python -m modelling.frequency_hybrid.summarize
```

Default output: `runs/diagnostics/frequency_hybrid_0p10_rank10_v1`.
Use `--output` to create a new study. `prepare` also accepts six `--bands`
values, `--order`, and `--edge-seconds`. Component checkpoints can be evaluated
or explicitly continued with the ordinary neural-SDE checkpoint commands; their
data resolver restores the component target and checks its identity. Ordinary
checkpoint evaluation conditions on offline component histories and must not be
reported as a causal forecast from raw observations.

The pair uses fresh TCN63 models, 64 channels, velocity-state damped-residual
parameterization, diagonal diffusion, learning rate 1e-4, batch size 256 and
seed 0. Both use one-step NLL, short multistep likelihood (2/5/10/20 native
steps, eight particles, weight .03) and the existing 20/50/100-step endpoint
mean/variance calibration. Multistep batches are sampled at fraction .005 with
the trainer's inverse-probability weighting. This reduces rollout launch
overhead relative to evaluating tiny example subsets on every minibatch.
The successful historical checkpoint used a longer training/fine-tuning lineage;
these five-epoch fresh fits are not reproductions of that training history.

## Interpretation

- The sum of measured components reconstructs the preprocessed signal exactly;
  their *autospectra* do not sum exactly because inter-component cross terms
  remain. `preflight.json` quantifies this independence approximation.
- The same filters are applied to every coordinate. They retain all spatial
  modes; no SPOD change of basis or hidden-state inversion is introduced.
- Oscillators fit total band power, including background under the peaks. Their
  widths are effective filtered-component parameters, not physical damping.
  Real shared-frequency dynamics cannot fit arbitrary imaginary cross-spectra.
- A second-order white-driven oscillator has spectral tails. Generated output is
  never post-filtered. Stopband log errors must be read alongside actual tail
  power and within-band fit errors.
- The remainder contains notches and all unselected peaks. Success requires the
  neural SDE to preserve this structure in autonomous generation.
- Stationary oscillator states are sampled analytically. The TCN starts from
  training histories and discards 10,000 generated steps. Early/late moments
  diagnose residual initialization effects; stationarity is not guaranteed.
- Conditional causal forecasting requires a separately defined causal filtering
  and initialization protocol, including replacing the stored offline mean
  high-pass if strict causal observation conditioning is required.

## Tests

```bash
python -m pytest modelling/acdm/tests/test_frequency_hybrid.py \
  modelling/acdm/tests/test_downsampled_preparation.py \
  modelling/latent_oscillators/tests/test_model.py -q
```

Tests cover reconstruction, physical timestamp alignment, training-only scaling,
source immutability, minibatch consistency, component checkpoint/evaluation
round trips, cache corruption detection, stationary oscillator covariance, and
synthetic frequency/damping recovery.
