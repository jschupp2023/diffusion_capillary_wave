# Stochastic operator inference

This package applies the original state-based stochastic OpInf equations to the
same normalized shared-POD trajectories used by ACDM and the Neural SDE.
`--rank R` retains `R` shared POD modes plus the `sqrt(N) * spatial_mean`
coordinate. The default whole-repetition split is 12 train, 2 validation, and
2 test for a 16-repetition experiment. Only training frames fit normalization
and operators.

Training divides each training repetition into non-overlapping 5,760-frame
windows, discards the incomplete tail, and computes the relative-time ensemble
mean and covariance. It preserves the original `A`, `AB`, `AN`, and `ABN`
forms, sixth-order mean derivative, covariance-based constant diffusion,
implicit Euler-Maruyama update, and training-mean-error regularization choice.
The input is

```text
u(t) = input_amplitude * cos(2*pi*input_frequency*t)
```

with defaults `input_amplitude=1` and `input_frequency=7e6` Hz. Relative time
starts at zero for every segment or evaluation window, as in the segmented
original workflow. The `A` form constructs but does not use `u(t)`; choose
`AB`, `AN`, or `ABN` when the input should enter the inferred dynamics.

```bash
python -m modelling.stochastic_opinf train \
  --experiment 0p20 --rank 50 --model-form A \
  --output runs/0p20_r50_stochastic_opinf

python -m modelling.stochastic_opinf evaluate \
  --checkpoint runs/0p20_r50_stochastic_opinf/best.pt \
  --split test --num-conditions 32 --ensemble-size 16 --horizon 50 \
  --output runs/0p20_r50_stochastic_opinf/evaluation

python -m data_analysis.plot_rollout \
  runs/0p20_r50_stochastic_opinf/evaluation
```

Training saves `best.pt`, `config.json`, `regularization.npz`, and a compact
regularization plot. Evaluation writes the common physical-coordinate
`rollout.npz` and `metrics.json` files.

For an input-dependent `ABN` fit using a stored high-pass spatial mean and
explicit operator regularization, for example:

```bash
python -m modelling.stochastic_opinf train \
  --experiment 0p20 --rank 20 --model-form ABN \
  --spatial-mean-highpass-hz 50 \
  --a-regularization 1e3 --b-regularization 0 \
  --n-regularization-min 1 --n-regularization-max 1e10 \
  --regularization-count 10 --h-regularization 1e4 \
  --output runs/0p20_r20_stochastic_opinf_abn_hp50
```

For forms containing `N`, the N regularization is swept logarithmically while
A and B regularization remain fixed. For `A` and `AB`, the original A grid
from `1e-1` through `1e5` is retained. The checkpoint records the full selected
regularization vector. Evaluation inherits the stored spatial-mean treatment.
