#!/usr/bin/env bash
set -u

repo=/home/jonas/ucsd_thesis/CapillaryWaveTurbulence
python_bin=/home/jonas/miniforge3/envs/diff-model-mit/bin/python

cd "$repo" || exit 1

"$python_bin" -u -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_tcn63_untrimmed_energy_score_study.json
warm_status=$?
echo "Warm-start energy-score controller exited with status $warm_status"

"$python_bin" -u -m modelling.neural_sde.experiments.adaptive_controller \
  modelling/neural_sde/experiments/0p10_tcn63_untrimmed_energy_score_fresh_study.json
fresh_status=$?
echo "Fresh untrimmed energy-score controller exited with status $fresh_status"

exit "$fresh_status"
