import json

import pytest

from modelling.neural_sde.experiments.ten_peak_remainder_architecture_study import (
    _ratio_penalty, extract_candidate)


def test_ratio_penalty_is_symmetric():
    assert _ratio_penalty(2) == pytest.approx(_ratio_penalty(.5))


def test_extract_candidate_uses_filtered_hybrid(tmp_path):
    quantitative = tmp_path/"quantitative.json"
    quantitative.write_text(json.dumps({
        "energy": {"untrimmed": {"generated_over_reference": 1.1}},
        "psd": {"signals": {
            "center": {"mean_absolute_log10_band_power_error_decades": .2},
            "center_without_spatial_mean": {
                "mean_absolute_log10_band_power_error_decades": .3}}}}))
    (tmp_path/"summary.json").write_text(json.dumps({"reports": {"hybrid_filtered": {
        "common_metrics": str(quantitative), "center_band_power_ratio": .9,
        "acf_rmse_first_4000_lags": .1, "increment_variance_ratio": 1.2,
        "normalized_csd_rmse": .4, "coherence_complex_rmse": .3,
        "peaks": {"peak_contrast_error_db": 3.0}}}}))
    result = extract_candidate(tmp_path)
    assert result["energy_ratio"] == 1.1
    assert result["priority_pass"] is True
    assert result["primary_score"] > 0
