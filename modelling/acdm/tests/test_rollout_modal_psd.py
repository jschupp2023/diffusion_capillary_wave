import json
from types import SimpleNamespace

import numpy as np
import pytest

from data_analysis.psd.rollout_modal_psd import analyze


def test_modal_psd_writes_per_mode_arrays_metrics_and_plot(tmp_path):
    fs = 128.
    samples = 2048
    time = np.arange(samples) / fs
    reference = np.empty((2, samples, 3))
    generated = np.empty((2, 3, samples, 3))
    for condition in range(2):
        reference[condition, :, 0] = 0
        reference[condition, :, 1] = np.sin(2 * np.pi * 4 * time + condition)
        reference[condition, :, 2] = .5 * np.sin(2 * np.pi * 12 * time + condition)
        for ensemble in range(3):
            generated[condition, ensemble, :, 0] = 0
            generated[condition, ensemble, :, 1] = (
                2 * np.sin(2 * np.pi * 4 * time + condition + .1 * ensemble))
            generated[condition, ensemble, :, 2] = (
                .5 * np.sin(2 * np.pi * 12 * time + condition + .1 * ensemble))
    rollout = SimpleNamespace(
        rank=2, include_spatial_mean=True, generated=generated,
        reference=reference, fs=fs, units="microns",
        path=tmp_path / "rollout.npz",
        provenance=lambda: {"rollout": "synthetic"},
    )
    output = analyze(
        rollout, modes=[1, 2], nperseg=256, lower_frequency=1.,
        output=tmp_path / "modal_psd")
    assert (output / "modal_psd.png").is_file()
    assert (output / "quick_summary.md").is_file()
    report = json.loads((output / "summary.json").read_text())
    assert report["selected_modes"] == [1, 2]
    assert report["modes"][0]["generated_over_reference_power"] == pytest.approx(4.)
    assert report["modes"][1]["generated_over_reference_power"] == pytest.approx(1.)
    with np.load(output / "psd.npz") as arrays:
        assert arrays["generated_psd"].shape == (2, 2, 3, 129)
        assert arrays["reference_psd"].shape == (2, 2, 129)


def test_modal_psd_rejects_duplicate_modes(tmp_path):
    rollout = SimpleNamespace(rank=2)
    with pytest.raises(ValueError, match="unique"):
        analyze(rollout, modes=[1, 1], output=tmp_path)
