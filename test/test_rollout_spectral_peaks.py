"""Scientific behavior of reference-guided resonance contrast."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from data_analysis.psd.rollout_spectral_peaks import analyze, peak_metrics


def spectrum(f, *, shift=0., width=15.):
    db = sum(6 * np.exp(-.5 * ((f - center - shift) / width)**2)
             for center in (250., 600., 1050.))
    return 10**(db / 10)


def test_contrast_separates_amplitude_from_structure():
    f = np.arange(0., 2049., 4.)
    r = spectrum(f)
    identical = peak_metrics(f, r, r)
    scaled = peak_metrics(f, r, 9 * r)
    assert identical["reference_peak_count"] == 3
    assert identical["peak_contrast_error_db"] == pytest.approx(0, abs=1e-12)
    assert scaled["peak_contrast_error_db"] == pytest.approx(0, abs=1e-12)
    assert scaled["generated_over_reference_band_power"] == pytest.approx(9)


@pytest.mark.parametrize("generated", ["missing", "shifted", "broad"])
def test_same_band_power_does_not_hide_failed_resonances(generated):
    f = np.arange(0., 2049., 4.); r = spectrum(f)
    g = {"missing": np.ones_like(f), "shifted": spectrum(f, shift=55),
         "broad": spectrum(f, width=150)}[generated]
    mask = (f >= 100) & (f <= 1500)
    g *= r[mask].sum() / g[mask].sum()
    result = peak_metrics(f, r, g)
    assert result["generated_over_reference_band_power"] == pytest.approx(1)
    assert result["peak_contrast_error_db"] > 1
    assert result["reference_peak_count"] == 3
    assert [p["peak_bin"] for p in result["peaks"]] == [p["peak_bin"] for p in peak_metrics(f, r, r)["peaks"]]


def test_no_reference_peaks_and_short_records_are_not_perfect_scores():
    f = np.arange(0., 2049., 4.)
    no_peaks = peak_metrics(f, np.ones_like(f), spectrum(f))
    assert no_peaks["status"] == "no_reference_peaks"
    assert no_peaks["reference_peak_count"] == 0
    assert no_peaks["peak_contrast_error_db"] is None
    short = peak_metrics(f, spectrum(f), spectrum(f), sufficient_samples=False)
    assert short["status"] == "insufficient_samples"
    assert short["peak_contrast_error_db"] is None
    assert short["generated_over_reference_band_power"] == pytest.approx(1)
    empty = peak_metrics(f, np.zeros_like(f), np.ones_like(f))
    assert empty["status"] == "no_reference_power"
    assert empty["generated_over_reference_band_power"] is None


def test_uncovered_band_is_explicit():
    f = np.arange(65.)
    result = peak_metrics(f, np.ones_like(f), np.ones_like(f))
    assert result["status"] == "insufficient_frequency_coverage"
    assert result["peak_contrast_error_db"] is None


@pytest.mark.parametrize("with_mean", [True, False])
def test_saved_signal_reconstruction_welch_and_outputs(tmp_path, with_mean):
    fs = 4096
    t = np.arange(8192) / fs
    # Different amplitudes per condition ensure equal-condition PSD aggregation.
    rng = np.random.default_rng(4)
    waves = np.stack([a * np.sin(2*np.pi*320*t) + .2*rng.normal(size=len(t)) for a in (1, 2)])
    coefficients = waves[..., None]
    if with_mean:
        coefficients = np.concatenate((2*coefficients, np.zeros_like(coefficients)), axis=-1)
    class Example(SimpleNamespace):
        def provenance(self):
            return dict(reference_repetitions=["rep13", "rep14"], sampling_frequency_hz=self.fs)
    rollout = Example(
        path=tmp_path/"rollout.npz", fs=fs, units="microns", n_space=4,
        x=np.arange(2), y=np.arange(2), x_units="mm", y_units="mm",
        modes=np.ones((1, 2, 2)), include_spatial_mean=with_mean,
        reference=coefficients, generated=np.repeat((3*coefficients)[:, None], 4, axis=1),
        metadata={"split": "validation"})
    out = analyze(rollout, nperseg=1024)
    report = json.loads((out/"summary.json").read_text())
    assert report["signal"] == ("center" if with_mean else "center_without_spatial_mean")
    assert report["reference_split"] == "validation"
    assert report["metrics"]["reference_peak_count"] >= 1
    assert report["metrics"]["peak_contrast_error_db"] == pytest.approx(0, abs=1e-10)
    assert report["metrics"]["generated_over_reference_band_power"] == pytest.approx(9)
    with np.load(out/"psd.npz") as z:
        assert z["generated_psd"].shape == (2, 4, 513)
        assert z["reference_psd"].shape == (2, 513)
        np.testing.assert_allclose(z["generated_mean_psd"],9*z["reference_mean_psd"])
    assert (out/"psd_peaks.png").stat().st_size > 1000
    assert "Peak-contrast error" in (out/"quick_summary.md").read_text()
    with pytest.raises(FileExistsError):
        analyze(rollout, nperseg=1024)
