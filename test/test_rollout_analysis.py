"""Scientific checks for rollout integration, using analytic waves and surfaces."""
import json

import h5py
import numpy as np
import pytest

from data_analysis.rollout import Rollout
from data_analysis.rollout_metrics import (
    analyze as analyze_metrics,
    energy_comparison,
    moment_comparison,
    reference_training_trim_mask,
    velocity_training_trim_mask_from_native,
)
from data_analysis.psd.rollout_psd import analyze as analyze_psd
from data_analysis.energy.rollout_energy import analyze as analyze_energy, compute_energies
from data_analysis.energy.rollout_modal import analyze as analyze_modal, modal_diagnostics
from data_analysis.plot_rollout import main as plot_rollout


@pytest.fixture
def rollout(tmp_path):
    x, y = np.linspace(-2, 2, 5), np.linspace(-1, 1, 4)
    modes = np.stack(np.meshgrid(x, y))
    modes /= np.linalg.norm(modes, axis=(1, 2), keepdims=True)
    basis_path = tmp_path / "shared.h5"
    names = np.array(["rep1", "rep2"])
    with h5py.File(basis_path, "w") as f:
        f.attrs["instantaneous_spatial_mean_removed"] = True
        f.create_dataset("pod/modes", data=modes)
        for axis, values in (("x", x), ("y", y)):
            f.create_dataset(f"grid/{axis}", data=values).attrs["units"] = "mm"
        for name in names:
            f.create_group(f"repetitions/{name}").attrs["signal_units"] = "microns"
    time = np.arange(257) / 128
    reference = np.empty((2, len(time), 3))
    for c in range(2):
        wave = np.sin(2 * np.pi * 8 * time)
        reference[c] = np.column_stack((np.sqrt(20) * wave, (c + 1) * wave, np.cos(2 * np.pi * 16 * time)))
    generated = np.repeat(reference[:, None], 3, axis=1)
    np.savez(tmp_path / "rollout.npz", trajectories=generated, reference=reference,
             reference_time=np.stack([time, time + 1]), repetition=names)
    (tmp_path / "metrics.json").write_text(json.dumps(dict(rank=2, physical_lag=1/128,
                                                         source_shared_basis=str(basis_path), time_units="seconds")))
    return Rollout(tmp_path)


def test_psd_peak_and_identical_ensemble_reference(rollout):
    out = analyze_psd(rollout, nperseg=128)
    with np.load(out / "psd.npz") as f:
        g, r = f["generated_spatial_mean"], f["reference_spatial_mean"]
        np.testing.assert_allclose(g.mean((0, 1)), r.mean(0), atol=1e-14)
        assert f["frequency_hz"][g.mean((0, 1)).argmax()] == 8
        assert g.shape == (2, 3, 65)
    report = json.loads((out / "summary.json").read_text())
    assert report["signals"]["spatial_mean"]["generated_mean_power"] == pytest.approx(.5)
    assert (out / "psd.png").exists()


def test_energy_comparison_and_exact_bound(rollout):
    out = analyze_energy(rollout, exact=True, batch_size=7)
    with np.load(out / "energy.npz") as f:
        for method in ("quadratic", "exact"):
            np.testing.assert_allclose(f[f"generated_{method}_J"][:, 0], f[f"reference_{method}_J"], rtol=1e-12)
        assert np.all(f["generated_exact_J"] <= f["generated_quadratic_J"] * (1 + 1e-12))
        assert np.all(f["generated_exact_J"] >= 0)
    report = json.loads((out / "summary.json").read_text())
    assert report["energy"]["quadratic"]["generated_over_reference"] == pytest.approx(1)


def test_pod_only_rollout_energy_uses_every_coefficient(rollout):
    path = rollout.path
    expected = compute_energies(rollout.reference, rollout.modes,
                                rollout.x * 1e-3, rollout.y * 1e-3, 1e-6, .0728)["quadratic"]
    np.savez(path, trajectories=rollout.generated[..., 1:], reference=rollout.reference[..., 1:],
             reference_time=rollout.time, repetition=rollout.repetitions)
    metadata_path = path.with_name("metrics.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["include_spatial_mean"] = False
    metadata_path.write_text(json.dumps(metadata))
    pod_only = Rollout(path)
    assert pod_only.include_spatial_mean is False
    psd_out = analyze_psd(pod_only)
    with np.load(psd_out / "psd.npz") as values:
        assert "generated_spatial_mean" not in values
        np.testing.assert_allclose(values["generated_center_without_spatial_mean"][:, 0],
                                   values["reference_center_without_spatial_mean"])
    assert json.loads((psd_out / "summary.json").read_text())["spatial_mean_available"] is False
    out = analyze_energy(pod_only)
    with np.load(out / "energy.npz") as values:
        np.testing.assert_allclose(values["reference_quadratic_J"], expected)
    assert (out / "traces.png").exists()


def test_modal_diagnostics_amplitude_ratio_and_energy_closure(rollout):
    generated = rollout.generated[..., 1:].copy()
    reference = rollout.reference[..., 1:]
    generated[..., 0] *= 2
    stiffness = np.array([[2., .25], [.25, 1.]])
    values = modal_diagnostics(generated, reference, stiffness, coefficient_scale=3.)
    assert values["mean_square_ratio"][0] == pytest.approx(4)
    assert values["rms_amplitude_ratio"][0] == pytest.approx(2)
    for trajectories, energy in ((generated, values["generated_energy_attribution_J"]),
                                 (reference, values["reference_energy_attribution_J"])):
        direct = .5 * np.einsum("...i,ij,...j->...", 3 * trajectories, stiffness,
                                3 * trajectories, optimize=True).mean()
        assert energy.sum() == pytest.approx(direct)


def test_modal_diagnostics_reference_mask_changes_reference_population(rollout):
    generated = rollout.generated[..., 1:]
    reference = rollout.reference[..., 1:]
    mask = np.zeros(reference.shape[:-1], dtype=bool)
    mask[:, ::2] = True
    stiffness = np.eye(rollout.rank)
    values = modal_diagnostics(
        generated, reference, stiffness, reference_mask=mask)
    selected = reference[mask]
    np.testing.assert_allclose(values["reference_mean"], selected.mean(0))
    np.testing.assert_allclose(values["reference_variance"], selected.var(0))
    with pytest.raises(ValueError, match="Reference mask"):
        modal_diagnostics(generated, reference, stiffness,
                          reference_mask=np.ones(reference.shape[:1], dtype=bool))


def test_modal_rollout_output(rollout):
    out = analyze_modal(rollout)
    assert (out / "modal_diagnostics.png").exists()
    with np.load(out / "modal_diagnostics.npz") as values:
        np.testing.assert_allclose(values["mean_square_ratio"], 1)
        np.testing.assert_allclose(values["rms_amplitude_ratio"], 1)
        np.testing.assert_allclose(values["generated_energy_attribution_J"],
                                   values["reference_energy_attribution_J"])
    report = json.loads((out / "summary.json").read_text())
    assert report["generated_over_reference"] == pytest.approx(1)
    assert report["comparison_used_for_plot_and_primary_fields"] == "untrimmed"
    assert report["generated_untrimmed_vs_reference_training_trimmed"] is None


def test_modal_rollout_uses_training_trim_as_primary_comparison(
        rollout, tmp_path, monkeypatch):
    mask = np.zeros(rollout.reference.shape[:-1], dtype=bool)
    mask[:, :100] = True
    trim = dict(removed_reference_frame_percent=100 * float((~mask).mean()),
                retained_reference_frames=int(mask.sum()))
    monkeypatch.setattr(
        "data_analysis.energy.rollout_modal.reference_training_trim_population",
        lambda _: (mask, trim))
    out = analyze_modal(rollout, output=tmp_path / "trimmed_modal")
    report = json.loads((out / "summary.json").read_text())
    assert (report["comparison_used_for_plot_and_primary_fields"]
            == "generated_untrimmed_vs_reference_training_trimmed")
    assert report["reference_training_trim"] == trim
    assert report["untrimmed"]["generated_over_reference"] == pytest.approx(1)
    assert report["generated_untrimmed_vs_reference_training_trimmed"] is not None
    with np.load(out / "modal_diagnostics.npz") as values:
        np.testing.assert_array_equal(values["reference_training_trim_mask"], mask)
        np.testing.assert_allclose(values["untrimmed_mean_square_ratio"], 1)


def test_quantitative_rollout_metrics_identical_populations(rollout):
    out = analyze_metrics(rollout, nperseg=128, psd_lower_frequency=1)
    report = json.loads((out / "summary.json").read_text())
    quick = (out / "quick_summary.md").read_text()
    assert "Central result" in quick
    assert "Mean quadratic energy" in quick
    assert "Half-decade PSD comparison" in quick
    assert report["energy"]["untrimmed"]["generated_over_reference"] == pytest.approx(1)
    assert report["energy"]["generated_untrimmed_vs_reference_training_trimmed"] is None
    statistics = report["coefficient_statistics"]["untrimmed"]
    assert statistics["mean_bias_l2"] == pytest.approx(0, abs=1e-14)
    assert statistics["covariance_relative_frobenius_error"] == pytest.approx(0, abs=1e-14)
    pod_statistics = report["coefficient_statistics"]["pod_fluctuation_coordinates"]["untrimmed"]
    assert pod_statistics["covariance_relative_frobenius_error"] == pytest.approx(0, abs=1e-14)
    mean_statistics = report["coefficient_statistics"]["spatial_mean_coordinate"]["untrimmed"]
    assert mean_statistics["mean_bias_l2"] == pytest.approx(0, abs=1e-14)
    for signal in report["psd"]["signals"].values():
        assert signal["bins_per_decade"] == 2
        assert signal["mean_absolute_log10_band_power_error_decades"] == pytest.approx(0, abs=1e-14)
        assert all(band["ratio"] == pytest.approx(1) for band in signal["bands"])
    with np.load(out / "metrics.npz") as values:
        np.testing.assert_allclose(values["generated_covariance"], values["reference_covariance"],
                                   atol=1e-14)


def test_training_trim_is_applied_only_to_reference_frames():
    reference = np.array([[[0.], [1.], [2.], [3.], [20.], [21.]]])
    specification = dict(cutoff=2., state_std=np.array([1.]), train_trim_percent=5.,
                         state_variable="velocity", lag_steps=1, config_path="config.json")
    reference_mask, trim = reference_training_trim_mask(reference, specification)
    assert reference_mask.tolist() == [[False, True, True, False, False, False]]
    assert trim["retained_reference_frames"] == 2
    generated_energy = np.array([[[1., 2., 3., 4., 5., 6.],
                                  [2., 3., 4., 5., 6., 7.]]])
    reference_energy = np.array([[1., 2., 3., 100., 200., 300.]])
    report = energy_comparison(generated_energy, reference_energy,
                               reference_mask, trim)
    comparison = report["generated_untrimmed_vs_reference_training_trimmed"]
    assert comparison["generated"]["frame_count"] == 12
    assert comparison["generated"]["mean_J"] == pytest.approx(4.)
    assert comparison["reference"]["frame_count"] == 2
    assert comparison["reference"]["mean_J"] == pytest.approx(2.5)


def test_lagged_velocity_trim_checks_every_native_increment():
    # Two coarse lag-2 transitions. Each current-state stencil contains the
    # incoming native increment plus the following two increments.
    native_paths = np.array([[[0.], [1.], [2.], [20.], [21.], [22.]]])
    specification = dict(cutoff=2., state_std=np.array([1.]),
                         state_variable="velocity", lag_steps=2)
    keep = velocity_training_trim_mask_from_native(
        native_paths, sample_count=3, specification=specification)
    # Frame zero sees increments [1, 1, 18] and is rejected. Frame one sees
    # [18, 1, 1] and is rejected. The final frame has no outgoing stencil.
    assert keep.tolist() == [[False, False, False]]


def test_moment_comparison_known_shift_and_covariance_scale():
    reference = np.array([[-1., -2.], [1., 2.], [-1., 2.], [1., -2.]])
    generated = 2 * reference + np.array([3., -1.])
    report = moment_comparison(generated, reference)
    np.testing.assert_allclose(report["mean_bias"], [3, -1])
    assert report["covariance_trace_ratio"] == pytest.approx(4)
    assert report["covariance_relative_frobenius_error"] == pytest.approx(3)


def test_one_command_creates_all_rollout_plots(rollout, tmp_path, monkeypatch):
    output = tmp_path / "all_plots"
    captured = {}

    def fake_video(reference, generated, x, y, times, *, output, **options):
        captured.update(reference=reference, generated=generated, x=x, y=y,
                        times=times, options=options)
        output.write_bytes(b"test video")
        return output

    monkeypatch.setattr(
        "data_analysis.plot_rollout.make_surface_comparison_video", fake_video)
    plot_rollout([str(rollout.path.parent), "--output", str(output),
                  "--modes", "1", "2", "--surface-video", "--video-steps", "5",
                  "--video-spatial-points", "3", "--video-fps", "12"])
    assert (output / "energy" / "energy.png").exists()
    assert (output / "modal" / "modal_diagnostics.png").exists()
    assert (output / "psd" / "psd.png").exists()
    assert (output / "coordinates" / "trajectories.png").exists()
    assert (output / "quantitative" / "summary.json").exists()
    assert (output / "quantitative" / "quick_summary.md").exists()
    assert (output / "surface_video" / "reference_vs_generated.mp4").exists()
    assert captured["reference"].shape == (5, 3, 3)
    np.testing.assert_allclose(captured["generated"], captured["reference"])
    np.testing.assert_allclose(captured["times"], np.arange(5) / rollout.fs)
    assert captured["options"]["fps"] == 12
    with pytest.raises(FileExistsError, match="--overwrite"):
        plot_rollout([str(rollout.path.parent), "--output", str(output)])


def test_piston_has_no_energy_and_plane_matches_analytic_area():
    x, y = np.linspace(0, 2, 5), np.linspace(0, 3, 4)
    mode = np.broadcast_to(x, (len(y), len(x)))[None]
    # Coordinates are metres here, plane slope a=.4, area=6, gamma=.07.
    coefficients = np.array([[100., .4], [-100., .4], [5., 0.]])
    result = compute_energies(coefficients, mode, x, y, 1, .07, exact=True, batch_size=1)
    np.testing.assert_allclose(result["quadratic"], [.5*.07*6*.4**2]*2+[0], atol=1e-15)
    np.testing.assert_allclose(result["exact"], [.07*6*(np.sqrt(1+.4**2)-1)]*2+[0], atol=1e-15)


def test_discard_metadata_and_reject_bad_timing(rollout):
    trimmed = Rollout(rollout.path, discard=10)
    assert trimmed.generated.shape[2] == 247
    np.testing.assert_array_equal(trimmed.reference, rollout.reference[:, 10:])
    path = rollout.path.with_name("metrics.json")
    metadata = json.loads(path.read_text())
    metadata["physical_lag"] *= 2
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sampling interval"):
        Rollout(rollout.path)
