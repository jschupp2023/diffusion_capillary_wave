"""Finite-particle diagnostics for the Neural-SDE multistep mixture."""
import json

import numpy as np
import torch

from modelling.acdm.conditional_edm.checkpoint import save_checkpoint
from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.losses import MonteCarloMultistepNLL
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.multistep_particle_diagnostic import diagnose
from modelling.neural_sde.multistep_gradient_diagnostic import diagnose_gradients
from modelling.neural_sde.multistep_batch_fraction_diagnostic import diagnose_batch_fraction
from modelling.neural_sde.train import _fit_sde_normalization


def test_particle_diagnostics_match_marginal_mixture_and_cv_identity():
    model = NeuralSDE(SDEConfig(
        reduced_dim=3, dt=.01, history_steps=1, state_variable="velocity",
        hidden_layers=(4,), diffusion_type="full", diffusion_init=.3))
    model.drift_model.zero_output()
    with torch.no_grad():
        model.diffusion_model.lower_entries.copy_(torch.tensor([.15, -.08, .11]))
    current = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    history = current[:, None] - torch.tensor([.2, -.1, .3])
    targets = torch.tensor([[[9., .25, -7.]], [[-8., -.4, 6.]]])
    objective = MonteCarloMultistepNLL(
        (2,), particles=7, mode_counts=(1,), pod_coordinate_offset=1)

    diagnostics = {}
    losses = objective.per_horizon(
        model, current, targets, history_states=history,
        generator=torch.Generator().manual_seed(19),
        particle_diagnostics=diagnostics)
    unchanged = objective.per_horizon(
        model, current, targets, history_states=history,
        generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(losses[2], unchanged[2])
    values = diagnostics[2]
    torch.testing.assert_close(losses[2], -values["log_likelihood"].mean())
    torch.testing.assert_close(
        values["coefficient_of_variation_squared"],
        7 / values["effective_sample_size"] - 1)
    torch.testing.assert_close(
        values["relative_standard_error"].square(),
        1 / values["effective_sample_size"] - 1 / 7)
    assert torch.all((1 <= values["effective_sample_size"])
                     & (values["effective_sample_size"] <= 7))
    assert torch.all((0 < values["largest_component_fraction"])
                     & (values["largest_component_fraction"] <= 1))

    changed_excluded = targets.clone()
    changed_excluded[..., 0] += 1000
    changed_excluded[..., 2] -= 1000
    repeated = {}
    repeated_losses = objective.per_horizon(
        model, current, changed_excluded, history_states=history,
        generator=torch.Generator().manual_seed(19),
        particle_diagnostics=repeated)
    torch.testing.assert_close(repeated_losses[2], losses[2])
    for name, value in values.items():
        torch.testing.assert_close(repeated[2][name], value)


def test_checkpoint_diagnostic_inherits_schedules_and_saves_per_window_arrays(
        shared_data, tmp_path):
    config = SDEConfig(
        shared_data.n_coordinates, dt=shared_data.native_dt,
        history_steps=1, state_variable="velocity", hidden_layers=(4,))
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    checkpoint = tmp_path / "run/best.pt"
    save_checkpoint(checkpoint, {
        "model_config": config.to_dict(),
        "data_config": shared_data.source_config,
        "data_signature": shared_data.signature(),
        "model_state": model.state_dict(),
        "training": {
            "multistep_target": "velocity",
            "multistep_horizons": [2, 4],
            "multistep_mode_counts": [2, 1],
            "multistep_particles": 7,
            "multistep_particle_counts": [7, 3],
            "train_trim_norm_cutoff": None,
            "train_trim_state_std": None,
        },
    })

    output = diagnose(
        checkpoint.parent, tmp_path / "diagnostic", split="validation",
        batch_size=3, max_windows=5, seeds=(11, 12), device="cpu")
    report = json.loads((output / "metrics.json").read_text())
    assert report["horizons"] == [2, 4]
    assert report["mode_counts"] == [2, 1]
    assert report["particle_counts"] == [7, 3]
    assert report["window_count"] == 5
    assert report["per_horizon"]["2"]["scored_dimension"] == 2
    assert report["per_horizon"]["4"]["particle_count"] == 3
    assert report["per_horizon"]["2"]["seed_stability"][
        "empirical_likelihood_relative_sd"] is not None

    arrays = np.load(output / "per_window_diagnostics.npz")
    assert arrays["h2_effective_sample_size"].shape == (2, 5)
    assert arrays["h4_relative_standard_error"].shape == (2, 5)
    assert arrays["current_state"].shape == (5, shared_data.n_coordinates)
    np.testing.assert_allclose(
        arrays["h4_coefficient_of_variation_squared"],
        3 / arrays["h4_effective_sample_size"] - 1,
        rtol=1e-5, atol=1e-6)


def test_gradient_diagnostic_uses_fixed_batch_and_leaves_checkpoint_unchanged(
        shared_data, tmp_path):
    config = SDEConfig(
        shared_data.n_coordinates, dt=shared_data.native_dt,
        history_steps=1, state_variable="velocity", hidden_layers=(4,),
        drift_type="damped_residual")
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    checkpoint = tmp_path / "gradient_run/best.pt"
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()}
    save_checkpoint(checkpoint, {
        "model_config": config.to_dict(),
        "data_config": shared_data.source_config,
        "data_signature": shared_data.signature(),
        "model_state": model.state_dict(),
        "training": {
            "multistep_target": "velocity",
            "multistep_horizons": [2],
            "multistep_mode_counts": [1],
            "multistep_particles": 3,
            "multistep_particle_counts": [3],
            "train_trim_norm_cutoff": None,
            "train_trim_state_std": None,
        },
    })

    output = diagnose_gradients(
        checkpoint.parent, tmp_path / "gradient_diagnostic", split="validation",
        windows=3, microbatch_size=2, seeds=(1, 2), particle_multipliers=(1, 2),
        reference_multiplier=2, reference_seeds=(11, 12), device="cpu")
    report = json.loads((output / "metrics.json").read_text())
    assert (output / "quick_summary.md").is_file()
    assert report["objective_name"] == "gmm_nll"
    assert report["saved_training_context"]["multistep_objective"] == "gmm_nll"
    assert report["window_count"] == 3
    assert report["gradient_microbatch_size"] == 2
    assert report["parameter_counts"]["drift_stiffness"] == shared_data.n_coordinates
    assert report["parameter_counts"]["drift_damping"] == shared_data.n_coordinates
    assert report["parameter_counts"]["drift_structured"] == 2 * shared_data.n_coordinates
    assert report["parameter_counts"]["drift_residual"] > 0
    assert report["base_particle_counts"] == [3]
    assert report["by_particle_multiplier"]["2"]["particle_counts"] == [6]
    for group in ("all", "drift", "diffusion"):
        summary = report["by_particle_multiplier"]["1"]["gradient_groups"][group]
        assert summary["cosine_to_reference"]["finite_count"] == 2
        assert summary["gradient_norm"]["minimum"] >= 0
    for group in ("drift_structured", "drift_stiffness", "drift_damping",
                  "drift_residual"):
        assert group in report["by_particle_multiplier"]["1"]["gradient_groups"]

    arrays = np.load(output / "gradients.npz")
    assert arrays["current_state"].shape[0] == 3
    assert arrays["m1_samples_all"].shape[0] == 2
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    for name, value in original_state.items():
        torch.testing.assert_close(saved[name], value)


def test_gradient_diagnostic_inherits_energy_score_objective(shared_data, tmp_path):
    config = SDEConfig(
        shared_data.n_coordinates, dt=shared_data.native_dt,
        history_steps=1, state_variable="velocity", hidden_layers=(4,),
        drift_type="damped_residual")
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    checkpoint = tmp_path / "energy_gradient_run/best.pt"
    save_checkpoint(checkpoint, {
        "model_config": config.to_dict(),
        "data_config": shared_data.source_config,
        "data_signature": shared_data.signature(),
        "model_state": model.state_dict(),
        "training": {
            "multistep_objective": "energy_score",
            "multistep_target": "velocity",
            "multistep_horizons": [2],
            "multistep_mode_counts": None,
            "multistep_particles": 2,
            "multistep_particle_counts": [2],
            "train_trim_norm_cutoff": None,
            "train_trim_state_std": None,
        },
    })

    output = diagnose_gradients(
        checkpoint.parent, tmp_path / "energy_gradient_diagnostic",
        split="validation", windows=3, microbatch_size=2,
        seeds=(1, 2), particle_multipliers=(1, 2),
        reference_multiplier=2, reference_seeds=(11, 12), device="cpu")
    report = json.loads((output / "metrics.json").read_text())
    assert report["objective_name"] == "energy_score"
    assert report["saved_training_context"]["multistep_objective"] == "energy_score"
    assert report["base_particle_counts"] == [2]
    assert report["by_particle_multiplier"]["2"]["particle_counts"] == [4]
    assert "conditional velocity energy scores" in report["objective"]
    assert "`energy_score`" in (output / "quick_summary.md").read_text()
    for group in ("all", "drift", "diffusion"):
        summary = report["by_particle_multiplier"]["1"]["gradient_groups"][group]
        assert summary["cosine_to_reference"]["finite_count"] == 2
        assert np.isfinite(summary["gradient_norm"]["mean"])


def test_batch_fraction_diagnostic_separates_particle_and_window_noise(
        shared_data, tmp_path):
    config = SDEConfig(
        shared_data.n_coordinates, dt=shared_data.native_dt,
        history_steps=1, state_variable="velocity", hidden_layers=(4,),
        drift_type="damped_residual")
    model = NeuralSDE(config)
    model.set_normalization(_fit_sde_normalization(shared_data, config))
    checkpoint = tmp_path / "batch_fraction_run/best.pt"
    save_checkpoint(checkpoint, {
        "model_config": config.to_dict(),
        "data_config": shared_data.source_config,
        "data_signature": shared_data.signature(),
        "model_state": model.state_dict(),
        "training": {
            "batch_size": 2,
            "multistep_objective": "energy_score",
            "multistep_target": "velocity",
            "multistep_horizons": [2],
            "multistep_mode_counts": None,
            "multistep_particles": 2,
            "multistep_particle_counts": [2],
            "multistep_weight": .03,
            "multistep_window_fraction": .5,
            "multistep_window_sampling": "batches",
            "normalization_retained_transition_count": 8,
            "train_trim_norm_cutoff": None,
            "train_trim_state_std": None,
        },
    })

    output = diagnose_batch_fraction(
        checkpoint.parent, tmp_path / "batch_fraction_diagnostic",
        split="validation", windows_per_batch=2, microbatch_size=1,
        window_seeds=(1, 2), particle_seeds=(11, 12), device="cpu",
        fractions=(.25, .5))
    report = json.loads((output / "metrics.json").read_text())
    assert report["objective_name"] == "energy_score"
    assert report["retained_window_counts"] == [2, 2]
    assert report["saved_training_context"]["estimated_epoch_batch_count"] == 4
    assert report["fraction_projection"][0]["expected_scored_batches_per_epoch"] == 1
    assert report["fraction_projection"][1][
        "selected_update_multistep_coefficient"] == .06
    for group in ("all", "drift", "diffusion"):
        summary = report["gradient_groups"][group]
        assert summary["single_scored_update"]["gradient_norm"]["finite_count"] == 4
        assert summary["particle_averaged_batch"][
            "pairwise_seed_cosine"]["finite_count"] == 1
    assert (output / "gradients.npz").is_file()
    assert "Connection to batch fraction" in (output / "quick_summary.md").read_text()
