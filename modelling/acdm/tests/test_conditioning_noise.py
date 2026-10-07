from dataclasses import replace

import torch
import pytest

from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.model import ConditionalEDM


def config(**kwargs):
    return EDMConfig(reduced_dim=2, param_dim=1, history_conditioning=True, history_steps=2,
                     hidden_dim=8, num_blocks=1, num_sampling_steps=3, **kwargs)


def test_joint_loss_denoises_one_condition_target_state():
    model = ConditionalEDM(config())
    batch = dict(current_state=torch.randn(4, 2), next_state=torch.randn(4, 2),
                 history_states=torch.randn(4, 2, 2), params=torch.randn(4, 1))
    loss, metrics = model.loss(batch, sigma=.3, generator=torch.Generator().manual_seed(7))
    loss.backward()
    assert model.backbone.input.in_features == model.config.joint_dim + model.config.noise_embedding_dim
    assert model.backbone.output.out_features == model.config.joint_dim
    torch.testing.assert_close(loss, metrics["target_loss"] + metrics["conditioning_loss"])
    assert model.backbone.output.weight.grad.abs().sum() > 0


def test_sampling_rebuilds_clamped_condition_and_discards_its_prediction():
    class Capture(ConditionalEDM):
        def denoise(self, x, sigma, *args):
            condition, target = self.split_joint(x)
            self.calls.append((sigma.clone(), condition.clone()))
            return torch.cat((torch.full_like(condition, 1.e6), torch.zeros_like(target)), -1)

    model = Capture(config())
    model.calls = []
    current = torch.ones(1, 2)
    history = torch.ones(1, 2, 2) * .5
    params = torch.ones(1, 1)
    model.sample_next(current, params, history_states=history, num_samples=2, seed=51)
    clean = model.conditioning_vector(
        model.normalize_state(current).repeat_interleave(2, 0),
        model.normalize_params(params).repeat_interleave(2, 0),
        model.normalize_history(current, history).repeat_interleave(2, 0),
    )
    sigma0, condition0 = model.calls[0]
    epsilon = (condition0 - clean) / sigma0
    for sigma, condition in model.calls:
        torch.testing.assert_close(condition, clean + sigma * epsilon)


def test_clean_mode_keeps_target_only_head_and_legacy_config_loading():
    cfg = config(conditioning_mode="clean")
    model = ConditionalEDM(cfg)
    assert model.backbone.output.out_features == cfg.reduced_dim
    values = cfg.to_dict()
    del values["conditioning_mode"]
    assert EDMConfig.from_dict(values).conditioning_mode == "clean"
    legacy_joint = values | {"conditioning_noise": True,
                             "conditioning_noise_scale": 1., "conditioning_loss_weight": 1.}
    assert EDMConfig.from_dict(legacy_joint).conditioning_mode == "joint_noised"
    assert replace(cfg, conditioning_mode="joint_noised").joint_dim == cfg.joint_dim


def test_kohl_ddpm_schedule_and_joint_noise_loss():
    model = ConditionalEDM(config(diffusion_formulation="ddpm"))
    torch.testing.assert_close(model.ddpm_betas[[0, -1]], torch.tensor([.0025, .5]))
    with torch.no_grad():
        model.backbone.output.weight.zero_()
        model.backbone.output.bias.zero_()
    batch = dict(current_state=torch.randn(4, 2), next_state=torch.randn(4, 2),
                 history_states=torch.randn(4, 2, 2), params=torch.randn(4, 1))
    captured = []
    hook = model.backbone.register_forward_pre_hook(lambda _, args: captured.append(args))
    loss, metrics = model.loss(batch, diffusion_step=10,
                              generator=torch.Generator().manual_seed(7))
    hook.remove()
    assert captured[0][0].shape == (4, model.config.joint_dim)
    assert torch.equal(captured[0][1], torch.full((4,), 9.))
    torch.testing.assert_close(loss, metrics["target_loss"] + metrics["conditioning_loss"])
    with pytest.raises(ValueError, match="sigma is not used"):
        model.loss(batch, sigma=.3)


def test_kohl_sampling_reuses_one_conditioning_noise_tensor():
    class Capture(ConditionalEDM):
        def predict_noise(self, x, step):
            condition, target = self.split_joint(x)
            self.calls.append((step, condition.clone()))
            return torch.cat((torch.full_like(condition, 1.e6), torch.zeros_like(target)), -1)

    model = Capture(EDMConfig(reduced_dim=2, diffusion_formulation="ddpm",
                              hidden_dim=8, num_blocks=0, noise_embedding_dim=4))
    model.calls = []
    current = torch.ones(1, 2)
    model.sample_next(current, num_samples=2, seed=51)
    clean = model.normalize_state(current).repeat_interleave(2, 0)
    step0, condition0 = model.calls[0]
    index0 = step0 - 1
    epsilon = ((condition0 - model.ddpm_alpha_bars[index0].sqrt() * clean)
               / (1 - model.ddpm_alpha_bars[index0]).sqrt())
    assert [step for step, _ in model.calls] == list(range(20, 0, -1))
    for step, condition in model.calls:
        index = step - 1
        expected = (model.ddpm_alpha_bars[index].sqrt() * clean
                    + (1 - model.ddpm_alpha_bars[index]).sqrt() * epsilon)
        torch.testing.assert_close(condition, expected)
