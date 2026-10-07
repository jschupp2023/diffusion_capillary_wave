from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modelling.acdm.experiments.conditioning_errors import build_error_bank, perturb_batch


class LinearData:
    splits = {"train": ["training"], "validation": ["heldout"], "test": ["test"]}
    frame_counts = {"training": 100, "heldout": 100, "test": 100}

    def read_coordinates(self, name, start, stop):
        assert name == "training", "Error collection must never access held-out data"
        t = np.arange(start, stop)
        return t, t[:, None] * np.array([1., 2.])


class BiasedModel:
    state_mean = torch.zeros(2)
    state_std = torch.ones(2)
    target_std = torch.full((2,), .5)

    def __init__(self, history=2, nonfinite=False):
        self.config = SimpleNamespace(lag_steps=1, history_steps=history, param_dim=0,
                                      history_conditioning=bool(history), num_sampling_steps=4)
        self.nonfinite = nonfinite

    def rollout(self, initial, *, horizon, **kwargs):
        t = torch.arange(horizon + 1)
        values = initial[:, None] + t[None, :, None] * torch.tensor([1.25, 2.5])
        if self.nonfinite:
            values[:, 2] = torch.inf
        return values[:, None]


def test_error_windows_preserve_history_and_true_next():
    bank = build_error_bank(BiasedModel(), LinearData(), num_conditions=2, horizon=3)
    assert bank.errors.shape == (6, 3, 2)
    expected = torch.tensor([[.75, 1.5], [.5, 1.], [.25, .5]])
    torch.testing.assert_close(bank.errors[2], expected)
    torch.testing.assert_close(bank.errors[0, 1:], torch.zeros(2, 2))
    assert bank.summary["source_repetitions"] == ["training", "training"]
    # At t=3, the generated current has accumulated 3*bias; target remains real.
    torch.testing.assert_close(bank.replay["next_state"][2] - bank.replay["current_state"][2],
                               torch.tensor([.25, .5]))


def test_filter_rejects_large_or_nonfinite_windows_without_clipping():
    bank = build_error_bank(BiasedModel(), LinearData(), num_conditions=1, horizon=3,
                            max_error_std=.5)
    assert len(bank) == 1 and bank.summary["rejected"] == 2
    torch.testing.assert_close(bank.errors[0, 0], torch.tensor([.25, .5]))
    nonfinite = build_error_bank(BiasedModel(nonfinite=True), LinearData(), num_conditions=1, horizon=3)
    assert len(nonfinite) == 1  # An invalid history also invalidates its later window.
    with pytest.raises(ValueError, match="No finite"):
        build_error_bank(BiasedModel(), LinearData(), horizon=2, max_error_std=.1)
    target_scaled = build_error_bank(BiasedModel(), LinearData(), horizon=2, num_conditions=1,
                                     max_error_std=1., error_scale="target")
    assert len(target_scaled) == 1


def test_perturbation_is_reproducible_and_keeps_next_state_fixed():
    bank = build_error_bank(BiasedModel(), LinearData(), num_conditions=1, horizon=3)
    batch = dict(current_state=torch.zeros(16, 2), history_states=torch.zeros(16, 2, 2),
                 next_state=torch.ones(16, 2))
    rng_state = torch.get_rng_state()
    one = perturb_batch(batch, bank, strength=.5, generator=torch.Generator().manual_seed(5))
    two = perturb_batch(batch, bank, strength=.5, generator=torch.Generator().manual_seed(5))
    for key in one:
        torch.testing.assert_close(one[key], two[key])
    torch.testing.assert_close(torch.get_rng_state(), rng_state)
    torch.testing.assert_close(one["next_state"], batch["next_state"])
    assert not batch["current_state"].any() and not batch["history_states"].any()
    assert 0 < int(one["current_state"].any(-1).sum()) < len(one["current_state"])
    for current, history in zip(one["current_state"], one["history_states"]):
        window = torch.cat((current[None], history))
        assert not window.any() or any(torch.equal(window, .5 * e) for e in bank.errors)


def test_no_history_replay_and_noop():
    bank = build_error_bank(BiasedModel(history=0), LinearData(), num_conditions=1, horizon=2).to("cpu")
    replay = bank.sample_replay_batch(4, generator=torch.Generator().manual_seed(9))
    assert set(replay) == {"current_state", "next_state"}
    assert replay["current_state"].shape == (4, 2)
    result = perturb_batch(replay, bank, probability=0)
    assert result["current_state"] is replay["current_state"]
    with pytest.raises(ValueError, match="match"):
        perturb_batch(dict(replay, history_states=torch.zeros(4, 2, 2)), bank)
