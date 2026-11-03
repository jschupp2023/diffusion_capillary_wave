"""Controller provenance, matched controls and finite process budgets."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import load_model
from modelling.neural_sde.experiments.fixed_memory_study import materialize_variant, assert_matched, run_study
from modelling.neural_sde.experiments.adaptive_controller import run_logged


def test_exploratory_probe_preserves_selected_and_frozen_parent(tmp_path):
    model = NeuralSDE(SDEConfig(2, .001, history_steps=1, state_variable="velocity",
        fixed_memory_frequencies=(40.,), fixed_memory_damping=(10.,), fixed_memory_window=8))
    fit = tmp_path/"fit"
    fit.mkdir()
    payload = dict(model_config=model.config.to_dict(), model_state=model.state_dict(), training={}, data_config={})
    torch.save(payload, fit/"best.pt")
    (fit/"report.json").write_text(json.dumps(dict(selected_penalty=None, scores=[
        dict(penalty=None, validation_error=1.), dict(penalty=.1, validation_error=1.1)])))
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32,2))
    np.savez(fit/"regression.npz", train_features=x, train_residuals=x*.1)
    probe = tmp_path/"probe"
    details = materialize_variant(fit, probe)
    assert details["variant"] == "exploratory_nonzero_rejected_by_one_step_selection"
    loaded, _ = load_model(probe/"best.pt")
    assert torch.count_nonzero(loaded.fixed_memory.readout.weight)
    original, _ = load_model(fit/"best.pt")
    assert not torch.count_nonzero(original.fixed_memory.readout.weight)
    for name, value in model.state_dict().items():
        if not name.startswith("fixed_memory."):
            torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    baseline = tmp_path/"parent"
    materialize_variant(fit, baseline, zero=True)
    zero, _ = load_model(baseline/"best.pt")
    assert zero.config.history_steps == loaded.config.history_steps
    assert not torch.count_nonzero(zero.fixed_memory.readout.weight)


def test_reference_matching_detects_different_origins(tmp_path):
    a, b = tmp_path/"a.npz", tmp_path/"b.npz"
    values = dict(reference=np.zeros((2,4,2)), reference_time=np.arange(4),
                  repetition=np.array(["rep13","rep14"]), start_index=np.array([8,9]))
    np.savez(a, **values)
    np.savez(b, **values)
    assert assert_matched(a,b)["start_indices"] == [8,9]
    values["start_index"][1] = 10
    np.savez(b, **values)
    with pytest.raises(ValueError, match="start_index"):
        assert_matched(a,b)


def test_timeout_kills_a_process_ignoring_term(tmp_path):
    started = time.monotonic()
    command = [sys.executable, "-u", "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"]
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_logged(command, tmp_path/"timeout.log", tmp_path, timeout_seconds=.5)
    assert caught.value.returncode == 124
    assert time.monotonic()-started < 12
    assert "TIMEOUT" in (tmp_path/"timeout.log").read_text()


def test_study_dry_run_does_not_launch_or_write(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[3]
    source = repo/"modelling/neural_sde/experiments/0p10_fixed_memory_study.json"
    study = json.loads(source.read_text())
    study.update(repository=str(tmp_path), output_root="unused")
    path = tmp_path/"study.json"
    path.write_text(json.dumps(study))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    run_study(path, dry_run=True)
    assert not (tmp_path/"unused").exists()
