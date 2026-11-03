"""Bounded integration check on real windows; not a fitted-model comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from modelling.data_preparation.prepare_shared_pod_training import SharedPODTrainingData
from .config import SDEConfig
from .imported_oscillator import configure_imported_oscillator
from .losses import EulerMaruyamaNLL
from .model import NeuralSDE
from .train import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=Path(
        'runs/diagnostics/peak_background_fit_0p10_v1/cap10_seed1/model.npz'))
    parser.add_argument('--normalization-checkpoint', type=Path, default=Path(
        'runs/adaptive/neural_sde_0p10_tcn63_untrimmed_structured/trials/'
        'untrimmed_variance_w0p001_lr1e6/best.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--residual-input', choices=('state', 'core_innovation'), default='state')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    saved = torch.load(args.normalization_checkpoint, map_location='cpu', weights_only=True)
    data = SharedPODTrainingData(**saved['data_config'], batch_size=64)
    stats = SimpleNamespace(**{name: saved['model_state'][name] for name in
        ('state_mean', 'state_std', 'target_mean', 'target_std')})
    # Read a fixed, small set of native-rate windows from every eligible rep.
    starts = (1000, 25000, 50000, 75000)
    history_frames = 63 + int(args.residual_input == 'core_innovation')
    def windows(names):
        rows = [data.read_coordinates(name, start-history_frames, start+2)[1]
                for name in names for start in starts]
        x = torch.tensor(np.stack([r[history_frames] for r in rows]), dtype=torch.float32)
        y = torch.tensor(np.stack([r[history_frames+1] for r in rows]), dtype=torch.float32)
        h = torch.tensor(np.stack([r[:history_frames][::-1].copy() for r in rows]), dtype=torch.float32)
        return x, y, h
    training, validation = windows(data.train_repetitions), windows(data.validation_repetitions)
    report = dict(purpose='Integration smoke only: five optimizer updates on 48 training windows; no PSD conclusion',
        normalization_checkpoint=str(args.normalization_checkpoint.resolve()),
        normalization_only=True, neural_weights_imported=False,
        train_repetitions=list(data.train_repetitions), validation_repetitions=list(data.validation_repetitions),
        starts_native_frames=list(starts), history_frames=history_frames,
        residual_input=args.residual_input, optimizer_steps=5, models={})
    for architecture in (('tcn',) if args.residual_input == 'core_innovation' else ('mlp', 'tcn')):
        torch.manual_seed(17)
        cfg = SDEConfig(reduced_dim=data.n_coordinates, dt=data.native_dt,
            drift_type='imported_oscillator_residual', state_variable='velocity',
            history_steps=history_frames, history_offsets=(1, 8, 32, 63), hidden_layers=(16,16),
            residual_input=args.residual_input,
            residual_architecture=architecture, tcn_context_frames=63, tcn_channels=16,
            tcn_kernel_size=3, tcn_dilations=(1,2,4,8,16), diffusion_type='diagonal')
        model = NeuralSDE(cfg)
        model.set_normalization(stats)
        provenance = configure_imported_oscillator(model, data, args.artifact)
        model.tcn_rollout_cache = architecture == 'tcn'
        fixed_before = model.drift_model.physical_stiffness.clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        objective = EulerMaruyamaNLL()
        x,y,h = training
        def score(batch):
            with torch.no_grad():
                a,b,c = batch
                return float(objective(model,a,b,history_states=c))
        initial = dict(train_nll=score(training),validation_nll=score(validation))
        gradients = []
        for _ in range(5):
            optimizer.zero_grad()
            loss = objective(model,x,y,history_states=h)
            loss.backward()
            gradients.append({name: float(torch.sqrt(sum(p.grad.square().sum() for p in module.parameters()
                if p.grad is not None))) for name,module in
                [('residual',model.drift_model),('diffusion',model.diffusion_model)]})
            optimizer.step()
        fixed_unchanged = torch.equal(fixed_before, model.drift_model.physical_stiffness)
        assert fixed_unchanged
        a,b,c = validation
        with torch.no_grad():
            fixed = model.imported_oscillator_drift(a,c)
            residual = model.drift(a,c)-fixed
            rollout = model.rollout(a[:2],horizon=1024,num_trajectories=2,history_states=c[:2],seed=19)
            normalized = (rollout-model.state_mean)/model.state_std
        path = args.output/f'{architecture}_smoke.pt'
        torch.save(dict(model_config=cfg.to_dict(),model_state=model.state_dict(),
            training=dict(imported_oscillator=provenance,tcn_rollout_cache=model.tcn_rollout_cache,
                          smoke_only=True), data_config=saved['data_config']),path)
        loaded,_ = load_model(path)
        torch.testing.assert_close(loaded.drift(a,c),model.drift(a,c))
        report['models'][architecture] = dict(initial=initial,final=dict(train_nll=score(training),
            validation_nll=score(validation)),gradient_norms=gradients,fixed_operators_unchanged=fixed_unchanged,
            fixed_normalized_rms=float((fixed/model.target_std).square().mean().sqrt()),
            residual_normalized_rms=float((residual/model.target_std).square().mean().sqrt()),
            rollout_horizon=1024,rollout_paths=4,rollout_finite=bool(torch.isfinite(rollout).all()),
            rollout_max_absolute_standardized_position=float(normalized.abs().max()),
            checkpoint_drift_reload_matches=True,model_config=cfg.to_dict(),provenance=provenance)
        print(architecture, report['models'][architecture]['final'], flush=True)
    report['elapsed_seconds']=time.perf_counter()-started
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"Smoke completed in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
