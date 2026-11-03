"""Imported physical poles: units, residual learning, caching, and saved training."""
import copy
import hashlib
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from modelling.neural_sde.config import SDEConfig
from modelling.neural_sde.imported_oscillator import configure_imported_oscillator
from modelling.neural_sde.losses import EulerMaruyamaNLL, MonteCarloMultistepNLL
from modelling.neural_sde.model import NeuralSDE
from modelling.neural_sde.train import (
    _epoch, _validation_change_batches, load_model, main, train,
)


def config(**kwargs):
    values = dict(reduced_dim=3, dt=.01, hidden_layers=(5,), state_variable='velocity',
                  history_steps=1, drift_type='imported_oscillator_residual')
    values.update(kwargs)
    return SDEConfig(**values)


def configure(model):
    model.drift_model.configure(
        [[100., 20., 0.], [0., 80., -10.], [2., 0., 60.]],
        [[2., .2, 0.], [0., 3., .1], [.1, 0., 1.]], [1., -2., 3.], model)
    return model


def artifact_for(data, directory):
    directory.mkdir(parents=True)
    d = data.n_coordinates
    t = np.eye(d); t[0, 1] = .4
    omega, gamma = np.arange(1, d+1)*10., np.arange(1, d+1)*.5
    scale, mean = np.arange(1, d+1)*.7, np.arange(d)-.5
    artifact = directory/'model.npz'
    np.savez(artifact, T=t, K=(t*omega**2)@np.linalg.inv(t),
             D=(t*(2*gamma))@np.linalg.inv(t), omega_per_second=omega,
             gamma_per_second=gamma, mean=mean, scale=scale,
             prescribed_peaks_hz=omega[:-1]/(2*np.pi), fs=1/data.native_dt,
             resonant_spectrum=np.zeros((1,d,d)))
    source = dict(single_rollout=dict(shared_basis=str(data.basis_path), rank=data.rank,
                  signal_units=data.signal_units, spatial_mean_highpass_hz=data.spatial_mean_highpass_hz,
                  sampling_frequency_hz=1/data.native_dt),
                  training_repetitions=[int(n.split('rep')[-1]) for n in data.train_repetitions],
                  validation_repetitions=[int(n.split('rep')[-1]) for n in data.validation_repetitions],
                  downsample_factor=1,
                  coordinate_units=f'all coefficients divided by sqrt(N)={np.sqrt(data.n_space):g}; coordinate 0 is physical spatial mean')
    protocol = dict(nominal_peaks_hz=(omega[:-1]/(2*np.pi)).tolist(),
                    background_protocol=dict(source_protocol=dict(source_protocol=dict(source_provenance=source))))
    (directory.parent/'protocol.json').write_text(json.dumps(protocol))
    return artifact


def forcing_artifact_for(oscillator_artifact, directory, dimension, dt):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'modal_biquad.npz'
    np.savez(path, h0=np.ones(dimension), h1=np.full(dimension, -.4),
             h2=np.full(dimension, .1), d1=np.full(dimension, -.2),
             d2=np.full(dimension, .02), dt_seconds=dt, dimension=dimension,
             oscillator_artifact_sha256=np.asarray(
                 hashlib.sha256(oscillator_artifact.read_bytes()).hexdigest()))
    return path


def test_affine_conversion_preserves_physical_drift_after_normalization_change():
    model = configure(NeuralSDE(config()).double())
    x, h = torch.randn(4,3,dtype=torch.double), torch.randn(4,1,3,dtype=torch.double)
    expected = (-.01**2*(x-model.drift_model.equilibrium)@model.drift_model.physical_stiffness.T
                -.01*(x-h[:,0])@model.drift_model.physical_damping.T)
    for shift in (0., 3.):
        model.set_normalization(SimpleNamespace(state_mean=torch.tensor([2.,-4.,3.])+shift,
            state_std=torch.tensor([2.,4.,6.]), target_mean=torch.tensor([.2,-.3,.1])+shift,
            target_std=torch.tensor([.3,.7,1.1])))
        torch.testing.assert_close(model.drift(x,h), expected, rtol=1e-10, atol=1e-10)
    assert all('physical_' not in name for name,_ in model.named_parameters())
    assert torch.count_nonzero(model.nonlinear_drift(x,h)) == 0


@pytest.mark.parametrize('architecture,residual_input', [
    ('mlp', 'state'), ('tcn', 'state'), ('tcn', 'core_innovation')])
@pytest.mark.parametrize('diffusion', ['diagonal','full','state_diagonal','bounded_state_diagonal'])
def test_learning_gradients_and_rollout_cache(architecture, residual_input, diffusion):
    if residual_input == 'core_innovation' and diffusion not in {'diagonal', 'full'}:
        with pytest.raises(ValueError, match='constant diffusion'):
            config(residual_architecture=architecture, residual_input=residual_input,
                   diffusion_type=diffusion)
        return
    torch.manual_seed(7)
    model = configure(NeuralSDE(config(residual_architecture=architecture,
        residual_input=residual_input,
        diffusion_type=diffusion, tcn_context_frames=3, tcn_channels=4,
        tcn_kernel_size=3, tcn_dilations=(1,), history_offsets=(1,3), history_steps=3)).double())
    x = torch.randn(4,3,dtype=torch.double)
    h = torch.randn(4,model.config.history_steps,3,dtype=torch.double)
    y = x+.1*torch.randn_like(x)
    loss = EulerMaruyamaNLL()(model,x,y,history_states=h)
    loss.backward()
    assert sum(float(p.grad.square().sum()) for p in model.drift_model.parameters() if p.grad is not None)>0
    assert sum(float(p.grad.square().sum()) for p in model.diffusion_model.parameters() if p.grad is not None)>0
    for p in model.drift_model.parameters():
        with torch.no_grad(): p.uniform_(-.03,.03)
    cached = copy.deepcopy(model); cached.tcn_rollout_cache = True
    outputs=[]
    targets=torch.randn(4,2,3,dtype=torch.double)
    for candidate in (model,cached):
        candidate.zero_grad()
        objective=MonteCarloMultistepNLL((1,5),particle_counts=(3,2))
        value=objective(candidate,x,targets,history_states=h,generator=torch.Generator().manual_seed(8))
        value.backward()
        outputs.append((value.detach(),[p.grad for p in candidate.parameters()]))
    torch.testing.assert_close(outputs[0][0],outputs[1][0])
    for a,b in zip(outputs[0][1],outputs[1][1]):
        if a is None: assert b is None
        else: torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
    torch.testing.assert_close(model.rollout(x,horizon=15,num_trajectories=2,history_states=h,seed=5),
        cached.rollout(x,horizon=15,num_trajectories=2,history_states=h,seed=5))


def test_artifact_conversion_and_completion_exclusion(shared_data,tmp_path):
    artifact=artifact_for(shared_data,tmp_path/'fit'/'candidate')
    model=NeuralSDE(config(dt=shared_data.native_dt)).double()
    report=configure_imported_oscillator(model,shared_data,artifact)
    with np.load(artifact) as a:
        scales=np.sqrt(shared_data.n_space)*a['scale']
        direction=scales*a['T'][:,-1]
        np.testing.assert_allclose(model.drift_model.physical_stiffness@direction,0,atol=1e-10)
        np.testing.assert_allclose(model.drift_model.physical_damping@direction,0,atol=1e-10)
        for j in range(2):
            direction=scales*a['T'][:,j]
            np.testing.assert_allclose(model.drift_model.physical_stiffness@direction,a['omega_per_second'][j]**2*direction)
        np.testing.assert_allclose(model.drift_model.equilibrium,np.sqrt(shared_data.n_space)*a['mean'])
    assert report['resonant_directions']==2
    assert not report['background_and_spectral_forcing_imported']
    bad=copy.copy(shared_data); bad.signal_units='radians'
    with pytest.raises(ValueError,match='units'): configure_imported_oscillator(model,bad,artifact)
    bad=copy.copy(shared_data); bad.downsample_factor=8
    with pytest.raises(ValueError,match='native'): configure_imported_oscillator(model,bad,artifact)


def test_diffusion_warmup_and_checkpoint_without_source(shared_data,tmp_path):
    artifact=artifact_for(shared_data,tmp_path/'fit'/'candidate')
    cfg=config(dt=shared_data.native_dt)
    path=train(shared_data,tmp_path/'diffusion',cfg,oscillator_artifact=artifact,
               epochs=1,batch_size=8,train_mode='diffusion')
    model,checkpoint=load_model(path)
    metrics={}
    # Epoch validation must keep the physical prior active during diffusion-only fitting.
    actual=_epoch(model,shared_data,'validation',8,'cpu',EulerMaruyamaNLL(),train_mode='diffusion',metric_output=metrics)
    expected=_epoch(model,shared_data,'validation',8,'cpu',EulerMaruyamaNLL(),train_mode='joint')
    assert actual==pytest.approx(expected)
    assert metrics['imported_fixed_normalized_rms_first_batch']>0
    assert metrics['imported_residual_normalized_rms_first_batch']==0
    for batch,innovation in _validation_change_batches(shared_data,model,8):
        x=torch.as_tensor(batch['current_state'],dtype=torch.float32)
        h=torch.as_tensor(batch['history_states'],dtype=torch.float32)
        raw=(batch['next_state']-batch['next_history_states'][:,0])-(batch['current_state']-batch['history_states'][:,0])
        np.testing.assert_allclose(innovation,raw-model.imported_oscillator_drift(x,h).detach().numpy())
    artifact.unlink(); artifact.parent.parent.joinpath('protocol.json').unlink()
    reloaded,_=load_model(path)
    assert reloaded.drift_model.ready
    torch.testing.assert_close(reloaded.drift_model.physical_stiffness,model.drift_model.physical_stiffness)
    drift_path=train(shared_data,tmp_path/'drift',cfg,epochs=1,batch_size=8,train_mode='drift',initial_checkpoint=checkpoint)
    trained,_=load_model(drift_path)
    torch.testing.assert_close(trained.drift_model.physical_stiffness,model.drift_model.physical_stiffness)
    assert any(not torch.equal(a,b) for a,b in zip(trained.drift_model.parameters(),model.drift_model.parameters()))


@pytest.mark.parametrize('residual_input', ['state', 'core_innovation'])
def test_cli_training_with_trim(shared_data,tmp_path,residual_input):
    with h5py.File(shared_data.basis_path, 'r+') as basis:
        for name in basis['repetitions']:
            basis[f'repetitions/{name}'].attrs['signal_units'] = 'microns'
    artifact=artifact_for(shared_data,tmp_path/'fit'/'candidate')
    output=tmp_path/'cli'
    extra = ([] if residual_input == 'state' else [
        '--residual-architecture','tcn','--residual-input','core_innovation',
        '--tcn-context-frames','3','--tcn-channels','4','--tcn-dilations','1',
        '--tcn-rollout-cache'])
    main(['train','--experiment',shared_data.experiment,'--rank','2','--data-root',shared_data.source_config['data_root'],
          '--state-variable','velocity','--history','1','--drift-type','imported_oscillator_residual',
          '--oscillator-artifact',str(artifact),'--hidden-layers','5','--epochs','1','--batch-size','8',
          '--train-trim-percent','1','--device','cpu','--output',str(output), *extra])
    model,checkpoint=load_model(output/'best.pt')
    assert model.config.residual_input == residual_input
    assert checkpoint['training']['imported_oscillator']['fixed']
    log=json.loads((output/'metrics.jsonl').read_text())
    assert log['validation_imported_fixed_normalized_rms_first_batch']>0
    old=model.drift_model.stiffness.clone(); model.drift_model.refresh_normalization(model)
    torch.testing.assert_close(old,model.drift_model.stiffness)
    evaluation=tmp_path/'evaluation'
    main(['evaluate','--checkpoint',str(output/'best.pt'),'--split','validation',
          '--num-conditions','2','--ensemble-size','2','--horizon','4',
          '--no-metrics','--device','cpu','--output',str(evaluation)])
    report=json.loads((evaluation/'metrics.json').read_text())
    assert report['residual_input'] == residual_input
    np.testing.assert_allclose(report['physical_stiffness_matrix'], model.drift_model.physical_stiffness)
    assert report['imported_oscillator']['resonant_directions']==2


@pytest.mark.parametrize('frames', [1, 2, 3])
def test_core_innovation_recovers_forcing_and_ignores_homogeneous_motion(frames):
    """Same driving sequence, different oscillator initial states => same TCN inputs."""
    torch.manual_seed(12)
    model=configure(NeuralSDE(config(residual_architecture='tcn',
        residual_input='core_innovation', tcn_context_frames=frames,
        tcn_channels=4, tcn_kernel_size=1 if frames == 1 else 3,
        tcn_dilations=(1,))).double())
    model.set_normalization(SimpleNamespace(state_mean=torch.tensor([2.,-4.,3.]),
        state_std=torch.tensor([2.,4.,6.]), target_mean=torch.tensor([.2,-.3,.1]),
        target_std=torch.tensor([.3,.7,1.1])))
    for p in model.drift_model.parameters():
        with torch.no_grad(): p.uniform_(-.03,.03)
    forcing=torch.randn(8,2,3,dtype=torch.double)*.01
    outputs=[]
    for shift in (0.,5.):
        x=torch.randn(2,3,dtype=torch.double)+shift
        u=torch.randn_like(x)*.01
        positions=[x-u,x]
        for e in forcing:
            core=(-model.config.dt**2*(x-model.drift_model.equilibrium)
                  @model.drift_model.physical_stiffness.T
                  -model.config.dt*u@model.drift_model.physical_damping.T)
            u=u+core+e
            x=x+u
            positions.append(x)
        history=torch.stack(positions[-2:-frames-3:-1],dim=-2)
        tokens=model._tcn_input(x,history,u)
        torch.testing.assert_close(tokens,forcing[-frames:].transpose(0,1)/model.target_std,
                                   atol=1e-12,rtol=1e-10)
        outputs.append(model.nonlinear_drift(x,history,u))
        cached=copy.deepcopy(model); cached.tcn_rollout_cache=True
        torch.testing.assert_close(model.rollout(x,horizon=8,num_trajectories=2,history_states=history,seed=3),
                                   cached.rollout(x,horizon=8,num_trajectories=2,history_states=history,seed=3))
    torch.testing.assert_close(*outputs,atol=1e-12,rtol=1e-10)
    with pytest.raises(ValueError,match='preceding states'):
        model._tcn_input(x,history[...,:frames,:])


def test_core_innovation_config_is_explicit_and_backward_compatible():
    legacy=config().to_dict()
    legacy.pop('residual_input')
    assert SDEConfig.from_dict(legacy).residual_input == 'state'
    cfg=config(residual_architecture='tcn',residual_input='core_innovation')
    assert cfg.history_steps == 64
    assert SDEConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError,match='imported oscillator TCN'):
        config(residual_input='core_innovation')
    with pytest.raises(ValueError,match='imported oscillator TCN'):
        config(residual_input='core_innovation',residual_architecture='tcn',drift_type='mlp')


def test_modal_biquad_filter_inverse_likelihood_and_rollout():
    torch.manual_seed(23)
    cfg = config(residual_architecture='tcn', residual_input='core_innovation',
                 forcing_filter='modal_biquad', forcing_filter_burnin=4,
                 tcn_context_frames=3, tcn_channels=4, tcn_kernel_size=3,
                 tcn_dilations=(1,))
    model = NeuralSDE(cfg).double()
    model.drift_model.configure(
        np.diag([40., 60., 80.]), np.diag([1., 2., 3.]), np.zeros(3), model,
        modal_transform=np.array([[1., .2, 0.], [0., 1., .1], [.1, 0., 1.]]))
    model.drift_model.configure_forcing_filter(
        [1., 1., 1.], [-1.8, -1.7, -1.6], [.82, .75, .68],
        [-.4, -.3, -.2], [.06, .04, .02])
    raw = torch.randn(7, 3, dtype=torch.double)
    history = torch.randn(7, 2, 3, dtype=torch.double)
    filtered_history = torch.randn(7, 2, 3, dtype=torch.double)
    filtered = model.drift_model.filter_forcing(raw, history, filtered_history)
    torch.testing.assert_close(
        model.drift_model.inverse_filter_forcing(filtered, history, filtered_history), raw)
    forcing_history = torch.randn(7, 3, 3, dtype=torch.double)
    target = torch.randn(7, 3, dtype=torch.double)
    x = torch.randn(7, 3, dtype=torch.double)
    h = torch.randn(7, cfg.history_steps, 3, dtype=torch.double)
    y = torch.randn_like(x)
    loss = EulerMaruyamaNLL()(model, x, y, history_states=h,
                              forcing_history=forcing_history,
                              raw_forcing_target=target)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in model.drift_model.residual.parameters())
    result = model.rollout(x[:2], horizon=8, num_trajectories=2,
                           history_states=h[:2], seed=4)
    assert result.shape == (2, 2, 9, 3)
    assert torch.isfinite(result).all()


def test_modal_biquad_config_is_opt_in_and_rejects_unsupported_forms():
    legacy = config().to_dict()
    legacy.pop('forcing_filter')
    legacy.pop('forcing_filter_burnin')
    assert SDEConfig.from_dict(legacy).forcing_filter == 'none'
    with pytest.raises(ValueError, match='core-innovation'):
        config(forcing_filter='modal_biquad')


def test_modal_biquad_training_cache_and_checkpoint(shared_data, tmp_path):
    oscillator = artifact_for(shared_data, tmp_path/'fit'/'candidate')
    forcing = forcing_artifact_for(
        oscillator, tmp_path/'filter', shared_data.n_coordinates, shared_data.native_dt)
    cfg = config(dt=shared_data.native_dt, residual_architecture='tcn',
                 residual_input='core_innovation', forcing_filter='modal_biquad',
                 forcing_filter_burnin=4, tcn_context_frames=3, tcn_channels=4,
                 tcn_kernel_size=3, tcn_dilations=(1,))
    path = train(shared_data, tmp_path/'filtered', cfg,
                 oscillator_artifact=oscillator, forcing_filter_artifact=forcing,
                 epochs=1, batch_size=8)
    loaded, checkpoint = load_model(path)
    assert loaded.drift_model.filter_ready
    cache_report = checkpoint['training']['forcing_filter']['cache']
    assert cache_report['retained_count'] > 0
    assert cache_report['inverse_reconstruction_relative_error'] < 1e-12
    assert np.isfinite(json.loads((tmp_path/'filtered'/'metrics.jsonl').read_text())[
        'validation_nll'])


def test_unconfigured_and_invalid_modes():
    model=NeuralSDE(config())
    with pytest.raises(RuntimeError,match='configured'): model.drift(torch.zeros(1,3),torch.zeros(1,1,3))
    with pytest.raises(ValueError,match='native-lag'): config(lag_steps=2)
    with pytest.raises(ValueError,match='velocity'): config(state_variable='state')


def test_full_resonant_import_with_explicit_rank_selection(shared_data,tmp_path):
    artifact=artifact_for(shared_data,tmp_path/'fit'/'candidate')
    with np.load(artifact) as a:
        values={key:a[key] for key in a.files}
    values['prescribed_peaks_hz']=values['omega_per_second']/(2*np.pi)
    np.savez(artifact,**values)
    path=artifact.parent.parent/'protocol.json'
    protocol=json.loads(path.read_text())
    protocol['nominal_peaks_hz']=values['prescribed_peaks_hz'].tolist()
    protocol['background_protocol']['source_protocol']['source_protocol']['source_provenance']['single_rollout']['rank']=3
    protocol['coordinate_selection']=dict(kind='leading_with_spatial_mean',source_dimension=4,dimension=3,pod_rank=2)
    path.write_text(json.dumps(protocol))
    model=NeuralSDE(config(dt=shared_data.native_dt))
    report=configure_imported_oscillator(model,shared_data,artifact)
    assert report['excluded_completion_directions']==0
    assert np.linalg.matrix_rank(model.drift_model.physical_stiffness.numpy())==3
    protocol['coordinate_selection']['dimension']=4
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='coordinate-selection'):
        configure_imported_oscillator(model,shared_data,artifact)
