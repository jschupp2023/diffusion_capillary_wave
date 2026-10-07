import torch
from modelling.acdm.conditional_edm.config import EDMConfig
from modelling.acdm.conditional_edm.model import ConditionalEDM
from modelling.acdm.conditional_edm.checkpoint import save_checkpoint
from modelling.acdm.experiments.run_stability_experiments import ResidentTransitions, warm_start


def test_resident_sampler_matches_sources_and_preserves_boundaries(shared_data):
    cfg = EDMConfig(reduced_dim=3, history_conditioning=True, history_steps=2, lag_steps=2)
    data = ResidentTransitions(shared_data, torch.device('cpu'))
    batch = data.sample(cfg, 100, torch.Generator().manual_seed(3))
    eligible = []
    for name in shared_data.train_repetitions:
        _, values = shared_data.read_coordinates(name, 0, shared_data.frame_counts[name])
        v = torch.as_tensor(values, dtype=torch.float32)
        for i in range(4, len(v)-2):
            eligible.append((v[i],v[i+2],v[[i-2,i-4]]))
    for current,nxt,history in zip(batch['current_state'],batch['next_state'],batch['history_states']):
        assert any(torch.equal(current,a) and torch.equal(nxt,b) and torch.equal(history,c) for a,b,c in eligible)


def test_extended_history_starts_with_identical_target_prediction(tmp_path):
    cfg = EDMConfig(reduced_dim=2, history_conditioning=True, history_steps=2,
                    conditioning_mode="clean", hidden_dim=8,num_blocks=1)
    old = ConditionalEDM(cfg).eval()
    path=tmp_path/'model.pt'
    save_checkpoint(path,{'model_config':cfg.to_dict(),'model_state':old.state_dict()})
    new,_=warm_start(path,8,torch.device('cpu'))
    new.eval()
    current=torch.randn(3,2); history=torch.randn(3,8,2); x=torch.randn(3,2)
    a=old.denoise(x,.3,old.normalize_state(current),history_increments=old.normalize_history(current,history[:,:2]))
    b=new.denoise(x,.3,new.normalize_state(current),history_increments=new.normalize_history(current,history))
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_ddpm_extended_history_preserves_target_noise_prediction(tmp_path):
    cfg = EDMConfig(reduced_dim=2, history_conditioning=True, history_steps=2,
                    diffusion_formulation="ddpm", hidden_dim=8, num_blocks=1)
    old = ConditionalEDM(cfg).eval()
    path = tmp_path / "ddpm.pt"
    save_checkpoint(path, {"model_config": cfg.to_dict(), "model_state": old.state_dict()})
    new, _ = warm_start(path, 8, torch.device("cpu"))
    current = torch.randn(3, 2)
    old_history, new_history = torch.randn(3, 4), torch.randn(3, 16)
    new_history[:, :4] = old_history
    target = torch.randn(3, 2)
    old_x = torch.cat((current, old_history, target), -1)
    new_x = torch.cat((current, new_history, target), -1)
    torch.testing.assert_close(old.split_joint(old.predict_noise(old_x, 7))[1],
                               new.split_joint(new.predict_noise(new_x, 7))[1], rtol=0, atol=0)
