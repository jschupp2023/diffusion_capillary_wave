"""Separate TF32 convolution effects from streaming/compilation errors."""
import argparse
import json

import torch

from modelling.acdm.conditional_edm.data import checkpoint_data
from modelling.neural_sde.batches import indexed_velocity_batches
from modelling.neural_sde.losses import MonteCarloMultistepNLL
from modelling.neural_sde.train import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    model, checkpoint = load_model(args.checkpoint, "cuda")
    data = checkpoint_data(checkpoint)
    data.cache_coordinates(2.)
    t = checkpoint["training"]
    batches = indexed_velocity_batches(data, model.config, "train", 512, 0,
                                       tuple(t["multistep_horizons"]), t["train_trim_norm_cutoff"],
                                       t["train_trim_state_std"], t["multistep_window_fraction"],
                                       torch.Generator().manual_seed(0))
    for _ in range(11):
        batch = next(batches)
    x, h, y = (batch[k].cuda() for k in ("current_state", "history_states", "target_velocities"))
    loss = MonteCarloMultistepNLL(tuple(t["multistep_horizons"]),
                                particle_counts=tuple(t["multistep_particle_counts"]))
    results = {}
    for name, cache, compile_, tf32 in [
            ("reference_default", False, False, True),
            ("reference_ieee", False, False, False),
            ("cached_eager", True, False, False),
            ("cached_compiled", True, True, False)]:
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        model.tcn_rollout_cache, model.tcn_rollout_compile = cache, compile_
        value = loss(model, x, y, history_states=h,
                     generator=torch.Generator(device="cuda").manual_seed(29))
        gradients = torch.autograd.grad(value, tuple(model.parameters()), allow_unused=True)
        vector = torch.cat([g.flatten() for g in gradients if g is not None]).detach()
        results[name] = (value.detach(), vector)
        ref = results.get("reference_ieee", results["reference_default"])
        print(json.dumps(dict(name=name, loss=value.item(),
                              relative_loss=((value-ref[0]).abs()/ref[0].abs()).item(),
                              relative_gradient=((vector-ref[1]).norm()/ref[1].norm()).item())), flush=True)
    ref, base = results["reference_ieee"], results["reference_default"]
    print("DEFAULT_VS_IEEE", ((base[1]-ref[1]).norm()/ref[1].norm()).item(), flush=True)


if __name__ == "__main__":
    main()
