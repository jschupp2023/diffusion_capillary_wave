"""Fixed, finite-window resonant filters, not inferred physical oscillator states.

h[t] = gain * sum_{j=0}^{L-1} rho**j * P.T @ standardized_v[t-j].
rho = exp((-gamma + 2 pi i f) dt). The expired forcing is subtracted in
rollout, so cached observations and recurrent generation use exactly the
same finite-history definition. No gradients through a long rollout are needed.
"""
import numpy as np
import torch
from torch import nn
from scipy.signal import lfilter


class FixedOscillatorMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window = config.fixed_memory_window
        f = np.asarray(config.fixed_memory_frequencies)
        g = np.asarray(config.fixed_memory_damping)
        rho = np.exp((-g + 2j * np.pi * f) * config.dt)
        self.register_buffer("pole", torch.tensor(np.c_[rho.real, rho.imag], dtype=torch.float32))
        # Derive truncation from the serialized poles, rather than a slightly
        # different float64 pole, to retain the finite-window cancellation.
        rounded = self.pole.numpy()
        tail = (rounded[:, 0].astype(np.float64) + 1j * rounded[:, 1]) ** self.window
        self.register_buffer("tail", torch.tensor(np.c_[tail.real, tail.imag], dtype=torch.float32))
        self.register_buffer("gain", torch.tensor(np.sqrt(1 - abs(rho)**2), dtype=torch.float32))
        # Set explicitly by the fitter; serialized alongside the readout.
        self.register_buffer("projection", torch.zeros(config.reduced_dim, len(f)))
        self.register_buffer("feature_scale", torch.ones(2 * len(f)))
        self.readout = nn.Linear(2 * len(f), config.reduced_dim, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forcing(self, velocity, mean, std):
        return ((velocity - mean) / std) @ self.projection * self.gain

    def initialize(self, state, history, mean, std):
        if history is None or history.shape[-2] < self.window:
            raise ValueError("fixed memory requires its complete observed history window")
        positions = torch.cat((state.unsqueeze(-2), history[..., :self.window, :]), -2)
        forcing = self.forcing(positions[..., :-1, :] - positions[..., 1:, :], mean, std)
        pole = torch.view_as_complex(self.pole.contiguous())
        powers = pole[None, :] ** torch.arange(self.window, device=state.device)[:, None]
        value = (forcing * powers).sum(-2)
        return torch.view_as_real(value), forcing

    def advance(self, memory, new_forcing, expired_forcing):
        real, imag = memory.unbind(-1)
        a, b = self.pole.unbind(-1)
        return torch.stack((a * real - b * imag + new_forcing - self.tail[:, 0] * expired_forcing,
                            b * real + a * imag - self.tail[:, 1] * expired_forcing), -1)

    def forward(self, memory):
        return self.readout(memory.flatten(-2) / self.feature_scale)

    def cache(self, positions, mean, std):
        """One record only. Feature t uses positions at or before t, never t+1."""
        values = torch.tensor(np.asarray(positions), dtype=torch.float32)
        # Match training's float32 position subtraction before normalizing.
        with torch.no_grad():
            forcing = self.forcing(values[1:] - values[:-1], mean.cpu(), std.cpu()).cpu().numpy()
        rho = self.pole.cpu().numpy()
        rho = rho[:, 0].astype(np.float64) + 1j * rho[:, 1].astype(np.float64)
        # Use the serialized float32 tail for the same recurrence as rollout.
        tail = self.tail.cpu().numpy()
        tail = tail[:, 0] + 1j * tail[:, 1]
        result = np.zeros((len(values), len(rho)), dtype=np.complex128)
        for k, pole in enumerate(rho):
            drive = forcing[:, k].astype(np.complex128)
            drive[self.window:] -= tail[k] * forcing[:-self.window, k]
            result[1:, k] = lfilter([1.], [1., -pole], drive)
        return np.stack((result.real, result.imag), -1).astype(np.float32)
