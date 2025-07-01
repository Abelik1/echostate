import torch
from .utils import compute_spectral_radius

class Reservoir:
    def __init__(self,
                 input_dim,
                 reservoir_size,
                 output_dim,
                 spectral_radius,
                 sparsity,
                 input_scaling,
                 bias_scaling,
                 seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        # input weights and bias
        self.W_in = torch.empty(reservoir_size, input_dim).uniform_(-input_scaling, input_scaling)
        self.W_bias = torch.empty(reservoir_size, 1).uniform_(-bias_scaling, bias_scaling)
        # recurrent reservoir
        self.W = self._initialize_reservoir(reservoir_size,
                                            spectral_radius,
                                            sparsity)

    def _initialize_reservoir(self, size, spectral_radius, sparsity):
        W = torch.randn(size, size)
        mask = torch.rand_like(W) < sparsity
        W = W * mask
        radius = compute_spectral_radius(W)
        W = W * (spectral_radius / radius)
        return W

    def update_batch(self, x, u, leak_rate):
        # x: (B, R), u: (B, input_dim)
        pre = u @ self.W_in.T + x @ self.W.T + self.W_bias.T  # broadcast bias
        return (1 - leak_rate) * x + leak_rate * torch.tanh(pre)