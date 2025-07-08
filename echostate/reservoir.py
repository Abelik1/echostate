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
                 seed=None,
                 device=torch.device("cpu")):
        
        if seed is not None:
            torch.manual_seed(seed)
        print(device)
        self.device= device
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim

        # Move weights to device after creation
        self.W_in = torch.empty(reservoir_size, input_dim).uniform_(-input_scaling, input_scaling).to(self.device)
        self.W_bias = torch.empty(reservoir_size, 1).uniform_(-bias_scaling, bias_scaling).to(self.device)
        self.W = self._initialize_reservoir(reservoir_size, spectral_radius, sparsity).to(self.device)

    def _initialize_reservoir(self, size, spectral_radius, sparsity):
        W = torch.randn(size, size)
        mask = torch.rand_like(W) < sparsity
        W = W * mask
        radius = compute_spectral_radius(W)
        W = W * (spectral_radius / radius)
        return W

    def update_batch(self, x, u, leak_rate):
        """
        Vectorized reservoir update for a batch of inputs, or a single 1-D x.
        """
        # Make sure bias is 1-D of length R
        x = x.to(self.device)
        u = u.to(self.device)
        bias = self.W_bias.squeeze().to(self.device)            # shape (R,)

        # If x is 1-D, treat it as batch of size 1
        single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)                  # (1, R) #TODO REMOVE ALL THESE CHECKS FOR LATER
            u = u.unsqueeze(0)                  # (1, input_dim)
            single = True

        # now u @ W_in.T is (B, R), same for x @ W.T
        pre = u @ self.W_in.T + x @ self.W.T + bias

        x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)

        return x_new.squeeze(0) if single else x_new