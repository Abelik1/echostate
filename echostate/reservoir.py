import torch
from .utils import compute_spectral_radius

class Reservoir:
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        spectral_radius: float,
        sparsity: float,
        input_scaling: float,
        bias_scaling: float,
        seed: int = None,
        device: torch.device = torch.device('cpu'),
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device
        self.reservoir_size = reservoir_size

        # input weights
        self.W_in = torch.empty(reservoir_size, input_dim, device=self.device)
        self.W_in.uniform_(-input_scaling, input_scaling)

        # bias weights
        self.W_bias = torch.empty(reservoir_size, 1, device=self.device)
        self.W_bias.uniform_(-bias_scaling, bias_scaling)
        self._W_bias = self.W_bias.squeeze()  # shape (reservoir_size,)

        # reservoir recurrent weights on CPU for eigen-decomp
        W = self._initialize_reservoir(reservoir_size, spectral_radius, sparsity)
        self.W = W.to(self.device)

    def _initialize_reservoir(self, size, spectral_radius, sparsity):
        W = torch.randn(size, size)              # CPU
        mask = (torch.rand(size, size) < sparsity)
        W = W * mask
        radius = compute_spectral_radius(W)
        W = W * (spectral_radius / radius)
        return W

    def update_batch(self, x: torch.Tensor, u: torch.Tensor, leak_rate: float) -> torch.Tensor:
        # ensure on correct device
        if x.device != self.device: #TODO REMOVE THESE LATER CHECK, OVERHEAD
            x = x.to(self.device)
        if u.device != self.device:
            u = u.to(self.device)

        single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            single = True

        pre = u @ self.W_in.T + x @ self.W.T + self._W_bias
        x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)
        return x_new.squeeze(0) if single else x_new
