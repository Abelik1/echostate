import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from ..utils import compute_spectral_radius

LOGGER = logging.getLogger(__name__)

class Reservoir(nn.Module):
    """
    Sparse random reservoir with leaky-tanh dynamics.
    """
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        spectral_radius: float,
        sparsity: float,
        input_scaling: float,
        bias_scaling: float,
        seed: int | None = None,
        device: torch.device = torch.device("cpu"),
        profile: bool = False,
        use_amp: bool = True,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.device = device
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.profile = profile
        self.use_amp = use_amp
        self._warned_dev = False

        # Input weights
        W_in = torch.empty(reservoir_size, input_dim)
        W_in.uniform_(-input_scaling, input_scaling)
        self.register_buffer("W_in", W_in.to(device))

        # Bias weights
        W_bias = torch.empty(reservoir_size, 1)
        W_bias.uniform_(-bias_scaling, bias_scaling)
        self.register_buffer("W_bias", W_bias.to(device))

        # Recurrent matrix (scaled to target spectral radius)
        W = self._initialize_reservoir(reservoir_size, spectral_radius, sparsity)
        self.register_buffer("W", W.to(device))

        LOGGER.debug(
            "Reservoir init",
            extra={"extra": {
                "reservoir_size": reservoir_size,
                "input_dim": input_dim,
                "sparsity": float(sparsity),
                "spectral_radius": float(spectral_radius),
            }},
        )

    @property
    def bias_vec(self) -> torch.Tensor:
        return self.W_bias.squeeze(1)

    def _initialize_reservoir(self, size: int, spectral_radius: float, sparsity: float):
        W = torch.randn(size, size)
        mask = (torch.rand(size, size) < sparsity)
        W = W * mask

        before = compute_spectral_radius(W)
        if before == 0:
            before = 1.0
        W = W * (spectral_radius / before)

        after = compute_spectral_radius(W)
        LOGGER.info(
            "Reservoir radius scaled",
            extra={"extra": {"size": size, "sparsity": float(sparsity), "before": float(before), "after": float(after)}},
        )
        return W

    def update_batch(self, x: torch.Tensor, u: torch.Tensor, leak_rate: float) -> torch.Tensor:
        moved = False
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True); moved = True
        if u.device != self.device:
            u = u.to(self.device, non_blocking=True); moved = True
        if moved and self.profile and not self._warned_dev:
            print(f"[Reservoir] Moved tensors to {self.device} inside update_batch(). "
                  "Pass tensors already on the device to avoid overhead.")
            self._warned_dev = True

        if x.dim() == 1: x = x.unsqueeze(0)
        if u.dim() == 1: u = u.unsqueeze(0)
        if self.use_amp and x.is_cuda:
            with autocast("cuda", dtype=torch.float16):
                pre = torch.addmm(u @ self.W_in.T + self.bias_vec, x, self.W.mT)
                x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)
                x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)
        else:
            pre = torch.addmm(u @ self.W_in.T + self.bias_vec, x, self.W.mT)
            x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)

        return x_new
