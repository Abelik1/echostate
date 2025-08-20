# echostate/reservoir.py
import logging
import torch
import torch.nn as nn
from .utils import compute_spectral_radius

LOGGER = logging.getLogger(__name__)

class Reservoir(nn.Module):
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
        profile: bool = False,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        # store config (useful for reconstruction/logging)
        self.device = device
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.profile = profile
        self._warned_dev = False
        self.use_amp = True

        # --- Input weights (buffer → saved in state_dict, moved by .to(device)) ---
        W_in = torch.empty(reservoir_size, input_dim, device=self.device)
        W_in.uniform_(-input_scaling, input_scaling)
        self.register_buffer("W_in", W_in)

        # --- Bias weights (buffer) ---
        W_bias = torch.empty(reservoir_size, 1, device=self.device)
        W_bias.uniform_(-bias_scaling, bias_scaling)
        self.register_buffer("W_bias", W_bias)

        # --- Recurrent/reservoir matrix (buffer), scaled to target spectral radius ---
        W = self._initialize_reservoir(reservoir_size, spectral_radius, sparsity)
        self.register_buffer("W", W.to(self.device))

        LOGGER.debug(
            "Reservoir init",
            extra={"extra": {
                "reservoir_size": reservoir_size,
                "input_dim": input_dim,
                "sparsity": float(sparsity),
                "spectral_radius_target": float(spectral_radius),
                "W_in_shape": tuple(self.W_in.shape),
                "W_bias_shape": tuple(self.W_bias.shape),
                "W_shape": tuple(self.W.shape),
            }},
        )

    # Safer than caching a squeezed view; always reflects current device/buffer
    @property
    def bias_vec(self) -> torch.Tensor:
        # shape: (reservoir_size,)
        return self.W_bias.squeeze(1)

    def _initialize_reservoir(self, size, spectral_radius, sparsity):
        # build on CPU for eig/spectral radius, then moved to device as buffer
        W = torch.randn(size, size)
        mask = (torch.rand(size, size) < sparsity)
        W = W * mask

        radius_before = compute_spectral_radius(W)
        if radius_before == 0:
            radius_before = 1.0
        W = W * (spectral_radius / radius_before)

        radius_after = compute_spectral_radius(W)
        LOGGER.info(
            "Reservoir spectral radius scaled",
            extra={"extra": {
                "size": size,
                "sparsity": float(sparsity),
                "radius_before": float(radius_before),
                "radius_after": float(radius_after),
                "target": float(spectral_radius),
            }},
        )
        return W

    def update_batch(self, x: torch.Tensor, u: torch.Tensor, leak_rate: float) -> torch.Tensor:
        # Device auto-fix + one-time warning if we had to move tensors
        moved = False
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True); moved = True
        if u.device != self.device:
            u = u.to(self.device, non_blocking=True); moved = True
        if moved and self.profile and not self._warned_dev:
            print(f"[PROFILE][Reservoir] Moved tensors to {self.device} inside update_batch(). "
                  f"This adds CPU overhead. Ensure caller passes tensors already on {self.device}.")
            self._warned_dev = True

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)

        # Reservoir pre-activation and leaky tanh update (AMP on CUDA)
        if self.use_amp and x.is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pre = torch.addmm(u @ self.W_in.T + self.bias_vec, x, self.W.mT)  # (B,R)
                x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)
        else:
            pre = torch.addmm(u @ self.W_in.T + self.bias_vec, x, self.W.mT)
            x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)

        if LOGGER.isEnabledFor(5):  # TRACE
            try:
                from .logging_utils import tensor_stats
                LOGGER.trace("Reservoir step",
                    extra={"extra": {
                        "u_stats": tensor_stats(u),
                        "x_stats": tensor_stats(x),
                        "pre_stats": tensor_stats(pre),
                        "x_new_stats": tensor_stats(x_new),
                        "leak_rate": float(leak_rate),
                    }},
                )
            except Exception:
                pass

        return x_new  # (B, reservoir_size)
