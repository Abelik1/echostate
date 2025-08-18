# echostate/reservoir.py
import logging
import torch
from .utils import compute_spectral_radius
import time

LOGGER = logging.getLogger(__name__)

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
        profile: bool = False,    
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device
        self.reservoir_size = reservoir_size
        
        self.profile = profile                     
        self._warned_dev = False 
        # input weights
        self.W_in = torch.empty(reservoir_size, input_dim, device=self.device)
        self.W_in.uniform_(-input_scaling, input_scaling)

        # bias weights
        self.W_bias = torch.empty(reservoir_size, 1, device=self.device)
        self.W_bias.uniform_(-bias_scaling, bias_scaling)
        self._W_bias = self.W_bias.squeeze()  # shape (reservoir_size,)
        self.use_amp = True
        # reservoir recurrent weights on CPU for eigen-decomp
        W = self._initialize_reservoir(reservoir_size, spectral_radius, sparsity)
        self.W = W.to(self.device)
        self.W_csr = W.to_sparse_csr().to(self.device) 
        # print(device)
        LOGGER.debug(
            "Reservoir init",
            extra={"extra": {
                "reservoir_size": reservoir_size,
                "input_dim": input_dim,
                "sparsity": sparsity,
                "spectral_radius_target": spectral_radius,
                "W_in_shape": tuple(self.W_in.shape),
                "W_bias_shape": tuple(self.W_bias.shape),
            }},
        )

    def _initialize_reservoir(self, size, spectral_radius, sparsity):
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

        if x.dim() == 1:  # make batch dim
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)

        
        if self.use_amp and x.is_cuda:
            # AMP for reservoir math only
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pre = torch.addmm(u @ self.W_in.T + self._W_bias, x, self.W.mT)  # (B,R)
                x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)
        else:
            pre = torch.addmm(u @ self.W_in.T + self._W_bias, x, self.W.mT)
            x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)
            
        # x_new = (1 - leak_rate) * x + leak_rate * torch.tanh(pre)

        # Ultra-verbose per-step logging (guarded by TRACE)
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
