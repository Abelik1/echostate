import logging
import os
import time
import torch

from .reservoir import Reservoir
from .trainer import Trainer
from ..utils import mean_absolute_error, mean_squared_error
from ..esn_logging.utils import log_tensor, tensor_stats

LOGGER = logging.getLogger(__name__)

def _env_profile_on() -> bool:
    return os.getenv("ESN_PROFILE", "0").lower() in ("1", "true", "yes", "y")

def _fmt_tensor(name, t):
    if t is None: return f"{name}: None"
    if not isinstance(t, torch.Tensor): return f"{name}: (not a tensor) {type(t)}"
    pinned = getattr(t, "is_pinned", lambda: False)()
    return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
            f"pinned={pinned}, requires_grad={t.requires_grad}")

class ESN(torch.nn.Module):
    """
    Echo State Network with streamed covariance training and implicit closed-loop inference.
    - Training: teacher-forced state collection after washout, ridge-regularized readout.
    - Inference: **implicitly closed-loop after washout** (no flags).
    """
    def __init__(
        self,
        *,
        base_input_dim: int,
        reservoir_size: int,
        output_dim: int,
        feedback: int = 0,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.2,
        ridge_param: float = 1e-6,
        learning_algo: str = "inv",
        leak_rate: float = 1.0,
        washout: int = 50,
        chunk_K: int = 128,
        seed: int | None = None,
        device: torch.device | None = None,
        use_amp: bool = True,
        profile: bool = False,
    ):
        super().__init__()
        self.device = device
        if seed is not None:
            torch.manual_seed(seed)

        # Hyperparameters
        self.base_input_dim = base_input_dim
        self.output_dim = output_dim
        self.feedback = feedback
        self.input_dim = base_input_dim + feedback * output_dim
        self.reservoir_size = reservoir_size
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.leak_rate = leak_rate
        self.washout = washout
        self.learning_algo = learning_algo
        self.ridge_param = ridge_param
        self.seed = seed
        self.profile = profile or _env_profile_on()
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.chunk_K = chunk_K
        self.use_amp = use_amp

        # Components
        self.reservoir = Reservoir(
            input_dim=self.input_dim,
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            seed=seed,
            device=self.device,
            profile=self.profile,
            use_amp=use_amp,
        )
        self.trainer = Trainer(
            ridge_param=ridge_param,
            learning_algo=learning_algo,
            device=self.device,
            profile=self.profile,
        )

        # Readout weights buffer
        self.register_buffer("W_out", torch.empty(0, device=self.device))

        # Per-batch bias helper (1 column of ones * bias_scaling)
        self._batch_bias = None

        LOGGER.info("ESN init",
            extra={"extra": {
                "device": str(self.device),
                "base_input_dim": base_input_dim,
                "output_dim": output_dim,
                "input_dim": self.input_dim,
                "reservoir_size": reservoir_size,
                "feedback": feedback,
                "leak_rate": leak_rate,
                "washout": washout,
                "learning_algo": learning_algo,
            }},
        )

    # --------- Config & helpers ---------
    def get_config(self) -> dict:
        return dict(
            base_input_dim=self.base_input_dim,
            reservoir_size=self.reservoir_size,
            output_dim=self.output_dim,
            feedback=self.feedback,
            spectral_radius=self.spectral_radius,
            sparsity=self.sparsity,
            input_scaling=self.input_scaling,
            bias_scaling=self.bias_scaling,
            ridge_param=self.ridge_param,
            learning_algo=self.learning_algo,
            leak_rate=self.leak_rate,
            washout=self.washout,
            chunk_K=self.chunk_K,
            seed=self.seed,
            use_amp=self.use_amp,
        )

    def _ensure_batch_bias(self, batch_sz: int) -> torch.Tensor:
        if self._batch_bias is None or self._batch_bias.shape[0] != batch_sz:
            self._batch_bias = torch.ones(batch_sz, 1, device=self.device) * self.bias_scaling
        return self._batch_bias

    # --------- Training (teacher-forced state collection) ---------
    def fit(self, inputs: torch.Tensor | list[torch.Tensor], targets: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        prof_on = self.profile
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
            targets = torch.stack(targets, dim=0)

        B, T, Din = inputs.shape
        assert Din == self.base_input_dim, "inputs last dim must match base_input_dim"
        assert T > self.washout, "Need at least washout+1 steps for training."

        X = inputs.to(self.device, non_blocking=True)
        Y = targets.to(self.device, non_blocking=True)

        LOGGER.info("ESN.fit start", extra={"extra": {"B": B, "T": T, "Din": Din, "washout": self.washout}})
        log_tensor(LOGGER, X, "inputs(batch)", level=logging.DEBUG)
        log_tensor(LOGGER, Y, "targets(batch)", level=logging.DEBUG)

        # State & feedback
        x = torch.zeros(B, self.reservoir_size, device=self.device, dtype=X.dtype)
        prev_fb = torch.zeros(B, self.feedback * self.output_dim, device=self.device, dtype=X.dtype) if self.feedback > 0 else None

        # Accumulators for streamed normal equations
        n_feat = self.reservoir_size + 1
        xTx = torch.zeros(n_feat, n_feat, device=self.device, dtype=X.dtype)
        xTy = torch.zeros(n_feat, self.output_dim, device=self.device, dtype=X.dtype)

        # Light diagnostics
        diag_count = 0
        run_mean = torch.zeros(self.reservoir_size, device=self.device, dtype=X.dtype)
        run_m2   = torch.zeros(self.reservoir_size, device=self.device, dtype=X.dtype)

        # Optional CUDA timing
        if prof_on and self.device.type == "cuda":
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        K = int(self.chunk_K)
        Xbuf, Ybuf = [], []

        for t in range(T):
            u_base = X[:, t, :]
            u = torch.cat([u_base, prev_fb], dim=1) if self.feedback > 0 else u_base

            x = self.reservoir.update_batch(x, u, self.leak_rate)

            if t >= self.washout:
                xb = torch.cat([x, self._ensure_batch_bias(x.size(0))], dim=1)  # (B, R+1)
                yt = Y[:, t, :]                                                # (B, Dout)

                Xbuf.append(xb)
                Ybuf.append(yt)
                if len(Xbuf) == K or t == T - 1:
                    Xstk = torch.cat(Xbuf, dim=0)  # (B*K, R+1)
                    Ystk = torch.cat(Ybuf, dim=0)  # (B*K, Dout)
                    xTx += Xstk.mT @ Xstk
                    xTy += Xstk.mT @ Ystk
                    Xbuf.clear(); Ybuf.clear()

                # running stats
                bsz = x.shape[0]
                diag_count += bsz
                delta = x.mean(dim=0) - run_mean
                run_mean += (bsz / max(diag_count, 1)) * delta
                run_m2 += ((x - run_mean).pow(2)).sum(dim=0)

            if self.feedback > 0:
                prev_fb = torch.cat([prev_fb[:, self.output_dim:], Y[:, t, :]], dim=1)

            LOGGER.debug("fit.step", extra={"extra": {"t": t, "post_washout": t >= self.washout}})

        # CUDA timing end
        if prof_on and self.device.type == "cuda":
            ev1.record(); torch.cuda.synchronize()
            ms = ev0.elapsed_time(ev1)
            print(f"[ESN.fit] reservoir loop CUDA time: {ms/1000:.6f}s")

        # Diagnostics
        if diag_count > 0:
            var = (run_m2 / max(diag_count, 1)).clamp_min(0)
            n_dead = int((var < 1e-10).sum().item())
        else:
            n_dead = 0
        n_rows = (B * max(T - self.washout, 0))
        LOGGER.debug("Design matrix diagnostics (streamed)", extra={"extra": {
            "rows": n_rows, "features": n_feat, "dead_features": n_dead
        }})

        # Solve for readout
        self.trainer._profile = prof_on
        W = self.trainer.fit_from_cov(xTx, xTy, n_feat)
        with torch.no_grad():
            self.W_out = W.detach()
        return self.W_out

    # --------- Inference (implicitly closed-loop after washout) ---------
    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run inference (closed-loop after washout). Returns predictions from `washout:` onward.
        If base_input_dim == output_dim, the model will feed its own last prediction as input
        after washout. If feedback > 0, a shifting output feedback vector is also used.
        """
        assert isinstance(self.W_out, torch.Tensor) and self.W_out.numel() > 0, "Call fit() before infer()."
        T, Din = inputs.shape
        assert Din == self.base_input_dim, "inputs last dim must match base_input_dim"

        X = inputs.to(self.device, non_blocking=True)
        x = torch.zeros(1, self.reservoir_size, device=self.device, dtype=X.dtype)
        prev_fb = torch.zeros(self.feedback * self.output_dim, device=self.device, dtype=X.dtype) if self.feedback > 0 else None

        preds = []
        last_y = None

        for t in range(T):
            use_closed = (t >= self.washout) and (last_y is not None) and (self.base_input_dim == self.output_dim)
            u_base = last_y.squeeze(0) if use_closed else X[t, :].unsqueeze(0)

            if self.feedback > 0:
                u = torch.cat([u_base, prev_fb.unsqueeze(0)], dim=1)
            else:
                u = u_base

            x = self.reservoir.update_batch(x, u, self.leak_rate)
            xb = torch.cat([x, self._ensure_batch_bias(x.size(0))], dim=1)
            y = xb @ self.W_out.T  # (1, Dout)
            preds.append(y.squeeze(0))
            last_y = y

            if self.feedback > 0:
                prev_fb = torch.cat([prev_fb[self.output_dim:], y.squeeze(0)], dim=0)

            LOGGER.debug("infer.step", extra={"extra": {"t": t}})

        out = torch.stack(preds, dim=0)
        return out[self.washout:]

    # PyTorch semantics sugar
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.infer(inputs)

    # --------- Batch predict with optional metrics ---------
    def predict(self, input_list: list[torch.Tensor], target_list: list[torch.Tensor] | None = None):
        preds = []
        with torch.no_grad():
            for seq in input_list:
                preds.append(self.infer(seq))

        if target_list is not None:
            P = torch.cat(preds, dim=0)
            y_true = torch.cat([t[self.washout:] for t in target_list], dim=0).to(P.device)
            mae = mean_absolute_error(P, y_true).item()
            mse = mean_squared_error(P, y_true).item()
            LOGGER.info("predict.metrics", extra={"extra": {"mae": float(mae), "mse": float(mse)}})
            return preds, {"mae": mae, "mse": mse}
        return preds
