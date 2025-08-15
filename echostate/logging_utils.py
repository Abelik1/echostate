# echostate/logging_utils.py
import logging
import hashlib
from typing import Dict, Any, Sequence, Optional
import torch

LOGGER = logging.getLogger(__name__)

def _to_cpu_float(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(t):
        t = t.float()
    if t.device.type != "cpu":
        t = t.detach().cpu()
    else:
        t = t.detach()
    return t

def tensor_digest(t: torch.Tensor) -> str:
    """Stable short hash so we can reference the same large tensor without dumping it repeatedly."""
    t_cpu = _to_cpu_float(t).contiguous()
    h = hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()
    return h[:16]

def tensor_stats(
    t: torch.Tensor,
    quantiles: Sequence[float] = (0.0, 0.01, 0.5, 0.99, 1.0),
    sat_thresh: float = 0.99,
) -> Dict[str, Any]:
    """
    Summarize potentially-huge tensors without dumping full content.
    """
    info: Dict[str, Any] = {}
    try:
        info["shape"] = tuple(t.shape)
        info["dtype"] = str(t.dtype).replace("torch.", "")
        info["device"] = t.device.type
        info["digest"] = tensor_digest(t)
        tc = _to_cpu_float(t).view(-1)
        if tc.numel() > 0:
            finite = torch.isfinite(tc)
            if finite.any():
                tc = tc[finite]
                info["finite_frac"] = float(finite.float().mean().item())
                info["mean"] = float(tc.mean().item())
                info["std"] = float(tc.std(unbiased=False).item())
                info["min"] = float(tc.min().item())
                info["max"] = float(tc.max().item())
                qs = torch.tensor(quantiles)
                qv = torch.quantile(tc, qs)
                info["quantiles"] = {str(float(q)): float(v) for q, v in zip(qs, qv)}
                # zeros and saturation
                info["zero_frac"] = float((tc == 0).float().mean().item())
                info["sat_frac_|x|>%.2f" % sat_thresh] = float((tc.abs() > sat_thresh).float().mean().item())
            else:
                info["finite_frac"] = 0.0
        # small head/tail sample
        k = min(6, tc.numel())
        if k > 0:
            info["sample"] = [float(v) for v in tc[:k].tolist()]
    except Exception as e:
        info["error"] = f"failed_stats:{e}"
    return info

def log_tensor(
    logger: logging.Logger,
    t: torch.Tensor,
    name: str,
    level: int = logging.DEBUG,
    include_values: bool = False,
) -> None:
    stats = tensor_stats(t)
    if not include_values and "sample" in stats:
        stats = {k: v for k, v in stats.items() if k != "sample"}
    logger.log(level, f"{name} => {stats}", extra={"extra": {"tensor": name, **stats}})

def log_hparams(logger: logging.Logger, **kwargs):
    logger.info("Hyperparameters", extra={"extra": {"hparams": kwargs}})

def safe_len(x) -> Optional[int]:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None
