from typing import Iterable, Tuple, List
import torch

def sliding_windows(
    series: torch.Tensor,
    *,
    input_len: int,
    pred_len: int,
    step: int = 1,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Turn a single time series (T, D) into (X, Y) windows:
      X: (input_len, D), Y: (pred_len, D), sliding with stride 'step'.
    """
    T, D = series.shape
    pairs = []
    i = 0
    while i + input_len + pred_len <= T:
        x = series[i : i + input_len]
        y = series[i + input_len : i + input_len + pred_len]
        pairs.append((x, y))
        i += step
    return pairs

def batchify(pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    X = torch.stack([p[0] for p in pairs], dim=0)
    Y = torch.stack([p[1] for p in pairs], dim=0)
    return X, Y
