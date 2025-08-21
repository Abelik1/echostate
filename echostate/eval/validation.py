from typing import List, Tuple
import torch

def rolling_origin_splits(T: int, n_splits: int, min_train: int, horizon: int) -> List[Tuple[slice, slice]]:
    """
    Produce rolling-origin (walk-forward) splits over indices [0..T).
    Returns list of (train_slice, test_slice).
    """
    splits = []
    step = max((T - min_train - horizon) // max(n_splits, 1), 1)
    start = min_train
    for k in range(n_splits):
        train_end = start + k * step
        test_start = train_end
        test_end = min(test_start + horizon, T)
        if test_end <= test_start: break
        splits.append((slice(0, train_end), slice(test_start, test_end)))
    return splits
