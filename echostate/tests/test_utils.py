import torch
import pytest
from echostate.utils import compute_spectral_radius, mean_absolute_error, mean_squared_error

def test_compute_spectral_radius_identity():
    W = torch.eye(4)
    radius = compute_spectral_radius(W)
    assert radius == pytest.approx(1.0)

def test_error_metrics():
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
    mae = mean_absolute_error(preds, targets)
    mse = mean_squared_error(preds, targets)
    assert mae == pytest.approx((0 + 1 + 1 + 1) / 4)
    assert mse == pytest.approx((0 + 1 + 1 + 1) / 4)