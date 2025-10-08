import torch
import numpy as np
from echostate.trainer import Trainer

def test_trainer_fit_inverse():
    # X: random, Y = A X + noise
    np.random.seed(0)
    T, dim = 100, 5
    X = torch.randn(T, dim)
    true_W = torch.randn(1, dim)
    Y = X @ true_W.T + 0.01 * torch.randn(T, 1)
    trainer = Trainer(ridge_param=1e-6, learning_algo="inv")
    W_out = trainer.fit(X, Y)
    # Check shape
    assert W_out.shape == (1, dim)
    # Predictions close to true
    Y_pred = X @ W_out.T
    mse = torch.mean((Y_pred - Y) ** 2).item()
    assert mse < 0.1