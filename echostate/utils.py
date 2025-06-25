import torch

def compute_spectral_radius(W: torch.Tensor) -> float:
    """Compute the spectral radius (max abs eigenvalue) of weight matrix W."""
    eigs = torch.linalg.eigvals(W).abs()
    return eigs.max().item()


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between predictions and targets."""
    return torch.mean(torch.abs(predictions - targets))


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predictions and targets."""
    return torch.mean((predictions - targets) ** 2)