import torch

def compute_spectral_radius(W: torch.Tensor) -> float:
    eigs = torch.linalg.eigvals(W).abs()
    return eigs.max().item()

def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(predictions.device)
    return torch.mean(torch.abs(predictions - targets))

def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(predictions.device)
    return torch.mean((predictions - targets) ** 2)
