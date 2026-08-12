import torch

def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(predictions.device)
    return torch.mean(torch.abs(predictions - targets))

def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets = targets.to(predictions.device)
    return torch.mean((predictions - targets) ** 2)
