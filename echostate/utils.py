import torch

def compute_spectral_radius(W: torch.Tensor) -> float:
    eigs = torch.linalg.eigvals(W).abs()
    return eigs.max().item()
