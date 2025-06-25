import torch
from echostate.reservoir import Reservoir

def test_spectral_radius_scaling():
    size = 50
    spectral_radius = 0.5
    res = Reservoir(input_dim=1,
                    reservoir_size=size,
                    output_dim=1,
                    spectral_radius=spectral_radius,
                    sparsity=0.2,
                    input_scaling=1.0,
                    bias_scaling=0.1,
                    seed=0)
    eigs = torch.linalg.eigvals(res.W).abs()
    max_eig = eigs.max().item()
    # Allow small numeric error
    assert pytest.approx(max_eig, rel=1e-2) == spectral_radius