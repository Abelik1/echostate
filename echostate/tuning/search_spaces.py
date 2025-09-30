from dataclasses import dataclass

@dataclass
class DefaultSpace:
    reservoir_size: tuple[int, int] = (64, 1024)          # int range
    spectral_radius: tuple[float, float] = (0.1, 1.2)     # float range
    sparsity: tuple[float, float] = (0.05, 0.3)
    leak_rate: tuple[float, float] = (0.2, 1.0)
    input_scaling: tuple[float, float] = (0.1, 2.0)
    bias_scaling: tuple[float, float] = (0.0, 1.0)
    ridge_param: tuple[float, float] = (1e-8, 1e-2)       # sampled log-uniform
    feedback: tuple[int, int] = (0, 0)                    # keep 0 by default
