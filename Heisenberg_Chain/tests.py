# tests/test_heisenberg_chain.py
import numpy as np
from Heisenberg_Chain.Heisenberg_sim import HeisenbergChain

def test_short_run_norm_and_lengths():
    N, k, dt, steps = 5, 0, 0.1, 10
    hc = HeisenbergChain(N, k, dt=dt, measure="sz")
    hc.evolve(steps)
    assert abs(np.vdot(hc.psi, hc.psi).real - 1.0) < 1e-6
    assert len(hc.energy_history) == steps + 1
    assert len(hc.obs_history) == steps + 1
