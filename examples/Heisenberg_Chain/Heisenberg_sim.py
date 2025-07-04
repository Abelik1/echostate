from qutip import Qobj, basis, tensor, sigmax, sigmay, sigmaz, qeye, expect
import numpy as np
import matplotlib.pyplot as plt

class HeisenbergChain:
    """
    Simulate pure-state evolution of an N-qubit Heisenberg chain,
    storing only data for a selected qubit to save memory.
    """
    def __init__(self, num_qubits, target_qubit, J=1.0, dt=0.01, dtype=np.complex64):
        self.N = num_qubits
        self.k = target_qubit
        self.dt = dt
        self.J = J
        self.dtype = dtype

        # initial random pure state vector
        vec = (np.random.randn(2**self.N) + 1j*np.random.randn(2**self.N)).astype(self.dtype)
        vec /= np.linalg.norm(vec)
        # print("vec: ", vec)
        self.psi = vec
        
        # build Hamiltonian once
        H_qobj = self._build_hamiltonian()
        self.H = H_qobj
        # Precompute U = exp(-i H dt)
        self.U = (-1j * self.H * self.dt).expm()
        
        # History of expectation values or reduced density matrices
        self.sz_history = []

    def _build_hamiltonian(self) -> Qobj:
        H = 0
        for j in range(self.N):
            jp1 = (j + 1) % self.N
            for op in (sigmax, sigmay, sigmaz):
                ops = [qeye(2)] * self.N
                ops[j] = op()
                ops[jp1] = op()
                H += -0.5 * self.J * tensor(*ops)
        return H

    def evolve(self, steps: int, store_reduced=False):
        sz_op = tensor(*[sigmaz() if idx == self.k else qeye(2) for idx in range(self.N)])

        # Store initial state
        psi_qobj = Qobj(self.psi, dims=[[2]*self.N, [1]*self.N])
        if store_reduced:
            rho_k = psi_qobj.ptrace(self.k)
            self.sz_history.append(rho_k.full())
        else:
            val = expect(sz_op, psi_qobj)
            self.sz_history.append(val)

        # Then evolve
        for _ in range(steps):
            psi_qobj = self.U * psi_qobj
            self.psi = psi_qobj.full().flatten()
            self.psi /= np.linalg.norm(self.psi)

            if store_reduced:
                rho_k = psi_qobj.ptrace(self.k)
                self.sz_history.append(rho_k.full())
            else:
                val = expect(sz_op, psi_qobj)
                self.sz_history.append(val)

    def get_sz(self, t=None):
        """Get stored ⟨σ_z⟩ or reduced density matrix at time t (or full history if t is None)."""
        if t is None:
            return np.array(self.sz_history)
        return self.sz_history[t]

    def plot(self):
        """Plot ⟨σ_z⟩ vs time for the target qubit."""
        times = np.arange(len(self.sz_history)) * self.dt
        plt.figure()
        plt.plot(times, self.sz_history)
        plt.xlabel('Time')
        plt.ylabel(f"⟨σ_z⟩ (qubit {self.k})")
        plt.title('Spin Expectation Evolution')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    from scipy.interpolate import interp1d
    import pickle

    # Simulation parameters
    N = 10
    T = 100
    qubit = 0
    dt_list = [0.01, 0.15]
    seed = 31415

    all_z = []
    all_times = []
    errors = []

    # Paths for caching
    base = './examples/Heisenberg_Chain/cache'
    histories_path_time = f'{base}/Historydata({seed})_N{N}_alltimes.pkl'
    histories_path_z    = f'{base}/Historydata({seed})_N{N}_allz.pkl'

    # Load or generate trajectories
    try:
        with open(histories_path_time, 'rb') as f:
            all_times = pickle.load(f)
        with open(histories_path_z, 'rb') as f:
            all_z    = pickle.load(f)
    except FileNotFoundError:
        for dt in dt_list:
            steps = int(T / dt)
            print(f"Processing dt={dt}, steps={steps}")
            np.random.seed(seed)
            chain = HeisenbergChain(N, qubit, J=1.0, dt=dt)
            chain.evolve(steps)
            chain.plot()
            z_vals = chain.get_sz()
            times = np.arange(len(z_vals)) * dt
            all_z.append(z_vals)
            all_times.append(times)

        # Cache results
        import os
        os.makedirs(base, exist_ok=True)
        with open(histories_path_time, 'wb') as f:
            pickle.dump(all_times, f)
        with open(histories_path_z, 'wb') as f:
            pickle.dump(all_z, f)

    # Reference trajectory
    ref_z = all_z[0]
    ref_t = all_times[0]

    # Compute mean absolute errors
    for i, (t_arr, z_arr) in enumerate(zip(all_times, all_z)):
        if i == 0:
            errors.append(0.0)
            continue
        f_interp = interp1d(t_arr, z_arr, bounds_error=False, fill_value="extrapolate")
        errors.append(np.mean(np.abs(f_interp(ref_t) - ref_z)))

    # Plot error vs dt
    plt.figure()
    plt.plot(dt_list, errors, marker='o')
    plt.xlabel('dt')
    plt.ylabel('Mean Absolute Error vs smallest dt')
    plt.title('Fidelity loss with increasing dt')
    plt.grid(True)
    plt.tight_layout()

    # Compare two trajectories visually
    plt.figure()
    plt.plot(all_times[0], all_z[0], label=f"dt={dt_list[0]}")
    plt.plot(all_times[1], all_z[1], label=f"dt={dt_list[1]}")
    plt.xlabel('Time')
    plt.ylabel(f"⟨σ_z⟩ (qubit {qubit})")
    plt.legend()
    plt.show()
