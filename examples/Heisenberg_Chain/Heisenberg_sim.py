from qutip import Qobj, basis, tensor, sigmax, sigmay, sigmaz, qeye, expect
import numpy as np
import matplotlib.pyplot as plt

class HeisenbergChain:
    """
    Simulate density-matrix evolution of an N-qubit Heisenberg chain,
    storing history as NumPy arrays for efficiency.
    """
    def __init__(self, num_qubits, J = 1.0, dt = 0.01):
        self.N = num_qubits
        self.dt = dt
        self.J = J
        # dims for reconstructing Qobj from array
        self.dims = [[2]*self.N, [2]*self.N]

        # build Hamiltonian once (Qobj)
        self.H = self._build_hamiltonian()
        
        # initial random pure state
        vec = (np.random.randn(2**self.N) + 1j*np.random.randn(2**self.N))
        vec /= np.linalg.norm(vec)
        print("Vec: ", vec[:3])
        psi = Qobj(vec, dims=[[2]*self.N, [1]*self.N])
        rho0 = psi * psi.dag()

        # store history as arrays
        self.history = [rho0.full()]

    def _build_hamiltonian(self) -> Qobj:
        H = 0
        for j in range(self.N):
            jp1 = (j + 1) % self.N
            for op in (sigmax, sigmay, sigmaz):
                ops = [qeye(2)] * self.N
                ops[j]   = op()
                ops[jp1] = op()
                H += -0.5 * self.J * tensor(*ops)
        return H

    def evolve(self, steps: int):
        """Perform evolution and append full-state arrays."""
        # convert last array back to Qobj for evolution
        p = Qobj(self.history[-1], dims=self.dims)
        U = (-1j * self.H * self.dt).expm()
        for _ in range(steps):
            p = U * p * U.dag()
            self.history.append(p.full())
        # keep last Qobj for future ops
        self._last_rho = p

    def get_rho_qobj(self, t: int) -> Qobj:
        """Reconstruct Qobj density matrix at time t from array."""
        return Qobj(self.history[t], dims=self.dims)

    def check_conservation(self) -> dict:
        """Compute conserved quantities across history."""
        traces = []
        purities = []
        energies = []
        Sz_tot = sum(
            tensor(*[sigmaz() if k==j else qeye(2) for k in range(self.N)])
            for j in range(self.N)
        )
        mags = []
        for arr in self.history:
            rho = Qobj(arr, dims=self.dims)
            traces.append(float(rho.tr()))
            purities.append(float((rho*rho).tr()))
            energies.append(float((self.H*rho).tr().real))
            mags.append(float(expect(Sz_tot, rho)))
        return {
            'trace': (min(traces), max(traces)),
            'purity': (min(purities), max(purities)),
            'energy': (min(energies), max(energies)),
            'magnetization': (min(mags), max(mags)),
        }

    def plot_spin_grid(self, size):
        """Plot ⟨σ_z⟩ for each qubit vs time as grid using array history."""
        data = []
        for t, arr in enumerate(self.history):
            rho = Qobj(arr, dims=self.dims)
            for q in range(self.N):
                rz = rho.ptrace(q)
                data.append((q, t, expect(sigmaz(), rz)))
        arr_data = np.array(data)[:size]
        # print(arr_data) #TODO REMOVE
        x, y, c = arr_data[:,0], arr_data[:,1], arr_data[:,2]
        plt.figure(figsize=(self.N, len(arr_data)*0.15+2))
        sc = plt.scatter(x, y, c=c, cmap='bwr', vmin=-1, vmax=1,
                         s=200, marker='s')
        plt.colorbar(sc, label='⟨σ_z⟩')
        plt.xlabel('Qubit')
        plt.ylabel('Timestep')
        plt.title('Spin Evolution')
        plt.grid(which='both', ls='--', lw=0.5)
        plt.tight_layout()
        