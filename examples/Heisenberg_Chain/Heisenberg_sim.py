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
        # print("Vec: ", vec[:self.N])
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
    
    
if __name__ == '__main__': 
    from scipy.interpolate import interp1d
    import pickle
    N = 5
    T = 5000
    qubit = 0
    # dt_list = np.arange(0.01, 1.8, 0.01) 
    dt_list = [0.05,1.5]
    seed = 3141
    target_points = 10_000

    all_z = []
    all_times = []
    errors = []

    histories_path_time = f'./examples/Heisenberg_Chain/cache/Historydata({seed})_alltimes.pkl'  
    histories_path_z = f'./examples/Heisenberg_Chain/cache/Historydata({seed})_allz.pkl'  

    
    # Load or generate all trajectories
    try:
        with open(histories_path_time, 'rb') as f:
            all_times = pickle.load(f)
        with open(histories_path_z, 'rb') as f:
            all_z = pickle.load(f)
    except:
        for dt in dt_list:
            steps = int(T / dt)
            print(f"Processing dt: {dt}")
            np.random.seed(seed)
            chain = HeisenbergChain(N, dt=dt)
            name = f"Qbts{N}_dt{round(dt, 5)}".replace(".", "_", 1)
            histories_path = f'./examples/Heisenberg_Chain/cache/Historydata({seed})_{name}.pkl'
            chain.evolve(steps=steps)

            z_vals = np.array([
                float(expect(sigmaz(), Qobj(rho, dims=chain.dims).ptrace(qubit)))
                for rho in chain.history
            ])
            times = np.arange(len(z_vals)) * dt
            all_z.append(z_vals)
            all_times.append(times)
    
        with open(histories_path_time, 'wb') as f:
            pickle.dump(all_times, f)
        with open(histories_path_z, 'wb') as f:
            pickle.dump(all_z, f)
        
    # Use smallest dt as reference
    ref_z = all_z[0]
    ref_t = all_times[0]

    for i, (t_arr, z_arr) in enumerate(zip(all_times, all_z)):
        if i == 0:
            errors.append(0.0)  # zero error for reference
            continue

        # Interpolate current trajectory to reference times
        f = interp1d(t_arr, z_arr, bounds_error=False, fill_value="extrapolate")
        z_interp = f(ref_t)

        mae = np.mean(np.abs(z_interp - ref_z))
        errors.append(mae)

    # Plot error vs dt
    plt.figure()
    plt.plot(dt_list, errors, marker='o')
    plt.xlabel("dt")
    plt.ylabel("Mean Absolute Error vs smallest dt")
    plt.title("Fidelity loss with increasing dt")
    plt.grid(True)
    plt.tight_layout()
    
    num = 1
    plt.figure()
    plt.plot(all_times[0], all_z[0], label = f"Best {all_times[0][1]}")
    plt.plot(all_times[num], all_z[num], label = f"Less {all_times[num][1]}")
    plt.legend()
    plt.show()

    