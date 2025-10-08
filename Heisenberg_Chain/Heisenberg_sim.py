from qutip import Qobj, basis, tensor, sigmax, sigmay, sigmaz, qeye, expect
import numpy as np
import matplotlib.pyplot as plt

class HeisenbergChain:
    """
    Simulate pure-state evolution of an N-qubit Heisenberg chain.
    You can record either a Pauli expectation ('sx','sy','sz') for a target qubit
    or the reduced density matrix ('rho') for that qubit.
    """
    def __init__(self, periodic, num_qubits, target_qubit, J=1.0, dt=0.01,
                 dtype=np.complex64, measure='sz'):
        self.N = num_qubits
        self.k = target_qubit
        self.dt = dt
        self.J = J
        self.dtype = dtype
        self.measure = measure  # 'sx' | 'sy' | 'sz' | 'rho'

        vec = (np.random.randn(2**self.N) + 1j*np.random.randn(2**self.N)).astype(self.dtype)
        vec /= np.linalg.norm(vec)
        self.psi = vec
        H_qobj = self._build_hamiltonian(periodic=periodic)
        self.H = H_qobj
        self.U = (-1j * self.H * self.dt).expm()

        self.norm_history = []
        self.energy_history = []
        self.obs_history = []  # either list of floats (⟨σ⟩) or 2x2 matrices (ρ)

        # prebuild the measurement operator for speed if we're in expectation mode
        if self.measure in ('sx', 'sy', 'sz'):
            pauli = {'sx': sigmax, 'sy': sigmay, 'sz': sigmaz}[self.measure]
            self._pauli_op = tensor(*[pauli() if i == self.k else qeye(2) for i in range(self.N)])
        else:
            raise ValueError("Wrong measure")

    def set_initial_state(self, vec: np.ndarray):
        if vec.shape != (2**self.N,):
            raise ValueError(f"Input vector shape mismatch: expected {(2**self.N,)}, got {vec.shape}")
        self.psi = vec.astype(self.dtype) / np.linalg.norm(vec)

    def _build_hamiltonian(self, periodic: bool = False) -> Qobj:
        H = Qobj(np.zeros((2**self.N, 2**self.N)), dims=[[2]*self.N, [2]*self.N])
        N = self.N
        pairs = [(j, j+1) for j in range(N-1)]
        if periodic:
            pairs.append((N-1, 0))    # add the wrap only if requested

        for j, jp1 in pairs:
            for op in (sigmax, sigmay, sigmaz):
                ops = [qeye(2) for _ in range(N)]
                ops[j]  = op()
                ops[jp1] = op()
                H += -0.5 * self.J * tensor(*ops)
        return H

    def _record_observable(self, psi_qobj: Qobj):
        if self.measure == 'rho':
            rho_k = psi_qobj.ptrace(self.k).full()
            self.obs_history.append(rho_k)
        else:
            val = expect(self._pauli_op, psi_qobj)
            self.obs_history.append(float(np.real(val)))

    def evolve(self, steps: int):
        psi_qobj = Qobj(self.psi, dims=[[2]*self.N, [1]*self.N])

        self.norm_history.append(float(np.vdot(self.psi, self.psi).real))
        e0 = expect(self.H, psi_qobj)
        self.energy_history.append(float(np.real(e0)))
        self._record_observable(psi_qobj)

        for _ in range(steps):
            psi_qobj = self.U @ psi_qobj
            self.psi = psi_qobj.full().flatten()
            self.psi /= np.linalg.norm(self.psi)

            self.norm_history.append(float(np.vdot(self.psi, self.psi).real))
            e = expect(self.H, psi_qobj)
            self.energy_history.append(float(np.real(e)))
            self._record_observable(psi_qobj)

    def get_observable(self, t=None):
        return np.array(self.obs_history) if t is None else self.obs_history[t]
#region TESTING
def generate_perturbed_states(N, base_seed=31415, epsilon=1e-6):
        """
        Returns:
            - base_vec: original pure state (normalized)
            - first_qubit_perturbation: small change in amplitudes affecting qubit 0
            - all_qubits_perturbation: small random change across all amplitudes
        """
        np.random.seed(base_seed)
        base_vec = (np.random.randn(2**N) + 1j * np.random.randn(2**N)).astype(np.complex64)
        base_vec /= np.linalg.norm(base_vec)

        # Perturbation only in the first qubit's relevant states
        perturb = np.zeros_like(base_vec)
        for i in range(len(perturb)):
            if (i >> (N - 1)) & 1:  # Most significant bit controls qubit 0
                perturb[i] += epsilon
        perturbed_first = base_vec + perturb
        perturbed_first /= np.linalg.norm(perturbed_first)

        # Small perturbation across all qubits
        noise = epsilon * (np.random.randn(2**N) + 1j * np.random.randn(2**N)).astype(np.complex64)
        perturbed_all = base_vec + noise
        perturbed_all /= np.linalg.norm(perturbed_all)

        return base_vec, perturbed_first, perturbed_all
import numpy as np

def _index_to_spin_string(i: int, N: int, bit_up='0->↑', msb_is_q0=True):
    """
    Convert basis index i to a spin ket like |↑↓…⟩.
    Conventions:
      - msb_is_q0=True: leftmost char is qubit 0 (MSB), rightmost is qubit N-1 (LSB).
      - bit_up chooses mapping:
          '0->↑' (default): bit 0 => ↑, bit 1 => ↓  (matches σ_z eigenvalues +1 for |0>=|↑>)
          '1->↑':          bit 1 => ↑, bit 0 => ↓
    """
    if msb_is_q0:
        bits = f"{i:0{N}b}"  # MSB...LSB
    else:
        bits = f"{i:0{N}b}"[::-1]  # LSB...MSB shown left-to-right as q0..q{N-1}

    if bit_up == '0->↑':
        # 0 -> ↑, 1 -> ↓
        spin_chars = ''.join('↑' if b == '0' else '↓' for b in bits)
    else:
        # 1 -> ↑, 0 -> ↓
        spin_chars = ''.join('↑' if b == '1' else '↓' for b in bits)
    return f"|{spin_chars}⟩"

def pretty_print_state(
    vec: np.ndarray,
    N: int,
    top_k: int | None = 16,
    tol: float = 0.0,
    msb_is_q0: bool = True,
    bit_up: str = '0->↑',
    phase_in_deg: bool = False,
    show_index: bool = True
):
    """
    Pretty-print the state vector as a sorted list of amplitudes times spin basis kets.

    Args:
      vec: complex ndarray of shape (2**N,)
      N: number of qubits
      top_k: print only the largest |amp| entries (None => print all passing tol)
      tol: skip entries with |amp| < tol
      msb_is_q0: choose qubit ordering in the printed ket
      bit_up: '0->↑' (default) or '1->↑' to control ↑/↓ mapping
      phase_in_deg: print phase in degrees (otherwise radians)
      show_index: include the integer basis index i

    Example line:
      |amp|=0.2314  arg= -1.57 rad   amp=-0.0000-0.2314i   [i=3]  |↑↓↓…⟩
    """
    vec = np.asarray(vec).reshape(-1)
    assert vec.size == (1 << N), f"vec length {vec.size} != 2**N"

    mags = np.abs(vec)
    mask = mags >= tol
    idxs = np.arange(vec.size)[mask]
    mags = mags[mask]

    # sort by magnitude descending
    order = np.argsort(-mags)
    idxs = idxs[order]
    mags = mags[order]

    if top_k is not None:
        idxs = idxs[:top_k]
        mags = mags[:top_k]

    print(f"\nState in computational basis (N={N}):")
    print(f"norm^2 = {np.vdot(vec, vec).real:.8f} (should be ~1.0)")
    header = "  {:>12}  {:>12}  {:>28}  {}{}".format(
        "|amp|", "arg", "complex amplitude", "[i] " if show_index else "", "basis ket"
    )
    print(header)
    print("-" * len(header))

    for i, mag in zip(idxs, mags):
        amp = vec[i]
        arg = np.angle(amp)
        arg_str = f"{np.degrees(arg): .2f} deg" if phase_in_deg else f"{arg: .2f} rad"
        ket = _index_to_spin_string(i, N, bit_up=bit_up, msb_is_q0=msb_is_q0)
        idx_str = f"[{i}]" if show_index else ""
        print(f"  {mag:12.6f}  {arg_str:>12}  {amp.real: .6f}{amp.imag:+.6f}i  {idx_str:>4}  {ket}")

def print_qubit_marginal_probs(vec: np.ndarray, N: int, k: int, bit_up='0->↑'):
    """
    Quick sanity check: compute P(↑) and P(↓) for qubit k from the full state.
    Uses the same bit/↑ mapping and MSB qubit order (q0 is MSB).
    """
    vec = np.asarray(vec).reshape(-1)
    assert vec.size == (1 << N)
    p_up = 0.0
    p_dn = 0.0
    for i, amp in enumerate(vec):
        # bit position for qubit k when q0 is MSB:
        # MSB index is N-1, so the mask for qubit k is at bit (N-1-k)
        bit = (i >> (N - 1 - k)) & 1
        if bit_up == '0->↑':
            # bit 0 => ↑
            if bit == 0: p_up += (amp.conjugate() * amp).real
            else:        p_dn += (amp.conjugate() * amp).real
        else:
            # bit 1 => ↑
            if bit == 1: p_up += (amp.conjugate() * amp).real
            else:        p_dn += (amp.conjugate() * amp).real
    print(f"Qubit {k}:  P(↑)={p_up:.6f},  P(↓)={p_dn:.6f},  P(sum)={p_up+p_dn:.6f}")


if __name__ == '__main__':
    from scipy.interpolate import interp1d
    import pickle
    import os

    # Simulation parameters
    N = 5
    T = 20
    qubit = 0
    dt_list = [0.1]  # can add more, e.g., [0.05, 0.1, 0.2]
    seed = 314
    periodic = False
    if True:  # Used for standard testing
        all_z = []
        all_times = []
        errors = []

        # Paths for caching
        base = './examples/Heisenberg_Chain/cache'
        os.makedirs(base, exist_ok=True)
        histories_path_time = f'{base}/Historydata({seed})_N{N}_alltimes.pkl'
        histories_path_z    = f'{base}/Historydata({seed})_N{N}_allz.pkl'

        # Load or generate trajectories
        try:
            with open(histories_path_time, 'rb') as f:
                all_times = pickle.load(f)
            with open(histories_path_z, 'rb') as f:
                all_z = pickle.load(f)
        except FileNotFoundError:
            for dt in dt_list:
                steps = int(T / dt)
                print(f"Processing dt={dt}, steps={steps}")
                np.random.seed(seed)

                # measure='sz' ensures the chain records ⟨σ_z⟩ for qubit `qubit`
                chain = HeisenbergChain(periodic = periodic, num_qubits=N, target_qubit=qubit, J=1.0, dt=dt, measure='sz')
                chain.evolve(steps)

                z_vals = chain.get_observable()          # ⟨σ_z⟩(t) for target qubit
                times = np.arange(len(z_vals)) * dt
                all_z.append(z_vals)
                all_times.append(times)

            # Cache results
            with open(histories_path_time, 'wb') as f:
                pickle.dump(all_times, f)
            with open(histories_path_z, 'wb') as f:
                pickle.dump(all_z, f)

        # Pretty-print final state and single-qubit marginals
        pretty_print_state(chain.psi, N, top_k=20, tol=0.0, msb_is_q0=True, bit_up='0->↑', phase_in_deg=True)
        print_qubit_marginal_probs(chain.psi, N, k=qubit, bit_up='0->↑')
        print_qubit_marginal_probs(chain.psi, N, k=1, bit_up='0->↑')

        # Reference trajectory for error comparison (first dt in list)
        ref_z = all_z[0]
        ref_t = all_times[0]

        # Compute mean absolute errors vs the first trajectory
        for i, (t_arr, z_arr) in enumerate(zip(all_times, all_z)):
            if i == 0:
                errors.append(0.0)
                continue
            f_interp = interp1d(t_arr, z_arr, bounds_error=False, fill_value="extrapolate")
            errors.append(np.mean(np.abs(f_interp(ref_t) - ref_z)))

        # Plot error vs dt (useful when dt_list has multiple values)
        plt.figure()
        plt.plot(dt_list, errors, marker='o')
        plt.xlabel('dt')
        plt.ylabel('Mean Absolute Error vs smallest dt')
        plt.title('Fidelity loss with increasing dt')
        plt.grid(True)
        plt.tight_layout()

        # Compare one trajectory visually
        plt.figure()
        plt.plot(all_times[0], all_z[0], label=f"dt={dt_list[0]}")
        plt.xlabel('Time')
        plt.ylabel(f"⟨σ_z⟩ (qubit {qubit})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if False:  # Used for perturbed testing
        N = 5
        qubit = 0
        T = 1000
        dt = 0.2
        steps = int(T / dt)

        # Generate initial states
        base_vec, pert1, pert2 = generate_perturbed_states(N, base_seed=31415, epsilon=1e-2)

        # Initialize chains with the different initial states
        chains = []
        for psi in [base_vec, pert1, pert2]:
            chain = HeisenbergChain(num_qubits=N, target_qubit=qubit, dt=dt, measure='sz')
            chain.set_initial_state(psi)
            chain.evolve(steps)
            chains.append(chain)

        # Compare ⟨σ_z⟩ values
        zs = [c.get_observable() for c in chains]
        times = np.arange(len(zs[0])) * dt

        plt.figure(figsize=(10, 5))
        labels = ['Original', 'Perturbed Qubit 0', 'Perturbed All Qubits']
        for i, z in enumerate(zs):
            plt.plot(times, z, label=labels[i])
        plt.xlabel("Time")
        plt.ylabel(f"⟨σ_z⟩ for qubit {qubit}")
        plt.title("Sensitivity to Initial State Perturbations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.xlim(500, 600)

        mae_01 = np.mean(np.abs(zs[0] - zs[1]))
        mae_02 = np.mean(np.abs(zs[0] - zs[2]))
        print(f"MAE (Original vs Perturbed Qubit 0): {mae_01:.6e}")
        print(f"MAE (Original vs Perturbed All):     {mae_02:.6e}")
        plt.show()
      
        
        
        
        
