# echostate
Custom method for echo state networks using pytorch

To reproduce requirements I should use pip freeze > requirements.txt

# 🧠 HeisenbergChain: N-Qubit Heisenberg Model Simulator

`HeisenbergChain` simulates the evolution of an N-qubit Heisenberg spin chain using QuTiP and stores the time history efficiently as NumPy arrays.

---

## ✅ Constructor

```python
HeisenbergChain(num_qubits, J=1.0, dt=0.01)
```

| Parameter    | Type    | Default | Description                                      |
| ------------ | ------- | ------- | ------------------------------------------------ |
| `num_qubits` | `int`   | —       | Number of qubits in the chain (N).               |
| `J`          | `float` | `1.0`   | Coupling strength of the Heisenberg interaction. |
| `dt`         | `float` | `0.01`  | Time step for evolution.                         |

👉 The chain is initialized with a random pure state.

---

## ✅ Methods

### `evolve(steps)`

```python
evolve(steps: int)
```

Evolves the system for the specified number of time steps. Each step applies the unitary evolution:

$$
U = e^{-i H dt}
$$

The resulting density matrix is stored in `self.history` as a NumPy array.

---

### `get_rho_qobj(t)`

```python
get_rho_qobj(t: int) -> Qobj
```

Reconstructs and returns the density matrix at time index `t` as a QuTiP `Qobj`.

---

### `check_conservation()`

```python
check_conservation() -> dict
```

Returns min and max values across the history for:

* **Trace** (should stay \~1)
* **Purity**
* **Energy**
* **Total magnetization ⟨Sᶻ⟩**

Example output:

```python
{
  'trace': (0.999, 1.001),
  'purity': (0.5, 1.0),
  'energy': (-1.23, -1.23),
  'magnetization': (-0.1, 0.1)
}
```

---

### `plot_spin_grid(size)`

```python
plot_spin_grid(size: int)
```

Visualizes ⟨σ\_z⟩ for each qubit over time as a color grid scatter plot.

| Parameter | Description                                           |
| --------- | ----------------------------------------------------- |
| `size`    | Max number of data points to plot (controls density). |

---

## ✅ Example usage

```python
chain = HeisenbergChain(num_qubits=3, J=1.0, dt=0.01)
chain.evolve(steps=100)
chain.plot_spin_grid(size=300)
stats = chain.check_conservation()
print(stats)
```




Some plans
Find the fidelity of dt for Heisenberg chain that retains information

Train a ESN now, see if when you input a different dt can it track a higher dt level


