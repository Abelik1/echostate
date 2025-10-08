# 🧠 EchoState & HeisenbergChain

A PyTorch-based toolkit for physics-informed Echo State Networks (ESNs), featuring a built-in simulator for quantum spin chains using the Heisenberg model.

---

## 🧭 Table of Contents

* [📦 Installation](#-installation)
* [♻️ What are Echo State Networks?](#-what-are-echo-state-networks)
* [📚 Module: `echostate`](#-module-echostate)

  * [✅ Components](#-components)
  * [✅ Basic Usage](#-basic-usage)
  * [⚖️ Advanced Features](#-advanced-features)
* [🧪 Physics Module: `HeisenbergChain`](#-physics-module-heisenbergchain)

  * [✅ Constructor](#-constructor)
  * [✅ Methods](#-methods)
  * [✅ Example Usage](#-example-usage)
* [🌀 Fidelity and Time Resolution Study](#-fidelity-and-time-resolution-study)
* [🧠 Learning Quantum Dynamics with ESNs](#-learning-quantum-dynamics-with-esns)
* [🗺️ Future Directions](#-future-directions)

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ♻️ What are Echo State Networks?

**Echo State Networks (ESNs)** are a class of recurrent neural networks (RNNs) designed to handle temporal sequence learning with minimal training overhead. Unlike standard RNNs, only the output layer is trained—while the recurrent “reservoir” is fixed and randomly initialized with special properties like:

* Sparse connectivity
* Spectral radius control (to ensure echo-state property)
* Efficient linear regression-based output learning

This architecture is highly suited for chaotic, nonlinear, or temporal physics data—such as quantum systems, particle trajectories, or spin chains.

---

## 📚 Module: `echostate`

### ✅ Components

* `ESN`: Core Echo State Network model
* `Reservoir`: Dynamical reservoir layer
* `Trainer`: Linear output trainer (ridge regression)
* `utils`: Error metrics and spectral radius utilities

### ✅ Basic Usage

```python
from echostate import ESN
import torch

# Example data
inputs = [torch.randn(100, 1)]
targets = [torch.sin(torch.linspace(0, 10, 100)).unsqueeze(1)]

# Initialize ESN
model = ESN(base_input_dim=1, output_dim=1, reservoir_size=200, feedback=1)

# Train
model.fit(inputs, targets)

# Predict
predictions, metrics = model.predict(inputs, targets)
print(metrics)  # {'mae': ..., 'mse': ...}
```

### ⚖️ Advanced Features

* **Feedback**: Incorporate previous outputs into the input stream
* **Washout**: Discard initial transient states
* **Hyperparameter Tuning**: Integrated Optuna-based tuner with `ESN.tune(...)`
* **Diagnostics**: `Trainer.debug_covariance()` helps assess conditioning

---

## 🧪 Physics Module: `HeisenbergChain`

> Simulates a quantum Heisenberg spin chain and exports evolution history for use in machine learning or analysis.

---

### 🧪 Theoretical Background: The Heisenberg Model

The **Heisenberg model** is a fundamental model in quantum magnetism that describes interacting spins on a lattice. Each spin is represented by a quantum two-level system (qubit), and interactions occur between neighboring spins.

The **Hamiltonian** for an N-qubit Heisenberg spin chain with periodic boundary conditions is given by:

$$
H = -\frac{J}{2} \sum_{j=1}^{N} \left( \sigma_j^x \sigma_{j+1}^x + \sigma_j^y \sigma_{j+1}^y + \sigma_j^z \sigma_{j+1}^z \right)
$$

Where:

* $J$ is the coupling constant (positive for ferromagnetic interaction).
* $\sigma_j^x, \sigma_j^y, \sigma_j^z$ are the Pauli matrices acting on the $j$-th qubit.
* Periodic boundary conditions imply $\sigma_{N+1} = \sigma_1$.

This Hamiltonian conserves total magnetization $\sum_j \sigma_j^z$, energy, and the trace of the density matrix, making it a useful testbed for quantum dynamics and machine learning predictions.

The **unitary time evolution** of the quantum system follows Schrödinger's equation:

$$
\rho(t + \Delta t) = U \rho(t) U^\dagger, \quad \text{where} \quad U = e^{-i H \Delta t}
$$

The system is initialized with a **random pure state** $|\psi_0\rangle$, from which the initial density matrix is built:

$$
\rho_0 = |\psi_0\rangle \langle \psi_0|
$$

Each subsequent state is evolved step-by-step using the fixed unitary operator $U$.

---

### ✅ Constructor

```python
HeisenbergChain(num_qubits, J=1.0, dt=0.01)
```

| Parameter    | Type    | Default | Description                    |
| ------------ | ------- | ------- | ------------------------------ |
| `num_qubits` | `int`   | —       | Number of qubits in the chain. |
| `J`          | `float` | `1.0`   | Heisenberg coupling constant.  |
| `dt`         | `float` | `0.01`  | Timestep of simulation.        |

👉 The chain initializes with a random pure state $|\psi_0\rangle$, stored as a density matrix $\rho_0$.

---

## ✅ Methods

### `evolve(steps: int)`

Evolves the system for the specified number of time steps. Each step applies the unitary evolution:

$$
U = e^{-i H \Delta t}
$$

The resulting density matrix is stored in `self.history` as a NumPy array.

---

### `get_rho_qobj(t: int) -> Qobj`

Reconstructs and returns the density matrix at time index `t` as a QuTiP `Qobj`.

---

### `check_conservation() -> dict`

Returns minimum and maximum values across the history for:

* **Trace** (should stay $\approx 1$)
* **Purity** $\text{Tr}(\rho^2)$
* **Energy** $\text{Tr}(H \rho)$
* **Total magnetization** $\langle S^z \rangle = \sum_j \langle \sigma_j^z \rangle$

**Example output:**

```python
{
  'trace': (0.999, 1.001),
  'purity': (0.5, 1.0),
  'energy': (-1.23, -1.23),
  'magnetization': (-0.1, 0.1)
}
```

---

### `plot_spin_grid(size: int)`

Visualizes $\langle \sigma_z \rangle$ for each qubit over time as a color grid scatter plot.

| Parameter | Description                                           |
| --------- | ----------------------------------------------------- |
| `size`    | Max number of data points to plot (controls density). |

---

## ✅ Example Usage

```python
chain = HeisenbergChain(num_qubits=3, J=1.0, dt=0.01)
chain.evolve(steps=100)
chain.plot_spin_grid(size=300)
stats = chain.check_conservation()
print(stats)
```

---


---

## 🌀 Fidelity and Time Resolution Study

The file `Heisenberg_sim.py` also supports studying how the time step `dt` affects simulation fidelity. It compares ⟨σ\_z⟩ evolution of qubits at different `dt` values and calculates error relative to a high-resolution reference.

### Highlights

* Caches simulation histories using `pickle`
* Interpolates ⟨σ\_z⟩ curves to compare different time resolutions
* Computes mean absolute error (MAE)
* Visualizes loss of fidelity as `dt` increases

---

## 🧠 Learning Quantum Dynamics with ESNs

The `Solving_Solution.py` script defines a pipeline to train and evaluate an Echo State Network (ESN) on the time evolution of a single qubit's ⟨σ\_z⟩ expectation value, derived from a Heisenberg spin chain simulation. It uses simulation data from `Heisenberg_sim.py` and prepares the inputs for ESN training.

---

### ✅ `ESNPredictor` Class

#### Constructor

```python
ESNPredictor(
    steps: int,
    dt: float,
    N: int,
    history_arrays: list,
    dims: list,
    qubit: int,
    reservoir_size: int = 100,
    spectral_radius: float = 0.9,
    input_scaling: float = 1.0,
    ridge_param: float = 1e-3,
    leak_rate: float = 0.9,
    sparsity: float = 1.0,
    feedback: int = 1,
    washout: int = 0,
    batch_size: int = 1,
    training_depth: int = 1,
    model_path = None,
    seed = None
)
```

#### Parameters

| Parameter         | Type    | Description                                              |
| ----------------- | ------- | -------------------------------------------------------- |
| `steps`           | `int`   | Number of time steps to simulate per training history.   |
| `dt`              | `float` | Time resolution per step.                                |
| `N`               | `int`   | Number of qubits in the Heisenberg chain.                |
| `history_arrays`  | `list`  | List of density matrix arrays from prior simulation.     |
| `dims`            | `list`  | Qobj-compatible dimension structure.                     |
| `qubit`           | `int`   | Index of the qubit to extract ⟨σ\_z⟩ dynamics from.      |
| `reservoir_size`  | `int`   | Number of internal reservoir units.                      |
| `spectral_radius` | `float` | Controls memory depth of reservoir (eigenvalue scaling). |
| `input_scaling`   | `float` | Scaling factor for input projection.                     |
| `ridge_param`     | `float` | L2 regularization strength for output layer.             |
| `leak_rate`       | `float` | Leaky integrator blending factor (0 = slow, 1 = fast).   |
| `sparsity`        | `float` | Fraction of non-zero recurrent weights.                  |
| `feedback`        | `int`   | Number of past outputs included in input.                |
| `washout`         | `int`   | Number of initial timesteps ignored during training.     |
| `batch_size`      | `int`   | Number of training sequences. Must match training depth. |
| `training_depth`  | `int`   | Number of simulated histories used for training.         |
| `model_path`      | `str`   | File path for saving or loading the trained model.       |
| `seed`            | `int`   | Random seed for reproducibility.                         |

---

### 🔧 `train()`

```python
ESNPredictor.train()
```

* Builds a dataset of inputs and targets from quantum histories.
* Calls `fit()` on the ESN using teacher-forced sequences.
* Ensures batch size consistency.

---

### 📈 `predict_and_plot()`

```python
ESNPredictor.predict_and_plot()
```

* Uses the trained ESN to predict the future ⟨σ\_z⟩ values of the selected test qubit.
* Plots the predicted vs true trajectory.
* Saves figure to cache.
* Prints mean absolute error (MAE).

---

### 🛠 `debug()`

```python
ESNPredictor.debug()
```

* Prints the condition number of the ESN's covariance matrix.
* Useful for diagnosing overfitting or rank deficiency.

---

### 🎯 `optuna()` Function

```python
optuna()
```

This script-level function invokes the built-in `ESN.tune()` method using [Optuna](https://optuna.org/) to find optimal hyperparameters for the ESN.

#### Optimization Targets

* **Minimize**: Mean Absolute Error (MAE)
* Runs `n_trials` trials using random + Bayesian optimization.
* Stores results in a local SQLite database for reusability.

#### Key Arguments (inside `ESN.tune(...)`)

| Parameter               | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| `reservoir_limit`       | Range for number of reservoir neurons.                     |
| `spectral_radius_limit` | Controls memory capacity via recurrent matrix eigenvalues. |
| `feedback_limit`        | Number of past outputs included in input.                  |
| `input_scaling_limit`   | Adjusts strength of input projection.                      |
| `ridge_param_limit`     | L2 penalty regularization range (log-scaled).              |
| `leak_rate_limit`       | Controls dynamical speed of reservoir states.              |
| `sparsity_limit`        | Fraction of active (non-zero) weights in reservoir matrix. |

#### Output

* Best parameters
* Best validation MAE

You can customize the number of trials, search bounds, and direction (minimize/maximize) as needed.


### SOME NOTES

Graphing names:
Errors_Seed{}_T{}_Qbts({}){}_dt{}_dpth{}.pdf
-They should be scattering plots, saving without washout. Should probably save the prob plots
-In the plot should include what fidelity high res is

Cache naming
Historydata_Seed{}_T{}_Qbts({}){}_dt{}.pkl
Historydata_Seed{}_T{}_Qbts(ALL){}_dt{}.pkl

Trained_esns cache
bestparams_Seed{}_Qbts({}){}_dt{}_dpth{}_wsht{}.json

trainedmodel_Seed{}_Qbts({}){}_dt{}_dpth{}_wsht{}.pt


esnStudy
esnStudy_Seed{}_Qbts({}){}_dt{}_dpth{}_wsht{}.pt

Where all 
dt{round(dt,5)} has some attribute to change .replace('.', '_', 1)


## ESN Hyperparameter Guide

This table explains each hyperparameter, what it does, and the typical effects of low vs. high values.

| Hyperparameter | Definition | Low Value Effects | High Value Effects | Key Interactions |
| --- | --- | --- | --- | --- |
| **`reservoir_size`** | Number of recurrent neurons in the reservoir (feature dimension of the nonlinear expansion). | Fast training, less overfit risk, but low capacity and poor modeling of complex/long-term dynamics. | Richer dynamics, better at nonlinear tasks, but slower training and higher overfit risk on small datasets. | Larger values often require higher `ridge_param` to avoid overfitting. |
| **`spectral_radius`** | Scales the largest eigenvalue of the recurrent weight matrix; controls memory length. | Quick state decay, short memory, more reactive to input. | Longer memory; too high (>1.2) may cause instability/chaos. | High values + high `leak_rate` → very slow state decay. |
| **`feedback`** | Number of past output steps fed back into the reservoir. | No autoregressive context. | Can improve multi-step prediction but risks error accumulation in closed-loop. | Requires `base_input_dim == output_dim` if replacing base input with predictions. |
| **`input_scaling`** | Scales input-to-reservoir weights (`W_in`); controls input drive strength. | Weak input drive, reservoir dominated by internal dynamics; may fail if signal is small. | Strong input drive, more reactive but less memory; too high can saturate activation function. | Low `input_scaling` + high `spectral_radius` → reservoir acts like long-term memory bank. |
| **`bias_scaling`** | Range for bias term in reservoir neurons. | Symmetric activations around zero; reduced diversity. | Breaks symmetry and increases diversity; too high may saturate neurons. | Works with `input_scaling` to set operating range of `tanh`. |
| **`ridge_param`** | Regularization λ in ridge regression for readout weights. | Low: fits noise, unstable if `X` is ill-conditioned. | High: more stable, less overfit, but can underfit and oversmooth. | Larger datasets and reservoirs often need higher λ. |
| **`leak_rate`** | Proportion of new activation replacing old state each step. | Slow updates, smoother dynamics, long memory; may miss fast changes. | Fast updates, responsive, but short memory unless `spectral_radius` is high. | Low leak + high spectral radius → very slow-changing reservoir. |
| **`sparsity`** | Fraction of recurrent connections set to zero. | Dense: richer dynamics, but more instability risk at high spectral radius. | Sparse: faster compute, simpler dynamics; too sparse may reduce expressivity. | Very sparse + small reservoir can harm performance. |
| **`washout`** | Number of timesteps ignored during training to discard unstable transients. | More training data kept but includes unstable states. | Safer training but fewer samples for regression. | Needed washout increases with high `spectral_radius` and low `leak_rate`. |
| **`learning_algo`** | Solver for ridge regression (`"inv"` or `"cholesky"`). | `"inv"`: fast but less stable for ill-conditioned matrices. | `"cholesky"`: stable and often faster for SPD matrices. | Prefer `"cholesky"` for large datasets or high `reservoir_size`. |
| **`seed`** | RNG seed for reproducible reservoir initialization. | Random each run; more variety, less reproducibility. | Fixed seed ensures repeatable experiments. | Different seeds can drastically change performance for small reservoirs. |

**Tips:**
- Larger datasets benefit from slightly higher `ridge_param` and feature standardization before regression.
- When increasing `reservoir_size`, adjust `ridge_param` to control overfitting.
- Always verify washout is long enough for your spectral radius and leak rate combination.



When to use which (quick guide)

cholesky / solve: Fastest for well‑conditioned SPD systems (yours usually is with λ>0).

eigh: Very stable and transparent; easy to clamp tiny eigenvalues if conditioning is poor.

cg: Scales to large reservoirs when forming factorizations is expensive; stop early for a bias–variance trade‑off (useful in noisy physics data).

svd: Gold standard for numerical stability; handles rank deficiency gracefully; best if you can keep X around.

tsvd: Adds explicit low‑rank regularization; great when your design has lots of near‑collinear features (common with long washouts).

qr: Solid for least squares; with diagonal loading it behaves like a fast approximate ridge.

pinv: Simple and robust; with the Tikhonov augmentation it reproduces ridge via a single pinv.

Implementation notes

If you want qr/svd/pinv, keep self.X and self.Y (post‑washout, with the bias column) alongside xTx/xTy.

For cg, set sensible defaults (e.g., rtol=1e-6, maxiter=1000–5000) and log residual norms so your physics runs are auditable.

For extra safety, add a small jitter to A before factorizations: A = A + 1e-12 * I (scale by A.norm() if needed).

If you compute multiple outputs, all solvers above handle multiple RHS (xTy has shape (R+1, d_out)), so they’re vectorized.


TODO
Get all the graphs we have ever needed.
So we have the graph with histograms and fourier analysis
One plot of 1 esn at tuned values on trained values
ont plot of 1 esn at tuned values on non trianed values
^Train this on variation of Large times steps, lots of batches.

Then make plots with a lot of trained esns, without tolerance
MAke plots with a lot of trained esns with tolerance