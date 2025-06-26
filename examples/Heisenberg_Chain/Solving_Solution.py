import os
import pickle
import re

import torch
from torch.utils.data import DataLoader, TensorDataset
from qutip import Qobj, sigmaz, expect
import matplotlib.pyplot as plt
from tqdm import tqdm

from echostate import ESN  # <-- our new ESN module
from .Heisenberg_sim import HeisenbergChain

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class ESNPredictor:
    """
    Train and evaluate an ESN on single-qubit ⟨σ_z⟩ histories
    produced by a HeisenbergChain simulation.
    """
    def __init__(self,
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
                 model_path = None,):
        self.steps = steps
        self.dt = dt
        self.N = N
        self.test_history = history_arrays
        self.dims = dims
        self.qubit = qubit
        self.training_depth = training_depth

        # ESN hyperparameters
        self.washout = washout
        self.batch_size = batch_size
        if not os.path.exists(model_path):
            self.esn = ESN(
                base_input_dim=1,                # only ⟨σ_z⟩ at time t
                reservoir_size=reservoir_size,
                output_dim=1,                    # predicting ⟨σ_z⟩ at t+1
                feedback=feedback,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                leak_rate=leak_rate,
                input_scaling=input_scaling,
                ridge_param=ridge_param,
                washout=washout,
                batch_size=batch_size
            ).to(device)
        else:
            self.esn = torch.load(model_path)
            self.esn.eval() 

        # Pre-generate quantum histories if needed
        self.histories = []
        for _ in range(training_depth):
            chain = HeisenbergChain(N, dt)
            chain.evolve(steps=steps)
            self.histories.append(chain.history)
        print(f"Collected {training_depth} simulation histories.")

    def _build_dataset(self):
        """
        Convert self.histories into lists of (input_seq, target_seq) tensors.
        Each input_seq is shape (T,1), target_seq is (T,1).
        """
        input_list, target_list = [], []

        for run_hist in self.histories:
            # iterate qubits if qubit==0 else just the specified qubit
            qubits = range(self.N) if self.qubit == 0 else [self.qubit]
            for q in qubits:
                # extract ⟨σ_z⟩ time series
                z_series = [float(expect(sigmaz(),
                                         Qobj(rho, dims=self.dims).ptrace(q)))
                            for rho in run_hist]

                # teacher-forced training data:
                # input at t is z_t, target at t is z_{t+1}
                # (we lose the last point as target)
                X = torch.tensor(z_series[:-1], dtype=torch.float32)    .unsqueeze(-1)
                Y = torch.tensor(z_series[1:],  dtype=torch.float32)    .unsqueeze(-1)

                input_list.append(X.to(device))
                target_list.append(Y.to(device))

        # ensure it matches batch_size
        assert len(input_list) == self.batch_size, \
            f"Expected batch_size={self.batch_size}, got {len(input_list)} sequences"

        return input_list, target_list

    def train(self):
        input_list, target_list = self._build_dataset()
        print(f"Training ESN on {len(input_list)} sequences (washout={self.washout})")
        self.esn.fit(input_list, target_list)
        

    def predict_and_plot(self):
        """
        Run prediction on self.test_history, then plot true vs predicted.
        """
        # build a single test sequence from test_history
        z_test = [float(expect(sigmaz(),
                              Qobj(rho, dims=self.dims).ptrace(self.qubit)))
                  for rho in self.test_history]
        print(z_test[:10])
        X_test = torch.tensor(z_test[:-1], dtype=torch.float32).unsqueeze(-1).to(device)
        preds = self.esn.predict([X_test])[0].cpu().numpy().flatten()

        # true targets
        true = z_test[1 + self.washout:]  # align with washout shift

        # plot
        plt.figure(figsize=(8,4))
        
        plt.plot(preds, label='predicted ⟨σ_z⟩')
        plt.plot(true,  label='true ⟨σ_z⟩')
        plt.legend()
        plt.title('ESN Prediction of Single‐Qubit Dynamics')
        plt.savefig(re.sub(r"(Qbts)", r"\1({})".format(qubit+1), f"./examples/Heisenberg_Chain/cache/Errors_{name}.png"))
        plt.show()

    def debug(self):
        """Check covariance conditioning after training."""
        self.esn.trainer.debug_covariance()


if __name__ == '__main__':
    # Example usage
    steps = 10000
    N = 3
    dt = 0.01
    qubit = 1
    washout = 200

    # simulate once or load from pickle
    name = f"Stps{int(steps/1000.0)}_Qbts{N}_dt{dt}".replace(".","_",1)
    histories_path = f'./examples/Heisenberg_Chain/cache/Historydata_{name}.pkl'
    model_path = f'./examples/Heisenberg_Chain/cache/trainedmodel_{name}.pt'
    chain = HeisenbergChain(N, dt)
    try:
        with open(histories_path, 'rb') as f:
            chain.history = pickle.load(f)
    except FileNotFoundError:
        chain.evolve(steps=steps)
        with open(histories_path, 'wb') as f:
            pickle.dump(chain.history, f)
            
    chain.plot_spin_grid(None)
    # print(chain.history[1][:10])
    
    predictor = ESNPredictor(
        steps=steps,
        dt=dt,
        N=N,
        history_arrays=chain.history,
        dims=chain.dims,
        qubit=qubit,
        reservoir_size=208,
        spectral_radius=0.4,
        input_scaling=4.8,
        ridge_param=3e-5,
        leak_rate=0.25,
        sparsity=0.1,
        feedback=1,
        washout=washout,
        batch_size=1,
        training_depth=1,
        model_path = model_path
    )

    if not os.path.exists(model_path):
        predictor.train()
        torch.save(predictor.esn, model_path)
        
    predictor.debug()
    predictor.predict_and_plot()
    
