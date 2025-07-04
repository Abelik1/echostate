import os
import json
import pickle
import numpy as np
import torch
from echostate import ESN  # <-- our new ESN module
from echostate.utils import mean_absolute_error
from .Heisenberg_sim import HeisenbergChain
import matplotlib.pyplot as plt
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESNPredictor:
    """
    Train and evaluate an ESN on single-qubit ⟨σ_z⟩ histories
    produced by a HeisenbergChain simulation.
    """
    def __init__(self,
                 steps: int,
                 dt: float,
                 N: int,
                 qubit: int,
                 history_values: list = None,
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
                 model_path=None,
                 seed=None):
        self.steps = steps
        self.dt = dt
        self.N = N
        self.qubit = qubit
        self.washout = washout
        self.batch_size = batch_size
        self.training_depth = training_depth

        # Initialize or load ESN
        if model_path is None or not os.path.exists(model_path):
            self.esn = ESN(
                base_input_dim=1,
                reservoir_size=reservoir_size,
                output_dim=1,
                feedback=feedback,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                input_scaling=input_scaling,
                ridge_param=ridge_param,
                leak_rate=leak_rate,
                washout=washout,
                batch_size=batch_size,
                seed=seed,
            ).to(device)
        else:
            self.esn = torch.load(model_path)
            self.esn.eval()

        # Prepare test history if provided
        self.test_history = history_values

        # Generate training histories if needed
        self.histories = []
        if history_values is None or training_depth > 0:
            np.random.seed(seed)
            for _ in range(training_depth):
                chain = HeisenbergChain(
                    num_qubits=N,
                    target_qubit=qubit,
                    dt=dt
                )
                chain.evolve(steps)
                # store expectation values
                self.histories.append(chain.get_sz())
            print(f"Collected {len(self.histories)} simulation histories.")

    def _build_dataset(self):
        """
        Convert stored ⟨σ_z⟩ arrays into teacher-forced sequences for ESN.
        Returns lists of Tensors (each shape (T,1)).
        """
        inputs, targets = [], []
        for z_seq in self.histories:
            arr = np.asarray(z_seq)
            X = torch.tensor(arr[:-1], dtype=torch.float32).unsqueeze(-1)
            Y = torch.tensor(arr[1:], dtype=torch.float32).unsqueeze(-1)
            inputs.append(X)
            targets.append(Y)
        # ensure batch size
        assert len(inputs) == self.batch_size, \
            f"Expected batch_size={self.batch_size}, got {len(inputs)} sequences"
        return inputs, targets

    def train(self):
        inputs, targets = self._build_dataset()
        print(f"Training ESN on {len(inputs)} sequences (washout={self.washout})")
        self.esn.fit(inputs, targets)

    def predict_and_plot(self, acc_history=None, acc_chain=None, name="test"):
        """
        Predict with ESN on self.test_history, then plot true vs predicted
        with optional comparison to high‐resolution accurate data.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import torch
        from echostate.utils import mean_absolute_error
        from qutip import Qobj, sigmaz, expect
        import re

        if self.test_history is None:
            raise ValueError("No test history provided.")

        # If your test_history is already ⟨σ_z⟩ values, then this is fine:
        z_test = np.asarray(self.test_history)

        # --- build high‐res reference if given (same as old code)
        if acc_history is not None and acc_chain is not None:
            acc_z = [float(expect(sigmaz(),
                                Qobj(rho, dims=acc_chain.dims).ptrace(self.qubit)))
                    for rho in acc_history]
            acc_dt = acc_chain.dt

        # --- build X_test exactly as you did (batch=1, seq length = len(z_test)-1)
        device = next(self.esn.parameters()).device
        X_test = torch.tensor(
            z_test[:-1], dtype=torch.float32, device=device
        ).unsqueeze(-1).unsqueeze(0)
        preds = self.esn.predict(X_test)[0].cpu().numpy().flatten()

        # --- true trajectory (after washout) just like before
        true = z_test[self.washout+1 : self.washout+1+len(preds)]

        # --- time axes
        coarse_t = np.arange(len(preds)) * self.dt
        true_t   = np.arange(len(true)) * self.dt

        # --- plot
        plt.figure(figsize=(8, 4))

        if acc_history is not None and acc_chain is not None:
            # same offset logic as old
            washout_acc_steps = int((self.washout * self.dt) / acc_dt)
            # plus one coarse‐step offset like you had
            extra = int(self.dt / acc_dt)
            acc_z_trim = acc_z[washout_acc_steps + extra : washout_acc_steps + extra + len(preds)]
            acc_t = np.arange(len(acc_z_trim)) * acc_dt
            plt.plot(acc_t, acc_z_trim, "-o", label="Fully Accurate", markersize=1)

        plt.plot(coarse_t, preds, label='Predicted ⟨σ_z⟩')
        plt.plot(true_t, true,   label='True ⟨σ_z⟩')

        # restore your zoom window:
        plt.xlim(800, 900)

        plt.xlabel("Time")
        plt.ylabel("⟨σ_z⟩")
        plt.title('ESN Prediction of Single‐Qubit Dynamics')
        plt.legend()

        # save exactly like old
        out_dir = './examples/Heisenberg_Chain/cache'
        os.makedirs(out_dir, exist_ok=True)
        # embed the qubit in the filename as you had with the regex
        fname = re.sub(r"(Qbts)", r"\1({})".format(self.qubit + 1),
                    f"Errors_{name}.pdf")
        plt.savefig(f"{out_dir}/{fname}", format="pdf")

        # MAE
        mae = mean_absolute_error(
            torch.tensor(preds), torch.tensor(true)
        )
        print(f"MAE on test: {mae.item():.4f}")

        plt.show()

    def debug(self):
        """Check covariance conditioning after training."""
        self.esn.trainer.debug_covariance()

def Heisen_tune(predictor, study_name, washout, seed, n_trials, plots = False):
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice, plot_contour, plot_edf
    best_params_dict = {}
    input_list, target_list = predictor._build_dataset()

    # ------------ Run Optuna
    study = ESN.tune(input_list, target_list, n_trials=n_trials, direction="minimize",study_name = study_name, washout = washout, seed = seed,
                    reservoir_limit = [100,1500],
                    spectral_radius_limit = [0.1, 1.7],
                    feedback_limit = 1,
                    input_scaling_limit = [0.05, 5.0],
                    ridge_param_limit = [1e-7, 1],
                    leak_rate_limit = [0.2, 1.0],
                    sparsity_limit = [0.1,1.0],
                    )
    if plots:
        plot_optimization_history(study).show()
        plot_param_importances(study).show()
        plot_parallel_coordinate(study).show()
        plot_slice(study).show()
        plot_contour(study).show()
        # plot_edf(study).show()
    # ----- Print best params
    
    print(dt)
    print("Best hyperparameters:", study.best_params)
    print("Best MAE:", study.best_value)
    
    # Save all best parameters to JSON
    best_params_dict[str(round(dt, 5))] = study.best_params
    
    output_path = f'./examples/Heisenberg_Chain/trained_esns/best{seed}_Qbts{N}_params_by_dt.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)
    return study


def dt_loop():
    best_params_dict = {}

    for dt in np.arange(0.1, 0.4, 0.05):
        steps = int(T / dt)
        name = f"Seed{seed}_Qbts{N}_dt{round(dt,5)}".replace(".", "_", 1)
        study_name = f"esnStudy_Seed{seed}_Qbts{N}_dt{round(dt,5)}_dpth{training_depth}"
        model_path = f'./examples/Heisenberg_Chain/trained_esns/trainedmodel_{name}.pt'

        # simulate chain
        np.random.seed(seed)
        chain = HeisenbergChain(N, dt=dt)
        chain.evolve(steps=steps)

        # downsample history
        history = []
        for element in chain.history:
            history.append(element[::int(dt / 0.05)])

        predictor = ESNPredictor(
            steps=steps,
            dt=dt,
            N=N,
            history_arrays=history,
            dims=chain.dims,
            qubit=qubit,
            batch_size=training_depth,
            training_depth=training_depth,
            model_path=model_path,
            seed=seed,
        )

        study = Heisen_tune(predictor, study_name=study_name, washout=washout, seed=seed, n_trials=n_trials)
        best_params_dict[str(round(dt, 5))] = study.best_params

    # Save all best parameters to JSON
    output_path = f'./examples/Heisenberg_Chain/trained_esns/best{seed}_Qbts{N}_params_by_dt.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)
     
   
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Setup parameters
    T = 100
    N = 10
    seed = 31415
    qubit = 0
    washout = 75
    dt = 0.15
    training_depth = 60
    n_trials = -1  # no tuning by default

    np.random.seed(seed)
    # high-resolution reference
    acc_dt = 0.05
    acc_chain = HeisenbergChain(N, qubit, dt=acc_dt)
    acc_steps = int(T / acc_dt)
    acc_chain.evolve(acc_steps)

    # simulate or load low-res history
    steps = int(T / dt)
    name = f"Seed{seed}_Qbts{N}_dt{round(dt,5)}".replace('.', '_', 1)
    histories_path = f"./examples/Heisenberg_Chain/cache/Historydata_{name}.pkl"
    model_path = f'./examples/Heisenberg_Chain/trained_esns/trainedmodel_{name}.pt'
    try:
        with open(histories_path, 'rb') as f:
            z_history = pickle.load(f)
    except FileNotFoundError:
        chain = HeisenbergChain(N, qubit, dt=dt)
        chain.evolve(steps)
        z_history = chain.get_sz()
        os.makedirs(os.path.dirname(histories_path), exist_ok=True)
        with open(histories_path, 'wb') as f:
            pickle.dump(z_history, f)
            
    #dt_loop()
    # Load or fallback best params
    best_params_path = f'./examples/Heisenberg_Chain/trained_esns/best{seed}_Qbts{N}_params_by_dt.json'
    try:
        with open(best_params_path, 'r') as f:
            all_best = json.load(f)
        best = all_best.get(str(round(dt,5)), {})
    except (FileNotFoundError, json.JSONDecodeError):
        best = {}

    predictor = ESNPredictor(
        steps=steps,
        dt=dt,
        N=N,
        qubit=qubit,
        history_values=z_history,
        reservoir_size=best.get('reservoir_size', 983),
        spectral_radius=best.get('spectral_radius', 1.25033),
        input_scaling=best.get('input_scaling', 0.546107),
        ridge_param=best.get('ridge_param', 0.170278),
        leak_rate=best.get('leak_rate', 0.946),
        sparsity=best.get('sparsity', 0.2),
        feedback=best.get('feedback', 1),
        washout=washout,
        batch_size=training_depth,
        training_depth=training_depth,
        model_path= model_path,
        seed=seed,
    )

    
    
    study_name = f"esnStudy_Seed{seed}_Qbts{N}_dt{dt}_dpth{training_depth}"
    # optional tuning
    if n_trials >= 0:
        Heisen_tune(predictor, study_name= study_name, washout=washout, seed=seed, n_trials=max(0, n_trials), plots=False)
    else:
        if not os.path.exists(model_path):
            predictor.train()
            torch.save(predictor.esn, model_path)

        predictor.debug()
        predictor.predict_and_plot(acc_history=acc_chain.get_sz(), acc_chain=acc_chain, name=name)

        
