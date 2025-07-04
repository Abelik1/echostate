import os
import pickle
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from qutip import Qobj, sigmaz, expect
import matplotlib.pyplot as plt
from tqdm import tqdm

from echostate import ESN  # <-- our new ESN module
from .Heisenberg_sim import HeisenbergChain
from echostate.utils import mean_absolute_error

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
                 seed = None,):
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
                batch_size=batch_size,
                seed = seed,
            ).to(device)
        else:
            self.esn = torch.load(model_path)
            self.esn.eval() 

        # Pre-generate quantum histories if needed
        self.histories = []
        np.random.seed(seed)
        for _ in range(training_depth):
            chain = HeisenbergChain(N, dt=dt)
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
            qubits = range(self.N) if self.qubit == -1 else [self.qubit]
            for q in qubits:
                # extract ⟨σ_z⟩ time series
                z_series = [float(expect(sigmaz(),
                                         Qobj(rho, dims=self.dims).ptrace(q)))
                            for rho in run_hist]

                # teacher-forced training data:
                # input at t is z_t, target at t is z_{t+1}
                X = torch.tensor(z_series[:-1], dtype=torch.float32)    .unsqueeze(-1)
                Y = torch.tensor(z_series[1:],  dtype=torch.float32)    .unsqueeze(-1)

                input_list.append(X)
                target_list.append(Y)
        # Pad and batch into single tensors
        X_batch = torch.nn.utils.rnn.pad_sequence(input_list, batch_first=True).to(device)
        Y_batch = torch.nn.utils.rnn.pad_sequence(target_list, batch_first=True).to(device)
        
        # ensure it matches batch_size
        assert len(input_list) == self.batch_size, \
            f"Expected batch_size={self.batch_size}, got {len(input_list)} sequences"

        return X_batch, Y_batch

    def train(self):
        input_list, target_list = self._build_dataset()
        print(f"Training ESN on {len(input_list)} sequences (washout={self.washout})")
        self.esn.fit(input_list, target_list)
        

    def predict_and_plot(self, acc_z_history=None):
        """
        Run prediction on self.test_history, then plot true vs predicted,
        with optional comparison to high-resolution accurate data.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import re
        from qutip import Qobj, sigmaz, expect
        from echostate.utils import mean_absolute_error

        acc_dt = acc_chain.dt
        # build ⟨σ_z⟩ trajectory from test history
        z_test = [float(expect(sigmaz(),
                            Qobj(rho, dims=self.dims).ptrace(self.qubit)))
                for rho in self.test_history]

        # build accurate ⟨σ_z⟩ trajectory if provided
        if acc_z_history:
            acc_z = [float(expect(sigmaz(),
                                Qobj(rho, dims=acc_chain.dims).ptrace(self.qubit)))
                    for rho in acc_z_history]
            

        # generate ESN predictions
        X_test, _ = self._build_dataset()
        preds = self.esn.predict(X_test)[0].cpu().numpy().flatten()

        # true targets for comparison
        true = z_test[self.washout+1: len(preds) + self.washout+1]

        # time axes
        coarse_t = np.arange(len(preds)) * self.dt
        true_t = np.arange(len(true)) * self.dt
        
        if acc_z_history:
            washout_acc_steps = int((self.washout*self.dt)/acc_chain.dt)
            acc_z = acc_z[int(washout_acc_steps+(self.dt/acc_chain.dt)):]
            acc_t = np.arange(len(acc_z)) * acc_dt
        # plot
        plt.figure(figsize=(8, 4))
        if acc_z_history:
            plt.plot(acc_t, acc_z,"-o", label="Fully Accurate", markersize = "1")
        plt.plot(coarse_t, preds, label='Predicted ⟨σ_z⟩')
        plt.plot(true_t, true, label='True ⟨σ_z⟩')
        plt.xlim(800,900)
        plt.xlabel("Time")
        plt.ylabel("⟨σ_z⟩")
        plt.legend()
        plt.title('ESN Prediction of Single‐Qubit Dynamics')

        # Save and show
        plt.savefig(re.sub(r"(Qbts)", r"\1({})".format(self.qubit + 1),
                        f"./examples/Heisenberg_Chain/cache/Errors_{name}.png"),format="pdf")
        mae = mean_absolute_error(torch.tensor(preds), torch.tensor(true))
        print(f"MAE on test: {mae.item():.4f}")
        plt.show()

    def debug(self):
        """Check covariance conditioning after training."""
        self.esn.trainer.debug_covariance()


def Heisen_tune(predictor, study_name, washout, seed, n_trials, plots = False):
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice, plot_contour, plot_edf

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
    output_path = "./examples/Heisenberg_Chain/trained_esns/best_params_by_dt.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)
     
   
if __name__ == '__main__':
    import os
    import json
    import pickle
    import torch
    import numpy as np

    # ---- Initialization
    T = 100
    N = 10
    seed = 31415
    qubit = 0
    washout = 200
    dt = 0.2
    training_depth = 5
    n_trials = -1  # set > 0 to tune and save new best hyperparams ( -1 just prints current best)

    np.random.seed(seed)
    
    acc_dt = 0.05
    acc_chain = HeisenbergChain(N,dt=acc_dt)
    acc_steps = int(T / acc_dt)
    acc_chain.evolve(steps=acc_steps)
    # ---- Run tuning loop and save best hyperparameters
    # dt_loop()

    # ---- Setup names and paths
    steps = int(T / dt)
    name = f"Seed{seed}_Qbts{N}_dt{round(dt,5)}".replace(".", "_", 1)
    model_path = f'./examples/Heisenberg_Chain/trained_esns/trainedmodel_{name}.pt'
    histories_path = f'./examples/Heisenberg_Chain/cache/Historydata_{name}.pkl'

    # ---- Generate or load data
    np.random.seed(seed)
    chain = HeisenbergChain(N, dt=dt)

    try:
        with open(histories_path, 'rb') as f:
            chain.history = pickle.load(f)
    except FileNotFoundError:
        chain.evolve(steps=steps)
        with open(histories_path, 'wb') as f:
            pickle.dump(chain.history, f)

    # ---- Load best params from JSON if available
    best_params_path = f'./examples/Heisenberg_Chain/trained_esns/best{seed}_Qbts{N}_params_by_dt.json'
    try:
        with open(best_params_path, 'r') as f:
            all_best_params = json.load(f)
        best_params = all_best_params.get(str(round(dt, 5)), {})
        print("Found best parameters")
    
    except (FileNotFoundError, json.JSONDecodeError):
        best_params = {}

    # ---- Manual fallback if no best params found
    predictor = ESNPredictor(
        steps=steps,
        dt=dt,
        N=N,
        history_arrays=chain.history,
        dims=chain.dims,
        qubit=qubit,

        reservoir_size=best_params.get('reservoir_size', 983),
        spectral_radius=best_params.get('spectral_radius', 1.25033),
        feedback=best_params.get('feedback', 1),
        input_scaling=best_params.get('input_scaling', 0.546107),
        ridge_param=best_params.get('ridge_param', 0.170278),
        leak_rate=best_params.get('leak_rate', 0.946),
        sparsity=best_params.get('sparsity', 0.2),

        washout=washout,
        batch_size=training_depth,
        training_depth=training_depth,
        model_path=model_path,
        seed=seed,
    )

    # ---- Optional: train and save
    if not os.path.exists(model_path):
        predictor.train()
        torch.save(predictor.esn, model_path)

    #---- Optional: debug and plot
    predictor.debug()
    predictor.predict_and_plot(acc_z_history = acc_chain.history )

    # ---- Optional: tuning (no-op if n_trials == 0)
    study_name = f"esnStudy_Seed{seed}_Qbts{N}_dt{dt}_dpth{training_depth}"
    if n_trials == -1:
        Heisen_tune(predictor, study_name=study_name, washout=washout, seed=seed, n_trials=0, plots = True)
    elif n_trials > 0:
        Heisen_tune(predictor, study_name=study_name, washout=washout, seed=seed, n_trials=n_trials, plots = True)
        
