import os
import json
import pickle
import numpy as np
import torch
from echostate import ESN  # <-- our new ESN module
from echostate.utils import mean_absolute_error
from .Heisenberg_sim import HeisenbergChain
import matplotlib.pyplot as plt

device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# Extra check
if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")



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
                 bias_scaling: float = 0.2,
                 washout: int = 0,
                 batch_size: int = 1,
                 training_depth: int = 1,
                 model_path=None,
                 cache_dir="./examples/Heisenberg_Chain/cache/",
                 seed=None,
                 device = device,):
        
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
                device=device,
                base_input_dim=1,
                reservoir_size=reservoir_size,
                output_dim=1,
                feedback=feedback,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                input_scaling=input_scaling,
                ridge_param=ridge_param,
                leak_rate=leak_rate,
                bias_scaling = bias_scaling,
                washout=washout,
                batch_size=batch_size,
                seed=seed,
            ).to(device)
        else:
            self.esn = torch.load(model_path)
            self.esn.to(device).eval()

        # Prepare test history if provided
        self.test_history = history_values

        # Load or generate training histories
        self.histories = []
        # History cache follows ALL-qubit format
        fmt_dt_val = str(round(self.dt, 5)).replace(".", "_", 1)
        qubit_tag = f"Qbts(dpth{training_depth}){self.N}"
        cache_name = f"Historydata_Seed{seed}_T{T}_{qubit_tag}_dt{fmt_dt_val}.pkl"
        cache_path = os.path.join(cache_dir, cache_name)

        if history_values is None or training_depth > 0:
            # Check for cached histories
            if cache_path is not None and os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.histories = pickle.load(f)
                print(f"Loaded {len(self.histories)} training histories from cache.")
            else:
                # Generate new histories
                np.random.seed(seed)
                for _ in range(training_depth):
                    chain = HeisenbergChain(
                        num_qubits=N,
                        target_qubit=qubit,
                        dt=dt
                    )
                    chain.evolve(steps)
                    self.histories.append(chain.get_sz())
                print(f"Collected {len(self.histories)} simulation histories.")
                
                # Save to cache
                if cache_path is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.histories, f)
                    print(f"Saved training histories to {cache_path}.")

    def _build_dataset(self):
        """
        Convert stored ⟨σ_z⟩ arrays into teacher-forced sequences for ESN.
        Returns lists of Tensors (each shape (T,1)).
        """
        inputs, targets = [], []
        for z_seq in self.histories:
            arr = np.asarray(z_seq)
            X = torch.tensor(arr[:-1], dtype=torch.float32, device=device).unsqueeze(-1)
            Y = torch.tensor(arr[1:], dtype=torch.float32, device=device).unsqueeze(-1)
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
#region Plotting
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

        # Use test history directly if it's already ⟨σ_z⟩ values or reduced density matrices
        if isinstance(self.test_history[0], (float, np.floating, np.complexfloating)):
            z_test = np.asarray(self.test_history)
        else:
            # Convert reduced density matrices to ⟨σ_z⟩ values
            z_test = np.asarray([
                            float(expect(sigmaz(), Qobj(rho, dims=[[2], [2]])))
                            for rho in self.test_history
        ])

        # --- build high‐res reference if given
        if acc_history is not None and acc_chain is not None:
            acc_z = [float(expect(sigmaz(), Qobj(rho, dims=[[2], [2]])))
                    for rho in acc_history]
            acc_dt = acc_chain.dt

        # --- build X_test (batch=1, seq length = len(z_test)-1)
        
        X_test = torch.tensor(z_test[:-1], dtype=torch.float32, device=device).unsqueeze(-1).unsqueeze(0)
        preds = self.esn.predict(X_test)[0].cpu().numpy().flatten()

        # --- true trajectory (after washout)
        # print("Z_test:", z_test[:20])
        true = z_test[self.washout+1 : self.washout+1+len(preds)]
        # print("True:", true[:20])
        # --- time axes
        coarse_t = np.arange(len(preds)) * self.dt
        true_t   = np.arange(len(true)) * self.dt

        # --- plot
        plt.figure(figsize=(8, 4))

        if acc_history is not None and acc_chain is not None:
            washout_acc_steps = int((self.washout * self.dt) / acc_dt)
            extra = int(self.dt / acc_dt)
            acc_z_trim = acc_z[washout_acc_steps + extra : washout_acc_steps + extra + len(preds)]
            acc_t = np.arange(len(acc_z_trim)) * acc_dt
            # print("acc_z: ",acc_z[:20])
            plt.plot(acc_t, acc_z_trim, label=f"Fully Accurate dt={acc_dt}")
            
        # print("preds: ", preds[:20])
        plt.plot(coarse_t, preds,"-o", label='Predicted ⟨σ_z⟩', markersize = "5")
        plt.plot(true_t, true, "-o",   label='True ⟨σ_z⟩', markersize = "1")

        plt.xlim(50, 70)
        plt.xlabel("Time")
        plt.ylabel("⟨σ_z⟩")
        plt.title(f'ESN Prediction of Single‐Qubit({self.qubit}){self.N} at T:{T} and dt:{self.dt} Dynamics')
        plt.legend()

        out_dir = './examples/Heisenberg_Chain/cache'
        os.makedirs(out_dir, exist_ok=True)
        fname = f"Errors_{name}.pdf"
        plt.savefig(f"{out_dir}/{fname}", format="pdf")

        mae = mean_absolute_error(
            torch.tensor(preds), torch.tensor(true)
        )
        print(f"MAE on test: {mae.item():.4f}")

        # plt.show()
        
        # plt.figure()
        # plt.plot(z_test, label='Z test raw')
        # plt.plot(np.arange(self.washout + 1, self.washout + 1 + len(preds)), preds, label='Predicted (aligned)')
        # plt.legend()
        # plt.title("Raw history and prediction overlay")
        # plt.show()


    def debug(self):
        """Check covariance conditioning after training."""
        self.esn.trainer.debug_covariance()




#region EXAMPLE
# -------------Example CASE------------------------------------------
def plot_hyperparams_vs_N(perm_dir):
    import re
    """
    Scan a directory for bestparams_*.json files, extract system size N from filenames,
    load hyperparameters, and plot how they vary with N.

    Parameters:
        perm_dir (str): Directory containing bestparams_*.json files
    """
    param_files = [f for f in os.listdir(perm_dir) if f.startswith("bestparams_") and f.endswith(".json")]
    param_data = {}

    for fname in param_files:
        try:
            # Extract system size N from "Qbts(X)Y" → Y
            match = re.search(r'Qbts\(\d+\)(\d+)', fname)
            if not match:
                continue
            N_value = int(match.group(1))

            with open(os.path.join(perm_dir, fname), 'r') as f:
                data = json.load(f)

            # Assume structure like { "0.2": { ...params... } }
            for key, val in data.items():
                param_data[N_value] = val

        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")

    if not param_data:
        print("No valid parameter files found.")
        return

    sorted_N = sorted(param_data.keys())
    param_names = list(next(iter(param_data.values())).keys())

    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 2.5 * len(param_names)), sharex=True)
    if len(param_names) == 1:
        axes = [axes]

    for i, param in enumerate(param_names):
        y = [param_data[N][param] for N in sorted_N]
        axes[i].plot(sorted_N, y, marker='o')
        axes[i].set_ylabel(param)
        axes[i].grid(True)

    axes[-1].set_xlabel("System Size (N)")
    fig.suptitle("ESN Hyperparameters vs System Size (N)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
def Heisen_tune(predictor, study_name, study_loc, washout, seed, n_trials, plots = False):
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice, plot_contour, plot_edf
    import optuna
    best_params_dict = {}
    input_list, target_list = predictor._build_dataset()

    # ------------ Run Optuna
    try:
        study = ESN.tune(input_list, target_list, n_trials=n_trials, direction="minimize",study_name = study_name, study_loc= study_loc, washout = washout, seed = seed,
                        reservoir_limit = [700,1000],
                        spectral_radius_limit = [0.1, 2],
                        feedback_limit = 1,
                        input_scaling_limit = [1.0, 3.0],
                        ridge_param_limit = [1e-8, 1],
                        leak_rate_limit = [0.2, 1.0],
                        sparsity_limit = [0.1,1.0],
                        bias_scaling_limit= [0.2,0.8],
                        device = device,
                        learning_algo="inv"
                        )
    except KeyboardInterrupt:
        print("Interrupted! Loading best trial so far...")
        # Save all best parameters to JSON
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_loc}/{study_name}.db")
        best_params_dict[str(round(dt, 5))] = study.best_params
        
        output_path = f'./examples/Heisenberg_Chain/trained_esns/bestparams_{name}.json'
        print(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(best_params_dict, f, indent=4)
            print("Saved Best parameters")
        exit()
    
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
    
    output_path = f'./examples/Heisenberg_Chain/trained_esns/bestparams_{name}.json'
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)
        print("Saved Best parameters")
    return study

def dt_loop(): # OUtdated
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
    
#region CONTROL  
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Setup parameters
    T = 100
    N = 5
    seed = 31415
    qubit = 0
    washout = 75
    dt = 0.2
    training_depth = 300
    n_trials = 100 # no tuning by default

    for N in [4]:

        print(f"Solving N: {N}") 
        np.random.seed(seed)
        # high-resolution reference
        acc_dt = 0.05
        acc_chain = HeisenbergChain(N, qubit, dt=acc_dt)
        acc_steps = int(T / acc_dt)

        acc_chain.evolve(acc_steps, store_reduced=True)

        # acc_chain.plot()
        # plt.show()
        # --- Simulation setup ---
        steps = int(T / dt)
        fmt_dt_val = str(round(dt, 5)).replace(".", "_", 1)
        qubit_tag = f"Qbts({qubit + 1}){N}"
        
        
        name = f"Seed{seed}_{qubit_tag}_dt{fmt_dt_val}_dpth{training_depth}_wsht{washout}"
        
        
        # --- File locations ---
        cache_dir = "./examples/Heisenberg_Chain/cache/"
        perm_dir = "./examples/Heisenberg_Chain/trained_esns/"

        # Low-resolution history file for specific qubit
        histories_path = f"{cache_dir}Historydata_Seed{seed}_T{T}_{qubit_tag}_dt{fmt_dt_val}.pkl"

        # Trained ESN model file
        model_path = f"{cache_dir}trainedmodel_{name}.pt"

        # Best hyperparameter JSON
        best_params_path = f"{perm_dir}bestparams_{name}.json"

        # Optuna study name
        st_name = f"Seed31415_{qubit_tag}_dt{fmt_dt_val}_dpth50_wsht{washout}"
        study_name = f"esnStudy_{st_name}"
        
        try:
            with open(histories_path, 'rb') as f:
                z_history = pickle.load(f)
            print("Loading z history...")
        except FileNotFoundError:
            np.random.seed(seed)
            chain = HeisenbergChain(N, qubit, dt=dt)
            chain.evolve(steps, store_reduced= True)
            z_history = chain.get_sz()
            
            os.makedirs(os.path.dirname(histories_path), exist_ok=True)
            with open(histories_path, 'wb') as f:
                pickle.dump(z_history, f)
                
        #dt_loop()
        # Load or fallback best params
        
        try:
            param_name = f"Seed31415_{qubit_tag}_dt{fmt_dt_val}_dpth300_wsht{washout}"
            best_params_path = f"{perm_dir}bestparams_{param_name}.json"
            
            with open(best_params_path, 'r') as f:
                all_best = json.load(f)
            best = all_best.get(str(round(dt,5)), {})
            if best != {}:
                print("Found best parameters")
            print(best)
            
        except (FileNotFoundError, json.JSONDecodeError):
            best = {}
        
        predictor = ESNPredictor(
            steps=steps,
            dt=dt,
            N=N,
            qubit=qubit,
            history_values=z_history,
            reservoir_size=best.get('reservoir_size', 900),
            spectral_radius=best.get('spectral_radius', 1.25033),
            input_scaling=best.get('input_scaling', 0.546107),
            ridge_param=best.get('ridge_param', 0.170278),
            leak_rate=best.get('leak_rate', 0.946),
            sparsity=best.get('sparsity', 0.2),
            feedback=best.get('feedback', 1),
            # bias_scaling=best.get("bias_scaling", 0.45),
            washout=washout,
            batch_size=training_depth,
            training_depth=training_depth,
            model_path= model_path,
            seed=seed,
            device=device,
        )
        
        
        # optional tuning
        if n_trials > 0:
            Heisen_tune(predictor, study_name= study_name, study_loc= perm_dir, washout=washout, seed=seed, n_trials=n_trials, plots=False)
        elif n_trials == 0:
            Heisen_tune(predictor, study_name= study_name, study_loc= perm_dir, washout=washout, seed=seed, n_trials=0, plots=True)
        else:
            if not os.path.exists(model_path):
                print("Training ESN...")
                predictor.train()
                torch.save(predictor.esn, model_path)

            predictor.debug()
            print("Plotting..")
            predictor.predict_and_plot(acc_history=acc_chain.get_sz(), acc_chain=acc_chain, name=name)
    
    plot_hyperparams_vs_N("./examples/Heisenberg_Chain/trained_esns/")
    plt.show()
    
        
