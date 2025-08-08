import os
import json
import pickle
import numpy as np
import torch
from echostate import ESN  # <-- our new ESN module
from echostate.utils import mean_absolute_error
from .Heisenberg_sim import HeisenbergChain
import matplotlib.pyplot as plt

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')

# print(f"Using device: {device}")
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# # Extra check
# if device.type == "cuda":
#     print(f"GPU name: {torch.cuda.get_device_name(0)}")
#     print(f"Memory usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")



class ESNPredictor:
    """
    Train and evaluate an ESN on single-qubit ⟨σ_z⟩ histories
    produced by a HeisenbergChain simulation.

    Now accepts two separate seeds:
      * history_seed    -> for generating/loading training histories
      * reservoir_seed  -> for initializing reservoir random weights
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
                 history_seed: int = None,
                 reservoir_seed: int = None,
                 model_path=None,
                 cache_dir="./examples/Heisenberg_Chain/cache/",
                 device=torch.device('cpu')):
        # Core simulation settings
        self.steps = steps
        self.dt = dt
        self.N = N
        self.qubit = qubit
        self.washout = washout
        self.batch_size = batch_size
        self.training_depth = training_depth

        # Separate seeds for data vs. reservoir initialization
        self.history_seed   = history_seed if history_seed is not None else reservoir_seed
        self.reservoir_seed = reservoir_seed if reservoir_seed is not None else history_seed

        # Initialize or load ESN (use reservoir_seed for reproducibility)
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
                bias_scaling=bias_scaling,
                washout=washout,
                batch_size=batch_size,
                seed=self.reservoir_seed,
            ).to(device)
        else:
            self.esn = torch.load(model_path, weights_only=False)
            self.esn.to(device).eval()

        # Prepare test history if provided
        self.test_history = history_values

        # Load or generate training histories
        self.histories = []
        fmt_dt_val = str(round(self.dt, 5)).replace(".", "_", 1)
        qubit_tag = f"Qbts(dpth{training_depth}){self.N}"
        cache_name = (
            f"Historydata_Seed{self.history_seed}_T{steps*dt}_{qubit_tag}_dt{fmt_dt_val}.pkl"
        )
        cache_path = os.path.join(cache_dir, cache_name)

        # Always seed the RNG for reproducible history generation
        np.random.seed(self.history_seed)

        if history_values is None or training_depth > 0:
            if cache_path and os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.histories = pickle.load(f)
                print(f"Loaded {len(self.histories)} training histories from cache.")
            else:
                for _ in range(training_depth):
                    chain = HeisenbergChain(
                        num_qubits=N,
                        target_qubit=qubit,
                        dt=dt
                    )
                    chain.evolve(steps)
                    self.histories.append(chain.get_sz())
                print(f"Collected {len(self.histories)} simulation histories.")

                # Save to cache for next time
                os.makedirs(cache_dir, exist_ok=True)
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
        plt.plot(coarse_t, preds,"o", label='Predicted ⟨σ_z⟩', markersize = "4")
        plt.plot(true_t, true, "-o",   label='True ⟨σ_z⟩', markersize = "1")

        plt.xlim(50, 70)
        plt.xlabel("Time")
        plt.ylabel("⟨σ_z⟩")
        plt.title(f'ESN Prediction of Single‐Qubit({self.qubit+1}){self.N} at T:{T} and dt:{self.dt} Dynamics')
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




#region Tune/HyperPara
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
        try:
            y = [param_data[N][param] for N in sorted_N]
            axes[i].plot(sorted_N, y, marker='o')
            axes[i].set_ylabel(param)
            axes[i].grid(True)
        except:
            continue

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
        
        output_path = f'./examples/Heisenberg_Chain/trained_esns/bestparams_{param_name}.json'
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
    
    output_path = f'./examples/Heisenberg_Chain/trained_esns/bestparams_{param_name}.json'
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)
        print("Saved Best parameters")
    return study
  
#region CONTROL  
if __name__ == '__main__':
    # ─── Configuration ──────────────────────────────────────────────────────
    T              = 100
    N_list         = [5]
    train_seed     = 319
    test_seed      = 31
    qubit_list     = [0,1]       # list of qubit indices
    washout        = 3
    dt             = 0.2
    acc_dt         = 0.05
    training_depth = 1000 # Number of time series used to train 1 ESN
    testing_depth  = 1 # Number of ESNs trained

    # Modes: set exactly one of these to True
    do_tune        = False  # run Optuna tuning
    do_plot_hyper  = False  # just plot hyper‐vs‐N
    official_run   = True   # run ensemble of ESNs & shaded plot
    single_run     = not (do_tune or do_plot_hyper or official_run)

    ignore_qubit = True
    ignore_washout = True
    # optuna params (only used if do_tune)
    n_trials   = 50
    study_name = f"esn_tune_T{T}_dt{dt}"
    study_dir  = "./examples/Heisenberg_Chain/studies/"

    # ─── Preload high-res reference & (for single/official) test history ───
    from qutip import Qobj, sigmaz, expect
    np.random.seed(test_seed)
    acc_chain = HeisenbergChain(num_qubits=N_list[0],
                            target_qubit=qubit_list[0],
                            dt=acc_dt)
    acc_chain.evolve(int(T / acc_dt), store_reduced=True)

    # convert reduced density matrices → scalar ⟨σ_z⟩ values
    raw = acc_chain.get_sz()
    z_test = np.asarray([
        float(expect(sigmaz(), Qobj(rho, dims=[[2], [2]])))
        for rho in raw
    ])

    # ─── Dispatch based on mode ─────────────────────────────────────────────
    if do_plot_hyper:
        # very lightweight: just call your existing plot_hyperparams_vs_N
        plot_hyperparams_vs_N("./examples/Heisenberg_Chain/trained_esns/")
#region tune
    elif do_tune:
        # build a dummy predictor just to feed tuning
        predictor = ESNPredictor(
            steps=int(T/dt), dt=dt,
            N=N_list[0], qubit=qubit_list[0],
            history_values=None,
            reservoir_size=None, spectral_radius=None,
            input_scaling=None, ridge_param=None,
            leak_rate=None, sparsity=None,
            feedback=None, washout=washout,
            batch_size=training_depth,
            training_depth=training_depth,
            history_seed=train_seed,
            reservoir_seed=train_seed,
        )
        Heisen_tune(predictor,
                    study_name=study_name,
                    study_loc=study_dir,
                    washout=washout,
                    seed=train_seed,
                    n_trials=n_trials,
                    plots=True)
#region official run
    elif official_run:
        import torch
        import matplotlib.pyplot as plt
        import pandas as pd
        from qutip import Qobj, sigmaz, expect
        import json

        model_dir = "./examples/Heisenberg_Chain/cache/"
        param_dir = "./examples/Heisenberg_Chain/trained_esns/"
        os.makedirs(model_dir, exist_ok=True)

        mae_records = []
        config_records = []
        n_rows = len(N_list)
        n_cols = len(qubit_list)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for row_idx, N in enumerate(N_list):
            for col_idx, qubit in enumerate(qubit_list):
                print(f"\nSolving N={N}, qubit={qubit}")

                steps = int(T / dt)
                fmt_dt_val = str(round(dt, 5)).replace(".", "_", 1)
                qubit_tag = f"Qbts({(0 if ignore_qubit else qubit) + 1}){N}"
                if ignore_washout:
                    extra_wash  = 75
                else:
                    extra_wash = washout
                param_name = f"Seed31415_{qubit_tag}_dt{fmt_dt_val}_dpth50_wsht{extra_wash}"
                best_param_file = os.path.join(param_dir, f"bestparams_{param_name}.json")

                # Load best hyperparameters
                try:
                    with open(best_param_file, 'r') as f:
                        all_best = json.load(f)
                    best = all_best.get(str(round(dt, 5)), {})
                    print("Loaded best parameters:" if best else "No best parameters found for current dt.")
                except (FileNotFoundError, json.JSONDecodeError):
                    best = {}
                    print("No best parameter file found. Using defaults.")

                # High-res reference test history (new initial condition)
                np.random.seed(train_seed)
                acc_chain = HeisenbergChain(num_qubits=N, target_qubit=qubit, dt=acc_dt)
                acc_chain.evolve(int(T / acc_dt), store_reduced=True)
                raw = acc_chain.get_sz()
                z_test = np.asarray([
                    float(expect(sigmaz(), Qobj(rho, dims=[[2], [2]])))
                    for rho in raw
                ])
                print(z_test[0])

                seeds = [train_seed + i for i in range(testing_depth)]
                all_preds = []
                base_name = f"N{N}_{qubit_tag}_dt{fmt_dt_val}"

                for i, rseed in enumerate(seeds):
                    model_path = os.path.join(model_dir, f"trainedmodel_{base_name}_Ex{i}.pt")

                    predictor = ESNPredictor(
                        steps=steps,
                        dt=dt,
                        N=N,
                        qubit=qubit if not ignore_qubit else 0,
                        history_values=None,
                        reservoir_size=best.get("reservoir_size", 900),
                        spectral_radius=best.get("spectral_radius", 1.25),
                        input_scaling=best.get("input_scaling", 0.55),
                        ridge_param=best.get("ridge_param", 1e-1),
                        leak_rate=best.get("leak_rate", 0.9),
                        sparsity=best.get("sparsity", 0.2),
                        feedback=best.get("feedback", 1),
                        bias_scaling=best.get("bias_scaling", 0.4),
                        washout=washout,
                        batch_size=training_depth,
                        training_depth=training_depth,
                        history_seed=train_seed,
                        reservoir_seed=rseed,
                        model_path=model_path if os.path.exists(model_path) else None,
                        device=device
                    )

                    if os.path.exists(model_path):
                        print(f"Loading existing ESN Ex{i} (seed={rseed})")
                        predictor.esn = torch.load(model_path)
                    else:
                        print(f"Training ESN Ex{i} (seed={rseed})")
                        predictor.train()
                        torch.save(predictor.esn, model_path)

                    # Save hyperparameters and config
                    config_records.append({
                        "N": N, "qubit": qubit, "Ex": i, "seed": rseed,
                    "reservoir_size": best.get("reservoir_size", 900),
                    "spectral_radius": best.get("spectral_radius", 1.25),
                    "input_scaling": best.get("input_scaling", 0.55),
                    "ridge_param": best.get("ridge_param", 1e-1),
                    "leak_rate": best.get("leak_rate", 0.9),
                    "sparsity": best.get("sparsity", 0.2),
                    "feedback": best.get("feedback", 1),
                    "bias_scaling": best.get("bias_scaling", 0.4)
                    })
    

                    # Predict
                    X_test = torch.tensor(z_test[:-1], dtype=torch.float32, device=device)
                    X_test = X_test.unsqueeze(-1).unsqueeze(0)
                    pred = predictor.esn.predict(X_test)[0].cpu().numpy().flatten()
                    all_preds.append(pred)

                    # MAE calculation
                    true = z_test[washout + 1: washout + 1 + len(pred)]
                    mae = mean_absolute_error(torch.tensor(pred), torch.tensor(true)).item()
                    mae_records.append({"N": N, "qubit": qubit, "Ex": i, "seed": rseed, "MAE": mae})

                # Overlay all ESNs with fill_between
                ax = axs[row_idx][col_idx]
                all_preds = np.stack(all_preds)
                times = np.arange(all_preds.shape[1]) * dt
                mean_pred = all_preds.mean(axis=0)
                std_pred = all_preds.std(axis=0)
                true = z_test[washout+1 : washout+1+len(pred)]
                true_t = np.arange(len(true)) * dt

                ax.fill_between(times, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, label="±1σ")
                ax.plot(times, mean_pred, 'o-', markersize=2, label="Mean Prediction")
                
                for i, ipreds in enumerate(all_preds):
                    ax.plot(times,ipreds, label = f"ESNS {i}")
                ax.plot(true_t, true, '-', linewidth=1.2, label='True ⟨σ_z⟩')
                ax.set_xlim(0,15)
                ax.set_title(f"N={N}, Qubit={qubit}")
                ax.set_xlabel("Time")
                ax.set_ylabel("⟨σ_z⟩")
                ax.legend()

                fig_path = os.path.join(model_dir, f"official_overlay_{base_name}.pdf")
                fig.savefig(fig_path)
                print(f"Saved overlay plot to {fig_path}")

        # Save MAE results and configs to files
        mae_df = pd.DataFrame(mae_records)
        mae_df.to_csv(os.path.join(model_dir, "official_run_mae_summary.csv"), index=False)
        print("Saved MAE summary to official_run_mae_summary.csv")

        with open(os.path.join(model_dir, "official_run_esn_configs.json"), 'w') as f:
            json.dump(config_records, f, indent=4)
            print("Saved ESN config log to official_run_esn_configs.json")




    elif single_run:
        # exactly your old single‐ESN workflow
        for N in N_list:
            for qubit in qubit_list:
                # load or simulate low-res history for this (train vs test seed)
                try:
                    with open(f"./cache/Historydata_Seed{test_seed}_T{T}_Qbts({qubit+1}){N}_dt{dt:.5f}.pkl", 'rb') as f:
                        z_hist = pickle.load(f)
                except FileNotFoundError:
                    chain = HeisenbergChain(N, qubit, dt=dt)
                    chain.evolve(int(T/dt), store_reduced=True)
                    z_hist = chain.get_sz()

                predictor = ESNPredictor(
                    steps=int(T/dt),
                    dt=dt,
                    N=N,
                    qubit=qubit,
                    history_values=z_hist,
                    reservoir_size=900,
                    spectral_radius=1.25,
                    input_scaling=0.55,
                    ridge_param=1e-1,
                    leak_rate=0.9,
                    sparsity=0.2,
                    feedback=1,
                    washout=washout,
                    batch_size=training_depth,
                    training_depth=training_depth,
                    history_seed=train_seed,
                    reservoir_seed=train_seed,
                    model_path=f"./cache/esn_model_{train_seed}_{N}_{qubit}.pt",
                )

                if n_trials > 0:
                    Heisen_tune(predictor,
                                study_name=study_name,
                                study_loc=study_dir,
                                washout=washout,
                                seed=train_seed,
                                n_trials=n_trials,
                                plots=False)
                elif n_trials == 0:
                    Heisen_tune(predictor,
                                study_name=study_name,
                                study_loc=study_dir,
                                washout=washout,
                                seed=train_seed,
                                n_trials=0,
                                plots=True)
                else:
                    if not os.path.exists(predictor.model_path):
                        predictor.train()
                        torch.save(predictor.esn, predictor.model_path)
                    predictor.debug()
                    predictor.predict_and_plot(
                        acc_history=acc_chain.get_sz(),
                        acc_chain=acc_chain,
                        name=f"Seed{test_seed}_N{N}_Q{qubit}")
    
    plt.show()