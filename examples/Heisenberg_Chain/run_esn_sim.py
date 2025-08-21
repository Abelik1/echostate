import os
import json
import pickle
import numpy as np
import torch
from echostate import ESN  # <-- our new ESN module
from echostate.utils import mean_absolute_error
from .Heisenberg_sim import HeisenbergChain
from .heisen_utils import *
import matplotlib.pyplot as plt
from qutip import Qobj, sigmaz, expect
import pandas as pd
import time

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print(f"Using device: {device}")
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# # Extra check
# if device.type == "cuda":
#     print(f"GPU name: {torch.cuda.get_device_name(0)}")
#     print(f"Memory usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

from torch.serialization import add_safe_globals
from echostate import ESN  # your ESN class


# ---------- Small, focused utilities (behavior-preserving) ----------
def _safe_get_config(model):
    """Return a config dict even for legacy pickled ESNs that lacked get_config/seed."""
    # Preferred: modern API
    if hasattr(model, "get_config"):
        try:
            cfg = model.get_config()
            # ensure plain str for device
            if "device" in cfg and not isinstance(cfg["device"], str):
                cfg["device"] = str(cfg["device"])
            return cfg
        except Exception:
            pass
        
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def format_dt_val(dt: float) -> str:
    # Matches your exact formatting everywhere
    return str(dt).rstrip("0").rstrip(".").replace(".", "_", 1)

def qubit_tag_str(ignore_qubit: bool, qubit: int, N: int) -> str:
    # Matches f"Qbts({(0 if ignore_qubit else qubit) + 1}){N}"
    return f"Qbts({(qubit_focus if ignore_qubit else qubit) + 1}){N}"

def base_name_str(learning_algo: str, train_batch_size: int, qubit_tag: str, fmt_dt: str) -> str:
    # Matches f"{learning_algo}_batchSize{train_batch_size}_{qubit_tag}_dt{fmt_dt}"
    return f"{learning_algo}_batchSize{train_batch_size}_{qubit_tag}_dt{fmt_dt}"

def param_name_for_tune(learning_algo: str, qubit_tag: str, fmt_dt: str, extra_wash: int) -> str:
    # Matches tune's: f"{learning_algo}_Seed3141_{qubit_tag}_dt{fmt_dt}_dpth50_wsht{extra_wash}"
    return f"{learning_algo}_Seed3141_{qubit_tag}_dt{fmt_dt}_dpth50_wsht{extra_wash}"

def param_name_for_best(learning_algo: str, N: int, fmt_dt: str, extra_wash: int) -> str:
    # Matches official/predictions: f"{learning_algo}_Seed3141_Qbts({1}){N}_dt{fmt_dt}_dpth50_wsht{extra_wash}"
    return f"{learning_algo}_Seed3141_Qbts({1}){N}_dt{fmt_dt}_dpth50_wsht{extra_wash}"

def best_params_file(param_dir: str, param_name: str) -> str:
    return os.path.join(param_dir, f"bestparams_{param_name}.json")

def load_best_params(param_dir: str, param_name: str, dt: float) -> dict:
    """Load best params dict for given dt; return {} if not found/invalid — exact behavior preserved."""
    try:
        with open(best_params_file(param_dir, param_name), 'r') as f:
            all_best = json.load(f)
        return all_best.get(str(round(dt, 5)), {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def generate_high_res_test(N: int, qubit: int, acc_dt: float, T: float, seed: int, measure: str = "sz"):
    np.random.seed(seed)
    acc_chain = HeisenbergChain(num_qubits=N, target_qubit=qubit, dt=acc_dt, measure=measure)
    # i0 = 21
    # acc_chain.psi[i0] = np.conj(acc_chain.psi[i0])
    acc_chain.evolve(int(T / acc_dt))
    return np.asarray(acc_chain.get_observable()), acc_chain

def z_eval_from_z_test(z_test: np.ndarray, dt: float, acc_dt: float) -> np.ndarray:
    """Downsample z_test to dt (raises if dt not integer multiple of acc_dt), behavior preserved."""
    if not np.isclose(dt, acc_dt):
        ratio = dt / acc_dt
        k = int(round(ratio))
        if not np.isclose(k * acc_dt, dt):
            raise ValueError(f"dt/acc_dt not integer: dt={dt}, acc_dt={acc_dt}")
    else:
        k = 1
    return z_test[::k]

def select_series_for_qubit(z_arr: np.ndarray, q_idx: int, N: int) -> np.ndarray:
    """Mirror original selection logic for 1D series out of (T,N)/(N,T)/1D shapes."""
    if z_arr.ndim == 2:
        if z_arr.shape[1] == N:
            return z_arr[:, q_idx]
        elif z_arr.shape[0] == N:
            return z_arr[q_idx, :]
        else:
            return z_arr.squeeze()
    return z_arr.squeeze()

def extract_rseed_from_filename(path: str):
    """Parse ..._rSeedXXXX_... integer if present; else None (behavior preserved)."""
    parts = os.path.basename(path).split("_")
    for part in parts:
        if part.startswith("rSeed"):
            try:
                return int(part.replace("rSeed", ""))
            except ValueError:
                return None
    return None

def save_esn(model, path, extra=None):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if getattr(model, "W_out", None) is not None and "W_out" not in sd:
        sd["W_out"] = model.W_out.detach().cpu()
    payload = {"state_dict": sd, "config": _safe_get_config(model), "extra": extra or {}}
    torch.save(payload, path)  # ← remove pickle_protocol=5
    # (PyTorch defaults to protocol 2, which the safe loader understands)
    
def _install_W_out(esn, W):
    if isinstance(getattr(esn, "W_out", None), torch.Tensor) and esn.W_out.numel() > 0:
        # if shape mismatch, re-register
        if esn.W_out.shape != W.shape:
            try:
                delattr(esn, "W_out")
            except Exception:
                pass
            try:
                esn.register_buffer("W_out", W)
            except Exception:
                esn.W_out = W
        else:
            esn.W_out.data.copy_(W)
    else:
        # no usable buffer yet → register
        try:
            if hasattr(esn, "W_out"):
                delattr(esn, "W_out")
            esn.register_buffer("W_out", W)
        except Exception:
            esn.W_out = W

def load_esn(path, device="cpu", *, migrate_old=True):
    obj = torch.load(path, map_location="cpu")  # no weights_only
    if not (isinstance(obj, dict) and "state_dict" in obj and "config" in obj):
        raise ValueError(f"Unsupported ESN file format at {path}")

    cfg = obj["config"]
    sd  = obj["state_dict"]

    esn = ESN(**cfg)

    # First try to load everything (now that W_out placeholder exists, strict can be True)
    missing, unexpected = esn.load_state_dict(sd, strict=False)  # keep False to be lenient across versions

    # If W_out didn't get loaded (older payloads or name mismatch), install manually
    if ("W_out" in sd) and (not isinstance(esn.W_out, torch.Tensor) or esn.W_out.numel() == 0):
        W = sd["W_out"]
        # Ensure shape is (Dout, R+1)
        expected = (esn.output_dim, esn.reservoir_size + 1)
        if W.shape == expected:
            _install_W_out(esn, W)
        elif W.T.shape == expected:
            _install_W_out(esn, W.T)
        else:
            raise RuntimeError(f"W_out shape {tuple(W.shape)} incompatible with expected {expected}")

    # Final safety net: assert W_out is a non-empty tensor
    if not (isinstance(esn.W_out, torch.Tensor) and esn.W_out.numel() > 0):
        raise RuntimeError("Loaded ESN has no trained W_out. Was the model saved after fit()?")

    return esn.to(device).eval()

from safetensors.torch import load_file
def load_esn_st(path, config_json, device="cpu"):
    state = load_file(path)  # dict[str, Tensor] on CPU
    esn = ESN(**config_json)
    esn.load_state_dict(state, strict=True)
    return esn.to(device).eval()
#region ESNPredictor
class ESNPredictor:
    def __init__(self,
                 steps: int,
                 dt: float,
                 N: int,
                 qubit: int,
                 history_values: list = None,
                 washout: int = 0,
                 batch_size: int = 1,
                 train_batch_size: int = 1,
                 history_seed: int = None,
                 reservoir_seed: int = None,
                 cache_dir: str = "./examples/Heisenberg_Chain/cache/",
                 learning_algo = "inv",
                 device: torch.device = torch.device('cpu'),
                 train_op: str = "sz",
                 pred_op: str = "sz"):

        # Core settings
        self.steps = steps
        self.dt = dt
        self.N = N
        self.qubit = qubit
        self.washout = washout
        self.batch_size = batch_size
        self.train_batch_size = train_batch_size
        self.device = device
        self.learning_algo = learning_algo
        self.train_op = train_op
        self.pred_op  = pred_op
        # Seeds
        self.history_seed   = history_seed if history_seed is not None else reservoir_seed
        self.reservoir_seed = reservoir_seed if reservoir_seed is not None else history_seed

        # Provided test history (optional)
        self.test_history = history_values

        # Cache naming
        self.cache_dir = cache_dir
        fmt_dt_val = str(round(self.dt, 5)).replace(".", "_", 1)
        qubit_tag = f"Qbts(dpth{train_batch_size}){self.N}"
        self.cache_name = f"Historydata_Seed{self.history_seed}_T{steps*dt}_{qubit_tag}_op({self.train_op})_dt{fmt_dt_val}.pkl"
        self.cache_path = os.path.join(cache_dir, self.cache_name)

        # Histories are built on demand
        self.histories = None

    def prepare_histories(self):
        """
        Load or generate TRAINING histories (operator = self.train_op) for the training qubit.
        Returns list of sequences (each sequence is 1D np.array).
        """
        if self.histories is not None:
            return self.histories

        np.random.seed(self.history_seed)
        self.histories = []

        if self.test_history is None or self.train_batch_size > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    self.histories = pickle.load(f)
                print(f"Loaded {len(self.histories)} training histories from cache: {self.cache_path}")
            else:
                for _ in range(self.train_batch_size):
                    chain = HeisenbergChain(num_qubits=self.N,
                                            target_qubit=self.qubit,
                                            dt=self.dt,
                                            measure=self.train_op)
                    # keep your “conjugate one amplitude” tweak
                    # i0 = 21 #TODO REMOVE LATER TEST
                    # chain.psi[i0] = np.conj(chain.psi[i0])
                    chain.evolve(self.steps)
                    self.histories.append(chain.get_observable())
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.histories, f)
                print(f"Saved training histories to {self.cache_path}.")
        return self.histories

    def build_dataset(self):
        """
        Convert histories into teacher-forced (inputs, targets).
        Returns lists of Tensors of shape (T, 1).
        """
        self.prepare_histories()
        inputs, targets = [], []
        for z_seq in self.histories:
            arr = np.asarray(z_seq)
            X = torch.tensor(arr[:-1], dtype=torch.float32, device=self.device).unsqueeze(-1)
            Y = torch.tensor(arr[1:], dtype=torch.float32, device=self.device).unsqueeze(-1)
            inputs.append(X)
            targets.append(Y)
        assert len(inputs) == self.batch_size, \
            f"Expected batch_size={self.batch_size}, got {len(inputs)} sequences"
        return inputs, targets

    # ---------- ESN factory & helpers ----------
    def make_esn(self, *,
                 reservoir_size=900, spectral_radius=1.0, input_scaling=1.0,
                 ridge_param=1e-3, leak_rate=0.9, sparsity=0.2,
                 feedback=0, bias_scaling=0.4, seed=None):
        """
        Lazily construct an ESN instance with given hyperparams.
        """
        from echostate import ESN
        esn = ESN(
            device=self.device,
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
            washout=self.washout,
            batch_size=self.batch_size,
            learning_algo=self.learning_algo,
            seed=(self.reservoir_seed if seed is None else seed),
            step_log_every=ESN_STEP_LOG_EVERY,
        ).to(self.device)
        return esn

    def train_esn(self, esn):
        """Fit a provided ESN on the prepared dataset."""
        inputs, targets = self.build_dataset()
        print(f"Training ESN on {len(inputs)} sequences (washout={self.washout})")
        esn.fit(inputs, targets, profile=False)

    def predict_sequence(self, esn, z_test):
        """
        Predict next-step σ_z on a given test scalar sequence z_test.
        Returns numpy array predictions and aligned true sequence (after washout).
        """
        X_test = torch.tensor(z_test[:-1], dtype=torch.float32, device=self.device).unsqueeze(-1).unsqueeze(0)
        preds = esn.predict(X_test)[0].detach().cpu().numpy().flatten()
        true = z_test[self.washout+1: self.washout+1 + len(preds)]
        return preds, true

def _truth_cache_name(base_dir, seed, N, qubit_list, op, dt):
    qmin, qmax = min(qubit_list), max(qubit_list)
    fmt_dt = str(round(dt, 5)).replace(".", "_", 1)
    fname = f"TrueHist_Seed{seed}_N{N}_qubits({qmin}-{qmax})_op({op})_dt{fmt_dt}.pkl"
    return os.path.join(base_dir, fname)

def cache_prediction_truth(N, qubit_list, dt, T, seed, op, base_dir):
    """
    Generate or load true operator time series for all qubits in qubit_list.
    Returns: dict {qubit_index: np.array length steps+1}
    """
    path = _truth_cache_name(base_dir, seed, N, qubit_list, op, dt)
    steps = int(T / dt)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        # quick sanity checks
        if payload.get("N") == N and payload.get("dt") == dt and payload.get("op") == op:
            print(f"[truth-cache] Loaded {path}")
            return payload["series"]

    # build (independently per qubit to keep the class simple)
    print(f"[truth-cache] Generating true series for N={N}, qubits={qubit_list}, op={op}, dt={dt}")
    
    series = {}
    for q in qubit_list:
        np.random.seed(seed)
        chain = HeisenbergChain(num_qubits=N, target_qubit=q, dt=dt, measure=op)
        # keep your i0 tweak consistent for prediction seeds too
        # i0 = 21
        # chain.psi[i0] = np.conj(chain.psi[i0])
        chain.evolve(steps)
        series[q] = chain.get_observable()

    payload = {"N": N, "dt": dt, "T": T, "op": op, "seed": seed, "qubits": list(qubit_list), "series": series}
    os.makedirs(base_dir, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[truth-cache] Saved → {path}")
    return series




# -------------Example CASE------------------------------------------
#region Hyperparams vs N
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

#region Tune/HyperPara
def Heisen_tune(predictor, study_name, study_loc, washout, seed, n_trials, param_name, dt=None, plots=False, learning_algo="inv"):
    import optuna
    from optuna.visualization import (
        plot_optimization_history, plot_param_importances,
        plot_parallel_coordinate, plot_slice, plot_contour, plot_edf
    )
    from echostate import ESN

    if dt is None:
        dt = predictor.dt

    best_params_dict = {}
    input_list, target_list = predictor.build_dataset()
    try:
        study = ESN.tune(
            input_list, target_list,
            n_trials=n_trials, direction="minimize",
            study_name=study_name, study_loc=study_loc,
            washout=washout, seed=seed,
            reservoir_limit= [500,1000],
            spectral_radius_limit=[0.6,1.7],
            feedback_limit=0,
            input_scaling_limit=[0.1, 0.3],
            ridge_param_limit=[1e-11, 1e-3],
            leak_rate_limit=[ 0.2, 1.0],
            sparsity_limit= [0.01,0.1],
            bias_scaling_limit= 0.0,
            device=predictor.device,
            learning_algo= learning_algo
        )
    except KeyboardInterrupt:
        print("Interrupted! Loading best trial so far...")
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{study_loc}/{study_name}.db"
        )
        best_params_dict[str(round(dt, 5))] = study.best_params
        output_path = f'./examples/Heisenberg_Chain/trained_esns/bestparams_{param_name}.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(best_params_dict, f, indent=4)
            print("Saved Best parameters")
        return study

    if plots:
        plot_optimization_history(study).show()
        plot_param_importances(study).show()
        plot_parallel_coordinate(study).show()
        plot_slice(study).show()
        plot_contour(study).show()
        # plot_edf(study).show()

    print(dt)
    print("Best hyperparameters:", study.best_params)
    print("Best MAE:", study.best_value)

    best_params_dict[str(round(dt, 5))] = study.best_params
    output_path = f'./examples/Heisenberg_Chain/trained_esns/bestparams_{param_name}.json'
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_params_dict, f, indent=4)
        print("Saved Best parameters")

    return study


# Example usage:
# scorecard_physics("physics_summary.json")


#region CONTROL
if __name__ == '__main__':
    # ─── Configuration ──────────────────────────────────────────────────────
    T              = 100
    N_list         = [5]
    train_seed     = 3141
    reservoir_seed  = 314
    pred_seed     = 314
    qubit_list     = [0,1,2,3,4]       # list of qubit indices
    qubit_focus = 0
    washout        = 120
    dt             = 0.2
    acc_dt         = 0.05
    train_batch_size = 1000 # Number of time series used to train 1 ESN
    test_esns  = 1 # Number of ESNs trained
    
    train_op = "sz"   # operator for training histories (focus qubit)
    predict_op = "sz" # operator for truth used in prediction/plots

    # Modes: set exactly one of these to True
    do_tune        = False  # run Optuna tuning
    do_plot_hyper  = False  # just plot hyper‐vs‐N
    do_predictions = False
    official_run   = True   # run ensemble of ESNs & shaded plot

    learning_algo = "pinv" # inv / cholesky / solve / eigh / cg / svd / tsvd / qr / pinv

    ignore_qubit = True
    ignore_washout = True # Applies only to hyperparameters so far
    # optuna params (only used if do_tune)
    n_trials   = 500
    num_pred   = 20 # Number of final test series
    # Plot filtering (set to None to disable filtering)
    PLOT_MAE_TOL = 0.03   # e.g., only plot ESNs with MAE <= 0.002
    USE_TOL_FOR_STATS = True # if True, mean/std shading uses only filtered ESNs
    SKIP_PREDICTIONS_IF_HIST = True 
    
    # ---------- Names & Paths (centralized control block) ----------
    # You can change folder names/roots here (loop-dependent names still built later)
    LOG_DIR   = "./logs"
    MODEL_DIR = "./examples/Heisenberg_Chain/cache/"
    PARAM_DIR = "./examples/Heisenberg_Chain/trained_esns/"
    CACHE_DIR = "./examples/Heisenberg_Chain/cache/"  # used by ESNPredictor default

    #Logging Settings
    import logging, os
    from echostate.logging_config import setup_logging

    VERBOSITY = "CRITICAL"          # one of: "INFO", "DEBUG"  (avoid TRACE unless deep dive), "CRITICAL" if none
    STEP_LOG_EVERY = 50           # log ESN step stats every N steps
    SILENCE_3P = True             # silence matplotlib/PIL/optuna/etc. at WARNING


    fmt_dt_val = format_dt_val(dt)
    run_name = (
        f"official"
        f"_N{','.join(map(str, N_list))}"
        f"_Q{','.join(map(str, qubit_list))}"
        f"_dt{fmt_dt_val}"
        f"_depth{train_batch_size}"
        f"_seed{train_seed}"
    )

    paths = setup_logging(
        log_dir=LOG_DIR,
        run_name=run_name,
        console_level="CRITICAL",      # console stays readable # Critical
        file_level=VERBOSITY,      # file detail = DEBUG (good default)
        jsonl_file=True,
        plain_file=True,
    )

    # 1) Silence noisy third-party loggers
    if SILENCE_3P:
        for name in (
            "matplotlib", "PIL", "PIL.Image", "PIL.PngImagePlugin",
            "numba", "qutip", "optuna", "asyncio", "urllib3", "parso",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    # 2) Make reservoir quiet; keep ESN/trainer informative
    logging.getLogger("echostate.reservoir").setLevel(logging.WARNING)   # hide “Reservoir step”
    logging.getLogger("echostate.ESN").setLevel(logging.DEBUG)           # shapes, diagnostics, sampled steps
    logging.getLogger("echostate.trainer").setLevel(logging.DEBUG)       # covariance stats, solves

    # 3) Pass step sampling down to ESN (see small ESN change below)
    ESN_STEP_LOG_EVERY = STEP_LOG_EVERY
    train_qubit = qubit_focus if ignore_qubit else qubit
    # ─── Preload high-res reference & (for single/official) test history ───
    from qutip import Qobj, sigmaz, expect

    # ---------- Mode runners (no behavior changes, just wrapped) ----------
    def run_plot_hyper_mode():
        plot_hyperparams_vs_N(PARAM_DIR)
#region Tune Run
    def run_tune_mode():
        for N in N_list:
            train_qubit = qubit_focus if ignore_qubit else qubit_list[0]
            predictor = ESNPredictor(
                steps=int(T/dt), dt=dt, N=N, qubit=train_qubit,
                history_values=None, washout=washout,
                batch_size=train_batch_size, train_batch_size=train_batch_size,
                history_seed=train_seed, reservoir_seed=reservoir_seed,
                device=device,
                train_op=train_op, pred_op=predict_op   # ← add
            )

            fmt_dt = format_dt_val(dt)
            qubit_tag = qubit_tag_str(ignore_qubit, train_qubit, N)
            extra_wash = 75 if ignore_washout else washout
            param_name = param_name_for_tune(learning_algo, qubit_tag, fmt_dt, extra_wash)
            study_name = f"esnStudy_{learning_algo}_Seed{train_seed}_{qubit_tag}_dt{fmt_dt}_dpth50_wsht75"
            print(study_name)
            study_dir  = PARAM_DIR
            print(param_name)

            Heisen_tune(
                predictor,
                study_name=study_name,
                study_loc=study_dir,
                washout=washout,
                seed=train_seed,
                n_trials=n_trials,
                param_name=param_name,
                dt=dt,
                plots=False,
                learning_algo=learning_algo,
            )
#region Prediction Run
    def run_predictions_mode():
        from glob import glob

        model_dir = ensure_dir(MODEL_DIR)
        param_dir = ensure_dir(PARAM_DIR)

        n_rows = len(N_list)
        n_cols = len(qubit_list)
        fig, axs = plt.subplots(n_rows+1, n_cols,
                                figsize=(5 * n_cols, 4 * (n_rows+1)), squeeze=False)

        all_scores = []  # accumulate across all (N, qubit)

        for row_idx, N in enumerate(N_list):
            truth_series = cache_prediction_truth(
                N=N, qubit_list=qubit_list, dt=dt, T=T, seed=pred_seed,
                op=predict_op, base_dir=CACHE_DIR
            )
            for col_idx, qubit in enumerate(qubit_list):
                print(f"\n[do_predictions] N={N}, qubit={qubit}")

                steps = int(T / dt)
                fmt_dt = format_dt_val(dt)
                qubit_tag = qubit_tag_str(ignore_qubit, qubit, N)
                extra_wash = 75 if ignore_washout else washout

                # === match official_run naming ===
                param_name = param_name_for_best(learning_algo, N, fmt_dt, extra_wash)
                best_param_file_path = best_params_file(param_dir, param_name)

                # base_name used in model/fig paths
                base_name = base_name_str(learning_algo, train_batch_size, qubit_tag, fmt_dt)
                # =================================
                
                # ---- Fast-path: use cached histogram CSV if present (avoid ESN load) ----
                hist_csv = os.path.join(model_dir, f"hist_logMAE_N{N}_q{qubit}_dataset0.csv")
                if os.path.exists(hist_csv):
                    print(f"[fast-path] Using existing histogram data → {hist_csv}")
                    hist_df = pd.read_csv(hist_csv)

                    # Prefer log10 if available; otherwise convert from natural log; otherwise compute from MAE.
                    if "log10_MAE" in hist_df.columns:
                        log_maes = hist_df["log10_MAE"].to_numpy(dtype=float)
                    elif "ln_MAE" in hist_df.columns:
                        log_maes = (hist_df["ln_MAE"].to_numpy(dtype=float)) / np.log(10.0)
                    elif "MAE" in hist_df.columns:
                        log_maes = np.log10(hist_df["MAE"].to_numpy(dtype=float) + 1e-18)
                    else:
                        log_maes = None
                        print("[fast-path] CSV missing MAE columns; will fall back to ESN recompute.")

                    if log_maes is not None:
                        # Draw only once per column (final N row), same as original logic
                        if row_idx == n_rows - 1:
                            ax_hist = axs[n_rows][col_idx]
                            ax_hist.hist(log_maes, bins='auto')
                            ax_hist.set_xlabel("log10(MAE)")
                            ax_hist.set_ylabel("Frequency")
                            ax_hist.set_title(f"MAE dist — op={predict_op}, N={N}, q={qubit}")
                            if PLOT_MAE_TOL is not None:
                                ax_hist.axvline(np.log10(PLOT_MAE_TOL + 1e-18), linestyle='--')

                        # If you only wanted the histogram, you can skip ESN work entirely:
                        if SKIP_PREDICTIONS_IF_HIST:
                            print("[fast-path] Histogram plotted from cache. Skipping ESN loading/predictions for this (N,q).")
                            # Continue to next (N, qubit)
                            continue
                # -------------------------------------------------------------------------

                # Load best params if available
                best = load_best_params(param_dir, param_name, dt)
                print("Loaded best parameters for do_predictions." if best else "No best parameter file found. Using defaults.")
                
                # Predictor for using the (pretrained) ESN to predict
                predictor_model = ESNPredictor(
                    steps=steps, dt=dt, N=N, qubit=qubit,
                    history_values=None, washout=washout,
                    batch_size=train_batch_size, train_batch_size=train_batch_size,
                    history_seed=train_seed, reservoir_seed=reservoir_seed,
                    learning_algo=learning_algo, device=device,
                    train_op=train_op, pred_op=predict_op  # ← add
                )
                predictor_pred = ESNPredictor(
                    steps=steps, dt=dt, N=N, qubit=qubit,
                    history_values=None, washout=washout,
                    batch_size=num_pred, train_batch_size=num_pred,
                    history_seed=pred_seed, reservoir_seed=pred_seed,
                    learning_algo=learning_algo, device=device,
                    train_op=predict_op, pred_op=predict_op  # ← key change
                )

                # Load the ESN named after the TRAIN seed (Ex0 by convention) -- for the per-dataset metrics
                model_path = os.path.join(
                    model_dir,
                    f"trainedmodel_Seed{train_seed}_rSeed{reservoir_seed}_{base_name}.pt"
                )

                if os.path.exists(model_path):
                    print(f"Loading ESN from {model_path}")
                    esn_single = load_esn(model_path, device=device)
                else:
                    raise FileNotFoundError(
                        f"Expected trained ESN not found:\n  {model_path}\n"
                        "Please run your training/official block first."
                    )
                # High-res reference (same approach as official_run, per qubit)
                z_test_hi, _acc_tmp = generate_high_res_test(N, qubit, acc_dt, T, pred_seed, measure=predict_op)
                z_eval_dt = truth_series[qubit]  # already at dt, correct operator
                pred_hi, true_hi = ESNPredictor(
                    steps=int(T/dt), dt=dt, N=N, qubit=qubit, washout=washout,
                    batch_size=train_batch_size, train_batch_size=train_batch_size,
                    history_seed=train_seed, reservoir_seed=reservoir_seed,
                    learning_algo=learning_algo, device=device,
                    train_op=train_op, pred_op=predict_op
                ).predict_sequence(esn_single, z_eval_dt)
                # Generate/load PREDICTION datasets CACHED exactly like training (using pred_seed)
                predictor_pred = ESNPredictor(
                    steps=steps,
                    dt=dt,
                    N=N,
                    qubit=qubit,                 # generate datasets for this actual qubit
                    history_values=None,
                    washout=washout,
                    batch_size=num_pred,         # number of sequences we want
                    train_batch_size=num_pred,     # drives how many histories are generated/cached
                    history_seed=pred_seed,      # NEW: prediction datasets seed
                    reservoir_seed=pred_seed,    # not used for caching; harmless
                    learning_algo=learning_algo, # <-- match official_run signature
                    device=device
                )
                pred_histories = predictor_pred.prepare_histories()  # list length == num_pred
                # Compute MAE for the plotted single ESN
                mae_hi = mean_absolute_error(torch.tensor(pred_hi), torch.tensor(true_hi[:len(pred_hi)])).item()
                plot_ok = (PLOT_MAE_TOL is None) or (mae_hi <= PLOT_MAE_TOL)

                if not plot_ok:
                    print(f"[plot-skip] do_predictions: N={N}, q={qubit} single-ESN MAE={mae_hi:.4g} > tol={PLOT_MAE_TOL}")

                # --- Plot using high-res truth (per qubit) only if passes threshold ---
                if plot_ok:
                    ax = axs[row_idx][col_idx]
                    t_dt   = np.arange(len(pred_hi)) * dt
                    t0     = (washout + 1) * dt
                    start_hi = int(round(t0 / acc_dt))
                    t_fine_rel = np.arange(len(z_test_hi)) * acc_dt - t0

                    ax.plot(t_fine_rel[start_hi:], z_test_hi[start_hi:], '-', lw=1.2, color='green', label='True (acc_dt)')
                    ax.plot(t_dt, true_hi[:len(t_dt)], 'o', ms=1, color='darkgreen', alpha=0.8)
                    ax.plot(t_dt, pred_hi[:len(t_dt)], 'o-', ms=2, label="Prediction (single ESN)")

                    ax.set_xlim(65, 86)
                    ax.set_title(f"N={N}, Qubit={qubit}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel(f"⟨σ_{predict_op[-1]}⟩")
                    ax.legend()

                # ── Part A: original do_predictions behavior — per-dataset metrics with a single ESN ──
                ax = axs[row_idx][col_idx]
                first_plot_done = False
                records = []

                q_idx = qubit  # mirror official_run's qubit selection

                for d, z_seq in enumerate(pred_histories):
                    z_arr = np.asarray(z_seq)
                    z_eval = select_series_for_qubit(z_arr, q_idx, N)

                    pred_d, true_d = predictor_model.predict_sequence(esn_single, z_eval)
                    mae_d = mean_absolute_error(torch.tensor(pred_d), torch.tensor(true_d)).item()
                    rmse_d = float(np.sqrt(np.mean((pred_d - true_d)**2)))

                    records.append({
                        "N": N, "qubit": qubit,
                        "dataset_index": d,
                        "seed": pred_seed + d,
                        "MAE": mae_d,
                        "RMSE": rmse_d,
                        "T_effective": len(pred_d)
                    })

                    # Plot ONLY the first dataset
                    if not first_plot_done:
                        # --- Plot using high-res truth (per qubit), matching official_run style ---
                        ax = axs[row_idx][col_idx]
                        t_dt   = np.arange(len(pred_hi)) * dt
                        t0     = (washout + 1) * dt
                        start_hi = int(round(t0 / acc_dt))
                        t_fine_rel = np.arange(len(z_test_hi)) * acc_dt - t0

                        # High-res truth
                        ax.plot(t_fine_rel[start_hi:], z_test_hi[start_hi:], '-', lw=1.2, color='green', label='True (acc_dt)')
                        # Aligned dt truth points
                        ax.plot(t_dt, true_hi[:len(t_dt)], 'o', ms=1, color='darkgreen', alpha=0.8)
                        # Prediction
                        ax.plot(t_dt, pred_hi[:len(t_dt)], 'o-', ms=2, label="Prediction (single ESN)")

                        ax.set_xlim(65, 86)
                        ax.set_title(f"N={N}, Qubit={qubit}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel(f"⟨σ_{predict_op[-1]}⟩")
                        ax.legend()
                        first_plot_done = True

                # Save per-dataset metrics for this (N, qubit)
                df = pd.DataFrame(records)
                out_csv = os.path.join(
                    model_dir,
                    f"pred_eval_N{N}_q{qubit}_predSeed{pred_seed}_num{num_pred}.csv"
                )
                df.to_csv(out_csv, index=False)
                print(f"Saved do_predictions scores → {out_csv}")

                # Print aggregate
                all_scores.extend(records)
                maes = df["MAE"].values
                print(f"(N={N}, q={qubit}) MAE over {num_pred} datasets — "
                      f"mean={np.mean(maes):.4f}, std={np.std(maes):.4f}, "
                      f"min={np.min(maes):.4f}, max={np.max(maes):.4f}")

                # ── Part B: histogram over ESNs with varying rseed (dataset 0 only) ──
                pattern = os.path.join(model_dir, f"trainedmodel_Seed{train_seed}_rSeed*_{base_name}.pt")
                esn_paths = sorted(glob(pattern))

                if not esn_paths:
                    print(f"[hist] No ESN files found for pattern: {pattern}")
                else:
                    print(f"[hist] Found {len(esn_paths)} ESN files for histogram.")

                    if len(pred_histories) == 0:
                        print("[hist] No prediction datasets available; skipping histogram.")
                    else:
                        z0 = np.asarray(pred_histories[0])
                        z0_eval = select_series_for_qubit(z0, q_idx, N)

                        raw_maes = []
                        rseed_list = []

                        for pth in esn_paths:
                            try:
                                esn_i = load_esn(pth, device=device)
                                pred_i, true_i = predictor_model.predict_sequence(esn_i, z0_eval)
                                mae_i = mean_absolute_error(torch.tensor(pred_i), torch.tensor(true_i)).item()
                                raw_maes.append(mae_i)
                                rseed_list.append(extract_rseed_from_filename(pth))
                            except Exception as e:
                                print(f"[hist] Skipping {pth} due to error: {e}")

                        if raw_maes:
                            raw_maes = np.asarray(raw_maes, dtype=float)
                            log10_maes = np.log10(raw_maes + 1e-18)
                            ln_maes = np.log(raw_maes + 1e-18)

                            # draw only once per column to avoid overwriting (pick the final N row)
                            if row_idx == n_rows - 1:
                                ax_hist = axs[n_rows][col_idx]   # last row, same column
                                ax_hist.hist(log10_maes, bins='auto')
                                ax_hist.set_xlabel("log10(MAE)")
                                ax_hist.set_ylabel("Frequency")
                                ax_hist.set_title(f"MAE dist — N={N}, q={qubit}")
                                if PLOT_MAE_TOL is not None:
                                    ax_hist.axvline(np.log10(PLOT_MAE_TOL + 1e-18), linestyle='--')

                            hist_path = os.path.join(model_dir, f"hist_logMAE_op{predict_op}_N{N}_q{qubit}_dataset0.png")
                            hist_csv  = os.path.join(model_dir, f"hist_logMAE_op{predict_op}_N{N}_q{qubit}_dataset0.csv")
                            fig.savefig(hist_path, dpi=150)
                            print(f"[hist] Saved histogram subplot → {hist_path}")

                            # Save both logs for maximum compatibility with future fast-path loads
                            hist_df = pd.DataFrame({
                                "rseed": rseed_list[:len(raw_maes)],
                                "MAE": raw_maes,
                                "log10_MAE": log10_maes,
                                "ln_MAE": ln_maes,
                            })
                            hist_csv = os.path.join(model_dir, f"hist_logMAE_N{N}_q{qubit}_dataset0.csv")
                            hist_df.to_csv(hist_csv, index=False)
                            print(f"[hist] Saved histogram data → {hist_csv}")




        # Combined CSV across all (N, qubit)
        if all_scores:
            combo = pd.DataFrame(all_scores)
            combo_path = os.path.join(
                model_dir, f"pred_eval_ALL_predSeed{pred_seed}_num{num_pred}.csv"
            )
            combo.to_csv(combo_path, index=False)
            print(f"Saved combined do_predictions scores → {combo_path}")
#region Official Run
    def run_official_run_mode():
        logger = logging.getLogger("official_run")
        logger.info("Starting official run")
        model_dir = ensure_dir(MODEL_DIR)
        param_dir = ensure_dir(PARAM_DIR)

        mae_records = []
        config_records = []
        n_cols = len(N_list)
        n_rows = len(qubit_list)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows), squeeze=False)
        diagnostic_rows = []
        for row_idx, N in enumerate(N_list):
            truth_series = cache_prediction_truth(
                N=N, qubit_list=qubit_list, dt=dt, T=T, seed=pred_seed,
                op=predict_op, base_dir=CACHE_DIR
            )

            per_qubit_true = {}
            per_qubit_pred = {}
            for col_idx, qubit in enumerate(qubit_list):
                logger.info("Solving", extra={"extra": {"N": N, "qubit": qubit}})

                steps = int(T / dt)
                fmt_dt = format_dt_val(dt)
                qubit_tag = qubit_tag_str(ignore_qubit, qubit, N)
                extra_wash = 75 if ignore_washout else washout
                param_name = param_name_for_best(learning_algo, N, fmt_dt, extra_wash)
                print(param_name)
                best_param_file_path = best_params_file(param_dir, param_name)

                # Load best hyperparameters
                logger.info(f"Param name: {param_name}")
                best = load_best_params(param_dir, param_name, dt)
                if best:
                    logger.info("Loaded best parameters", extra={"extra": {"file": best_param_file_path}})
                else:
                    print("No best parameter file found. Using defaults.")
                    logger.warning("Best parameter file not found or invalid. Using defaults.",
                               extra={"extra": {"file": best_param_file_path}})
                
                # Predictor only for data orchestration
                predictor = ESNPredictor(
                    steps=steps, dt=dt, N=N,
                    qubit=(qubit_focus if ignore_qubit else qubit),
                    history_values=None, washout=washout,
                    batch_size=train_batch_size, train_batch_size=train_batch_size,
                    history_seed=train_seed, reservoir_seed=reservoir_seed,
                    learning_algo=learning_algo, device=device,
                    train_op=train_op, pred_op=predict_op
                )

                predictor.prepare_histories()
                # truth at dt for THIS plotted qubit and chosen predict operator
                z_eval = truth_series[qubit]  # already at dt
                # optional: high-res overlay (now operator-aware)
                # high‑res truth for plotting (operator-aware; returns floats already)
                z_test_hi, acc_chain_hi = generate_high_res_test(N, qubit, acc_dt, T, pred_seed, measure=predict_op)

                # purity requires reduced density matrices; generate a separate rho chain
                np.random.seed(pred_seed)
                acc_chain_rho = HeisenbergChain(num_qubits=N, target_qubit=qubit, dt=acc_dt, measure='rho')
                # i0 = 21
                # acc_chain_rho.psi[i0] = np.conj(acc_chain_rho.psi[i0])
                acc_chain_rho.evolve(int(T / acc_dt))
                rho_seq = acc_chain_rho.get_observable()  # list of 2x2 arrays

                purity = [float(np.real(np.trace(r @ r))) for r in rho_seq]
                # true-unitary checks for this qubit’s run
                sim_diag = {
                    "N": N, "qubit": qubit,
                    "norm_summary": summarize(acc_chain_hi.norm_history),
                    "energy_summary": summarize(acc_chain_hi.energy_history),
                    "norm_drift": drift_stats(acc_chain_hi.norm_history),
                    "energy_drift": drift_stats(acc_chain_hi.energy_history),
                }

            
                sim_diag["purity_summary"] = summarize(purity)
                sim_diag["purity_min"] = float(np.min(purity))
                sim_diag["purity_max"] = float(np.max(purity))

                diagnostic_rows.append({"simulator_checks": sim_diag})

                seeds = [reservoir_seed + i for i in range(test_esns)]
                all_preds = []
                base_name = base_name_str(learning_algo, train_batch_size, qubit_tag, fmt_dt)

                # Build the dt stream used for ESN evaluation ONCE
                # z_eval = z_eval_from_z_test(z_test_hi, dt, acc_dt)

                for i, rseed in enumerate(seeds):
                    model_path = os.path.join(model_dir, f"trainedmodel_Seed{train_seed}_rSeed{rseed}_{base_name}.pt")

                    # fresh ESN for each seed (lazily constructed)
                    esn = predictor.make_esn(
                        reservoir_size=best.get("reservoir_size", 500),
                        spectral_radius=best.get("spectral_radius", 0.99),
                        input_scaling=best.get("input_scaling", 1.01),
                        ridge_param=best.get("ridge_param", 0.00857),
                        leak_rate=best.get("leak_rate", 0.64),
                        sparsity=best.get("sparsity", 0.17),
                        feedback=best.get("feedback", 0),
                        bias_scaling=best.get("bias_scaling", 0.0),
                        seed=rseed
                    )

                    if os.path.exists(model_path):
                        print(f"Loading existing ESN Ex{i} (seed={rseed})")
                        logger.info("Loading existing ESN", extra={"extra": {"ex": i, "seed": rseed, "model_path": model_path}})
                        esn = load_esn(model_path, device=device)
                    else:
                        logger.info("Training ESN", extra={"extra": {"ex": i, "seed": rseed}})
                        predictor.train_esn(esn)
                        save_esn(esn, model_path)
                        logger.info("Saved trained ESN", extra={"extra": {"model_path": model_path}})

                    # Log the params you ACTUALLY used (not the old defaults)
                    config_records.append({
                        "N": N, "qubit": qubit, "Ex": i, "seed": rseed,
                        "reservoir_size": best.get("reservoir_size", 2347),
                        "spectral_radius": best.get("spectral_radius", 1.56526),
                        "input_scaling": best.get("input_scaling", 0.9480),
                        "ridge_param": best.get("ridge_param", 1e-1),
                        "leak_rate": best.get("leak_rate", 0.1947),
                        "sparsity": best.get("sparsity", 0.15286),
                        "feedback": best.get("feedback", 0),
                        "bias_scaling": best.get("bias_scaling", 0.277)
                    })

                    pred, true = predictor.predict_sequence(esn, z_eval)

                    # trainer diagnostics (behavior preserved)
                    if esn.trainer.xTx is not None:
                        esn.trainer.debug_covariance()
                    stats = esn.trainer.covariance_stats()
                    print(stats)

                    mae = mean_absolute_error(torch.tensor(pred), torch.tensor(true)).item()
                    mae_records.append({"N": N, "qubit": qubit, "Ex": i, "seed": rseed, "MAE": mae})
                    passed_plot_tol = (PLOT_MAE_TOL is None) or (mae <= PLOT_MAE_TOL)
                    if passed_plot_tol or not USE_TOL_FOR_STATS:
                        # collect for plotting/stats if passed, or if we keep all for stats
                        all_preds.append(pred)

                    # Optional: report skipped members
                    if not passed_plot_tol:
                        print(f"[plot-skip] N={N}, q={qubit}, Ex={i}, seed={rseed} MAE={mae:.4g} > tol={PLOT_MAE_TOL}")
                    
                if len(all_preds) == 0:
                    print(f"[plot-fallback] All members filtered (tol={PLOT_MAE_TOL}). Showing the last ESN anyway.")
                    all_preds = [pred]  # 'pred' from the last loop iteration
                # ---------- Plot overlay (dt metrics + high-res visuals) ----------
                ax = axs[col_idx][row_idx]
                all_preds_np = np.stack(all_preds)  # either filtered or unfiltered per the logic above
                mean_pred = all_preds_np.mean(axis=0)
                std_pred  = all_preds_np.std(axis=0)

                # truth aligned to predictions at dt (washout already applied inside predict)
                true_dt = z_eval[washout+1 : washout+1 + mean_pred.shape[0]]
                T_eff   = min(len(true_dt), len(mean_pred))
                true_dt = true_dt[:T_eff]
                mean_pred = mean_pred[:T_eff]
                std_pred  = std_pred[:T_eff]

                # record per-qubit series at dt
                per_qubit_true[qubit] = true_dt.copy()
                per_qubit_pred[qubit] = mean_pred.copy()

                # one per-qubit diagnostic block (no duplicates)
                per_qubit_diag = {
                    "N": N, "qubit": qubit,
                    "pred_summary": summarize(mean_pred),
                    "true_summary": summarize(true_dt),
                    "pred_bounds": check_bounds(mean_pred),
                    "acf1_pred": autocorr_lag1(mean_pred),
                    "acf1_true": autocorr_lag1(true_dt),
                    "series_error": compare_series(mean_pred, true_dt),
                }
                diagnostic_rows.append(per_qubit_diag)

                # time axes — RELATIVE (start at t = (washout+1)*dt)
                t_dt   = np.arange(len(mean_pred)) * dt
                t0     = (washout+1 ) * dt
                start_hi = int(round(t0 / acc_dt))
                t_fine_rel = np.arange(len(z_test_hi)) * acc_dt - t0

                # draw
                ax.plot(t_fine_rel[start_hi:], z_test_hi[start_hi:], '-', lw=1.2, color='green', label='True (acc_dt)')
                ax.plot(t_dt, true_dt, 'o', ms=1, color='darkgreen', alpha=0.8)

                ax.fill_between(t_dt, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, label="±1σ")
                # ax.plot(t_dt, mean_pred, 'o-', ms=2, label="Mean Prediction") #TODO Rememebr this

                # label ensemble members only once to avoid legend spam
                for i, ipreds in enumerate(all_preds_np):
                    ax.plot(t_dt, ipreds[:len(t_dt)], alpha=0.5, lw=0.9, label=f"ESN members {i}")

                # ax.set_xlim(65, 86)
                ax.set_title(f"N={N}, Qubit={qubit}")
                ax.set_xlabel("Time")
                ax.set_ylabel(f"⟨σ_{predict_op[-1]}⟩")
                ax.legend()

                fig_path = os.path.join(model_dir, f"overlay_pSeed{pred_seed}_op{predict_op}_tol{PLOT_MAE_TOL}_{base_name}.pdf")
                plt.savefig(fig_path)
                logger.info("Saved overlay plot", extra={"extra": {"fig_path": fig_path}})

                # ---------- (OPTIONAL) Global magnetization ----------
                if len(per_qubit_true) >= 2 and len(per_qubit_pred) >= 2:
                    T_min_true = min(len(v) for v in per_qubit_true.values())
                    T_min_pred = min(len(v) for v in per_qubit_pred.values())
                    T_min = min(T_min_true, T_min_pred)

                    Mz_true = magnetization_from_qubit_series(per_qubit_true, t_len=T_min)
                    Mz_pred = magnetization_from_qubit_series(per_qubit_pred, t_len=T_min)

                    mag_diag = {
                        "mag_true_summary": summarize(Mz_true),
                        "mag_pred_summary": summarize(Mz_pred),
                        "mag_true_drift": drift_stats(Mz_true),
                        "mag_pred_drift": drift_stats(Mz_pred),
                        "mag_series_error": compare_series(Mz_pred, Mz_true)
                    }
                    diagnostic_rows.append({"global_magnetization": mag_diag})

        # Save diagnostics summary
        diag_path_json = os.path.join(model_dir, "physics_summary.json")
        with open(diag_path_json, "w") as f:
            json.dump(diagnostic_rows, f, indent=2)
        print(f"Saved physics diagnostics to {diag_path_json}")

        mae_df = pd.DataFrame(mae_records)
        mae_df.to_csv(os.path.join(model_dir, "official_run_mae_summary.csv"), index=False)
        print("Saved MAE summary to official_run_mae_summary.csv")

        with open(os.path.join(model_dir, "official_run_esn_configs.json"), 'w') as f:
            json.dump(config_records, f, indent=4)
            print("Saved ESN config log to official_run_esn_configs.json")

        logger.info("Saved physics diagnostics", extra={"extra": {"path": diag_path_json}})
        logger.info("Saved MAE summary CSV", extra={"extra": {"path": os.path.join(model_dir, 'official_run_mae_summary.csv')}})
        logger.info("Saved ESN config log", extra={"extra": {"path": os.path.join(model_dir, 'official_run_esn_configs.json')}})

    # ─── Dispatch based on mode ─────────────────────────────────────────────
    if do_plot_hyper:
        run_plot_hyper_mode()
    elif do_tune:
        run_tune_mode()
    elif do_predictions:
        run_predictions_mode()
    elif official_run:
        run_official_run_mode()

    # render_physics_report("./examples/Heisenberg_Chain/cache/physics_summary.json")
    # scorecard_physics("./examples/Heisenberg_Chain/cache/physics_summary.json")
    plt.show()
