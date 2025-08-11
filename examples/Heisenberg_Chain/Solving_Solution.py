import os
import json
import pickle
import numpy as np
import torch
from echostate import ESN  # <-- our new ESN module
from echostate.utils import mean_absolute_error
from .Heisenberg_sim import HeisenbergChain
import matplotlib.pyplot as plt
from qutip import Qobj, sigmaz, expect
import pandas as pd

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning
)

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
    Orchestrates data generation/caching and provides helpers for training & prediction.
    Does NOT instantiate an ESN in __init__ (lazy construction only).
    """

    def __init__(self,
                 steps: int,
                 dt: float,
                 N: int,
                 qubit: int,
                 history_values: list = None,
                 washout: int = 0,
                 batch_size: int = 1,
                 training_depth: int = 1,
                 history_seed: int = None,
                 reservoir_seed: int = None,
                 cache_dir: str = "./examples/Heisenberg_Chain/cache/",
                 device: torch.device = torch.device('cpu')):
        # Core settings
        self.steps = steps
        self.dt = dt
        self.N = N
        self.qubit = qubit
        self.washout = washout
        self.batch_size = batch_size
        self.training_depth = training_depth
        self.device = device

        # Seeds
        self.history_seed   = history_seed if history_seed is not None else reservoir_seed
        self.reservoir_seed = reservoir_seed if reservoir_seed is not None else history_seed

        # Provided test history (optional)
        self.test_history = history_values

        # Cache naming
        self.cache_dir = cache_dir
        fmt_dt_val = str(round(self.dt, 5)).replace(".", "_", 1)
        qubit_tag = f"Qbts(dpth{training_depth}){self.N}"
        self.cache_name = f"Historydata_Seed{self.history_seed}_T{steps*dt}_{qubit_tag}_dt{fmt_dt_val}.pkl"
        self.cache_path = os.path.join(cache_dir, self.cache_name)

        # Histories are built on demand
        self.histories = None

    # ---------- Data ----------
    def prepare_histories(self):
        """
        Load or generate training histories. Idempotent.
        """
        if self.histories is not None:
            return self.histories

        np.random.seed(self.history_seed)
        self.histories = []

        if self.test_history is None or self.training_depth > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    self.histories = pickle.load(f)
                print(f"Loaded {len(self.histories)} training histories from cache.")
            else:
                for _ in range(self.training_depth):
                    chain = HeisenbergChain(num_qubits=self.N,
                                            target_qubit=self.qubit,
                                            dt=self.dt)
                    chain.evolve(self.steps)
                    self.histories.append(chain.get_sz())
                print(f"Collected {len(self.histories)} simulation histories.")

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
                 feedback=1, bias_scaling=0.4, seed=None):
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
            seed=(self.reservoir_seed if seed is None else seed),
        ).to(self.device)
        return esn

    def train_esn(self, esn):
        """Fit a provided ESN on the prepared dataset."""
        inputs, targets = self.build_dataset()
        print(f"Training ESN on {len(inputs)} sequences (washout={self.washout})")
        esn.fit(inputs, targets)

    def predict_sequence(self, esn, z_test):
        """
        Predict next-step σ_z on a given test scalar sequence z_test.
        Returns numpy array predictions and aligned true sequence (after washout).
        """
        X_test = torch.tensor(z_test[:-1], dtype=torch.float32, device=self.device).unsqueeze(-1).unsqueeze(0)
        preds = esn.predict(X_test)[0].detach().cpu().numpy().flatten()
        true = z_test[self.washout + 1: self.washout + 1 + len(preds)]
        return preds, true


#region Physical TESTS
def summarize(arr):
    arr = np.asarray(arr)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max())
    }

def check_bounds(z_pred, eps=1e-6):
    z_pred = np.asarray(z_pred)
    violations = np.where(np.abs(z_pred) > 1 + eps)[0]
    max_excess = float((np.abs(z_pred) - 1).clip(min=0).max()) if violations.size else 0.0
    return {
        "num_violations": int(violations.size),
        "violation_indices_sample": violations[:10].tolist(),
        "max_excess_over_1": max_excess
    }

def autocorr_lag1(x):
    x = np.asarray(x)
    x = x - x.mean()
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(np.correlate(x[:-1], x[1:])[0] / ((len(x)-1)*x.std()*x.std()))

def magnetization_from_qubit_series(z_per_qubit, t_len=None):
    """
    z_per_qubit: dict {q: np.array(seq_len)} for same timeline (after washout alignment if needed).
    returns: Mz(t) array
    """
    qs = sorted(z_per_qubit.keys())
    Z = np.stack([z_per_qubit[q][:t_len] for q in qs], axis=0)  # (Q, T)
    return Z.sum(axis=0)

def drift_stats(series):
    """How constant is a series? report mean abs diff and slope via linear fit."""
    y = np.asarray(series)
    diffs = np.abs(np.diff(y))
    # simple slope
    t = np.arange(len(y))
    slope = float(np.polyfit(t, y, 1)[0]) if len(y) >= 2 else 0.0
    return {"mean_abs_step": float(diffs.mean() if len(diffs) else 0.0), "linear_slope": slope}

def compare_series(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    T = min(len(y_pred), len(y_true))
    y_pred = y_pred[:T]; y_true = y_true[:T]
    mae = mean_absolute_error(torch.tensor(y_pred), torch.tensor(y_true)).item()
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    return {"mae": float(mae), "rmse": rmse}

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
    
def Heisen_tune(predictor, study_name, study_loc, washout, seed, n_trials, param_name, dt=None, plots=False):
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
            reservoir_limit=[700, 1000],
            spectral_radius_limit=[0.1, 2],
            feedback_limit=1,
            input_scaling_limit=[1.0, 3.0],
            ridge_param_limit=[1e-8, 1],
            leak_rate_limit=[0.2, 1.0],
            sparsity_limit=[0.1, 1.0],
            bias_scaling_limit=[0.2, 0.8],
            device=predictor.device,
            learning_algo="inv"
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

def render_physics_report(json_path="./examples/Heisenberg_Chain/cache/physics_summary.json",
                          out_dir=None,
                          show=True):
    """
    Read physics_summary.json and produce a compact visual report:
      - Per-(N,qubit) ESN MAE/RMSE
      - Bounds violations (|<σz>|>1)
      - Simulator norm & energy drift summaries
      - Purity statistics
      - Global magnetization summaries (if present)

    Creates a multi-page PDF and a few PNGs in out_dir, and prints a concise text summary.
    """
    import os, json, math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(json_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Load ----------
    with open(json_path, "r") as f:
        rows = json.load(f)

    # ---------- Buckets ----------
    # per-qubit ESN diag
    esn_rows = []   # dicts with keys: N, qubit, series_error{mae,rmse}, pred_bounds{num_violations,...}, summaries, acf1...
    # simulator per-qubit diag
    sim_rows = []   # dicts inside {"simulator_checks": {...}}
    # global magnetization rows
    mag_rows = []   # dicts inside {"global_magnetization": {...}}

    for item in rows:
        if "simulator_checks" in item:
            sim_rows.append(item["simulator_checks"])
        elif "global_magnetization" in item:
            mag = item["global_magnetization"]
            # inject N if missing
            if "N" not in mag:
                mag["N"] = None
            mag_rows.append(mag)
        else:
            # assume per-qubit ESN diag shape
            # require minimal keys to count it as ESN diag
            if all(k in item for k in ["N", "qubit", "series_error", "pred_bounds"]):
                esn_rows.append(item)

    # helper to tag
    def tag(n, q):
        return f"N{n}_q{q}"

    # ---------- Aggregate helpers ----------
    def collect_esn_metric(name):
        xs, vals = [], []
        for r in esn_rows:
            xs.append(tag(r["N"], r["qubit"]))
            vals.append(float(r["series_error"].get(name, np.nan)))
        return xs, np.array(vals)

    def collect_bounds():
        xs, cnts, max_exc = [], [], []
        for r in esn_rows:
            xs.append(tag(r["N"], r["qubit"]))
            pb = r.get("pred_bounds", {})
            cnts.append(int(pb.get("num_violations", 0)))
            max_exc.append(float(pb.get("max_excess_over_1", 0.0)))
        return xs, np.array(cnts), np.array(max_exc)

    def collect_sim_drift(key="energy_drift", sub="linear_slope"):
        xs, vals = [], []
        for r in sim_rows:
            xs.append(tag(r["N"], r["qubit"]))
            vals.append(abs(float(r.get(key, {}).get(sub, 0.0))))
        return xs, np.array(vals)

    def collect_purity(summary_stat="mean"):
        xs, vals = [], []
        for r in sim_rows:
            xs.append(tag(r["N"], r["qubit"]))
            vals.append(float(r.get("purity_summary", {}).get(summary_stat, np.nan)))
        return xs, np.array(vals)

    # ---------- Prepare data ----------
    x_mae_labels, mae_vals = collect_esn_metric("mae")
    _, rmse_vals = collect_esn_metric("rmse")
    x_bounds_labels, bound_cnts, bound_maxexc = collect_bounds()
    x_energy_labels, energy_slope = collect_sim_drift("energy_drift", "linear_slope")
    x_norm_labels, norm_step = collect_sim_drift("norm_drift", "mean_abs_step")
    x_purity_mean, purity_mean = collect_purity("mean")
    _, purity_std = collect_purity("std")
    _, purity_min = collect_purity("min")
    _, purity_max = collect_purity("max")

    # ---------- Plotting ----------
    pdf_path = os.path.join(out_dir, "physics_summary_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # 1) ESN error bars
        fig, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(len(x_mae_labels))
        ax.bar(idx - 0.2, mae_vals, width=0.4, label="MAE")
        ax.bar(idx + 0.2, rmse_vals, width=0.4, label="RMSE")
        ax.set_xticks(idx)
        ax.set_xticklabels(x_mae_labels, rotation=45, ha="right")
        ax.set_title("ESN Prediction Error per (N, qubit)")
        ax.set_ylabel("Error")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_esn_errors.png")); plt.close(fig)

        # 2) Bounds violations
        fig, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(len(x_bounds_labels))
        ax.bar(idx, bound_cnts, label="#(|⟨σz⟩|>1)")
        ax.plot(idx, bound_maxexc, marker="o", linestyle="--", label="Max excess over 1")
        ax.set_xticks(idx)
        ax.set_xticklabels(x_bounds_labels, rotation=45, ha="right")
        ax.set_title("ESN Physical Bounds Checks")
        ax.set_ylabel("Count / Excess")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_bounds.png")); plt.close(fig)

        # 3) Simulator drifts (energy & norm)
        fig, ax = plt.subplots(figsize=(10, 4))
        idxE = np.arange(len(x_energy_labels))
        ax.bar(idxE - 0.2, energy_slope, width=0.4, label="|Energy slope|")
        idxN = np.arange(len(x_norm_labels))
        ax.bar(idxN + 0.2, norm_step, width=0.4, label="Mean |Δ Norm|")
        xticks = list(dict.fromkeys(x_energy_labels + x_norm_labels))  # preserve order and uniqueness
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, ha="right")
        ax.set_title("Simulator Unitarity Checks")
        ax.set_ylabel("Drift")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_drift.png")); plt.close(fig)

        # 4) Purity summary (mean ± range)
        fig, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(len(x_purity_mean))
        ax.bar(idx, purity_mean, width=0.5, label="Purity mean")
        # error bars as min/max whiskers
        err_low = purity_mean - purity_min
        err_high = purity_max - purity_mean
        ax.errorbar(idx, purity_mean, yerr=[err_low, err_high], fmt="none", capsize=4, label="Range (min–max)")
        ax.set_xticks(idx)
        ax.set_xticklabels(x_purity_mean, rotation=45, ha="right")
        ax.set_title("Single‑Qubit Purity (from simulator ρ_k)")
        ax.set_ylabel("Tr(ρ²)")
        ax.set_ylim(0.45, 1.05)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig); fig.savefig(os.path.join(out_dir, "plot_purity.png")); plt.close(fig)

        # 5) Global magnetization summaries (if present)
        if mag_rows:
            # One page per N (or once if N is None)
            Ns = sorted(set(m.get("N") for m in mag_rows))
            for n in Ns:
                mags = [m for m in mag_rows if m.get("N") == n]
                # take first (they're summaries)
                m = mags[0]
                def fmt_summary(prefix, d):
                    return f"{prefix} mean={d.get('mean', 'NA'):.4g}, std={d.get('std','NA'):.4g}, slope={m.get(prefix+'_drift',{}).get('linear_slope',0):.3e}"
                fig, ax = plt.subplots(figsize=(8, 3))
                parts = []
                if "mag_true_summary" in m:
                    parts.append(fmt_summary("mag_true", m["mag_true_summary"]))
                if "mag_pred_summary" in m:
                    parts.append(fmt_summary("mag_pred", m["mag_pred_summary"]))
                if "mag_series_error" in m:
                    parts.append(f"MAE={m['mag_series_error'].get('mae',np.nan):.4g}, RMSE={m['mag_series_error'].get('rmse',np.nan):.4g}")
                text = "\n".join(parts) if parts else "No magnetization summary found."
                ax.axis("off")
                ax.set_title(f"Global Magnetization Summary (N={n})")
                ax.text(0.02, 0.5, text, va="center", ha="left", family="monospace")
                fig.tight_layout()
                pdf.savefig(fig); fig.savefig(os.path.join(out_dir, f"mag_summary_N{n}.png")); plt.close(fig)

    # ---------- Console summary ----------
    def topk(labels, values, k=5, reverse=False):
        order = np.argsort(values)
        if reverse:
            order = order[::-1]
        order = order[:k]
        return [(labels[i], float(values[i])) for i in order]

    print("\n=== Physics Report ===")
    if len(mae_vals):
        worst_mae = topk(x_mae_labels, mae_vals, k=min(5, len(mae_vals)), reverse=True)
        print("Worst MAE (top 5):")
        for lbl, val in worst_mae:
            print(f"  {lbl}: {val:.4g}")
    if len(bound_cnts) and bound_cnts.sum() > 0:
        offenders = [(x_bounds_labels[i], int(bound_cnts[i])) for i in np.where(bound_cnts>0)[0]]
        print("Bounds violations (|<σz>| > 1):")
        for lbl, cnt in offenders:
            print(f"  {lbl}: {cnt}")
    if len(energy_slope):
        worst_energy = topk(x_energy_labels, energy_slope, k=min(5, len(energy_slope)), reverse=True)
        print("Largest |energy slope| (top 5):")
        for lbl, val in worst_energy:
            print(f"  {lbl}: {val:.3e}")
    if len(norm_step):
        worst_norm = topk(x_norm_labels, norm_step, k=min(5, len(norm_step)), reverse=True)
        print("Largest mean |Δ norm| (top 5):")
        for lbl, val in worst_norm:
            print(f"  {lbl}: {val:.3e}")

    print(f"\nSaved report PDF → {pdf_path}")
    print(f"PNGs saved in     → {out_dir}")

    if show:
        # Open the last PNG (arbitrary) to pop a window in interactive sessions
        try:
            import webbrowser
            webbrowser.open_new(pdf_path)
        except Exception:
            pass

import json
from pathlib import Path
from typing import Union, Dict, Any, List

def scorecard_physics(summary_json: Union[str, Path, Dict, List],
                      *,
                      # ---- thresholds you can tweak ----
                      norm_std_tol=1e-12,
                      norm_slope_tol=1e-15,
                      energy_std_tol=1e-10,
                      energy_slope_tol=1e-12,
                      purity_min_floor=0.0,
                      purity_max_ceiling=1.0,
                      mae_ok=0.03, mae_warn=0.08,
                      rmse_ok=0.04, rmse_warn=0.10,
                      acf_diff_ok=0.01, acf_diff_warn=0.03,
                      mag_slope_tol=5e-6,
                      quiet=False) -> Dict[str, Any]:
    """
    Read a physics_summary.json (either a path or an already-loaded list/dict)
    and print a concise scorecard, returning a structured summary.

    Status semantics:
      ✅ = pass (green)   | ⚠️ = caution (yellow) | ❌ = fail (red)

    Thresholds are defaults you can tune to your system’s scale.
    """
    def _status(ok: bool=None, warn: bool=False):
        if ok is True:  return "✅"
        if warn:        return "⚠️"
        return "❌"

    # Load JSON
    if isinstance(summary_json, (str, Path)):
        data = json.loads(Path(summary_json).read_text())
    else:
        data = summary_json

    # Normalize list of blocks
    if isinstance(data, dict):
        blocks = [data]
    else:
        blocks = list(data)

    # Extract blocks by type
    sim_checks = [b["simulator_checks"] for b in blocks if "simulator_checks" in b]
    mag_block  = next((b["global_magnetization"] for b in blocks if "global_magnetization" in b), None)
    qubit_series = [b for b in blocks if all(k in b for k in ("qubit","pred_summary","true_summary","series_error","acf1_pred","acf1_true","pred_bounds"))]

    report = {"simulator": {}, "qubits": {}, "magnetization": {}, "overall": {"ok": True, "notes": []}}

    # ---------- Simulator checks per qubit ----------
    for sc in sim_checks:
        q = sc["qubit"]
        ns = sc["norm_summary"]
        es = sc["energy_summary"]
        nd = sc["norm_drift"]
        ed = sc["energy_drift"]
        ps = sc.get("purity_summary", {})
        # Norm
        norm_ok  = (ns["std"] <= norm_std_tol) and (abs(nd["linear_slope"]) <= norm_slope_tol)
        norm_warn = (ns["std"] <= 10*norm_std_tol) and (abs(nd["linear_slope"]) <= 10*norm_slope_tol)
        # Energy
        en_ok  = (es["std"] <= energy_std_tol) and (abs(ed["linear_slope"]) <= energy_slope_tol)
        en_warn = (es["std"] <= 10*energy_std_tol) and (abs(ed["linear_slope"]) <= 10*energy_slope_tol)
        # Purity in range
        purity_ok = (ps.get("min", 0.0) >= purity_min_floor - 1e-9) and (ps.get("max", 1.0) <= purity_max_ceiling + 1e-9)

        report["simulator"][q] = {
            "norm":   {"status": _status(ok=norm_ok, warn=norm_warn), "std": ns["std"], "slope": nd["linear_slope"]},
            "energy": {"status": _status(ok=en_ok, warn=en_warn), "std": es["std"], "slope": ed["linear_slope"]},
            "purity": {"status": _status(ok=purity_ok), "mean": ps.get("mean"), "min": ps.get("min"), "max": ps.get("max")},
        }

    # ---------- Per-qubit ESN series ----------
    for qb in qubit_series:
        q = qb["qubit"]
        mae  = qb["series_error"]["mae"]
        rmse = qb["series_error"]["rmse"]
        acf_diff = abs(qb["acf1_pred"] - qb["acf1_true"])
        bounds_ok = (qb["pred_bounds"]["num_violations"] == 0)

        # MAE / RMSE grading
        mae_status  = _status(ok=(mae <= mae_ok),  warn=(mae_ok < mae <= mae_warn))
        rmse_status = _status(ok=(rmse <= rmse_ok), warn=(rmse_ok < rmse <= rmse_warn))
        acf_status  = _status(ok=(acf_diff <= acf_diff_ok), warn=(acf_diff_ok < acf_diff <= acf_diff_warn))
        bounds_status = _status(ok=bounds_ok)

        report["qubits"][q] = {
            "mae":  {"status": mae_status,  "value": mae},
            "rmse": {"status": rmse_status, "value": rmse},
            "acf1_diff": {"status": acf_status, "value": acf_diff},
            "bounds": {"status": bounds_status, "violations": qb["pred_bounds"]["num_violations"]},
            "pred_summary": qb["pred_summary"],
            "true_summary": qb["true_summary"],
        }

    # ---------- Global magnetization ----------
    if mag_block:
        mtrue = mag_block["mag_true_summary"]
        mpred = mag_block["mag_pred_summary"]
        mte   = mag_block["mag_true_drift"]["linear_slope"]
        mpe   = mag_block["mag_pred_drift"]["linear_slope"]
        m_mae = mag_block["mag_series_error"]["mae"]
        m_rmse= mag_block["mag_series_error"]["rmse"]

        drift_ok = (abs(mte) <= mag_slope_tol and abs(mpe) <= mag_slope_tol)
        report["magnetization"] = {
            "drift": {"status": _status(ok=drift_ok, warn=abs(mte)<=10*mag_slope_tol and abs(mpe)<=10*mag_slope_tol),
                      "true_slope": mte, "pred_slope": mpe},
            "error": {
                "mae":  {"status": _status(ok=(m_mae<=mae_ok),  warn=(mae_ok<m_mae<=mae_warn)),  "value": m_mae},
                "rmse": {"status": _status(ok=(m_rmse<=rmse_ok), warn=(rmse_ok<m_rmse<=rmse_warn)), "value": m_rmse},
            },
            "true_summary": mtrue, "pred_summary": mpred,
        }

    # ---------- Print summary ----------
    if not quiet:
        print("\n=== Physics Summary Scorecard ===")
        # Simulator
        print("\n-- Simulator checks (per qubit) --")
        for q in sorted(report["simulator"].keys()):
            s = report["simulator"][q]
            print(f"Qubit {q}: "
                  f"Norm {s['norm']['status']} (std={s['norm']['std']:.2e}, slope={s['norm']['slope']:.2e}) | "
                  f"Energy {s['energy']['status']} (std={s['energy']['std']:.2e}, slope={s['energy']['slope']:.2e}) | "
                  f"Purity {s['purity']['status']} (mean={s['purity']['mean']:.3f}, "
                  f"range=[{s['purity']['min']:.3f},{s['purity']['max']:.3f}])")

        # Qubit series
        print("\n-- ESN predictions vs. truth (per qubit) --")
        for q in sorted(report["qubits"].keys()):
            s = report["qubits"][q]
            print(f"Qubit {q}: "
                  f"MAE {s['mae']['status']}={s['mae']['value']:.4f} | "
                  f"RMSE {s['rmse']['status']}={s['rmse']['value']:.4f} | "
                  f"ACFΔ {s['acf1_diff']['status']}={s['acf1_diff']['value']:.4f} | "
                  f"Bounds {s['bounds']['status']} (violations={s['bounds']['violations']})")

        # Magnetization
        if report["magnetization"]:
            m = report["magnetization"]
            print("\n-- Global magnetization --")
            print(f"Drift {m['drift']['status']} (true slope={m['drift']['true_slope']:.2e}, "
                  f"pred slope={m['drift']['pred_slope']:.2e}) | "
                  f"MAE {m['error']['mae']['status']}={m['error']['mae']['value']:.4f} | "
                  f"RMSE {m['error']['rmse']['status']}={m['error']['rmse']['value']:.4f}")

        print("\nLegend: ✅ pass | ⚠️ caution | ❌ fail\n")

    # Overall ok?
    any_fail = False
    for section in ("simulator","qubits","magnetization"):
        sec = report.get(section, {})
        if isinstance(sec, dict):
            for _, v in sec.items():
                if isinstance(v, dict):
                    # look for nested 'status'
                    for vv in v.values() if "status" not in v else [v]:
                        if isinstance(vv, dict) and "status" in vv and vv["status"] == "❌":
                            any_fail = True
                elif "status" in v and v["status"] == "❌":
                    any_fail = True
    report["overall"]["ok"] = not any_fail
    return report

# Example usage:
# scorecard_physics("physics_summary.json")

#region CONTROL  
if __name__ == '__main__':
    # ─── Configuration ──────────────────────────────────────────────────────
    T              = 100
    N_list         = [5]
    train_seed     = 31
    reservoir_seed  = 310
    pred_seed     = 31415
    qubit_list     = [0,1,2]       # list of qubit indices
    washout        = 75
    dt             = 0.2
    acc_dt         = 0.05
    training_depth = 400 # Number of time series used to train 1 ESN
    testing_depth  = 200 # Number of ESNs trained

    # Modes: set exactly one of these to True
    do_tune        = False  # run Optuna tuning
    do_plot_hyper  = False  # just plot hyper‐vs‐N
    official_run   = False   # run ensemble of ESNs & shaded plot
    do_predictions = True
    
    ignore_qubit = True
    ignore_washout = True # Applies only to hyperparameters so far
    # optuna params (only used if do_tune)
    n_trials   = 3
    num_pred      = 40 

    # ─── Preload high-res reference & (for single/official) test history ───
    from qutip import Qobj, sigmaz, expect
    np.random.seed(train_seed)
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
        predictor = ESNPredictor(
            steps=int(T/dt), dt=dt,
            N=N_list[0], qubit=qubit_list[0],
            history_values=None,
            washout=washout,
            batch_size=training_depth,
            training_depth=training_depth,
            history_seed=train_seed,
            reservoir_seed=reservoir_seed,
            device=device
        )

        fmt_dt_val = str(dt).rstrip("0").rstrip(".").replace(".", "_", 1)
        qubit_tag = f"Qbts({(0 if ignore_qubit else qubit_list[0]) + 1}){N_list[0]}"
        extra_wash = 75 if ignore_washout else washout
        param_name = f"Seed31415_{qubit_tag}_dt{fmt_dt_val}_dpth50_wsht{extra_wash}"
        study_name = f"esnStudy_Seed31415_{qubit_tag}_dt{fmt_dt_val}_dpth50_wsht75"
        print(study_name)
        study_dir  = "./examples/Heisenberg_Chain/trained_esns/"
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
            plots=False
        )
#region official run
    elif official_run:

        model_dir = "./examples/Heisenberg_Chain/cache/"
        param_dir = "./examples/Heisenberg_Chain/trained_esns/"
        os.makedirs(model_dir, exist_ok=True)

        mae_records = []
        config_records = []
        n_rows = len(N_list)
        n_cols = len(qubit_list)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        diagnostic_rows = []
        for row_idx, N in enumerate(N_list):
            per_qubit_true = {}
            per_qubit_pred = {}
            for col_idx, qubit in enumerate(qubit_list):
                print(f"\nSolving N={N}, qubit={qubit}")

                steps = int(T / dt)
                fmt_dt_val = str(dt).rstrip("0").rstrip(".").replace(".", "_", 1)
                qubit_tag = f"Qbts({(0 if ignore_qubit else qubit) + 1}){N}"
                extra_wash = 75 if ignore_washout else washout
                param_name = f"Seed31415_{qubit_tag}_dt{fmt_dt_val}_dpth50_wsht{extra_wash}"
                print(param_name)
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

                # Predictor only for data orchestration
                predictor = ESNPredictor(
                    steps=steps,
                    dt=dt,
                    N=N,
                    qubit=(qubit if not ignore_qubit else 0),
                    history_values=None,
                    washout=washout,
                    batch_size=training_depth,
                    training_depth=training_depth,
                    history_seed=train_seed,
                    reservoir_seed=reservoir_seed,
                    device=device
                )

                # High-res reference (test) sequence for this (N,qubit)
                np.random.seed(train_seed)
                acc_chain = HeisenbergChain(num_qubits=N, target_qubit=qubit, dt=acc_dt)
                acc_chain.evolve(int(T / acc_dt), store_reduced=True)
                raw = acc_chain.get_sz()
                z_test = np.asarray([
                    float(expect(sigmaz(), Qobj(rho, dims=[[2], [2]])))
                    for rho in raw
                ])
                
                # true-unitary checks for this qubit’s run
                sim_diag = {
                    "N": N, "qubit": qubit,
                    "norm_summary": summarize(acc_chain.norm_history),
                    "energy_summary": summarize(acc_chain.energy_history),
                    "norm_drift": drift_stats(acc_chain.norm_history),
                    "energy_drift": drift_stats(acc_chain.energy_history),
                }

                # single-qubit purity when store_reduced=True
                # acc_chain.sz_history contains ρ_k(t) (complex 2x2 arrays)
                rho_seq = acc_chain.get_sz()  # list/array of 2x2
                purity = []
                for rho in rho_seq:
                    # rho is 2x2 complex ndarray
                    purity.append(float(np.real(np.trace(rho @ rho))))
                sim_diag["purity_summary"] = summarize(purity)
                sim_diag["purity_min"] = float(np.min(purity))
                sim_diag["purity_max"] = float(np.max(purity))

                diagnostic_rows.append({"simulator_checks": sim_diag})
                seeds = [reservoir_seed + i for i in range(testing_depth)]
                all_preds = []
                base_name = f"N{N}_{qubit_tag}_dt{fmt_dt_val}"

                for i, rseed in enumerate(seeds):
                    model_path = os.path.join(model_dir, f"trainedmodel_Seed{train_seed}_rSeed{rseed}_{base_name}.pt")

                    # fresh ESN for each seed (lazily constructed)
                    esn = predictor.make_esn(
                        reservoir_size=best.get("reservoir_size", 900),
                        spectral_radius=best.get("spectral_radius", 1.25),
                        input_scaling=best.get("input_scaling", 0.55),
                        ridge_param=best.get("ridge_param", 1e-1),
                        leak_rate=best.get("leak_rate", 0.9),
                        sparsity=best.get("sparsity", 0.2),
                        feedback=best.get("feedback", 1),
                        bias_scaling=best.get("bias_scaling", 0.4),
                        seed=rseed
                    )

                    if os.path.exists(model_path):
                        print(f"Loading existing ESN Ex{i} (seed={rseed})")
                        esn = torch.load(model_path, weights_only=False)
                        esn.to(device).eval()
                    else:
                        print(f"Training ESN Ex{i} (seed={rseed})")
                        predictor.train_esn(esn)
                        torch.save(esn, model_path)

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

                    pred, true = predictor.predict_sequence(esn, z_test)
                    all_preds.append(pred)

                    mae = mean_absolute_error(torch.tensor(pred), torch.tensor(true)).item()
                    mae_records.append({"N": N, "qubit": qubit, "Ex": i, "seed": rseed, "MAE": mae})

                # Only do this if you actually evaluated more than one qubit
                if len(per_qubit_true) >= 1 and len(per_qubit_pred) >= 1:
                    # equalize length across qubits (use min T)
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

                    # Optional: assert invariants (fail-fast)
                    # assert abs(mag_diag["mag_true_drift"]["linear_slope"]) < 1e-8, "Energy drift too large"
                
                # Plot overlay (unchanged style, just uses arrays we built)
                ax = axs[row_idx][col_idx]
                all_preds = np.stack(all_preds)
                
                mean_pred = all_preds.mean(axis=0)
                true_aligned = z_test[washout + 1: washout + 1 + mean_pred.shape[0]]

                per_qubit_true[qubit] = true_aligned.copy()
                per_qubit_pred[qubit] = mean_pred.copy()

                # quick per-qubit ESN sanity
                per_qubit_diag = {
                    "N": N, "qubit": qubit,
                    "pred_summary": summarize(mean_pred),
                    "true_summary": summarize(true_aligned),
                    "pred_bounds": check_bounds(mean_pred),
                    "acf1_pred": autocorr_lag1(mean_pred),
                    "acf1_true": autocorr_lag1(true_aligned),
                    "series_error": compare_series(mean_pred, true_aligned)
                }
                # append it to a list you’ll dump later
                diagnostic_rows.append(per_qubit_diag)
                
                
                times = np.arange(all_preds.shape[1]) * dt
                mean_pred = all_preds.mean(axis=0)
                std_pred = all_preds.std(axis=0)
                true = z_test[washout + 1: washout + 1 + all_preds.shape[1]]
                true_t = np.arange(len(true)) * dt

                ax.fill_between(times, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, label="±1σ")
                ax.plot(times, mean_pred, 'o-', markersize=2, label="Mean Prediction")
                for i, ipreds in enumerate(all_preds):
                    ax.plot(times, ipreds, label=f"ESNS {i}")
                ax.plot(true_t, true, '-', linewidth=1.2, label='True ⟨σ_z⟩')
                ax.set_xlim(0, 15)
                ax.set_title(f"N={N}, Qubit={qubit}")
                ax.set_xlabel("Time")
                ax.set_ylabel("⟨σ_z⟩")
                ax.legend()

                fig_path = os.path.join(model_dir, f"official_overlay_{base_name}.pdf")
                fig.savefig(fig_path)
                print(f"Saved overlay plot to {fig_path}")
                
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
#region Predictions
    elif do_predictions:
        
        from glob import glob

        model_dir = "./examples/Heisenberg_Chain/cache/"
        param_dir = "./examples/Heisenberg_Chain/trained_esns/"
        os.makedirs(model_dir, exist_ok=True)

        n_rows = len(N_list)
        n_cols = len(qubit_list)
        fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        all_scores = []  # accumulate across all (N, qubit)

        for row_idx, N in enumerate(N_list):
            for col_idx, qubit in enumerate(qubit_list):
                print(f"\n[do_predictions] N={N}, qubit={qubit}")

                steps = int(T / dt)
                fmt_dt_val = str(dt).rstrip("0").rstrip(".").replace(".", "_", 1)
                qubit_tag = f"Qbts({(0 if ignore_qubit else qubit) + 1}){N}"
                extra_wash = 75 if ignore_washout else washout
                param_name = f"Seed31415_{qubit_tag}_dt{fmt_dt_val}_dpth50_wsht{extra_wash}"
                best_param_file = os.path.join(param_dir, f"bestparams_{param_name}.json")
                base_name = f"N{N}_{qubit_tag}_dt{fmt_dt_val}"

                # Load best params if available
                try:
                    with open(best_param_file, 'r') as f:
                        all_best = json.load(f)
                    best = all_best.get(str(round(dt, 5)), {})
                    print("Loaded best parameters for do_predictions.")
                except (FileNotFoundError, json.JSONDecodeError):
                    best = {}
                    print("No best parameter file found. Using defaults.")

                # Predictor for using the (pretrained) ESN to predict
                predictor_model = ESNPredictor(
                    steps=steps,
                    dt=dt,
                    N=N,
                    qubit=(qubit if not ignore_qubit else 0),
                    history_values=None,
                    washout=washout,
                    batch_size=training_depth,
                    training_depth=training_depth,
                    history_seed=train_seed,        # training dataset seed (not used here for data gen)
                    reservoir_seed=reservoir_seed,  # ESN reservoir seed base (documentary)
                    device=device
                )

                # Load the ESN named after the TRAIN seed (Ex0 by convention) -- for the per-dataset metrics
                model_path = os.path.join(model_dir, f"trainedmodel_Seed{train_seed}_rSeed{reservoir_seed}_{base_name}.pt")
                # if not os.path.exists(model_path):
                #     # fall back to original single-model name if needed
                #     alt_path = os.path.join(model_dir, f"trainedmodel_{N}_{qubit_tag}.pt")
                #     if os.path.exists(alt_path):
                #         model_path = alt_path

                if os.path.exists(model_path):
                    print(f"Loading ESN from {model_path}")
                    esn_single = torch.load(model_path, weights_only=False)
                    esn_single.to(device).eval()
                else:
                    raise FileNotFoundError(
                        f"Expected trained ESN not found:\n  {model_path}\n"
                        "Please run your training/official block first."
                    )

                # Generate/load PREDICTION datasets CACHED exactly like training (using pred_seed)
                predictor_pred = ESNPredictor(
                    steps=steps,
                    dt=dt,
                    N=N,
                    qubit=qubit,                 # generate datasets for this actual qubit
                    history_values=None,
                    washout=washout,
                    batch_size=num_pred,         # number of sequences we want
                    training_depth=num_pred,     # drives how many histories are generated/cached
                    history_seed=pred_seed,      # NEW: prediction datasets seed
                    reservoir_seed=pred_seed,    # not used for caching; harmless
                    device=device
                )
                pred_histories = predictor_pred.prepare_histories()  # list length == num_pred

                # ── Part A: original do_predictions behavior — per-dataset metrics with a single ESN ──
                ax = axs[row_idx][col_idx]
                first_plot_done = False
                records = []

                for d, z_seq in enumerate(pred_histories):
                    pred_d, true_d = predictor_model.predict_sequence(esn_single, np.asarray(z_seq))
                    mae_d = mean_absolute_error(torch.tensor(pred_d), torch.tensor(true_d)).item()
                    rmse_d = float(np.sqrt(np.mean((pred_d - true_d)**2)))

                    records.append({
                        "N": N, "qubit": qubit,
                        "dataset_index": d,
                        "seed": pred_seed + d,  # implicit generation order
                        "MAE": mae_d,
                        "RMSE": rmse_d,
                        "T_effective": len(pred_d)
                    })

                    # Plot ONLY the first dataset
                    if not first_plot_done:
                        times0 = np.arange(len(pred_d)) * dt
                        ax.plot(times0, pred_d, 'o-', markersize=2, label="Prediction (dataset 0)")
                        ax.plot(times0, true_d, '-', linewidth=1.2, label='True ⟨σ_z⟩ (dataset 0)')
                        ax.set_xlim(0, 15)
                        ax.set_title(f"N={N}, Qubit={qubit}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("⟨σ_z⟩")
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

                # ── Part B: NEW — histogram over ESNs with varying rseed (dataset 0 only) ──
                # Find all ESNs trained by official_run for this (N, qubit), any rseed / Ex index
                pattern = os.path.join(model_dir, f"trainedmodel_Seed{train_seed}_rSeed*_{base_name}.pt")
                esn_paths = sorted(glob(pattern))
                if not esn_paths:
                    print(f"[hist] No ESN files found for pattern: {pattern}")
                else:
                    print(f"[hist] Found {len(esn_paths)} ESN files for histogram.")

                    # Use the FIRST prediction dataset as requested
                    if len(pred_histories) == 0:
                        print("[hist] No prediction datasets available; skipping histogram.")
                    else:
                        z_seq0 = np.asarray(pred_histories[0])

                        ln_maes = []
                        raw_maes = []
                        rseed_list = []

                        for pth in esn_paths:
                            try:
                                esn_i = torch.load(pth, weights_only=False)
                                esn_i.to(device).eval()
                                pred_i, true_i = predictor_model.predict_sequence(esn_i, z_seq0)
                                mae_i = mean_absolute_error(torch.tensor(pred_i), torch.tensor(true_i)).item()
                                raw_maes.append(mae_i)
                                # Extract rseed from filename: trainedmodel_Seed{rseed}_...
                                # fall back to None if pattern unexpected
                                rseed_str = os.path.basename(pth).split("_")[1]
                                if rseed_str.startswith("Seed"):
                                    rseed_list.append(int(rseed_str.replace("Seed", "")))
                                else:
                                    rseed_list.append(None)
                            except Exception as e:
                                print(f"[hist] Skipping {pth} due to error: {e}")

                        if raw_maes:
                            # Natural log of MAE; guard against zero
                            raw_maes = np.asarray(raw_maes, dtype=float)
                            ln_maes = np.log(raw_maes + 1e-18)

                            plt.figure(figsize=(6,4))
                            plt.hist(ln_maes, bins='auto')
                            plt.xlabel("ln(MAE)")
                            plt.ylabel("Frequency")
                            plt.title(f"ESN rseed variation — N={N}, q={qubit}, dataset 0")
                            hist_path = os.path.join(model_dir, f"hist_lnMAE_N{N}_q{qubit}_dataset0.png")
                            plt.tight_layout()
                            plt.savefig(hist_path, dpi=150)
                            print(f"[hist] Saved histogram → {hist_path}")

                            # Optional: also write a CSV of (rseed, MAE, lnMAE) for debugging
                            hist_df = pd.DataFrame({
                                "rseed": rseed_list[:len(raw_maes)],
                                "MAE": raw_maes,
                                "ln_MAE": ln_maes
                            })
                            hist_csv = os.path.join(model_dir, f"hist_lnMAE_N{N}_q{qubit}_dataset0.csv")
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




    # render_physics_report("./examples/Heisenberg_Chain/cache/physics_summary.json")
    # scorecard_physics("./examples/Heisenberg_Chain/cache/physics_summary.json")
    plt.show()