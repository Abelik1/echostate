from __future__ import annotations
import math
from typing import Callable, Sequence
import torch

try:
    import optuna
except Exception as e:
    raise RuntimeError("optuna is required for tuning. Install with `pip install optuna`.") from e

from ..core.metrics import mean_absolute_error, mean_squared_error

def run_optuna_tuning(
    *,
    build_model: Callable[[dict], torch.nn.Module],
    inputs: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    space,
    n_trials: int = 50,
    direction: str = "minimize",
    study_name: str | None = None,
    storage: str | None = None,
    device: torch.device | None = None,
    washout: int | None = None,
    algo: str = "inv",
    seed: int | None = 31415,
    eval_subset: int | None = 10,
    **study_kwargs,
):
    """
    Generic tuner: you provide a build_model(config)->ESN factory, data, and a search space object.
    Returns (study, best_config).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    X_tensor = torch.stack(list(inputs), dim=0).to(device)
    Y_tensor = torch.stack(list(targets), dim=0).to(device)

    def suggest(trial):
        cfg = {}
        # ints
        rs_min, rs_max = space.reservoir_size
        cfg["reservoir_size"] = trial.suggest_int("reservoir_size", rs_min, rs_max)
        fb_min, fb_max = space.feedback
        cfg["feedback"] = trial.suggest_int("feedback", fb_min, fb_max)
        # floats
        sr_min, sr_max = space.spectral_radius
        cfg["spectral_radius"] = trial.suggest_float("spectral_radius", sr_min, sr_max)
        sp_min, sp_max = space.sparsity
        cfg["sparsity"] = trial.suggest_float("sparsity", sp_min, sp_max)
        lr_min, lr_max = space.leak_rate
        cfg["leak_rate"] = trial.suggest_float("leak_rate", lr_min, lr_max)
        is_min, is_max = space.input_scaling
        cfg["input_scaling"] = trial.suggest_float("input_scaling", is_min, is_max)
        bs_min, bs_max = space.bias_scaling
        cfg["bias_scaling"] = trial.suggest_float("bias_scaling", bs_min, bs_max)
        rp_min, rp_max = space.ridge_param
        cfg["ridge_param"] = trial.suggest_float("ridge_param", rp_min, rp_max, log=True)
        return cfg

    def objective(trial):
        cfg = suggest(trial)
        # Base dims taken from data
        cfg.update({
            "base_input_dim": X_tensor.shape[-1],
            "output_dim": Y_tensor.shape[-1],
            "washout": (washout if washout is not None else 50),
            "learning_algo": algo,
            "seed": seed,
            "device": device,
        })
        model = build_model(cfg)
        model.fit(X_tensor, Y_tensor)

        k = eval_subset or len(inputs)
        preds, metrics = model.predict(list(inputs)[:k], list(targets)[:k])
        # Log-friendly objective (stabilize scale)
        return math.log1p(metrics["mae"])

    study = optuna.create_study(direction=direction, study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, **study_kwargs)

    best_cfg = study.best_trial.params
    best_cfg.update({
        "base_input_dim": X_tensor.shape[-1],
        "output_dim": Y_tensor.shape[-1],
        "washout": (washout if washout is not None else 50),
        "learning_algo": algo,
        "seed": seed,
        "device": str(device),
    })
    return study, best_cfg
