# echostate/ESN.py
import logging
import torch
from .reservoir import Reservoir
from .trainer import Trainer
from .utils import mean_absolute_error, mean_squared_error
from .logging_utils import log_tensor, tensor_stats

LOGGER = logging.getLogger(__name__)

class ESN(torch.nn.Module):
    """
    Echo State Network with optimized training path (vectorized covariance, inv solver,
    minimal device transfers, preallocated bias/feedback).
    """
    def __init__(
        self,
        device=None,
        base_input_dim: int = None,
        reservoir_size: int = None,
        output_dim: int = None,
        feedback: int = 0,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scaling: float = 1.0,
        bias_scaling: float = 0.2,
        ridge_param: float = 1e-6,
        learning_algo: str = 'inv',
        leak_rate: float = 1.0,
        washout: int = 50,
        batch_size: int = None,
        seed: int = None,
        step_log_every: int | None = None,
    ):
        super().__init__()
        # device + optional seeding
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        if seed is not None:
            torch.manual_seed(seed)

        # hyperparameters
        self.base_input_dim = base_input_dim
        self.output_dim = output_dim
        self.feedback = feedback
        self.input_dim = base_input_dim + feedback * output_dim
        self.reservoir_size = reservoir_size
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.leak_rate = leak_rate
        self.washout = washout
        self.batch_size = batch_size
        self.learning_algo = learning_algo
        self.step_log_every = step_log_every
        # components
        self.reservoir = Reservoir(
            input_dim=self.input_dim,
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            seed=seed,
            device=self.device,
        )
        self.trainer = Trainer(
            ridge_param=ridge_param,
            learning_algo=learning_algo,
            device=self.device,
        )

        # readout weights and buffers
        self.W_out = None
        self._batch_bias = None

        LOGGER.info("ESN init",
            extra={"extra": {
                "device": str(self.device),
                "base_input_dim": base_input_dim,
                "output_dim": output_dim,
                "input_dim": self.input_dim,
                "reservoir_size": reservoir_size,
                "feedback": feedback,
                "leak_rate": leak_rate,
                "washout": washout,
                "learning_algo": learning_algo,
            }},
        )

    def _ensure_batch_bias(self, batch_sz: int) -> torch.Tensor:
        if self._batch_bias is None or self._batch_bias.shape[0] != batch_sz:
            self._batch_bias = torch.ones(batch_sz, 1, device=self.device) * self.bias_scaling
        return self._batch_bias

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced training with progressive history:
        inputs/targets are assumed to be pre-aligned by the dataset (e.g., X[t] -> Y[t]).
        We collect states after washout and solve a single ridge regression.
        """
        # stack list input
        if isinstance(inputs, list):
            inputs  = torch.stack(inputs,  dim=0)
            targets = torch.stack(targets, dim=0)

        B, T, Din = inputs.shape
        assert T > self.washout, "Need at least washout+1 steps for next-step training."
        X = inputs.to(self.device)
        Y = targets.to(self.device)

        LOGGER.info("ESN.fit start",
            extra={"extra": {"B": B, "T": T, "Din": Din, "washout": self.washout}}
        )
        log_tensor(LOGGER, X, "inputs(batch)", level=logging.DEBUG)
        log_tensor(LOGGER, Y, "targets(batch)", level=logging.DEBUG)

        # initial reservoir state + feedback buffer (holds past outputs)
        x = torch.zeros(B, self.reservoir_size, device=self.device)
        prev_fb = torch.zeros(B, self.feedback * self.output_dim, device=self.device) if self.feedback > 0 else None

        state_list, target_list = [], []

        for t in range(T):
            u_base = X[:, t, :]  # inputs at time t
            if self.feedback > 0:
                u = torch.cat([u_base, prev_fb], dim=1)
            else:
                u = u_base

            x = self.reservoir.update_batch(x, u, self.leak_rate)

            if t >= self.washout:
                bias_vec = self._ensure_batch_bias(x.shape[0])
                xb = torch.cat([x, bias_vec], dim=1)
                state_list.append(xb)
                target_list.append(Y[:, t, :])

            if self.feedback > 0:
                prev_fb = torch.cat([prev_fb[:, self.output_dim:], Y[:, t, :]], dim=1)

            if LOGGER.isEnabledFor(logging.DEBUG):  # or 5 if you keep TRACE
                if self.step_log_every and (t % self.step_log_every != 0):
                    pass  # skip most steps
                else:
                    LOGGER.debug("fit.step",
                        extra={"extra": {
                            "t": t,
                            "post_washout": t >= self.washout,
                            "x_stats": tensor_stats(x),
                        }},
                    )

        # solve readout in one shot
        X_all = torch.cat(state_list,  dim=0)   # (B*(T-washout), res+1)
        Y_all = torch.cat(target_list, dim=0)   # (B*(T-washout), out_dim)

        # Pre-solve diagnostics (dead features / saturation)
        var = X_all[:, :-1].var(dim=0)
        n_dead = int((var < 1e-10).sum().item())
        n_rows = X_all.shape[0]
        n_feat = X_all.shape[1]
        xvals = X_all[:, :-1]
        sat = ((xvals > 0.99) | (xvals < -0.99)).float().mean().item()

        LOGGER.debug("Design matrix diagnostics",
            extra={"extra": {
                "rows": n_rows, "features": n_feat, "dead_features": n_dead,
                "saturation_fraction": float(sat)
            }},
        )
        log_tensor(LOGGER, X_all, "X_all", level=logging.DEBUG)
        log_tensor(LOGGER, Y_all, "Y_all", level=logging.DEBUG)

        self.W_out = self.trainer.fit(X_all, Y_all)

        # Covariance stats (post-solve)
        cov_stats = self.trainer.covariance_stats(safe=True)
        LOGGER.info("Trainer covariance stats", extra={"extra": {"cov_stats": cov_stats}})

        return self.W_out

    def forward(self, inputs: torch.Tensor, *, closed_loop_after_washout: bool = True) -> torch.Tensor:
        """
        Inference with closed-loop after washout.
        Returns predictions trimmed by washout: out[washout:].
        """
        assert self.W_out is not None, "Model must be fit before forward()"
        T, Din = inputs.shape
        X = inputs.to(self.device)

        LOGGER.info("ESN.forward start", extra={"extra": {"T": T, "Din": Din, "washout": self.washout}})

        x = torch.zeros(1, self.reservoir_size, device=self.device)
        preds = []

        prev_fb = torch.zeros(self.feedback * self.output_dim, device=self.device) if self.feedback > 0 else None
        last_y = None

        for t in range(T):
            use_closed = closed_loop_after_washout and (t >= self.washout) and (last_y is not None)
            replace_base = use_closed and (self.base_input_dim == self.output_dim)
            u_base = last_y if replace_base else X[t:t+1, :]

            if self.feedback > 0:
                u = torch.cat([u_base, prev_fb.unsqueeze(0)], dim=1)
            else:
                u = u_base

            x = self.reservoir.update_batch(x, u, self.leak_rate)
            bias_vec = self._ensure_batch_bias(x.shape[0])
            xb = torch.cat([x, bias_vec], dim=1)
            y = xb @ self.W_out.T
            preds.append(y.squeeze(0))

            if self.feedback > 0:
                prev_fb = torch.cat([prev_fb[self.output_dim:], y.squeeze(0)], dim=0)

            last_y = y

            if LOGGER.isEnabledFor(logging.DEBUG):
                if self.step_log_every and (t % self.step_log_every != 0):
                    pass
                else:
                    LOGGER.debug("forward.step",
                        extra={"extra": {"t": t, "use_closed": use_closed, "y_stats": tensor_stats(y)}}
                    )

        out = torch.stack(preds, dim=0)
        return out[self.washout:]

    def predict(self, input_list, target_list=None):
        """
        Batch-predict with optional metrics.
        """
        preds = []
        with torch.no_grad():
            for i, seq in enumerate(input_list):
                out = self.forward(seq)
                preds.append(out)
                if LOGGER.isEnabledFor(5):
                    LOGGER.trace("predict.seq_done", extra={"extra": {"idx": i, "out_stats": tensor_stats(out)}})

        if target_list is not None:
            P = torch.cat(preds, dim=0)
            y_true = torch.cat([t[self.washout:] for t in target_list], dim=0)
            mae = mean_absolute_error(P, y_true).item()
            mse = mean_squared_error(P, y_true).item()
            LOGGER.info("predict.metrics", extra={"extra": {"mae": float(mae), "mse": float(mse)}})
            return preds, {'mae': mae, 'mse': mse}
        return preds

#region Tune
    # ------------ Tune (adds logging) ------------
    @staticmethod
    def tune(input_list,
             target_list,
             device=None,
             n_trials=50,
             direction='minimize',
             study_name=None,
             study_loc=None,
             washout=0,
             seed=31415,
             reservoir_limit=200,
             spectral_radius_limit=0.9,
             feedback_limit=0,
             sparsity_limit=0.1,
             leak_rate_limit=1.0,
             input_scaling_limit=1.0,
             bias_scaling_limit=0.2,
             ridge_param_limit=1e-6,
             learning_algo='inv',
             **study_kwargs):
        import optuna
        LOGGER.info("ESN.tune start",
            extra={"extra": {
                "n_trials": n_trials, "washout": washout,
                "direction": direction, "study_name": study_name
            }},
        )
        X_tensor = torch.stack(input_list, dim=0).to(device)
        Y_tensor = torch.stack(target_list, dim=0).to(device)

        def objective(trial):
            # suggest hyperparams
            reservoir_size = trial.suggest_int('reservoir_size', *reservoir_limit) if isinstance(reservoir_limit, list) else reservoir_limit
            spectral_radius = trial.suggest_float('spectral_radius', *spectral_radius_limit) if isinstance(spectral_radius_limit, list) else spectral_radius_limit
            feedback = trial.suggest_int('feedback', *feedback_limit) if isinstance(feedback_limit, list) else feedback_limit
            sparsity = trial.suggest_float('sparsity', *sparsity_limit) if isinstance(sparsity_limit, list) else sparsity_limit
            leak_rate = trial.suggest_float('leak_rate', *leak_rate_limit) if isinstance(leak_rate_limit, list) else leak_rate_limit
            input_scaling = trial.suggest_float('input_scaling', *input_scaling_limit) if isinstance(input_scaling_limit, list) else input_scaling_limit
            bias_scaling = trial.suggest_float('bias_scaling', *bias_scaling_limit) if isinstance(bias_scaling_limit, list) else bias_scaling_limit
            ridge_param = trial.suggest_float('ridge_param', *ridge_param_limit, log=True) if isinstance(ridge_param_limit, list) else ridge_param_limit

            LOGGER.debug("Trial params",
                extra={"extra": {
                    "trial": trial.number,
                    "reservoir_size": reservoir_size,
                    "spectral_radius": float(spectral_radius),
                    "feedback": int(feedback),
                    "sparsity": float(sparsity),
                    "leak_rate": float(leak_rate),
                    "input_scaling": float(input_scaling),
                    "bias_scaling": float(bias_scaling),
                    "ridge_param": float(ridge_param),
                }},
            )

            model = ESN(
                device=device,
                base_input_dim= input_list[0].shape[1],
                reservoir_size=reservoir_size,
                output_dim= target_list[0].shape[1],
                feedback=feedback,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                leak_rate=leak_rate,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                washout=washout,
                ridge_param=ridge_param,
                learning_algo=learning_algo,
                batch_size=len(input_list),
                seed=seed,
            ).to(device)

            model.fit(X_tensor, Y_tensor)
            _, metrics = model.predict(input_list, target_list)

            LOGGER.info("Trial result",
                extra={"extra": {"trial": trial.number, "mae": float(metrics['mae']), "mse": float(metrics['mse'])}}
            )
            return metrics['mae']

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=f'sqlite:///{study_loc}{study_name}.db' if study_name else None,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials, **study_kwargs)
        return study
