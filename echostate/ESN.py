import torch
from .reservoir import Reservoir
from .trainer import Trainer
from .utils import mean_absolute_error, mean_squared_error

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
        learning_algo: str = 'inv',  # 'inv' default for speed
        leak_rate: float = 1.0,
        washout: int = 50,
        batch_size: int = None,
        seed: int = None,
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

    def _ensure_batch_bias(self, B):
        size = self.batch_size or B
        if self._batch_bias is None or self._batch_bias.shape[0] != size:
            bias = torch.ones(size, 1, device=self.device) * self.bias_scaling
            self._batch_bias = bias
        return self._batch_bias

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced training: vectorized state collection, single covariance solve.
        inputs: (B, T, base_input_dim) or list of seqs
        targets: (B, T, output_dim)
        """
        # stack list input
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
            targets = torch.stack(targets, dim=0)

        B, T, _ = inputs.shape
        # move once to device + scale
        X = inputs.to(self.device) * self.input_scaling
        Y = targets.to(self.device)

        # initial reservoir state
        x = torch.zeros(B, self.reservoir_size, device=self.device)
        # initialize feedback
        if self.feedback > 0:
            prev_fb = torch.zeros(B, self.feedback * self.output_dim, device=self.device)
        else:
            prev_fb = None

        # collect flattened states/targets
        state_list = []  # each entry (B, res+1)
        target_list = []
        bias_vec = None

        for t in range(T):
            u_base = X[:, t, :]
            # feedback concat
            if self.feedback > 0:
                u = torch.cat([u_base, prev_fb], dim=1)
            else:
                u = u_base

            # reservoir update
            x = self.reservoir.update_batch(x, u, self.leak_rate)

            # after washout, collect state
            if t >= self.washout:
                bias_vec = self._ensure_batch_bias(B)
                xb = torch.cat([x, bias_vec], dim=1)  # (B, res+1)
                state_list.append(xb)
                target_list.append(Y[:, t, :])

            # update feedback teacher-forced
            if self.feedback > 0:
                prev_fb = torch.cat([
                    prev_fb[:, self.output_dim:],
                    Y[:, t, :]
                ], dim=1)

        # concatenate all steps
        X_all = torch.cat(state_list, dim=0)  # (B*(T-washout), res+1)
        Y_all = torch.cat(target_list, dim=0)  # (B*(T-washout), output_dim)

        # solve for output weights in one go
        self.W_out = self.trainer.fit(X_all, Y_all)
        return self.W_out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Auto-regressive generation: predict sequence with feedback.
        inputs: (T, base_input_dim)
        returns: (T - washout, output_dim)
        """
        assert self.W_out is not None, "Model must be fit before forward()"
        T, _ = inputs.shape
        X = inputs.to(self.device) * self.input_scaling

        x = torch.zeros(self.reservoir_size, device=self.device)
        preds = []
        prev_fb = torch.zeros(self.feedback * self.output_dim, device=self.device) if self.feedback>0 else None

        for t in range(T):
            u_base = X[t:t+1, :]
            if self.feedback>0:
                u = torch.cat([u_base, prev_fb.unsqueeze(0)], dim=1)
            else:
                u = u_base

            x = self.reservoir.update_batch(x, u, self.leak_rate)
            bias_vec = self._ensure_batch_bias(1)
            xb = torch.cat([x, bias_vec[:1]], dim=1)
            y = xb @ self.W_out.T
            preds.append(y.squeeze(0))

            if self.feedback>0:
                prev_fb = torch.cat([prev_fb[self.output_dim:], y.squeeze(0)], dim=0)

        # trim initial washout if desired
        out = torch.stack(preds, dim=0)
        return out[self.washout:]

    def predict(self, input_list, target_list=None):
        """
        Batch-predict with optional metrics.
        """
        preds = []
        with torch.no_grad():
            for seq in input_list:
                out = self.forward(seq)
                preds.append(out)

        if target_list is not None:
            P = torch.cat(preds, dim=0)
            T = torch.cat([t[self.washout:] for t in target_list], dim=0)
            return preds, {
                'mae': mean_absolute_error(P, T).item(),
                'mse': mean_squared_error(P, T).item()
            }
        return preds

    @staticmethod
    def tune(input_list,
             target_list,
             device=None,
             n_trials=50,
             direction='minimize',
             study_name=None,
             study_loc=None,
             washout=0,
             seed=None,
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

            model = ESN(
                device=device,
                base_input_dim=input_list[0].shape[1],
                reservoir_size=reservoir_size,
                output_dim=target_list[0].shape[1],
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
                seed=seed
            )
            # training and evaluation
            X_batch = torch.stack(input_list, dim=0)
            Y_batch = torch.stack(target_list, dim=0)
            model.fit(X_batch, Y_batch)
            _, metrics = model.predict(input_list, target_list)
            return metrics['mae']

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=f'sqlite:///{study_loc}{study_name}.db' if study_name else None,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials, **study_kwargs)
        return study