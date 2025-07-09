import torch
from .reservoir import Reservoir
from .trainer import Trainer
from .utils import mean_absolute_error, mean_squared_error

class ESN(torch.nn.Module):
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
        learning_algo: str = 'cholesky',
        leak_rate: float = 1.0,
        washout: int = 50,
        batch_size: int = None,
        seed: int = None,
    ):
        super().__init__()
        # device + optional seeding
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is not None:
            torch.manual_seed(seed)

        # dimensions & hyper-parameters
        self.base_input_dim = base_input_dim
        self.feedback      = feedback             # number of past outputs to feed back
        self.input_dim     = base_input_dim + feedback * output_dim
        self.res_size      = reservoir_size
        self.output_dim    = output_dim
        self.input_scaling = input_scaling
        self.bias_scaling  = bias_scaling
        self.leak_rate     = leak_rate
        self.washout       = washout
        self.batch_size    = batch_size
        self.learning_algo = learning_algo

        # reservoir + trainer
        self.reservoir = Reservoir(
            input_dim=self.input_dim,
            reservoir_size=self.res_size,
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            input_scaling=self.input_scaling,
            bias_scaling=self.bias_scaling,
            seed=seed,
            device=self.device,
        )
        self.trainer = Trainer(
            ridge_param=ridge_param,
            learning_algo=learning_algo,
            device=self.device
        )

        # readout
        self.W_out = None
        self._batch_bias = None

    def _ensure_batch_bias(self, B):
        size = self.batch_size or B
        if self._batch_bias is None or self._batch_bias.shape[0] != size:
            bias = torch.ones(size, 1, device=self.device) * self.bias_scaling
            self._batch_bias = bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Auto-regressive generation: feeds back predictions instead of true targets.
        inputs: (T, base_input_dim)
        returns: (T, output_dim)
        """
        # ensure trained
        assert self.W_out is not None, "Model must be fit before calling forward()"

        T, _ = inputs.shape
        inputs = inputs.to(self.device) * self.input_scaling

        # initial states
        x = torch.zeros(1, self.res_size, device=self.device)
        if self.feedback > 0:
            prev_y = torch.zeros(1, self.feedback * self.output_dim, device=self.device)
        else:
            prev_y = None

        outputs = []
        for t in range(T):
            u_ext = inputs[t:t+1, :]
            if self.feedback > 0:
                u = torch.cat([u_ext, prev_y], dim=1)
            else:
                u = u_ext

            x = self.reservoir.update_batch(x, u, self.leak_rate)
            # append bias, compute readout
            self._ensure_batch_bias(1)
            xb = torch.cat([x, self._batch_bias[:1]], dim=1)  # (1, res_size+1)
            y = xb @ self.W_out.T                             # (1, output_dim)
            outputs.append(y)

            if self.feedback > 0:
                prev_y = torch.cat([prev_y[:, self.output_dim:], y], dim=1)

        return torch.cat(outputs, dim=0)

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced training: uses the true target at each step for feedback.
        inputs: (B, T, base_input_dim)
        targets: (B, T, output_dim)
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
            targets = torch.stack(targets, dim=0)
        B, T, _ = inputs.shape
        inputs = inputs.to(self.device) * self.input_scaling
        targets = targets.to(self.device)

        x = torch.zeros(B, self.res_size, device=self.device)
        # initialize feedback buffer if needed
        if self.feedback > 0:
            prev_targets = torch.zeros(B, self.feedback * self.output_dim, device=self.device)
        else:
            prev_targets = None
        states = []

        for t in range(T):
            u_ext = inputs[:, t, :]
            # concatenate sliding window of past targets
            if self.feedback > 0:
                u = torch.cat([u_ext, prev_targets], dim=1)
            else:
                u = u_ext

            # update reservoir
            x = self.reservoir.update_batch(x, u, self.leak_rate)

            # collect state after washout
            if t >= self.washout:
                self._ensure_batch_bias(B)
                xb = torch.cat([x, self._batch_bias[:B]], dim=1)
                states.append(xb)

            # teacher-forcing: slide buffer window
            if self.feedback > 0:
                # drop oldest feedback, append current target
                new_fb = targets[:, t, :]
                prev_targets = torch.cat([
                    prev_targets[:, self.output_dim:],
                    new_fb
                ], dim=1)

        # stack states and train readout
        Xb = torch.stack(states, dim=1)                     # (B, T-washout, res_size+1)
        Yb = targets[:, self.washout:, :]                   # (B, T-washout, output_dim)
        self.W_out = self.trainer.fit(Xb, Yb)               # (output_dim, res_size+1)
        return self.W_out

    def predict(self, input_list, target_list=None):
        """
        Generate predictions with washout removed.
        input_list: list of (T, base_input_dim) sequences
        target_list: optional list of (T, output_dim) sequences
        Returns preds (list of (T-washout, output_dim)) or (preds, metrics)
        """
        # compute full outputs then trim initial washout
        trimmed_preds = []
        for seq in input_list:
            out = self.forward(seq)
            trimmed_preds.append(out[self.washout:])

        if target_list is not None:
            trimmed_targets = [t[self.washout:] for t in target_list]
            # concat for metrics
            P = torch.cat(trimmed_preds, dim=0)
            T = torch.cat(trimmed_targets, dim=0)
            return trimmed_preds, {
                'mae': mean_absolute_error(P, T).item(),
                'mse': mean_squared_error(P, T).item()
            }
        return trimmed_preds
        

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
             learning_algo='cholesky',
             **study_kwargs):
        import optuna

        def objective(trial):
            reservoir_size = trial.suggest_int('reservoir_size', *reservoir_limit) if isinstance(reservoir_limit, list) else reservoir_limit
            spectral_radius = trial.suggest_float('spectral_radius', *spectral_radius_limit) if isinstance(spectral_radius_limit, list) else spectral_radius_limit
            feedback = trial.suggest_int('feedback', *feedback_limit) if isinstance(feedback_limit, list) else feedback_limit
            sparsity = trial.suggest_float('sparsity', *sparsity_limit) if isinstance(sparsity_limit, list) else sparsity_limit
            leak_rate = trial.suggest_float('leak_rate', *leak_rate_limit) if isinstance(leak_rate_limit, list) else leak_rate_limit
            input_scaling = trial.suggest_float('input_scaling', *input_scaling_limit) if isinstance(input_scaling_limit, list) else input_scaling_limit
            bias_scaling = trial.suggest_float('bias_scaling', *bias_scaling_limit) if isinstance(bias_scaling_limit, list) else bias_scaling_limit
            ridge_param = trial.suggest_float('ridge_param', *ridge_param_limit, log = True) if isinstance(ridge_param_limit, list) else ridge_param_limit

            model = ESN(device=device,
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
                        seed=seed)
            X_batch = torch.stack(input_list, dim=0).to(device)
            Y_batch = torch.stack(target_list, dim=0).to(device)
            model.fit(X_batch, Y_batch)
            with torch.no_grad():
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