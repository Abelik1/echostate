import torch

from .reservoir import Reservoir
from .trainer import Trainer
from .utils import mean_absolute_error, mean_squared_error

class ESN(torch.nn.Module):
    def __init__(self,
                 base_input_dim=1,
                 reservoir_size=100,
                 output_dim=1,
                 feedback=0,
                 spectral_radius=0.9,
                 sparsity=0.1,
                 leak_rate=1.0,
                 input_scaling=1.0,
                 bias_scaling=0.2,
                 ridge_param=1e-6,
                 learning_algo="inv",
                 washout=0,
                 batch_size=1,
                 seed=None):
        super().__init__()
        self.base_input_dim = base_input_dim
        self.output_dim = output_dim
        self.feedback = feedback
        self.input_dim = base_input_dim + feedback * output_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.ridge_param = ridge_param
        self.learning_algo = learning_algo
        self.washout = washout
        self.batch_size = batch_size

        # initialize reservoir and trainer
        self.reservoir = Reservoir(self.input_dim,
                                   reservoir_size,
                                   output_dim,
                                   spectral_radius,
                                   sparsity,
                                   input_scaling,
                                   bias_scaling,
                                   seed)
        self.trainer = Trainer(ridge_param,
                               learning_algo)
        self.W_out = None

    def reset_state(self):
        # return fresh reservoir state
        return torch.zeros(self.reservoir.reservoir_size)

    def fit(self, X_batch, Y_batch):
        """
        X_batch: (B, T, base_input_dim)
        Y_batch: (B, T, output_dim)
        """
        B, T, base_input_dim = X_batch.shape
        device = X_batch.device
        output_dim = self.output_dim

        x = torch.zeros(B, self.reservoir.reservoir_size, device=device)

        state_list = []
        target_list = []

        # prepare previous outputs buffer
        prev_outputs = torch.zeros(B, self.feedback * output_dim, device=device)

        for t in range(T):
            u_base = X_batch[:, t, :]  # (B, base_input_dim)

            if self.feedback > 0:
                if t >= self.feedback:
                    fb = []
                    for j in range(1, self.feedback + 1):
                        fb.append(Y_batch[:, t - j, :])  # use teacher forcing
                    u_fb = torch.cat(fb, dim=1)  # (B, feedback * output_dim)
                else:
                    u_fb = torch.zeros(B, self.feedback * output_dim, device=device)
                u = torch.cat([u_base, u_fb], dim=1)
            else:
                u = u_base

            x = self.reservoir.update_batch(x, u, self.leak_rate)

            if t >= self.washout:
                xb = torch.cat([x, torch.ones(B, 1, device=device)], dim=1)  # bias
                state_list.append(xb)
                target_list.append(Y_batch[:, t, :])

        X_all = torch.cat(state_list, dim=0)  # (B*(T-washout), R+1)
        Y_all = torch.cat(target_list, dim=0)  # (B*(T-washout), output_dim)

        self.W_out = self.trainer.fit(X_all, Y_all)

    def forward(self, inputs):
        """
        Run ESN on one sequence (T, base_input_dim) with autoregressive feedback.
        Returns Tensor (T - washout, output_dim).
        """
        T = inputs.shape[0]
        device = inputs.device

        # start with a 1-D reservoir state
        x = torch.zeros(self.reservoir.reservoir_size, device=device)
        prev_outputs = []
        outputs = []

        for t in range(T):
            u_base = inputs[t]                               # (base_input_dim,)
            # --- build feedback portion ---
            if self.feedback > 0:
                fb = []
                for j in range(1, self.feedback + 1):
                    if len(prev_outputs) >= j:
                        fb.append(prev_outputs[-j])         # each is (output_dim,)
                    else:
                        fb.append(torch.zeros(self.output_dim, device=device))
                u = torch.cat([u_base, *fb], dim=0)         # (input_dim,)
            else:
                u = u_base                                  # (input_dim,)

            # vectorized update_batch handles 1-D just like a batch of 1
            x = self.reservoir.update_batch(x, u, self.leak_rate)  # returns (R,)

            # readout
            bias = torch.tensor([1.0], device=device)      # shape (1,)
            xb   = torch.cat([x, bias], dim=0)             # shape (R+1,)
            y    = self.W_out @ xb                         # (output_dim,)
            prev_outputs.append(y)

            if t >= self.washout:
                outputs.append(y)

        return torch.stack(outputs)                        # (T-washout, output_dim)

    def predict(self, input_list, target_list=None):
        """
        Generate predictions for a batch of input sequences.
        Optionally compute error metrics against target_list.
        Returns predictions: list of Tensors
        metrics (optional): dict with 'mae' and 'mse'
        """
        predictions = []
        for seq in input_list:
            preds = self.forward(seq)
            predictions.append(preds)

        if target_list is not None:
            preds_cat = torch.cat(predictions, dim=0)
            targets_cat = torch.cat([t[self.washout:] for t in target_list], dim=0)
            mae = mean_absolute_error(preds_cat, targets_cat)
            mse = mean_squared_error(preds_cat, targets_cat)
            return predictions, {'mae': mae.item(), 'mse': mse.item()}

        return predictions

    @staticmethod
    def tune(input_list,
         target_list,
         n_trials=50,
         direction="minimize",
         study_name=None,
         washout=0,
         seed=None,
         reservoir_limit=200,
         spectral_radius_limit=0.9,
         feedback_limit = 0,
         sparsity_limit = 0.1,
         leak_rate_limit = 1.0,
         input_scaling_limit= 1.0,
         bias_scaling_limit=0.2,
         ridge_param_limit=  1e-6,
         learning_algo="inv",
         **study_kwargs):
        
        import optuna


        def objective(trial):
            # Suggest or use fixed hyperparameters
            reservoir_size = trial.suggest_int("reservoir_size", *reservoir_limit) if isinstance(reservoir_limit, list) else reservoir_limit
            spectral_radius = trial.suggest_float("spectral_radius", *spectral_radius_limit) if isinstance(spectral_radius_limit, list) else spectral_radius_limit
            feedback = trial.suggest_int("feedback", *feedback_limit) if isinstance(feedback_limit, list) else feedback_limit
            sparsity = trial.suggest_float("sparsity", *sparsity_limit) if isinstance(sparsity_limit, list) else sparsity_limit
            leak_rate = trial.suggest_float("leak_rate", *leak_rate_limit) if isinstance(leak_rate_limit, list) else leak_rate_limit
            input_scaling = trial.suggest_float("input_scaling", *input_scaling_limit) if isinstance(input_scaling_limit, list) else input_scaling_limit
            bias_scaling = trial.suggest_float("bias_scaling", *bias_scaling_limit) if isinstance(bias_scaling_limit, list) else bias_scaling_limit
            ridge_param = trial.suggest_float("ridge_param", *ridge_param_limit, log=True) if isinstance(ridge_param_limit, list) else ridge_param_limit


            model = ESN(
                base_input_dim=input_list[0].shape[1],
                reservoir_size=reservoir_size,
                output_dim=target_list[0].shape[1],
                feedback=feedback,
                spectral_radius=spectral_radius,
                sparsity=sparsity,
                leak_rate=leak_rate,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                ridge_param=ridge_param,
                learning_algo=learning_algo,
                washout=washout,
                batch_size=len(input_list),
                seed=seed
            )

            model.fit(input_list, target_list)
            _, metrics = model.predict(input_list, target_list)
            return metrics['mae']

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=f"sqlite:///examples/Heisenberg_Chain/trained_esns/{study_name}.db" if study_name else None,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials, **study_kwargs)
        return study
