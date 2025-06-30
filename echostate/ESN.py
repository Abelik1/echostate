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

    def fit(self, input_list, target_list):
        """
        Train ESN on a batch of sequences with teacher-forced feedback.
        input_list: list of Tensors, each (T, base_input_dim)
        target_list: list of Tensors, each (T, output_dim)
        """
        all_states = []
        all_targets = []

        for seq_idx in range(self.batch_size):
            base_inputs = input_list[seq_idx]
            targets = target_list[seq_idx]
            T = base_inputs.shape[0]
            x = self.reset_state()
            states = []

            for t in range(T):
                # build input with feedback
                u_base = base_inputs[t]
                if self.feedback > 0:
                    fb_vals = []
                    for j in range(1, self.feedback+1):
                        idx = t - j
                        if idx >= 0:
                            fb_vals.append(targets[idx])
                        else:
                            fb_vals.append(torch.zeros(self.output_dim, device=u_base.device))
                    u = torch.cat([u_base] + fb_vals, dim=0)
                else:
                    u = u_base
                x = self.reservoir.update(x, u, self.leak_rate)
                if t >= self.washout:
                    states.append(x.clone())
                    
            states = torch.stack(states)  # Now it's a tensor
            bias = torch.ones(states.size(0), 1, device=states.device)
            states = torch.cat([states, bias], dim=1)

            all_states.append(states)  # No need to stack again
            all_targets.append(targets[self.washout:])
        
        
        X = torch.cat(all_states, dim=0)
        Y = torch.cat(all_targets, dim=0)
        self.W_out = self.trainer.fit(X, Y)

    def forward(self, inputs):
        """
        Run ESN on a single sequence with autoregressive feedback.
        inputs: Tensor (T, base_input_dim)
        returns: Tensor (T-washout, output_dim)
        """
        T = inputs.shape[0]
        x = self.reset_state()
        prev_outputs = []
        outputs = []

        for t in range(T):
            u_base = inputs[t]
            if self.feedback > 0:
                fb_vals = []
                for j in range(1, self.feedback+1):
                    if len(prev_outputs) >= j:
                        fb_vals.append(prev_outputs[-j])
                    else:
                        fb_vals.append(torch.zeros(self.output_dim, device=u_base.device))
                u = torch.cat([u_base] + fb_vals, dim=0)
            else:
                u = u_base
            x = self.reservoir.update(x, u, self.leak_rate)
            xb = torch.cat([x, torch.tensor([1.0], device=x.device)])
            y  = self.W_out @ xb
            prev_outputs.append(y)
            if t >= self.washout:
                outputs.append(y)

        return torch.stack(outputs)

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
            study_name = None,
            washout=0,
            seed = None,
            reservoir_limit = [50,1000],
            spectral_radius_limit = [0.1, 1.4],
            feedback_limit = [0, 5],
            input_scaling_limit = [0.05, 5.0],
            ridge_param_limit = [1e-7, 1.0],
            leak_rate_limit = [0.1, 1.0],
            sparsity_limit = [0.05, 1.0],
            **study_kwargs):
        import optuna
        
        
        """
        Hyperparameter tuning using Optuna.
        Returns the Optuna Study object.
        """
        def objective(trial):
            reservoir_size = trial.suggest_int("reservoir_size", reservoir_limit[0], reservoir_limit[1])
            spectral_radius = trial.suggest_float("spectral_radius", spectral_radius_limit[0],spectral_radius_limit[1])
            feedback = trial.suggest_int("feedback", feedback_limit[0], feedback_limit[1])
            input_scaling = trial.suggest_float("input_scaling", input_scaling_limit[0], input_scaling_limit[1])
            ridge_param = trial.suggest_float("ridge_param", ridge_param_limit[0], ridge_param_limit[1], log=True)
            leak_rate = trial.suggest_float("leak_rate", leak_rate_limit[0], leak_rate_limit[1])
            sparsity = trial.suggest_float("sparsity", sparsity_limit[0], sparsity_limit[1])

            model = ESN(base_input_dim=input_list[0].shape[1],
                        reservoir_size=reservoir_size,
                        output_dim=target_list[0].shape[1],
                        feedback=feedback,
                        spectral_radius=spectral_radius,
                        sparsity=sparsity,
                        input_scaling=input_scaling,
                        ridge_param=ridge_param,
                        leak_rate=leak_rate,
                        washout= washout,
                        batch_size=len(input_list),
                        seed = seed)
                        
            model.fit(input_list, target_list)
            _, metrics = model.predict(input_list, target_list)
            return metrics['mae']

        study = optuna.create_study(direction=direction, study_name=study_name,
                storage=f"sqlite:///examples/Heisenberg_Chain/trained_esns/{study_name}.db", #TODO REMOVE THE PATH
                load_if_exists=True)
        study.optimize(objective, n_trials=n_trials, **study_kwargs)
        return study