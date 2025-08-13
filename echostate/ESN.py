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

    def _ensure_batch_bias(self, size: int):
        """
        Return a bias column of shape (size, 1) on the correct device.
        Rebuilds if the cached buffer isn't the requested size.
        """
        if (self._batch_bias is None) or (self._batch_bias.shape[0] != size):
            self._batch_bias = torch.ones(size, 1, device=self.device) * self.bias_scaling
        return self._batch_bias

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor, *, debug: bool = False) -> torch.Tensor:
        """
        Teacher-forced training with explicit *per-series* reset and washout.
        For each series b:
        - reset x and feedback
        - run 'washout' steps (not collected)
        - collect states/targets for t in [washout, T-1]
        Solves a single ridge regression on the stacked design matrix.
        """
        # Accept list[(T,D)] or tensor (B,T,D)
        if isinstance(inputs, list):
            inputs  = torch.stack(inputs,  dim=0)
            targets = torch.stack(targets, dim=0)

        B, T, _ = inputs.shape
        X = inputs.to(self.device)
        Y = targets.to(self.device)

        if self.washout >= T:
            raise ValueError(f"washout ({self.washout}) must be < sequence length T ({T}).")

        state_rows = []
        target_rows = []

        if debug:
            print(f"[ESN.fit] B={B}, T={T}, base_in={self.base_input_dim}, out_dim={self.output_dim}, "
                f"reservoir={self.reservoir_size}, washout={self.washout}, leak={self.leak_rate}, fb={self.feedback}")

        # ---- loop over series; hard reset per-series ----
        for b in range(B):
            # reset reservoir state and feedback for THIS series
            x = torch.zeros(1, self.reservoir_size, device=self.device)
            prev_fb = torch.zeros(1, self.feedback * self.output_dim, device=self.device) if self.feedback > 0 else None

            # 1) washout (not collected)
            for t in range(self.washout):
                u_base = X[b:b+1, t, :]                       # (1, base_in)
                if self.feedback > 0:
                    u = torch.cat([u_base, prev_fb], dim=1)   # (1, base_in + fb*out_dim)
                else:
                    u = u_base
                x = self.reservoir.update_batch(x, u, self.leak_rate)
                if self.feedback > 0:
                    # NOTE: preserves your current "input-as-feedback" choice.
                    prev_fb = torch.cat([prev_fb[:, self.output_dim:], u_base], dim=1)

            if debug and (b < 3 or b == B-1):  # sample a few series
                xm = float(x.abs().mean().item())
                xs = float(x.std(unbiased=False).item())
                print(f"[ESN.fit] After washout — series {b}: mean|x|={xm:.3e}, std(x)={xs:.3e}")

            # 2) collect (t = washout .. T-1)
            for t in range(self.washout, T):
                u_base = X[b:b+1, t, :]
                if self.feedback > 0:
                    u = torch.cat([u_base, prev_fb], dim=1)
                else:
                    u = u_base

                x = self.reservoir.update_batch(x, u, self.leak_rate)

                xb = torch.cat([x, self._ensure_batch_bias(1)], dim=1)  # (1, res+1)
                state_rows.append(xb)
                target_rows.append(Y[b:b+1, t, :])                      # aligned within series

                if self.feedback > 0:
                    prev_fb = torch.cat([prev_fb[:, self.output_dim:], u_base], dim=1)

        # Stack and solve
        X_all = torch.cat(state_rows,  dim=0)  # (#rows, res+1)
        Y_all = torch.cat(target_rows, dim=0)  # (#rows, out_dim)

        if debug:
            has_nan = torch.isnan(X_all).any() or torch.isnan(Y_all).any()
            has_inf = torch.isinf(X_all).any() or torch.isinf(Y_all).any()
            print(f"[ESN.fit] Design rows={X_all.shape[0]}, features={X_all.shape[1]}; "
                f"NaN? {bool(has_nan)}, Inf? {bool(has_inf)}")

        self.W_out = self.trainer.fit(X_all, Y_all)

        if debug:
            try:
                self.trainer.debug_covariance()
            except Exception as e:
                print(f"[ESN.fit][warn] covariance debug failed: {e}")
            w_norm = float(self.W_out.norm().item())
            w_max  = float(self.W_out.abs().max().item())
            print(f"[ESN.fit] W_out: ||W||_F={w_norm:.3e}, max|W|={w_max:.3e}")

        return self.W_out



    def forward(self, inputs: torch.Tensor, *, closed_loop_after_washout: bool = True, debug: bool = False) -> torch.Tensor:
        assert self.W_out is not None, "Model must be fit before forward()"
        T, _ = inputs.shape
        X = inputs.to(self.device)

        x = torch.zeros(self.reservoir_size, device=self.device)
        preds = []
        prev_fb = torch.zeros(self.feedback * self.output_dim, device=self.device) if self.feedback > 0 else None
        last_y = None

        if debug:
            print(f"[ESN.forward] T={T}, washout={self.washout}, closed_after={closed_loop_after_washout}")

        for t in range(T):
            use_closed = closed_loop_after_washout and (t >= self.washout) and (last_y is not None)
            u_base = last_y if use_closed else X[t:t+1, :]

            if self.feedback > 0:
                u = torch.cat([u_base, prev_fb.unsqueeze(0)], dim=1)
            else:
                u = u_base

            x = self.reservoir.update_batch(x, u, self.leak_rate)
            xb = torch.cat([x, self._ensure_batch_bias(1)], dim=1)    # bias is (1,1)
            y = xb @ self.W_out.T
            preds.append(y.squeeze(0))

            if self.feedback > 0:
                # same "input-as-feedback" convention as fit()
                prev_fb = torch.cat([prev_fb[self.output_dim:], u_base.squeeze(0)], dim=0)

            last_y = y

            if debug and t in (self.washout-1, self.washout, self.washout+1):
                print(f"[ESN.forward] t={t} {'(switch)' if use_closed else '(warmup)'} "
                    f"| mean|x|={float(x.abs().mean()):.3e}, y.max={float(y.abs().max()):.3e}")

        out = torch.stack(preds, dim=0)
        if debug:
            op = out[self.washout:]
            print(f"[ESN.forward] post-washout preds: shape={op.shape}, "
                f"max|pred|={float(op.abs().max()):.3e}, mean|pred|={float(op.abs().mean()):.3e}")
        return out[self.washout:]


    def predict(self, input_list, target_list=None):
        import matplotlib.pyplot as plt
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
            if False:
                 # Plot the first `plot_limit` sequences individually
                plot_limit =2
                for idx in range(min(plot_limit, len(preds))):
                    pred_seq = preds[idx].cpu().numpy()
                    targ_seq = target_list[idx][self.washout:].cpu().numpy()

                    plt.figure(figsize=(8, 3))
                    plt.plot(targ_seq, label=f"Target seq {idx}", lw=1.2)
                    plt.plot(pred_seq, label=f"Prediction seq {idx}", lw=1.0)
                    plt.legend()
                    plt.title(f"Prediction vs Target (sequence {idx}, post-washout)")
                    plt.xlabel("Time steps")
                    plt.ylabel("Value")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()
            return preds, {
                'mae': mean_absolute_error(P, T).item(),
                'mse': mean_squared_error(P, T).item()
            }
        return preds
#region Tune
    @staticmethod
    def tune(input_list,
             target_list,
             device=None,
             n_trials=50,
             direction='minimize',
             study_name=None,
             study_loc=None,
             washout=0,
             seed=31415, #TODO Set to NONE
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
                seed=seed
            ).to(device)
            # training and evaluation
            # X_batch = torch.stack(input_list, dim=0)
            # Y_batch = torch.stack(target_list, dim=0)
            

            model.fit(X_tensor, Y_tensor)
            _, metrics = model.predict(input_list, target_list)
            return metrics['mae'] #mae

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=f'sqlite:///{study_loc}{study_name}.db' if study_name else None,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=n_trials, **study_kwargs)
        return study