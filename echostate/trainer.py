import torch

class Trainer:
    def __init__(self, ridge_param: float = 1e-6,
                 learning_algo: str = 'inv',
                 device: torch.device = torch.device('cpu')):
        self.ridge_param = ridge_param
        self.learning_algo = learning_algo
        self.device = device
        self.xTx = None
        self.xTy = None
        self._I = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X: (B, T, features) or (N, features)
        X = X.to(self.device)
        Y = Y.to(self.device)

        # flatten (B,T) → (B*T)
        if X.dim() == 3:
            B, T, D = X.shape
            X_flat = X.reshape(B * T, D)
            Y_flat = Y.reshape(B * T, -1)
        else:
            X_flat, Y_flat = X, Y

        # accumulate covariance
        Xt = X_flat.transpose(0, 1)       # (features, N)
        self.xTx = Xt @ X_flat            # (features, features)
        self.xTy = Xt @ Y_flat            # (features, output_dim)

        # cache identity
        n = self.xTx.shape[0]
        if self._I is None or self._I.shape[0] != n:
            self._I = torch.eye(n, device=self.device, dtype=self.xTx.dtype)

            
        A = self.xTx + self.ridge_param * self._I

        # solve A W = xTy
        if self.learning_algo == 'cholesky':
            # try Cholesky (fast & stable if A is SPD)
            try:
                chol = torch.linalg.cholesky(A)
                W = torch.cholesky_solve(self.xTy, chol)
            except RuntimeError:
                # fallback to general solver if not PD
                W = torch.linalg.solve(A, self.xTy)
        elif self.learning_algo == 'solve':
            # always use general solver
            W = torch.linalg.solve(A, self.xTy)
        elif self.learning_algo == 'inv':
            W = (torch.linalg.inv(self.xTx + self.ridge_param * self._I) @ self.xTy)
        else:
            raise NotImplementedError(f"Learning algorithm '{self.learning_algo}' not implemented.")

        return W.T  # (output_dim, features)

    def debug_covariance(self):
        s = torch.linalg.svdvals(self.xTx)
        tol = s.max() * max(self.xTx.shape) * torch.finfo(s.dtype).eps
        rank = int((s > tol).sum().item())
        cond = (s.max() / s.min()).item()
        print(f"[DEBUG] covariance shape: {tuple(self.xTx.shape)}, rank: {rank}, condition #: {cond:.2e}")
