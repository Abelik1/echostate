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

    def fit(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        I = torch.eye(X.shape[1], device=self.device)
        self.xTx = X.T @ X
        self.xTy = X.T @ Y

        if self.learning_algo == "inv":
            return (torch.linalg.inv(self.xTx + self.ridge_param * I) @ self.xTy).T
        elif self.learning_algo == "cholesky":
            chol = torch.linalg.cholesky(self.xTx + self.ridge_param * I)
            return torch.cholesky_solve(self.xTy, chol).T
        else:
            raise NotImplementedError(f"Learning algorithm '{self.learning_algo}' not implemented.")


    def debug_covariance(self):
        s = torch.linalg.svdvals(self.xTx)
        tol = s.max() * max(self.xTx.shape) * torch.finfo(s.dtype).eps
        rank = int((s > tol).sum().item())
        cond = (s.max() / s.min()).item()
        print(f"[DEBUG] covariance shape: {tuple(self.xTx.shape)}, rank: {rank}, condition #: {cond:.2e}")
