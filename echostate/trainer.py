import torch
import numpy as np

class Trainer:
    def __init__(self, ridge_param=1e-6, learning_algo="inv"):
        self.ridge_param = ridge_param
        self.learning_algo = learning_algo
        self.xTx = None
        self.xTy = None

    def fit(self, X, Y):
        # accumulate covariance
        Xt = X.T
        self.xTx = Xt @ X
        self.xTy = Xt @ Y
        I = torch.eye(self.xTx.shape[0], device=self.xTx.device)

        if self.learning_algo == "inv":
            W_out = (torch.linalg.inv(self.xTx + self.ridge_param * I) @ self.xTy).T
        else:
            raise NotImplementedError(f"Learning algorithm '{self.learning_algo}' not implemented.")
        return W_out

    def debug_covariance(self):
        # inspect covariance for conditioning issues
        cov = self.xTx
        cov_arr = cov.cpu().numpy()
        rank = np.linalg.matrix_rank(cov_arr)
        cond = np.linalg.cond(cov_arr)
        print(f"[DEBUG] covariance shape: {cov_arr.shape}, rank: {rank}, condition #: {cond:.2e}")