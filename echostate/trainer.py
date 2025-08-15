# echostate/trainer.py
import logging
import torch

LOGGER = logging.getLogger(__name__)

class Trainer:
    def __init__(self, ridge_param: float = 1e-6,
                 learning_algo: str = 'inv',
                 device: torch.device = torch.device('cpu')):
        self.ridge_param = ridge_param
        self.learning_algo = learning_algo
        self.device = device
        self.xTx = None
        self.xTy = None

    def fit(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        I = torch.eye(X.shape[1], device=self.device)
        self.xTx = X.T @ X
        self.xTy = X.T @ Y

        LOGGER.debug("Trainer.fit: formed X^T X and X^T Y",
            extra={"extra": {
                "X_shape": tuple(X.shape),
                "Y_shape": tuple(Y.shape),
                "ridge_param": float(self.ridge_param),
                "algo": self.learning_algo,
            }},
        )

        try:
            if self.learning_algo == "inv":
                W = (torch.linalg.inv(self.xTx + self.ridge_param * I) @ self.xTy).T
            elif self.learning_algo == "cholesky":
                chol = torch.linalg.cholesky(self.xTx + self.ridge_param * I)
                W = torch.cholesky_solve(self.xTy, chol).T
            else:
                raise NotImplementedError(f"Learning algorithm '{self.learning_algo}' not implemented.")
        except Exception as e:
            # Log conditioning stats to help debug failures
            stats = self.covariance_stats(safe=True)
            LOGGER.exception("Trainer.fit failed", extra={"extra": {"cov_stats": stats}})
            raise

        # Log final stats
        try:
            from .logging_utils import tensor_stats
            LOGGER.debug("Readout weights computed",
                extra={"extra": {"W_out_stats": tensor_stats(W)}}
            )
        except Exception:
            pass
        return W

    def covariance_stats(self, safe: bool = False):
        """
        Return dict of conditioning stats for XᵀX used in the last fit().
        """
        if self.xTx is None:
            return {"error": "xTx is None"}
        with torch.no_grad():
            try:
                s = torch.linalg.svdvals(self.xTx)
                s_sorted, _ = torch.sort(s, descending=True)
                tol = s_sorted[0] * max(self.xTx.shape) * torch.finfo(s.dtype).eps
                rank = int((s_sorted > tol).sum().item())
                cond = (s_sorted[0] / s_sorted[-1]).item() if s_sorted[-1] > 0 else float('inf')
                return {
                    "shape": tuple(self.xTx.shape),
                    "rank": rank,
                    "cond": float(cond),
                    "sigma_max": float(s_sorted[0].item()),
                    "sigma_min": float(s_sorted[-1].item()),
                }
            except Exception as e:
                if safe:
                    return {"error": f"cov_stats_failed:{e}"}
                raise
