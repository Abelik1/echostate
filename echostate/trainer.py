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
        self.X = X
        self.Y = Y
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
                
            # ========= Methods that only need xTx and xTy (no full X) =========
            elif self.learning_algo == "solve":
                # Solve (X^T X + λI) W^T = X^T Y without factoring explicitly
                A = self.xTx + self.ridge_param * I
                W = torch.linalg.solve(A, self.xTy).T

            elif self.learning_algo == "eigh":
                # Eigenvalue (symmetric) factorization: A = Q Λ Q^T
                A = self.xTx + self.ridge_param * I
                # torch.linalg.eigh -> ascending eigenvalues
                evals, Q = torch.linalg.eigh(A)
                # Guard very small eigenvalues
                eps = torch.finfo(A.dtype).eps
                inv_evals = 1.0 / torch.clamp(evals, min=eps)
                W = (Q @ (inv_evals.unsqueeze(-1) * (Q.mT @ self.xTy))).T

            elif self.learning_algo == "cg":
                # Conjugate Gradient on SPD normal matrix (good for large R)
                # Solves A Z = xTy with A = xTx + λI, multiple RHS columns
                A = self.xTx + self.ridge_param * I
                B = self.xTy  # shape: (R+1, d_out)

                # Try to use torch.linalg.cg if available (PyTorch >= 2.1)
                try:
                    Z, info = torch.linalg.cg(A, B, maxiter=2000, rtol=1e-6, atol=0.0)
                    # info == 0 means converged
                    W = Z.T
                except Exception:
                    # Fallback: simple batch CG loop (shared A, loop over RHS)
                    def cg_single(b, maxiter=2000, tol=1e-6):
                        x = torch.zeros_like(b)
                        r = b - A @ x
                        p = r.clone()
                        rs_old = r @ r
                        for _ in range(maxiter):
                            Ap = A @ p
                            alpha = rs_old / (p @ Ap + 1e-30)
                            x = x + alpha * p
                            r = r - alpha * Ap
                            rs_new = r @ r
                            if torch.sqrt(rs_new) < tol:
                                break
                            p = r + (rs_new / (rs_old + 1e-30)) * p
                            rs_old = rs_new
                        return x
                    Z_cols = [cg_single(B[:, j]) for j in range(B.shape[1])]
                    W = torch.stack(Z_cols, dim=1).T

            # ========= Methods that require the full post-washout design X and Y =========
            # (Store them as self.X (N x (R+1)) and self.Y (N x d_out) when you build xTx/xTy.)

            elif self.learning_algo == "qr":
                # Economy QR: X = Q R, solve R W^T = Q^T Y  (numerically stable LS)
                Q, R = torch.linalg.qr(self.X, mode='reduced')
                W = torch.linalg.solve(R + self.ridge_param * torch.eye(R.size(-1), dtype=R.dtype, device=R.device),
                                    Q.mT @ self.Y).T
                # Note: classic QR is for λ=0. For ridge, the line above implements a common
                # "QR + diagonal loading" variant; for exact ridge, prefer SVD below.

            elif self.learning_algo == "svd":
                # SVD ridge (most stable): X = U S V^T, W = V diag(S/(S^2+λ)) U^T Y
                U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)   # Vh = V^T
                # Ridge shrinkage on singular values
                denom = (S * S + self.ridge_param)
                S_shrink = S / denom
                W = (Vh.mT @ (S_shrink.unsqueeze(-1) * (U.mT @ self.Y))).T

            elif self.learning_algo == "tsvd":
                # Truncated SVD ridge: drop tiny singular values to reduce noise
                U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)
                # Keep components above a relative threshold
                rcond = 1e-6
                keep = S > (rcond * S.max())
                U_r = U[:, keep]
                S_r = S[keep]
                V_r = Vh.mT[:, keep]
                S_shrink = S_r / (S_r * S_r + self.ridge_param)
                W = (V_r @ (S_shrink.unsqueeze(-1) * (U_r.mT @ self.Y))).T

            elif self.learning_algo == "pinv":
                # Moore-Penrose pseudoinverse (effectively SVD with default rcond)
                # Ridge-like effect by adding λI before pinv if desired
                X_reg = self.X
                if self.ridge_param > 0:
                    # Tikhonov trick: augment X and Y so that pinv gives ridge
                    n_feat = self.X.shape[1]
                    X_reg = torch.cat([self.X, torch.sqrt(self.ridge_param) * torch.eye(n_feat, device=self.X.device, dtype=self.X.dtype)], dim=0)
                    Y_reg = torch.cat([self.Y, torch.zeros(n_feat, self.Y.shape[1], device=self.Y.device, dtype=self.Y.dtype)], dim=0)
                else:
                    Y_reg = self.Y
                W = (torch.linalg.pinv(X_reg) @ Y_reg).T
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
    def debug_covariance(self): 
        s = torch.linalg.svdvals(self.xTx) 
        tol = s.max() * max(self.xTx.shape) * torch.finfo(s.dtype).eps 
        rank = int((s > tol).sum().item()) 
        cond = (s.max() / s.min()).item() 
        print(f"[DEBUG] covariance shape: {tuple(self.xTx.shape)}, rank: {rank}, condition #: {cond:.2e}")
        
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
