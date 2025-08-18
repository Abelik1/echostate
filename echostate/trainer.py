# echostate/trainer.py
import logging
import torch
import time

LOGGER = logging.getLogger(__name__)
def _fmt_tensor(name, t):
    if t is None:
        return f"{name}: None"
    if not isinstance(t, torch.Tensor):
        return f"{name}: (not a tensor) {type(t)}"
    pinned = getattr(t, "is_pinned", lambda: False)()
    return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
            f"pinned={pinned}, requires_grad={t.requires_grad}")
class Trainer:
    def __init__(self, ridge_param: float = 1e-6,
                 learning_algo: str = 'inv',
                 device: torch.device = torch.device('cpu'),profile: bool = False):
        self.ridge_param = ridge_param
        self.learning_algo = learning_algo
        self.device = device
        self.xTx = None
        self.xTy = None
        self._profile = profile   

    def fit(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        I = torch.eye(X.shape[1], device=self.device, dtype=X.dtype)
        # print(self.device)
        # Cache normal equations and full X/Y for algorithms that need them
        self.xTx = X.T @ X
        self.xTy = X.T @ Y
        self.X = X
        self.Y = Y

        LOGGER.debug(
            "Trainer.fit: formed X^T X and X^T Y",
            extra={"extra": {
                "X_shape": tuple(X.shape),
                "Y_shape": tuple(Y.shape),
                "ridge_param": float(self.ridge_param),
                "algo": self.learning_algo,
            }},
        )

        try:
            algo = self.learning_algo
            lam = float(self.ridge_param)

            if algo == "inv":
                W = (torch.linalg.inv(self.xTx + lam * I) @ self.xTy).T

            elif algo == "cholesky":
                chol = torch.linalg.cholesky(self.xTx + lam * I)
                W = torch.cholesky_solve(self.xTy, chol).T

            # ========= Methods that only need xTx and xTy (no full X) =========
            elif algo == "solve":
                # Solve (X^T X + λI) W^T = X^T Y without factoring explicitly
                A = self.xTx + lam * I
                W = torch.linalg.solve(A, self.xTy).T

            elif algo == "eigh":
                # Eigenvalue (symmetric) factorization: A = Q Λ Q^T
                A = self.xTx + lam * I
                evals, Q = torch.linalg.eigh(A)  # ascending
                eps = torch.finfo(A.dtype).eps
                inv_evals = 1.0 / torch.clamp(evals, min=eps)
                W = (Q @ (inv_evals.unsqueeze(-1) * (Q.mT @ self.xTy))).T

            elif algo == "cg":
                # Conjugate Gradient on SPD normal matrix (good for large R)
                A = self.xTx + lam * I
                B = self.xTy  # shape: (R+1, d_out)
                try:
                    Z, info = torch.linalg.cg(A, B, maxiter=2000, rtol=1e-6, atol=0.0)
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
            elif algo == "qr":
                # Economy QR: X = Q R, solve R W^T = Q^T Y  (numerically stable LS)
                Qm, R = torch.linalg.qr(self.X, mode='reduced')
                W = torch.linalg.solve(
                    R + lam * torch.eye(R.size(-1), dtype=R.dtype, device=R.device),
                    Qm.mT @ self.Y
                ).T
                # Note: classic QR is for λ=0; the diagonal loading above is a common variant.

            elif algo == "svd":
                # SVD ridge (most stable): X = U S V^T, W = V diag(S/(S^2+λ)) U^T Y
                U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)   # Vh = V^T
                denom = (S * S + lam)
                S_shrink = S / denom
                W = (Vh.mT @ (S_shrink.unsqueeze(-1) * (U.mT @ self.Y))).T

            elif algo == "tsvd":
                # Truncated SVD ridge: drop tiny singular values to reduce noise
                U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)
                rcond = 1e-12 if self.X.dtype == torch.float64 else 1e-6
                keep = S > (rcond * S.max())
                U_r = U[:, keep]
                S_r = S[keep]
                V_r = Vh.mT[:, keep]
                S_shrink = S_r / (S_r * S_r + lam)
                W = (V_r @ (S_shrink.unsqueeze(-1) * (U_r.mT @ self.Y))).T

            elif algo == "pinv":
                # Moore–Penrose via covariance eigendecomposition:
                # W^T = V diag(1/(σ^2 + λ)) V^T (X^T Y), with truncation for tiny σ.
                # This matches torch.linalg.pinv(X) @ Y (minimum-norm) when λ=0.
                evals, V = torch.linalg.eigh(self.xTx)  # evals = σ^2 (ascending)
                s2_max = torch.amax(evals)
                rcond = 1e-12 if self.xTx.dtype == torch.float64 else 1e-6
                keep = evals > (rcond * s2_max)

                inv = torch.zeros_like(evals)
                if lam == 0.0:
                    inv[keep] = 1.0 / evals[keep]              # 1/σ^2 (pseudo-inverse on support)
                else:
                    inv[keep] = 1.0 / (evals[keep] + lam)      # ridge-regularized pseudoinverse

                W = (V @ (inv.unsqueeze(-1) * (V.mT @ self.xTy))).T

            else:
                raise NotImplementedError(f"Learning algorithm '{algo}' not implemented.")

        except Exception:
            # Log conditioning stats to help debug failures
            stats = self.covariance_stats(safe=True)
            LOGGER.exception("Trainer.fit failed", extra={"extra": {"cov_stats": stats}})
            raise

        # Log final stats
        try:
            from .logging_utils import tensor_stats
            LOGGER.debug("Readout weights computed", extra={"extra": {"W_out_stats": tensor_stats(W)}})
        except Exception:
            pass

        return W

    def fit_from_cov(self, xTx: torch.Tensor, xTy: torch.Tensor, n_feat: int, rcond: float | None = None):
        self.xTx = xTx
        self.xTy = xTy
        I = torch.eye(n_feat, device=xTx.device, dtype=xTx.dtype)
        lam = float(self.ridge_param)
        algo = self.learning_algo
        prof_on = getattr(self, "_profile", False)

        if prof_on:
            print(f"[PROFILE][Trainer.fit_from_cov] algo={algo}, lam={lam:g}")
            print(_fmt_tensor("xTx", xTx))
            print(_fmt_tensor("xTy", xTy))
            print(_fmt_tensor("I", I))
            if xTx.device.type == "cuda":
                idx = xTx.device.index or 0
                alloc = torch.cuda.memory_allocated(idx)/1e9
                resv  = torch.cuda.memory_reserved(idx)/1e9
                print(f"[PROFILE][Trainer] cuda_alloc={alloc:.3f}GB, cuda_reserved={resv:.3f}GB")

        # Time the solve (with CUDA sync for accurate wall time)
        if xTx.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if algo == "inv":
            A = xTx + lam * I
            W = (torch.linalg.inv(A) @ xTy).T

        elif algo == "cholesky":
            A = xTx + lam * I
            chol = torch.linalg.cholesky(A)
            W = torch.cholesky_solve(xTy, chol).T

        elif algo == "solve":
            A = xTx + lam * I
            W = torch.linalg.solve(A, xTy).T

        elif algo == "eigh":
            A = xTx + lam * I
            evals, Q = torch.linalg.eigh(A)
            eps = torch.finfo(A.dtype).eps
            inv = 1.0 / torch.clamp(evals, min=eps)
            W = (Q @ (inv.unsqueeze(-1) * (Q.mT @ xTy))).T

        elif algo == "cg":
            A = xTx + lam * I
            Z, _ = torch.linalg.cg(A, xTy, maxiter=2000, rtol=1e-6, atol=0.0)
            W = Z.T

        elif algo == "pinv":
            evals, V = torch.linalg.eigh(xTx)  # ascending, evals = σ^2
            s2_max = torch.amax(evals)
            if rcond is None:
                rcond = 1e-12 if xTx.dtype == torch.float64 else 1e-6
            keep = evals > (rcond * s2_max)
            inv = torch.zeros_like(evals)
            inv[keep] = 1.0 / (evals[keep] + lam)
            W = (V @ (inv.unsqueeze(-1) * (V.mT @ xTy))).T

        else:
            raise NotImplementedError(f"Learning algorithm '{algo}' not supported from covariance.")

        if xTx.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if prof_on:
            print(f"[PROFILE][Trainer.fit_from_cov] solve({algo}) wall time: {(t1 - t0):.6f}s")
            print(_fmt_tensor("W_out", W))
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
