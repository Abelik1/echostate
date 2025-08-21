import logging
import time
import torch

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
    """
    Linear readout trainer (ridge regression) with several solvers.
    Supports fitting from either the full design matrix X,Y or the
    streamed normal equations xTx, xTy.
    """
    def __init__(self, ridge_param: float = 1e-6,
                 learning_algo: str = "inv",
                 device: torch.device = torch.device("cpu"),
                 profile: bool = False):
        self.ridge_param = ridge_param
        self.learning_algo = learning_algo
        self.device = device
        self._profile = profile
        self.xTx = None
        self.xTy = None
        self.X = None
        self.Y = None

    # Full-design fit (optional path)
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        Y = Y.to(self.device)
        self.xTx = X.T @ X
        self.xTy = X.T @ Y
        self.X, self.Y = X, Y
        return self._solve_from_cov(self.xTx, self.xTy)

    # Streamed fit from normal equations
    def fit_from_cov(self, xTx: torch.Tensor, xTy: torch.Tensor, n_feat: int, rcond: float | None = None):
        self.xTx, self.xTy = xTx, xTy
        return self._solve_from_cov(xTx, xTy, n_feat, rcond=rcond)

    # Diagnostics
    def debug_covariance(self):
        if self.xTx is None:
            print("[DEBUG] covariance: None")
            return
        s = torch.linalg.svdvals(self.xTx)
        tol = s.max() * max(self.xTx.shape) * torch.finfo(s.dtype).eps
        rank = int((s > tol).sum().item())
        cond = (s.max() / s.min()).item()
        print(f"[DEBUG] xTx shape={tuple(self.xTx.shape)}, rank={rank}, cond={cond:.2e}")

    def covariance_stats(self, safe: bool = False):
        if self.xTx is None:
            return {"error": "xTx is None"}
        try:
            s = torch.linalg.svdvals(self.xTx)
            s_sorted, _ = torch.sort(s, descending=True)
            tol = s_sorted[0] * max(self.xTx.shape) * torch.finfo(s.dtype).eps
            rank = int((s_sorted > tol).sum().item())
            cond = (s_sorted[0] / s_sorted[-1]).item() if s_sorted[-1] > 0 else float("inf")
            return {
                "shape": tuple(self.xTx.shape),
                "rank": rank,
                "cond": float(cond),
                "sigma_max": float(s_sorted[0].item()),
                "sigma_min": float(s_sorted[-1].item()),
            }
        except Exception as e:
            return {"error": f"cov_stats_failed:{e}"} if safe else (_ for _ in ()).throw(e)

    # Internal solver
    def _solve_from_cov(self, xTx: torch.Tensor, xTy: torch.Tensor, n_feat: int | None = None, rcond: float | None = None):
        I = torch.eye(xTx.shape[0], device=xTx.device, dtype=xTx.dtype)
        lam = float(self.ridge_param)
        algo = self.learning_algo
        prof_on = self._profile

        if prof_on and xTx.device.type == "cuda":
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
            evals, V = torch.linalg.eigh(xTx)  # evals = σ^2 ascending
            s2_max = torch.amax(evals)
            if rcond is None:
                rcond = 1e-12 if xTx.dtype == torch.float64 else 1e-6
            keep = evals > (rcond * s2_max)
            inv = torch.zeros_like(evals)
            inv[keep] = 1.0 / (evals[keep] + lam)
            W = (V @ (inv.unsqueeze(-1) * (V.mT @ xTy))).T
        elif algo in {"qr", "svd", "tsvd"}:
            # These need X and Y
            assert self.X is not None and self.Y is not None, f"{algo} requires full design X,Y"
            lamI = lam * torch.eye(self.X.shape[1], device=self.X.device, dtype=self.X.dtype)
            if algo == "qr":
                Qm, R = torch.linalg.qr(self.X, mode="reduced")
                W = torch.linalg.solve(R + lamI, Qm.mT @ self.Y).T
            elif algo == "svd":
                U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)
                S_shrink = S / (S * S + lam)
                W = (Vh.mT @ (S_shrink.unsqueeze(-1) * (U.mT @ self.Y))).T
            else:  # tsvd
                U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)
                rcond = 1e-12 if self.X.dtype == torch.float64 else 1e-6
                keep = S > (rcond * S.max())
                U_r, S_r, V_r = U[:, keep], S[keep], Vh.mT[:, keep]
                S_shrink = S_r / (S_r * S_r + lam)
                W = (V_r @ (S_shrink.unsqueeze(-1) * (U_r.mT @ self.Y))).T
        else:
            raise NotImplementedError(f"Algorithm '{algo}' not supported.")

        if prof_on and xTx.device.type == "cuda":
            torch.cuda.synchronize()
        if prof_on:
            print(f"[Trainer] solve({algo}) time: {time.perf_counter() - t0:.6f}s")
            print(_fmt_tensor("W_out", W))
        return W
