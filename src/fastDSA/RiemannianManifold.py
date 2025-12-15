# simdist_Landing.py
# Landing-style similarity transform distance using a Riemannian optimizer
# on the Stiefel manifold (orthonormal columns). Accepts NumPy or Torch inputs.

from __future__ import annotations
import time
from typing import Optional, Literal, Tuple, Union

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import geoopt
import ot  # POT: optimal transport

ArrayLike = Union[np.ndarray, torch.Tensor]


# ----------------------------- Landing Optimizer -----------------------------

class LandingSGD(Optimizer):
    """
    Riemannian SGD with momentum on the Canonical Stiefel manifold (orthonormal columns).
    - Uses geoopt's Stiefel manifold to compute Riemannian gradients and retractions.
    - Keeps parameters on the manifold at every step (Landing-style).
    - Optional: Nesterov, dampening, weight decay (Euclidean), safe step, periodic stabilization.

    Expected usage:
        stiefel = geoopt.manifolds.stiefel.CanonicalStiefel()
        C = geoopt.ManifoldParameter(stiefel.random(n, n), manifold=stiefel)
        opt = LandingSGD([{"params": [C], "lr": 1e-2, "momentum": 0.0, ... }])
    """

    def __init__(self, params, lr=1e-2, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov: bool = False,
                 safe_step: Optional[float] = None, stabilize: Optional[int] = 50):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            safe_step=safe_step,
            stabilize=stabilize,
            step_count=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            safe_step = group["safe_step"]
            stabilize = group["stabilize"]

            group["step_count"] += 1
            k = group["step_count"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Must be a Stiefel manifold parameter
                if not hasattr(p, "manifold"):
                    raise TypeError("Parameter must be a geoopt ManifoldParameter with a Stiefel manifold.")
                if not isinstance(p.manifold, geoopt.manifolds.stiefel.CanonicalStiefel):
                    raise TypeError("LandingSGD supports only Canonical Stiefel manifold parameters.")

                # Euclidean gradient
                grad = p.grad
                # Euclidean weight decay, then map to Riemannian grad
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Map to Riemannian gradient in the tangent space at p
                rgrad = p.manifold.egrad2rgrad(p, grad)

                # Momentum buffer (tangent space)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                else:
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(rgrad, alpha=1.0 - dampening)

                # Nesterov direction if requested
                direction = rgrad.add(buf, alpha=momentum) if nesterov else buf

                # Optional "safe step": shrink step a bit to avoid overshoot
                step_size = lr
                if safe_step is not None and safe_step > 0:
                    step_size = lr / (1.0 + safe_step * lr)

                # Retraction update to stay on manifold
                new_p = p.manifold.retr(p, -step_size * direction)
                p.copy_(new_p)

                # Periodic exact projection (stabilization)
                if stabilize and k % stabilize == 0:
                    p.copy_(p.manifold.projx(p))

        return loss


# -------------------- Similarity Transform Distance (Landing) --------------------

class SimilarityTransformDist(torch.nn.Module):
    """
    Landing-style optimizer on Stiefel to align B to A by minimizing:
        min_C ||A* - C B* C^T||_F^2
    (A*, B* optionally Frobenius-normalized inside optimization).

    After optimization, returns a distance/score chosen by `score_method`:
        - "angular"   : arccos(<A, C B C^T> / (||A||_F ||C B C^T||_F))
        - "frobenius" : ||A - C B C^T||_F
        - "wasserstein": OT distance between spectral distributions of A, B
                         (eigenvalues as points in R^2 if compare="eig",
                          singular values in R if compare="sv"; no optimization needed)

    Parameters
    ----------
    iters : int
        Optimization steps.
    lr : float
        Learning rate for LandingSGD.
    momentum, dampening, nesterov, weight_decay, safe_step, stabilize :
        Optimizer knobs (see LandingSGD).
    verbose : bool
        Print progress and final timing/score.
    device : torch.device | None
        Device for computation (defaults to CPU).
    normalize : bool
        If True, internally Frobenius-normalize A and B before optimization.
        If False, optimize on raw A and B. (The returned angle is always scale-invariant.)
    so : bool
        If True and C is square, enforce det(C)=+1 by flipping one column when necessary.
    init : {"orthogonal","identity","random"}
        Initialization for C on the Stiefel manifold.
    seed : int | None
        Seed for deterministic init.
    """

    def __init__(
        self,
        iters: int = 500,
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        safe_step: Optional[float] = None,
        stabilize: Optional[int] = 50,
        verbose: bool = False,
        device: Optional[torch.device] = None,
        normalize: bool = True,
        so: bool = False,
        init: Literal["orthogonal", "identity", "random"] = "identity",
        seed: Optional[int] = None,
        # NEW:
        score_method: Literal["angular", "frobenius", "wasserstein"] = "angular",
        wasserstein_compare: Literal["eig", "sv"] = "eig",
    ):
        super().__init__()
        self.iters = int(iters)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.dampening = float(dampening)
        self.nesterov = bool(nesterov)
        self.weight_decay = float(weight_decay)
        self.safe_step = safe_step
        self.stabilize = stabilize
        self.verbose = bool(verbose)
        self.device = device if device is not None else torch.device("cpu")
        self.normalize = bool(normalize)
        self.so = bool(so)
        self.init = init
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # scoring config
        self.score_method = score_method
        self.wasserstein_compare = wasserstein_compare

        # Stiefel manifold
        self._stiefel = geoopt.manifolds.stiefel.CanonicalStiefel()

    # ------------------------ helpers ------------------------

    @staticmethod
    def _fro_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, ord="fro")

    def _maybe_normalize(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize:
            return A, B
        An = A / (self._fro_norm(A) + 1e-12)
        Bn = B / (self._fro_norm(B) + 1e-12)
        return An, Bn

    def _init_C(self, n: int, dtype: torch.dtype, device: torch.device) -> geoopt.ManifoldParameter:
        if self.init == "identity":
            C0 = torch.eye(n, dtype=dtype, device=device)
            C0 = self._stiefel.projx(C0)
        elif self.init == "orthogonal":
            C0 = self._stiefel.random(n, n, dtype=dtype, device=device)
        elif self.init == "random":
            C0 = torch.randn((n, n), dtype=dtype, device=device)
            C0 = self._stiefel.projx(C0)
        else:
            raise ValueError(f"Unknown init='{self.init}'.")
        return geoopt.ManifoldParameter(C0, manifold=self._stiefel)

    @staticmethod
    def _as_tensor(x: ArrayLike, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            if not np.issubdtype(x.dtype, np.floating):
                x = x.astype(np.float32, copy=False)
            t = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            t = x
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")
        return t.to(device=device, dtype=dtype, copy=False)

    @torch.no_grad()
    def _enforce_so(self, C: torch.Tensor) -> None:
        if not self.so:
            return
        if C.shape[0] == C.shape[1]:
            if torch.linalg.det(C) < 0:
                C[:, -1].mul_(-1)

    @staticmethod
    def _wasserstein_spectral_distance(
        A: torch.Tensor, B: torch.Tensor, compare: Literal["eig","sv"] = "eig"
    ) -> float:
        with torch.no_grad():
            if compare == "eig":
                a = torch.linalg.eigvals(A)
                b = torch.linalg.eigvals(B)
                a_np = torch.stack((a.real, a.imag), dim=1).detach().cpu().numpy()
                b_np = torch.stack((b.real, b.imag), dim=1).detach().cpu().numpy()
            elif compare == "sv":
                a_np = torch.linalg.svdvals(A).detach().cpu().numpy().reshape(-1, 1)
                b_np = torch.linalg.svdvals(B).detach().cpu().numpy().reshape(-1, 1)
            else:
                raise ValueError("wasserstein_compare must be 'eig' or 'sv'.")

        M  = ot.dist(a_np, b_np)                          # cost matrix
        wa = np.ones(a_np.shape[0]) / a_np.shape[0]       # uniform masses
        wb = np.ones(b_np.shape[0]) / b_np.shape[0]
        return float(ot.emd2(wa, wb, M))

    # ------------------------ public API ------------------------

    def fit_score(self, A: ArrayLike, B: ArrayLike, return_time: bool = False):
        """
        If score_method == 'wasserstein':
            - Skip optimization; return spectral OT distance between A and B.

        Else (angular|frobenius):
            - Optimize C with LandingSGD on Stiefel to align B to A
              by minimizing ||A* - C B* C^T||_F^2, then compute the chosen score.

        Returns
        -------
        float (score)                         if return_time == False
        (float score, float elapsed_seconds)  if return_time == True
        """
        t0 = time.time()

        # shape checks first (non-wasserstein requires same square shape)
        def _shape(x: ArrayLike) -> Tuple[int, ...]:
            if isinstance(x, np.ndarray):     return x.shape
            if isinstance(x, torch.Tensor):   return tuple(x.shape)
            return ()

        # dtype preference
        prefer_float64 = (
            (isinstance(A, np.ndarray) and A.dtype == np.float64) or
            (isinstance(B, np.ndarray) and B.dtype == np.float64) or
            (isinstance(A, torch.Tensor) and A.dtype == torch.float64) or
            (isinstance(B, torch.Tensor) and B.dtype == torch.float64)
        )
        dtype = torch.float64 if prefer_float64 else torch.float32

        # convert
        A = self._as_tensor(A, self.device, dtype)
        B = self._as_tensor(B, self.device, dtype)

        sm = self.score_method.lower()
        if sm == "wasserstein":
            # works even if sizes differ; (optionally) normalize before spectra
            Aopt, Bopt = self._maybe_normalize(A, B)
            score = self._wasserstein_spectral_distance(Aopt, Bopt, self.wasserstein_compare)
            dt = time.time() - t0
            return (score, dt) if return_time else score

        # angular / frobenius path: need same square shape
        if len(_shape(A)) != 2 or len(_shape(B)) != 2:
            raise ValueError("A and B must be 2D matrices.")
        if _shape(A)[0] != _shape(A)[1] or _shape(B)[0] != _shape(B)[1] or _shape(A) != _shape(B):
            raise ValueError("For angular/frobenius, A and B must be square and of the same shape.")

        # optimization copies (maybe normalized)
        Aopt, Bopt = self._maybe_normalize(A, B)
        n = A.shape[0]
        C = self._init_C(n, dtype, self.device)

        # LandingSGD optimizer (unchanged logic)
        opt = LandingSGD(
            [{"params": [C],
              "lr": self.lr,
              "momentum": self.momentum,
              "dampening": self.dampening,
              "nesterov": self.nesterov,
              "weight_decay": self.weight_decay,
              "safe_step": self.safe_step,
              "stabilize": self.stabilize}],
        )

        for it in range(self.iters):
            opt.zero_grad(set_to_none=True)
            CBCt = C @ Bopt @ C.T
            loss = torch.sum((Aopt - CBCt) ** 2)
            loss.backward()
            opt.step()
            if self.so:
                self._enforce_so(C)
            if self.verbose and (it % 100 == 0 or it + 1 == self.iters):
                print(f"[Landing] iter={it+1}/{self.iters} loss={loss.item():.6e}")

        # compute requested score on ORIGINAL A,B (not Aopt/Bopt)
        with torch.no_grad():
            CBCt_full = C @ B @ C.T
            CBCt_full_opt = C @ Bopt @ C.T 
            if sm == "angular":
                num = torch.sum(A * CBCt_full)
                den = self._fro_norm(A) * self._fro_norm(CBCt_full) + 1e-12
                cosang = torch.clamp(num / den, -1.0, 1.0)
                score = torch.arccos(cosang).item()
            elif sm == "frobenius":
                # score = torch.linalg.norm(A - CBCt_full, ord="fro").item()
                # score = torch.norm(Aopt - CBCt_full_opt, p='fro').item()
                # diff = A - CBCt_full
                # score = diff.pow(2).sum().sqrt().item()   # = ||A - C B C^T||_F
                # num = diff.pow(2).sum().sqrt()
                # den = (A.pow(2).sum().sqrt() * CBCt_full.pow(2).sum().sqrt()).clamp_min(1e-12)
                # score = (num / den).item()
                    # unit-Frobenius normalization for each term
                A_n   = A   / A.pow(2).sum().sqrt().clamp_min(1e-12)
                CBC_n = CBCt / CBCt.pow(2).sum().sqrt().clamp_min(1e-12)

                score = (A_n - CBC_n).pow(2).sum().sqrt().item()   # in [0, 2]
            else:
                raise ValueError("score_method must be 'angular', 'frobenius', or 'wasserstein'.")

        dt = time.time() - t0
        if self.verbose:
            print(f"[Landing] score={score:.6f}, time={dt:.3f}s")

        return (score, dt) if return_time else score
    
# # --- quick smoke test ---
# if __name__ == "__main__":
#     import numpy as np
#     import torch

#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.set_default_dtype(torch.float64)  # numerically safer
#     torch.manual_seed(930)

#     n = 10
#     A = torch.randn(n, n, device=dev)
#     B_same = A.clone()
#     B_noisy = A + 0.05 * torch.randn_like(A)

#     # 1) Angular distance (optimization on Stiefel)
#     ang_model = SimilarityTransformDist(
#         iters=200, lr=0.02, verbose=False, device=dev,
#         score_method="angular", normalize=True, so=False, init="identity"
#     )
#     ang_same, t_ang_same = ang_model.fit_score(A, B_same, return_time=True)
#     ang_noisy, t_ang_noisy = ang_model.fit_score(A, B_noisy, return_time=True)
#     print(f"[{dev}] ANGULAR: A vs A => {ang_same:.6f} rad (time {t_ang_same:.3f}s)")
#     print(f"[{dev}] ANGULAR: A vs noisy(A) => {ang_noisy:.6f} rad (time {t_ang_noisy:.3f}s)")

#     # 2) Frobenius distance (also optimized)
#     fro_model = SimilarityTransformDist(
#         iters=200, lr=0.02, verbose=False, device=dev,
#         score_method="frobenius", normalize=True, so=False, init="identity"
#     )
#     fro_same, t_fro_same = fro_model.fit_score(A, B_same, return_time=True)
#     fro_noisy, t_fro_noisy = fro_model.fit_score(A, B_noisy, return_time=True)
#     print(f"[{dev}] FROBENIUS: A vs A => {fro_same:.6f} (time {t_fro_same:.3f}s)")
#     print(f"[{dev}] FROBENIUS: A vs noisy(A) => {fro_noisy:.6f} (time {t_fro_noisy:.3f}s)")

#     # 3) Wasserstein spectral distance (no optimization; fast)
#     was_model = SimilarityTransformDist(
#         score_method="wasserstein", wasserstein_compare="sv", device=dev
#     )
#     was_same, t_was_same = was_model.fit_score(A, B_same, return_time=True)
#     was_noisy, t_was_noisy = was_model.fit_score(A, B_noisy, return_time=True)
#     print(f"[{dev}] WASSERSTEIN(SV): A vs A => {was_same:.6e} (time {t_was_same:.3f}s)")
#     print(f"[{dev}] WASSERSTEIN(SV): A vs noisy(A) => {was_noisy:.6e} (time {t_was_noisy:.3f}s)")

