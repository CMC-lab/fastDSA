from __future__ import annotations

import time
import numpy as np
from typing import List, Tuple, Optional, Iterable, Dict

import torch

try:
    import ot  # POT (optional)
    _HAS_OT = True
except Exception:  # pragma: no cover
    _HAS_OT = False

__all__ = ["SimilarityTransformDist", "run_landing"]


# ---------------------------------------
# Utilities: device/dtype + conversions
# ---------------------------------------

def _default_device(user_device: Optional[str] = None) -> torch.device:
    if user_device is not None:
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if you want MPS fallback on Apple, uncomment next two lines:
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")

def _to_tensor(x, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").numpy()


# ---------------------------------------
# Core math (Torch)
# ---------------------------------------

def orthogonalize_iterative(M: torch.Tensor, steps: int = 100) -> torch.Tensor:
    """Polynomial (Newton-Schulz-like) iterative orthogonalization."""
    a, b, c = 3.0, -16.0 / 5.0, 6.0 / 5.0

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / torch.linalg.norm(M)

    for _ in range(steps):
        A = M.T @ M
        I = torch.eye(A.shape[0], device=M.device, dtype=M.dtype)
        M = M @ (a * I + b * A + c * (A @ A))

    if transpose:
        M = M.T
    return M


def sym(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * (X + X.T)


def skew(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * (X - X.T)


def grad_euclid(A: torch.Tensor, C: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Closed-form Euclidean gradient d/ dC ||A - C B C^T||_F^2."""
    E = A - C @ B @ C.T
    return -2.0 * (E @ C @ B.T + E.T @ C @ B)


def compute_angular_score(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> float:
    with torch.no_grad():
        CBCt = C @ B @ C.T
        num = torch.trace(A.T @ CBCt)
        den = (torch.linalg.norm(A, ord="fro") * torch.linalg.norm(CBCt, ord="fro")).clamp_min(1e-12)
        val = torch.clamp(num / (den + 1e-12), -1.0, 1.0)
        score = torch.arccos(val)
        s = float(score.item())
        if np.isnan(s):  # extremely defensive
            s = float(np.pi) if float(num / (den + 1e-12)) < 0 else 0.0
        return s


def compute_frobenius_score(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, normalized: bool = True) -> float:
    with torch.no_grad():
        CBCt = C @ B @ C.T
        if normalized:
            A_n   = A    / torch.linalg.norm(A,   ord="fro").clamp_min(1e-12)
            CBC_n = CBCt / torch.linalg.norm(CBCt, ord="fro").clamp_min(1e-12)
            eu = torch.linalg.norm(A_n - CBC_n, ord="fro").item()
        else:
            eu = torch.linalg.norm(A - CBCt, ord="fro").item()
    return float(eu)


def _wasserstein_1d_uniform(u: np.ndarray, v: np.ndarray) -> float:
    """Fast 1D W1 for equally-weighted samples via quantile matching."""
    u = np.ravel(u)
    v = np.ravel(v)
    m = min(len(u), len(v))
    if m == 0:
        return 0.0
    u_s = np.sort(u)[:m]
    v_s = np.sort(v)[:m]
    return float(np.mean(np.abs(u_s - v_s)))


def compute_wasserstein_distance(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                                 spectrum: str = "eig", p: int = 1) -> float:
    """Wasserstein distance (default 1-Wasserstein) between spectra of A and C B C^T.

    Robust to non-finite values by returning NaN for that iteration.
    """
    with torch.no_grad():
        CBCt = C @ B @ C.T
        # Guard: if matrix contains NaN/Inf, skip metric this iteration
        if not torch.isfinite(CBCt).all():
            return float("nan")
        try:
            if spectrum == "sv":
                vals_A = torch.linalg.svdvals(A)
                vals_B = torch.linalg.svdvals(CBCt)
            else:
                vals_A = torch.linalg.eigvals(A).real
                vals_B = torch.linalg.eigvals(CBCt).real
        except torch._C._LinAlgError:
            return float("nan")

    a = vals_A.to("cpu").numpy()
    b = vals_B.to("cpu").numpy()

    if _HAS_OT:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        M = ot.dist(a, b, metric="euclidean")
        wa = np.ones(a.shape[0]) / max(1, a.shape[0])
        wb = np.ones(b.shape[0]) / max(1, b.shape[0])
        return float(ot.emd2(wa, wb, M))
    else:
        return _wasserstein_1d_uniform(a, b)


# ---------------------------------------
# Landing step(s) with momentum (Torch)
# ---------------------------------------

def landing_step_momentum(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    Avel: torch.Tensor,
    eta: float,
    gamma: float = 0.99,
    lam: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-C update using the explicit Euclidean gradient and a skew-momentum on the tangent.
    """
    with torch.no_grad():
        G = grad_euclid(A, C, B)
        psi = skew(G @ C.T)
        Avel.mul_(gamma).add_(psi, alpha=(1.0 - gamma))  # Avel = (1-gamma)*psi + gamma*Avel
        C = C - eta * ((Avel @ C) + lam * ((C @ C.T - torch.eye(C.shape[0], device=C.device, dtype=C.dtype)) @ C))
    return C, Avel


def landing_step_multiple_momentum(
    A: torch.Tensor,
    B: torch.Tensor,
    C_list: List[torch.Tensor],
    Avels: List[torch.Tensor],
    eta: float,
    gamma: float = 0.99,
    lam: float = 1.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Chain update for C_total = C1 @ C2 @ ... @ Cn:
      - Uses torch autograd to compute grads wrt each Ci.
      - Momentum update in skew space, plus soft orthogonality penalty.
    """
    # Ensure grad-enabled copies
    Cs = [Ci.detach().requires_grad_(True) for Ci in C_list]

    # Forward: objective = ||A - C_total B C_total^T||_F^2
    Ctot = Cs[0]
    for Ci in Cs[1:]:
        Ctot = Ctot @ Ci
    E = A - Ctot @ B @ Ctot.T
    loss = torch.sum(E * E)

    # Backprop to each Ci
    grads = torch.autograd.grad(loss, Cs, retain_graph=False, create_graph=False)

    # Update (no_grad)
    with torch.no_grad():
        new_Cs, new_Avels = [], []
        I = torch.eye(C_list[0].shape[0], device=C_list[0].device, dtype=C_list[0].dtype)
        for Ci, Ai, Gi in zip(Cs, Avels, grads):
            psi = skew(Gi @ Ci.T)
            Ai.mul_(gamma).add_(psi, alpha=(1.0 - gamma))
            Ci = Ci - eta * ((Ai @ Ci) + lam * ((Ci @ Ci.T - I) @ Ci))
            new_Cs.append(Ci)
            new_Avels.append(Ai)
    return new_Cs, new_Avels


# ---------------------------------------
# Landing loop scaffolding (Torch)
# ---------------------------------------

def _landing(
    A1: torch.Tensor,
    A2: torch.Tensor,
    C_opt: List[torch.Tensor],
    Avels: List[torch.Tensor],
    eta: float = 0.01,
    gamma: float = 0.99,
    n_Cmats: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if n_Cmats == 1:
        C_next, Avel = landing_step_momentum(A1, A2, C_opt[0], Avels[0], eta=eta, gamma=gamma)
        Avels[0] = Avel
        C_opt = [C_next]
    else:
        C_opt, Avels = landing_step_multiple_momentum(A1, A2, C_opt, Avels, eta=eta, gamma=gamma)
    return C_opt, Avels


def optimize(
    A1,
    A2,
    algo=_landing,
    eta: float = 0.01,
    gamma: float = 0.99,
    n_Cmats: int = 1,
    its: int = 1000,
    verbose: bool = False,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    metrics: Iterable[str] = ("angular", "frobenius"),
    wasserstein_spectrum: str = "eig",
    normalize_fro: bool = True,
    retract_every: Optional[int] = None,
) -> Tuple[List, Dict[str, List[float]], List[float], List[float], Tuple[float, float, float]]:
    """
    Runs the landing optimizer on the chosen device (CUDA if available).
    Inputs A1, A2 can be numpy arrays or torch tensors.

    Parameters
    ----------
    metrics: iterable of {'angular','frobenius','euclidean','wasserstein'}
        Which metrics to compute per-iteration. Only the selected ones are computed to save time.
    wasserstein_spectrum: {'eig','sv'}
        Spectrum to compare when computing Wasserstein.
    normalize_fro: bool
        If True, the Frobenius score compares normalized matrices (unit Fro norm).

    Returns
    -------
      Cs            : [initial_chain(list of C_i), C_total@it=1, C_total@it=2, ...]  (tensors on device)
      scores        : dict mapping metric -> list of values per iteration
      orthogs       : list of ||C^T C - I||_F per iteration (floats)
      losses        : list of squared Frobenius losses ||A1 - C A2 C^T||_F^2 per iteration
      timing        : (t_start_all, t_end_all, seconds_total)
    """
    dev = _default_device(device)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    A1_t = _to_tensor(A1, device=dev, dtype=dtype)
    A2_t = _to_tensor(A2, device=dev, dtype=dtype)

    n = A1_t.shape[0]
    assert A1_t.shape == A2_t.shape and A1_t.ndim == 2 and n == A1_t.shape[1], \
        "A1 and A2 must be the same square shape (n x n)."
    
    #normalize for the optimization dynamics
    A1n = A1_t / torch.linalg.norm(A1_t, ord="fro").clamp_min(1e-12)
    A2n = A2_t / torch.linalg.norm(A2_t, ord="fro").clamp_min(1e-12)

    # Initialize chain of C_i exactly as identity on device (so if B==A, we're at a fixed point)
    C_opt: List[torch.Tensor] = [torch.eye(n, device=dev, dtype=dtype) for _ in range(n_Cmats)]

    # Momentum buffers
    Avels: List[torch.Tensor] = [torch.zeros((n, n), device=dev, dtype=dtype) for _ in range(n_Cmats)]

    # Metric trackers
    metrics_set = set(m.lower() for m in metrics)
    if "euclidean" in metrics_set:
        metrics_set.add("frobenius")  # alias
    scores: Dict[str, List[float]] = {m: [] for m in sorted(metrics_set)}
    orthogs: List[float] = []
    losses: List[float] = []

    # Store initial chain as-is (list of tensors)
    Cs: List = [C_opt]

    # Progress iterator
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(range(its))
        except Exception:
            iterator = range(its)
    else:
        iterator = range(its)

    t_all0 = time.time()
    iteration_times: List[float] = []

    for _ in iterator:
        t0 = time.time()

        C_opt, Avels = algo(A1_t, A2_t, C_opt, Avels, eta=eta, gamma=gamma, n_Cmats=n_Cmats)

        # Total transform
        C_tot = C_opt[0]
        for Ci in C_opt[1:]:
            C_tot = C_tot @ Ci

        # Optional: retract to the orthogonal manifold to prevent drift/explosions
        if retract_every is not None and (len(Cs) % max(1, retract_every) == 0):
            # Retract each factor Ci via QR for stability
            with torch.no_grad():
                new_chain = []
                for Ci in C_opt:
                    try:
                        Q, _ = torch.linalg.qr(Ci)
                        new_chain.append(Q)
                    except Exception:
                        new_chain.append(Ci)
                C_opt[:] = new_chain
                # recompute total after retraction
                C_tot = C_opt[0]
                for Ci in C_opt[1:]:
                    C_tot = C_tot @ Ci

        Cs.append(C_tot)

        with torch.no_grad():
            I = torch.eye(n, device=dev, dtype=dtype)
            orth = torch.linalg.norm(C_tot.T @ C_tot - I, ord="fro").item()
            orthogs.append(float(orth))

            # Raw reconstruction loss (un-normalized)
            # E = A1_t - C_tot @ A2_t @ C_tot.T
            E = A1n - C_tot @ A2n @ C_tot.T

            losses.append(float(torch.sum(E * E).item()))

            # Selective metrics
            if "angular" in metrics_set:
                try:
                    scores["angular"].append(compute_angular_score(A1_t, A2_t, C_tot))
                except Exception:
                    scores["angular"].append(float("nan"))
            if "frobenius" in metrics_set:
                try:
                    scores["frobenius"].append(
                        compute_frobenius_score(A1_t, A2_t, C_tot, normalized=normalize_fro)
                    )
                except Exception:
                    scores["frobenius"].append(float("nan"))
            if "wasserstein" in metrics_set:
                val = compute_wasserstein_distance(A1_t, A2_t, C_tot, spectrum=wasserstein_spectrum)
                scores["wasserstein"].append(val)

        t1 = time.time()
        iteration_times.append(t1 - t0)

    t_all1 = time.time()
    timing = (t_all0, t_all1, t_all1 - t_all0)

    # We tack iteration_times into scores dict for convenient access without breaking signature
    scores["iteration_time"] = iteration_times

    return Cs, scores, orthogs, losses, timing


# ---------------------------------------
# Public API: Class + wrapper
# ---------------------------------------

class SimilarityTransformDist:
    """
    CUDA-accelerated optimizer (PyTorch) for aligning A ≈ C B C^T.
    - Accepts numpy arrays or tensors; runs on GPU if available (or device passed).
    - Single-C step uses explicit gradient; multi-C chain uses torch autograd.
    - Tracks *selectable* metrics (angular, Frobenius/Euclidean, Wasserstein), orthogonality, per-iteration loss & time.
    - Records (t_start, t_end, seconds) via time.time().
    """

    def __init__(
        self,
        its: int = 1000,
        eta: float = 0.01,
        gamma: float = 0.99,
        n_Cmats: int = 1,
        verbose: bool = False,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        metrics: Iterable[str] = ("angular", "frobenius"),
        wasserstein_spectrum: str = "eig",
        normalize_fro: bool = True,
    ):
        self.its = int(its)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.n_Cmats = int(n_Cmats)
        self.verbose = bool(verbose)
        self.seed = seed
        self.device = _default_device(device)
        self.dtype = dtype

        # Metric config
        self.metrics = tuple(metrics)
        self.wasserstein_spectrum = wasserstein_spectrum
        self.normalize_fro = normalize_fro

        # Results
        self.C_chain_init: Optional[List[torch.Tensor]] = None
        self.C_total_traj: Optional[List[torch.Tensor]] = None
        self.C_final: Optional[torch.Tensor] = None
        self.scores: Optional[Dict[str, List[float]]] = None
        self.orthogs: Optional[List[float]] = None
        self.losses: Optional[List[float]] = None
        self.timing: Optional[Tuple[float, float, float]] = None

    def fit(self, A, B) -> "SimilarityTransformDist":
        Cs, scores, orthogs, losses, timing = optimize(
            A1=A, A2=B, algo=_landing, eta=self.eta, gamma=self.gamma,
            n_Cmats=self.n_Cmats, its=self.its, verbose=self.verbose,
            seed=self.seed, device=str(self.device), dtype=self.dtype,
            metrics=self.metrics, wasserstein_spectrum=self.wasserstein_spectrum,
            normalize_fro=self.normalize_fro,
        )
        # Save results
        self.C_chain_init = Cs[0] if isinstance(Cs[0], list) else None
        self.C_total_traj = Cs[1:]
        self.C_final = self.C_total_traj[-1] if self.C_total_traj else None
        self.scores = scores
        self.orthogs = orthogs
        self.losses = losses
        self.timing = timing
        return self

    def score(self, A, B, method: str = "angular") -> float:
        if self.C_final is None:
            self.fit(A, B)
        C = self.C_final
        A_t = _to_tensor(A, device=self.device, dtype=self.dtype)
        B_t = _to_tensor(B, device=self.device, dtype=self.dtype)
        m = method.lower()
        if m == "angular":
            return compute_angular_score(A_t, B_t, C)
        elif m in ("euclidean", "frobenius"):
            return compute_frobenius_score(A_t, B_t, C, normalized=self.normalize_fro)
        elif m == "wasserstein":
            return compute_wasserstein_distance(A_t, B_t, C, spectrum=self.wasserstein_spectrum)
        else:
            raise ValueError(f"Unknown score method: {method}")

    def fit_score(self, A, B, method: str = "angular") -> float:
        self.fit(A, B)
        return self.score(A, B, method=method)

    # Convenience
    def orthogonality_score(self) -> float:
        if not self.orthogs:
            return float("nan")
        return float(self.orthogs[-1])

    def timings(self) -> Tuple[float, float, float]:
        return self.timing if self.timing is not None else (float("nan"), float("nan"), float("nan"))

    def per_iteration_times(self) -> List[float]:
        if not self.scores or "iteration_time" not in self.scores:
            return []
        return list(self.scores["iteration_time"])  # copy

    def C(self) -> Optional[torch.Tensor]:
        return self.C_final


def run_landing(
    A,
    B,
    its: int = 1000,
    eta: float = 0.01,
    gamma: float = 0.99,
    n_Cmats: int = 1,
    verbose: bool = False,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    metrics: Iterable[str] = ("angular", "frobenius"),
    wasserstein_spectrum: str = "eig",
    normalize_fro: bool = True,
) -> Tuple[Dict[str, float], List[torch.Tensor], List[float], List[float], Tuple[float, float, float]]:
    """
    Convenience wrapper to mirror experiment loops.

    Returns
    -------
      finals_by_metric : {metric: last_value}
      C_total_traj     : list of C_t per iteration
      orthogs          : list of ||C^T C - I||_F per iteration
      losses           : list of reconstruction losses per iteration
      timing           : (t0, t1, dt)
    """
    solver = SimilarityTransformDist(
        its=its, eta=eta, gamma=gamma, n_Cmats=n_Cmats, verbose=verbose,
        seed=seed, device=device, dtype=dtype, metrics=metrics,
        wasserstein_spectrum=wasserstein_spectrum, normalize_fro=normalize_fro,
    )
    solver.fit(A, B)

    finals = {m: (solver.scores[m][-1] if m in solver.scores and solver.scores[m] else float("nan"))
              for m in solver.scores if m != "iteration_time"}

    return finals, solver.C_total_traj, solver.orthogs, solver.losses, solver.timing


# # --- quick smoke test ---
# if __name__ == "__main__":
#     dev = _default_device()
#     torch.manual_seed(931)
#     n = 100
#     A = torch.randn(n, n, device=dev)
#     B = torch.randn(n, n, device=dev)
#     # B = A

#     # model = SimilarityTransformDist(
#     #     its=200, eta=0.02, gamma=0.98, n_Cmats=2, verbose=False, device=str(dev),
#     #     metrics=("angular", "wasserstein"), wasserstein_spectrum="sv"
#     # )
#     model = SimilarityTransformDist(
#         its=200, eta=0.002, gamma=0.98, n_Cmats=2, verbose=False, device=str(dev),
#         metrics=("angular",)
#     )
#     final_ang = model.fit_score(A, B, method="angular")
#     final_was = model.score(A, B, method="wasserstein")

#     t0, t1, dt = model.timings()
#     print(f"[{dev}] Angular (rad): {final_ang:.6f} | Wasserstein: {final_was:.6e} |"
#           f" Orthog: {model.orthogonality_score():.2e} | Time: {dt:.3f}s |"
#           f" Per-iter time mean: {np.mean(model.per_iteration_times() or [np.nan]):.6f}s")