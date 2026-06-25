# src/fastDSA/kwdsa.py
from __future__ import annotations

import numpy as np
import torch

# POT / ot
import ot

# ---- kooplearn compatibility layer -----------------------------------------
# Newer kooplearn docs use kooplearn.kernel and kernel="rbf" (string) :contentReference[oaicite:1]{index=1}
try:
    # Newer (documented) API
    from kooplearn.kernel import NystroemKernelRidge as _NystroemModel
    _KERNEL_DEFAULT = "rbf"
    _SUPPORTS_KERNEL_OBJECT = False
except Exception:
    # Older API (what your current code assumed)
    from kooplearn.models import NystroemKernel as _NystroemModel
    _SUPPORTS_KERNEL_OBJECT = True
    try:
        from kooplearn.kernels import RBF as _RBF
        _KERNEL_DEFAULT = _RBF()
    except Exception:
        # If kernels module is absent even though kooplearn.models exists, fall back to string
        _KERNEL_DEFAULT = "rbf"
        _SUPPORTS_KERNEL_OBJECT = False


def _as_real_point_cloud(x) -> np.ndarray:
    """Convert real values or complex eigenvalues to a finite real point cloud."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x).reshape(-1)

    if np.iscomplexobj(x):
        pts = np.column_stack([x.real, x.imag])
    else:
        pts = x.astype(float, copy=False).reshape(-1, 1)

    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    if pts.shape[0] == 0:
        raise ValueError("Cannot compute Wasserstein distance: no finite spectral values were provided.")

    return pts.astype(float, copy=False)


def compute_wasserstein_distance(a, b) -> float:
    """
    Wasserstein distance between spectral point clouds.

    Real spectra are treated as 1D points. Complex eigenvalues are treated as
    2D points (real part, imaginary part), which avoids invalid complex-valued
    ground-cost matrices in POT.
    """
    a = _as_real_point_cloud(a)
    b = _as_real_point_cloud(b)

    M = ot.dist(a, b, metric="euclidean")
    aw = np.ones(a.shape[0], dtype=float) / a.shape[0]
    bw = np.ones(b.shape[0], dtype=float) / b.shape[0]
    return float(ot.emd2(aw, bw, M))


class KernelDMD(_NystroemModel):
    """
    Kernel DMD wrapper that works with both:
      - kooplearn.kernel.NystroemKernelRidge (new docs API; kernel is string "rbf")
      - kooplearn.models.NystroemKernel (older API; kernel object possibly supported)
    """

    def __init__(
        self,
        data,
        n_delays: int,
        kernel=_KERNEL_DEFAULT,
        delay_interval: int = 1,
        rank: int = 10,
        verbose: bool = False,
        # common kernel hyperparams (new API uses gamma/alpha naming)
        gamma: float = 1.0,
        alpha: float = 1e-7,
        n_centers: int = 600,
        random_state: int = 0,
        reduced_rank: bool = True,
        eigen_solver: str = "arpack",
        **kwargs,
    ):
        self.data = data
        self.n_delays = n_delays
        self.context_window_len = n_delays + 1
        self.delay_interval = delay_interval
        self.verbose = verbose
        self.rank = rank

        self.gamma = gamma
        self.alpha = alpha
        self.n_centers = n_centers
        self.random_state = random_state
        self.reduced_rank = reduced_rank
        self.eigen_solver = eigen_solver

        # Call parent constructor in a way compatible with the underlying model
        if _SUPPORTS_KERNEL_OBJECT:
            # Older model likely accepts (kernel, reduced_rank_reg, rank, ...)
            super().__init__(kernel=kernel, reduced_rank_reg=True, rank=rank, **kwargs)
        else:
            # New documented model (NystroemKernelRidge) accepts sklearn-like args
            super().__init__(
                n_components=rank,
                reduced_rank=reduced_rank,
                kernel=kernel,         # "rbf"
                gamma=gamma,
                alpha=alpha,
                eigen_solver=eigen_solver,
                n_centers=n_centers,
                random_state=random_state,
                lag_time=delay_interval,  # close analogue; you already manage delay embedding
                **kwargs,
            )

    def fit(self, data=None, **kwargs):
        data = self.data if data is None else data
        # For the new API, fit(data) is enough; for the old API your old code did hankel itself.
        # Keep it simple: rely on kooplearn's internal handling.
        return super().fit(data, **kwargs)


def _num_available_contexts(x, n_delays: int, delay_interval: int) -> int:
    """Conservative count of usable transition/context rows."""
    x = np.asarray(x)
    if x.ndim == 2:
        T = x.shape[0]
        return max(1, T - (int(n_delays) - 1) * int(delay_interval) - 1)
    if x.ndim == 3:
        T = x.shape[1]
        return max(1, x.shape[0] * (T - (int(n_delays) - 1) * int(delay_interval) - 1))
    return max(1, x.shape[0] - 1)


def fit_kernel_dmd(x, n_delays: int, rank: int, delay_interval: int = 1, **kwargs):
    # Keep kooplearn's randomized Nyström model in a feasible regime for short
    # trajectories. This prevents n_centers/n_components from exceeding the
    # number of available contexts.
    n_contexts = _num_available_contexts(x, n_delays=n_delays, delay_interval=delay_interval)
    rank = int(max(1, min(int(rank), int(n_contexts))))
    kwargs.setdefault("n_centers", int(max(rank, min(600, n_contexts))))

    dmd = KernelDMD(
        x,
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=rank,
        **kwargs,
    )
    dmd.fit()
    # Your downstream code expects an operator matrix. Depending on kooplearn version,
    # this may differ. The simplest robust output is to return the learned Koopman matrix
    # if exposed; otherwise raise a clear error.
    if hasattr(dmd, "A_v"):
        return dmd.A_v
    if hasattr(dmd, "koopman_operator_"):
        return dmd.koopman_operator_
    raise AttributeError(
        "KernelDMD fit succeeded, but no operator attribute was found. "
        "Expected one of: A_v, koopman_operator_. Inspect the fitted object to adapt."
    )
