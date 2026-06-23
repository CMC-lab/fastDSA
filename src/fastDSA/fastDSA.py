"""
fastDSA.py

fastDSA with default automatic eigendelay q_star selection.

Default behavior
----------------
If the user does not provide n_delays or delay_interval, fastDSA automatically
selects Hankel parameters using q_star:

    q_star selection: smallest q near the best holdout NRMSE
    rank selection:   SVHT on the eigendelay/SVD spectrum

Manual behavior
---------------
If the user provides n_delays or delay_interval, fastDSA switches to manual
Hankel parameters unless use_q_star=True is explicitly set.

Examples
--------
Default automatic q_star:

    model = fastDSA(X, Y)

Manual delay embedding:

    model = fastDSA(X, Y, n_delays=50, delay_interval=2)

Force q_star even if manual-like values are present:

    model = fastDSA(X, Y, use_q_star=True, q_vals=range(20, 301, 20))
"""

from __future__ import annotations

from typing import Literal, Optional, Iterable, Any

import numpy as np
import torch
from omegaconf.listconfig import ListConfig
from scipy.linalg import svdvals

try:
    from fastDSA.dmd import DMD, embed_signal_torch
    from fastDSA.kerneldmd import KernelDMD
    from fastDSA.simdist import SimilarityTransformDist
    from fastDSA.q_star import select_q_star_for_collection, svht_threshold
except Exception:  # pragma: no cover - fallback for package-local imports
    from .dmd import DMD, embed_signal_torch
    from .kerneldmd import KernelDMD
    from .simdist import SimilarityTransformDist
    from .q_star import select_q_star_for_collection, svht_threshold


def svht(X, sv=None):
    """
    Backward-compatible SVHT threshold function.

    Kept here because older code may import `svht` from fastDSA.py.
    The implementation delegates to q_star.svht_threshold.
    """
    return svht_threshold(X, sv=sv)


class fastDSA:
    """
    Computes fast dynamical similarity analysis between two sets of data.

    This version uses automatic eigendelay q_star selection by default.

    Parameters most relevant to Hankel selection
    --------------------------------------------
    use_q_star : bool or None, default None
        If None, q_star is used only when both n_delays and delay_interval are
        omitted. If True, q_star is forced. If False, manual parameters are used.
    n_delays : int, list, or None, default None
        Manual number of Hankel rows. If provided and use_q_star is None,
        q_star is disabled.
    delay_interval : int, list, or None, default None
        Manual delay spacing. If provided and use_q_star is None, q_star is
        disabled.
    q_vals : iterable or None
        Candidate q values for q_star. Defaults to range(20, 201, 10).
    q_nrmse_tol : float
        q_star plateau tolerance. q_star chooses the smallest q with NRMSE
        <= (1 + q_nrmse_tol) * best_NRMSE.
    q_shared : bool
        If True, all datasets use one shared q after per-dataset diagnostics.
    q_rank_strategy : {'max', 'median'}
        How to convert per-dataset SVHT ranks to a shared rank.
    """

    def __init__(
        self,
        X,
        Y=None,
        n_delays=None,
        delay_interval=None,
        use_q_star: Optional[bool] = None,
        q_vals: Optional[Iterable[int]] = None,
        q_train_ratio: float = 0.7,
        q_nrmse_tol: float = 0.05,
        q_roughness_quantile: Optional[float] = 0.75,
        q_min_rank: int = 2,
        q_max_rank_cap: Optional[int] = None,
        q_delay_interval: int = 1,
        q_shared: bool = True,
        q_shared_strategy: Literal["median", "max", "min"] = "median",
        q_rank_strategy: Literal["max", "median"] = "max",
        rank=None,
        rank_thresh=None,
        rank_explained_variance=None,
        lamb=0.0,
        send_to_cpu=True,
        iters=1500,
        score_method: Literal["angular", "euclidean", "wasserstein"] = "euclidean",
        lr=5e-3,
        group: Literal["GL(n)", "O(n)", "SO(n)"] = "O(n)",
        zero_pad=False,
        device="cpu",
        verbose=False,
        reduced_rank_reg=False,
        kernel=None,
        num_centers=0.1,
        svd_solver="arnoldi",
        wasserstein_compare: Literal["sv", "eig", None] = None,
    ):
        self.X = X
        self.Y = Y
        self.check_method()

        if self.method == "self-pairwise":
            self.data = [self.X]
        else:
            self.data = [self.X, self.Y]

        # Decide whether to use q_star.
        manual_hankel_requested = (n_delays is not None) or (delay_interval is not None)
        if use_q_star is None:
            self.use_q_star = not manual_hankel_requested
        else:
            self.use_q_star = bool(use_q_star)

        self.user_provided_rank = rank is not None

        # Manual fallback defaults.
        if self.use_q_star:
            # Placeholders. They will be overwritten by auto_select_q_star().
            if n_delays is None:
                n_delays = 1
            if delay_interval is None:
                delay_interval = int(q_delay_interval)
        else:
            if n_delays is None:
                n_delays = 1
            if delay_interval is None:
                delay_interval = 1

        # Broadcast parameters to match data structure.
        self.n_delays = self.broadcast_params(n_delays, cast=int)
        self.delay_interval = self.broadcast_params(delay_interval, cast=int)
        self.rank = self.broadcast_params(rank, cast=int)
        self.rank_thresh = self.broadcast_params(rank_thresh)
        self.rank_explained_variance = self.broadcast_params(rank_explained_variance)
        self.lamb = self.broadcast_params(lamb)

        self.send_to_cpu = send_to_cpu
        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self.zero_pad = zero_pad
        self.group = group
        self.reduced_rank_reg = reduced_rank_reg
        self.kernel = kernel
        self.num_centers = num_centers
        self.svd_solver = svd_solver
        self.wasserstein_compare = wasserstein_compare

        # q_star settings.
        self.q_vals = list(q_vals) if q_vals is not None else list(range(20, 201, 10))
        self.q_train_ratio = q_train_ratio
        self.q_nrmse_tol = q_nrmse_tol
        self.q_roughness_quantile = q_roughness_quantile
        self.q_min_rank = q_min_rank
        self.q_max_rank_cap = q_max_rank_cap
        self.q_delay_interval = int(q_delay_interval)
        self.q_shared = q_shared
        self.q_shared_strategy = q_shared_strategy
        self.q_rank_strategy = q_rank_strategy

        self.q_star_result = None
        self.hankel_selection_table = None
        self.hankel_diagnostics_table = None

        if self.use_q_star:
            self.auto_select_q_star()

        self.dmds = self._make_dmds()

        self.simdist = SimilarityTransformDist(
            iters,
            score_method,
            lr,
            device,
            verbose,
            group,
            lambda_reg=0.01,
        )

    # ------------------------------------------------------------------
    # Data/method handling
    # ------------------------------------------------------------------
    def check_method(self):
        """Determine comparison method and normalize X/Y into lists."""
        tensor_or_np = lambda x: isinstance(x, (np.ndarray, torch.Tensor))

        if isinstance(self.X, list):
            if self.Y is None:
                self.method = "self-pairwise"
            elif isinstance(self.Y, list):
                self.method = "bipartite-pairwise"
            elif tensor_or_np(self.Y):
                self.method = "list-to-one"
                self.Y = [self.Y]
            else:
                raise ValueError("unknown type of Y")

        elif tensor_or_np(self.X):
            self.X = [self.X]
            if self.Y is None:
                raise ValueError("only one element provided; provide Y or pass X as a list for self-pairwise")
            elif isinstance(self.Y, list):
                self.method = "one-to-list"
            elif tensor_or_np(self.Y):
                self.method = "default"
                self.Y = [self.Y]
            else:
                raise ValueError("unknown type of Y")
        else:
            raise ValueError("unknown type of X")

    def broadcast_params(self, param, cast=None):
        """Broadcast scalar/list parameters to match self.X/self.Y structure."""
        out = []

        is_scalar = isinstance(param, (int, float, np.integer, np.floating)) or param is None

        if is_scalar:
            out.append([param] * len(self.X))
            if self.Y is not None:
                out.append([param] * len(self.Y))

        elif isinstance(param, (tuple, list, np.ndarray, ListConfig)):
            if self.method == "self-pairwise" and len(param) >= len(self.X):
                out = [list(param)[: len(self.X)]]
            else:
                if len(param) > 2:
                    raise AssertionError("parameter lists can have at most two top-level elements")
                for i, data in enumerate([self.X, self.Y]):
                    if data is None:
                        continue
                    if isinstance(param[i], (int, float, np.integer, np.floating)) or param[i] is None:
                        out.append([param[i]] * len(data))
                    elif isinstance(param[i], (list, np.ndarray, tuple, ListConfig)):
                        if len(param[i]) < len(data):
                            raise AssertionError("nested parameter list is shorter than data list")
                        out.append(list(param[i])[: len(data)])
                    else:
                        raise ValueError("unknown nested parameter type")
        else:
            raise ValueError("unknown type entered for parameter")

        if cast is not None and param is not None:
            out = [[cast(x) if x is not None else None for x in dat] for dat in out]

        return out

    # ------------------------------------------------------------------
    # q_star integration
    # ------------------------------------------------------------------
    def auto_select_q_star(self):
        """
        Automatically select n_delays/q and rank using q_star.

        Updates:
            self.n_delays
            self.delay_interval
            self.rank, unless user explicitly provided rank
            self.q_star_result
            self.hankel_selection_table
            self.hankel_diagnostics_table
        """
        result = select_q_star_for_collection(
            self.data,
            q_vals=self.q_vals,
            delay_interval=self.q_delay_interval,
            train_ratio=self.q_train_ratio,
            min_rank=self.q_min_rank,
            max_rank_cap=self.q_max_rank_cap,
            nrmse_tol=self.q_nrmse_tol,
            roughness_quantile=self.q_roughness_quantile,
            shared_q=self.q_shared,
            shared_q_strategy=self.q_shared_strategy,
            rank_strategy=self.q_rank_strategy,
            device=self.device,
            verbose=self.verbose,
        )

        self.q_star_result = result
        self.hankel_selection_table = result["selected_rows"]
        self.hankel_diagnostics_table = result["diagnostic_rows"]

        self.n_delays = result["n_delays"]
        self.delay_interval = result["delay_interval"]

        if not self.user_provided_rank:
            self.rank = result["rank"]

        if self.verbose:
            print("[fastDSA] q_star selected n_delays:", self.n_delays)
            print("[fastDSA] q_star selected delay_interval:", self.delay_interval)
            print("[fastDSA] q_star selected rank:", self.rank)

        return result

    # ------------------------------------------------------------------
    # DMD construction/fitting
    # ------------------------------------------------------------------
    def _make_dmds(self):
        """Create DMD or KernelDMD objects from current parameters."""
        if self.kernel is None:
            return [
                [
                    DMD(
                        Xi,
                        self.n_delays[i][j],
                        delay_interval=self.delay_interval[i][j],
                        rank=self.rank[i][j],
                        rank_thresh=self.rank_thresh[i][j],
                        rank_explained_variance=self.rank_explained_variance[i][j],
                        reduced_rank_reg=self.reduced_rank_reg,
                        lamb=self.lamb[i][j],
                        device=self.device,
                        verbose=self.verbose,
                        send_to_cpu=self.send_to_cpu,
                    )
                    for j, Xi in enumerate(dat)
                ]
                for i, dat in enumerate(self.data)
            ]

        return [
            [
                KernelDMD(
                    Xi,
                    self.n_delays[i][j],
                    kernel=self.kernel,
                    num_centers=self.num_centers,
                    delay_interval=self.delay_interval[i][j],
                    rank=self.rank[i][j],
                    reduced_rank_reg=self.reduced_rank_reg,
                    lamb=self.lamb[i][j],
                    verbose=self.verbose,
                    svd_solver=self.svd_solver,
                )
                for j, Xi in enumerate(dat)
            ]
            for i, dat in enumerate(self.data)
        ]

    def fit_dmds(
        self,
        X=None,
        Y=None,
        n_delays=None,
        delay_interval=None,
        rank=None,
        rank_thresh=None,
        rank_explained_variance=None,
        reduced_rank_reg=None,
        lamb=None,
        device=None,
        verbose=None,
        send_to_cpu=None,
        use_q_star: Optional[bool] = None,
    ):
        """
        Recompute DMDs with optionally overridden data/parameters.

        This method updates `self.dmds` and returns the newly fitted DMD list.
        """
        if X is not None:
            self.X = X
        if Y is not None:
            self.Y = Y

        if X is not None or Y is not None:
            self.check_method()
            self.data = [self.X] if self.method == "self-pairwise" else [self.X, self.Y]

        if n_delays is not None:
            self.n_delays = self.broadcast_params(n_delays, cast=int)
        if delay_interval is not None:
            self.delay_interval = self.broadcast_params(delay_interval, cast=int)
        if rank is not None:
            self.rank = self.broadcast_params(rank, cast=int)
            self.user_provided_rank = True
        if rank_thresh is not None:
            self.rank_thresh = self.broadcast_params(rank_thresh)
        if rank_explained_variance is not None:
            self.rank_explained_variance = self.broadcast_params(rank_explained_variance)
        if reduced_rank_reg is not None:
            self.reduced_rank_reg = reduced_rank_reg
        if lamb is not None:
            self.lamb = self.broadcast_params(lamb)
        if device is not None:
            self.device = device
        if verbose is not None:
            self.verbose = verbose
        if send_to_cpu is not None:
            self.send_to_cpu = send_to_cpu

        do_q_star = self.use_q_star if use_q_star is None else bool(use_q_star)
        if do_q_star:
            self.auto_select_q_star()

        self.dmds = self._make_dmds()

        for dmd_sets in self.dmds:
            for dmd in dmd_sets:
                dmd.fit()

        return self.dmds

    def fit_score(self):
        """Fit all DMD models and then compute the similarity score."""
        for dmd_sets in self.dmds:
            for dmd in dmd_sets:
                dmd.fit()
        return self.score()

    def score(self, iters=None, lr=None, score_method=None):
        """Compute similarity score using already fitted DMDs."""
        iters = self.iters if iters is None else iters
        lr = self.lr if lr is None else lr
        score_method = self.score_method if score_method is None else score_method

        ind2 = 1 - int(self.method == "self-pairwise")
        self.sims = np.zeros((len(self.dmds[0]), len(self.dmds[ind2])))

        for i, dmd1 in enumerate(self.dmds[0]):
            for j, dmd2 in enumerate(self.dmds[ind2]):
                if self.method == "self-pairwise" and j >= i:
                    continue

                if self.verbose:
                    print(f"Computing similarity between DMDs {i} and {j}")

                self.sims[i, j] = self.simdist.fit_score(
                    dmd1.A_v,
                    dmd2.A_v,
                    iters,
                    lr,
                    score_method,
                    zero_pad=self.zero_pad,
                )

                if self.method == "self-pairwise":
                    self.sims[j, i] = self.sims[i, j]

        if self.method == "default":
            return self.sims[0, 0]

        return self.sims
