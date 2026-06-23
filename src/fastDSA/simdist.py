# src/fastDSA/simdist.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union, Any, List

import numpy as np
import torch

from .dmd import DMD
from .kwdsa import fit_kernel_dmd, compute_wasserstein_distance as compute_kw_wasserstein_distance

# Each of these files defines a class named SimilarityTransformDist.
# We import and alias them to avoid name collisions.
from .RegularizationTerm import SimilarityTransformDist as RegularizationSimilarityTransformDist
from .RiemannianManifold import SimilarityTransformDist as RiemannianSimilarityTransformDist
from .LandingAlgorithm import SimilarityTransformDist as LandingSimilarityTransformDist

try:
    from .q_star import select_q_star_for_collection
except Exception:  # pragma: no cover - allows import diagnostics if q_star.py is missing
    select_q_star_for_collection = None


ArrayLike = Union[np.ndarray, torch.Tensor]
MethodType = Literal["ro", "rim", "land", "kw"]
SharedQStrategy = Literal["median", "max", "min"]
RankStrategy = Literal["max", "median"]


# ---------------------------------------------------------------------------
# Delay embedding utilities
# ---------------------------------------------------------------------------

def delay_embed_traj(traj: np.ndarray, n_delays: int, delay_interval: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (X, Y) Hankel blocks from a single trajectory.

    Parameters
    ----------
    traj : np.ndarray
        2D array of shape (T, n) where T = timepoints, n = channels.
    n_delays : int
        Number of Hankel rows / delay vectors. In q_star mode this is q_star.
    delay_interval : int
        Spacing between raw delays. q_star defaults to consecutive delays, i.e. 1.

    Returns
    -------
    X, Y : np.ndarray
        X, Y both of shape (n * n_delays, L - 1),
        with L = T - (n_delays - 1) * delay_interval.
    """
    traj = np.asarray(traj)
    if traj.ndim != 2:
        raise ValueError(f"traj must be 2D (T, n); got shape {traj.shape}")

    n_delays = int(n_delays)
    delay_interval = int(delay_interval)
    if n_delays < 1:
        raise ValueError("n_delays must be >= 1")
    if delay_interval < 1:
        raise ValueError("delay_interval must be >= 1")

    T, n = traj.shape
    L = T - (n_delays - 1) * delay_interval
    if L <= 1:
        raise ValueError(
            f"Not enough time points (T={T}) for n_delays={n_delays}, delay_interval={delay_interval}."
        )

    # Hankel-like delayed matrix Z of shape (L, n * n_delays)
    Z = np.zeros((L, n * n_delays), dtype=float)
    for d in range(n_delays):
        start = (n_delays - 1 - d) * delay_interval
        end = start + L
        Z[:, d * n : (d + 1) * n] = traj[start:end, :]

    X = Z[:-1, :].T  # (n * n_delays, L - 1)
    Y = Z[1:, :].T   # (n * n_delays, L - 1)
    return X, Y


# ---------------------------------------------------------------------------
# SVHT rank selection
# ---------------------------------------------------------------------------

def omega_beta(beta: float) -> float:
    """Gavish-Donoho cubic approximation for SVHT."""
    return float(0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43)


def svht_threshold(M: np.ndarray, s: Optional[np.ndarray] = None) -> float:
    """
    Gavish-Donoho SVHT threshold tau = omega(beta) * median(s).
    """
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError(f"svht_threshold expects a 2D matrix; got {M.shape}")
    m, n = M.shape
    beta = min(m, n) / max(m, n)
    if s is None:
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    tau = omega_beta(beta) * np.median(s)
    return float(tau)


def svht_rank_single(M: np.ndarray, min_rank: int = 1) -> int:
    """Rank of a single matrix M using GD-SVHT."""
    s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    tau = svht_threshold(M, s=s)
    r = int((s > tau).sum())
    return max(r, int(min_rank))


def svht_ranks_for_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    which: Literal["X", "Y", "concat"] = "X",
    min_rank: int = 1,
) -> Sequence[int]:
    """
    pairs: list of (X, Y)
    which: "X" | "Y" | "concat" (concat = [X | Y] horizontally)
    returns: list of ranks (one per pair)
    """
    ranks = []
    for X, Y in pairs:
        if which == "X":
            M = X
        elif which == "Y":
            M = Y
        elif which == "concat":
            M = np.hstack([X, Y])
        else:
            raise ValueError("which must be 'X', 'Y', or 'concat'")
        ranks.append(svht_rank_single(M, min_rank=min_rank))
    return ranks


# ---------------------------------------------------------------------------
# DMD utilities
# ---------------------------------------------------------------------------

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch or array-like to a NumPy array on CPU."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _default_device_str(device: Optional[str]) -> str:
    if device is not None:
        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_rank_for_traj(
    traj: np.ndarray,
    n_delays: int,
    delay_interval: int,
    desired_rank: int,
    steps_ahead: int = 1,
) -> int:
    """
    Cap the desired rank so it is compatible with the trajectory and Hankel dimension.
    """
    traj = np.asarray(traj)
    if traj.ndim != 2:
        raise ValueError(f"traj must be 2D (T, n) inside _safe_rank_for_traj, got {traj.shape}")

    T, n = traj.shape
    L = T - (int(n_delays) - 1) * int(delay_interval) - int(steps_ahead)
    if L <= 0:
        raise ValueError(
            f"Not enough time points (T={T}) for n_delays={n_delays}, delay_interval={delay_interval}, "
            f"steps_ahead={steps_ahead}."
        )
    hankel_dim = n * int(n_delays)
    max_rank = min(hankel_dim, L)
    return int(max(1, min(int(desired_rank), max_rank)))


def _cap_global_rank_for_trajs(
    trajs: Sequence[np.ndarray],
    n_delays: int,
    delay_interval: int,
    desired_rank: int,
    steps_ahead: int = 1,
) -> int:
    """Cap a shared rank so all trajectories can use the same operator dimension."""
    if len(trajs) == 0:
        raise ValueError("No trajectories supplied for rank capping.")
    caps = [
        _safe_rank_for_traj(
            tr,
            n_delays=n_delays,
            delay_interval=delay_interval,
            desired_rank=10**12,
            steps_ahead=steps_ahead,
        )
        for tr in trajs
    ]
    return int(max(1, min(int(desired_rank), min(caps))))


def compute_dmd_matrix_for_traj(
    traj: np.ndarray,
    n_delays: int,
    delay_interval: int,
    desired_rank: int,
    device: str,
    steps_ahead: int = 1,
) -> Tuple[torch.Tensor, int]:
    """
    Returns:
      A_op: torch.Tensor on `device`
      used_rank: int

    This wrapper is compatible with the current DMD implementation which
    expects `self.data` (torch.Tensor) to be set before compute_hankel().
    """
    if isinstance(traj, np.ndarray) and traj.dtype != np.float32:
        traj = traj.astype(np.float32)

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    used_rank = _safe_rank_for_traj(traj, n_delays, delay_interval, desired_rank, steps_ahead)

    # IMPORTANT: your DMD expects torch tensor stored in self.data.
    traj_t = torch.as_tensor(traj, dtype=torch.float32, device=device)

    dmd = DMD(
        traj_t,
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=None,
        reduced_rank_reg=False,
        lamb=0.0,
        device=device,
        verbose=False,
        send_to_cpu=False,
        steps_ahead=steps_ahead,
    )

    # Also ensure attribute exists even if __init__ doesn't set it.
    dmd.data = traj_t

    # Many DMD implementations ignore the passed `data` and rely on self.data,
    # but we pass it anyway for compatibility.
    dmd.fit(
        data=traj_t,
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=used_rank,
        device=device,
    )

    # Your code uses A_havok_dmd as the operator.
    A = dmd.A_havok_dmd
    return A, used_rank


def _aggregate_operators(ops: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Aggregate multiple DMD operators into a single representative operator.
    Currently: element-wise mean over trajectories.
    """
    if len(ops) == 0:
        raise ValueError("No operators to aggregate.")
    if len(ops) == 1:
        return ops[0]
    shapes = {tuple(op.shape) for op in ops}
    if len(shapes) != 1:
        raise ValueError(
            "Cannot aggregate DMD operators with different shapes. "
            f"Shapes found: {sorted(shapes)}. Use shared q/rank settings."
        )
    stacked = torch.stack(ops, dim=0)
    return stacked.mean(dim=0)


# ---------------------------------------------------------------------------
# Config and main class
# ---------------------------------------------------------------------------

@dataclass
class SimDistConfig:
    """
    Configuration for the similarity pipeline.

    Default behavior
    ----------------
    If `n_delays` is None, the pipeline automatically runs q_star-SVHT and uses
    the selected q_star as the number of Hankel rows. If the user supplies
    `n_delays`, the pipeline switches to manual delay embedding.

    method:
        'ro'   -> RegularizationTerm-based SimilarityTransformDist
        'rim'  -> Riemannian manifold optimizer
        'land' -> Landing-style optimizer
        'kw'   -> Kernel DMD + Wasserstein distance between spectra
    """
    # Delay/DMD controls. None means automatic q_star mode.
    n_delays: Optional[int] = None
    delay_interval: Optional[int] = None
    rank: Optional[int] = None
    method: MethodType = "ro"

    # q_star controls. Used when use_q_star=True, or when use_q_star is None
    # and n_delays is None.
    use_q_star: Optional[bool] = None
    q_min: Optional[int] = None
    q_prop_constant: float = 2.0
    q_prop_exponent: float = 1.0 / 3.0
    q_max_acf_lag_fraction: float = 0.25
    q_period_prominence_ratio: float = 8.0
    q_period_min_cycles: int = 3
    q_period_min_period: int = 8
    q_min_svht_rank: int = 1
    q_max_rank_cap: Optional[int] = None
    q_shared: bool = True
    q_shared_strategy: SharedQStrategy = "min"
    q_rank_strategy: RankStrategy = "max"
    q_time_axis: Optional[int] = 0  # trajs are converted to (T, channels)

    # Common optimization knobs
    iters: int = 500
    lr: float = 1e-2

    # Landing-specific (optional; eta defaults to lr if None)
    eta: Optional[float] = None
    gamma: float = 0.98
    n_Cmats: int = 1

    # RegularizationTerm-specific
    ro_score_method: Literal["angular", "frobenius"] = "angular"
    ro_lambda_reg: float = 0.01
    ro_group: str = "O(n)"

    # RiemannianManifold-specific
    rim_score_method: Literal["angular", "frobenius", "wasserstein"] = "angular"
    rim_wasserstein_compare: Literal["eig", "sv"] = "eig"
    rim_normalize: bool = True
    rim_so: bool = False
    rim_init: Literal["orthogonal", "identity", "random"] = "identity"

    # LandingAlgorithm-specific
    land_score_method: Literal["angular", "frobenius", "wasserstein"] = "angular"
    land_wasserstein_spectrum: str = "eig"
    land_normalize_fro: bool = True

    # Kernel-Wasserstein-specific
    kw_use_sv: bool = False  # if True, you could extend kw to use singular values

    # Misc
    device: str = "cuda"
    steps_ahead: int = 1
    svht_which: Literal["X", "Y", "concat"] = "X"
    verbose: bool = False


class FastDSASimilarity:
    """
    Main entrypoint for fastDSA similarity.

    Workflow:
      1. Accept datasets A and B as (channels, timepoints) or batches thereof.
      2. If n_delays is None, automatically select q_star and r_star using
         q_star-SVHT with no BIC/RMSE/grid search.
      3. Build Hankel (X, Y) pairs for each trajectory via delay_embed_traj.
      4. If rank is still None, compute global_max_rank using GD-SVHT on the
         actual Hankel pairs.
      5. For each trajectory, compute a DMD operator A.
      6. Depending on `method`:
         - 'ro'   -> RegularizationTerm.SimilarityTransformDist
         - 'rim'  -> RiemannianManifold.SimilarityTransformDist
         - 'land' -> LandingAlgorithm.SimilarityTransformDist
         - 'kw'   -> KernelDMD + Wasserstein distance between eigenvalues
    """

    def __init__(self, config: SimDistConfig):
        self.config = config

        self.global_max_rank_: Optional[int] = None
        self.ranks_A_: Optional[Sequence[int]] = None
        self.ranks_B_: Optional[Sequence[int]] = None
        self.score_: Optional[float] = None

        self.used_q_star_: bool = False
        self.selected_n_delays_: Optional[int] = None
        self.selected_delay_interval_: Optional[int] = None
        self.q_star_result_: Optional[dict] = None
        self.hankel_selection_table_: Optional[Sequence[dict]] = None
        self.hankel_diagnostics_table_: Optional[Sequence[dict]] = None

    # -------------------- public API -------------------- #

    def fit_score(self, data_A: ArrayLike, data_B: ArrayLike) -> Tuple[float, int]:
        """
        Compute similarity score between two datasets.

        Parameters
        ----------
        data_A, data_B
            Either:
              - np.ndarray or torch.Tensor of shape (channels, timepoints)
              - np.ndarray/torch.Tensor of shape (n_traj, channels, timepoints)
              - sequence of 2D arrays/tensors, each (channels, timepoints)

        Returns
        -------
        score : float
            Similarity score (definition depends on `method`).
        used_rank : int
            Global rank used across all trajectories.
        """
        method = self.config.method.lower()
        if method not in ("ro", "rim", "land", "kw"):
            raise ValueError(f"Unknown method '{self.config.method}'. Must be one of: 'ro', 'rim', 'land', 'kw'.")

        if method == "kw":
            score, used_rank = self._fit_score_kw(data_A, data_B)
        else:
            score, used_rank = self._fit_score_operator_based(data_A, data_B)

        self.score_ = float(score)
        self.global_max_rank_ = int(used_rank)
        return self.score_, self.global_max_rank_

    # -------------------- internal helpers -------------------- #

    def _should_use_q_star(self) -> bool:
        if self.config.use_q_star is not None:
            return bool(self.config.use_q_star)
        return self.config.n_delays is None

    def _prepare_trajs_list(self, data: ArrayLike) -> Sequence[np.ndarray]:
        """
        Convert input data into a list of trajectories, each shaped (T, n).

        Accepts:
          - 2D (channels, timepoints)
          - 3D (n_traj, channels, timepoints)
          - list/tuple of 2D arrays/tensors (channels, timepoints)
        """
        if isinstance(data, (list, tuple)):
            trajs = []
            for d in data:
                arr = _to_numpy(d)
                if arr.ndim != 2:
                    raise ValueError("Each trajectory in a list must be 2D (channels, timepoints).")
                C, T = arr.shape
                trajs.append(arr.T.copy())  # (T, C)
            return trajs

        arr = _to_numpy(data)
        if arr.ndim == 2:
            C, T = arr.shape
            return [arr.T.copy()]  # single trajectory (T, C)
        elif arr.ndim == 3:
            # (n_traj, channels, timepoints) -> list[(T, C)]
            n_traj, C, T = arr.shape
            return [arr[i].T.copy() for i in range(n_traj)]
        else:
            raise ValueError(
                "data must be 2D (channels, timepoints), 3D (n_traj, channels, timepoints), "
                "or a list/tuple of 2D arrays."
            )

    def _build_pairs(
        self,
        trajs: Sequence[np.ndarray],
        n_delays: int,
        delay_interval: int,
    ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """Compute (X, Y) Hankel pairs for each trajectory."""
        pairs = []
        for traj in trajs:
            X, Y = delay_embed_traj(traj, int(n_delays), int(delay_interval))
            pairs.append((X, Y))
        return pairs

    def _auto_detect_rank(
        self,
        pairs_A: Sequence[Tuple[np.ndarray, np.ndarray]],
        pairs_B: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> int:
        """Compute global_max_rank from SVHT on (X, Y) pairs of both datasets."""
        ranks_A = svht_ranks_for_pairs(pairs_A, which=self.config.svht_which, min_rank=self.config.q_min_svht_rank)
        ranks_B = svht_ranks_for_pairs(pairs_B, which=self.config.svht_which, min_rank=self.config.q_min_svht_rank)
        global_max_rank = max(max(ranks_A), max(ranks_B))

        self.ranks_A_ = ranks_A
        self.ranks_B_ = ranks_B
        return int(global_max_rank)

    def _resolve_embedding_params(
        self,
        trajs_A: Sequence[np.ndarray],
        trajs_B: Sequence[np.ndarray],
    ) -> Tuple[int, int, Optional[int]]:
        """
        Resolve n_delays, delay_interval, and optional q_star rank.

        Returns
        -------
        n_delays : int
        delay_interval : int
        q_star_rank : Optional[int]
            Rank suggested by q_star. None in manual mode.
        """
        delay_interval = int(self.config.delay_interval) if self.config.delay_interval is not None else 1

        if not self._should_use_q_star():
            if self.config.n_delays is None:
                raise ValueError("Manual mode requires config.n_delays to be set.")
            n_delays = int(self.config.n_delays)
            self.used_q_star_ = False
            self.selected_n_delays_ = n_delays
            self.selected_delay_interval_ = delay_interval
            return n_delays, delay_interval, None

        if select_q_star_for_collection is None:
            raise ImportError(
                "q_star mode requested, but fastDSA.q_star could not be imported. "
                "Place q_star.py next to simdist.py inside the fastDSA package."
            )

        q_result = select_q_star_for_collection(
            [trajs_A, trajs_B],
            delay_interval=delay_interval,
            q_min=self.config.q_min,
            c_prop=float(self.config.q_prop_constant),
            q_prop_exponent=float(self.config.q_prop_exponent),
            max_acf_lag_fraction=float(self.config.q_max_acf_lag_fraction),
            period_prominence_ratio=float(self.config.q_period_prominence_ratio),
            dominant_period_min_cycles=int(self.config.q_period_min_cycles),
            min_period=int(self.config.q_period_min_period),
            min_rank=int(self.config.q_min_svht_rank),
            max_rank_cap=self.config.q_max_rank_cap,
            shared_q=bool(self.config.q_shared),
            shared_q_strategy=self.config.q_shared_strategy,
            rank_strategy=self.config.q_rank_strategy,
            time_axis=self.config.q_time_axis,
            device=_default_device_str(self.config.device),
            verbose=bool(self.config.verbose),
        )

        q_nested = q_result["n_delays"]
        r_nested = q_result["rank"]

        # This simdist wrapper aggregates operators, so it requires a shared q/r.
        # select_q_star_for_collection returns shared values by default.
        n_delays = int(q_nested[0][0])
        q_star_rank = int(r_nested[0][0])

        # Final safety guard: a shared q must be feasible for every trajectory.
        # This should already be enforced inside q_star.py, but keeping the
        # check here prevents impossible Hankel matrices if an older q_star.py
        # is accidentally imported.
        all_trajs = list(trajs_A) + list(trajs_B)
        q_feasible_by_traj = [
            max(1, ((int(tr.shape[0]) - 2) // delay_interval) + 1)
            for tr in all_trajs
        ]
        q_feasible_global = int(min(q_feasible_by_traj))
        if n_delays > q_feasible_global:
            if self.config.verbose:
                print(
                    f"[fastDSA] q_star={n_delays} is infeasible for the shortest "
                    f"trajectory; clipping to {q_feasible_global}."
                )
            n_delays = q_feasible_global
            # Rank will be capped after Hankel construction; avoid trusting a
            # rank estimated at an infeasible/larger q.
            q_star_rank = None

        self.used_q_star_ = True
        self.selected_n_delays_ = n_delays
        self.selected_delay_interval_ = delay_interval
        self.q_star_result_ = q_result
        self.hankel_selection_table_ = q_result.get("selected_rows", None)
        self.hankel_diagnostics_table_ = q_result.get("diagnostic_rows", None)

        if self.config.verbose:
            print(
                f"[fastDSA] q_star selected n_delays={n_delays}, "
                f"delay_interval={delay_interval}, rank={q_star_rank}"
            )

        return n_delays, delay_interval, q_star_rank

    # -------------------- operator-based path (ro / rim / land) -------------------- #

    def _fit_score_operator_based(self, data_A: ArrayLike, data_B: ArrayLike) -> Tuple[float, int]:
        """
        Path for methods 'ro', 'rim', 'land':
          - optionally select q_star/r_star,
          - build Hankel pairs,
          - detect global rank if needed,
          - compute DMD operators,
          - call the respective metric's fit_score.
        """
        device_str = _default_device_str(self.config.device)

        # 1) Convert to list of trajectories (T, n)
        trajs_A = self._prepare_trajs_list(data_A)
        trajs_B = self._prepare_trajs_list(data_B)
        all_trajs = list(trajs_A) + list(trajs_B)

        # 2) Resolve embedding parameters, defaulting to q_star-SVHT.
        n_delays, delay_interval, q_star_rank = self._resolve_embedding_params(trajs_A, trajs_B)

        # 3) Build Hankel pairs and detect global rank if needed.
        pairs_A = self._build_pairs(trajs_A, n_delays, delay_interval)
        pairs_B = self._build_pairs(trajs_B, n_delays, delay_interval)

        if self.config.rank is not None:
            global_rank = int(self.config.rank)
        elif q_star_rank is not None:
            global_rank = int(q_star_rank)
        else:
            global_rank = self._auto_detect_rank(pairs_A, pairs_B)
            if self.config.verbose:
                print(f"[fastDSA] Detected global rank via SVHT: {global_rank}")

        # Make sure all trajectories can use the same rank.
        global_rank = _cap_global_rank_for_trajs(
            all_trajs,
            n_delays=n_delays,
            delay_interval=delay_interval,
            desired_rank=global_rank,
            steps_ahead=self.config.steps_ahead,
        )

        if self.config.verbose:
            print(f"[fastDSA] Using n_delays={n_delays}, delay_interval={delay_interval}, rank={global_rank}")

        # 4) Compute DMD operators for all trajectories.
        A_ops, B_ops = [], []
        for tr in trajs_A:
            A_op, _ = compute_dmd_matrix_for_traj(
                tr,
                n_delays=n_delays,
                delay_interval=delay_interval,
                desired_rank=global_rank,
                device=device_str,
                steps_ahead=self.config.steps_ahead,
            )
            A_ops.append(A_op)

        for tr in trajs_B:
            B_op, _ = compute_dmd_matrix_for_traj(
                tr,
                n_delays=n_delays,
                delay_interval=delay_interval,
                desired_rank=global_rank,
                device=device_str,
                steps_ahead=self.config.steps_ahead,
            )
            B_ops.append(B_op)

        # Aggregate operators into a single representative operator per dataset.
        A_mean = _aggregate_operators(A_ops)
        B_mean = _aggregate_operators(B_ops)

        # 5) Instantiate and run the chosen metric.
        method = self.config.method.lower()

        if method == "ro":
            metric = RegularizationSimilarityTransformDist(
                iters=self.config.iters,
                score_method=self.config.ro_score_method,
                lr=self.config.lr,
                device=device_str,
                verbose=self.config.verbose,
                group=self.config.ro_group,
                lambda_reg=self.config.ro_lambda_reg,
                rank=None,
            )
            score = metric.fit_score(A_mean, B_mean)

        elif method == "rim":
            rim_device = torch.device(device_str)
            metric = RiemannianSimilarityTransformDist(
                iters=self.config.iters,
                lr=self.config.lr,
                verbose=self.config.verbose,
                device=rim_device,
                normalize=self.config.rim_normalize,
                so=self.config.rim_so,
                init=self.config.rim_init,
                score_method=self.config.rim_score_method,
                wasserstein_compare=self.config.rim_wasserstein_compare,
            )
            # rim's fit_score can optionally return (score, time); we use score only.
            out = metric.fit_score(A_mean, B_mean)
            score = out[0] if isinstance(out, tuple) else out

        elif method == "land":
            eta = self.config.eta if self.config.eta is not None else self.config.lr
            metrics_list = (self.config.land_score_method,)

            metric = LandingSimilarityTransformDist(
                its=self.config.iters,
                eta=eta,
                gamma=self.config.gamma,
                n_Cmats=self.config.n_Cmats,
                verbose=self.config.verbose,
                device=device_str,
                metrics=metrics_list,
                wasserstein_spectrum=self.config.land_wasserstein_spectrum,
                normalize_fro=self.config.land_normalize_fro,
            )
            score = metric.fit_score(A_mean, B_mean, method=self.config.land_score_method)

        else:
            raise RuntimeError("Internal error: _fit_score_operator_based called with invalid method.")

        return float(score), int(global_rank)

    # -------------------- kwDSA path (kernel Wasserstein) -------------------- #

    def _to_trajs_3d_for_kw_from_list(self, trajs: Sequence[np.ndarray]) -> np.ndarray:
        """Stack list[(T, C)] into (n_traj, T, C) for KernelDMD."""
        if len(trajs) == 0:
            raise ValueError("No trajectories supplied for kwDSA.")
        shapes = {tuple(t.shape) for t in trajs}
        if len(shapes) != 1:
            raise ValueError(
                "kwDSA path requires all trajectories in a dataset to have the same shape. "
                f"Shapes found: {sorted(shapes)}"
            )
        return np.stack(trajs, axis=0)

    def _fit_score_kw(self, data_A: ArrayLike, data_B: ArrayLike) -> Tuple[float, int]:
        """
        Kernel-DMD + Wasserstein distance between eigenvalue distributions.
        """
        trajs_A = self._prepare_trajs_list(data_A)
        trajs_B = self._prepare_trajs_list(data_B)
        all_trajs = list(trajs_A) + list(trajs_B)

        n_delays, delay_interval, q_star_rank = self._resolve_embedding_params(trajs_A, trajs_B)

        # Detect rank if needed using the selected q.
        if self.config.rank is not None:
            global_rank = int(self.config.rank)
        elif q_star_rank is not None:
            global_rank = int(q_star_rank)
            if self.config.verbose:
                print(f"[fastDSA-kw] Using q_star SVHT rank: {global_rank}")
        else:
            pairs_A = self._build_pairs([trajs_A[0]], n_delays, delay_interval)
            pairs_B = self._build_pairs([trajs_B[0]], n_delays, delay_interval)
            global_rank = self._auto_detect_rank(pairs_A, pairs_B)
            if self.config.verbose:
                print(f"[fastDSA-kw] Detected rank via SVHT: {global_rank}")

        global_rank = _cap_global_rank_for_trajs(
            all_trajs,
            n_delays=n_delays,
            delay_interval=delay_interval,
            desired_rank=global_rank,
            steps_ahead=self.config.steps_ahead,
        )

        if self.config.verbose:
            print(f"[fastDSA-kw] Using n_delays={n_delays}, delay_interval={delay_interval}, rank={global_rank}")

        trajs_A_3d = self._to_trajs_3d_for_kw_from_list(trajs_A)
        trajs_B_3d = self._to_trajs_3d_for_kw_from_list(trajs_B)

        # Fit kernel DMD for both datasets.
        kernel_A = fit_kernel_dmd(
            trajs_A_3d,
            n_delays,
            global_rank,
            delay_interval=delay_interval,
        )
        kernel_B = fit_kernel_dmd(
            trajs_B_3d,
            n_delays,
            global_rank,
            delay_interval=delay_interval,
        )

        eigvals_A = np.linalg.eigvals(kernel_A)
        eigvals_B = np.linalg.eigvals(kernel_B)

        score = compute_kw_wasserstein_distance(eigvals_A, eigvals_B)
        return float(score), int(global_rank)
