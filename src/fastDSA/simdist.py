# src/fastDSA/simdist.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .dmd import DMD
from .kwdsa import fit_kernel_dmd, compute_wasserstein_distance as compute_kw_wasserstein_distance

# Each of these files defines a class named SimilarityTransformDist.
# We import and alias them to avoid name collisions.
from .RegularizationTerm import SimilarityTransformDist as RegularizationSimilarityTransformDist  # :contentReference[oaicite:1]{index=1}
from .RiemannianManifold import SimilarityTransformDist as RiemannianSimilarityTransformDist      # :contentReference[oaicite:2]{index=2}
from .LandingAlgorithm import SimilarityTransformDist as LandingSimilarityTransformDist          # :contentReference[oaicite:3]{index=3}


ArrayLike = Union[np.ndarray, torch.Tensor]
MethodType = Literal["ro", "rim", "land", "kw"]


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
    delay_interval : int

    Returns
    -------
    X, Y : np.ndarray
        X, Y both of shape (n * n_delays, L - 1),
        with L = T - (n_delays - 1) * delay_interval.
    """
    traj = np.asarray(traj)
    if traj.ndim != 2:
        raise ValueError(f"traj must be 2D (T, n); got shape {traj.shape}")

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
    """Gavish–Donoho cubic approximation for SVHT."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def svht_threshold(M: np.ndarray, s: Optional[np.ndarray] = None) -> float:
    """
    Gavish–Donoho SVHT threshold tau = omega(beta) * median(s).

    Parameters
    ----------
    M : np.ndarray
        2D matrix.
    s : np.ndarray, optional
        Precomputed singular values.

    Returns
    -------
    float
        Threshold tau.
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
    return max(r, min_rank)


def svht_ranks_for_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    which: Literal["X", "Y", "concat"] = "X",
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
        ranks.append(svht_rank_single(M))
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
    Cap the desired rank so it's compatible with the trajectory and Hankel dimension.
    """
    traj = np.asarray(traj)
    if traj.ndim != 2:
        raise ValueError(f"traj must be 2D (T, n) inside _safe_rank_for_traj, got {traj.shape}")

    T, n = traj.shape
    L = T - (n_delays - 1) * delay_interval - steps_ahead
    if L <= 0:
        raise ValueError(
            f"Not enough time points (T={T}) for n_delays={n_delays}, delay_interval={delay_interval}, "
            f"steps_ahead={steps_ahead}."
        )
    hankel_dim = n * n_delays
    max_rank = min(hankel_dim, L)
    return int(max(1, min(desired_rank, max_rank)))


def compute_dmd_matrix_for_traj(
    traj: np.ndarray,
    n_delays: int,
    delay_interval: int,
    desired_rank: int,
    device: str = "cuda",
    steps_ahead: int = 1,
):
    """
    Compute the HAVOK-DMD operator matrix for a single trajectory.

    Parameters
    ----------
    traj : np.ndarray
        2D array of shape (T, n) on CPU.
    n_delays : int
    delay_interval : int
    desired_rank : int
        Global desired rank (will be capped per-trajectory by _safe_rank_for_traj).
    device : str
        'cuda' or 'cpu'.
    steps_ahead : int

    Returns
    -------
    A : torch.Tensor
        DMD operator (H_dims x H_dims) on `device`.
    rank : int
        Rank actually used for this trajectory.
    """
    if isinstance(traj, np.ndarray) and traj.dtype != np.float32:
        traj = traj.astype(np.float32)

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    rank = _safe_rank_for_traj(traj, n_delays, delay_interval, desired_rank, steps_ahead)

    dmd = DMD(
        traj,
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=None,                 # set via fit() to be explicit
        reduced_rank_reg=False,
        lamb=0.0,
        device=device,
        verbose=False,
        send_to_cpu=False,
        steps_ahead=steps_ahead,
    )

    dmd.fit(
        data=traj,
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=rank,
        device=device,
    )

    A = dmd.A_havok_dmd  # torch.Tensor on `device`
    return A, rank


def _aggregate_operators(ops: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Aggregate multiple DMD operators into a single representative operator.
    Currently: element-wise mean over trajectories.
    """
    if len(ops) == 0:
        raise ValueError("No operators to aggregate.")
    if len(ops) == 1:
        return ops[0]
    stacked = torch.stack(ops, dim=0)
    return stacked.mean(dim=0)


# ---------------------------------------------------------------------------
# Config and main class
# ---------------------------------------------------------------------------

@dataclass
class SimDistConfig:
    """
    Configuration for the similarity pipeline.

    method:
        'ro'   -> RegularizationTerm-based SimilarityTransformDist (RegularizationTerm.py)
        'rim'  -> Riemannian manifold optimizer (RiemannianManifold.py)
        'land' -> Landing-style optimizer (LandingAlgorithm.py)
        'kw'   -> Kernel DMD + Wasserstein distance between spectra
    """
    n_delays: int
    delay_interval: int = 1
    rank: Optional[int] = None
    method: MethodType = "ro"

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
      2. Build Hankel (X, Y) pairs for each trajectory via delay_embed_traj.
      3. If rank is None, compute global_max_rank using Gavish–Donoho SVHT:
         global_max_rank = max(max(ranks_A), max(ranks_B)).
      4. For each trajectory, compute a DMD operator A via compute_dmd_matrix_for_traj.
      5. Depending on `method`:
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
            Global rank used across all trajectories (or for kw, the rank passed to KernelDMD).
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

    def _build_pairs(self, trajs: Sequence[np.ndarray]) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """Compute (X, Y) Hankel pairs for each trajectory."""
        pairs = []
        for traj in trajs:
            X, Y = delay_embed_traj(traj, self.config.n_delays, self.config.delay_interval)
            pairs.append((X, Y))
        return pairs

    def _auto_detect_rank(
        self,
        pairs_A: Sequence[Tuple[np.ndarray, np.ndarray]],
        pairs_B: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> int:
        """Compute global_max_rank from SVHT on (X, Y) pairs of both datasets."""
        ranks_A = svht_ranks_for_pairs(pairs_A, which=self.config.svht_which)
        ranks_B = svht_ranks_for_pairs(pairs_B, which=self.config.svht_which)
        global_max_rank = max(max(ranks_A), max(ranks_B))

        self.ranks_A_ = ranks_A
        self.ranks_B_ = ranks_B
        return int(global_max_rank)

    # -------------------- operator-based path (ro / rim / land) -------------------- #

    def _fit_score_operator_based(self, data_A: ArrayLike, data_B: ArrayLike) -> Tuple[float, int]:
        """
        Path for methods 'ro', 'rim', 'land':
          - build Hankel pairs,
          - detect global rank if needed,
          - compute DMD operators,
          - call the respective metric's fit_score.
        """
        device_str = _default_device_str(self.config.device)

        # 1) Convert to list of trajectories (T, n)
        trajs_A = self._prepare_trajs_list(data_A)
        trajs_B = self._prepare_trajs_list(data_B)

        # 2) Build Hankel pairs and detect global rank (if rank not supplied)
        pairs_A = self._build_pairs(trajs_A)
        pairs_B = self._build_pairs(trajs_B)

        if self.config.rank is None:
            global_rank = self._auto_detect_rank(pairs_A, pairs_B)
            if self.config.verbose:
                print(f"[fastDSA] Detected global rank via SVHT: {global_rank}")
        else:
            global_rank = int(self.config.rank)

        # 3) Compute DMD operators for all trajectories
        A_ops, B_ops = [], []
        for tr in trajs_A:
            A_op, _ = compute_dmd_matrix_for_traj(
                tr,
                n_delays=self.config.n_delays,
                delay_interval=self.config.delay_interval,
                desired_rank=global_rank,
                device=device_str,
                steps_ahead=self.config.steps_ahead,
            )
            A_ops.append(A_op)

        for tr in trajs_B:
            B_op, _ = compute_dmd_matrix_for_traj(
                tr,
                n_delays=self.config.n_delays,
                delay_interval=self.config.delay_interval,
                desired_rank=global_rank,
                device=device_str,
                steps_ahead=self.config.steps_ahead,
            )
            B_ops.append(B_op)

        # Aggregate operators into a single representative operator per dataset
        A_mean = _aggregate_operators(A_ops)
        B_mean = _aggregate_operators(B_ops)

        # 4) Instantiate and run the chosen metric
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

    def _to_trajs_3d_for_kw(self, data: ArrayLike) -> np.ndarray:
        """
        Convert input into a 3D array (n_traj, T, n) for KernelDMD.compute_hankel.

        Accepts:
          - 2D (channels, timepoints)
          - 3D (n_traj, channels, timepoints)
          - list/tuple of 2D (channels, timepoints)
        """
        if isinstance(data, (list, tuple)):
            trajs = []
            for d in data:
                arr = _to_numpy(d)
                if arr.ndim != 2:
                    raise ValueError("Each trajectory in a list must be 2D (channels, timepoints).")
                C, T = arr.shape
                trajs.append(arr.T)  # (T, C)
            return np.stack(trajs, axis=0)

        arr = _to_numpy(data)
        if arr.ndim == 2:
            C, T = arr.shape
            return arr.T[None, :, :]  # (1, T, C)
        elif arr.ndim == 3:
            # (n_traj, channels, timepoints) -> (n_traj, T, C)
            return np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(
                "data must be 2D (channels, timepoints), 3D (n_traj, channels, timepoints), "
                "or a list/tuple of 2D arrays."
            )

    def _fit_score_kw(self, data_A: ArrayLike, data_B: ArrayLike) -> Tuple[float, int]:
        """
        Kernel-DMD + Wasserstein distance between eigenvalue distributions.
        """
        trajs_A_3d = self._to_trajs_3d_for_kw(data_A)
        trajs_B_3d = self._to_trajs_3d_for_kw(data_B)

        # Detect rank if needed (cheap: use first trajectory in each dataset)
        if self.config.rank is None:
            trajs_A = [trajs_A_3d[0]]
            trajs_B = [trajs_B_3d[0]]
            pairs_A = self._build_pairs(trajs_A)
            pairs_B = self._build_pairs(trajs_B)
            global_rank = self._auto_detect_rank(pairs_A, pairs_B)
            if self.config.verbose:
                print(f"[fastDSA-kw] Detected rank via SVHT: {global_rank}")
        else:
            global_rank = int(self.config.rank)

        # Fit kernel DMD for both datasets
        kernel_A = fit_kernel_dmd(
            trajs_A_3d,
            self.config.n_delays,
            global_rank,
            delay_interval=self.config.delay_interval,
        )
        kernel_B = fit_kernel_dmd(
            trajs_B_3d,
            self.config.n_delays,
            global_rank,
            delay_interval=self.config.delay_interval,
        )

        eigvals_A = np.linalg.eigvals(kernel_A)
        eigvals_B = np.linalg.eigvals(kernel_B)

        score = compute_kw_wasserstein_distance(eigvals_A, eigvals_B)
        return float(score), int(global_rank)
