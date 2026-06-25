# src/fastDSA/kwdsa.py
from __future__ import annotations

import numpy as np
import torch

import ot

try:
    # kooplearn >= 2
    from kooplearn.kernel import NystroemKernelRidge as _NystroemModel

    _KOOPLEARN_V2 = True
    TensorContextDataset = None
    traj_to_contexts = None
except ImportError:
    # kooplearn 1.1.x
    from kooplearn.data import TensorContextDataset, traj_to_contexts
    from kooplearn.models import NystroemKernel as _NystroemModel

    _KOOPLEARN_V2 = False

from sklearn.gaussian_process.kernels import RBF


def _as_real_point_cloud(x, use_complex_plane: bool = False) -> np.ndarray:
    """Convert real values or complex eigenvalues to a finite real point cloud."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x).reshape(-1)

    if use_complex_plane:
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
    use_complex_plane = np.iscomplexobj(a) or np.iscomplexobj(b)
    a = _as_real_point_cloud(a, use_complex_plane=use_complex_plane)
    b = _as_real_point_cloud(b, use_complex_plane=use_complex_plane)

    M = ot.dist(a, b, metric="euclidean")
    aw = np.ones(a.shape[0], dtype=float) / a.shape[0]
    bw = np.ones(b.shape[0], dtype=float) / b.shape[0]
    return float(ot.emd2(aw, bw, M))


class KernelDMD(_NystroemModel):
    """Kernel DMD wrapper supporting kooplearn 1.1 and 2.x."""

    def __init__(
        self,
        data,
        n_delays: int,
        kernel=None,
        delay_interval: int = 1,
        rank: int = 10,
        verbose: bool = False,
        gamma: float = 1.0,
        alpha: float = 1e-7,
        n_centers: int | float = 600,
        random_state: int = 0,
        reduced_rank: bool = True,
        eigen_solver: str = "arpack",
        **kwargs,
    ):
        # Accept kooplearn-native aliases while retaining the public kwDSA
        # argument names used by the rest of this package.
        num_centers = kwargs.pop("num_centers", n_centers)
        tikhonov_reg = kwargs.pop("tikhonov_reg", alpha)
        rng_seed = kwargs.pop("rng_seed", random_state)
        reduced_rank = kwargs.pop("reduced_rank_reg", reduced_rank)
        svd_solver = kwargs.pop("svd_solver", eigen_solver)
        rank = int(kwargs.pop("n_components", rank))

        if _KOOPLEARN_V2:
            if kernel is None:
                kernel = "rbf"
            if svd_solver == "full":
                svd_solver = "dense"
            elif svd_solver == "arnoldi":
                svd_solver = "arpack"

            super().__init__(
                n_components=rank,
                lag_time=int(delay_interval),
                reduced_rank=reduced_rank,
                kernel=kernel,
                gamma=gamma,
                alpha=tikhonov_reg,
                eigen_solver=svd_solver,
                n_centers=num_centers,
                random_state=rng_seed,
                **kwargs,
            )
        else:
            if kernel is None or (isinstance(kernel, str) and kernel.lower() == "rbf"):
                if gamma <= 0:
                    raise ValueError("gamma must be positive for the RBF kernel.")
                kernel = RBF(length_scale=np.sqrt(1.0 / (2.0 * gamma)))
            elif isinstance(kernel, str):
                raise ValueError(
                    f"Unsupported kernel string {kernel!r}; pass 'rbf' or a "
                    "scikit-learn kernel object."
                )

            if svd_solver == "arpack":
                svd_solver = "arnoldi"
            elif svd_solver in ("auto", "dense"):
                svd_solver = "full"

            super().__init__(
                kernel=kernel,
                reduced_rank=reduced_rank,
                rank=rank,
                tikhonov_reg=tikhonov_reg,
                svd_solver=svd_solver,
                num_centers=num_centers,
                rng_seed=rng_seed,
                **kwargs,
            )

        self.raw_data = data
        self.n_delays = int(n_delays)
        self.context_window_len = self.n_delays + 1
        self.delay_interval = int(delay_interval)
        self.verbose = verbose
        self._pair_x_indices = None
        self._pair_y_indices = None
        self._training_row_count = None

    def fit(self, data=None, **kwargs):
        data = self.raw_data if data is None else data
        if _KOOPLEARN_V2:
            training_data = self._build_delay_embedded_training_data(data)
            super().fit(training_data)
            self.A_v = (
                self.V_.T @ self.kernel_YX_ @ self.U_ / len(self.kernel_YX_)
            )
        else:
            contexts = self._build_context_dataset(data)
            super().fit(contexts)
            self.A_v = self.V.T @ self.kernel_YX @ self.U / len(self.kernel_YX)
        return self

    def _build_context_dataset(self, data) -> TensorContextDataset:
        """
        Build delay contexts independently for each trajectory.

        Concatenating fixed-size contexts supports variable-length trajectories
        without padding, truncation, or transitions across trial boundaries.
        """
        trajectories = _as_trajectory_list(data)
        context_blocks = []

        for trajectory in trajectories:
            contexts = traj_to_contexts(
                trajectory,
                context_window_len=self.context_window_len,
                time_lag=self.delay_interval,
            )
            context_blocks.append(contexts.data)

        context_data = np.concatenate(context_blocks, axis=0)
        self.data = TensorContextDataset(context_data)
        return self.data

    def _build_delay_embedded_training_data(self, data) -> np.ndarray:
        """
        Build flattened delay vectors for kooplearn 2.x.

        Pair indices are tracked per trajectory so the estimator never creates
        artificial transitions between two trials after concatenation.
        """
        trajectories = _as_trajectory_list(data)
        embedded_blocks = []
        pair_x_indices = []
        pair_y_indices = []
        offset = 0

        for trajectory in trajectories:
            T, n_channels = trajectory.shape
            n_rows = T - (self.n_delays - 1) * self.delay_interval
            if n_rows <= self.delay_interval:
                raise ValueError(
                    f"Not enough time points (T={T}) for "
                    f"n_delays={self.n_delays}, "
                    f"delay_interval={self.delay_interval}."
                )

            embedded = np.empty(
                (n_rows, n_channels * self.n_delays),
                dtype=trajectory.dtype,
            )
            for delay_index in range(self.n_delays):
                start = delay_index * self.delay_interval
                embedded[
                    :,
                    delay_index * n_channels : (delay_index + 1) * n_channels,
                ] = trajectory[start : start + n_rows]

            n_pairs = n_rows - self.delay_interval
            x_indices = offset + np.arange(n_pairs)
            pair_x_indices.append(x_indices)
            pair_y_indices.append(x_indices + self.delay_interval)
            embedded_blocks.append(embedded)
            offset += n_rows

        training_data = np.concatenate(embedded_blocks, axis=0)
        self._pair_x_indices = np.concatenate(pair_x_indices)
        self._pair_y_indices = np.concatenate(pair_y_indices)
        self._training_row_count = training_data.shape[0]
        self.data = training_data
        return training_data

    def _split_trajectory(self, X):
        """Respect trajectory boundaries when using kooplearn 2.x."""
        if (
            _KOOPLEARN_V2
            and self._pair_x_indices is not None
            and X.shape[0] == self._training_row_count
        ):
            return X[self._pair_x_indices], X[self._pair_y_indices]
        return super()._split_trajectory(X)


def _as_trajectory_list(data):
    """Normalize data to list[(timepoints, channels)]."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            trajectories = [data]
        elif data.ndim == 3:
            trajectories = [data[i] for i in range(data.shape[0])]
        else:
            raise ValueError(
                "KernelDMD data must be a 2D trajectory, a 3D batch, "
                "or a sequence of 2D trajectories."
            )
    elif isinstance(data, (list, tuple)):
        trajectories = []
        for trajectory in data:
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.detach().cpu().numpy()
            trajectory = np.asarray(trajectory)
            if trajectory.ndim != 2:
                raise ValueError(
                    "Each KernelDMD trajectory must be 2D "
                    "(timepoints, channels)."
                )
            trajectories.append(trajectory)
    else:
        trajectory = np.asarray(data)
        if trajectory.ndim != 2:
            raise ValueError(
                "KernelDMD data must be a 2D trajectory, a 3D batch, "
                "or a sequence of 2D trajectories."
            )
        trajectories = [trajectory]

    if not trajectories:
        raise ValueError("KernelDMD requires at least one trajectory.")

    n_channels = trajectories[0].shape[1]
    for index, trajectory in enumerate(trajectories):
        if trajectory.shape[1] != n_channels:
            raise ValueError(
                "All KernelDMD trajectories must have the same number of "
                f"channels; trajectory 0 has {n_channels}, while trajectory "
                f"{index} has {trajectory.shape[1]}."
            )

    return trajectories


def _num_available_contexts(x, n_delays: int, delay_interval: int) -> int:
    """Count context windows produced by kooplearn for all trajectories."""
    trajectories = _as_trajectory_list(x)
    n_delays = int(n_delays)
    delay_interval = int(delay_interval)

    counts = [
        int(trajectory.shape[0]) - n_delays * delay_interval
        for trajectory in trajectories
    ]
    if min(counts) < 1:
        shortest = min(int(trajectory.shape[0]) for trajectory in trajectories)
        raise ValueError(
            f"Not enough time points (shortest T={shortest}) for "
            f"n_delays={n_delays}, delay_interval={delay_interval}."
        )
    return int(sum(counts))


def fit_kernel_dmd(x, n_delays: int, rank: int, delay_interval: int = 1, **kwargs):
    # Keep kooplearn's randomized Nyström model in a feasible regime for short
    # trajectories. This prevents n_centers/n_components from exceeding the
    # number of available contexts.
    n_contexts = _num_available_contexts(x, n_delays=n_delays, delay_interval=delay_interval)
    rank = int(max(1, min(int(rank), int(n_contexts))))
    n_centers = int(max(rank, min(600, n_contexts)))
    kwargs.setdefault("n_centers", n_centers)

    # kooplearn's Arnoldi implementation needs room above the requested rank.
    arnoldi_buffer = max(10, 4 * int(np.sqrt(rank)))
    if n_centers <= rank + arnoldi_buffer + 1:
        kwargs.setdefault("eigen_solver", "full")

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
