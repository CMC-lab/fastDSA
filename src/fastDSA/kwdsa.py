# src/fastdsa/kwdsa.py

from __future__ import annotations

import numpy as np
import torch
import ot

# These imports assume you're using kooplearn; adjust paths to match your environment.
from kooplearn.models import NystroemKernel
from kooplearn.kernels import RBF

# Make sure traj_to_contexts is importable; if it's in your codebase, import from there.
from kooplearn.data import traj_to_contexts  # adapt if needed


def compute_wasserstein_distance(a, b) -> float:
    """
    Computes the Wasserstein distance between two distributions.

    Parameters
    ----------
    a : np.ndarray or torch.Tensor
        First distribution (e.g., singular values or eigenvalues).
    b : np.ndarray or torch.Tensor
        Second distribution (e.g., singular values or eigenvalues).

    Returns
    -------
    float
        Wasserstein distance between the two distributions.
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()

    a = np.asarray(a).reshape(-1, 1)
    b = np.asarray(b).reshape(-1, 1)

    # Pairwise distance matrix
    M = ot.dist(a, b)

    # Uniform weights
    a_weights = np.ones(a.shape[0]) / a.shape[0]
    b_weights = np.ones(b.shape[0]) / b.shape[0]

    # Sanity checks
    assert a_weights.shape[0] == M.shape[0], "Mismatch between weights and cost matrix rows"
    assert b_weights.shape[0] == M.shape[1], "Mismatch between weights and cost matrix columns"

    # Wasserstein distance (quadratic cost)
    wasserstein_distance = ot.emd2(a_weights, b_weights, M)
    return float(wasserstein_distance)


class KernelDMD(NystroemKernel):
    def __init__(
        self,
        data,
        n_delays: int,
        kernel=None,
        num_centers: float | int = 0.1,
        delay_interval: int = 1,
        rank: int = 10,
        reduced_rank_reg: bool = True,
        lamb: float = 1e-10,
        verbose: bool = False,
        svd_solver: str = "arnoldi",
    ):
        """
        Subclass of kooplearn.NystroemKernel that uses a kernel to compute a DMD-like model.

        This uses Reduced Rank Regression (as opposed to Principal Component Regression).

        Parameters
        ----------
        data : array-like
            Initial data, understood as trajectories for Hankel embeddings.
        n_delays : int
            Number of delays (context_window_len = n_delays + 1).
        kernel : kernel object, optional
            Kernel to use; defaults to RBF().
        num_centers : float or int
            Number of Nyström centers or fraction.
        delay_interval : int
        rank : int
        reduced_rank_reg : bool
        lamb : float
            Regularization parameter.
        verbose : bool
        svd_solver : str
        """
        if kernel is None:
            kernel = RBF()

        super().__init__(
            kernel=kernel,
            reduced_rank_reg=reduced_rank_reg,
            rank=rank,
            lamb=lamb,
            svd_solver=svd_solver,
            num_centers=num_centers,
        )

        self.n_delays = n_delays
        self.context_window_len = n_delays + 1
        self.delay_interval = delay_interval
        self.verbose = verbose
        self.rank = rank
        self.lamb = 0.0 if lamb is None else lamb

        self.data = data

    def fit(self, data=None, lamb: float | None = None):
        """
        Fits the Kernel DMD model to the provided data.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor, optional
            Trajectories; if None, uses self.data.
        lamb : float, optional
            Regularization parameter for ridge regression.
        """
        data = self.data if data is None else data
        lamb = self.lamb if lamb is None else lamb

        self.compute_hankel(data)
        self.compute_kernel_dmd(lamb)

    def compute_hankel(self, trajs):
        """
        Computes delay embeddings for given trajectories.

        Parameters
        ----------
        trajs : np.ndarray or torch.Tensor or list
            If 2D: (T, d) single trajectory.
            If 3D: (n_traj, T, d) multiple trajectories.
        """
        if isinstance(trajs, torch.Tensor):
            trajs = trajs.detach().cpu().numpy()

        trajs = np.asarray(trajs)
        if trajs.ndim == 2:
            # single trajectory -> add batch dimension
            trajs = trajs[np.newaxis, :, :]

        # First trajectory
        data = traj_to_contexts(
            trajs[0],
            context_window_len=self.context_window_len,
            time_lag=self.delay_interval,
        )
        idx = np.zeros(data.idx_map.shape)
        data.idx_map = np.concatenate((idx, data.idx_map), axis=-1)

        # Remaining trajectories
        for i in range(1, len(trajs)):
            new_traj = traj_to_contexts(
                trajs[i],
                context_window_len=self.context_window_len,
                time_lag=self.delay_interval,
            )

            data.data = np.concatenate((data.data, new_traj.data), axis=0)

            idx = np.zeros(new_traj.idx_map.shape) + 1
            new_traj.idx_map = np.concatenate((idx, new_traj.idx_map), axis=-1)
            data.idx_map = np.concatenate((data.idx_map, new_traj.idx_map), axis=0)

        self.data = data

        if self.verbose:
            print("Hankel matrix computed")

    def compute_kernel_dmd(self, lamb: float | None = None):
        """
        Computes the kernel-based DMD operator A_v.
        """
        self.tikhonov_reg = self.lamb if lamb is None else lamb
        super().fit(self.data)
        # A_v: operator in feature space
        self.A_v = self.V.T @ self.kernel_YX @ self.U / len(self.kernel_YX)

        if self.verbose:
            print("Kernel regression complete")

    def predict(self, test_data, reseed=None):
        """
        Predict future trajectories using the Kernel DMD model.

        Parameters
        ----------
        test_data : np.ndarray or torch.Tensor
            Input test data, shape (T, d) or (n_traj, T, d).
        reseed : int or None
            Currently only reseed=1 is supported.

        Returns
        -------
        pred_data : np.ndarray
            Predictions generated by the Kernel DMD model.
        """
        if reseed is None:
            reseed = 1
        else:
            raise NotImplementedError("Reseeding values other than 1 are not implemented.")

        if isinstance(test_data, torch.Tensor):
            test_data = test_data.detach().cpu().numpy()
        if isinstance(test_data, list):
            test_data = np.array(test_data)

        isdim2 = test_data.ndim == 2
        if isdim2:
            test_data = test_data[np.newaxis, :, :]

        # Shape: (n_traj, T, d)
        pred_data = np.zeros(test_data.shape)
        pred_data[:, 0 : self.n_delays] = test_data[:, 0 : self.n_delays]

        self.compute_hankel(test_data)

        pred = super().predict(self.data)
        pred = pred.reshape(
            test_data.shape[0],
            test_data.shape[1] - self.n_delays,
            test_data.shape[2],
        )
        pred_data[:, self.n_delays :] = pred

        return pred_data


def fit_kernel_dmd(x, n_delays: int, rank: int, delay_interval: int = 1):
    """
    Convenience wrapper to fit KernelDMD and return A_v.

    Parameters
    ----------
    x : array-like
        Trajectories, shape:
          - (T, d) single trajectory
          - (n_traj, T, d) multiple trajectories
    n_delays : int
    rank : int
    delay_interval : int

    Returns
    -------
    A_v : np.ndarray
        Kernel DMD operator matrix (as a NumPy array).
    """
    dmd = KernelDMD(
        x,
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=rank,
        verbose=True,
    )
    dmd.fit()
    # A_v is typically a NumPy array already, but be defensive
    A_v = dmd.A_v
    if isinstance(A_v, torch.Tensor):
        A_v = A_v.detach().cpu().numpy()
    return A_v
