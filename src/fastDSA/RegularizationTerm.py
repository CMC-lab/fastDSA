import numpy as np
import ot
import torch
import os
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import TruncatedSVD

class SimilarityTransformDist:
    """
    Computes the Similarity Transform for Matrix Comparison under Orthogonal Groups (O(n) or SO(n)),
    with hybrid optimization and optional low-rank approximation.
    """
    def __init__(self,
                 iters=200,
                 score_method: str = "angular",
                 lr=0.01,
                 device: str = 'cpu',
                 verbose=False,
                 group: str = "O(n)",
                 lambda_reg: float = 0.01,
                 rank: int = None,
                 save_path: str = None,
                 run_name: str = None):
        """
        Parameters
        ----------
        iters : int
            Number of iterations for optimization.
        score_method : {'angular', 'frobenius'}
            Metric to evaluate similarity.
        device : {'cpu', 'cuda'}
            Device to perform computation.
        verbose : bool
            Whether to print progress and results.
        group : {'O(n)', 'SO(n)'}
            Specifies the group of matrices to optimize over.
        lambda_reg : float
            Regularization parameter to enforce near-orthogonality.
        rank : int or None
            Rank for low-rank approximation. If None, no approximation is applied.
        save_path : str, optional
            Base directory to save results. If None, results won't be saved.
        """
        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self.group = group
        self.lambda_reg = lambda_reg
        self.rank = rank
        self.save_path = Path(save_path) if save_path else None
        self.run_name = run_name
        self.losses = []
        self.scores = []

    def _to_tensor(self, array):
        """
        Converts a numpy array to a torch tensor if necessary.
        """
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32, device=self.device)
        return array

    def _low_rank_approximation(self, A, B):
        """
        Applies low-rank approximation to matrices A and B if rank is specified.
        """
        if self.rank is not None:
            svd = TruncatedSVD(n_components=self.rank)
            A_reduced = torch.tensor(svd.fit_transform(A.cpu().numpy()), dtype=torch.float32, device=self.device)
            B_reduced = torch.tensor(svd.fit_transform(B.cpu().numpy()), dtype=torch.float32, device=self.device)
            return A_reduced, B_reduced
        return A, B

    def save_results(self, timestamp=None):
        """Save optimization results to files with timestamps and optional run name prefix."""
        if self.save_path is None or self.C_star is None:
            return
            
        # Create timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory name with optional run_name prefix
        dir_name = f"{self.run_name}_{timestamp}" if self.run_name else timestamp
            
        # Create save directory with run_name and timestamp
        save_dir = self.save_path / dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save C matrix
        torch.save(self.C_star, save_dir / "C_star.pt")
        
        # Save losses and scores as numpy arrays
        if self.losses:
            np.save(save_dir / "losses.npy", np.array(self.losses))
        if self.scores:
            np.save(save_dir / "scores.npy", np.array(self.scores))
            
        return str(save_dir)

    def fit(self, A, B):
        """
        Computes the optimal orthogonal matrix C under the specified group (O(n) or SO(n))
        using a hybrid approach that includes regularization for near-orthogonality and optional low-rank approximation.

        Parameters
        ----------
        A : torch.Tensor or np.ndarray
            First matrix (n x n).
        B : torch.Tensor or np.ndarray
            Second matrix (n x n).

        Returns
        -------
        C_star : torch.Tensor
            The optimal orthogonal transformation matrix.
        """
        A = self._to_tensor(A)
        B = self._to_tensor(B)

        # Apply low-rank approximation if rank is specified
        A, B = self._low_rank_approximation(A, B)

        assert A.shape == B.shape, "Matrices A and B must have the same shape."
        assert A.shape[0] == A.shape[1], "Matrices A and B must be square."

        # Initialize C as an identity matrix
        C = torch.eye(A.shape[0], device=self.device, requires_grad=True)
        
        # Define optimizer
        optimizer = torch.optim.Adam([C], lr=self.lr)
        
        self.losses = []  # Reset losses
        self.scores = []  # Reset scores
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        import time
        self.losses = []
        self.scores = []
        self.iteration_times = []   # <-- new list for per-iteration durations
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        for _ in range(self.iters):
            optimizer.zero_grad()
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            t0 = time.time()          # start timing
            optimizer.zero_grad()
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            # Calculate loss
            loss_transform = torch.norm(A - C @ B @ C.T, p="fro")**2
            loss_orthogonality = self.lambda_reg * torch.norm(C.T @ C - torch.eye(C.shape[0], device=self.device), p="fro")**2
            loss = loss_transform + loss_orthogonality
            
            # Store loss and current score
            self.losses.append(loss.item())
            
            # Calculate and store current score
            with torch.no_grad():
                current_score = self.score(A, B, current_C=C)
                self.scores.append(current_score)
            
            loss.backward()
            optimizer.step()
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            t1 = time.time()          # end timing
            self.iteration_times.append(t1 - t0)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
        # Ensure final C is orthogonal if required
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(C)
            C_star = U @ Vh

            if self.group == "SO(n)" and torch.det(C_star) < 0:
                U[:, -1] *= -1
                C_star = U @ Vh

        self.C_star = C_star.detach()

        if self.verbose:
            print(f"Optimal C computed for group {self.group} with regularization and low-rank approximation.")

        # Save results if path is provided
        if self.save_path:
            self.save_results()
            
        return self.C_star

    def score(self, A, B, current_C=None):
        """
        Computes the similarity score between matrices A and B using the optimal transformation matrix C.

        Parameters
        ----------
        A : torch.Tensor or np.ndarray
            First matrix (n x n).
        B : torch.Tensor or np.ndarray
            Second matrix (n x n).

        Returns
        -------
        score : float
            The similarity score (angular distance or Frobenius norm).
        """
        A = self._to_tensor(A)
        B = self._to_tensor(B)
        
        # Apply low-rank approximation if rank is specified
        A, B = self._low_rank_approximation(A, B)
        
        if current_C is not None:
            C = current_C
        else:
            assert self.C_star is not None, "fit() must be called before score()"
            C = self.C_star
        C = C.to(self.device)

        # Transform B
        B_transformed = C @ B @ C.T

        if self.score_method == "angular":
            # Angular distance
            num = torch.trace(A.T @ B_transformed)
            den = torch.norm(A, p="fro") * torch.norm(B_transformed, p="fro")
            cos_theta = num / den
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure numerical stability
            score = torch.arccos(cos_theta).item()
        elif self.score_method == "frobenius":
            # Frobenius norm
            score = torch.norm(A - B_transformed, p="fro").item()
        else:
            raise ValueError("Invalid score_method. Choose 'angular' or 'frobenius'.")

        return score

    def fit_score(self, A, B):
        """
        Convenience method to compute both the optimal C and the similarity score.

        Parameters
        ----------
        A : torch.Tensor or np.ndarray
            First matrix (n x n).
        B : torch.Tensor or np.ndarray
            Second matrix (n x n).

        Returns
        -------
        score : float
            The similarity score (angular distance or Frobenius norm).
        """
        self.fit(A, B)
        return self.score(A, B)
    

def compute_wasserstein_distance(a, b):
    """
    Computes the Wasserstein distance between two distributions.

    Parameters
    ----------
    a : np.ndarray
        First distribution (e.g., singular values or eigenvalues).
    b : np.ndarray
        Second distribution (e.g., singular values or eigenvalues).

    Returns
    -------
    float
        Wasserstein distance between the two distributions.
    """
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()

    # Reshape to ensure compatibility with distance computation
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)

    # Pairwise distance matrix
    M = ot.dist(a, b)

    # Define uniform weights for the distributions
    a_weights = np.ones(a.shape[0]) / a.shape[0]
    b_weights = np.ones(b.shape[0]) / b.shape[0]

    # Ensure dimensions of weights and cost matrix match
    assert a_weights.shape[0] == M.shape[0], "Mismatch between weights and cost matrix rows"
    assert b_weights.shape[0] == M.shape[1], "Mismatch between weights and cost matrix columns"

    # Compute Wasserstein distance
    wasserstein_distance = ot.emd2(a_weights, b_weights, M)
    return wasserstein_distance
