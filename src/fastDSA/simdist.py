import torch
import numpy as np


class SimilarityTransformDist:
    """
    Computes the Similarity Transform for Matrix Comparison under Orthogonal Groups (O(n) or SO(n))
    """
    def __init__(self, iters=200, score_method='angular', device='cpu', verbose=False, group='O(n)', lambda_reg=0.01):

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
        """
        self.iters = iters
        self.score_method = score_method
        self.device = device
        self.verbose = verbose
        self.group = group
        self.lambda_reg = lambda_reg

    def _to_tensor(self, array):
        """
        Converts a numpy array to a torch tensor if necessary.
        """
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32, device=self.device)
        return array

    def fit(self, A, B):
        """
        Computes the optimal orthogonal matrix C under the specified group (O(n) or SO(n))
        using a hybrid approach that includes regularization for near-orthogonality.

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

        assert A.shape == B.shape, "Matrices A and B must have the same shape."
        assert A.shape[0] == A.shape[1], "Matrices A and B must be square."

        # Initialize C as an identity matrix
        C = torch.eye(A.shape[0], device=self.device, requires_grad=True)

        # Define optimizer
        optimizer = torch.optim.Adam([C], lr=1e-2)

        for _ in range(self.iters):
            optimizer.zero_grad()

            # Enforce near-orthogonality using a regularization term
            loss_transform = torch.norm(A - C @ B @ C.T, p="fro")**2
            loss_orthogonality = self.lambda_reg * torch.norm(C.T @ C - torch.eye(C.shape[0], device=self.device), p="fro")**2
            loss = loss_transform + loss_orthogonality

            loss.backward()
            optimizer.step()

        # Ensure final C is orthogonal if required
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(C)
            C_star = U @ Vh

            if self.group == "SO(n)" and torch.det(C_star) < 0:
                U[:, -1] *= -1
                C_star = U @ Vh

        self.C_star = C_star.to(self.device)

        if self.verbose:
            print(f"Optimal C computed for group {self.group} with regularization.")

        return self.C_star

    def score(self, A, B):
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

        assert self.C_star is not None, "fit() must be called before score()."

        # Transform B
        B_transformed = self.C_star @ B @ self.C_star.T

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