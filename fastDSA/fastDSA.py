import numpy as np
import torch
from fastDSA.dmd import DMD, embed_signal_torch
from fastDSA.kerneldmd import KernelDMD
from fastDSA.simdist import SimilarityTransformDist
from typing import Literal
from omegaconf.listconfig import ListConfig
from scipy.linalg import svdvals

def svht(X, sv=None):
    """
    Singular Value Hard Thresholding (SVHT) function.
    """
    m, n = X.shape
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median_sv = np.median(sv) if sv is not None else np.median(svdvals(X))
    return omega * median_sv

class fastDSA:
    """
    Computes the Dynamical Similarity Analysis (DSA) for two data matrices.
    
    This version automatically determines the optimal rank for the DMD 
    representations using SVHT. It loops over all input matrices, computes each
    matrix’s embedded Hankel matrix, uses svdvals and SVHT to obtain its optimal rank,
    selects the maximum rank, and then uses that rank when constructing the final DMD objects.
    """
    def __init__(self,
                 X,
                 Y=None,
                 n_delays=1,
                 delay_interval=1,
                 rank=None,
                 rank_thresh=None,
                 rank_explained_variance=None,
                 lamb=0.0,
                 send_to_cpu=True,
                 iters=1500,
                 score_method: Literal["angular", "euclidean", "wasserstein"] = "angular",
                 lr=5e-3,
                 group: Literal["GL(n)", "O(n)", "SO(n)"] = "O(n)",
                 zero_pad=False,
                 device='cpu',
                 verbose=False,
                 reduced_rank_reg=False,
                 kernel=None,
                 num_centers=0.1,
                 svd_solver='arnoldi',
                 wasserstein_compare: Literal['sv', 'eig', None] = None
                 ):
        # Save input data; if Y is not provided, compare X to itself.
        self.X = X
        self.Y = Y if Y is not None else X
        self.check_method()
        if self.method == 'self-pairwise':
            self.data = [self.X]
        else:
            self.data = [self.X, self.Y]
        
        # Broadcast parameters to align with data dimensions.
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

        # Create DMD objects for each data matrix based on whether a kernel is provided.
        if kernel is None:
            self.dmds = [[DMD(Xi,
                               self.n_delays[i][j],
                               delay_interval=self.delay_interval[i][j],
                               rank=self.rank[i][j],
                               rank_thresh=self.rank_thresh[i][j], 
                               rank_explained_variance=self.rank_explained_variance[i][j],
                               reduced_rank_reg=self.reduced_rank_reg,
                               lamb=self.lamb[i][j],
                               device=self.device,
                               verbose=self.verbose,
                               send_to_cpu=self.send_to_cpu) for j, Xi in enumerate(dat)] for i, dat in enumerate(self.data)]
        else:
            self.dmds = [[KernelDMD(Xi,
                                     self.n_delays[i][j],
                                     kernel=self.kernel,
                                     num_centers=self.num_centers,
                                     delay_interval=self.delay_interval[i][j],
                                     rank=self.rank[i][j],
                                     reduced_rank_reg=self.reduced_rank_reg,
                                     lamb=self.lamb[i][j],
                                     verbose=self.verbose,
                                     svd_solver=self.svd_solver) for j, Xi in enumerate(dat)] for i, dat in enumerate(self.data)]

        # Initialize the similarity transform module.
        self.simdist = SimilarityTransformDist(iters, score_method, lr, device, verbose, group, lambda_reg=0.01)

    def check_method(self):
        """
        Determines the method based on the types of X and Y.
        """
        tensor_or_np = lambda x: isinstance(x, (np.ndarray, torch.Tensor))
        if isinstance(self.X, list):
            if self.Y is None:
                self.method = 'self-pairwise'
            elif isinstance(self.Y, list):
                self.method = 'bipartite-pairwise'
            elif tensor_or_np(self.Y):
                self.method = 'list-to-one'
                self.Y = [self.Y]
            else:
                raise ValueError('unknown type of Y')
        elif tensor_or_np(self.X):
            self.X = [self.X]
            if self.Y is None:
                raise ValueError('only one element provided')
            elif isinstance(self.Y, list):
                self.method = 'one-to-list'
            elif tensor_or_np(self.Y):
                self.method = 'default'
                self.Y = [self.Y]
            else:
                raise ValueError('unknown type of Y')
        else:
            raise ValueError('unknown type of X')

    def broadcast_params(self, param, cast=None):
        """
        Aligns the dimensionality of the parameter with the data.
        """
        out = []
        if isinstance(param, (int, float, np.integer)) or param is None:
            out.append([param] * len(self.X))
            if self.Y is not None:
                out.append([param] * len(self.Y))
        elif isinstance(param, (tuple, list, np.ndarray, ListConfig)):
            if self.method == 'self-pairwise' and len(param) >= len(self.X):
                out = [param]
            else:
                assert len(param) <= 2  # only 2 elements max
                for i, data in enumerate([self.X, self.Y]):
                    if data is None:
                        continue
                    if isinstance(param[i], (int, float)):
                        out.append([param[i]] * len(data))
                    elif isinstance(param[i], (list, np.ndarray, tuple)):
                        assert len(param[i]) >= len(data)
                        out.append(param[i][:len(data)])
        else:
            raise ValueError("unknown type entered for parameter")
        if cast is not None and param is not None:
            out = [[cast(x) for x in dat] for dat in out]
        return out

    def fit_dmds(self,
                 X=None,
                 Y=None,
                 n_delays=None,
                 delay_interval=None,
                 rank=None,
                 rank_thresh=None,
                 rank_explained_variance=None,
                 reduced_rank_reg=None,
                 lamb=None,
                 device='cpu',
                 verbose=False,
                 send_to_cpu=True):
        """
        Recomputes only the DMDs with a single set of hyperparameters.
        
        Before instantiating the DMD objects, this method loops over all data matrices,
        computes each matrix’s embedded Hankel matrix, uses svdvals and SVHT to compute the optimal rank,
        and then sets the overall rank to the maximum found.
        """
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        n_delays = self.n_delays if n_delays is None else n_delays
        delay_interval = self.delay_interval if delay_interval is None else delay_interval
        rank = self.rank if rank is None else rank
        lamb = self.lamb if lamb is None else lamb
        data = []
        if isinstance(X, list):
            data.append(X)
        else:
            data.append([X])
        if Y is not None:
            if isinstance(Y, list):
                data.append(Y)
            else:
                data.append([Y])
        
        max_rank = 0
        for dat in data:
            for Xi in dat:
                # Compute the Hankel matrix for the data matrix Xi.
                # Ensure Xi is a torch tensor.
                Xi_tensor = torch.tensor(Xi, dtype=torch.float32, device=self.device) if not isinstance(Xi, torch.Tensor) else Xi
                # Compute embedded matrix using the provided embed_signal_torch function.
                _D = embed_signal_torch(Xi_tensor, n_delays, delay_interval)
                # Convert _D to numpy (if necessary) to use scipy's svdvals.
                _D_np = _D.cpu().detach().numpy()
                _sv = svdvals(_D_np)
                tau = svht(_D_np, sv=_sv)
                r = np.sum(_sv > tau)
                if r > max_rank:
                    max_rank = r
        if verbose:
            print(f"Maximum rank determined from all DMD embeddings: {max_rank}")
        # Use the maximum rank for all subsequent DMD constructions.
        rank = max_rank

        dmds = [[DMD(Xi, n_delays, delay_interval,
                     rank, rank_thresh, rank_explained_variance,
                     reduced_rank_reg, lamb, device, verbose, send_to_cpu)
                 for Xi in dat] for dat in data]
            
        for dmd_sets in dmds:
            for dmd in dmd_sets:
                dmd.fit()

        return dmds

    def fit_score(self):
        """
        Standard fitting function for both DMDs and the similarity transform.
        """
        for dmd_sets in self.dmds:
            for dmd in dmd_sets:
                dmd.fit()
        return self.score()

    def score(self, iters=None, lr=None, score_method=None):
        """
        Recomputes the similarity score using precomputed DMDs.
        """
        iters = self.iters if iters is None else iters
        lr = self.lr if lr is None else lr
        score_method = self.score_method if score_method is None else score_method

        ind2 = 1 - int(self.method == 'self-pairwise')
        self.sims = np.zeros((len(self.dmds[0]), len(self.dmds[ind2])))
        for i, dmd1 in enumerate(self.dmds[0]):
            for j, dmd2 in enumerate(self.dmds[ind2]):
                if self.method == 'self-pairwise' and j >= i:
                    continue
                if self.verbose:
                    print(f'Computing similarity between DMDs {i} and {j}')
                self.sims[i, j] = self.simdist.fit_score(dmd1.A_v, dmd2.A_v, iters, lr, score_method, zero_pad=self.zero_pad)
                if self.method == 'self-pairwise':
                    self.sims[j, i] = self.sims[i, j]
        if self.method == 'default':
            return self.sims[0, 0]
        return self.sims
