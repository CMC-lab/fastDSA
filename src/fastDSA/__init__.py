# src/fastdsa/__init__.py

from .simdist import FastDSASimilarity, SimDistConfig
from .dmd import DMD
from .kwdsa import KernelDMD, compute_wasserstein_distance
# from . import stats  # module-level, not star-import

__all__ = [
    "FastDSASimilarity",
    "SimDistConfig",
    "DMD",
    "KernelDMD",
    "compute_wasserstein_distance",
    "stats",
]
