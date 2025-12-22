# fastDSA

Fast Dynamical Similarity Analysis (**fastDSA**) is the reference implementation accompanying the paper **“Fast dynamical similarity analysis”**.

https://www.arxiv.org/abs/2511.22828

![MethodsSchematic](https://github.com/CMC-lab/fastDSA/blob/main/Tests/Figures/Methods_schematic.png)

To understand how neural systems process information, it is often essential to compare one circuit with another, one brain with another, or data with a model. Traditional similarity measures ignore the dynamical processes underlying neural representations. Dynamical similarity methods offer a framework to compare the temporal structure of dynamical systems by embedding their (possibly) nonlinear dynamics into a globally linear space and computing conjugacy metrics on the resulting linear operators. However, identifying the best embedding and computing these metrics can be computationally slow.

fastDSA is designed to be computationally more efficient than prior dynamical similarity methods while maintaining accuracy and robustness. It introduces two key components that boost efficiency:

1. **Automatic model-order selection for delay embedding** using a data-driven singular-value hard threshold (SVHT) that identifies an informative subspace and discards noise, reducing computational cost without sacrificing signal.
2. **A lightweight optimization objective and procedure** that replaces expensive exact orthogonality constraints with an efficient mechanism that keeps the search close to the space of orthogonal transformations.

We demonstrate that fastDSA is at least an order of magnitude faster than previous methods while preserving invariances and sensitivities relevant to dynamical similarity analysis.



### Citation

If you use this code, please cite:

```bibtex
@article{behrad2025fast,
  title={Fast dynamical similarity analysis},
  author={Behrad, Arman and Ostrow, Mitchell and Fakharian, Mohammad Taha and Fiete, Ila and Beste, Christian and Safavi, Shervin},
  journal={arXiv preprint arXiv:2511.22828},
  year={2025}
}
```


## Installation

Clone the repository and install in editable mode:

```
git clone https://github.com/CMC-lab/fastDSA.git
cd fastDSA
pip install -e .
```




## Quick Start

Data format

fastDSA expects each trajectory/trial in the shape:

`(channels, timepoints)`

```python
from fastDSA.simdist import SimDistConfig, FastDSASimilarity

# Example configuration (edit these values to match your experiment)
cfg = SimDistConfig(
    n_delays=15,
    delay_interval=1,
    rank=None,           # set to None to enable automatic SVHT rank selection
    method="ro",         # "ro", "rim", "land", or "kw"
    iters=200,
    lr=1e-2,
    device="cuda",       # recommended if available
    verbose=False,
)

sim = FastDSASimilarity(cfg)

# dataset_A and dataset_B can be:
#   - a single trajectory shaped (C, T), or
#   - a list of trajectories, each shaped (C, T), or
#   - a batch array shaped (n_trials, C, T)
score, used_rank = sim.fit_score(dataset_A, dataset_B)

print("fastDSA score:", score)
print("rank used:", used_rank)
```
For a more detailed tutorial—including applying fastDSA to data shaped `(trials, channels, timepoints)`—see:

https://github.com/CMC-lab/fastDSA/blob/main/Tutorial/Tutorial1.ipynb


**fastDSA uses PyTorch and supports CUDA acceleration. For large datasets, using a CUDA-enabled GPU is strongly recommended.**

The package supports multiple similarity backends via the `method` argument in `SimDistConfig`:

`"ro"`: RegularizationTerm-based similarity transform distance

`"rim"`: Riemannian manifold optimization-based distance

`"land"`: Landing-style optimization-based distance

`"kw"`: Kernel-based Wasserstein distance (kernel DMD)


For methodological details and guidance on choosing among these options, please refer to the paper.