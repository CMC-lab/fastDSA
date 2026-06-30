#!/usr/bin/env python3
"""Slurm-friendly entrypoint for the single-file fastDSA runner.

Run this from the repository root, or submit it through `submit_fastdsa.sbatch`.
The implementation reuses `Tutorial/single_py/run_fastdsa_single.py` so local
and HPC runs share the same command-line options.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Tutorial.single_py.run_fastdsa_single import build_parser, run_fastdsa_pairwise


if __name__ == "__main__":
    run_fastdsa_pairwise(build_parser().parse_args())
