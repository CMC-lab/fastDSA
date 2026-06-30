#!/usr/bin/env python3
"""Slurm-friendly entrypoint for the single-file fastDSA runner.

Run this from the repository root, or submit it through `submit_fastdsa.sbatch`.
The implementation reuses `Tutorial/single_py/run_fastdsa_single.py` so local
and HPC runs share the same implementation. The HPC entrypoint fixes
selection_iters=1 for the automatic setup stage.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Tutorial.single_py.run_fastdsa_single import build_parser, run_fastdsa_pairwise


def build_slurm_parser():
    parser = build_parser()

    action = parser._option_string_actions.pop("--selection-iters", None)
    if action is not None:
        parser._actions.remove(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)

    return parser


if __name__ == "__main__":
    args = build_slurm_parser().parse_args()
    args.selection_iters = 1
    run_fastdsa_pairwise(args)
