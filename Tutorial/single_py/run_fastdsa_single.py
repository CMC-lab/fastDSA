#!/usr/bin/env python3
"""Run fastDSA pairwise distances from one Python file.

Input datasets are folders containing `.npy` files, one file per trial. By
default, each file is assumed to be shaped `(timepoints, features)` and is
transposed before calling fastDSA, which expects `(channels, timepoints)`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.manifold import MDS

from fastDSA.simdist import FastDSASimilarity, SimDistConfig


def system_a(t, state, eps=0.5):
    x, y = state
    dx = -1.0 * x + eps * x * y
    dy = -2.0 * y - eps * x**2
    return [dx, dy]


def system_b(t, state, eps=0.5):
    x, y = state
    dx = -1.5 * x + 0.5 * y - eps * x**2
    dy = 0.5 * x - 1.5 * y - eps * y**2
    return [dx, dy]


def generate_trajectory(system, x0, total_time=10.0, dt=0.01, eps=0.5):
    t_eval = np.arange(0, total_time, dt)
    sol = solve_ivp(system, (0, total_time), x0, t_eval=t_eval, args=(eps,))
    return sol.y.astype(np.float32)


def make_synthetic_datasets(num_per_set=10, total_time=10.0, dt=0.01, eps=0.5, seed=42):
    rng = np.random.default_rng(seed)
    dataset_a = [
        generate_trajectory(system_a, rng.uniform(-4, 4, size=2), total_time, dt, eps)
        for _ in range(num_per_set)
    ]
    dataset_b = [
        generate_trajectory(system_b, rng.uniform(-4, 4, size=2), total_time, dt, eps)
        for _ in range(num_per_set)
    ]
    return dataset_a, dataset_b


def load_dataset(folder: Path, input_shape: str):
    paths = sorted(folder.glob("*.npy"))
    if not paths:
        raise ValueError(f"No .npy files found in {folder}")

    dataset = []
    for path in paths:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be 2D; got shape {arr.shape}")
        if input_shape == "time_features":
            arr = arr.T
        dataset.append(np.asarray(arr, dtype=np.float32))
    return dataset


def resolve_device(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; using CPU.")
        return "cpu"
    return device


def select_embedding_and_rank(dataset_a, dataset_b, args):
    if args.n_delays is not None and args.rank is not None:
        return int(args.n_delays), int(args.delay_interval or 1), int(args.rank), None

    cfg_select = SimDistConfig(
        n_delays=args.n_delays,
        delay_interval=args.delay_interval,
        rank=args.rank,
        method=args.method,
        iters=args.selection_iters,
        lr=args.lr,
        device=args.device,
        verbose=args.verbose,
    )
    sim_select = FastDSASimilarity(cfg_select)
    _, used_rank = sim_select.fit_score(dataset_a, dataset_b)

    selected_n_delays = getattr(sim_select, "selected_n_delays_", None)
    selected_delay_interval = getattr(sim_select, "selected_delay_interval_", None)
    if selected_n_delays is None:
        selected_n_delays = args.n_delays
    if selected_delay_interval is None:
        selected_delay_interval = args.delay_interval or 1
    if selected_n_delays is None:
        raise RuntimeError("Could not resolve n_delays. Provide --n-delays or enable q-star mode.")

    diagnostics = {
        "q_star_result": getattr(sim_select, "q_star_result_", None),
        "hankel_selection_table": getattr(sim_select, "hankel_selection_table_", None),
    }
    return int(selected_n_delays), int(selected_delay_interval), int(used_rank), diagnostics


def compute_pairwise_distance_matrix(all_trials, cfg_pairwise):
    n_trials = len(all_trials)
    dist_mat = np.zeros((n_trials, n_trials), dtype=float)

    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            sim = FastDSASimilarity(cfg_pairwise)
            distance, _ = sim.fit_score([all_trials[i]], [all_trials[j]])
            dist_mat[i, j] = dist_mat[j, i] = float(distance)
            print(f"pair ({i + 1}, {j + 1}) distance = {distance:.6g}", flush=True)

    return dist_mat


def plot_heatmap(dist_mat, labels, output_dir):
    plt.figure(figsize=(8, 7))
    im = plt.imshow(dist_mat, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, label="fastDSA distance")
    n_a = sum(label.startswith("A") for label in labels)
    plt.axhline(n_a - 0.5, color="white", linewidth=1.5)
    plt.axvline(n_a - 0.5, color="white", linewidth=1.5)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Trial")
    plt.ylabel("Trial")
    plt.title("Pairwise fastDSA distance matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "distance_matrix_heatmap.png", dpi=200)
    plt.savefig(output_dir / "distance_matrix_heatmap.pdf")
    plt.close()


def plot_mds(dist_mat, labels, output_dir, random_state=42):
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state).fit_transform(dist_mat)
    np.savetxt(
        output_dir / "mds_coordinates.csv",
        coords,
        delimiter=",",
        header="MDS-1,MDS-2",
        comments="",
    )

    labels = np.asarray(labels)
    is_a = np.char.startswith(labels, "A")
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[is_a, 0], coords[is_a, 1], label="Dataset A", alpha=0.85)
    plt.scatter(coords[~is_a, 0], coords[~is_a, 1], label="Dataset B", alpha=0.85)
    for label, (x_coord, y_coord) in zip(labels, coords):
        plt.annotate(label, (x_coord, y_coord), fontsize=8, alpha=0.8)
    plt.xlabel("MDS-1")
    plt.ylabel("MDS-2")
    plt.title("MDS of fastDSA distances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mds_plot.png", dpi=200)
    plt.savefig(output_dir / "mds_plot.pdf")
    plt.close()


def run_fastdsa_pairwise(args):
    args.device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        dataset_a, dataset_b = make_synthetic_datasets(
            num_per_set=args.num_per_set,
            total_time=args.total_time,
            dt=args.dt,
            eps=args.eps,
            seed=args.seed,
        )
    else:
        if args.dataset_a is None or args.dataset_b is None:
            raise ValueError("Provide --dataset-a and --dataset-b, or use --synthetic.")
        dataset_a = load_dataset(Path(args.dataset_a), args.input_shape)
        dataset_b = load_dataset(Path(args.dataset_b), args.input_shape)

    labels = [f"A{i + 1}" for i in range(len(dataset_a))] + [f"B{i + 1}" for i in range(len(dataset_b))]
    selected_n_delays, selected_delay_interval, used_rank, diagnostics = select_embedding_and_rank(
        dataset_a,
        dataset_b,
        args,
    )

    cfg_pairwise = SimDistConfig(
        n_delays=selected_n_delays,
        delay_interval=selected_delay_interval,
        rank=used_rank,
        method=args.method,
        iters=args.iters,
        lr=args.lr,
        eta=args.eta,
        device=args.device,
        verbose=args.verbose,
    )

    dist_mat = compute_pairwise_distance_matrix(dataset_a + dataset_b, cfg_pairwise)
    np.save(output_dir / "distance_matrix.npy", dist_mat)
    np.savetxt(output_dir / "distance_matrix.csv", dist_mat, delimiter=",")
    (output_dir / "trial_labels.txt").write_text("\n".join(labels) + "\n")

    run_info = {
        "method": args.method,
        "device": args.device,
        "n_delays": selected_n_delays,
        "delay_interval": selected_delay_interval,
        "rank": used_rank,
        "n_dataset_a": len(dataset_a),
        "n_dataset_b": len(dataset_b),
        "diagnostics": diagnostics,
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str) + "\n")

    if args.plot_heatmap:
        plot_heatmap(dist_mat, labels, output_dir)
    if args.plot_mds:
        plot_mds(dist_mat, labels, output_dir, random_state=args.seed)

    print(f"Saved outputs to {output_dir.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description="Compute pairwise fastDSA distances from one Python file.")
    parser.add_argument("--dataset-a", type=Path, default=None, help="Folder of .npy trials for Dataset A.")
    parser.add_argument("--dataset-b", type=Path, default=None, help="Folder of .npy trials for Dataset B.")
    parser.add_argument("--synthetic", action="store_true", help="Use the built-in synthetic two-system demo.")
    parser.add_argument(
        "--input-shape",
        choices=["time_features", "channels_time"],
        default="time_features",
        help="Shape of loaded .npy files. fastDSA receives channels_time.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("fastdsa_single_results"))
    parser.add_argument("--method", choices=["ro", "rim", "land", "kw"], default="kw")
    parser.add_argument("--n-delays", type=int, default=None, help="Manual delay count. Omit for q-star.")
    parser.add_argument("--delay-interval", type=int, default=None, help="Delay spacing. Omit for default 1.")
    parser.add_argument("--rank", type=int, default=None, help="Manual DMD rank. Omit for SVHT.")
    parser.add_argument("--iters", type=int, default=200, help="Optimization iterations for pairwise distances.")
    parser.add_argument("--selection-iters", type=int, default=50, help="Cheap setup iterations for automatic selection.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eta", type=float, default=None, help="Landing-method eta. Defaults to package behavior.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or a CUDA device such as 'cuda:0'.")
    parser.add_argument("--plot-heatmap", action="store_true", help="Save PNG/PDF heatmap of the distance matrix.")
    parser.add_argument("--plot-mds", action="store_true", help="Save PNG/PDF MDS plot from the distance matrix.")
    parser.add_argument("--num-per-set", type=int, default=10, help="Synthetic trials per dataset.")
    parser.add_argument("--total-time", type=float, default=10.0, help="Synthetic trajectory duration.")
    parser.add_argument("--dt", type=float, default=0.01, help="Synthetic sampling interval.")
    parser.add_argument("--eps", type=float, default=0.5, help="Synthetic system epsilon.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    run_fastdsa_pairwise(build_parser().parse_args())
