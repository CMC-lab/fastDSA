# Running fastDSA From One Python File

This folder contains a standalone script for computing a pairwise fastDSA distance matrix from two datasets:

```bash
python Tutorial/single_py/run_fastdsa_single.py --help
```

Each input dataset should be a folder of `.npy` files, with one file per trial.

By default, each `.npy` file is assumed to have shape:

```text
(timepoints, features)
```

The script transposes each trial before calling fastDSA, because fastDSA expects:

```text
(channels, timepoints)
```

If your files are already shaped `(channels, timepoints)`, pass:

```bash
--input-shape channels_time
```

## Quick Synthetic Run

Use this command to run the built-in two-system synthetic example and save both plots:

```bash
python Tutorial/single_py/run_fastdsa_single.py \
  --synthetic \
  --method kw \
  --output-dir Tutorial/single_py/results_synthetic \
  --plot-heatmap \
  --plot-mds
```

## Run On Your Own `.npy` Files

```bash
python Tutorial/single_py/run_fastdsa_single.py \
  --dataset-a /path/to/dataset_A_npy_files \
  --dataset-b /path/to/dataset_B_npy_files \
  --input-shape time_features \
  --method kw \
  --output-dir Tutorial/single_py/results_real_data \
  --plot-heatmap \
  --plot-mds
```

## Main Options

- `--method`: fastDSA backend. Use one of `ro`, `rim`, `land`, or `kw`.
- `--n-delays`: manual number of delays. Omit this to use automatic q-star selection.
- `--delay-interval`: manual delay spacing. Omit this to use the package default.
- `--rank`: manual DMD rank. Omit this to use SVHT rank selection.
- `--device`: use `auto`, `cpu`, `cuda`, or a specific CUDA device such as `cuda:0`.
- `--iters`: optimization iterations for pairwise distance computation.
- `--selection-iters`: cheaper iterations used during automatic setup.

## Plotting Options

- `--plot-heatmap`: saves `distance_matrix_heatmap.png` and `distance_matrix_heatmap.pdf`.
- `--plot-mds`: saves `mds_plot.png`, `mds_plot.pdf`, and `mds_coordinates.csv`.

## Outputs

The output folder contains:

- `distance_matrix.npy`: NumPy distance matrix.
- `distance_matrix.csv`: CSV version of the same matrix.
- `trial_labels.txt`: row/column labels for the matrix.
- `run_info.json`: selected q-star/delay/rank settings and run metadata.
- Optional heatmap and MDS files if the plotting flags are enabled.
