# Running fastDSA On HPC With Slurm

This folder contains a Slurm-oriented entrypoint and an `sbatch` template for running the same pairwise fastDSA workflow on a cluster.

Files:

- `run_fastdsa_slurm.py`: Python entrypoint used by the Slurm job.
- `submit_fastdsa.sbatch`: editable Slurm submission script.

The Slurm entrypoint reuses the local runner in `Tutorial/single_py/run_fastdsa_single.py`, with one HPC-specific setting: the automatic setup stage uses `selection_iters=1`.

## Data Format

Each dataset should be a folder of `.npy` files, with one file per trial.

By default, each file is assumed to have shape:

```text
(timepoints, features)
```

The script transposes each trial before calling fastDSA. If your files are already shaped `(channels, timepoints)`, set:

```bash
export INPUT_SHAPE=channels_time
```

## Submit A Synthetic Test Job

From the repository root:

```bash
sbatch Tutorial/hpc_slurm/submit_fastdsa.sbatch
```

When no dataset folders are provided, the job uses the built-in synthetic two-system demo.

## Submit A Real Data Job

```bash
sbatch Tutorial/hpc_slurm/submit_fastdsa.sbatch \
  /path/to/dataset_A_npy_files \
  /path/to/dataset_B_npy_files \
  /path/to/output_folder
```

## Common Slurm Settings

Edit these lines in `submit_fastdsa.sbatch` for your cluster:

```bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
##SBATCH --gres=gpu:1
```

If your cluster uses GPUs, uncomment and adjust the GPU line. If your cluster requires modules or conda activation, add those commands before the final `srun` line.

For example:

```bash
module load python
source /path/to/venv/bin/activate
```

or:

```bash
module load anaconda
conda activate fastdsa
```

## Runtime Options

The batch script reads environment variables so you can change common settings without editing the file:

```bash
METHOD=kw DEVICE=cuda ITERS=200 \
sbatch Tutorial/hpc_slurm/submit_fastdsa.sbatch /path/to/A /path/to/B /path/to/out
```

Available environment variables:

- `PYTHON_BIN`: Python executable. Defaults to `python`.
- `METHOD`: fastDSA backend, one of `ro`, `rim`, `land`, or `kw`.
- `DEVICE`: `cuda`, `cuda:0`, `cpu`, or `auto`.
- `ITERS`: pairwise optimization iterations.
- `INPUT_SHAPE`: `time_features` or `channels_time`.

## Plotting Options

Heatmap and MDS plotting are enabled by default in the Slurm template.

Disable either plot by setting the corresponding environment variable to `0`:

```bash
PLOT_HEATMAP=0 PLOT_MDS=1 \
sbatch Tutorial/hpc_slurm/submit_fastdsa.sbatch /path/to/A /path/to/B /path/to/out
```

Plot outputs:

- `distance_matrix_heatmap.png`
- `distance_matrix_heatmap.pdf`
- `mds_plot.png`
- `mds_plot.pdf`
- `mds_coordinates.csv`

## Outputs

The output folder contains:

- `distance_matrix.npy`: NumPy distance matrix.
- `distance_matrix.csv`: CSV version of the same matrix.
- `trial_labels.txt`: row/column labels for the matrix.
- `run_info.json`: selected q-star/delay/rank settings and run metadata.
- Optional heatmap and MDS files, depending on `PLOT_HEATMAP` and `PLOT_MDS`.
