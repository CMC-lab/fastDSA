#!/bin/bash
#SBATCH --job-name=fastdsa
#SBATCH --output=Tutorial/hpc_slurm/fastdsa_%j.out
#SBATCH --error=Tutorial/hpc_slurm/fastdsa_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
##SBATCH --gres=gpu:1

set -euo pipefail

# Submit from the repository root:
#   sbatch Tutorial/hpc_slurm/submit_fastdsa.sbatch /path/to/A /path/to/B /path/to/output
#
# If dataset folders are omitted, the job runs the built-in synthetic example.

cd "${SLURM_SUBMIT_DIR}"

DATASET_A="${1:-}"
DATASET_B="${2:-}"
OUTPUT_DIR="${3:-Tutorial/hpc_slurm/results_${SLURM_JOB_ID}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
METHOD="${METHOD:-kw}"
DEVICE="${DEVICE:-cuda}"
ITERS="${ITERS:-200}"
SELECTION_ITERS="${SELECTION_ITERS:-50}"
INPUT_SHAPE="${INPUT_SHAPE:-time_features}"
PLOT_HEATMAP="${PLOT_HEATMAP:-1}"
PLOT_MDS="${PLOT_MDS:-1}"

ARGS=(
  Tutorial/hpc_slurm/run_fastdsa_slurm.py
  --output-dir "${OUTPUT_DIR}"
  --method "${METHOD}"
  --device "${DEVICE}"
  --iters "${ITERS}"
  --selection-iters "${SELECTION_ITERS}"
  --input-shape "${INPUT_SHAPE}"
)

if [[ -n "${DATASET_A}" && -n "${DATASET_B}" ]]; then
  ARGS+=(--dataset-a "${DATASET_A}" --dataset-b "${DATASET_B}")
else
  ARGS+=(--synthetic)
fi

if [[ "${PLOT_HEATMAP}" == "1" ]]; then
  ARGS+=(--plot-heatmap)
fi

if [[ "${PLOT_MDS}" == "1" ]]; then
  ARGS+=(--plot-mds)
fi

srun "${PYTHON_BIN}" "${ARGS[@]}"
