#!/bin/bash
# =============================================================================
# R6: Matched-Distribution Downstream Evaluation
# Self-generated training data eliminates distribution shift
# 1 GPU, 48G mem (generation + training + beam search in one job)
# =============================================================================
#SBATCH --job-name=r6_matched
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-statelift}"
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # set to 1 on clusters without internet (then LLAMA_PATH must be a local dir)
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
# set HF_HOME / TORCH_HOME here if needed
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export NCCL_P2P_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1


echo "[INFO] Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "[INFO] Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

rm -f /dev/shm/psm_* 2>/dev/null || true
rm -f /dev/shm/vllm_* 2>/dev/null || true

export LLAMA_PATH="${LLAMA_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_PATH="${LLAMA_PATH}"
SCRIPT_PATH="scripts/training/r6_matched_downstream.py"

if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" ]]; then echo "[FAIL] Model: ${MODEL_PATH}"; exit 1; fi
if [ ! -f "${SCRIPT_PATH}" ]; then echo "[FAIL] Script: ${SCRIPT_PATH}"; exit 1; fi
echo "[OK] Paths validated"

python -u "${SCRIPT_PATH}"

rm -rf "/tmp/vllm_shm_$(id -u)" 2>/dev/null || true
echo "[OK] Completed at $(date)"
