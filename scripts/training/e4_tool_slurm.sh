#!/bin/bash
# =============================================================================
# E4: PRM vs ORM Math Reward Model (Llama-3.1-8B on GSM8K)
# 1 GPU, ~16GB model + LoRA training
# =============================================================================
#SBATCH --job-name=e4_tool_rm
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Conda environment ----
CONDA_ENV="${CONDA_ENV:-statelift}"
source activate "${CONDA_ENV}" 2>/dev/null || conda activate "${CONDA_ENV}"

# ---- Offline mode (optional) ----
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # set to 1 on clusters without internet (then LLAMA_PATH must be a local dir)
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
# set HF_HOME / TORCH_HOME here if needed
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0

# ---- Multi-GPU fixes (MANDATORY for TP > 1) ----
export NCCL_P2P_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0


# ---- Diagnostics ----
echo "[INFO] Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "[INFO] Conda env: ${CONDA_ENV}"
echo "[INFO] Python: $(which python)"
echo "[INFO] GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ---- Stale shared memory cleanup ----
rm -f /dev/shm/psm_* 2>/dev/null || true
rm -f /dev/shm/vllm_* 2>/dev/null || true

# ---- Validate paths ----
export LLAMA_PATH="${LLAMA_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_PATH="${LLAMA_PATH}"
DATA_PATH="data/e4_tool_data.pt"
SCRIPT_PATH="scripts/training/e4_tool_train.py"

if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" ]]; then
    echo "[FAIL] Model not found: ${MODEL_PATH}"
    exit 1
fi
if [ ! -f "${DATA_PATH}" ]; then
    echo "[FAIL] Data not found: ${DATA_PATH}"
    exit 1
fi
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "[FAIL] Script not found: ${SCRIPT_PATH}"
    exit 1
fi

echo "[OK] All paths validated"

# ===========================================================================
# RUN
# ===========================================================================
python -u "${SCRIPT_PATH}"

# ---- Cleanup ----
rm -rf "/tmp/vllm_shm_$(id -u)" 2>/dev/null || true
echo "[OK] Job completed at $(date)"
