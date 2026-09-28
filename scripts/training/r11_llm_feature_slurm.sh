#!/bin/bash
# =============================================================================
# R11: State-lift with frozen LLM hidden states (resolution hypothesis)
# 1 GPU, ~16GB for Llama-8B inference (no training)
# =============================================================================
#SBATCH --job-name=r11_llmsl
#SBATCH --partition=${SLURM_PARTITION:-gpu}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
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


echo "[INFO] Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

export LLAMA_PATH="${LLAMA_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_PATH="${LLAMA_PATH}"
DATA_PATH="data/PRM800K/phase2_train.jsonl"
SCRIPT_PATH="scripts/training/r11_llm_feature_sl.py"

if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" ]]; then echo "[FAIL] Model: ${MODEL_PATH}"; exit 1; fi
for P in "${DATA_PATH}" "${SCRIPT_PATH}"; do
    if [ ! -e "${P}" ]; then
        echo "[FAIL] Not found: ${P}"
        exit 1
    fi
done
echo "[OK] Paths validated"

python -u "${SCRIPT_PATH}"

echo "[OK] Completed at $(date)"
