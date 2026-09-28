#!/bin/bash
# Download the two raw text corpora that are not fetched through HF `datasets`.
# Run from the repo root:  bash scripts/data/download_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "${ROOT}/data"
cd "${ROOT}/data"

# Download PersuasionForGood (public HF dataset mirror)
wget "https://huggingface.co/datasets/spawn99/PersuasionForGood/resolve/main/Full%20Dialog.csv" \
  -O persuasion_full.csv 2>&1

# Download DealOrNoDeal
wget -q "https://raw.githubusercontent.com/facebookresearch/end-to-end-negotiator/master/src/data/negotiate/train.txt" \
  -O dealornodeal_train.txt 2>&1

echo "DONE"
ls -la persuasion_full.csv dealornodeal_train.txt 2>/dev/null
