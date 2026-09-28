#!/bin/bash
# Submit multi-seed training jobs to SLURM queue
# 4 domains x 3 seeds = 12 jobs

QUEUE_BASE="${QUEUE_BASE:-logs/queue}"   # staging dir for the job-queue helper (cluster-specific)
DATA_DIR="data"
SCRIPT="scripts/training/e6_multiseed_train.py"

DOMAINS="casino dealornodeal esconv hhrlhf"
SEEDS="42 123 7"

DATA_FILES="e4_casino_data.pt e5_dealornodeal_data.pt e4_esconv_data.pt e4_hh_data.pt"

for domain in $DOMAINS; do
    for seed in $SEEDS; do
        TS=$(date +%Y%m%d_%H%M%S)_${domain}_s${seed}
        JOB_DIR="${QUEUE_BASE}/${TS}"
        mkdir -p "${JOB_DIR}/payload"

        # Copy script
        cp "$SCRIPT" "${JOB_DIR}/payload/"

        # Copy the right data file
        case $domain in
            casino) cp "${DATA_DIR}/e4_casino_data.pt" "${JOB_DIR}/payload/" ;;
            dealornodeal) cp "${DATA_DIR}/e5_dealornodeal_data.pt" "${JOB_DIR}/payload/" ;;
            esconv) cp "${DATA_DIR}/e4_esconv_data.pt" "${JOB_DIR}/payload/" ;;
            hhrlhf) cp "${DATA_DIR}/e4_hh_data.pt" "${JOB_DIR}/payload/" ;;
        esac

        # Create meta.yaml
        cat > "${JOB_DIR}/meta.yaml" << EOF
type: slurm
command: python -u e6_multiseed_train.py --domain ${domain} --seed ${seed}
partition: ${SLURM_PARTITION:-gpu}
gres: gpu:1
conda_env: ${CONDA_ENV:-statelift}
time: "04:00:00"
mem: 48G
cpus_per_task: 4
EOF

        echo "Staged: ${domain} seed=${seed} -> ${TS}"
        sleep 1  # unique timestamps
    done
done

echo "Done: 12 jobs staged"
