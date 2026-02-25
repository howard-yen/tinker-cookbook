#!/usr/bin/env bash
set -euo pipefail

# training runs, do lr search
for lr in 1e-6 5e-6 1e-5; do
    for rank in 16 32; do
        LOG_LEVEL=10 BZ=4 GS=8 EGS=8 NS=25 TRAIN_SPLIT=bcplus EVAL_SPLIT=browsecomp_plus EVAL_EVERY=5 N_BATCHES=50 LR=$lr LR_SCHEDULE=cosine WARMUP_STEPS=5 RANK=$rank bash run.sh
    done
done
