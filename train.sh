#!/usr/bin/env bash
set -euo pipefail

# training runs, do lr search
for rank in 32 ; do
    for lr in 1e-6 5e-6 1e-5; do
        LOG_LEVEL=20 NUM_GROUPS_TO_LOG=4 BZ=16 GS=8 EGS=8 NS=25 TRAIN_SPLIT=browsecomp_plus_train EVAL_SPLIT=browsecomp_plus_test SAVE_EVERY=8 EVAL_EVERY=8 N_BATCHES=128 LR=$lr LR_SCHEDULE=cosine WARMUP_STEPS=13 RANK=$rank DO_FALLBACK_CHALLENGER=false bash run.sh
    done
done
