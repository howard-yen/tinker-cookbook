#!/usr/bin/env bash

# training runs, do lr search

for lr in 1e-6 5e-6 1e-5; do
    for rank in 16 32; do
        BZ=4 GS=8 EGS=8 NS=25 TRAIN_SPLIT=bcplus EVAL_SPLIT=browsecomp_plus EVAL_EVERY=5 N_BATCHES=50 LR=$lr RANK=$rank bash run.sh
    done
done
