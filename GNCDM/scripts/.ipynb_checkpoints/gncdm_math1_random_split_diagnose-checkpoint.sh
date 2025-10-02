#!/bin/bash

python diagnose.py \
    --evidence_file ~/data/math1/random_split/train.csv \
    --Q_matrix ~/data/math1/Q_matrix.npy \
    --model_path ./result/math1_random_split/params_32_32.pt\
    --output_path ./result/gncdm_diagnose_math1_random_split/ \
    --n_user 4209 \
    --n_item 20 \
    --n_know 11 \