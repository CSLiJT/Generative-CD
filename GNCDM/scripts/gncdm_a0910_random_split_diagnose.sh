#!/bin/bash

python diagnose.py \
    --evidence_file ~/data/a0910/new_random_split/train.csv \
    --Q_matrix ~/data/a0910/Q_matrix.npy \
    --model_path ./result/a0910_random_split/params_32_32.pt\
    --output_path ./result/gncdm_diagnose_a0910_random_split/ \
    --n_user 4163 \
    --n_item 17746 \
    --n_know 123 