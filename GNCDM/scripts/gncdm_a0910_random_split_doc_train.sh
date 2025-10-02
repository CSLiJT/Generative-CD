#!/bin/bash

python reliability.py \
    --train_file ~/data/a0910/new_random_split/train.csv \
    --valid_file ~/data/a0910/new_random_split/valid.csv \
    --test_file ~/data/a0910/new_random_split/test.csv \
    --Q_matrix ~/data/a0910/Q_matrix.npy \
    --model_path ./result/a0910_random_split/params_32_32.pt\
    --n_user 4163 \
    --n_item 17746 \
    --n_know 123 \
    --theta_evidence train
