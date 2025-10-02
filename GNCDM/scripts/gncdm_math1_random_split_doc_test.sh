#!/bin/bash

python reliability.py \
    --train_file ~/data/math1/random_split/train.csv \
    --valid_file ~/data/math1/random_split/valid.csv \
    --test_file ~/data/math1/random_split/test.csv \
    --Q_matrix ~/data/math1/Q_matrix.npy \
    --model_path ./result/math1_random_split/params_32_32.pt\
    --n_user 4209 \
    --n_item 20 \
    --n_know 11 \
    --theta_evidence test
