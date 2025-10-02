#!/bin/bash

python run.py \
    --train_file ~/data/math1/random_split/train.csv \
    --valid_file ~/data/math1/random_split/valid.csv \
    --test_file ~/data/math1/random_split/test.csv \
    --Q_matrix ~/data/math1/Q_matrix.npy \
    --save_path ./result/math1_random_split \
    --n_user 4209 \
    --n_item 20 \
    --n_know 11 \
    --user_dim 32 \
    --item_dim 32 \
    --alpha 0.8 \
    --training_config config/training_config_math1.json
