#!/bin/bash

python run.py \
    --train_file ~/data/a0910/new_user_split/train.csv \
    --valid_file ~/data/a0910/new_user_split/valid.csv \
    --test_file ~/data/a0910/new_user_split/test.csv \
    --Q_matrix ~/data/a0910/Q_matrix.npy \
    --save_path ./result/a0910_user_split \
    --n_user 4163 \
    --n_item 17746 \
    --n_know 123 \
    --user_dim 32 \
    --item_dim 32 \
    --alpha 0.9 \
    --training_config config/training_config_a0910.json
