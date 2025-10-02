#!/bin/bash

python run.py \
    --train_file ~/data/algebra/random_split/train.csv \
    --valid_file ~/data/algebra/random_split/valid.csv \
    --test_file ~/data/algebra/random_split/test.csv \
    --Q_matrix ~/data/algebra/Q_matrix.npy \
    --save_path ./result/algebra_random_split \
    --n_user 1336 \
    --n_item 100000 \
    --n_know 491 \
    --user_dim 32 \
    --item_dim 32 \
    --alpha 0.9 \
    --batch_size 64 \
    --lr 0.0002 \
    --epoch 1 \
    --device cuda:2 \
