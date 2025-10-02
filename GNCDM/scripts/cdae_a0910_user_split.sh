#!/bin/bash

python run_ae.py \
    --model CDAE \
    --train_file ~/data/a0910/new_user_split/train.csv \
    --valid_file ~/data/a0910/new_user_split/valid.csv \
    --test_file ~/data/a0910/new_user_split/test.csv \
    --batch_size 32 \
    --lr 0.01 \
    --n_epoch 5 \
    --device 'cuda:0' \
