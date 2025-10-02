#!/bin/bash

python run_ae.py \
    --model UAutoRec \
    --train_file ~/data/math1/user_split/train.csv \
    --valid_file ~/data/math1/user_split/valid.csv \
    --test_file ~/data/math1/user_split/test.csv \
    --batch_size 16 \
    --lr 0.0001 \
    --n_epoch 1 \
    --device 'cuda:0' \
