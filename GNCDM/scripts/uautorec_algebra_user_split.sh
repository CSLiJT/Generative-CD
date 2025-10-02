#!/bin/bash

python run_ae.py \
    --model UAutoRec \
    --train_file ~/data/algebra/user_split/train.csv \
    --valid_file ~/data/algebra/user_split/valid.csv \
    --test_file ~/data/algebra/user_split/test.csv \
    --batch_size 32 \
    --lr 0.01 \
    --n_epoch 1 \
    --device 'cuda:3' \
