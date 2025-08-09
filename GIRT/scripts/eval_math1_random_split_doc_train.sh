#!/bin/bash

python reliability.py \
    --train_file ./data/Math1-Random-Split/train.txt \
    --test_file ./data/Math1-Random-Split/test.txt \
    --model_path ./checkpoint/girt2pl-math1-random-split/checkpoint-epoch-10.pt\
    --theta_evidence train
