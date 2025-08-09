#!/bin/bash

python reliability.py \
    --train_file ./data/ASSIST0910-Random-Split/train.txt \
    --test_file ./data/ASSIST0910-Random-Split/test.txt \
    --model_path ./checkpoint/girt2pl-assist0910-random-split/checkpoint-epoch-20.pt\
    --theta_evidence test
