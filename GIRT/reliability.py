# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
# 
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# information and shall use it only in accordance with the terms of
# the license agreement.



import argparse
import numpy as np 
import pandas as pd 
import torch 
from model.girt_arch import GIRTDataset, GenItemResponseTheoryModel2PL
from tqdm import tqdm
from tools import degree_of_consistency

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',help='the path of the train file')
    parser.add_argument('--test_file',help='the path of the test file')
    parser.add_argument('--model_path',help='the model path')
    parser.add_argument('--theta_path',help='The theta .npy file path. Leave blank if you want to test the model.', default=None)
    parser.add_argument('--theta_evidence', help='Where does diagnosis data come from. Either "train" or "test".')

    args = parser.parse_args()

    train_sm = np.loadtxt(args.train_file)
    eval_sm = np.loadtxt(args.test_file)

    n_user, n_item = train_sm.shape

    
    if args.theta_evidence == 'test':
        evidence_sm = eval_sm
    elif args.theta_evidence == 'train':
        evidence_sm = train_sm
    # dataloader = DataLoader(
    #     dataset = dataset, batch_size = 1, shuffle = False)
    theta_list = []
    
    theta_mat = None
    ### Load model
    if args.theta_path is None:
        net = GenItemResponseTheoryModel2PL()
        net.load(args.model_path)
        for user_id in tqdm(range(n_user)):
            user_log = np.array([evidence_sm[user_id]])
            theta = net.diagnose_learner(user_log)
            theta_list.append(theta)
        theta_mat = np.concatenate(theta_list, axis=0)
    
    else:
        theta_mat = np.load(args.theta_path)

    test_user_know_hit = np.ones(shape=(n_user, 1))
    Q_mat = np.ones(shape=(n_item, 1))
    
    doc = degree_of_consistency(\
        theta_mat = theta_mat, \
        user_know_hit = test_user_know_hit, \
        log_mat = eval_sm, \
        Q_mat = Q_mat, \
        know_list = None)
    
    print(f'>>> Degree of Consistency = {doc:.3f}')
