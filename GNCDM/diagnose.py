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
import os
import pandas as pd 
import torch 
import model 
from tqdm import tqdm
from train import IDCDataset
from torch.utils.data import DataLoader
from tools import degree_of_consistency

def add_knowledge_code(data: pd.DataFrame, Q_mat):
    knowledge = []
    for i in range(data.shape[0]):
        knowledge.append(Q_mat[data.loc[i,'item_id']])
    data['knowledge'] = knowledge
    return data 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence_file',help='the path of the evidence file')
    parser.add_argument('--Q_matrix',help='the path of the q-matrix', default='')
    parser.add_argument('--model_path',help='the model path')
    parser.add_argument('--output_path',help='the model path')
    parser.add_argument('--n_user', help='the number of students in the entire dataset')
    parser.add_argument('--n_item', help='the number of exercises in the entire dataset')
    parser.add_argument('--n_know', help='the number of knowledge points in the entire dataset')

    args = parser.parse_args()

    df_train = pd.read_csv(args.evidence_file)

    n_user = int(args.n_user)
    n_item = int(args.n_item)
    n_know = int(args.n_know)

    Q_mat = np.load(args.Q_matrix) if args.Q_matrix !='' else np.ones((n_item, n_know))

    df_train = add_knowledge_code(df_train, Q_mat)

    ### Load model
    net = torch.load(args.model_path, weights_only=False)

    dataset = IDCDataset(
        df_train,n_user=n_user, n_item=n_item, Q_mat=Q_mat)
    
    theta_list = []
    device = net.device
    
    for user_id in tqdm(range(n_user)):
        user_log = torch.Tensor([dataset.log_mat[user_id]])
        theta = net.diagnose_theta(user_log.to(device))\
            .detach().cpu().numpy()
        theta_list.append(theta)
    
    theta_mat = np.concatenate(theta_list, axis=0)

    score_rates = []
    for user_id in tqdm(range(n_user)):
        score_vec = dataset.log_mat[user_id]
        n_correct = len(score_vec[score_vec > 0])
        n_all = len(score_vec[score_vec != 0])
        acc = n_correct/n_all if n_all != 0 else -0.2
        score_rates.append(acc)
    score_rates = np.array(score_rates)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    np.save(os.path.join(args.output_path, 'theta.npy'), theta_mat)
    np.save(os.path.join(args.output_path, 'score_rate.npy'), score_rates)
    

