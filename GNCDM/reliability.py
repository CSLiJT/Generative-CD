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
    parser.add_argument('--train_file',help='the path of the train file')
    parser.add_argument('--test_file',help='the path of the test file')
    parser.add_argument('--valid_file',help='the path of the valid file', default='')
    parser.add_argument('--Q_matrix',help='the path of the q-matrix', default='')
    parser.add_argument('--model_path',help='the model path')
    parser.add_argument('--theta_path',help='The theta .npy file path. Leave along if you want to test the model.', default=None)
    parser.add_argument('--n_user', help='the number of students in the entire dataset')
    parser.add_argument('--n_item', help='the number of exercises in the entire dataset')
    parser.add_argument('--n_know', help='the number of knowledge points in the entire dataset')
    parser.add_argument('--theta_evidence', help='Where does diagnosis data come from. Either "train" or "test".')

    args = parser.parse_args()

    df_train = pd.read_csv(args.train_file)
    df_valid = pd.read_csv(args.valid_file)
    df_test = pd.read_csv(args.test_file)

    n_user = int(args.n_user)
    n_item = int(args.n_item)
    n_know = int(args.n_know)

    Q_mat = np.load(args.Q_matrix) if args.Q_matrix !='' else np.ones((n_item, n_know))

    df_train = add_knowledge_code(df_train, Q_mat)
    df_valid = add_knowledge_code(df_valid, Q_mat)
    df_test = add_knowledge_code(df_test, Q_mat)

    ### Load model
    if args.theta_path is None:
        net = torch.load(args.model_path, weights_only=False)

        if args.theta_evidence == 'test':
            dataset = IDCDataset(
                df_test,n_user=n_user, n_item=n_item, Q_mat=Q_mat)
        elif args.theta_evidence == 'train':
            dataset = IDCDataset(
                df_train,n_user=n_user, n_item=n_item, Q_mat=Q_mat)
        # dataloader = DataLoader(
        #     dataset = dataset, batch_size = 1, shuffle = False)
        theta_list = []
        device = net.device
        
        for user_id in tqdm(range(n_user)):
            user_log = torch.Tensor([dataset.log_mat[user_id]])
            theta = net.diagnose_theta(user_log.to(device))\
                .detach().cpu().numpy()
            theta_list.append(theta)
        
        theta_mat = np.concatenate(theta_list, axis=0)
    
    else: # Designed for IRT
        theta_mat = np.load(args.theta_path)
        user_know_hit = np.ones(shape=(n_user,1))
        Q_mat = np.ones(shape=(n_item,1))

    test_dataset = IDCDataset(
        df_test,n_user=n_user, n_item=n_item, Q_mat=Q_mat)
    test_user_know_hit = np.abs(test_dataset.log_mat) @ Q_mat

    
    doc = degree_of_consistency(\
        theta_mat = theta_mat, \
        user_know_hit = test_user_know_hit, \
        log_mat = test_dataset.log_mat, \
        Q_mat = Q_mat, \
        know_list = None)
    
    print(f'>>> Degree of Consistency = {doc:.3f}')

