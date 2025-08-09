# -*- coding: utf-8 -*-
# Copyright (c) 2025 Jiatong Li
# All rights reserved.
# 
# This software is the confidential and proprietary information
# of Jiatong Li. You shall not disclose such confidential
# information and shall use it only in accordance with the terms of
# the license agreement.


import os
import numpy as np
import time
import torch
from datetime import datetime
from tqdm import tqdm

class Logger:
    '''
    logger suitable for anywhere
    '''
    def __init__(self, path = './log/', mode='both'):
        self.fmt = "%Y-%m-%d-%H:%M:%S"
        self.begin_time = time.strftime(self.fmt,time.localtime())
        self.path = os.path.join(path,self.begin_time +'/')

    
    def write(self, message: str, mode = 'both'):
        '''
        @Param mode: 
            file(default): print to file
            console: print to screen
            both: print to both file and screen
        '''
        current_time = time.strftime(self.fmt,time.localtime())
        begin = datetime.strptime(self.begin_time,self.fmt)
        end = datetime.strptime(current_time, self.fmt)
        minutes = (end - begin).seconds
        record = '{} ({} s used) {}\n'.format(current_time , minutes, message)

        if mode == 'file' or mode == 'both':
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        if mode == 'file':
            with open(self.path+'log.txt','a') as f:
                f.write(record)

        elif mode == 'console':
            print(record, end='')
        
        elif mode == 'both':
            with open(self.path+'log.txt','a') as f:
                f.write(record)
            print(record, end='')
        
        else:
            print('Logger error! [mode] must be \'file\' or \'console\' or \'both\'.')

def labelize(y_pred: torch.DoubleTensor, threshold = 0.5)->np.ndarray:
    return (y_pred > threshold).to('cpu').detach().numpy().astype(np.int).reshape(-1,)

def to_numpy(y_pred: torch.DoubleTensor)->np.ndarray:
    return y_pred.to('cpu').detach().numpy().reshape(-1,)

def degree_of_consistency(theta_mat: np.array, user_know_hit: np.array, \
    log_mat: np.array, Q_mat: np.array, know_list = None):
    '''
    theta_mat: (n_user, n_know): the diagnostic result matrix
    user_know_hit: (n_user, n_know): the (i,j) element indicate \
        the number of hits of the i-th user on the j-th attribute
    log_mat: (n_user, n_exer): the matrix indicating whether the \
        student has correctly answered the exercise (+1) or not(-1) 
    Q_mat: (n_exer, n_know)
    '''
    n_user, n_know = theta_mat.shape 
    n_exer = log_mat.shape[1]
    doc_all = []
    item_list = list(range(n_exer)) 
    for item_id in item_list:
        know_list = np.where(Q_mat[item_id,:] > 0)[0]
        user_list = np.where(log_mat[:,item_id] != 0)[0]
        # import pdb
        # pdb.set_trace()
        n_u_k = len(user_list)
        pbar = tqdm(total = n_u_k * (n_u_k - 1), desc='item_id = %d'%item_id)
        doc_frac1 = 0
        doc_frac2 = 0
        for a in user_list:
            for b in user_list:
                # if m_ak != m_bk, then either m_ak > m_bk or m_bk > m_ak
                if a == b:
                    continue
                delta_r = int(log_mat[a, item_id] > log_mat[b, item_id])
                delta_t = 1e-9
                ndiff_t = 1e-9
                for know_id in know_list:
                    delta_t += int(theta_mat[a, know_id] > theta_mat[b, know_id])
                    ndiff_t += int(theta_mat[a, know_id] != theta_mat[b, know_id])
                pbar.update(1)
                doc_frac1 += delta_r * delta_t
                doc_frac2 += delta_r * ndiff_t
        pbar.close()
        if doc_frac2 == 0:
            continue
        doc = doc_frac1/doc_frac2
        doc_all.append(doc)
        print(f">>> DOC(E{item_id}) = {doc:.3f}")
        # import pdb
        # pdb.set_trace()
    return np.mean(doc_all)

# def degree_of_consistency(theta_mat: np.array, user_know_hit: np.array, \
#     log_mat: np.array, Q_mat: np.array, know_list = None):
#     '''
#     theta_mat: (n_user, n_know): the diagnostic result matrix
#     user_know_hit: (n_user, n_know): the (i,j) element indicate \
#         the number of hits of the i-th user on the j-th attribute
#     log_mat: (n_user, n_exer): the matrix indicating whether the \
#         student has correctly answered the exercise (+1) or not(-1) 
#     Q_mat: (n_exer, n_know)
#     '''
#     n_user, n_know = theta_mat.shape 
#     n_exer = log_mat.shape[1]
#     doc_all = []
#     know_list = list(range(n_know)) if know_list is None else know_list
#     for know_id in know_list:
#         Z = 1e-9
#         dm = 0
#         exer_list = np.where(Q_mat[:,know_id] > 0)[0]
#         user_list = np.where(user_know_hit[:,know_id]>0)[0]
#         n_u_k = len(user_list)
#         pbar = tqdm(total = n_u_k * (n_u_k - 1), desc='know_id = %d'%know_id)
#         for a in user_list:
#             for b in user_list:
#                 # if m_ak != m_bk, then either m_ak > m_bk or m_bk > m_ak
#                 if a == b:
#                     continue
#                 Z += (theta_mat[a, know_id] > theta_mat[b, know_id])
#                 nab = 1e-9
#                 dab = 1e-9
#                 for exer_id in exer_list:
#                     Jab = (log_mat[a,exer_id] * log_mat[b,exer_id] != 0)
#                     nab += Jab * (log_mat[a, exer_id] > log_mat[b, exer_id])
#                     dab += Jab * (log_mat[a, exer_id] != log_mat[b, exer_id])
#                 dm += (theta_mat[a, know_id] > theta_mat[b, know_id]) * nab / dab 
#                 pbar.update(1)

#         doa = dm / Z 
#         doc_all.append(doa)
#         import pdb
#         pdb.set_trace()
#         print(f">>> DOC(A{know_id}) = {doc_all:.3f}")
#     return doc_all
                