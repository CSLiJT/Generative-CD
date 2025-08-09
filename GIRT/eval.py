import argparse
import json
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model')
sys.path.insert(0, model_path)
sys.patn.insert(0, './')
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

from model.girt_arch import GIRTDataset, GenItemResponseTheoryModel2PL, set_seed

def evaluation(model_file:str, eval_data:np.array, train_data:np.array, device:str):

    model = GenItemResponseTheoryModel2PL()
    model.load(model_file)
    user_id, item_id, y_true, y_pred = model.eval(
        train_sm = train_data,
        eval_sm = eval_data,
        device = device
    )
    print(f'>>> y_pred = {y_pred}')
    y_plabel = (y_pred > 0.5).astype(int)
    nan_count = np.sum(np.isnan(y_pred))
    print(f">>> Data Count = {y_pred.shape[0]}")

    print(f">>> NaN Count for y_pred = {nan_count}")
    
    nan_count = np.sum(np.isnan(y_true))
    print(f">>> NaN Count for y_true = {nan_count}")

    acc = accuracy_score(y_true, y_plabel)
    print(f">>> ACC = {acc: .4f}")

    f1 = f1_score(y_true, y_plabel)
    print(f">>> F1 = {f1: .4f}")
    
    auc = roc_auc_score(y_true, y_pred)
    print(f">>> AUC = {auc: .4f}")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f">>> RMSE = {rmse: .4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file")
    parser.add_argument("-e", "--eval_data")
    parser.add_argument("-t", "--train_data")
    parser.add_argument("-d", "--device")
    args = parser.parse_args()
    train_sm = np.loadtxt(args.train_data)
    eval_sm = np.loadtxt(args.eval_data)
    evaluation(
        model_file = args.model_file,
        eval_data = eval_sm,
        train_data = train_sm,
        device = args.device
    )
