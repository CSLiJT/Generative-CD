import argparse
import numpy as np
import pandas as pd

from model import GNCDM, UAutoRec, CDAE
from train import train_AE, eval_AE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help = 'UAutoRec or CDAE')
    parser.add_argument('--train_file')
    parser.add_argument('--valid_file')
    parser.add_argument('--test_file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    train_df = pd.read_csv(args.train_file)
    valid_df = pd.read_csv(args.valid_file)
    test_df = pd.read_csv(args.test_file)
    all_df = pd.concat([train_df, test_df, valid_df], axis=0)
    n_user = np.max(all_df['user_id']) + 1
    n_item = np.max(all_df['item_id']) + 1
    model = eval(args.model)(n_user, n_item, hidden_dim=128, device=args.device)
    train_AE(model = model, train_data = train_df, valid_data = valid_df, \
             batch_size = args.batch_size, lr = args.lr, n_epoch = args.n_epoch)
    print('**** Evaluate Score Reconstruction (test -> test)****')
    results = eval_AE(model = model, train_data = test_df, test_data = test_df)
    print(f"Results = {results}")
    print('**** Evaluate Score Prediction (train -> test) ****')
    results = eval_AE(model = model, train_data = train_df, test_data = test_df)
    print(f"Results = {results}")
     
