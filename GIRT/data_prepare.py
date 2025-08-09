import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def fill_in_matrix_with_records(records:pd.DataFrame, matrix:np.array, uid_map:dict, iid_map:dict):
    for _, row in records.iterrows():
        score = 1 if row['score'] > 0.5 else -1
        x = uid_map[row['user_id']]
        y = iid_map[row['item_id']]
        matrix[x,y] = score
    return matrix

def transform_records_to_matrix(train_dir:str, valid_dir:str, test_dir:str, output_dir:str):
    '''
    Input file format: *.csv, columns include ["user_id", "item_id", "score"].
    Score is dichotomous. 1 -> 1; 0 -> -1
    Output: np.array(shape=(n_user, n_item))
    '''
    train_data = pd.read_csv(train_dir)
    valid_data = pd.read_csv(valid_dir)
    test_data = pd.read_csv(test_dir)
    all_data = pd.concat((train_data, valid_data, test_data), axis=0)
    user_ids = all_data['user_id'].unique()
    item_ids = all_data['item_id'].unique()
    print(f">>> n_user = {len(user_ids)}; n_item = {len(item_ids)}")
    user_id_map = {}
    for i, uid in enumerate(user_ids):
        user_id_map[uid] = i
    item_id_map = {}
    for i, iid in enumerate(item_ids):
        item_id_map[iid] = i
    train_matrix = np.zeros((len(user_ids),len(item_ids)))
    valid_matrix = np.zeros((len(user_ids),len(item_ids)))
    test_matrix = np.zeros((len(user_ids),len(item_ids)))
    train_matrix = fill_in_matrix_with_records(
        train_data, train_matrix, uid_map = user_id_map, iid_map = item_id_map)
    valid_matrix = fill_in_matrix_with_records(
        valid_data, valid_matrix, uid_map = user_id_map, iid_map = item_id_map)
    test_matrix = fill_in_matrix_with_records(
        test_data, test_matrix, uid_map = user_id_map, iid_map = item_id_map)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, "train.txt"), train_matrix, fmt="%d")
    np.savetxt(os.path.join(output_dir, "valid.txt"), valid_matrix, fmt="%d")
    np.savetxt(os.path.join(output_dir, "test.txt"), test_matrix, fmt="%d")
    with open(os.path.join(output_dir,'user_id_map.json'),'w') as fp:
        user_id_map = {int(k): v for k, v in user_id_map.items()}
        json.dump(user_id_map, fp)
    with open(os.path.join(output_dir,'item_id_map.json'),'w') as fp:
        item_id_map = {int(k): v for k, v in item_id_map.items()}
        json.dump(item_id_map, fp)
    return train_matrix, valid_matrix, test_matrix

def build_user_split_data(train_dir:str, valid_dir:str, test_dir:str, output_dir:str,\
    train_test_valid_ratio:tuple = (0.7,0.1,0.2)):
    '''
    Input file format: *.csv, columns include ["user_id", "item_id", "score"].
    All input files are concatenated together. Then re-split according to user ID.
    The user ID of rebuilt training file, test file and valid file is not overlapped.
    '''
    train_data = pd.read_csv(train_dir)
    valid_data = pd.read_csv(valid_dir)
    test_data = pd.read_csv(test_dir)
    all_data = pd.concat((train_data, valid_data, test_data), axis=0)
    rebuilt_train = pd.DataFrame()
    rebuilt_test = pd.DataFrame()
    rebuilt_valid = pd.DataFrame()
    group_user = all_data.groupby('user_id')
    
    # Get all unique user IDs
    unique_users = all_data['user_id'].unique()
    
    # Shuffle users for random splitting
    np.random.shuffle(unique_users)
    
    # Calculate split indices based on the provided ratio
    n_users = len(unique_users)
    train_end = int(n_users * train_test_valid_ratio[0])
    valid_end = train_end + int(n_users * train_test_valid_ratio[1])
    
    # Split users into train, valid, and test sets
    train_users = unique_users[:train_end]
    valid_users = unique_users[train_end:valid_end]
    test_users = unique_users[valid_end:]
    
    # Build the datasets based on user assignments
    for user_id, group in tqdm(group_user):
        if user_id in train_users:
            rebuilt_train = pd.concat([rebuilt_train, group], axis=0)
        elif user_id in valid_users:
            rebuilt_valid = pd.concat([rebuilt_valid, group], axis=0)
        else:  # user_id in test_users
            rebuilt_test = pd.concat([rebuilt_test, group], axis=0)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the rebuilt datasets
    rebuilt_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    rebuilt_valid.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    rebuilt_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Train set: {len(rebuilt_train)} rows, {len(train_users)} users")
    print(f"Valid set: {len(rebuilt_valid)} rows, {len(valid_users)} users")
    print(f"Test set: {len(rebuilt_test)} rows, {len(test_users)} users")
    
    return rebuilt_train, rebuilt_valid, rebuilt_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Data preparation")
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-t', '--task', default='transform', help='\'transform\' or \'rebuild\'')
    args = parser.parse_args()
    print(f"Prepare.task = {args.task}")
    if args.task == 'transform':
        transform_records_to_matrix(
            train_dir=os.path.join(args.input_dir,"train.csv"),
            valid_dir=os.path.join(args.input_dir,"valid.csv"),
            test_dir=os.path.join(args.input_dir,"test.csv"),
            output_dir=args.output_dir
        )
    else:
        build_user_split_data(
            train_dir=os.path.join(args.input_dir,"train.csv"),
            valid_dir=os.path.join(args.input_dir,"valid.csv"),
            test_dir=os.path.join(args.input_dir,"test.csv"),
            output_dir=args.output_dir     
        )

