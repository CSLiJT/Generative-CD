import argparse
import json
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model')
sys.path.insert(0, model_path)

from model.girt_arch import GenItemResponseTheoryModel2PL, set_seed

class TrainingConfig:
    def __init__(self, input_dir:str):
        self.device = None
        self.n_epoch = None
        self.lr = None
        self.batch_size = None
        self.shuffle = None
        self.num_workers = None
        self.load_from_json(input_dir)

    def load_from_json(self, input_dir:str):
        fp = open(input_dir, 'r')
        json_data = json.load(fp)
        for attr in vars(self).keys():
            if attr in json_data:
                setattr(self, attr, json_data[attr])
        fp.close()

    def save_as_json(self, path:str):
        with open(path,'w') as fp:
            json.dump({
                'device': self.device,
                'n_epoch': self.n_epoch,
                'lr': self.lr,
                'batch_size': self.batch_size,
                'shuffle': self.shuffle,
                'num_workers': self.num_workers
            }, fp)

    def __str__(self):
        # Obtain all attributes
        attributes = vars(self)
        # Build a string
        attr_str = '\n'.join(f"{key}: {value}" for key, value in attributes.items())
        return attr_str

def train(data:np.array, valid_data:np.array, \
          training_config:TrainingConfig, checkpoint_gap:int, \
          output_dir:str)->GenItemResponseTheoryModel2PL:
    # Save training configs
    training_config.save_as_json(os.path.join(output_dir, "training_config.json"))
    
    # Train Model
    model = GenItemResponseTheoryModel2PL()
    theta_epochs = model.fit(
        score_matrix = data,
        valid_score_matrix = valid_data,
        learning_rate = training_config.lr,
        n_epoch = training_config.n_epoch,
        batch_size = training_config.batch_size,
        shuffle = training_config.shuffle,
        device = training_config.device,
        checkpoint_gap = checkpoint_gap,
        checkpoint_dir = output_dir
    )
    
    # Save results
    model.save(os.path.join(output_dir, "model_state.pt"))
    np.save(os.path.join(output_dir, "theta_epochs.npy"), theta_epochs)
    return model
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "G-IRT training program.")
    parser.add_argument("-t", "--training_config", help="Path of the training configuration json file.")
    parser.add_argument("-d", "--data_dir", help="Path of the training data.")
    parser.add_argument("-o", "--output_dir", help="Path of output.")
    parser.add_argument("-c", "--checkpoint_gap", help="Frequency of epochs to save model state.")
    args = parser.parse_args()
    training_config = TrainingConfig(args.training_config)
    print(f">>> TrainingConfig = {training_config}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f">>> output_dir = {args.output_dir}")
    training_data = np.loadtxt(os.path.join(args.data_dir, "train.txt"))
    valid_data = np.loadtxt(os.path.join(args.data_dir, "valid.txt"))
    model = train(training_data, valid_data, training_config, \
                  int(args.checkpoint_gap), args.output_dir)
    


