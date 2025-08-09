import argparse
import numpy as np
import os
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model')
sys.path.insert(0, model_path)

from model.girt_arch import GIRTDataset, GenItemResponseTheoryModel2PL

def generative_diagnosis(model_file:str, score_vector:np.array, device:str):
    model = GenItemResponseTheoryModel2PL()
    model.load(model_file)
    start_time = time.time()
    theta = model.diagnose_learner(score_vector, device=device)
    end_time = time.time()
    print(f'>>> Generative diagnostic theta = \n>>> {theta}')
    print(f'>>> Cost time of diagnosis is {end_time-start_time: .4f} s.')
    return theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file")
    parser.add_argument("-s", "--score_matrix_file")
    parser.add_argument("-o", "--output_path", default = "./diagnosis/default/")
    parser.add_argument("-d", "--device")
    args = parser.parse_args()
    score_matrix = np.loadtxt(args.score_matrix_file)
    theta = generative_diagnosis(args.model_file, score_matrix, args.device)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    np.save(os.path.join(args.output_path, 'theta.npy'), theta)