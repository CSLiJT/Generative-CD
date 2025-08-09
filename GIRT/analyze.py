import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model')
sys.path.insert(0, model_path)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

from model.girt_arch import GIRTDataset, GenItemResponseTheoryModel2PL, set_seed

def draw_scatters(data:np.array, color_value, x_label:str, y_label:str, path:str):
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(data[:,0], data[:,1], c = color_value, cmap = "RdBu")
    plt.colorbar(scatter, label="Score Rate")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path, dpi=800)

def analyze_proxy(model_file:str, evidence_data:np.array, device:str, save_path):

    model = GenItemResponseTheoryModel2PL()
    model.load(model_file)
    dataset = GIRTDataset(evidence_data)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 4)
    user_trait = {}
    user_score = {}
    # user_proxy = {}
    item_a = {}
    item_b = {}
    item_score = {}
    # item_proxy = {}
    model.model = model.model.to(device)
    model.model.eval()
    count = 0
    for user_sv, item_sv, user_id, item_id, score in tqdm(dataloader, desc=f"Loading Latent Traits"):
        # count += 1
        # if count == 100:
        #     break
        uid = user_id.numpy().reshape(-1,)[0]
        if user_trait.get(uid, None) is None:
            user_trait[uid] = [0,0]
            user_trait[uid][0] = model.model.get_respondent_parameters(user_sv.to(device)).detach().cpu().numpy().reshape(-1,)[0]
            user_trait[uid][1] = model.model.get_proxy_theta(user_id.to(device)).detach().cpu().numpy().reshape(-1,)[0]
            raw_sv = user_sv.numpy()
            ob_sv = raw_sv[raw_sv != 0]
            user_score[uid] = (np.mean(ob_sv)+1)/2

        iid = item_id.cpu().numpy().reshape(-1,)[0]
        if item_a.get(iid, None) is None:
            item_a[iid] = [0,0]
            item_b[iid] = [0,0]
            a, b = model.model.get_item_parameters(item_sv.to(device),item_id.to(device))
            item_a[iid][0] = a.detach().cpu().numpy().reshape(-1,)[0]
            item_b[iid][0] = b.detach().cpu().numpy().reshape(-1,)[0]
            item_a[iid][1] = model.model.get_proxy_a(item_id.to(device)).detach().cpu().numpy().reshape(-1,)[0]
            item_b[iid][1] = model.model.get_proxy_b(item_id.to(device)).detach().cpu().numpy().reshape(-1,)[0]
            raw_sv = item_sv.numpy()
            ob_sv = raw_sv[raw_sv != 0]
            item_score[iid] = (np.mean(ob_sv)+1)/2
    theta_dual = np.array([elem for _, elem in user_trait.items()])
    a_dual = np.array([elem for _, elem in item_a.items()])
    b_dual = np.array([elem for _, elem in item_b.items()])
    user_score = np.array([elem for _, elem in user_score.items()])
    item_score = np.array([elem for _, elem in item_score.items()])

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "theta_dual.npy"), theta_dual)
    np.save(os.path.join(save_path, "a_dual.npy"), a_dual)
    np.save(os.path.join(save_path, "b_dual.npy"), b_dual)
    np.save(os.path.join(save_path, "user_score_rate.npy"), user_score)
    np.save(os.path.join(save_path, "item_score_rate.npy"), item_score)

    draw_scatters(theta_dual, color_value = user_score, x_label = "Generative Diagnosis", y_label="Proxy Parameter", path=os.path.join(save_path, "theta_dual.png"))
    draw_scatters(a_dual, color_value = item_score, x_label = "Generative Diagnosis", y_label="Proxy Parameter", path=os.path.join(save_path, "a_dual.png"))
    draw_scatters(b_dual, color_value = item_score, x_label = "Generative Diagnosis", y_label="Proxy Parameter", path=os.path.join(save_path,"b_dual.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file")
    parser.add_argument("-e", "--evidence_data")
    parser.add_argument("-d", "--device")
    parser.add_argument("-s", "--save_path")
    args = parser.parse_args()
    evidence_sm = np.loadtxt(args.evidence_data)
    analyze_proxy(args.model_file,evidence_sm,args.device,args.save_path)
