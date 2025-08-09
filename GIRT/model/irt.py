import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind, shapiro
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.girt_dataset import IRTppDataset

class IRTNet(nn.Module):
    def __init__(self, n_respondent, n_item, 
                 theta_min = 1, theta_max = 5):
        super(IRTNet, self).__init__()
        
        self.theta_min = theta_min
        self.theta_max = theta_max

        print(f'a_min = {self.a_min}')
        print(f'a_max = {self.a_max}')
        print(f'h_lambda_min = {self.h_lambda_min}')
        print(f'h_lambda_max = {self.h_lambda_max}')

        self.n_respondent = n_respondent
        self.n_item = n_item
        self.a_raw = nn.Embedding(
            num_embeddings = n_item, embedding_dim = 1)
        self.b_raw = nn.Embedding(
            num_embeddings = n_item, embedding_dim = 1)
        self.theta_raw = nn.Embedding(
            num_embeddings = n_respondent, embedding_dim = 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.constant_(self.a_raw.weight, 0)
        nn.init.constant_(self.b_raw.weight, -0.1)
        nn.init.constant_(self.theta_raw.weight, 0.1)

    def get_a(self, item_id = None):
        if item_id is None:
            return self.sigmoid(self.a_raw.weight) * (self.a_max - self.a_min) + self.a_min
        return (self.sigmoid(self.a_raw(item_id)) * (self.a_max - self.a_min) + self.a_min).squeeze(-1)
    
    def get_b(self, item_id = None):
        if item_id is None:
            return self.sigmoid(self.b_raw.weight) * (self.theta_max - self.theta_min) + self.theta_min
        return (self.sigmoid(self.b_raw(item_id)) * (self.theta_max - self.theta_min) + self.theta_min).squeeze(-1)
    
    def get_theta(self, respondent_id = None):
        if respondent_id is None:
            return self.sigmoid(self.theta_raw.weight) * (self.theta_max - self.theta_min) + self.theta_min
        return (self.sigmoid(self.theta_raw(respondent_id)) * (self.theta_max - self.theta_min) + self.theta_min).squeeze(-1)

    def irt2pl(self, theta, item_a, item_b):
        return self.sigmoid(item_a * (theta - item_b))

    
    def forward(self, respondent_ids: torch.LongTensor, item_ids: torch.LongTensor):
        '''
        response_svs: torch.Tensor, shape = (batch_size, n_item), the score vector of respondents
        item_svs: torch.Tensor, shape = (batch_size, n_respondent), the score vector of items
        '''
        theta = self.get_theta(respondent_ids)
        item_a = self.get_a(item_ids)
        item_b = self.get_b(item_ids)
        y_pred = self.irt2pl(theta, item_a, item_b)

        return y_pred
    
class IRT2PL:
    def __init__(self):
        self.net = None
    
    def save(self, path:str = 'IRTpp_default_path.pth'):
        # 保存
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'hyperparameters': {
                'n_respondent': self.net.n_respondent, 
                'n_item': self.net.n_item,
                'theta_min': self.net.theta_min,
                'theta_max': self.net.theta_max,
                'a_min': self.net.a_min
            }
        }, path)

    def load(self, path:str):
        # 加载
        checkpoint = torch.load(path, map_location = torch.device('cpu'))
        n_respondent = checkpoint['hyperparameters']['n_respondent']
        n_item = checkpoint['hyperparameters']['n_item']
        theta_min = checkpoint['hyperparameters']['theta_min']
        theta_max = checkpoint['hyperparameters']['theta_max']
        theta_min = checkpoint['hyperparameters']['theta_min']
        theta_max = checkpoint['hyperparameters']['theta_max']
        a_min = checkpoint['hyperparameters']['a_min']
        h_lambda = checkpoint['hyperparameters']['h_lambda']

        self.net = IRTppNet(n_respondent,n_item,theta_min,theta_max,
                            theta_min,theta_max,
                            a_min,h_lambda)
        self.net.load_state_dict(checkpoint['model_state_dict'])

    def fit(self, score_matrix: np.array, learning_rate = 0.01, \
            n_epoch = 5, batch_size = 32,shuffle = True, \
            num_workers = 4, theta_min = 1, theta_max = 5, \
            theta_min = 2.5, theta_max = 3.5, a_min = 1, \
            h_lambda = None, device:str = 'cpu', \
            checkpoint_gap=1,
            checkpoint_dir="./checkpoint/"):
        
        if self.net is None:
            self.net = IRTppNet(score_matrix.shape[0], score_matrix.shape[1],
                                theta_min, theta_max, theta_min, theta_max,
                                a_min,h_lambda)
        device = torch.device(device)
        self.net = self.net.to(device)
        dataset = IRTppDataset(score_matrix)
        train_loader = DataLoader(dataset, 
                                batch_size = batch_size,
                                shuffle = shuffle,
                                num_workers = num_workers)
        optimizer = optim.Adam(self.net.parameters(), lr = learning_rate)
        loss_func = nn.BCELoss()
        # ---
        theta_epochs = []
        # ---
        for epoch in range(1, n_epoch+1):
            self.net.train()
            train_loss = 0
            for respondent_sv, item_sv, respondent_id, item_id, score in tqdm(train_loader):
                optimizer.zero_grad()
                respondent_sv = respondent_sv.to(device)
                item_sv = item_sv.to(device)
                respondent_id = respondent_id.to(device)
                item_id = item_id.to(device)
                score = score.to(device)
                pred_prob = self.net(respondent_sv, item_sv, respondent_id, item_id)
                # print(f'pred_prob = {pred_prob}')
                # print(f'score = {score}')
                loss = loss_func(pred_prob, score)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            print(f'Epoch {epoch}: train_loss = {train_loss:.6f}')
            theta = self.net.get_respondent_parameters(torch.Tensor(score_matrix).to(device)).squeeze(-1).detach().cpu().numpy()
            theta_epochs.append(theta)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if epoch%checkpoint_gap == 0:
                self.save(os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}.pt"))

        return np.array(theta_epochs)
        
def set_seed(seed):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # 设置Python hash种子
    os.environ['PYTHONHASHSEED'] = str(seed)
        

def test_dataset():
    score_matrix = np.array([
        [1,0,0,1,1],
        [1,-1,0,1,-1],
        [1,0,-1,1,0],
        [1,1,1,-1,1],
        [1,1,1,1,1]])
    dataset = IRTppDataset(score_matrix)
    print(f'len(dataset) = {len(dataset)}')
    dataloader = DataLoader(dataset, 
                            batch_size=3, 
                            shuffle=True,
                            num_workers=4)
    for respondent_sv, item_sv, respondent_id, item_id, score in dataloader:
        print(f'rid = {respondent_id}, rsv = {respondent_sv}')
        print(f'iid = {item_id}, isv = {item_sv}')
        print(f'score = {score}')

def test_forward():
    score_matrix = np.array([
        [1,0,0,1,1],
        [1,-1,0,1,-1],
        [1,0,-1,1,0],
        [1,1,1,-1,1],
        [1,1,1,1,1]])
    score_tensor = torch.Tensor(score_matrix)
    n_respondent = score_tensor.size(0)
    n_item = score_tensor.size(1)
    irtpp = IRTppNet(n_respondent = n_respondent, n_item = n_item)
    print(f'a = {irtpp.get_a()}')
    print(f'b = {irtpp.get_b()}')
    print(f'theta = {irtpp.get_theta()}')
    respondent_svs = score_tensor
    print(f'respondent_rvs = {respondent_svs}')
    print(f'theta = {irtpp.get_respondent_parameters(respondent_svs)}')
    item_svs = score_tensor.t()
    print(f'item_svs = {item_svs}')
    item_ids = torch.LongTensor([i for i in range(n_item)]).unsqueeze(1)
    print(f'item_ids = {item_ids}')
    item_a, item_b = irtpp.get_item_parameters(item_svs, item_ids)
    print(f'item_a = {item_a}')
    print(f'item_b = {item_b}')

    y_pred = irtpp(respondent_svs, item_svs, None, item_ids)
    print(f'y_pred = {y_pred}')

def draw_theta_epoch(theta_epochs:np.array, path = '../checkpoint/theta_epoch.png'):
    plt.figure(figsize = (16,8))
    for id in range(theta_epochs.shape[1]):
        plt.plot([i for i in range(theta_epochs.shape[0])], theta_epochs[:,id],label = f'R{id}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Theta')
    plt.tight_layout()
    plt.savefig(path)


def test_theta_epoch():
    exam_id = 0
    theta = np.load(f'../checkpoint/irtpp2pl_english{exam_id}_theta.npy')
    theta_sample = theta[:,[10,20,30,40,50,60,70,80,90,100]]
    draw_theta_epoch(theta_sample, f'../checkpoint/english{exam_id}_head_theta_epoch.png')
    n_stable = 0
    for id in range(theta.shape[1]):
        n_sample = theta.shape[0]
        theta_x = theta[-int(0.2*n_sample):-int(0.1*n_sample),id]
        theta_y = theta[-int(0.1*n_sample):,id]
        # theta_x = theta[:int(0.1*n_sample),id]
        # theta_y = theta[int(0.1*n_sample):int(0.2*n_sample),id]
        splstats, splpvalue = shapiro(theta_x)
        sprstats, sprpvalue = shapiro(theta_y)
        tstats, pvalue = ttest_ind(theta_x, theta_y)
        stable = pvalue > 0.01
        n_stable += int(stable)
        print(f'RID {id}: splpvalue = {splpvalue:.4f}, sprpvalue = {sprpvalue:.4f}, t-stats = {tstats:.4f}, pvalue = {pvalue:.4f}, stable = {stable}')
    print(f'stable respondent: {n_stable/theta.shape[1]: .1%} ({n_stable}/{theta.shape[1]})')

def test_train():
    set_seed(3104)
    exam_id = 4
    score_matrix = np.load('../data/real/english_6times.npy')[exam_id]
    print(f'score_matrix.shape = {score_matrix.shape}')
    irtpp = IRTpp2PL()
    theta_epochs = irtpp.fit(score_matrix, device='cuda:2', n_epoch=100, learning_rate = 0.1, batch_size = 128)
    irtpp.save(f'../checkpoint/irtpp2pl_english{exam_id}.pth')
    irtpp.net = irtpp.net.to('cpu')
    print(f'theta = \n{irtpp.net.get_respondent_parameters(torch.Tensor(score_matrix))}')
    np.save(f'../checkpoint/irtpp2pl_english{exam_id}_theta.npy',theta_epochs)

def test_deploy():
    exam_id = 0
    score_matrix = np.load('../data/real/english_6times.npy')[exam_id]
    irtpp = IRTpp2PL()
    irtpp.load(f'../checkpoint/irtpp2pl_english{exam_id}.pth')
    score_vector = score_matrix[0]
    mask_left= np.ones_like(score_vector)
    mask_left[int(0.5*mask_left.shape[0]):] = 0
    score_vector_left = score_vector * mask_left
    mask_right = np.ones_like(score_vector)
    mask_right[:int(0.5*mask_left.shape[0])] = 0
    score_vector_right = score_vector * mask_right
    theta = irtpp.net.get_respondent_parameters(torch.Tensor([score_vector]))
    theta_left = irtpp.net.get_respondent_parameters(torch.Tensor([score_vector_left]))
    theta_right = irtpp.net.get_respondent_parameters(torch.Tensor([score_vector_right]))
    print(f'sv = {score_vector}, theta = {theta}')
    print(f'sv = {score_vector_left}, score = {np.sum(score_vector_left)}, theta = {theta_left}')
    print(f'sv = {score_vector_right}, score = {np.sum(score_vector_right)}, theta = {theta_right}')

if __name__ == "__main__":
    test_train()