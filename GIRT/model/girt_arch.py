import functools
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def array_to_triplets(arr):
    # Obtain 1-d index of non-zero elements
    nonzero_idx = arr.nonzero()
    # Obtain values
    values = arr[nonzero_idx]
    # Calculate row and column vectors
    rows = nonzero_idx[0]
    cols = nonzero_idx[1]
    return list(zip(rows, cols, values))

class GIRTDataset(Dataset):
    def __init__(self, score_matrix: np.array):
        '''Args
        score_matrix: np.array, shape = (n_respondent, n_item)
        '''
        self.score_matrix = score_matrix
        self.score_record = array_to_triplets(score_matrix)

    def __getitem__(self, idx):
        # Each row/column of self.score_matrix represents the response score vector of a learner/item. Record[0] is row index (learner id), while record[1] is column index (item id). 
        respondent_id, item_id, raw_score = self.score_record[idx]
        respondent_id = torch.LongTensor([respondent_id])
        item_id = torch.LongTensor([item_id])
        score = torch.Tensor([int(raw_score > 0)])
        respondent_sv = torch.Tensor(self.score_matrix[respondent_id])
        item_sv = torch.Tensor(self.score_matrix[:,item_id])
        return respondent_sv, item_sv, respondent_id, item_id, score

    def __len__(self):
        return len(self.score_record)

class GIRTDualDataset(Dataset):
    def __init__(self, score_matrix: np.array, score_matrix_dual:np.array):
        '''Args
        score_matrix: np.array, shape = (n_respondent, n_item)
        score_matrix_dual: np.array, shape = (n_respondent, n_item)
        For each data item, the current response score is from score_matrix, while the score vector is from score_matrix_dual. In this case, score_matrix is from evaluation dataset. score_matrix_dual is from training dataset
        This is for the evaluation of G-IRT.
        '''
        self.score_matrix = score_matrix
        self.score_record = array_to_triplets(score_matrix)
        self.score_matrix_dual = score_matrix_dual

    def __getitem__(self, idx):
        # Each row/column of self.score_matrix represents the response score vector of a learner/item. Record[0] is row index (learner id), while record[1] is column index (item id). 
        respondent_id, item_id, raw_score = self.score_record[idx]
        respondent_id = torch.LongTensor([respondent_id])
        item_id = torch.LongTensor([item_id])
        score = torch.Tensor([int(raw_score > 0)])
        respondent_sv = torch.Tensor(self.score_matrix_dual[respondent_id])
        item_sv = torch.Tensor(self.score_matrix_dual[:,item_id])
        return respondent_sv, item_sv, respondent_id, item_id, score

    def __len__(self):
        return len(self.score_record)

class GenItemResponseTheoryMeta(nn.Module):
    def __init__(self, n_respondent, n_item, 
                 theta_min = 1, theta_max = 5,
                 proxy_theta_min = 2.5, proxy_theta_max = 3.5,
                 proxy_a_min = 1, h_lambda = None):
        super(GenItemResponseTheoryMeta, self).__init__()
        assert(theta_min < proxy_theta_min 
               and proxy_theta_min < proxy_theta_max 
               and proxy_theta_max < theta_max
               and proxy_a_min > 0)
        
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.proxy_theta_min = proxy_theta_min
        self.proxy_theta_max = proxy_theta_max
        self.proxy_a_min = proxy_a_min
        self.proxy_a_max = proxy_a_min * min(
            2*(proxy_theta_min - theta_min)/(proxy_theta_max - proxy_theta_min),
            2*(theta_max - proxy_theta_max)/(proxy_theta_max - proxy_theta_min),)
        self.h_lambda_min = 0.5 * self.proxy_a_max * (proxy_theta_max - proxy_theta_min)
        self.h_lambda_max = min(self.proxy_a_min * (proxy_theta_min - theta_min),
                                self.proxy_a_min * (theta_max - proxy_theta_max))

        print(f'proxy_a_min = {self.proxy_a_min}')
        print(f'proxy_a_max = {self.proxy_a_max}')
        print(f'h_lambda_min = {self.h_lambda_min}')
        print(f'h_lambda_max = {self.h_lambda_max}')

        if h_lambda is None:
            self.h_lambda = 0.5 * (self.h_lambda_max + self.h_lambda_min)
        else:
            assert(h_lambda >= self.h_lambda_min
                   and h_lambda <= self.h_lambda_max)
            self.h_lambda = h_lambda

        self.n_respondent = n_respondent
        self.n_item = n_item
        self.proxy_a_raw = nn.Embedding(
            num_embeddings = n_item, embedding_dim = 1)
        self.proxy_b_raw = nn.Embedding(
            num_embeddings = n_item, embedding_dim = 1)
        self.proxy_theta_raw = nn.Embedding(
            num_embeddings = n_respondent, embedding_dim = 1)
        self.sigmoid = nn.Sigmoid()

        # nn.init.constant_(self.proxy_a_raw.weight, 0.0)
        # nn.init.constant_(self.proxy_b_raw.weight, -0.1)
        # nn.init.constant_(self.proxy_theta_raw.weight, 0.1)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
                print(f'>>> {name} initialized!')

    def get_proxy_a(self, item_id = None):
        if item_id is None:
            return self.sigmoid(self.proxy_a_raw.weight) \
                * (self.proxy_a_max - self.proxy_a_min) + self.proxy_a_min
        return (self.sigmoid(self.proxy_a_raw(item_id)) \
                    * (self.proxy_a_max - self.proxy_a_min) + self.proxy_a_min).squeeze(-1)
    
    def get_proxy_b(self, item_id = None):
        if item_id is None:
            return self.sigmoid(self.proxy_b_raw.weight) \
                * (self.proxy_theta_max - self.proxy_theta_min) + self.proxy_theta_min
        return (self.sigmoid(self.proxy_b_raw(item_id)) \
                * (self.proxy_theta_max - self.proxy_theta_min) + self.proxy_theta_min).squeeze(-1)
    
    def get_proxy_theta(self, respondent_id = None):
        if respondent_id is None:
            return self.sigmoid(self.proxy_theta_raw.weight) * (self.proxy_theta_max - self.proxy_theta_min) + self.proxy_theta_min
        return (self.sigmoid(self.proxy_theta_raw(respondent_id)) * (self.proxy_theta_max - self.proxy_theta_min) + self.proxy_theta_min).squeeze(-1)
    
    def get_respondent_parameters(self, respondent_svs: torch.Tensor):
        respondent_ovs = torch.square(respondent_svs)
        
        theta = (torch.matmul(respondent_ovs,self.get_proxy_b()) 
                 + self.h_lambda * torch.sum(respondent_svs / self.get_proxy_a().t(), axis = 1).unsqueeze(1))\
                    / torch.sum(respondent_ovs, axis = 1).unsqueeze(1)
        return theta
    
    def get_item_parameters(self, item_svs: torch.Tensor, item_ids: torch.LongTensor):
        item_ovs = torch.square(item_svs)
        
        proxy_theta = self.get_proxy_theta().t().repeat(item_ids.size(0),1)
        proxy_b = self.get_proxy_b(item_ids)
        frac = torch.clamp(torch.abs(proxy_theta - proxy_b), min = 1.0)
        item_a = torch.abs(self.h_lambda * item_svs) / frac
        
        frac = torch.clamp(torch.sum(item_ovs, axis = 1), min = 1.0)# Debug
        item_a = torch.sum(item_a, axis = 1) / frac #torch.sum(item_ovs, axis = 1)
        item_a = item_a.unsqueeze(1)

        item_b = (torch.matmul(item_ovs, self.get_proxy_theta()) \
                  - self.h_lambda * torch.sum(item_svs / self.get_proxy_a(item_ids), axis = 1).unsqueeze(1))
        item_b /= frac.unsqueeze(1)
        return item_a, item_b

    def irt2pl(self, theta, item_a, item_b):
        return self.sigmoid(item_a * (theta - item_b))

    
    def forward(self, respondent_svs: torch.Tensor, item_svs: torch.Tensor,
                respondent_ids: torch.LongTensor, item_ids: torch.LongTensor):
        '''
        response_svs: torch.Tensor, shape = (batch_size, n_item), the score vector of respondents
        item_svs: torch.Tensor, shape = (batch_size, n_respondent), the score vector of items
        '''
        theta = self.get_respondent_parameters(respondent_svs)
        item_a, item_b = self.get_item_parameters(item_svs, item_ids)
        # print(f'>>> theta = {theta}')
        # print(f'>>> item_a = {item_a}')
        # print(f'>>> item_b = {item_b}')
        y_pred = self.irt2pl(theta, item_a, item_b)

        return y_pred

class GenItemResponseTheoryModel2PL:
    def __init__(self):
        self.model = None
    
    def save(self, path:str = 'IRTpp_default_path.pth'):
        # 保存
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': {
                'n_respondent': self.model.n_respondent,  # int/float超参数
                'n_item': self.model.n_item,
                'theta_min': self.model.theta_min,
                'theta_max': self.model.theta_max,
                'proxy_theta_min': self.model.proxy_theta_min,
                'proxy_theta_max': self.model.proxy_theta_max,
                'proxy_a_min': self.model.proxy_a_min,
                'h_lambda': self.model.h_lambda
            }
        }, path)

    def load(self, path:str):
        # 加载
        checkpoint = torch.load(path, map_location = torch.device('cpu'))
        n_respondent = checkpoint['hyperparameters']['n_respondent']
        n_item = checkpoint['hyperparameters']['n_item']
        theta_min = checkpoint['hyperparameters']['theta_min']
        theta_max = checkpoint['hyperparameters']['theta_max']
        proxy_theta_min = checkpoint['hyperparameters']['proxy_theta_min']
        proxy_theta_max = checkpoint['hyperparameters']['proxy_theta_max']
        proxy_a_min = checkpoint['hyperparameters']['proxy_a_min']
        h_lambda = checkpoint['hyperparameters']['h_lambda']

        self.model = GenItemResponseTheoryMeta(n_respondent,n_item,theta_min,theta_max,
                            proxy_theta_min,proxy_theta_max,
                            proxy_a_min,h_lambda)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def fit(self, score_matrix: np.array, valid_score_matrix:np.array, 
            learning_rate = 0.01, n_epoch = 5, batch_size = 32,
            shuffle = True, num_workers = 4, theta_min = 1, theta_max = 5,
            proxy_theta_min = 2.5, proxy_theta_max = 3.5,
            proxy_a_min = 1, h_lambda = None, device:str = 'cpu', \
            checkpoint_gap=1,
            checkpoint_dir="./checkpoint/"):
        
        if self.model is None:
            self.model = GenItemResponseTheoryMeta(score_matrix.shape[0], score_matrix.shape[1],
                                theta_min, theta_max, proxy_theta_min, proxy_theta_max,
                                proxy_a_min,h_lambda)
        device = torch.device(device)
        self.model = self.model.to(device)
        dataset = GIRTDataset(score_matrix)
        train_loader = DataLoader(dataset, 
                                batch_size = batch_size,
                                shuffle = shuffle,
                                num_workers = num_workers)
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        loss_func = nn.BCELoss()
        # ---
        theta_epochs = []
        # ---
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.save(os.path.join(checkpoint_dir, "checkpoint-epoch-0.pt"))
        for epoch in range(1, n_epoch+1):
            self.model.train()
            train_loss = 0
            for respondent_sv, item_sv, respondent_id, item_id, score \
                in tqdm(train_loader,desc=f"Epoch: {epoch}"):
                # try:
                optimizer.zero_grad()
                respondent_sv = respondent_sv.to(device)
                item_sv = item_sv.to(device)
                respondent_id = respondent_id.to(device)
                item_id = item_id.to(device)
                score = score.to(device)
                pred_prob = self.model(respondent_sv, item_sv, respondent_id, item_id)
                loss = loss_func(pred_prob, score)
                loss.backward()
                optimizer.step()
                    
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            print(f'Epoch {epoch}: train_loss = {train_loss:.6f}')
            theta = self.model.get_respondent_parameters(torch.Tensor(score_matrix).to(device)).squeeze(-1).detach().cpu().numpy()
            theta_epochs.append(theta)
            try:
                _, _, y_true, y_pred = self.eval(
                    train_sm = score_matrix, eval_sm = valid_score_matrix, device = device)
                y_plabel = (y_pred > 0.5).astype(int)
                acc = accuracy_score(y_true, y_plabel)
                print(f">>> Prediction ACC = {acc: .4f}")
            except:
                continue
            if epoch%checkpoint_gap == 0:
                self.save(os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}.pt"))

        return np.array(theta_epochs)

    def diagnose_learner(self, score_vector:np.array, device:str='cpu'):
        '''
        Diagnose learnr ability for a new learner's score vector.
        Args:
            score_vector:np.array, shape = (n_item, 0)
        '''
        self.model = self.model.to(device)
        if score_vector.ndim == 1 and score_vector.shape[0] == self.model.n_item:
            score_tensor = torch.Tensor(score_vector).unsqueeze(0).to(device)
        elif score_vector.ndim == 2 and score_vector.shape[1] == self.model.n_item:
            score_tensor = torch.Tensor(score_vector).to(device)
        self.model = self.model.to(device)
        theta = self.model.get_respondent_parameters(score_tensor)
        theta = theta.detach().cpu().numpy()
        return theta
        
    def eval(self, train_sm: np.array, eval_sm: np.array, device:str):
        dataset = GIRTDualDataset(
            score_matrix = eval_sm, \
            score_matrix_dual = train_sm)
        assert(int(np.sum(eval_sm != 0)) == len(dataset))
        device = torch.device(device)
        eval_loader = DataLoader(
            dataset, 
            batch_size = 32,
            shuffle = False,
            num_workers = 1)
        all_respondent_id = []
        all_item_id = []
        all_ground_truth = []
        all_pred_prob = []
        self.model = self.model.to(device)
        self.model.eval()
        for respondent_sv, item_sv, respondent_id, item_id, score in tqdm(eval_loader):
            respondent_sv = respondent_sv.to(device)
            item_sv = item_sv.to(device)
            respondent_id = respondent_id.to(device)
            item_id = item_id.to(device)
            score = score.to(device)
            output = self.model(respondent_sv, item_sv, respondent_id, item_id)

            respondent_id = respondent_id.cpu().detach().numpy().reshape(-1)
            item_id = item_id.cpu().detach().numpy().reshape(-1)
            ground_truth = score.cpu().detach().numpy().reshape(-1)
            pred_prob = output.cpu().detach().numpy().reshape(-1)
            
            all_respondent_id.append(respondent_id)
            all_item_id.append(item_id)
            all_ground_truth.append(ground_truth)
            all_pred_prob.append(pred_prob)
        
        all_respondent_id = np.concatenate(all_respondent_id)
        all_item_id = np.concatenate(all_item_id)
        all_ground_truth = np.concatenate(all_ground_truth)
        all_pred_prob = np.concatenate(all_pred_prob)
        return all_respondent_id, all_item_id, all_ground_truth, all_pred_prob

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
    dataset = GIRTDataset(score_matrix)
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
    irtpp = GenItemResponseTheoryMeta(n_respondent = n_respondent, n_item = n_item)
    print(f'proxy_a = {irtpp.get_proxy_a()}')
    print(f'proxy_b = {irtpp.get_proxy_b()}')
    print(f'proxy_theta = {irtpp.get_proxy_theta()}')
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
