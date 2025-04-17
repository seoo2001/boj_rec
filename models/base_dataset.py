from torch.utils.data import Dataset
import numpy as np
import torch

class BaseDataset(Dataset):
    def __init__(self, data, transform=None, user_id_map=None, problem_id_map=None):
        self.data = data
        self.transform = transform
        
        if user_id_map is not None and problem_id_map is not None:
            self.user_id_map = user_id_map
            self.problem_id_map = problem_id_map
        else:
            self.user_id_map = {user: idx for idx, user in enumerate(self.user_id)}
            self.problem_id_map = {problem: idx for idx, problem in enumerate(self.problem_id)}
        
        self.users = torch.tensor(self.data['user_id'].map(self.user_id_map).values, dtype=torch.long)
        self.problems = torch.tensor(self.data['problem_id'].map(self.problem_id_map).values, dtype=torch.long)
        self.ratings = torch.ones(len(self.data), dtype=torch.float32)
    
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        problem = self.problems[idx]
        rating = self.ratings[idx]
        
        sample = {'user_id': user, 'problem_id': problem, 'rating': rating}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def to(self, device):
        self.users = self.users.to(device)
        self.problems = self.problems.to(device)
        self.ratings = self.ratings.to(device)
        return self