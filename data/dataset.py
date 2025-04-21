import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List
import os

class CFDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: 전처리된 CSV 파일들이 있는 디렉토리 경로
            split: "train" 또는 "test"
        """
        assert split in ["train", "test"], "split must be either 'train' or 'test'"
        
        # Load data
        self.interactions = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
        self.users = pd.read_csv(os.path.join(data_dir, "users.csv"))
        self.problems = pd.read_csv(os.path.join(data_dir, "problems.csv"))
        
        # Calculate max indices
        self.num_users = self.interactions['user_id'].max()
        self.num_items = self.interactions['problem_id'].max()
        
        # Convert to tensors for faster access
        self.user_indices = torch.LongTensor(self.interactions['user_id'].values)
        self.item_indices = torch.LongTensor(self.interactions['problem_id'].values)
        
        # Create mappings for inference
        self._handle_to_id_map = dict(zip(self.users['handle'], self.users['id']))
        self._id_to_problem_id_map = dict(zip(self.problems['id'], self.problems['problem_id']))
        
        # Create reverse mappings for inference
        self._id_to_handle_map = {v: k for k, v in self._handle_to_id_map.items()}
        
        # Store all user interactions for NDCG calculation
        self._user_interactions = {}
        for user_id, group in self.interactions.groupby('user_id'):
            self._user_interactions[user_id] = set(group['problem_id'].tolist())
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx]
    
    def get_user_id_from_handle(self, handle: str) -> int:
        """사용자 handle을 id로 변환"""
        return self._handle_to_id_map.get(handle)  # Unknown handle이면 None 반환
    
    def get_problem_id(self, idx: int) -> int:
        """problem의 id를 실제 problem_id로 변환"""
        return self._id_to_problem_id_map.get(idx)  # Unknown id면 None 반환
    
    def get_handle_from_id(self, user_id: int) -> str:
        """user_id로부터 handle을 가져옵니다"""
        return self._id_to_handle_map.get(user_id)
    
    def get_user_interactions(self, user_id: int) -> List[int]:
        """특정 사용자가 푼 문제 목록을 가져옵니다"""
        return list(self._user_interactions.get(user_id, set()))