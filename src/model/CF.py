import torch
import torch.nn as nn
from typing import List
from data.dataset import CFDataset

class CF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        """
        Args:
            num_users: 최대 user_id
            num_items: 최대 problem_id
            embedding_dim: 임베딩 차원
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)  # +1 for 1-based indexing
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)  # +1 for 1-based indexing
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_indices: (batch_size,) - user의 id
            item_indices: (batch_size,) - problem의 id
        Returns:
            predictions: (batch_size,)
        """
        user_embeds = self.user_embedding(user_indices)  # (batch_size, embedding_dim)
        item_embeds = self.item_embedding(item_indices)  # (batch_size, embedding_dim)
        
        # Compute dot product
        predictions = (user_embeds * item_embeds).sum(dim=1)  # (batch_size,)
        return torch.sigmoid(predictions)
    
    @torch.no_grad()
    def recommend(self, dataset: CFDataset, handle: str, top_k: int = 10, device: str = 'cpu') -> List[int]:
        """
        사용자 handle을 받아서 추천할 문제의 problem_id 목록을 반환합니다.
        
        Args:
            dataset: CFDataset 인스턴스
            handle: 사용자의 백준 handle
            top_k: 추천할 문제 개수
            device: 모델이 있는 device
            
        Returns:
            추천된 문제들의 problem_id 리스트
        """
        # handle을 id로 변환
        user_id = dataset.get_user_id_from_handle(handle)
        if user_id is None:  # Unknown user
            return []
            
        # 사용자 임베딩 가져오기 (device에 맞춰서)
        user_embedding = self.user_embedding(torch.tensor([user_id], device=device))
        
        # 모든 문제와의 유사도 계산 (device에 맞춰서)
        all_item_embeddings = self.item_embedding.weight.to(device)
        scores = torch.matmul(user_embedding, all_item_embeddings.t()).squeeze()
        
        # Top-K 문제의 인덱스 가져오기
        top_k_indices = torch.topk(scores, k=top_k).indices.tolist()
        
        # 인덱스를 실제 problem_id로 변환
        recommended_problems = []
        for idx in top_k_indices:
            problem_id = dataset.get_problem_id(idx)
            if problem_id is not None:
                recommended_problems.append(problem_id)
        
        return recommended_problems