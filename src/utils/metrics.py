import torch
from typing import List, Dict
import math
from collections import Counter

def calculate_ndcg_tensor(predicted: torch.Tensor, actual: torch.Tensor, k: int) -> float:
    """
    Calculate NDCG@K metric using tensor operations
    
    Args:
        predicted: (batch_size, k) - 추천된 문제 ID 텐서
        actual: (batch_size, n) - 실제 푼 문제 ID 텐서 (n은 가변적)
        k: 상위 K개 항목까지 평가
    
    Returns:
        NDCG@K 점수
    """
    if len(actual) == 0:
        return 0.0
    
    # (batch_size, k, n)
    matches = predicted.unsqueeze(-1) == actual.unsqueeze(1)
    
    # Calculate DCG - shape: (batch_size, k)
    log2_indices = torch.log2(torch.arange(2, k + 2, dtype=torch.float))
    dcg = (matches.any(-1).float() / log2_indices).sum(-1)
    
    # Calculate IDCG - shape: (batch_size,)
    n_relevant = torch.minimum(actual.shape[1] * torch.ones_like(dcg), torch.tensor([k], dtype=torch.float))
    idcg = (1 / log2_indices[:n_relevant.long()]).sum()
    
    # Average over batch
    return (dcg / idcg).mean().item() if idcg > 0 else 0.0

def calculate_entropy_tensor(recommendations: torch.Tensor) -> float:
    """
    Calculate entropy of recommendations using tensor operations
    
    Args:
        recommendations: (num_users, k) - 추천된 문제 ID 텐서
    
    Returns:
        추천의 다양성을 나타내는 entropy 점수
    """
    if len(recommendations) == 0:
        return 0.0
    
    # Flatten and get unique counts
    flat_recs = recommendations.reshape(-1)
    unique_items, counts = torch.unique(flat_recs, return_counts=True)
    
    if len(unique_items) == 0:
        return 0.0
    
    # Calculate probabilities
    probs = counts.float() / counts.sum()
    
    # Calculate entropy
    entropy = -(probs * torch.log2(probs)).sum()
    
    # Normalize by maximum possible entropy
    max_entropy = math.log2(len(unique_items))
    return (entropy / max_entropy).item() if max_entropy > 0 else 0.0

# Keep the original functions for non-tensor inputs
def calculate_ndcg(predicted: List[int], actual: List[int], k: int) -> float:
    """
    Original NDCG calculation for list inputs - converts to tensor and calls tensor version
    """
    if not actual:
        return 0.0
    
    predicted_tensor = torch.tensor([predicted[:k]])
    actual_tensor = torch.tensor([actual])
    return calculate_ndcg_tensor(predicted_tensor, actual_tensor, k)

def calculate_entropy(recommendations: List[List[int]]) -> float:
    """
    Original entropy calculation for list inputs - converts to tensor and calls tensor version
    """
    if not recommendations:
        return 0.0
    
    # Pad sequences to same length
    max_len = max(len(rec) for rec in recommendations)
    padded_recs = [rec + [0] * (max_len - len(rec)) for rec in recommendations]
    recommendations_tensor = torch.tensor(padded_recs)
    
    return calculate_entropy_tensor(recommendations_tensor)