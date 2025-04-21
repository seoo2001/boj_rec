import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.model.CF import CF
from data.dataset import CFDataset
from src.utils.metrics import calculate_ndcg_tensor, calculate_entropy_tensor
from typing import List, Dict
import os
from src.data.preprocess import create_processed_data

def train_epoch(model: CF, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer,
                device: str = 'cpu') -> float:
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for user_indices, item_indices in train_loader:
        user_indices = user_indices.to(device)
        item_indices = item_indices.to(device)
        
        # Forward pass
        predictions = model(user_indices, item_indices)
        
        # All interactions in the dataset are positive
        labels = torch.ones_like(predictions)
        
        # Binary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model: CF, 
            test_loader: DataLoader,
            dataset: CFDataset,
            device: str = 'cpu',
            k: int = 10) -> Dict[str, float]:
    """
    Evaluate model using multiple metrics
    
    Args:
        model: CF model
        test_loader: Test data loader
        dataset: Dataset instance for getting actual user interactions
        device: Device to run inference on
        k: Top-K items for NDCG calculation
        
    Returns:
        Dictionary containing different evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_recommendations = []
    all_actual_interactions = []
    
    with torch.no_grad():
        for user_indices, item_indices in test_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            
            # Calculate BCE loss
            predictions = model(user_indices, item_indices)
            labels = torch.ones_like(predictions)
            loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
            total_loss += loss.item()
            
            # Get recommendations and actual interactions for unique users in batch
            unique_users = user_indices.unique()
            batch_recommendations = []
            batch_interactions = []
            
            # Get recommendations in parallel for the batch
            for user_idx in unique_users:
                user_handle = dataset.get_handle_from_id(user_idx.item())
                if user_handle:
                    recommendations = model.recommend(dataset, user_handle, k, device)
                    actual_problems = dataset.get_user_interactions(user_idx.item())
                    if recommendations:
                        batch_recommendations.append(recommendations)
                        batch_interactions.append(actual_problems)
            
            if batch_recommendations:
                all_recommendations.extend(batch_recommendations)
                all_actual_interactions.extend(batch_interactions)
    
    # Calculate metrics using tensor operations
    if all_recommendations:
        # Prepare tensors for NDCG calculation
        max_actual_len = max(len(inter) for inter in all_actual_interactions)
        padded_actual = [inter + [0] * (max_actual_len - len(inter)) for inter in all_actual_interactions]
        
        recommendations_tensor = torch.tensor([rec[:k] for rec in all_recommendations])
        actual_tensor = torch.tensor(padded_actual)
        
        # Calculate NDCG and entropy using tensor operations
        ndcg = calculate_ndcg_tensor(recommendations_tensor, actual_tensor, k)
        diversity = calculate_entropy_tensor(recommendations_tensor)
    else:
        ndcg = 0.0
        diversity = 0.0
    
    avg_loss = total_loss / len(test_loader)
    
    return {
        'loss': avg_loss,
        f'ndcg@{k}': ndcg,
        'diversity': diversity
    }

def get_recommendations(model: CF,
                       dataset: CFDataset,
                       handles: List[str],
                       top_k: int = 10,
                       device: str = 'cpu') -> Dict[str, List[int]]:
    """
    Get recommendations for users
    
    Args:
        model: CF model
        dataset: CFDataset instance
        handles: List of user handles to get recommendations for
        top_k: Number of recommendations to generate per user
        device: Device to run inference on
        
    Returns:
        Dictionary mapping user handles to lists of recommended problem IDs
    """
    recommendations = {}
    
    for handle in handles:
        rec_problems = model.recommend(dataset, handle, top_k, device)
        if rec_problems:  # Only add if we got recommendations
            recommendations[handle] = rec_problems
    
    return recommendations

def main():
    # Hyperparameters
    EMBEDDING_DIM = 64
    BATCH_SIZE = 1024
    EPOCHS = 3
    LR = 0.001
    
    # Device configuration
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Prepare data directory
    data_dir = "data/processed"
    db_path = "baekjoon.db"
    
    # Create processed data if it doesn't exist
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("Preprocessing data...")
        create_processed_data(db_path, data_dir)
        print("Data preprocessing completed!")
    
    # Load datasets
    train_dataset = CFDataset(data_dir=data_dir, split="train")
    test_dataset = CFDataset(data_dir=data_dir, split="test")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    model = CF(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        metrics = evaluate(model, test_loader, train_dataset, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {metrics['loss']:.4f}")
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}, Diversity: {metrics['diversity']:.4f}")
    
    # Example: Get recommendations for sample users
    print("\nGetting recommendations...")
    sample_handles = ["cubelover", "kks227", "koosaga"]  # Example handles
    recommendations = get_recommendations(
        model,
        train_dataset,
        sample_handles,
        top_k=10,
        device=DEVICE
    )
    
    for handle, rec_problems in recommendations.items():
        print(f"\nRecommendations for user {handle}:")
        print(f"Recommended problem IDs: {rec_problems}")

if __name__ == "__main__":
    main()