import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.model.CF import CF
from data.dataset import CFDataset
import numpy as np
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
            device: str = 'cpu') -> float:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for user_indices, item_indices in test_loader:
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            
            predictions = model(user_indices, item_indices)
            labels = torch.ones_like(predictions)
            
            loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

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
        rec_problems = model.recommend(dataset, handle, top_k)
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
        test_loss = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
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