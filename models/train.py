from sklearn.model_selection import train_test_split
import pandas as pd
from base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import argparse
from CF import CFModel
import yaml
from tqdm import tqdm
from scipy.sparse import csr_matrix
import numpy as np

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            user_indices = batch['user_id'].to(device)
            item_indices = batch['problem_id'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()
            outputs = model(user_indices, item_indices)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
    return model

def top_k_recommendations(model, user_id, inter_csr, device, k=10):
    model.eval()
    model.to(device)
    with torch.no_grad():
        user_data = model.get_user_data(user_id)
        user_indices = user_data.to(device)
        item_indices = torch.arange(model.item_embedding.num_embeddings).to(device)
        scores = model(user_indices, item_indices)
        
        # interaction 있는 문제는 scores를 -inf로
        interaction_indices = inter_csr[user_id].indices
        scores[interaction_indices] = float('-inf')        
        
        top_k_indices = torch.topk(scores, k).indices
        return top_k_indices

    


def main():
    parser = argparse.ArgumentParser(description="Train a recommendation model")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    device = config['device']
    
    dataset = pd.read_csv("data/processed/interactions.csv")

    # 1/100
    dataset = dataset.sample(frac=0.01, random_state=42)

    # mapping
    user_id_map = {user: idx for idx, user in enumerate(dataset['user_id'].unique())}
    problem_id_map = {problem: idx for idx, problem in enumerate(dataset['problem_id'].unique())}

    labels = np.ones(len(dataset), dtype=np.float32)

    interaction_sparse = csr_matrix((labels, (dataset['user_id'].map(user_id_map), dataset['problem_id'].map(problem_id_map))), shape=(len(user_id_map), len(problem_id_map)))
    

    train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

    train_dataset = BaseDataset(train_df, user_id_map=user_id_map, problem_id_map=problem_id_map)
    test_dataset = BaseDataset(test_df, user_id_map=user_id_map, problem_id_map=problem_id_map)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_users = dataset['user_id'].nunique()
    num_items = dataset['problem_id'].nunique()
    
    model = CFModel(num_users, num_items)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trained_model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    user_id = 20 # Example user ID
    
    recommendations = top_k_recommendations(trained_model, user_id, interaction_sparse, device, k=10)
    
    user_name = list(user_id_map.keys())[list(user_id_map.values()).index(user_id)]
    print(f"Recommendations for user {user_name}:")
    for idx in recommendations:
        problem_id = list(problem_id_map.keys())[list(problem_id_map.values()).index(idx.item())]
        print(problem_id)
    
if __name__ == "__main__":
    main()