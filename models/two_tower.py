import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, embedding_dim=64):
        super(UserTower, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)


    def forward(self, user_indices):
        return self.user_embedding(user_indices)
    
    
class ItemTower(nn.Module):
    def __init__(self, num_items, embedding_dim=64):
        super(ItemTower, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_indices):
        return self.item_embedding(item_indices)
    
    
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, margin=0.2):
        super(TwoTowerModel, self).__init__()
        
        self.margin = margin
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, embedding_dim)
        

    def forward(self, user_indices, item_indices):
        pass
    
    def train_step(self, user_indices, item_indices, neg_item_indices):
        
        pos_score = self.forward(user_indices, item_indices)
        neg_score = self.forward(user_indices, neg_item_indices)
        
        loss = torch.sum(F.relu(self.margin - pos_score + neg_score))