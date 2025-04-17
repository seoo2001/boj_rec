import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(CFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)
        scores = torch.sum(user_embeds * item_embeds, dim=1)
        return scores
    
    def get_user_data(self, user_id):
        user_data = torch.tensor([user_id], dtype=torch.long)
        return user_data