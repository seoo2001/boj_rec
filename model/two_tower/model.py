# model.py
import torch
import torch.nn as nn

class ItemEncoder(nn.Module):
    def __init__(self, level_vocab_size, tag_vocab_size, embed_dim):
        super().__init__()
        self.level_embedding = nn.Embedding(level_vocab_size, embed_dim)
        self.tag_embedding = nn.Embedding(tag_vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, level, tag):  # level, tag: (B,)
        level_embed = self.level_embedding(level)
        tag_embed = self.tag_embedding(tag)
        x = torch.cat([level_embed, tag_embed], dim=-1)
        return self.fc(x)

class UserEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, solved_embeddings, mask=None):
        if mask is not None:
            solved_embeddings = solved_embeddings * mask.unsqueeze(-1)
            lengths = mask.sum(dim=1, keepdim=True) + 1e-8
        else:
            lengths = torch.tensor([solved_embeddings.size(1)], device=solved_embeddings.device).float()
        return solved_embeddings.sum(dim=1) / lengths

class TwoTowerRecommender(nn.Module):
    def __init__(self, item_encoder, user_encoder):
        super().__init__()
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder

    def forward(self, solved_embeddings, level, tag, mask=None):
        user_embed = self.user_encoder(solved_embeddings, mask)
        item_embed = self.item_encoder(level, tag)
        return (user_embed * item_embed).sum(dim=-1)