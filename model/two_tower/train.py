# train.py
import torch
from torch.utils.data import DataLoader
from model import ItemEncoder, UserEncoder, TwoTowerRecommender
from dataset import TwoTowerDataset
from entity import Users, Problems, UserProblem, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
import random

# --------------------------
# DB 및 데이터 불러오기
# --------------------------
engine = create_engine("sqlite:///your_db.sqlite3")
Session = sessionmaker(bind=engine)
session = Session()

users = session.query(Users).all()
problems = session.query(Problems).all()
user_problems = session.query(UserProblem).all()

user_to_problems = defaultdict(list)
for up in user_problems:
    user_to_problems[up.user_id].append(up.problem_id)

problem_meta = {
    p.problem_id: {
        "level": p.level,
        "tags": p.tags.split(",") if p.tags else ["unknown"]
    }
    for p in problems
}

TAG_SET = set(tag for p in problems for tag in (p.tags or "unknown").split(","))
TAG_TO_ID = {tag: idx for idx, tag in enumerate(TAG_SET)}

# --------------------------
# 모델 및 학습 설정
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 64

item_encoder = ItemEncoder(level_vocab_size=31, tag_vocab_size=len(TAG_TO_ID), embed_dim=embed_dim).to(device)
user_encoder = UserEncoder().to(device)
recommender = TwoTowerRecommender(item_encoder, user_encoder).to(device)
optimizer = torch.optim.Adam(recommender.parameters(), lr=1e-3)

# --------------------------
# Dataset & Loader
# --------------------------
filtered_users = [u for u in users if len(user_to_problems[u.id]) >= 6]
random.shuffle(filtered_users)
split_idx = int(len(filtered_users) * 0.8)
train_users = filtered_users[:split_idx]
valid_users = filtered_users[split_idx:]

train_dataset = TwoTowerDataset(train_users, user_to_problems, problem_meta, problem_meta.keys(), item_encoder)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# --------------------------
# 학습 루프
# --------------------------
from torch.nn.functional import binary_cross_entropy_with_logits

for epoch in range(10):
    recommender.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        pos_score = recommender(
            batch["solved_embeddings"].to(device),
            batch["pos_level"].to(device),
            batch["pos_tag"].to(device),
            batch["mask"].to(device)
        )

        neg_score = recommender(
            batch["solved_embeddings"].to(device),
            batch["neg_level"].to(device),
            batch["neg_tag"].to(device),
            batch["mask"].to(device)
        )

        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        score = torch.cat([pos_score, neg_score])
        loss = binary_cross_entropy_with_logits(score, label)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")