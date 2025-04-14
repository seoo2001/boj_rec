# dataset.py
import torch
from torch.utils.data import Dataset
import random

TAG_TO_ID = {}  # 사전에 만들어야 함

def get_tag_id(tags):
    if not tags:
        return TAG_TO_ID.get("unknown", 0)
    for t in tags:
        if t in TAG_TO_ID:
            return TAG_TO_ID[t]
    return TAG_TO_ID.get("unknown", 0)

class TwoTowerDataset(Dataset):
    def __init__(self, users, user_to_problems, problem_meta, all_problem_ids, item_encoder, num_solved=5):
        self.users = users
        self.user_to_problems = user_to_problems
        self.problem_meta = problem_meta
        self.all_problem_ids = list(all_problem_ids)
        self.num_solved = num_solved
        self.item_encoder = item_encoder

    def get_embedding(self, pid):
        meta = self.problem_meta.get(pid)
        level = torch.tensor(meta["level"])
        tag = torch.tensor(get_tag_id(meta["tags"]))
        return self.item_encoder(level.unsqueeze(0), tag.unsqueeze(0)).squeeze(0)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        solved = self.user_to_problems[user.id]
        if len(solved) < self.num_solved + 1:
            solved += random.choices(solved, k=self.num_solved + 1 - len(solved))

        pos_pid = random.choice(solved)
        neg_pid = random.choice([p for p in self.all_problem_ids if p not in solved])

        sampled_history = random.sample([p for p in solved if p != pos_pid], self.num_solved)
        solved_embeddings = torch.stack([self.get_embedding(pid) for pid in sampled_history])

        mask = torch.ones(self.num_solved)

        pos_meta = self.problem_meta[pos_pid]
        neg_meta = self.problem_meta[neg_pid]

        return {
            "solved_embeddings": solved_embeddings,
            "mask": mask,
            "pos_level": torch.tensor(pos_meta["level"]),
            "pos_tag": torch.tensor(get_tag_id(pos_meta["tags"])),
            "neg_level": torch.tensor(neg_meta["level"]),
            "neg_tag": torch.tensor(get_tag_id(neg_meta["tags"]))
        }