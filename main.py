import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os

from tqdm import tqdm
from solved_scraper.entity import Base, Problems, Users, UserProblem

# 1. 데이터 로드 및 전처리
class BaekjoonDataProcessor:
    def __init__(self, db_path="problems.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.bind = self.engine
        DBSession = sessionmaker(bind=self.engine)
        self.session = DBSession()
        
    def load_data(self):
        # 유저 데이터 로드
        users_query = self.session.query(Users).all()
        users_df = pd.DataFrame([{
            'user_id': user.id, 
            'handle': user.handle,
            'solved_count': user.solved_count,
            'tier': user.tier,
            'rating': user.rating,
            'user_class': user.user_class,
            'organization': user.organization
        } for user in users_query])
        
        # 문제 데이터 로드
        problems_query = self.session.query(Problems).all()
        problems_df = pd.DataFrame([{
            'problem_id': problem.problem_id,
            'title': problem.title,
            'level': problem.level,
            'accepted_user_count': problem.accepted_user_count,
            'average_tries': problem.average_tries,
            'tags': problem.tags
        } for problem in problems_query])
        
        # 유저-문제 인터랙션 데이터 로드
        interactions_query = self.session.query(UserProblem).all()
        interactions_df = pd.DataFrame([{
            'user_id': interaction.user_id,
            'problem_id': interaction.problem_id
        } for interaction in interactions_query])
        
        return users_df, problems_df, interactions_df
    
    def preprocess_data(self, users_df, problems_df, interactions_df):
        # 문제 태그 전처리 (쉼표로 구분된 태그 리스트를 개별 태그로 분리)
        problems_df['tags_list'] = problems_df['tags'].apply(lambda x: [] if pd.isna(x) else x.split(','))
        
        # 태그 원-핫 인코딩
        all_tags = set()
        for tags in problems_df['tags_list']:
            all_tags.update(tags)
        
        tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
        
        # 문제 특성 정규화
        problems_df['normalized_level'] = problems_df['level'] / 30  # 백준 난이도 최대값 기준
        problems_df['normalized_accepted'] = problems_df['accepted_user_count'] / problems_df['accepted_user_count'].max()
        problems_df['normalized_tries'] = problems_df['average_tries'] / problems_df['average_tries'].max()
        
        # 유저 특성 정규화
        users_df['normalized_tier'] = users_df['tier'] / 30  # 백준 티어 최대값 기준
        users_df['normalized_solved'] = users_df['solved_count'] / users_df['solved_count'].max()
        users_df['normalized_rating'] = users_df['rating'] / users_df['rating'].max()
        
        # 유저-문제 인터랙션 매트릭스 생성
        user_idx_map = {uid: i for i, uid in enumerate(users_df['user_id'].unique())}
        problem_idx_map = {pid: i for i, pid in enumerate(problems_df['problem_id'].unique())}
        
        n_users = len(user_idx_map)
        n_problems = len(problem_idx_map)
        
        interaction_matrix = np.zeros((n_users, n_problems))
        
        for _, row in interactions_df.iterrows():
            if row['user_id'] in user_idx_map and row['problem_id'] in problem_idx_map:
                u_idx = user_idx_map[row['user_id']]
                p_idx = problem_idx_map[row['problem_id']]
                interaction_matrix[u_idx, p_idx] = 1
                
        return users_df, problems_df, interaction_matrix, user_idx_map, problem_idx_map, tag_to_idx
    
    def create_problem_features(self, problems_df, tag_to_idx):
        # 문제 특성 벡터 생성
        n_problems = len(problems_df)
        n_tags = len(tag_to_idx)
        
        # 문제 특성: [난이도, 해결한 사용자 수, 평균 시도 횟수, 태그(원-핫)]
        problem_features = np.zeros((n_problems, 3 + n_tags))
        
        for i, (_, problem) in enumerate(problems_df.iterrows()):
            problem_features[i, 0] = problem['normalized_level']
            problem_features[i, 1] = problem['normalized_accepted']
            problem_features[i, 2] = problem['normalized_tries']
            
            # 태그 원-핫 인코딩
            for tag in problem['tags_list']:
                if tag in tag_to_idx:
                    problem_features[i, 3 + tag_to_idx[tag]] = 1
        
        return problem_features
    
    def create_user_features(self, users_df):
        # 유저 특성 벡터 생성
        user_features = np.column_stack([
            users_df['normalized_tier'],
            users_df['normalized_solved'],
            users_df['normalized_rating']
        ])
        
        return user_features
    
    def train_valid_split(self, interaction_matrix, test_size=0.01, random_state=42):
        n_users, n_problems = interaction_matrix.shape
        
        # 각 유저별로 문제를 훈련/검증 세트로 분할
        train_matrix = np.zeros_like(interaction_matrix)
        valid_matrix = np.zeros_like(interaction_matrix)
        
        for u in range(n_users):
            solved_problems = np.where(interaction_matrix[u] > 0)[0]
            
            if len(solved_problems) > 1:  # 최소 2개 이상의 문제를 풀었을 경우에만 분할
                n_valid = max(1, int(len(solved_problems) * test_size))
                
                # 랜덤하게 검증 문제 선택
                np.random.seed(random_state + u)  # 각 유저마다 다른 시드 사용
                valid_indices = np.random.choice(solved_problems, n_valid, replace=False)
                
                # 나머지는 훈련 세트로
                train_indices = np.array([p for p in solved_problems if p not in valid_indices])
                
                train_matrix[u, train_indices] = 1
                valid_matrix[u, valid_indices] = 1
            else:
                # 문제가 1개 이하면 훈련 세트에 모두 할당
                train_matrix[u] = interaction_matrix[u]
        
        return train_matrix, valid_matrix

# 2. 모델 정의
class BaekjoonRecommender(nn.Module):
    def __init__(self, n_users, n_problems, user_features_dim, problem_features_dim, embedding_dim=64):
        super(BaekjoonRecommender, self).__init__()
        
        # 유저 임베딩 레이어
        self.user_embedding = nn.Linear(user_features_dim, embedding_dim)
        
        # 문제 임베딩 레이어
        self.problem_embedding = nn.Linear(problem_features_dim, embedding_dim)
        
        # 유저 특성 변환 레이어
        self.user_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU()
        )
        
        # 문제 특성 변환 레이어
        self.problem_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU()
        )
    
    def encode_user(self, user_features):
        # 유저 특성을 임베딩 공간으로 변환
        user_emb = self.user_embedding(user_features)
        return self.user_encoder(user_emb)
    
    def encode_problem(self, problem_features):
        # 문제 특성을 임베딩 공간으로 변환
        problem_emb = self.problem_embedding(problem_features)
        return self.problem_encoder(problem_emb)
    
    def forward(self, user_features, problem_features):
        # 유저와 문제 임베딩 계산
        user_emb = self.encode_user(user_features)
        problem_emb = self.encode_problem(problem_features)
        
        # 내적으로 유사도 계산
        scores = torch.matmul(user_emb, problem_emb.t())
        return scores

# 3. 데이터셋 및 데이터로더
class BaekjoonDataset(Dataset):
    def __init__(self, user_features, problem_features, interaction_matrix, n_negatives=4):
        self.user_features = torch.FloatTensor(user_features)
        self.problem_features = torch.FloatTensor(problem_features)
        self.interaction_matrix = interaction_matrix
        self.n_negatives = n_negatives
        
        # 양성 샘플(유저가 푼 문제) 쌍 생성
        self.user_indices, self.pos_problem_indices = np.where(interaction_matrix > 0)
        
    def __len__(self):
        return len(self.user_indices)
    
    def __getitem__(self, idx):
        user_idx = self.user_indices[idx]
        pos_problem_idx = self.pos_problem_indices[idx]
        
        # 현재 유저에 대한 랜덤 부정 샘플(풀지 않은 문제) 선택
        neg_problem_indices = []
        for _ in range(self.n_negatives):
            neg_idx = np.random.randint(0, self.problem_features.shape[0])
            while self.interaction_matrix[user_idx, neg_idx] > 0:
                neg_idx = np.random.randint(0, self.problem_features.shape[0])
            neg_problem_indices.append(neg_idx)
        
        # 유저 특성과 문제 특성 반환
        user_feat = self.user_features[user_idx]
        pos_problem_feat = self.problem_features[pos_problem_idx]
        neg_problem_feats = self.problem_features[neg_problem_indices]
        
        return user_feat, pos_problem_feat, neg_problem_feats

# 4. 손실 함수 및 학습 루틴
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        
    def forward(self, pos_scores, neg_scores):
        # Bayesian Personalized Ranking 손실
        # log(sigmoid(pos_score - neg_score))
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

# 5. 학습 및 평가 함수
def train_model(model, train_dataset, valid_matrix, problem_features, user_features, 
                n_epochs=5, batch_size=1024, lr=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BPRLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model = model.to(device)
    
    best_ndcg = 0
    best_model_state = None
    
    problem_features_tensor = torch.FloatTensor(problem_features).to(device)
    user_features_tensor = torch.FloatTensor(user_features).to(device)
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        total_loss = 0
        
        for user_feat, pos_problem_feat, neg_problem_feats in train_loader:
            user_feat = user_feat.to(device)
            pos_problem_feat = pos_problem_feat.to(device)
            neg_problem_feats = neg_problem_feats.to(device)
            
            batch_size = user_feat.size(0)
            n_negatives = neg_problem_feats.size(1)
            
            # 예측 점수 계산
            pos_scores = model(user_feat, pos_problem_feat)
            
            # 부정 샘플에 대한 점수 계산
            neg_scores = []
            for i in range(n_negatives):
                neg_score = model(user_feat, neg_problem_feats[:, i])
                neg_scores.append(neg_score.unsqueeze(1))
            
            neg_scores = torch.cat(neg_scores, dim=1)
            
            # 손실 계산
            loss = 0
            for i in range(n_negatives):
                loss += criterion(pos_scores.view(-1), neg_scores[:, i].reshape(-1))
            loss /= n_negatives
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
        
        avg_loss = total_loss / len(train_dataset)
        
        # 검증 성능 평가
        model.eval()
        ndcg10 = evaluate_model(model, valid_matrix, problem_features_tensor, user_features_tensor)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, NDCG@10: {ndcg10:.4f}")
        
        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            best_model_state = model.state_dict()
    
    # 최고 성능 모델 복원
    model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, valid_matrix, problem_features_tensor, user_features_tensor, k=10):
    device = next(model.parameters()).device
    
    with torch.no_grad():
        all_ndcg = []
        
        # 검증 세트에 존재하는 유저만 평가
        valid_users = np.where(valid_matrix.sum(axis=1) > 0)[0]
        
        for user_idx in valid_users:
            user_feat = user_features_tensor[user_idx].unsqueeze(0)
            
            # 모든 문제에 대한 추천 점수 계산
            scores = model(user_feat, problem_features_tensor).cpu().numpy().flatten()
            
            # 훈련에 사용된 문제는 추천에서 제외
            train_indices = np.where(valid_matrix[user_idx] == 0)[0]
            scores[train_indices] = -np.inf
            
            # 최상위 k개 문제 선택
            top_k_items = np.argsort(-scores)[:k]
            
            # Ground truth와 예측 결과 준비
            y_true = valid_matrix[user_idx]
            y_score = np.zeros_like(y_true)
            y_score[top_k_items] = 1
            
            # nDCG@k 계산
            if np.sum(y_true) > 0:  # 검증 데이터가 있는 경우만
                finite_scores = scores[np.isfinite(scores)]
                if finite_scores.size > 0:
                    max_finite = np.max(finite_scores)
                    min_finite = np.min(finite_scores)
                else:
                    max_finite, min_finite = 0, 0

                scores = np.where(np.isposinf(scores), max_finite, scores)
                scores = np.where(np.isneginf(scores), min_finite, scores)
                scores = np.nan_to_num(scores, nan=0.0)
                
                
                ndcg = ndcg_score(y_true.reshape(1, -1), scores.reshape(1, -1), k=k)
                all_ndcg.append(ndcg)
        
        mean_ndcg = np.mean(all_ndcg) if all_ndcg else 0
        
    return mean_ndcg

# 6. 새로운 유저에 대한 추천 함수
def recommend_for_new_user(model, problem_features, problem_idx_map, user_meta, tag_to_idx, 
                          solved_problems=None, top_k=10):
    device = next(model.parameters()).device
    
    # 새로운 유저 특성 생성
    user_tier = user_meta.get('tier', 0) / 30
    user_solved = user_meta.get('solved_count', 0) / 5000  # 적당한 최대값으로 정규화
    user_rating = user_meta.get('rating', 0) / 3000  # 적당한 최대값으로 정규화
    
    user_features = torch.FloatTensor([[user_tier, user_solved, user_rating]]).to(device)
    
    # 이미 푼 문제 ID 목록
    if solved_problems is None:
        solved_problems = []
    
    # 유저가 푼 문제의 인덱스
    solved_indices = [problem_idx_map.get(pid, -1) for pid in solved_problems]
    solved_indices = [idx for idx in solved_indices if idx != -1]
    
    problem_features_tensor = torch.FloatTensor(problem_features).to(device)
    
    with torch.no_grad():
        # 모든 문제에 대한 추천 점수 계산
        scores = model(user_features, problem_features_tensor).cpu().numpy().flatten()
        
        # 이미 푼 문제 제외
        scores[solved_indices] = -np.inf
        
        # 최상위 k개 문제 선택
        top_k_indices = np.argsort(-scores)[:top_k]
        
        # 인덱스를 실제 문제 ID로 변환
        idx_to_problem = {v: k for k, v in problem_idx_map.items()}
        recommended_problems = [idx_to_problem[idx] for idx in top_k_indices]
        
    return recommended_problems, scores[top_k_indices]

# 7. 메인 함수: 데이터 처리, 모델 학습 및 저장
def main():
    # 데이터 로드 및 전처리
    processor = BaekjoonDataProcessor()
    users_df, problems_df, interactions_df = processor.load_data()
    
    users_df, problems_df, interaction_matrix, user_idx_map, problem_idx_map, tag_to_idx = processor.preprocess_data(
        users_df, problems_df, interactions_df
    )
    
    # 문제 및 유저 특성 생성
    problem_features = processor.create_problem_features(problems_df, tag_to_idx)
    user_features = processor.create_user_features(users_df)
    
    # 훈련/검증 데이터 분할
    train_matrix, valid_matrix = processor.train_valid_split(interaction_matrix)
    
    # 모델 초기화
    n_users = len(user_idx_map)
    n_problems = len(problem_idx_map)
    user_features_dim = user_features.shape[1]
    problem_features_dim = problem_features.shape[1]
    
    model = BaekjoonRecommender(n_users, n_problems, user_features_dim, problem_features_dim)
    
    # 데이터셋 생성
    train_dataset = BaekjoonDataset(user_features, problem_features, train_matrix)
    
    # 모델 학습
    model = train_model(model, train_dataset, valid_matrix, problem_features, user_features)
    
    # 모델 및 메타데이터 저장
    model_dir = "saved_model"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "baekjoon_recommender.pth"))
    
    metadata = {
        'user_idx_map': user_idx_map,
        'problem_idx_map': problem_idx_map,
        'tag_to_idx': tag_to_idx,
        'user_features_dim': user_features_dim,
        'problem_features_dim': problem_features_dim,
        'problem_features': problem_features
    }
    
    with open(os.path.join(model_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Model and metadata saved to {model_dir}")
    
    # 검증 성능 최종 평가
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    problem_features_tensor = torch.FloatTensor(problem_features).to(device)
    user_features_tensor = torch.FloatTensor(user_features).to(device)
    
    final_ndcg = evaluate_model(model, valid_matrix, problem_features_tensor, user_features_tensor)
    print(f"Final NDCG@10: {final_ndcg:.4f}")
    
    # 새로운 유저에 대한 추천 예시
    new_user_meta = {
        'tier': 8,  # 예시: 골드 5
        'solved_count': 100,
        'rating': 850
    }
    solved_problems = [1000, 1001, 1002]  # 예시: 이미 푼 문제들
    
    recommended_problems, scores = recommend_for_new_user(
        model, problem_features, problem_idx_map, new_user_meta, tag_to_idx, solved_problems
    )
    
    print("Recommended problems for new user:")
    for i, (problem_id, score) in enumerate(zip(recommended_problems, scores)):
        print(f"{i+1}. Problem ID: {problem_id}, Score: {score:.4f}")

# 8. API 서비스를 위한 클래스
class BaekjoonRecommenderService:
    def __init__(self, model_dir="saved_model"):
        # 메타데이터 로드
        with open(os.path.join(model_dir, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)
        
        # 모델 초기화 및 가중치 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        
        n_users = 1  # API 서비스에서는 한 번에 한 유저만 처리
        n_problems = len(self.metadata['problem_idx_map'])
        user_features_dim = self.metadata['user_features_dim']
        problem_features_dim = self.metadata['problem_features_dim']
        
        self.model = BaekjoonRecommender(n_users, n_problems, user_features_dim, problem_features_dim)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "baekjoon_recommender.pth")))
        self.model.to(self.device)
        self.model.eval()
        
        # 문제 특성 로드
        self.problem_features = torch.FloatTensor(self.metadata['problem_features']).to(self.device)
    
    def recommend(self, user_meta, solved_problems=None, top_k=10):
        """
        새로운 유저에게 문제 추천
        
        Args:
            user_meta: 유저 메타데이터 (티어, 푼 문제 수, 레이팅 등)
            solved_problems: 유저가 이미 푼 문제 ID 목록
            top_k: 추천할 문제 수
            
        Returns:
            추천된 문제 ID 목록과 점수
        """
        return recommend_for_new_user(
            self.model, 
            self.metadata['problem_features'],
            self.metadata['problem_idx_map'],
            user_meta,
            self.metadata['tag_to_idx'],
            solved_problems,
            top_k
        )

if __name__ == "__main__":
    main()