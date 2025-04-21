import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_processed_data(db_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    SQLite DB에서 데이터를 읽어와서 전처리하고 train/test CSV 파일을 생성합니다.
    
    Args:
        db_path: SQLite DB 파일 경로
        output_dir: 출력 CSV 파일들이 저장될 디렉토리 경로
        test_size: 테스트 셋 비율
        random_state: 랜덤 시드
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Load data from database
    interactions_query = """
    SELECT user_id, problem_id, timestamp
    FROM interactions
    GROUP BY user_id, problem_id
    """
    
    problems_query = "SELECT id, problem_id FROM problems"
    users_query = "SELECT id, handle FROM users"
    
    # Read data into pandas DataFrames
    interactions_df = pd.read_sql_query(interactions_query, conn)
    problems_df = pd.read_sql_query(problems_query, conn)
    users_df = pd.read_sql_query(users_query, conn)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        interactions_df,
        test_size=test_size,
        random_state=random_state,
        stratify=interactions_df['user_id']
    )
    
    # Save processed data
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    problems_df.to_csv(os.path.join(output_dir, 'problems.csv'), index=False)
    
    conn.close()

if __name__ == "__main__":
    db_path = "baekjoon.db"
    output_dir = "data/processed"
    create_processed_data(db_path, output_dir)
    print("Processed data files have been created successfully!")