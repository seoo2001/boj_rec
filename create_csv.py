import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scraper.entity import Users, Problems, Interactions

DATABASE_URL = "sqlite:///data/raw/baekjoon.db"

engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_csv():
    db = SessionLocal()

    # Query the database
    users = db.query(Users).all()
    problems = db.query(Problems).all()
    interactions = db.query(Interactions).all()

    # Convert to DataFrame
    user_df = pd.DataFrame([user.__dict__ for user in users])
    problem_df = pd.DataFrame([problem.__dict__ for problem in problems])
    interaction_df = pd.DataFrame([interaction.__dict__ for interaction in interactions])

    # Drop the SQLAlchemy internal columns
    user_df.drop(columns=['_sa_instance_state'], inplace=True)
    problem_df.drop(columns=['_sa_instance_state'], inplace=True)
    interaction_df.drop(columns=['_sa_instance_state'], inplace=True)

    # Save to CSV
    user_df.to_csv('data/processed/users.csv', index=False)
    problem_df.to_csv('data/processed/problems.csv', index=False)
    interaction_df.to_csv('data/processed/interactions.csv', index=False)

    db.close()
    
if __name__ == "__main__":
    create_csv()
    print("CSV files created successfully.")