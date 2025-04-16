# query.py
from sqlalchemy.orm import Session
from entity import Problems, Users

# 문제 ID로 문제 검색
def get_problem_by_problem_id(db: Session, problem_id: int):
    return db.query(Problems).filter(Problems.problem_id == problem_id).first()

# 문제 삽입
def insert_problem(db: Session, problem: Problems):
    db.add(problem)
    db.commit()

# 문제 업데이트
def update_problem(db: Session, problem: Problems):
    db.merge(problem)
    db.commit()

# 문제 삭제
def delete_problem(db: Session, problem: Problems):
    db.delete(problem)
    db.commit()
    
# 사용자 관련
def get_user_by_handle(db: Session, handle: str):
    return db.query(Users).filter(Users.handle == handle).first()

def insert_user(db: Session, user: Users):
    db.add(user)
    db.commit()

def update_user(db: Session, user: Users):
    db.merge(user)
    db.commit()