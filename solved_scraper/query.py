# query.py
from sqlalchemy.orm import Session
from entity import Problems

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