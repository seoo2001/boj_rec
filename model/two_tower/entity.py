# entity.py
from sqlalchemy import Column, Integer, String, Boolean, BigInteger, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import relationship
from datetime import datetime


Base = declarative_base()

class Problems(Base):
    __tablename__ = "problems"
    id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(Integer, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    is_solvable = Column(Boolean)
    accepted_user_count = Column(Integer)
    level = Column(Integer)
    average_tries = Column(Integer)
    tags = Column(String)

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    handle = Column(String(255), unique=True, index=True)
    solved_count = Column(Integer)
    user_class = Column(Integer)
    tier = Column(Integer)
    rating = Column(Integer)
    rating_by_problems_sum = Column(Integer)
    rating_by_class = Column(Integer)
    rating_by_solved_count = Column(Integer)
    rival_count = Column(Integer)
    reverse_rival_count = Column(Integer)
    max_streak = Column(Integer)
    rank = Column(Integer)
    organization = Column(String(2048), nullable=True)  # 길이 줄임
    
class UserProblem(Base):
    __tablename__ = "user_problem"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    problem_id = Column(Integer, ForeignKey("problems.problem_id"), nullable=False)