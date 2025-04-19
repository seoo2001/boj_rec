from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import DateTime

Base = declarative_base()

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    handle = Column(String(255), unique=True, nullable=False)
    solved_count = Column(Integer)
    user_class = Column(Integer)
    tier = Column(Integer)
    rating = Column(Integer)
    rival_count = Column(Integer)
    reverse_rival_count = Column(Integer)
    max_streak = Column(Integer)
    rank = Column(Integer)
    

class Problems(Base):
    __tablename__ = "problems"
    id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(Integer, unique=True, nullable=False)
    title = Column(String)
    is_solvable = Column(Integer)
    accepted_user_count = Column(Integer)
    level = Column(Integer)
    average_tries = Column(Integer)
    tags = Column(String(1023))  # tags를 String으로 변경
    
class Interactions(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    problem_id = Column(Integer)
    timestamp = Column(DateTime)