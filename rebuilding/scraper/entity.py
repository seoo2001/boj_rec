from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Users(Base):
    __tablename__ = "users"
    handle = Column(String(255), primary_key=True)
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
    problem_id = Column(Integer, primary_key=True)
    title = Column(String)
    is_solvable = Column(Integer)
    accepted_user_count = Column(Integer)
    level = Column(Integer)
    average_tries = Column(Integer)
    tags = Column(String(1023))  # tags를 String으로 변경
    
    
class Interactions(Base):
    __tablename__ = "solved"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255))
    problem_id = Column(Integer)
    
# Key를 유저 Handle로 설정했을 때
# 유저 Handle로 유저 정보 쉽게 접근 가능
# 하지만, 모델 학습 과정에서 one-hot 인코딩 후, 다시 돌리는 dictionary가 필요함.