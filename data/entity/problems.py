from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Problems(Base):
    __tablename__ = "problems"
    problem_id = Column(Integer, primary_key=True)
    title = Column(String)
    is_solvable = Column(Integer)
    accepted_user_count = Column(Integer)
    level = Column(Integer)
    average_tries = Column(Integer)
    tags = Column(String(1023))  # tags를 String으로 변경 