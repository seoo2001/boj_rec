# entity.py
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Problems(Base):
    __tablename__ = "problems"

    id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(Integer, unique=True, nullable=False)
    title = Column(String, nullable=False)
    is_solvable = Column(Boolean)
    accepted_user_count = Column(Integer)
    level = Column(Integer)
    average_tries = Column(Integer)
    tags = Column(String)  # 태그 문자열 저장용