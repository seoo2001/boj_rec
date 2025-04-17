from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Interactions(Base):
    __tablename__ = "solved"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255))
    problem_id = Column(Integer) 