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