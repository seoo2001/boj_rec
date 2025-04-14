# main.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from entity import Base  # Problems 모델이 여기에 있음
from problem_scraper import scrap_problem
# from interaction_scraper import scrap_user_problem_interactions
from interaction_scraper_2 import scrap_user_problem_interactions
from user_scraper import scrap_user


# 1. DB 연결 (SQLite는 경로만 주면 됨)
DATABASE_URL = "sqlite:///problems.db"
engine = create_engine(DATABASE_URL, echo=True)

# 2. 테이블 생성
Base.metadata.create_all(bind=engine)

# 3. 세션 생성 (DB에 접근할 때 사용)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

# 4. 문제 데이터 크롤링 및 저장
# scrap_problem(db, time_interval=1)  # 페이지마다 1초 대기

# 5. 사용자 데이터 크롤링 및 저장
# scrap_user(db, time_interval=1)  # 페이지마다 1초 대기

scrap_user_problem_interactions(db)

# 6. 세션 종료
db.close()

# 7. DB 연결 종료
engine.dispose()