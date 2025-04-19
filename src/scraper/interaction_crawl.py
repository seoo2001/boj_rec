from sqlalchemy.orm import Session
from src.utils import get_random_user_agent
from bs4 import BeautifulSoup
import requests
import time
from datetime import datetime
from src.data.entity import Users, Problems, Interactions
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

headers = {
    "Accept": "application/json",
    "User-Agent": get_random_user_agent(),
}


def crawl_first_ac_submissions(user_handle: str):
    submissions = {}
    top = None
    base_url = "https://www.acmicpc.net/status"
    
    for _ in range(1000):
        params = {
            "user_id": user_handle,
            "result_id": 4,  # 정답만
        }
        if top:
            params["top"] = top

        try:
            res = requests.get(base_url, params=params, headers=headers)
            res.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed for user {user_handle}")
            print(res.status_code)
            break
            
        
        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table#status-table tbody tr")

        if not rows:
            print(f"No more rows to scrape for user {user_handle}")
            break

        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 8:
                continue

            problem_id = tds[2].text.strip()
            # 이미 수집한 문제면 건너뜀
            if problem_id in submissions or problem_id == "":
                continue

            time_tag = tds[8].find("a")
            if not time_tag or not time_tag.has_attr("data-timestamp"):
                continue

            submission_time = time_tag["data-timestamp"]
            submission_time = datetime.fromtimestamp(int(submission_time))
            submissions[problem_id] = submission_time

        # 다음 페이지를 위한 top 값
        last_row_id = rows[-1].get("id", "")
        if last_row_id.startswith("solution-"):
            if top == last_row_id.replace("solution-", ""):
                break
            top = last_row_id.replace("solution-", "")
        else:
            break

        time.sleep(0.05)

    return submissions

def process_single_user(user, problem_mapper, db):
    """Process a single user and return the number of interactions added"""
    handle = user.handle
    print(f"Processing user {handle} (rank: {user.rank})")
    
    # interaction에 이미 있는지 확인
    interaction_found = db.query(Interactions).filter(Interactions.user_id == user.id).first()
    if interaction_found:
        print(f"User {handle} already processed, skipping")
        return 0
    
    try:
        # 새로운 크롤링 방식 적용
        submissions = crawl_first_ac_submissions(handle)
        
        if not submissions:
            print(f"No submissions found for user {handle}")
            return 0
        
        interactions = []
        for problem_id, timestamp in submissions.items():
            try:
                # 문제 ID는 숫자로 변환
                numeric_problem_id = int(problem_id)
                db_problem_id = problem_mapper.get(numeric_problem_id)
                if db_problem_id is None:
                    print(f"Problem ID {problem_id} not found in mapper.")
                    continue
                
                interactions.append(
                    Interactions(
                        user_id=user.id, 
                        problem_id=db_problem_id,
                        timestamp=timestamp
                    )
                )
            except ValueError:
                print(f"Invalid problem ID: {problem_id}")
                continue
        
        # 세션 충돌을 방지하기 위해 각 스레드에서 새로운 세션 생성
        with Session(db.get_bind().engine) as local_session:
            if interactions:
                local_session.add_all(interactions)
                local_session.commit()
                print(f"Added {len(interactions)} interactions for user {handle}")
                return len(interactions)
        
        # 서버 부하 방지를 위한 추가 대기 시간
        time.sleep(random.uniform(0.5, 2.0))
        
    except Exception as e:
        print(f"Error processing user {handle}: {e}")
        # 에러 발생 시 좀 더 긴 대기 시간
        time.sleep(random.uniform(5.0, 10.0))
        return 0

def parallel_crawl_interaction(db: Session, max_workers=50):
    """
    ThreadPoolExecutor를 사용하여 여러 사용자를 병렬로 처리
    
    Args:
        db: SQLAlchemy 세션
        max_workers: 동시에 실행할 최대 스레드 수
    """
    users = db.query(Users).all()
    
    problem_mapper = {}
    problems = db.query(Problems).all()
    for p in problems:
        problem_mapper[p.problem_id] = p.id
    
    total_processed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 유저를 처리하는 작업 제출
        future_to_user = {executor.submit(process_single_user, user, problem_mapper, db): user for user in users}
        
        # 완료된 작업 처리
        for future in as_completed(future_to_user):
            user = future_to_user[future]
            try:
                interactions_added = future.result()
                total_processed += interactions_added
            except Exception as exc:
                print(f'User {user.handle} generated an exception: {exc}')
    
    print(f"Total interactions processed: {total_processed}")
    return total_processed