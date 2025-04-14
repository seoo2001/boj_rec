# interaction_scraper.py

import requests
from sqlalchemy.orm import Session
from entity import Users, UserProblem, Problems
from tqdm import tqdm
from time import sleep

SOLVED_AC_SEARCH_URL = "https://solved.ac/api/v3/search/problem"
HEADERS = {
    "Accept": "application/json",
    "x-solvedac-language": "ko"
}

def get_user_solved_problem_ids(handle: str, solved_count: int) -> list[int]:
    """
    한 유저가 푼 문제 ID 리스트를 모두 가져옴 (페이지네이션 포함)
    """
    page = 1
    all_ids = []
    pages = solved_count // 50 + 1  # 페이지 수 계산 (20문제당 1페이지)
    for page in tqdm(range(1, pages + 1)):
        params = {
            "query": f"solved_by:{handle}",
            "page": str(page),
        }

        response = requests.get(SOLVED_AC_SEARCH_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"[{handle}] 페이지 {page} 요청 실패: {response.status_code}")
            break
        
        data = response.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            all_ids.append(item["problemId"])
        
        page += 1
        sleep(0.5)
        
    return all_ids

def scrap_user_problem_interactions(db: Session):
    """
    전체 유저에 대해 solved problem 정보를 긁어서 UserProblem 테이블에 저장
    """
    users = db.query(Users).all()
    problems = db.query(Problems).all()
    
    # problem_id, id 매핑
    problem_id_mapping = {problem.problem_id: problem.id for problem in problems}
    users = reversed(users)  # 역순으로 처리하여 최신 유저부터 처리
    for user in users:
        if user.solved_count > 2000:
            continue
        # db에 user 정보 있으면 continue
        if db.query(UserProblem).filter_by(user_id=user.id).first():
            print(f"[{user.handle}] already exists")
            continue
        print(f"[{user.handle}]")
        
        try:
            solved_ids = get_user_solved_problem_ids(user.handle, user.solved_count)
            print(f"{len(solved_ids)} Complete")

            for problem_id in solved_ids:
                # 중복 방지: 이미 같은 유저-문제 조합이 존재하는지 확인
                id = problem_id_mapping.get(problem_id)
                exists = db.query(UserProblem).filter_by(user_id=user.id, problem_id=id).first()
                if exists:
                    continue

                interaction = UserProblem(
                    user_id=user.id,
                    problem_id=id,
                )
                db.add(interaction)

            db.commit()
        except Exception as e:
            print(f"Error: {e}")
            db.rollback()