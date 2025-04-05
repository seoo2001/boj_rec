# scraper.py
import time
import json
import requests
from entity import Problems
from query import get_problem_by_problem_id, update_problem, insert_problem

from sqlalchemy.orm import Session

headers = {"Content-Type": "application/json"}
base_url = "https://solved.ac/api/v3/"
search_problem_url = "search/problem"

# 문제 ID 존재 여부 확인
def get_id_from_problem(db: Session, problem_id: int):
    problem_found = get_problem_by_problem_id(db, problem_id)
    if isinstance(problem_found, Problems):
        return problem_found.id
    return -1

# 특정 페이지 문제 스크래핑
def scrap_problem_per_page(db: Session, page: int):
    url = base_url + search_problem_url
    querystring = {"query": " ", "page": f"{page}"}

    response = requests.get(url, headers=headers, params=querystring)

    items = json.loads(response.text).get("items", [])
    for item in items:
        problem = Problems()
        problem.problem_id = int(item.get("problemId"))
        problem.title = item.get("titleKo")
        problem.is_solvable = item.get("isSolvable")
        problem.accepted_user_count = int(item.get("acceptedUserCount"))
        problem.level = int(item.get("level"))
        problem.average_tries = int(item.get("averageTries"))

        tags_data = item.get("tags", [])
        tags = [tag.get("key") for tag in tags_data]
        problem.tags = ",".join(tags)

        # 존재 여부 확인 후 삽입/업데이트
        id_exist = get_id_from_problem(db, problem.problem_id)
        if id_exist != -1:
            problem.id = id_exist
            update_problem(db, problem)
        else:
            insert_problem(db, problem)

# 전체 문제 크롤링
def scrap_problem(db: Session, time_interval: int = 1):
    url = base_url + search_problem_url
    querystring = {"query": " ", "page": "1"}

    try:
        response = requests.get(url, headers=headers, params=querystring)
        num_problem = json.loads(response.text).get("count", 0)
    except:
        print("API 연결 실패")
        return

    total_pages = (num_problem // 100) + (1 if num_problem % 100 > 0 else 0)

    # total pages 문제 있음. 
    # 이렇게 for문 구성하면 전체 page를 다 돌지 않음

    for page in range(1, total_pages + 1):
        try:
            print(f"[{page}/{total_pages}] 페이지 수집 중...")
            scrap_problem_per_page(db, page)
        except Exception as e:
            print(f"페이지 {page} 수집 실패: {e}")
        time.sleep(time_interval)