from sqlalchemy.orm import Session
import time
import json
import requests
from scraper.entity import Users, Problems, Interactions
from scraper.query import get_user_by_handle, update_user, insert_user, get_problem_by_problem_id, insert_problem, update_problem
from sqlalchemy.orm import Session
from bs4 import BeautifulSoup
import random

from datetime import datetime


headers = { "Accept": "application/json" }
base_url = "https://solved.ac/api/v3/"


search_user_url = "ranking/tier"
search_problem_url = "search/problem"
site_stats_url = "site/stats"


def get_problem_count():
    url = base_url + site_stats_url
    response = requests.get(url, headers=headers)
    return response.json().get("problemCount", 0)

def get_user_count():
    url = base_url + site_stats_url
    response = requests.get(url, headers=headers)
    return response.json().get("userCount", 0)


def get_id_from_user(db: Session, handle: str):
    user_found = get_user_by_handle(db, handle)
    if isinstance(user_found, Users):
        return user_found.handle
    return -1

def get_id_from_problem(db: Session, problem_id: int):
    problem_found = get_problem_by_problem_id(db, problem_id)
    if isinstance(problem_found, Problems):
        return problem_found.problem_id
    return -1

######################### problem ##########################

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

def scrap_problem(db: Session, start_page: int=1):
    # 전체 문제 수
    total_problem_count = get_problem_count()
    pages = total_problem_count // 50 + 1
    
    for page in range(start_page, pages + 1):
        url = base_url + search_problem_url
        querystring = {"query": " ", "page": str(page)}
        response = requests.get(url, headers=headers, params=querystring)
        items = json.loads(response.text).get("items", [])
        
        if not items:
            print(f"[Page {page}] No more problems to scrape.")
            break
        
        scrap_problem_per_page(db, page)
        print(f"[Page {page}] Problems scraped successfully.")
        time.sleep(1)


######################### user ##########################

def scrap_user_per_page(db: Session, page: int):
    url = base_url + search_user_url
    querystring = {"page": f"{page}", }

    response = requests.get(url, headers=headers, params=querystring)
    items = json.loads(response.text).get("items", [])

    for index, item in enumerate(items):
        user = Users()
        user.handle = item.get("handle")
        user.solved_count = int(item.get("solvedCount"))
        user.user_class = int(item.get("class"))
        user.tier = int(item.get("tier"))
        user.rating = int(item.get("rating"))
        user.rival_count = int(item.get("rivalCount"))
        user.reverse_rival_count = int(item.get("reverseRivalCount"))
        user.max_streak = int(item.get("maxStreak"))
        user.rank = int(item.get("rank"))

        id_exist = get_id_from_user(db, user.handle)
        if id_exist != -1:
            user.id = id_exist
            update_user(db, user)
        else:
            insert_user(db, user)

def scrap_user(db: Session, start_page: int=1):
    all_user_count = get_user_count()
    pages = all_user_count // 50 + 1
    for page in range(start_page, pages + 1):
        url = base_url + search_user_url
        querystring = {"page": str(page)}
        response = requests.get(url, headers=headers, params=querystring)
        items = json.loads(response.text).get("items", [])

        if not items:
            print(f"[Page {page}] No more users to scrape.")
            break

        scrap_user_per_page(db, page)
        print(f"[Page {page}] Users scraped successfully.")
        time.sleep(1)


########################## interaction ##########################

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1"
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
    }

def crawl_first_ac_submissions(user_id, max_pages=100, delay=1.0):
    submissions = {}
    top = None
    base_url = "https://www.acmicpc.net/status"
    
    headers = get_random_headers()

    for page in range(max_pages):
        params = {
            "user_id": user_id,
            "result_id": 4,  # 정답만
        }
        if top:
            params["top"] = top

        try:
            res = requests.get(base_url, params=params, headers=headers)
            res.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed for user {user_id}, page {page}: {e}")
            print(res.status_code)
            break
            
        
        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table#status-table tbody tr")

        if not rows:
            print(f"No more rows to scrape for user {user_id}")
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
                print(f"Already reached the last page for user {user_id}")
                break
            top = last_row_id.replace("solution-", "")
            print(f"Next top for {user_id}: {top}")
        else:
            print(f"No more pages to scrape for user {user_id}")
            break

        time.sleep(delay)

    return submissions

def scrap_interaction(db: Session, start_page: int=1, max_pages=1000, delay=0.1):
    # 유저 전체 60000명 중, 1/10만 사용, 균등하게 뽑아야 함.
    # DB에서 모든 유저 가져와서 rank순으로 정렬하고, 유저가 1/10명만 사용
    
    users = db.query(Users).order_by(Users.rank).all()
    users = users[start_page:]
    
    # problem mapper
    problem_mapper = {}
    
    problems = db.query(Problems).all()
    for problem in problems:
        problem_mapper[problem.problem_id] = problem.id
    
    for index, user in enumerate(users):
        # 1/10만 사용
        if index % 10 != 0: 
            continue
        
        # interaction에 이미 있으면 continue
        interaction_found = db.query(Interactions).filter(Interactions.user_id == user.id).first()
        if interaction_found:
            continue
        
        handle = user.handle
        print(f"Processing user {handle} (rank: {user.rank})")
        
        try:
            # 새로운 크롤링 방식 적용
            submissions = crawl_first_ac_submissions(handle, max_pages=max_pages, delay=delay)
            
            if not submissions:
                print(f"No submissions found for user {handle}")
                continue
            
            interactions = []
            for problem_id, timestamp in submissions.items():
                try:
                    # 문제 ID는 숫자로 변환
                    problem_id = problem_mapper.get(int(problem_id))
                    if problem_id is None:
                        print(f"Problem ID {problem_id} not found in mapper.")
                        continue
                    
                    interactions.append(
                        Interactions(
                            user_id=user.id, 
                            problem_id=problem_id,
                            timestamp=timestamp
                        )
                    )
                except ValueError:
                    print(f"Invalid problem ID: {problem_id}")
                    continue
            
            if interactions:
                db.add_all(interactions)
                db.commit()
                print(f"Added {len(interactions)} interactions for user {handle}")
            
            # 서버 부하 방지를 위한 추가 대기 시간
            time.sleep(random.uniform(0.5, 2.0))
            
        except Exception as e:
            print(f"Error processing user {handle}: {e}")
            db.rollback()
            # 에러 발생 시 좀 더 긴 대기 시간
            time.sleep(random.uniform(5.0, 10.0))