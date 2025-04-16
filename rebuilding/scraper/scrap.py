from sqlalchemy.orm import Session
import time
import json
import requests
from entity import Users, Problems, Interactions
from query import get_user_by_handle, update_user, insert_user, get_problem_by_problem_id, insert_problem, update_problem
from sqlalchemy.orm import Session
from bs4 import BeautifulSoup


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

def scrap_interaction(db: Session, start_page: int=1):
    # 유저 전체 60000명 중, 1/10만 사용, 균등하게 뽑아야 함.
    # DB에서 모든 유저 가져와서 rank순으로 정렬하고, 유저가 1/10명만 사용
    
    # https://www.acmicpc.net/user/{handle} 에서 problem-list div가져와서, result-ac <a> 태그 데이터 parsing
    
    users = db.query(Users).order_by(Users.rank).all()
    
    for index, user in enumerate(users):
        # 1/10만 사용
        if index % 10 != 0: 
            continue
        
        # interaction에 이미 있으면 continue
        interaction_found = db.query(Interactions).filter(Interactions.user_id == user.handle).first()
        if interaction_found:
            continue
        
        handle = user.handle
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        url = f"https://www.acmicpc.net/user/{handle}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[ERROR] Failed to fetch user page: {e}")
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        
        problem_list_div = soup.find('div', class_='problem-list')
        if not problem_list_div:
            continue
        
        interactions = []
        a_tags = problem_list_div.find_all('a')
        for a_tag in a_tags:
            problem_id = int(a_tag['href'].split('/')[-1])
            interactions.append(Interactions(user_id=user.handle, problem_id=problem_id))
            
        if interactions:
            db.add_all(interactions)
            db.commit()