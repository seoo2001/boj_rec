# scraper_user.py
import time
import json
import requests
from entity import Users
from query import get_user_by_handle, update_user, insert_user
from sqlalchemy.orm import Session

headers = { "Content-Type": "application/json" }
base_url = "https://solved.ac/api/v3/"
search_user_url = "ranking/tier"

def get_id_from_user(db: Session, handle: str):
    user_found = get_user_by_handle(db, handle)
    if isinstance(user_found, Users):
        return user_found.id
    return -1

def scrap_user_per_page(db: Session, page: int):
    url = base_url + search_user_url
    querystring = {"page": f"{page}"}

    response = requests.get(url, headers=headers, params=querystring)
    items = json.loads(response.text).get("items", [])

    for index, item in enumerate(items):
        user = Users()
        user.handle = item.get("handle")
        user.solved_count = int(item.get("solvedCount"))
        user.user_class = int(item.get("class"))
        user.tier = int(item.get("tier"))
        user.rating = int(item.get("rating"))
        user.rating_by_problems_sum = int(item.get("ratingByProblemsSum"))
        user.rating_by_class = int(item.get("ratingByClass"))
        user.rating_by_solved_count = int(item.get("ratingBySolvedCount"))
        user.rival_count = int(item.get("rivalCount"))
        user.reverse_rival_count = int(item.get("reverseRivalCount"))
        user.max_streak = int(item.get("maxStreak"))
        user.rank = int(item.get("rank"))

        # 조직 처리
        organizations_data = item.get("organizations")
        print(f"organizations_data: {organizations_data}")
        if organizations_data:
            orgs = [str(org.get("organizationId")) for org in organizations_data]
            user.organization = ",".join(orgs)
        else:
            user.organization = None

        id_exist = get_id_from_user(db, user.handle)
        if id_exist != -1:
            user.id = id_exist
            update_user(db, user)
        else:
            insert_user(db, user)



def scrap_user(db: Session, time_interval: int = 1):
    page = 1

    while True:
        try:
            print(f"[Page {page}] 사용자 수집 중...")
            url = base_url + search_user_url
            querystring = {"page": str(page)}
            response = requests.get(url, headers=headers, params=querystring)
            items = json.loads(response.text).get("items", [])

            # 더 이상 유저가 없으면 종료
            if not items:
                print("모든 사용자 수집 완료!")
                break

            # 기존 로직 호출
            scrap_user_per_page(db, page)

        except Exception as e:
            print(f"페이지 {page} 수집 실패: {e}")
        
        page += 1
        time.sleep(time_interval)