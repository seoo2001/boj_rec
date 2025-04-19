from src.data.entity import Users
from sqlalchemy.orm import Session
from typing import List
from src.utils import get_random_user_agent, get_random_proxy
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import time

headers = {
    "Accept": "application/json",
    "User-Agent": get_random_user_agent(),
    }

proxies = {
    "http": get_random_proxy(),
    "https": get_random_proxy(),
}

stats_url = "https://solved.ac/api/v3/site/stats"

def scrap_user_per_page(db: Session, page: int, user_id: List[int] = [0, 10, 20, 30, 40]):
    """
    Scrapes user data from the given page and stores it in the database.
    
    Args:
        db (Session): The database session to use for storing data.
        page (int): The page number to scrape.
        user_id (List[int], optional): List of user IDs to scrape. Defaults to [1].
    """
    url = "https://solved.ac/api/v3/ranking/tier"
    
    querystring = {
        "query": " ",
        "page": f"{page}",
    }
    try:
        response = requests.get(url, params=querystring, headers=headers, proxies=proxies, verify=False)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        response = requests.get(url, params=querystring, headers=headers)
    
    items = json.loads(response.text).get("items" ,[])
    
    for index in user_id:
        if index >= len(items):
            print(f"Index {index} out of range for items list.")
            break
        item = items[index]
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

        id_exist = db.query(Users.id).filter(Users.handle == user.handle).first()
        if id_exist != -1:
            user.id = id_exist
            db.merge(user)
        else:
            db.add(user)
        db.commit()
        
        time.sleep(1)
        
def parallel_scrap_user_per_page(db: Session):
    """
    Scrapes user data from multiple pages in parallel and stores it in the database.
    
    Args:
        db (Session): The database session to use for storing data.
    """
    stats_response = requests.get(stats_url, headers=headers)
    all_user_count = json.loads(stats_response.text).get("userCount", 0)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(scrap_user_per_page, db, page) for page in range(1, all_user_count // 50 + 1)]
        for future in futures:
            future.result()