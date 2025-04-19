from src.data.entity import Problems
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

def scrap_problem_per_page(db: Session, page: int):
    """
    Scrapes problem data from the given page and stores it in the database.
    
    Args:
        db (Session): The database session to use for storing data.
        page (int): The page number to scrape.
    """
    url = "https://solved.ac/api/v3/search/problem"
    
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
        
        problem_exists = db.query(Problems).filter(Problems.problem_id == problem.problem_id).first()
        if problem_exists != -1:
            problem.id = problem_exists.id
            db.merge(problem)
        else:
            db.add(problem)
    db.commit()
    
    time.sleep(1)
    
def parallel_scrap_problem_per_page(db: Session, max_workers: int = 10):
    """
    Scrapes problem data in parallel using multiple threads.
    
    Args:
        db (Session): The database session to use for storing data.
        max_workers (int, optional): The maximum number of threads to use. Defaults to 10.
    """
    
    all_problem_count_response = requests.get(stats_url, headers=headers)  
    
    all_problem_count = json.loads(all_problem_count_response.text).get("problemCount", 0)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(scrap_problem_per_page, db, page) for page in range(1, all_problem_count // 50 + 1)]
        for future in futures:
            future.result()