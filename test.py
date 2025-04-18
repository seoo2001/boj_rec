import requests
from bs4 import BeautifulSoup
import time
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
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

    for _ in range(max_pages):
        params = {
            "user_id": user_id,
            "result_id": 4,  # 정답만
        }
        if top:
            params["top"] = top

        res = requests.get(base_url, params=params, headers=headers)
        if res.status_code != 200:
            print(f"Request failed: {res.status_code}")
            break

        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table#status-table tbody tr")

        if not rows:
            print("No more rows to scrape.")
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

            submissions[problem_id] = submission_time

        # 다음 페이지를 위한 top 값
        last_row_id = rows[-1].get("id", "")
        if last_row_id.startswith("solution-"):
            top = last_row_id.replace("solution-", "")
            print(f"Next top: {top}")
        else:
            print("No more pages to scrape.")
            break

        time.sleep(delay)

    return submissions

# 사용 예시
result = crawl_first_ac_submissions("seo", max_pages=10)
for pid, time_str in result.items():
    print(f"{pid} -> {time_str}")

