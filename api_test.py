import requests
from typing import List
# https://solved.ac/api/v3/problem/lookup
import json

def test_api(path: List): # get test
    url = f"https://solved.ac/api/v3/problem/lookup"
    
    params = { 
        'problemIds': path
    }
    
    headers = {
        'Accept': 'application/json',
        'x-solvedac-language': 'ko'
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    return data

data = test_api([30001])

for d in data[0]:
    print(d)
    print(data[0][d])
    
# title ko,

# num_problem = json.loads(response.text).get("count", 0)
# url = base_url + search_problem_url
# querystring = {"query": " ", "page": "1"}
# response = requests.get(url, headers=headers, params=querystring)
def get_all_problem_count():
    
    headers = {"Content-Type": "application/json"}
    base_url = "https://solved.ac/api/v3/"
    search_problem_url = "search/problem"
    url = base_url + search_problem_url
    querystring = {"query": " ", "page": "1"}
    response = requests.get(url, headers=headers, params=querystring)
    num_problem = json.loads(response.text).get("count", 0)
    
    return num_problem

print(get_all_problem_count())