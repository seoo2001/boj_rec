import os
import random

def get_proxies():
    """
    Reads proxies from a file and returns them as a list.
    """
    proxies = []
    file_path = os.path.join(os.path.dirname(__file__), 'proxies.txt')
    
    with open(file_path, 'r') as file:
        for line in file:
            proxies.append(line.strip())
    
    return proxies

def get_random_proxy():
    """
    Returns a random proxy from the list of proxies.
    """
    proxies = get_proxies()
    return random.choice(proxies)