import os
import random

def get_user_agents():
    """
    Reads user agents from a file and returns them as a list.
    """
    user_agents = []
    file_path = os.path.join(os.path.dirname(__file__), 'agents.txt')
    
    with open(file_path, 'r') as file:
        for line in file:
            user_agents.append(line.strip())
    
    return user_agents

def get_random_user_agent():
    """
    Returns a random user agent from the list of user agents.
    """
    user_agents = get_user_agents()
    return random.choice(user_agents)

if __name__ == "__main__":
    # Example usage
    print(get_random_user_agent())