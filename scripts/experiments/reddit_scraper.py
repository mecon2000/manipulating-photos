import requests
import re
import sys

def get_reddit_info(username):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }
    # Search for comments and posts mentioning the username across reddit
    # This is a broad search to see what's public
    search_url = f"https://www.google.com/search?q=site:reddit.com+%22{username}%22"
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        # Extract potential subreddit links and comment links
        links = re.findall(r'reddit\.com/r/(\w+)/comments/(\w+)', response.text)
        return links
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    user = "Ron_p_wilder"
    print(f"Searching for public Reddit activity for {user}...")
    results = get_reddit_info(user)
    if isinstance(results, list):
        for sub, comm_id in set(results):
            print(f"Activity found in r/{sub} (ID: {comm_id})")
    else:
        print(f"Error: {results}")
