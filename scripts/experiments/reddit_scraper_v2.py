import urllib.request
import re

def search_reddit(username):
    url = f"https://www.google.com/search?q=site:reddit.com+%22{username}%22"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            # Look for subreddit patterns: /r/subredditname/
            subs = re.findall(r'/r/(\w+)/', html)
            return set(subs)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    user = "Ron_p_wilder"
    print(f"Subreddits associated with {user}:")
    results = search_reddit(user)
    print(results)
