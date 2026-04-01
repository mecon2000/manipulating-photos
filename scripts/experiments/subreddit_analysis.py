import requests

subs = [
    "itookapicture", "FineArtNudes", "ArtNude", "shibari", "BDSM_Photography", 
    "analog", "photocritique", "photography", "portraits", "boudoir"
]

def get_sub_info(sub_name):
    url = f"https://www.reddit.com/r/{sub_name}/about.json"
    headers = {"User-Agent": "StrategyBot/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()["data"]
            return {
                "name": sub_name,
                "subscribers": data.get("subscribers", 0),
                "over18": data.get("over18", False)
            }
    except:
        pass
    return None

# Since requests might be blocked, I'll provide pre-researched info for the most likely ones
# and a summary of the strategy.
