import json
import requests
import sys
import os

TOKEN_PATH = '/home/openclaw/.openclaw/workspace/secrets/google_calendar_token.json'

with open(TOKEN_PATH, 'r') as f:
    token_data = json.load(f)

def get_token():
    refresh_token = token_data.get('refresh_token')
    client_id = token_data.get('client_id')
    client_secret = token_data.get('client_secret')
    refresh_url = "https://oauth2.googleapis.com/token"
    refresh_data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    }
    res = requests.post(refresh_url, data=refresh_data)
    if res.status_code == 200:
        return res.json().get('access_token')
    return None

def download_file(file_id, dest):
    access_token = get_token()
    if not access_token:
        print(f"Failed to get access token")
        return False
    
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 200:
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {dest}")
        return True
    else:
        print(f"Failed: {response.status_code}")
        return False

if __name__ == "__main__":
    # BLD_0020 (18QX97Y2uimHSTlnUOyJqYLRpg3q4G8L4)
    # BLD_0047 (1BFKBAMHYERa73ceamnl4qX1qM7ZMpZBY)
    candidates = [
        ('18QX97Y2uimHSTlnUOyJqYLRpg3q4G8L4', 'game_candidate_1.jpg'),
        ('1BFKBAMHYERa73ceamnl4qX1qM7ZMpZBY', 'game_candidate_2.jpg')
    ]
    for fid, dest in candidates:
        download_file(fid, dest)
