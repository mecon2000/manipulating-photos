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
        print(f"Failed to get access token for {file_id}")
        return False
    
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 200:
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {file_id} to {dest}")
        return True
    else:
        print(f"Failed to download {file_id}: {response.status_code}")
        return False

if __name__ == "__main__":
    # Omry (BLD_7232E) and Raaia/Ksenia (BLD_6480)
    ids = [
        ('1EvO1kRG-Oy3s4xA8xjElZ9pZBmlwp2Wm', 'ref_omry.jpg'),
        ('1hqzkr19W9PDyddZnsIn3w3KBYfTd6-So', 'ref_ksenia.jpg')
    ]
    for fid, dest in ids:
        download_file(fid, dest)
