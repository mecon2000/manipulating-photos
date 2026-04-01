import json
import requests

with open('/home/openclaw/.openclaw/workspace/secrets/google_calendar_token.json', 'r') as f:
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

access_token = get_token()

def list_children(folder_id):
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        'q': f"'{folder_id}' in parents",
        'fields': 'files(id, name, mimeType)'
    }
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get('files', [])
    return []

print(list_children('1phZQQMxNFMO-Xj_3Y46EJAs-OnHsP_iO'))
