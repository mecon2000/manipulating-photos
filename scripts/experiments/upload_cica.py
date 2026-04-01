import json
import requests
import os

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

def create_folder(name, parent_id):
    url = "https://www.googleapis.com/drive/v3/files"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    data = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get('id')
    return None

def upload_file(path, folder_id):
    url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    file_metadata = {
        'name': os.path.basename(path),
        'parents': [folder_id]
    }
    files = {
        'data': ('metadata', json.dumps(file_metadata), 'application/json'),
        'file': open(path, 'rb')
    }
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        print(f"Uploaded {path}")
        return response.json().get('id')
    else:
        print(f"Error uploading {path}: {response.status_code} - {response.text}")
    return None

# Root folder: _photos from openclaw
root_id = '1phZQQMxNFMO-Xj_3Y46EJAs-OnHsP_iO'
folder_id = create_folder('CICA Submission Candidates', root_id)

if folder_id:
    files = [
        "working_art/cica_submission/Danielle_01_Form.jpg",
        "working_art/cica_submission/Danielle_02_Form.jpg",
        "working_art/cica_submission/Jenia_01_Form.jpg"
    ]
    for f in files:
        upload_file(f, folder_id)
    print(f"All files uploaded to folder: https://drive.google.com/drive/folders/{folder_id}")
