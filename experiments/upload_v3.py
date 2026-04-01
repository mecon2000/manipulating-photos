import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN_PATH = 'secrets/google_calendar_token.json'
PARENT_FOLDER_ID = '1QFTZlH5TZJEwBKg78uZIOsW4AdS8FeaU'
LOCAL_DIR = 'Contemporary Art Experiments 2026-03-15/v3'

def upload():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    service = build('drive', 'v3', credentials=creds)
    
    folder_metadata = {
        'name': 'v3',
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [PARENT_FOLDER_ID]
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f"Folder ID: {folder_id}")
    
    for filename in os.listdir(LOCAL_DIR):
        filepath = os.path.join(LOCAL_DIR, filename)
        if os.path.isfile(filepath):
            mime = 'image/jpeg' if filename.endswith('.jpg') else 'image/png'
            file_metadata = {'name': filename, 'parents': [folder_id]}
            media = MediaFileUpload(filepath, mimetype=mime)
            file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
            print(f"Uploaded: {filename} | {file.get('webViewLink')}")

if __name__ == "__main__":
    upload()
