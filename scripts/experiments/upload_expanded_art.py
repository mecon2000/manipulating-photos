import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN_PATH = 'secrets/google_calendar_token.json'
# Re-using folder ID from previous upload or creating new subfolder
PARENT_FOLDER_ID = '1QFTZlH5TZJEwBKg78uZIOsW4AdS8FeaU'
LOCAL_DIR = 'art_output/Contemporary Art Experiments 2026-03-15/Paper_Tear_Expanded/'

def upload():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    service = build('drive', 'v3', credentials=creds)
    
    # Create subfolder
    folder_metadata = {
        'name': 'Paper_Tear_Expanded',
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [PARENT_FOLDER_ID]
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    folder_id = folder.get('id')
    
    for filename in os.listdir(LOCAL_DIR):
        if filename.endswith('.png'):
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            media = MediaFileUpload(os.path.join(LOCAL_DIR, filename), mimetype='image/png')
            file = service.files().create(body=file_metadata, media_body=media, fields='id, webContentLink').execute()
            print(f"Uploaded: {filename} | Link: {file.get('webContentLink')}")

if __name__ == "__main__":
    upload()
