import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN_PATH = 'secrets/google_calendar_token.json'
FOLDER_ID = '1QFTZlH5TZJEwBKg78uZIOsW4AdS8FeaU' # The Contemporary Art Experiments folder
FILE_PATH = 'art_output/Contemporary Art Experiments 2026-03-15/Ruby_Paper_Tear/Ruby_Paper_Tear_Final.jpg'

def upload():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    service = build('drive', 'v3', credentials=creds)
    
    file_metadata = {
        'name': 'Ruby_Paper_Tear_Final.jpg',
        'parents': [FOLDER_ID]
    }
    media = MediaFileUpload(FILE_PATH, mimetype='image/jpeg')
    file = service.files().create(body=file_metadata, media_body=media, fields='id, webContentLink').execute()
    print(f"File ID: {file.get('id')}")
    print(f"Link: {file.get('webContentLink')}")

if __name__ == "__main__":
    upload()
