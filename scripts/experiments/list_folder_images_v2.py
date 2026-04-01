import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def list_images(folder_id, folder_name):
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)
    
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = service.files().list(q=query, pageSize=50, fields="files(id, name, thumbnailLink)").execute()
    items = results.get('files', [])
    
    print(f"FOLDER: {folder_name} ({len(items)} images)")
    for item in items:
        print(f"  [{item['id']}] {item['name']}")

if __name__ == '__main__':
    # Hani Processed
    list_images("16FtFGgGHLRAubem0_g68s7bIKnctTS07", "Hani Processed")
    print("-" * 20)
    # Mili Processed
    list_images("1XfBEsZRtIp3hji887DekFFEqqPNyCCtE", "Mili Processed")
    print("-" * 20)
    # Mili Tied Down
    list_images("1uqViFIHaZsHylkYW6_bzJGpjfd8R9Bl2", "Mili Tied Down")
