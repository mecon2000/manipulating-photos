import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def get_thumbs(folder_id):
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)
    
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = service.files().list(q=query, pageSize=10, fields="files(id, name, thumbnailLink)").execute()
    items = results.get('files', [])
    
    for item in items:
        print(f"{item['name']}: {item.get('thumbnailLink')}")

if __name__ == '__main__':
    print("HANI WHEEL:")
    get_thumbs("16FtFGgGHLRAubem0_g68s7bIKnctTS07")
    print("\nMILI MORIKAI:")
    get_thumbs("1XfBEsZRtIp3hji887DekFFEqqPNyCCtE")
