import os
import sys
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def get_service():
    if not os.path.exists(TOKEN_PATH):
        print(f"Error: Token not found at {TOKEN_PATH}")
        sys.exit(1)
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    return build("drive", "v3", credentials=creds)

def search_shibari():
    service = get_service()
    query = "name contains 'shibari' or name contains 'nude' or name contains 'boudoir'"
    results = service.files().list(q=query, pageSize=20, fields="nextPageToken, files(id, name, mimeType)").execute()
    items = results.get('files', [])
    
    if not items:
        print("No files found.")
    else:
        for item in items:
            print(f"[{item['id']}] {item['name']} ({item['mimeType']})")

if __name__ == '__main__':
    search_shibari()
