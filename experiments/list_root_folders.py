import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def get_service():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    return build("drive", "v3", credentials=creds)

def list_folders():
    service = get_service()
    query = "mimeType = 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query, pageSize=50, fields="files(id, name)").execute()
    items = results.get('files', [])
    for item in items:
        print(f"  [{item['id']}] {item['name']}")

if __name__ == '__main__':
    list_folders()
