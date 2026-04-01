import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def get_service():
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    return build("drive", "v3", credentials=creds)

def search_drive():
    service = get_service()
    queries = [
        "name contains 'Hani'",
        "name contains 'Mili'",
        "name contains 'Mori Kai'",
        "name contains 'circle'"
    ]
    
    for query in queries:
        print(f"SEARCHING: {query}")
        results = service.files().list(q=query, pageSize=20, fields="nextPageToken, files(id, name, mimeType)").execute()
        items = results.get('files', [])
        if not items:
            print("  No results.")
        else:
            for item in items:
                print(f"  [{item['id']}] {item['name']} ({item['mimeType']})")

if __name__ == '__main__':
    search_drive()
