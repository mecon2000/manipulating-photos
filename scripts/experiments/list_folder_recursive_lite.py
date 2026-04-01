import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def list_all(folder_id, folder_name):
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    service = build("drive", "v3", credentials=creds)
    
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, pageSize=50, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])
    
    print(f"FOLDER: {folder_name}")
    for item in items:
        print(f"  [{item['id']}] {item['name']} ({item['mimeType']})")

if __name__ == '__main__':
    list_all("1ndjBdjnWuRu_zgbUef5vZRuB68FGTrWf", "32 Hani on wheel")
    print("-" * 20)
    list_all("13fNzGtr_TuNFWyySdgDsiNGn9rsbKASg", "41 Mili Tipal, Morikai")
