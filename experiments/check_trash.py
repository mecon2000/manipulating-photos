import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def get_creds():
    return Credentials.from_authorized_user_file(TOKEN_PATH)

def check_trash_mail():
    creds = get_creds()
    service = build('gmail', 'v1', credentials=creds)
    results = service.users().messages().list(userId='me', q='Subject:"An event has been moved to Trash"', maxResults=1).execute()
    messages = results.get('messages', [])
    if not messages: return "No trash mail."
    
    msg = service.users().messages().get(userId='me', id=messages[0]['id']).execute()
    snippet = msg.get('snippet')
    return snippet

if __name__ == "__main__":
    print(check_trash_mail())
