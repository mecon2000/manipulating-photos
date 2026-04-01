import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def get_creds():
    return Credentials.from_authorized_user_file(TOKEN_PATH)

def read_last_email():
    creds = get_creds()
    service = build('gmail', 'v1', credentials=creds)
    results = service.users().messages().list(userId='me', q='from:mecon2000@gmail.com', maxResults=1).execute()
    messages = results.get('messages', [])
    if not messages: return "No mail from Ronnie."
    
    msg = service.users().messages().get(userId='me', id=messages[0]['id']).execute()
    snippet = msg.get('snippet')
    return snippet

if __name__ == "__main__":
    print(read_last_email())
