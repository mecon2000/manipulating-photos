import json
import datetime
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def get_creds():
    token_path = 'secrets/google_calendar_token.json'
    if not os.path.exists(token_path):
        raise Exception(f"{token_path} not found. Please run auth first.")
    
    creds = Credentials.from_authorized_user_file(token_path)
    
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    
    return creds

def list_calendars():
    creds = get_creds()
    service = build("calendar", "v3", credentials=creds)
    calendar_list = service.calendarList().list().execute()
    for calendar_list_entry in calendar_list['items']:
        print(f"ID: {calendar_list_entry['id']}, Summary: {calendar_list_entry['summary']}")

if __name__ == "__main__":
    list_calendars()
