import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta

CALENDAR_ID = '5ed930687646972eaf6f7f010d507c6ea8e78a5edeca252b69a30937f40f15ac@group.calendar.google.com'

def get_service():
    token_path = 'secrets/google_calendar_token.json'
    creds = Credentials.from_authorized_user_file(token_path)
    return build("calendar", "v3", credentials=creds)

service = get_service()
now = "2026-03-13T00:00:00Z"
then = "2026-03-16T00:00:00Z"

events_result = service.events().list(calendarId=CALENDAR_ID, timeMin=now, timeMax=then,
                                    singleEvents=True, orderBy='startTime').execute()
events = events_result.get('items', [])

for event in events:
    print(f"--- {event['summary']} ---")
    print(f"Start: {event['start'].get('dateTime', event['start'].get('date'))}")
    print(f"Description: {event.get('description', 'No description')}")
    print("-" * 20)
