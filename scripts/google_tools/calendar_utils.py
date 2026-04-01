import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime

CALENDAR_ID = '5ed930687646972eaf6f7f010d507c6ea8e78a5edeca252b69a30937f40f15ac@group.calendar.google.com'

def get_service():
    token_path = 'secrets/google_calendar_token.json'
    if not os.path.exists(token_path):
        return None
    creds = Credentials.from_authorized_user_file(token_path)
    return build("calendar", "v3", credentials=creds)

def mark_posted(event_id):
    service = get_service()
    if not service: return
    event = service.events().get(calendarId=CALENDAR_ID, eventId=event_id).execute()
    if not event['summary'].startswith("✅"):
        event['summary'] = f"✅ POSTED: {event['summary']}"
        service.events().update(calendarId=CALENDAR_ID, eventId=event_id, body=event).execute()

def get_upcoming_events(days=7):
    service = get_service()
    if not service: return []
    now = datetime.utcnow().isoformat() + 'Z'
    events_result = service.events().list(calendarId=CALENDAR_ID, timeMin=now,
                                        maxResults=10, singleEvents=True,
                                        orderBy='startTime').execute()
    return events_result.get('items', [])

def create_event(summary, start_time, end_time, description=""):
    service = get_service()
    if not service: return None
    event = {
        'summary': summary,
        'description': description,
        'start': {'dateTime': start_time},
        'end': {'dateTime': end_time},
    }
    return service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
