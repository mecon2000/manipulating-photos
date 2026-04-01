import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

CALENDAR_ID = '5ed930687646972eaf6f7f010d507c6ea8e78a5edeca252b69a30937f40f15ac@group.calendar.google.com'
EVENT_ID = 'da3mfgbskhjphub27fo6lffbu8'

def get_service():
    token_path = 'secrets/google_calendar_token.json'
    creds = Credentials.from_authorized_user_file(token_path)
    return build("calendar", "v3", credentials=creds)

service = get_service()
event = service.events().get(calendarId=CALENDAR_ID, eventId=EVENT_ID).execute()
print(json.dumps(event, indent=2))
