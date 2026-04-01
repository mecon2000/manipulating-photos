import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime

CALENDAR_ID = '5ed930687646972eaf6f7f010d507c6ea8e78a5edeca252b69a30937f40f15ac@group.calendar.google.com'

def get_service():
    token_path = 'secrets/google_calendar_token.json'
    creds = Credentials.from_authorized_user_file(token_path)
    return build("calendar", "v3", credentials=creds)

def update_elly_post(service):
    event_id = 'da3mfgbskhjphub27fo6lffbu8'
    event = service.events().get(calendarId=CALENDAR_ID, eventId=event_id).execute()
    
    new_caption = (
        "אל תתנו למראה ה-badass להטעות אתכם. נכון, יש ראסטות ויש קעקועים, אבל אלי היא אחת הבחורות הכי מתוקות, "
        "חכמות ומצחיקות שיצא לי לפגוש (ולצלם!). השילוב הזה של קשיחות חיצונית עם לב ענק הוא פשוט ממכר. "
        "היה תענוג של סשן, וכבר מחכה להזדמנות הבאה ליצור יחד. 🔥🖤"
    )
    
    tags = "@miss_grin17 @visionaryframes11"
    hashtags = "#portraitphotography #fineartphotography #naturallight #lightandshadow #moodyphotography #intimatephotography #boudoirphotography #artphotography #elegance #womenofinstagram #israel #telaviv"
    
    new_description = f"Type: Post\nModel: Elly\n\n--- CAPTION ---\n{new_caption}\n\nTags: {tags}\n\n--- HASHTAGS ---\n{hashtags}\n\n--- NOTES ---\nPhoto: BLD_5552. Scheduled via cron.\n\nGDrive Link: https://drive.google.com/file/d/1ATbWPkx7SQGZgimZIHjXdcygeM3KGqgY/view?usp=drivesdk"
    
    event['description'] = new_description
    updated_event = service.events().update(calendarId=CALENDAR_ID, eventId=event_id, body=event).execute()
    print(f"Updated event {event_id}: {updated_event.get('htmlLink')}")

def create_stories(service):
    stories = [
        {
            'summary': '[Story] A: Tough vs Soft Poll',
            'description': 'Visual: Elly photo (BLD_5616E).\nContent: Poll about "Tough vs Soft" vibes.',
            'start': '2026-03-14T10:00:00+02:00',
            'end': '2026-03-14T11:00:00+02:00',
            'colorId': '5' # Yellow for stories
        },
        {
            'summary': '[Story] B: Repost Elly Feed',
            'description': 'Content: Repost of the Elly feed post with a comment about her personality (badass vs sweet).',
            'start': '2026-03-14T21:00:00+02:00',
            'end': '2026-03-14T22:00:00+02:00',
            'colorId': '5'
        },
        {
            'summary': '[Story] C: Reel Teaser',
            'description': 'Content: Teaser for the Reel coming Sunday night.',
            'start': '2026-03-15T11:00:00+02:00',
            'end': '2026-03-15T12:00:00+02:00',
            'colorId': '5'
        }
    ]
    
    for story in stories:
        event_body = {
            'summary': story['summary'],
            'description': story['description'],
            'start': {
                'dateTime': story['start'],
                'timeZone': 'Asia/Jerusalem',
            },
            'end': {
                'dateTime': story['end'],
                'timeZone': 'Asia/Jerusalem',
            },
            'colorId': story['colorId']
        }
        created_event = service.events().insert(calendarId=CALENDAR_ID, body=event_body).execute()
        print(f"Created story event: {created_event.get('htmlLink')}")

if __name__ == "__main__":
    svc = get_service()
    update_elly_post(svc)
    create_stories(svc)
