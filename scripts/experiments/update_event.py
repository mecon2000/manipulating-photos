import sys
import calendar_utils

def update_event(event_id, summary=None, description=None):
    service = calendar_utils.get_service()
    if not service:
        print("Error: No service")
        return
    event = service.events().get(calendarId=calendar_utils.CALENDAR_ID, eventId=event_id).execute()
    if summary:
        event['summary'] = summary
    if description:
        event['description'] = description
    service.events().update(calendarId=calendar_utils.CALENDAR_ID, eventId=event_id, body=event).execute()
    print(f"Updated event {event_id}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 update_event.py [id] [summary|description] [text]")
    else:
        mode = sys.argv[2]
        text = " ".join(sys.argv[3:])
        if mode == "summary":
            update_event(sys.argv[1], summary=text)
        elif mode == "description":
            update_event(sys.argv[1], description=text)
