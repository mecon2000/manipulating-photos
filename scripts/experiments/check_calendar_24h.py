import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.append(os.getcwd())
import scripts.calendar_utils as cu

def main():
    events = cu.get_upcoming_events(days=1)
    if not events:
        print("No upcoming events found.")
        return
    
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    
    print(f"Upcoming events (next 24h):")
    count = 0
    for event in events:
        start_str = event['start'].get('dateTime', event['start'].get('date'))
        if 'T' in start_str:
            # Handle Z and offset manually or use fromisoformat correctly
            iso_str = start_str.replace('Z', '+00:00')
            start_dt = datetime.fromisoformat(iso_str)
        else:
            # All-day event
            start_dt = datetime.strptime(start_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            
        if start_dt <= tomorrow:
            print(f"- [{start_str}] {event['summary']}")
            count += 1
            
    if count == 0:
        print("None found in next 24h.")

if __name__ == "__main__":
    main()
