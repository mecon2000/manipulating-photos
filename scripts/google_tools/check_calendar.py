import json
import requests
import datetime

def check_calendar():
    try:
        with open('/home/openclaw/.openclaw/workspace/secrets/google_calendar_token.json', 'r') as f:
            token_data = json.load(f)
        
        access_token = token_data.get('token')
        if not access_token:
            print("No access token found.")
            return

        calendar_id = "5ed930687646972eaf6f7f010d507c6ea8e78a5edeca252b69a30937f40f15ac@group.calendar.google.com"
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        tomorrow = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).isoformat() + 'Z'
        
        url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
        params = {
            'timeMin': now,
            'timeMax': tomorrow,
            'singleEvents': True,
            'orderBy': 'startTime'
        }
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 401:
            # print("Token expired. Attempting refresh...")
            refresh_token = token_data.get('refresh_token')
            client_id = token_data.get('client_id')
            client_secret = token_data.get('client_secret')
            if refresh_token and client_id and client_secret:
                refresh_url = "https://oauth2.googleapis.com/token"
                refresh_data = {
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'refresh_token': refresh_token,
                    'grant_type': 'refresh_token'
                }
                refresh_res = requests.post(refresh_url, data=refresh_data)
                if refresh_res.status_code == 200:
                    new_token = refresh_res.json().get('access_token')
                    # print(f"Token refreshed: {new_token[:10]}...")
                    headers['Authorization'] = f'Bearer {new_token}'
                    response = requests.get(url, headers=headers, params=params)
                else:
                    # print(f"Refresh failed: {refresh_res.text}")
                    return
        
        if response.status_code == 200:
            events = response.json().get('items', [])
            if not events:
                print("No upcoming events found.")
            else:
                for event in events:
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    print(f"- {start}: {event['summary']}")
        else:
            # print(f"Error: {response.status_code} - {response.text}")
            pass
            
    except Exception as e:
        # print(f"Exception: {str(e)}")
        pass

if __name__ == "__main__":
    check_calendar()
