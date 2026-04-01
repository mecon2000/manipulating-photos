import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# If modifying these scopes, delete the file secrets/google_calendar_token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']

def main():
    creds = None
    token_path = 'secrets/google_calendar_token.json'
    creds_path = 'secrets/google_calendar_credentials.json'

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                print(f"Error: {creds_path} not found.")
                return
            
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path, 
                SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            print("\n1. Go to this URL in your browser:\n")
            print(auth_url)
            print("\n2. Log in and paste the code below.\n")
            
            code = input("Enter verification code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials
        
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    print("Authentication successful. Token saved.")

if __name__ == '__main__':
    main()
