import os
import base64
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText

def send_email():
    token_path = 'secrets/google_calendar_token.json'
    if not os.path.exists(token_path):
        print("Error: No Gmail token found. Need to auth first.")
        return

    creds = Credentials.from_authorized_user_file(token_path)
    service = build('gmail', 'v1', credentials=creds)

    spec_file = 'Echo_Cinema_Spec.txt'
    if not os.path.exists(spec_file):
        # Fallback to a placeholder if file doesn't exist
        spec_content = "Spec content missing."
    else:
        with open(spec_file, 'r') as f:
            spec_content = f.read()

    message = MIMEText(f"Hi Ronnie,\n\nAttached is the spec for the Reel script.\n\n---\n\n{spec_content}")
    message['to'] = 'mecon2000@gmail.com'
    message['subject'] = 'Echo Cinema - Reel Script Specification'
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    try:
        service.users().messages().send(userId='me', body={'raw': raw}).execute()
        print("Email sent!")
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == "__main__":
    send_email()
