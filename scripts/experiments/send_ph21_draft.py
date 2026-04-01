import os
import json
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

TOKEN_PATH = 'secrets/google_calendar_token.json'

def send_email():
    with open(TOKEN_PATH, 'r') as f:
        token_data = json.load(f)
    
    creds = Credentials.from_authorized_user_info(token_data)
    service = build('gmail', 'v1', credentials=creds)

    message = MIMEMultipart()
    message['to'] = 'mecon2000@gmail.com'
    message['subject'] = 'Submission Draft: Feminine / Masculine — PH21 Gallery'

    body = """שלום רוני,

לבקשתך, מצורפת טיוטת המייל להגשה לתערוכה בברצלונה, יחד עם כל 7 הקבצים המוכנים (בפורמט של 1280 פיקסלים כפי שנדרש).

אתה יכול פשוט לעשות Forward למייל הזה לכתובת: submission@ph21gallery.com
(רק אל תשכח למחוק את השורות האלו של ההסבר לפני השליחה).

בהצלחה!

---
Dear PH21 Gallery Team,

Please find attached my submission for the group exhibition "Feminine / Masculine." I have paid the entry fee (35 EUR) separately via PayPal.

Transaction ID: 72F70187B19765248

Photographer information:
- Name: Ron P. Wilder
- City, Country: Tel Aviv, Israel
- Website: ronpwilder.com
- Theme: Feminine / Masculine

Submitted photographs:

1. Column of Air (30x40 cm) - Rigger: Shibarium
2. Lattice (30x40 cm)
3. Counterbalance (30x40 cm) - Rigger: Shibarium
4. Surface Tension (30x40 cm)
5. Pinned Flight (30x40 cm)
6. The Weight of Expected Comfort (30x40 cm)
7. Bloom (30x40 cm) - Rigger: Shibarium

Thank you for reviewing my work. I look forward to the possibility of exhibiting at PH21 Gallery.

Warm regards,
Ron P. Wilder
ronpwilder.com | @ron.p.wilder
"""
    message.attach(MIMEText(body, 'plain'))

    folder = 'ph21_submission_files'
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path): continue
        with open(path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {filename}")
            message.attach(part)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
    print("Email sent successfully.")

if __name__ == '__main__':
    send_email()
