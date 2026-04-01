import sys
from google_auth_oauthlib.flow import InstalledAppFlow

credentials_info = {
    "installed": {
        "client_id": "955048968527-etvsjv2jfuui5ipgsd6nu68kdbeo0qss.apps.googleusercontent.com",
        "client_secret": "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",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}
# Exact verifier from previous step
verifier = "T_wZpCaBBxd7_qGwT-KBLVR2LLhNwDS4w9zjuJ~xU5ZTN-.T4y3S3bK-SogeC-EAnDJzDWponq2ndgYGohnzDGexMeGcaSi6Nv9_vR1gmHXDqOErEBbGRM5sBvaV.7uD"

flow = InstalledAppFlow.from_client_config(
    credentials_info, 
    scopes=['https://www.googleapis.com/auth/calendar'],
    redirect_uri='urn:ietf:wg:oauth:2.0:oob',
    code_verifier=verifier
)

auth_code = sys.argv[1].strip()
flow.fetch_token(code=auth_code)

creds = flow.credentials
with open('secrets/google_calendar_token.json', 'w') as f:
    f.write(creds.to_json())

print("SUCCESS")
