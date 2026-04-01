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

# The verifier MUST match the one used to generate the URL the user just clicked
verifier = "AO6Ye_wKoyiA202c_lCx-VK5NO7Jc85O1-dvgPjRQcoBB7cM4sprX4Z8r458McR5W0sgUKlkJVL~tf.~ftr2khYIl2FPlghKekuXfDQ0KEYxA9h6cZze_iNFNgfmGWsL"

flow = InstalledAppFlow.from_client_config(
    credentials_info, 
    scopes=['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/drive.readonly'],
    redirect_uri='urn:ietf:wg:oauth:2.0:oob',
    code_verifier=verifier
)

auth_code = sys.argv[1].strip()
flow.fetch_token(code=auth_code)

creds = flow.credentials
with open('secrets/google_calendar_token.json', 'w') as f:
    f.write(creds.to_json())

print("SUCCESS")
