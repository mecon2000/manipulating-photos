from google_auth_oauthlib.flow import InstalledAppFlow
import pickle

credentials_info = {
    "installed": {
        "client_id": "955048968527-etvsjv2jfuui5ipgsd6nu68kdbeo0qss.apps.googleusercontent.com",
        "client_secret": "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",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}
SCOPES = ['https://www.googleapis.com/auth/calendar']
flow = InstalledAppFlow.from_client_config(credentials_info, SCOPES, redirect_uri='urn:ietf:wg:oauth:2.0:oob')
auth_url, _ = flow.authorization_url(prompt='consent')

# Store the flow's internal verifier so we can finish later
with open('secrets/auth_flow_state.pkl', 'wb') as f:
    pickle.dump(flow, f)

print(f"URL: {auth_url}")
