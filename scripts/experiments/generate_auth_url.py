from google_auth_oauthlib.flow import InstalledAppFlow

# The credentials from the file
credentials_info = {
    "installed": {
        "client_id": "955048968527-etvsjv2jfuui5ipgsd6nu68kdbeo0qss.apps.googleusercontent.com",
        "client_secret": "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",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}

# Define the scope
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Create the flow
flow = InstalledAppFlow.from_client_config(credentials_info, SCOPES, redirect_uri='urn:ietf:wg:oauth:2.0:oob')

# Generate the authorization URL
auth_url, _ = flow.authorization_url(prompt='consent')

print(f"URL: {auth_url}")
