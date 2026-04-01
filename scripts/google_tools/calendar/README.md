# Google Calendar Sync for OpenClaw

## Setup Required
To enable Google Calendar syncing, Ronnie needs to:
1. Go to Google Cloud Console.
2. Enable Google Calendar API.
3. Create OAuth 2.0 Client ID (Desktop app).
4. Download `credentials.json` and place it in `/home/openclaw/.openclaw/workspace/secrets/google_calendar_credentials.json`.
5. Run `python3 scripts/calendar/auth.py` to generate `token.json`.

## Calendar Usage
- **Feed Posts:** Calendar events titled "IG Post: [Photo ID]"
- **Stories:** Calendar events titled "IG Story: [Concept]"
- **Submissions:** Calendar events titled "Exhibition: [Name]"

## Access
Sharing with: mecon2000@gmail.com
Status: PENDING (Waiting for auth)
