import os
import sys

# Simple script to notify Ronnie about a story via Telegram
# In a real setup, this would use a telegram bot API or sessions_send if integrated.
# For now, we'll log it and assume the system or a cron-based notification handler picks it up.

message = f"📢 TIME TO POST STORY: {sys.argv[1]}"
print(message)
# Assuming a 'telegram_notify' tool or similar exists in the environment or we use sessions_send
# Since I'm the main agent, I can't easily 'send' to a user outside of a turn, 
# but I can leave a log or use a dedicated notification script if configured.
