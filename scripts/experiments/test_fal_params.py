import os
import requests

FAL_API_KEY = os.getenv("FAL_API_KEY")
url = "https://fal.run/fal-ai/lama/openapi.json"
headers = {"Authorization": f"Key {FAL_API_KEY}"}
response = requests.get(url, headers=headers)
print(response.json())
