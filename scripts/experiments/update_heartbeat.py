import json
import time

path = 'memory/heartbeat-state.json'
try:
    with open(path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    data = {"lastChecks": {}}

now = int(time.time())
data['lastChecks']['calendar'] = now
data['lastChecks']['exhibition'] = now

with open(path, 'w') as f:
    json.dump(data, f, indent=2)
