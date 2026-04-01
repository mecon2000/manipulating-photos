import os
import random
import json
from datetime import datetime

PHOTO_ROOT = "photos/"
LOG_FILE = "state/daily_picker_log.json"

def get_all_photos():
    processed = []
    unprocessed = []
    
    if not os.path.exists(PHOTO_ROOT):
        return [], []

    for person in os.listdir(PHOTO_ROOT):
        person_path = os.path.join(PHOTO_ROOT, person)
        if not os.path.isdir(person_path):
            continue
            
        # Check for processed/unprocessed subdirectories
        for sub in ['processed', 'unprocessed']:
            sub_path = os.path.join(person_path, sub)
            if os.path.exists(sub_path) and os.path.isdir(sub_path):
                files = [os.path.join(sub_path, f) for f in os.listdir(sub_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if sub == 'processed':
                    processed.extend(files)
                else:
                    unprocessed.extend(files)
                    
    return processed, unprocessed

def pick_daily_photos(n=4, ratio=0.8):
    processed, unprocessed = get_all_photos()
    
    num_processed = int(n * ratio)
    num_unprocessed = n - num_processed
    
    selected = []
    
    if len(processed) >= num_processed:
        selected.extend(random.sample(processed, num_processed))
    else:
        selected.extend(processed)
        
    remaining = n - len(selected)
    if len(unprocessed) >= remaining:
        selected.extend(random.sample(unprocessed, remaining))
    else:
        selected.extend(unprocessed)
        
    return selected

if __name__ == "__main__":
    selected_photos = pick_daily_photos()
    print(f"Selected {len(selected_photos)} photos for {datetime.now().strftime('%Y-%m-%d')}:")
    for p in selected_photos:
        print(f" - {p}")
        
    # In a real scenario, this would then trigger the LLM/Generation step
