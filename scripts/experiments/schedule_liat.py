import subprocess
from datetime import datetime, timedelta
import random

def schedule():
    ideas = [
        "להביא פרחים לליאת",
        "להכין ארוחת ערב רומנטית",
        "לצאת להליכה רגועה יחד",
        "להזמין מקום למסעדה שבא לה",
        "ערב סרט בבית עם פופקורן",
        "לקנות שוקולד איכותי לקינוח",
        "לכתוב פתק קטן ונחמד",
        "מסאז' מפנק לכתפיים",
        "טיול שקיעה בים",
        "לשלוח הודעה באמצע היום רק כדי להגיד שאני אוהב",
        "לקנות לה משהו קטן שהיא הזכירה שבא לה",
        "לפנות זמן לשיחה עמוקה בלי טלפונים",
        "לצאת לקפה של בוקר יחד",
        "להפתיע עם קינוח שווה בסוף היום"
    ]
    random.shuffle(ideas)
    
    base_date = datetime.now()
    for i in range(14):
        event_date = base_date + timedelta(days=i)
        # Random hour between 10:00 and 19:00 (to stay safe between 8-8)
        hour = random.randint(10, 19)
        minute = random.choice([0, 15, 30, 45])
        
        start_time = event_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=1)
        
        # Format for ISO with Israel offset (+02:00 for March 2026 before DST)
        start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%S+02:00')
        end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%S+02:00')
        
        summary = ideas[i % len(ideas)]
        
        cmd = [
            "python3", "skills/google-calendar/scripts/calendar_tool.py", "create",
            "--summary", summary,
            "--start", start_iso,
            "--end", end_iso,
            "--description", "משימה נחמדה לליאת ❤️"
        ]
        
        print(f"Day {i+1}: Scheduling '{summary}' at {start_iso}")
        subprocess.run(cmd)

if __name__ == "__main__":
    schedule()
