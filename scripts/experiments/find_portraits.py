import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
# Search for Nitzan and other potential models with front-facing portraits
potential_models = ['Nitzan', 'Daniella', 'Elly', 'Maayan', 'Limor', 'Shira']
for model in potential_models:
    print(f"--- Searching for {model} ---")
    cursor.execute("SELECT folder, filename FROM photos WHERE folder LIKE ? AND (filename LIKE '%Face%' OR filename LIKE '%Portrait%' OR filename LIKE '%front%') LIMIT 5", (f'%{model}%',))
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"{row[0]} | {row[1]}")
    else:
        # If no specific filenames, just show a few from their main folders
        cursor.execute("SELECT folder, filename FROM photos WHERE folder LIKE ? LIMIT 3", (f'%{model}%',))
        for row in cursor.fetchall():
            print(f"Sample from main: {row[0]} | {row[1]}")

conn.close()
