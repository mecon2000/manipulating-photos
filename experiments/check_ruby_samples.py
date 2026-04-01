import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
target_folders = ['Ruby (Hila)/Lightly Processed', 'Ruby (Hila)/Processed']
for folder in target_folders:
    print(f"--- {folder} ---")
    cursor.execute("SELECT filename FROM photos WHERE folder = ? LIMIT 20", (folder,))
    for row in cursor.fetchall():
        print(row[0])
conn.close()
