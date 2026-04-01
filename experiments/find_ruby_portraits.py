import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
# Search for Ruby across all folders
cursor.execute("SELECT folder, filename, width, height FROM photos WHERE folder LIKE '%Ruby%'")
folders = {}
for row in cursor.fetchall():
    folder = row[0]
    folders[folder] = folders.get(folder, 0) + 1

for folder, count in folders.items():
    print(f"Folder: {folder} ({count} photos)")

conn.close()
