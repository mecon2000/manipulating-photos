import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("SELECT folder, filename, orig_path FROM photos WHERE folder LIKE '%Ruby%' LIMIT 10;")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
