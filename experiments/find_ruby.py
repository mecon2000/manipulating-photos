import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("SELECT folder, filename FROM photos WHERE folder LIKE '%Ruby%' AND (filename LIKE '%Face%' OR filename LIKE '%Portrait%' OR width < height) LIMIT 50")
for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]}")
conn.close()
