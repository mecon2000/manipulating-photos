import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(photos)")
for row in cursor.fetchall():
    print(row)
conn.close()
