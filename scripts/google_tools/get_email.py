import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("SELECT email FROM settings LIMIT 1")
print(cursor.fetchone()[0])
conn.close()