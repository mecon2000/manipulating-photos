import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute('SELECT folder, COUNT(*) as count FROM photos GROUP BY folder ORDER BY count DESC LIMIT 20')
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]} photos")
conn.close()
