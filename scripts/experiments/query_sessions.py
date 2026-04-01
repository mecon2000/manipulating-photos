import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
# Get distinct folders/sessions
cursor.execute("SELECT DISTINCT folder FROM photos ORDER BY folder;")
rows = cursor.fetchall()
for row in rows:
    print(row[0])
conn.close()
