import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("SELECT path FROM photos WHERE folder LIKE '%64 Ruby, Hanuka%'")
for r in cursor.fetchall(): print(r[0])
conn.close()