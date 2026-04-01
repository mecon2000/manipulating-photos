import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='accounts'")
if cursor.fetchone():
    cursor.execute("SELECT email FROM accounts")
    for r in cursor.fetchall(): print(r[0])
conn.close()