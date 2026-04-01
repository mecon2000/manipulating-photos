import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("SELECT folder, filename, orig_path, thumb_path, tags, notes FROM photos WHERE folder LIKE '%Ruby%' OR folder LIKE '%64 Ruby%' OR folder LIKE '%65 Ruby%' ORDER BY folder, filename LIMIT 60;")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()
