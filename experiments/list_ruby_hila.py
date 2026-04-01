import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
query = "SELECT folder, filename, orig_path FROM photos WHERE folder LIKE '%Ruby (Hila)%' AND (folder LIKE '%Processed%' OR folder LIKE '%Lightly Processed%') ORDER BY folder, filename LIMIT 50"
cursor.execute(query)
for r in cursor.fetchall(): print(f'{r[0]} | {r[1]} | {r[2]}')
conn.close()