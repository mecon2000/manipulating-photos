import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
# Find portrait-oriented photos from Nitzan or Daniella
cursor.execute("""
    SELECT folder, filename, width, height 
    FROM photos 
    WHERE (folder LIKE '%Nitzan%' OR folder LIKE '%Daniella%') 
    AND width < height 
    LIMIT 20
""")
for row in cursor.fetchall():
    print(f"{row[0]} | {row[1]} ({row[2]}x{row[3]})")
conn.close()
