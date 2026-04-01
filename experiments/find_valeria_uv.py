import sqlite3
conn = sqlite3.connect('catalog/photos.db')
cursor = conn.cursor()
cursor.execute("SELECT orig_path FROM photos WHERE filename = ?", ("IMG_8213 - I LIKE THESE.jpg",))
row = cursor.fetchone()
if row:
    print(row[0])
else:
    cursor.execute("SELECT orig_path FROM photos WHERE filename LIKE ?", ("%IMG_8213%",))
    row = cursor.fetchone()
    if row:
        print(row[0])
conn.close()
