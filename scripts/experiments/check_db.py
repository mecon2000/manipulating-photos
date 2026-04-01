import sqlite3
import json

def check_db():
    try:
        conn = sqlite3.connect('catalog/photos.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM photos WHERE filename LIKE '%BLD_4183%' LIMIT 1;")
        row = cursor.fetchone()
        if row:
            # Assuming typical catalog structure: id, filename, path, size, mtime, exif (json)
            # Let's try to get column names
            cursor.execute("PRAGMA table_info(photos);")
            columns = [c[1] for c in cursor.fetchall()]
            data = dict(zip(columns, row))
            print(json.dumps(data, indent=2))
        else:
            print("Photo not found in database.")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
