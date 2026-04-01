import sqlite3
import json

def find_photos():
    filenames = [
        "Rong_IMG_0379-Edit.jpg",
        "BLD_8893.jpg",
        "BLD_0177.jpg",
        "AP2A1467E.jpg",
        "Rong_IMG_0319-Edit.jpg",
        "BLD_8902E.jpg",
        "BLD_5307.jpg"
    ]
    
    conn = sqlite3.connect('catalog/photos.db')
    cursor = conn.cursor()
    
    results = {}
    for fn in filenames:
        cursor.execute("SELECT orig_path FROM photos WHERE filename = ?", (fn,))
        row = cursor.fetchone()
        if row:
            results[fn] = row[0]
        else:
            # Try partial match
            cursor.execute("SELECT orig_path FROM photos WHERE filename LIKE ?", (f"%{fn}%",))
            row = cursor.fetchone()
            if row:
                results[fn] = row[0]
    
    conn.close()
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    find_photos()
