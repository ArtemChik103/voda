import sqlite3
import os
import glob
import shutil
import tempfile

appdata = os.environ.get('LOCALAPPDATA', '')
paths = glob.glob(os.path.join(appdata, '*/*/User Data/*/History'))

for p in paths:
    try:
        tmp = tempfile.mktemp()
        shutil.copy2(p, tmp)
        conn = sqlite3.connect(tmp)
        c = conn.cursor()
        c.execute('SELECT target_path, tab_url, site_url FROM downloads ORDER BY start_time DESC LIMIT 30')
        rows = c.fetchall()
        if rows:
            print("DB:", p)
            for r in rows:
                print('  Path:', r[0])
                print('  Tab :', r[1])
                print('  Site:', r[2])
                print()
        conn.close()
        os.remove(tmp)
    except Exception as e:
        print("Error reading", p, e)
