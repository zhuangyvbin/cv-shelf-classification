import pymysql

def fetch_pending_images():
    conn = pymysql.connect(host="localhost", user="root", password="xxx", database="rack")
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path FROM image_tasks WHERE status='pending'")
    return cursor.fetchall(), conn, cursor

def update_result(cursor, id_, result):
    cursor.execute("UPDATE image_tasks SET status='done', prediction_result=%s WHERE id=%s", (result, id_))
