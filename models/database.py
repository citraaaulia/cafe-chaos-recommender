from flask import current_app
from flask_mysqldb import MySQL
import MySQLdb.cursors

mysql = MySQL()

def init_mysql(app):
    """Fungsi untuk inisialisasi MySQL dengan aplikasi Flask."""
    mysql.init_app(app)

def execute_query(query, params=None, fetch=None):
    try:
        mysql.connection.ping()  # coba aktifkan kembali koneksi
    except Exception as e:
        print("[MYSQL] koneksi error:", e)
        mysql.connection = mysql.connect  # reconnect (jika memungkinkan)

    """
    Menjalankan query database dengan aman dan efisien.
    - query: String SQL query.
    - params: Tuple parameter untuk query (mencegah SQL Injection).
    - fetch: 'one' (ambil 1 baris), 'all' (ambil semua baris), atau None (untuk INSERT/UPDATE/DELETE).
    """
    try:
        # Selalu gunakan DictCursor agar hasilnya berupa dictionary
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(query, params or ())
        
        if fetch == 'one':
            # Mengambil satu baris data sebagai dictionary
            result = cursor.fetchone()
        elif fetch == 'all':
            # Mengambil semua baris data sebagai list of dictionaries
            result = cursor.fetchall()
        else:
            # Jika bukan SELECT (INSERT, UPDATE, DELETE), commit perubahan
            mysql.connection.commit()
            result = cursor.rowcount  # Kembalikan jumlah baris terpengaruh
        
        cursor.close()
        return result

    except Exception as e:
        print(f"Database Error: {e}")
        # Jika query bukan SELECT, batalkan transaksi jika ada error
        if not fetch:
            mysql.connection.rollback()
        return None
    finally:
        # Pastikan cursor ditutup jika masih ada
        if 'cursor' in locals() and cursor:
            cursor.close()

def generate_next_menu_id():
    query = """
        SELECT MAX(CAST(SUBSTRING(menu_id, 4) AS UNSIGNED)) AS max_id
        FROM menu_items
        WHERE menu_id REGEXP '^MI_[0-9]+$'
    """
    result = execute_query(query, fetch='one')
    if result and result['max_id']:
        next_number = int(result['max_id']) + 1
        return f"MI_{next_number:02d}"
    return "MI_01"

