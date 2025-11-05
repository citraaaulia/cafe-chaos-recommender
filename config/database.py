from flask_mysqldb import MySQL
import MySQLdb.cursors

# Inisialisasi mysql di luar agar bisa diimpor
mysql = MySQL()

def init_mysql(app):
    """Fungsi untuk inisialisasi MySQL dengan aplikasi Flask."""
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # ⬅️ Tambahkan baris ini!
    mysql.init_app(app)


def execute_query(query, params=None, fetch=None):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(query, params or ())
        
        if fetch == 'one':
            result = cursor.fetchone()
        elif fetch == 'all':
            result = cursor.fetchall()
        else:
            mysql.connection.commit()
            result = cursor.rowcount
        
        cursor.close()

        # FIX: jika result berupa tuple dengan 1 dict di dalam, ambil isi dict-nya langsung
        if isinstance(result, tuple) and len(result) == 1 and isinstance(result[0], dict):
            return result[0]

        return result

    except Exception as e:
        print(f"Database Error: {e}")
        return None
