from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, current_app
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import os
import uuid
from math import ceil
from models.database import mysql
from werkzeug.security import generate_password_hash
from datetime import datetime

# Sesuaikan path import execute_query dengan struktur proyek Anda
from models.database import execute_query
from models.database import generate_next_menu_id

def get_db_connection():
    conn = mysql.connection
    try:
        conn.ping()
    except Exception as e:
        print(f"Koneksi DB gagal diperbarui: {e}")
        raise
    return conn


def query_db(query, args=(), one=False):
    """
    Fungsi helper untuk menjalankan query dan mengambil hasilnya.
    Menggunakan DictCursor secara otomatis jika sudah di-setting di app.config.
    """
    conn = get_db_connection()
    try:
        conn.ping()  # Tanpa reconnect karena pakai Flask-MySQLdb
    except Exception as e:
        print("[DB] Ping gagal:", e)
        raise
    cursor = conn.cursor()
    cursor.execute(query, args)
    results = cursor.fetchall()
    cursor.close()
    if one:
        return results[0] if results else None
    return results

def get_all_options():
    """Mengambil semua opsi dari tabel master untuk dropdown."""
    options = {
        'categories': query_db('SELECT * FROM categories ORDER BY name'),
        'main_ingredients': query_db('SELECT * FROM main_ingredients ORDER BY name'),
        'toppings': query_db('SELECT * FROM toppings ORDER BY name'),
        'aromas': query_db('SELECT * FROM aromas ORDER BY name'),
        'flavour_notes': query_db('SELECT * FROM flavour_notes ORDER BY name')
    }
    return options

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Mengecek apakah ekstensi file diizinkan."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Decorator untuk memeriksa apakah admin sudah login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            flash('Anda harus login untuk mengakses halaman ini.', 'warning')
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

# ====== RUTE LOGIN & LOGOUT ======
@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        admin = execute_query("SELECT * FROM admin WHERE username = %s", params=(username,), fetch='one')

        # --- PERBAIKAN PENTING DI SINI ---
        # Karena fetch='one' mengembalikan satu dictionary, kita tidak perlu [0]
        if admin and check_password_hash(admin['password'], password):
            session['admin_logged_in'] = True
            session['admin_id'] = admin['admin_id']
            session['username'] = admin['username']
            return redirect(url_for('admin.menu_list'))
        else:
            flash('❌ Username atau password salah', 'error')
    return render_template('admin/login.html')

@admin_bp.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Anda telah berhasil logout.', 'success')
    return redirect(url_for('admin.login'))

@admin_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = (request.form.get('current_password') or '').strip()
        new_password = (request.form.get('new_password') or '').strip()
        confirm_password = (request.form.get('confirm_password') or '').strip()

        admin_id = session.get('admin_id')
        admin = execute_query(
            "SELECT admin_id, username, password FROM admin WHERE admin_id = %s",
            params=(admin_id,),
            fetch='one'
        )

        if not admin:
            flash('Sesi tidak valid. Silakan login ulang.', 'error')
            return redirect(url_for('admin.logout'))

        if not check_password_hash(admin['password'], current_password):
            flash('Password saat ini tidak sesuai.', 'error')
            return render_template('admin/change_password.html')

        if len(new_password) < 8:
            flash('Password baru minimal 8 karakter.', 'error')
            return render_template('admin/change_password.html')

        if new_password != confirm_password:
            flash('Konfirmasi password tidak sama.', 'error')
            return render_template('admin/change_password.html')

        if check_password_hash(admin['password'], new_password):
            flash('Password baru tidak boleh sama dengan password lama.', 'error')
            return render_template('admin/change_password.html')

        new_hash = generate_password_hash(new_password)

        # Jika kolom audit tersedia
        try:
            execute_query(
                "UPDATE admin SET password = %s, password_changed_at = %s WHERE admin_id = %s",
                params=(new_hash, datetime.now(), admin_id),
                fetch=None
            )
        except Exception:
            # Fallback jika kolom audit belum ada
            execute_query(
                "UPDATE admin SET password = %s WHERE admin_id = %s",
                params=(new_hash, admin_id),
                fetch=None
            )

        session.clear()
        flash('Password berhasil diubah. Silakan login kembali.', 'success')
        return redirect(url_for('admin.login'))

    return render_template('admin/change_password.html')

# ====== RUTE MANAJEMEN MENU ======
@admin_bp.route('/menu')
@login_required
def menu_list():
    try:
        per_page = 10
        page = request.args.get('page', 1, type=int)
        search = request.args.get('search', '')
        offset = (page - 1) * per_page

        if search:
            query = "SELECT * FROM menu_items WHERE nama_minuman LIKE %s ORDER BY nama_minuman ASC LIMIT %s OFFSET %s"
            count_query = "SELECT COUNT(*) as total FROM menu_items WHERE nama_minuman LIKE %s"
            search_param = f"%{search}%"
            menu_items = execute_query(query, params=(search_param, per_page, offset), fetch='all')
            count_result = execute_query(count_query, params=(search_param,), fetch='one')
        else:
            query = "SELECT * FROM menu_items ORDER BY nama_minuman ASC LIMIT %s OFFSET %s"
            count_query = "SELECT COUNT(*) as total FROM menu_items"
            menu_items = execute_query(query, params=(per_page, offset), fetch='all')
            count_result = execute_query(count_query, fetch='one')

        total_count = count_result['total'] if count_result else 0
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1

        return render_template('admin/list_menu.html',
                               menu_items=menu_items,
                               page=page,
                               total_pages=total_pages,
                               search=search)
    except Exception as e:
        flash(f"Terjadi error saat memuat daftar menu: {e}", "error")
        print(f"Error in menu_list: {e}")
        return render_template('admin/list_menu.html',
                               menu_items=[], page=1, total_pages=1, search='')


# --- (Fungsi menu_add dan menu_edit Anda bisa diletakkan di sini, pastikan sudah benar) ---
# ... (Sisa kode untuk menu_add, menu_edit, riwayat, dll.)

@admin_bp.route('/menu/add', methods=['GET', 'POST'])
@login_required
def menu_add():
    if request.method == 'POST':
        conn = get_db_connection()  # Dapatkan koneksi di awal
        cursor = conn.cursor()
        
        try:
            # Langkah 1: Siapkan semua data dari form (Logika Asli Anda)
            menu_id = generate_next_menu_id()
            tekstur_selected = request.form.get('tekstur', '')
            harga_float = float(request.form['harga'])

            # Dictionary untuk data yang akan masuk ke tabel `menu_items`
            data_to_insert = {
                'menu_id': menu_id,
                'nama_minuman': request.form['nama_minuman'].strip(),
                'harga': harga_float,
                'availability': request.form['availability'],
                'temperatur_opsi': request.form['temperatur_opsi'],
                'rasa_asam': float(request.form.get('rasa_asam', 0.0)),
                'rasa_manis': float(request.form.get('rasa_manis', 0.0)),
                'rasa_pahit': float(request.form.get('rasa_pahit', 0.0)),
                'rasa_gurih': float(request.form.get('rasa_gurih', 0.0)),
                'kafein_score': float(request.form.get('kafein_score', 0.0)),
                'tekstur_LIGHT': 1 if tekstur_selected == 'tekstur_LIGHT' else 0,
                'tekstur_CREAMY': 1 if tekstur_selected == 'tekstur_CREAMY' else 0,
                'tekstur_BUBBLY': 1 if tekstur_selected == 'tekstur_BUBBLY' else 0,
                'tekstur_HEAVY': 1 if tekstur_selected == 'tekstur_HEAVY' else 0,
                'carbonated_score': 1.0 if request.form.get('carbonated_score') == 'ya' else 0.0,
                'sweetness_level': request.form['sweetness_level'],
                'harga_score': min(harga_float / 50000.0, 1.0),
                'foto': None # Akan diupdate setelah file di-save
            }

            # Handle upload file foto (Logika Asli Anda)
            if 'foto' in request.files and request.files['foto'].filename:
                file = request.files['foto']
                upload_dir = os.path.join(current_app.root_path, 'static', 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                filename = secure_filename(f"{menu_id}_{file.filename}")
                file.save(os.path.join(upload_dir, filename))
                data_to_insert['foto'] = filename
            
            # Langkah 2: Eksekusi INSERT ke tabel utama `menu_items`
            # Query ini HANYA berisi kolom-kolom yang ada di tabel menu_items.
            # Kolom seperti kategori, bahan_utama, dll., sudah tidak ada di sini.
            query = """
                INSERT INTO menu_items (menu_id, nama_minuman, harga, availability, temperatur_opsi, 
                                        rasa_asam, rasa_manis, rasa_pahit, rasa_gurih, kafein_score, 
                                        sweetness_level, tekstur_LIGHT, tekstur_CREAMY, tekstur_BUBBLY, 
                                        tekstur_HEAVY, carbonated_score, harga_score, foto)
                VALUES (%(menu_id)s, %(nama_minuman)s, %(harga)s, %(availability)s, %(temperatur_opsi)s, 
                        %(rasa_asam)s, %(rasa_manis)s, %(rasa_pahit)s, %(rasa_gurih)s, %(kafein_score)s,
                        %(sweetness_level)s, %(tekstur_LIGHT)s, %(tekstur_CREAMY)s, %(tekstur_BUBBLY)s,
                        %(tekstur_HEAVY)s, %(carbonated_score)s, %(harga_score)s, %(foto)s)
            """
            cursor.execute(query, data_to_insert)

            # Langkah 3: Ambil ID dari form dan insert ke tabel penghubung (Logika Baru)
            def insert_linking_data(table_name, column_name, id_list):
                if id_list:
                    # Menyiapkan data untuk multiple inserts yang lebih aman
                    records_to_insert = [(menu_id, item_id) for item_id in id_list]
                    sql_insert_link = f"INSERT INTO {table_name} (menu_item_id, {column_name}) VALUES (%s, %s)"
                    cursor.executemany(sql_insert_link, records_to_insert)

            insert_linking_data('menu_item_categories', 'category_id', request.form.getlist('categories'))
            insert_linking_data('menu_item_main_ingredients', 'main_ingredient_id', request.form.getlist('main_ingredients'))
            insert_linking_data('menu_item_toppings', 'topping_id', request.form.getlist('toppings'))
            insert_linking_data('menu_item_aromas', 'aroma_id', request.form.getlist('aromas'))
            insert_linking_data('menu_item_flavour_notes', 'flavour_note_id', request.form.getlist('flavour_notes'))

            # Langkah 4: Jika semua berhasil, commit transaksi
            conn.commit()
            flash("✅ Menu baru berhasil ditambahkan!", "success")
            return redirect(url_for('admin.menu_list'))

        except Exception as e:
            # Jika ada error di mana pun, batalkan semua perubahan
            conn.rollback()
            print(f"Error saat menambahkan menu: {e}")
            flash(f'❌ Terjadi kesalahan saat menyimpan: {str(e)}', 'error')
            # Kirim kembali data yang sudah diisi ke form agar user tidak perlu mengisi ulang
            options = get_all_options()
            return render_template('admin/menu_add.html', form_data=request.form, options=options)
        finally:
            # Pastikan koneksi ditutup
            cursor.close()
            conn.close()

    # Untuk method GET, tampilkan form kosong beserta pilihan dropdown
    options = get_all_options()
    return render_template('admin/menu_add.html', form_data={}, options=options)

@admin_bp.route("/menu/edit/<string:menu_id>", methods=["GET", "POST"])
@login_required
def menu_edit(menu_id):
    if request.method == "POST":
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            old = query_db("SELECT foto FROM menu_items WHERE menu_id=%s", (menu_id,), one=True)
            if not old:
                flash("❌ Menu yang akan diedit tidak ditemukan.", "error")
                return redirect(url_for("admin.menu_list"))

            tekstur = request.form.get("tekstur", "")
            harga = float(request.form.get("harga", 0))
            data = {
                "menu_id": menu_id,
                "nama_minuman": request.form.get("nama_minuman", "").strip(),
                "harga": harga,
                "availability": request.form.get("availability"),
                "temperatur_opsi": request.form.get("temperatur_opsi"),
                "rasa_asam": float(request.form.get("rasa_asam", 0) or 0),
                "rasa_manis": float(request.form.get("rasa_manis", 0) or 0),
                "rasa_pahit": float(request.form.get("rasa_pahit", 0) or 0),
                "rasa_gurih": float(request.form.get("rasa_gurih", 0) or 0),
                "kafein_score": float(request.form.get("kafein_score", 0) or 0),
                "tekstur_LIGHT": 1 if tekstur == "tekstur_LIGHT" else 0,
                "tekstur_CREAMY": 1 if tekstur == "tekstur_CREAMY" else 0,
                "tekstur_BUBBLY": 1 if tekstur == "tekstur_BUBBLY" else 0,
                "tekstur_HEAVY": 1 if tekstur == "tekstur_HEAVY" else 0,
                "carbonated_score": 1.0 if request.form.get("carbonated_score") == "ya" else 0.0,
                "sweetness_level": request.form.get("sweetness_level"),
                "harga_score": min(harga / 50000.0, 1.0),
                "foto": old.get("foto"),
            }

            # optional ganti foto
            file = request.files.get("foto")
            if file and file.filename:
                uploads = os.path.join(current_app.root_path, "static", "uploads")
                os.makedirs(uploads, exist_ok=True)
                # hapus lama
                if old.get("foto"):
                    oldpath = os.path.join(uploads, old["foto"])
                    if os.path.exists(oldpath):
                        try:
                            os.remove(oldpath)
                        except Exception:
                            pass
                # simpan baru
                filename = secure_filename(f"{menu_id}_{file.filename}")
                file.save(os.path.join(uploads, filename))
                data["foto"] = filename

            # update menu_items
            cur.execute(
                """
                UPDATE menu_items SET
                    nama_minuman=%(nama_minuman)s, harga=%(harga)s, availability=%(availability)s,
                    temperatur_opsi=%(temperatur_opsi)s, rasa_asam=%(rasa_asam)s, rasa_manis=%(rasa_manis)s,
                    rasa_pahit=%(rasa_pahit)s, rasa_gurih=%(rasa_gurih)s, kafein_score=%(kafein_score)s,
                    tekstur_LIGHT=%(tekstur_LIGHT)s, tekstur_CREAMY=%(tekstur_CREAMY)s, tekstur_BUBBLY=%(tekstur_BUBBLY)s,
                    tekstur_HEAVY=%(tekstur_HEAVY)s, carbonated_score=%(carbonated_score)s,
                    sweetness_level=%(sweetness_level)s, harga_score=%(harga_score)s, foto=%(foto)s
                WHERE menu_id=%(menu_id)s
                """,
                data,
            )

            # reset & insert pivot
            for table in [
                "menu_item_categories",
                "menu_item_main_ingredients",
                "menu_item_toppings",
                "menu_item_aromas",
                "menu_item_flavour_notes",
            ]:
                cur.execute(f"DELETE FROM {table} WHERE menu_item_id=%s", (menu_id,))

            def insert_pivot(table, col, ids):
                if not ids:
                    return
                vals = [(menu_id, i) for i in ids]
                cur.executemany(f"INSERT INTO {table} (menu_item_id, {col}) VALUES (%s, %s)", vals)

            insert_pivot("menu_item_categories",        "category_id",        request.form.getlist("categories"))
            insert_pivot("menu_item_main_ingredients",  "main_ingredient_id", request.form.getlist("main_ingredients"))
            insert_pivot("menu_item_toppings",          "topping_id",         request.form.getlist("toppings"))
            insert_pivot("menu_item_aromas",            "aroma_id",           request.form.getlist("aromas"))
            insert_pivot("menu_item_flavour_notes",     "flavour_note_id",    request.form.getlist("flavour_notes"))

            conn.commit()
            flash("✅ Menu berhasil diperbarui!", "success")
            return redirect(url_for("admin.menu_list"))
        except Exception as e:
            conn.rollback()
            print("[menu_edit] ERROR:", e)
            flash(f"❌ Gagal memperbarui menu: {e}", "error")
            return redirect(url_for("admin.menu_edit", menu_id=menu_id))
        finally:
            cur.close()

    # GET: render form + prefill
    menu = query_db("SELECT * FROM menu_items WHERE menu_id=%s", (menu_id,), one=True)
    if not menu:
        flash(f"❌ Menu dengan ID '{menu_id}' tidak ditemukan.", "error")
        return redirect(url_for("admin.menu_list"))

    options = get_all_options()
    selected_options = {
        "category_ids":     [r["category_id"] for r in query_db("SELECT category_id FROM menu_item_categories WHERE menu_item_id=%s", (menu_id,))],
        "ingredient_ids":   [r["main_ingredient_id"] for r in query_db("SELECT main_ingredient_id FROM menu_item_main_ingredients WHERE menu_item_id=%s", (menu_id,))],
        "topping_ids":      [r["topping_id"] for r in query_db("SELECT topping_id FROM menu_item_toppings WHERE menu_item_id=%s", (menu_id,))],
        "aroma_ids":        [r["aroma_id"] for r in query_db("SELECT aroma_id FROM menu_item_aromas WHERE menu_item_id=%s", (menu_id,))],
        "flavour_note_ids": [r["flavour_note_id"] for r in query_db("SELECT flavour_note_id FROM menu_item_flavour_notes WHERE menu_item_id=%s", (menu_id,))],
    }
    return render_template("admin/menu_edit.html", menu=menu, options=options, selected_options=selected_options)

# admin_controller.py
# admin_controller.py
@admin_bp.route('/riwayat', methods=['GET'])
@login_required
def riwayat():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 10
        offset = (page - 1) * per_page

        search = request.args.get('search', '').strip()
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        filter_status = request.args.get('feedback_status', '').strip()  # 'pending' | 'completed' | 'skipped'

        # --- Base per PREFERENSI ---
        base = """
            SELECT
                p.session_id,
                p.timestamp,                 -- waktu input
                p.pref_id,
                p.nama_customer,
                p.quiz_attempt,

                -- Turunkan status jika NULL: jika belum ada feedback -> pending, jika sudah ada -> completed
                COALESCE(
                    p.feedback_status,
                    CASE
                        WHEN COUNT(CASE WHEN r.feedback IS NOT NULL THEN 1 END) = 0 THEN 'pending'
                        ELSE 'completed'
                    END
                ) AS feedback_status,

                -- Teks feedback untuk tampilan:
                -- - 'Dilewati' bila status 'skipped'
                -- - 'Belum ada' bila belum ada satupun feedback
                -- - 'Membantu' bila rata-rata = 1
                -- - 'Tidak membantu' bila rata-rata = 0
                -- - fallback 'Belum ada' untuk kondisi lain/inkonsisten
                CASE
                    WHEN COALESCE(p.feedback_status, '') = 'skipped' THEN 'Dilewati'
                    WHEN COUNT(CASE WHEN r.feedback IS NOT NULL THEN 1 END) = 0 THEN 'Belum ada'
                    WHEN AVG(r.feedback) = 1 THEN 'Membantu'
                    WHEN AVG(r.feedback) = 0 THEN 'Tidak membantu'
                    ELSE 'Belum ada'
                END AS feedback_text

            FROM preferences p
            LEFT JOIN recommendations r ON r.pref_id = p.pref_id
        """

        where = []
        params = []

        if search:
            where.append("(p.nama_customer LIKE %s OR p.session_id LIKE %s OR p.pref_id LIKE %s)")
            s = f"%{search}%"
            params.extend([s, s, s])

        if start_date:
            where.append("p.timestamp >= %s")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            where.append("p.timestamp <= %s")
            params.append(f"{end_date} 23:59:59")

        if where:
            base += " WHERE " + " AND ".join(where)

        base += " GROUP BY p.pref_id"

        # Filter status feedback (opsional)
        if filter_status:
            base += " HAVING feedback_status = %s"
            params.append(filter_status)

        # Count untuk pagination
        count_sql = f"SELECT COUNT(*) AS total FROM ({base}) AS t"
        count_row = execute_query(count_sql, params=tuple(params), fetch='one')
        total_count = count_row['total'] if count_row else 0
        total_pages = max(1, (total_count + per_page - 1) // per_page)

        # Paging
        list_sql = base + " ORDER BY p.timestamp DESC LIMIT %s OFFSET %s"
        list_params = tuple(params) + (per_page, offset)
        rows = execute_query(list_sql, params=list_params, fetch='all')

        return render_template(
            'admin/riwayat.html',
            riwayat_items=rows,
            page=page,
            total_pages=total_pages,
            search=search,
            start_date=start_date,
            end_date=end_date,
            feedback_status=filter_status
        )
    except Exception as e:
        print(f"[riwayat] ERROR: {e}")
        flash("Terjadi kesalahan saat memuat riwayat.", "error")
        return render_template('admin/riwayat.html',
                               riwayat_items=[], page=1, total_pages=1,
                               search='', start_date='', end_date='', feedback_status='')
    
# ==== MANAJEMEN INPUT MASTER (categories, main_ingredients, toppings, aromas, flavour_notes) ====

@admin_bp.route('/manage-inputs')
@login_required
def manajemen_input():
    # Halaman list "apa yang mau dikelola"
    return render_template('admin/manajemen_input.html')


def _table_map():
    # mapping nama field -> tabel & label tampilannya
    return {
        'categories':        {'table': 'categories',        'label': 'Kategori'},
        'main_ingredients':  {'table': 'main_ingredients',  'label': 'Bahan Utama'},
        'toppings':          {'table': 'toppings',          'label': 'Topping'},
        'aromas':            {'table': 'aromas',            'label': 'Aroma'},
        'flavour_notes':     {'table': 'flavour_notes',     'label': 'Flavour Notes'},
    }


@admin_bp.route('/manage-inputs/<field>', methods=['GET', 'POST'])
@login_required
def manage_options(field):
    mapping = _table_map().get(field)
    if not mapping:
        flash('Field tidak valid.', 'error')
        return redirect(url_for('admin.manajemen_input'))

    table_name = mapping['table']
    label = mapping['label']

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Tambah / Update
        if request.method == 'POST':
            name = (request.form.get('name') or '').strip()
            opt_id = request.form.get('id')  # jika ada berarti update

            # validasi kosong
            if not name:
                flash('Nama tidak boleh kosong.', 'error')
                return redirect(url_for('admin.manage_options', field=field))

            # validasi duplikat
            if opt_id:
                cur.execute(f"SELECT 1 FROM {table_name} WHERE name=%s AND id<>%s LIMIT 1", (name, opt_id))
            else:
                cur.execute(f"SELECT 1 FROM {table_name} WHERE name=%s LIMIT 1", (name,))
            if cur.fetchone():
                flash('Nama sudah ada. Gunakan nama lain.', 'warning')
                return redirect(url_for('admin.manage_options', field=field))

            if opt_id:  # UPDATE
                cur.execute(f"UPDATE {table_name} SET name=%s WHERE id=%s", (name, opt_id))
                flash(f'{label} berhasil diperbarui.', 'success')
            else:       # INSERT
                cur.execute(f"INSERT INTO {table_name} (name) VALUES (%s)", (name,))
                flash(f'{label} baru berhasil ditambahkan.', 'success')

            conn.commit()
            return redirect(url_for('admin.manage_options', field=field))

        # GET: list + (optional) edit
        item_to_edit = None
        edit_id = request.args.get('edit')
        if edit_id:
            cur.execute(f"SELECT * FROM {table_name} WHERE id=%s", (edit_id,))
            row = cur.fetchone()
            if row:
                # MySQLdb DictCursor -> dict
                item_to_edit = dict(row)

        cur.execute(f"SELECT * FROM {table_name} ORDER BY name ASC")
        options = [dict(r) for r in cur.fetchall()]

        return render_template('admin/manage_options.html',
                               options=options,
                               field_name=field,
                               field_label=label,
                               item=item_to_edit)

    except Exception as e:
        conn.rollback()
        print('[manage_options] ERROR:', e)
        flash('Terjadi kesalahan saat memuat/menyimpan data.', 'error')
        return redirect(url_for('admin.manajemen_input'))
    finally:
        cur.close()


@admin_bp.route('/manage-inputs/<field>/delete/<int:opt_id>', methods=['POST'])
@login_required
def delete_option(field, opt_id):
    mapping = _table_map().get(field)
    if not mapping:
        flash('Field tidak valid.', 'error')
        return redirect(url_for('admin.manajemen_input'))

    table_name = mapping['table']
    label = mapping['label']

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # catatan: FK dari tabel pivot -> tabel master sebaiknya ON DELETE CASCADE
        # jika belum, hapus dulu dari pivot secara manual sebelum delete master

        cur.execute(f"DELETE FROM {table_name} WHERE id=%s", (opt_id,))
        conn.commit()
        flash(f'{label} berhasil dihapus.', 'success')
    except Exception as e:
        conn.rollback()
        # fallback: beri pesan kenapa gagal (FK)
        flash(f'Tidak bisa menghapus {label} karena masih dipakai. ({e})', 'error')
    finally:
        cur.close()
    return redirect(url_for('admin.manage_options', field=field))


# admin_controller.py
@admin_bp.route('/riwayat/detail/<string:pref_id>')
@login_required
def riwayat_detail(pref_id):
    try:
        # detail preferensi
        pref_query = """
            SELECT nama_customer, mood, rasa, tekstur, kafein, suhu, budget, timestamp 
            FROM preferences 
            WHERE pref_id = %s
        """
        pref_detail = execute_query(pref_query, params=(pref_id,), fetch='one')
        if not pref_detail:
            return jsonify({'success': False, 'message': 'Data preferensi tidak ditemukan'}), 404

        # rekomendasi untuk pref_id ini
        rec_query = """
            SELECT 
                m.nama_minuman,
                r.rank_position,
                r.similarity AS similarity_score,
                COALESCE(r.final_score, r.similarity) AS final_score
            FROM recommendations r
            JOIN menu_items m ON r.menu_id = m.menu_id
            WHERE r.pref_id = %s
            ORDER BY r.rank_position ASC
        """
        recs = execute_query(rec_query, params=(pref_id,), fetch='all')

        return jsonify({'success': True, 'data': {
            'preferences': pref_detail,
            'recommendations': recs
        }})
    except Exception as e:
        print(f"[riwayat_detail] ERROR for {pref_id}: {e}")
        return jsonify({'success': False, 'message': 'Terjadi kesalahan pada server'}), 500

def map_labels(options, ids, key='id'):
    idset = set((ids or []))
    return [row['name'] for row in options if row[key] in idset]

@admin_bp.route('/menu/detail/<string:menu_id>')
@login_required
def menu_detail(menu_id):
    # data utama
    menu = query_db("SELECT * FROM menu_items WHERE menu_id=%s", (menu_id,), one=True)
    if not menu:
        flash("❌ Menu tidak ditemukan.", "error")
        return redirect(url_for('admin.menu_list'))

    # ambil opsi & id terpilih
    options = get_all_options()
    selected_ids = {
        "categories":        [r["category_id"] for r in query_db("SELECT category_id FROM menu_item_categories WHERE menu_item_id=%s", (menu_id,))],
        "main_ingredients":  [r["main_ingredient_id"] for r in query_db("SELECT main_ingredient_id FROM menu_item_main_ingredients WHERE menu_item_id=%s", (menu_id,))],
        "toppings":          [r["topping_id"] for r in query_db("SELECT topping_id FROM menu_item_toppings WHERE menu_item_id=%s", (menu_id,))],
        "aromas":            [r["aroma_id"] for r in query_db("SELECT aroma_id FROM menu_item_aromas WHERE menu_item_id=%s", (menu_id,))],
        "flavour_notes":     [r["flavour_note_id"] for r in query_db("SELECT flavour_note_id FROM menu_item_flavour_notes WHERE menu_item_id=%s", (menu_id,))],
    }

    # konversi ke label untuk ditampilkan
    selected_labels = {
        "categories":       map_labels(options['categories'],       selected_ids['categories']),
        "main_ingredients": map_labels(options['main_ingredients'], selected_ids['main_ingredients']),
        "toppings":         map_labels(options['toppings'],         selected_ids['toppings']),
        "aromas":           map_labels(options['aromas'],           selected_ids['aromas']),
        "flavour_notes":    map_labels(options['flavour_notes'],    selected_ids['flavour_notes']),
    }

    return render_template('admin/menu_detail.html',
                           menu=menu,
                           selected_labels=selected_labels)

@admin_bp.route('/menu/delete/<string:menu_id>', methods=['POST'])
@login_required
def menu_delete(menu_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # ambil nama file foto (kalau ada) untuk dihapus dari storage
        row = query_db("SELECT foto FROM menu_items WHERE menu_id=%s", (menu_id,), one=True)
        if not row:
            flash("Menu tidak ditemukan.", "error")
            return redirect(url_for('admin.menu_list'))

        # hapus relasi di tabel pivot (kalau FK belum ON DELETE CASCADE)
        for table in [
            'menu_item_categories',
            'menu_item_main_ingredients',
            'menu_item_toppings',
            'menu_item_aromas',
            'menu_item_flavour_notes',
        ]:
            cur.execute(f"DELETE FROM {table} WHERE menu_item_id=%s", (menu_id,))

        # hapus menu utama
        cur.execute("DELETE FROM menu_items WHERE menu_id=%s", (menu_id,))
        conn.commit()

        # hapus file foto fisik
        if row.get('foto'):
            uploads = os.path.join(current_app.root_path, 'static', 'uploads')
            path = os.path.join(uploads, row['foto'])
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

        flash("✅ Menu berhasil dihapus.", "success")
        return redirect(url_for('admin.menu_list'))
    except Exception as e:
        conn.rollback()
        print("[menu_delete] ERROR:", e)
        flash(f"Gagal menghapus menu: {e}", "error")
        return redirect(url_for('admin.menu_detail', menu_id=menu_id))
    finally:
        cur.close()
