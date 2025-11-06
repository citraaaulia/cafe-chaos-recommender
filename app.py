import os
from dotenv import load_dotenv

import pymysql
pymysql.install_as_MySQLdb()

load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, session, jsonify

# Import dari file/modul lain dalam proyek Anda
from models.database import init_mysql
from model.recommender import get_recommendations, update_feedback, get_recommendation_analytics, init_recommender, create_session
from controllers.admin_controller import admin_bp

def create_app():
    """
    Application Factory: Fungsi untuk membuat dan mengkonfigurasi aplikasi Flask.
    """
    app = Flask(__name__)

    # --- 1. Konfigurasi Aplikasi ---
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-change-me')

    # Konfigurasi Database MySQL
    app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', '127.0.0.1')
    app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
    app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
    app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'db_ml_chaos')
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

    # --- 2. Inisialisasi Ekstensi ---
    init_mysql(app)
    init_recommender(app)

    # --- 3. Pendaftaran Blueprint ---
    app.register_blueprint(admin_bp)

    # --- 4. Rute Utama (Kuis Pengguna) ---
    
    @app.route('/')
    def home():
        session.clear()
        return redirect(url_for('index'))

    @app.route('/index', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            session.clear()
            # Create new session when user clicks start
            session_id = create_session()
            if session_id:
                session['session_id'] = session_id
                return redirect(url_for('step1_nama'))
            else:
                # Handle error creating session
                return render_template('index.html', error="Gagal membuat sesi. Silakan coba lagi.")
        return render_template('index.html')

    @app.route('/step1', methods=['GET', 'POST'])
    def step1_nama():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            nama = request.form['nama'].strip()
            if nama:
                session['nama'] = nama
                return redirect(url_for('step2_mood'))
            else:
                # Handle empty name
                return render_template('step1_nama.html', 
                                     selected=session.get('nama', ''),
                                     error="Nama tidak boleh kosong")
        return render_template('step1_nama.html', selected=session.get('nama', ''))

    @app.route('/step2', methods=['GET', 'POST'])
    def step2_mood():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            session['mood'] = request.form['mood']
            return redirect(url_for('step3_rasa'))
        return render_template('step2_mood.html', selected=session.get('mood', ''))

    @app.route('/step3', methods=['GET', 'POST'])
    def step3_rasa():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            session['rasa'] = request.form.get('rasa')
            return redirect(url_for('step4_tekstur'))
        return render_template('step3_rasa.html', selected=session.get('rasa', ''))

    @app.route('/step4', methods=['GET', 'POST'])
    def step4_tekstur():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            session['tekstur'] = request.form.get('tekstur')
            return redirect(url_for('step5_kafein'))
        return render_template('step4_tekstur.html', selected=session.get('tekstur', ''))

    @app.route('/step5', methods=['GET', 'POST'])
    def step5_kafein():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            session['kafein'] = request.form.get('kafein')
            return redirect(url_for('step6_suhu'))
        return render_template('step5_kafein.html', selected=session.get('kafein', ''))

    @app.route('/step6', methods=['GET', 'POST'])
    def step6_suhu():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            session['suhu'] = request.form.get('suhu')
            return redirect(url_for('step7_budget'))
        return render_template('step6_suhu.html', selected=session.get('suhu', ''))

    @app.route('/step7', methods=['GET', 'POST'])
    def step7_budget():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            session['budget'] = request.form.get('budget')
            return redirect(url_for('hasil'))
        return render_template('step7_budget.html', selected=session.get('budget', ''))
    
    @app.route('/submit_feedback', methods=['POST'])
    def submit_feedback():
        try:
            session_id = session.get('session_id')
            pref_id = session.get('pref_id')
            quiz_attempt = session.get('quiz_attempt')
        
            if not session_id or not pref_id or not quiz_attempt:
                return jsonify({
                    'success': False, 
                    'message': 'Sesi tidak ditemukan. Silakan ulangi kuis.'
                }), 400
        
        # 2. Ambil dan validasi data feedback dari form
            feedback = request.form.get('feedback')
            if not feedback or feedback not in ['positive', 'negative']:
                return jsonify({
                    'success': False,
                    'message': 'Feedback tidak valid. Harap berikan feedback "positive" atau "negative".'
                }), 400
            
        # 3. Proses feedback
            feedback_bool = (feedback == 'positive')
            success = update_feedback(session_id, pref_id, quiz_attempt, feedback_bool)
        
        # 4. Berikan respons berdasarkan hasil update
            if success:
            # Hapus session setelah feedback berhasil disimpan untuk mencegah pengiriman ganda
                session.pop('pref_id', None)
                session.pop('quiz_attempt', None)
            
                message = 'Terima kasih atas feedback positif Anda!' if feedback_bool else 'Terima kasih atas feedback Anda. Kami akan terus meningkatkan sistem rekomendasi.'
                return jsonify({
                    'success': True,
                    'message': message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Gagal menyimpan feedback ke database.'
                }), 500
            
        except Exception as e:
            print(f"ERROR in submit_feedback: {e}")
            return jsonify({
                'success': False,
                'message': 'Terjadi kesalahan internal pada sistem.'
            }), 500

    @app.route('/hasil')
    def hasil():
        if 'session_id' not in session:
            return redirect(url_for('index'))
            
        # FIXED: Hapus prioritas dari required_fields karena sudah tidak digunakan
        required_fields = ['nama', 'mood', 'rasa', 'tekstur', 'kafein', 'suhu', 'budget']
        for field in required_fields:
            if field not in session:
                print(f"Missing field: {field}")  # Debug log
                return redirect(url_for('step1_nama'))

        preferensi = {
            'mood': session.get('mood'),
            'rasa': session.get('rasa'),
            'tekstur': session.get('tekstur'),
            'kafein': session.get('kafein'),
            'suhu': session.get('suhu'),
            'budget': session.get('budget'),
            'session_id': session.get('session_id')
        }
        
        # Get recommendations with customer name for database tracking
        preferensi['nama_customer'] = session.get('nama')
        rekomendasi = get_recommendations(preferensi)
        
        # Definisikan nama_customer dengan benar
        nama_customer = session.get('nama', 'Customer')
        
        # Store recommendation session info for feedback
        if rekomendasi and len(rekomendasi) > 0:
            session['pref_id'] = rekomendasi[0].get('pref_id', '')
            session['quiz_attempt'] = rekomendasi[0].get('quiz_attempt', 1)
        
        return render_template('hasil.html', 
                             rekomendasi=rekomendasi,
                             nama_customer=nama_customer)
    
    @app.route('/ulangi_quiz')
    def ulangi_quiz():
        # Hapus semua data quiz kecuali session_id dan nama
        session_id = session.get('session_id')
        nama = session.get('nama')
        
        # Clear session data
        session.clear()
        
        # Restore session_id and nama
        if session_id:
            session['session_id'] = session_id
        if nama:
            session['nama'] = nama
            
        return redirect(url_for('step2_mood'))
    
    @app.route('/selesai')
    def selesai_quiz():
        session.clear()
        return redirect(url_for('index'))

    # --- 5. API Routes for Analytics (Optional) ---
    @app.route('/api/analytics')
    def api_analytics():
        """API endpoint to get recommendation analytics."""
        try:
            analytics = get_recommendation_analytics()
            return jsonify({
                'success': True,
                'data': analytics
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            })

    # --- 6. Error Handlers ---
    @app.errorhandler(404)
    def not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(e):
        return render_template('errors/500.html'), 500

    return app
app = create_app()