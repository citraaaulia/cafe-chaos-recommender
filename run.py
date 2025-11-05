# Di dalam file run.py

from app import create_app

# Buat instance aplikasi menggunakan factory
app = create_app()

if __name__ == '__main__':
    # Jalankan aplikasi dalam mode debug
    app.run(debug=True)