from werkzeug.security import generate_password_hash
import mysql.connector

# Ganti sesuai koneksi DB kamu
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="dbchaos"
)

cursor = conn.cursor()

username = 'admintest'
plain_password = 'admin123'
hashed_password = generate_password_hash(plain_password)

query = "INSERT INTO admin (username, password) VALUES (%s, %s)"
cursor.execute(query, (username, hashed_password))
conn.commit()

print("✅ Admin berhasil ditambahkan")
cursor.close()
conn.close()
